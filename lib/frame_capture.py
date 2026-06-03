"""frame_capture.py -- feed a shot's camera frames to the ExptServer (run-loop capture).

The pyctrl counterpart of MATLAB ``server_post_run`` (the Imag399-step after-end callback that,
in production, grabs the shot's frames and publishes them). In scenario 3 the FPGA pulses the
Orca trigger (``FPGA1/TTL54``, rising edge) at each imaging step, so after a shot the camera
buffer holds ``num_images`` frames; this module reads them and pushes each to
``ExptServer.store_imgs`` in the column-major ``[s1, s2, s3, pixels]`` wire format
(:func:`orca_camera.to_store_array`), then calls ``seq_finish()`` to publish the sequence.

Two pieces:
  * :func:`store_shot_frames` -- the core feeder (read N frames -> store_imgs* -> seq_finish);
    on a short read it ``seq_cancel()``s instead of publishing a partial shot (mirrors the
    MATLAB ``nFrames``-short path). NO-HARDWARE-testable with a fake camera + fake server.
  * :func:`make_capture_post_cb` -- wraps it as a ``run_scan_group`` ``post_cb`` (per-shot),
    pulling ``scan_id`` from a holder and ``seq_id`` from ``seq_config.G.seq_id`` (read BEFORE
    the loop bumps it). The runner arms the camera (external trigger) before the scan and
    disarms after; this hook fires once per completed shot.

Live exercise of the full chain (arm -> run a real seq that pulses TTL54 -> capture -> publish)
is the gated full-run acceptance test (it fires the whole experiment); this module is its
NO-HARDWARE-tested core.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time


def store_shot_frames(camera, server, num_images, scan_id, seq_id, *,
                      timeout=10.0, sleep=time.sleep, clock=time.monotonic):
    """Read ``num_images`` frames from ``camera`` and publish them via ``server``.

    Args:
        camera: an :class:`orca_camera.OrcaCamera` (or fake) exposing ``read_frames() ->
            list_of_ndarray`` (already armed for external trigger by the caller).
        server: an ExptServer-like hub with ``store_imgs(arr, scan_id, seq_id)`` /
            ``seq_finish()`` / ``seq_cancel()``.
        num_images: frames expected for this shot (from the scan's ``NumImages``).
        scan_id, seq_id: routing ids stamped on the sequence (store_imgs records them on the
            first call of the shot).
        timeout: seconds to wait for all ``num_images`` frames before giving up.

    Returns the number of frames published (``num_images`` on success, the partial count on a
    short read -- in which case ``seq_cancel()`` was called and NOTHING was published).
    """
    from orca_camera import to_store_array

    collected = []
    deadline = clock() + timeout
    while len(collected) < num_images and clock() < deadline:
        for f in camera.read_frames():
            collected.append(f)
            if len(collected) >= num_images:
                break
        if len(collected) < num_images:
            sleep(0.01)

    if len(collected) < num_images:
        # Short read: do NOT publish a partial sequence (mirror server_post_run's nFrames
        # guard -> seq_cancel clears the staged temp_imgs).
        server.seq_cancel()
        return len(collected)

    for f in collected[:num_images]:
        server.store_imgs(to_store_array(f), scan_id, seq_id)
    server.seq_finish()
    return num_images


def make_capture_post_cb(camera, server, num_images, scan_id, seq_config, *,
                         timeout=10.0):
    """Build a ``run_scan_group`` ``post_cb`` that captures + publishes one shot's frames.

    Args:
        camera / server / num_images / timeout: forwarded to :func:`store_shot_frames`.
        scan_id: the scan id (an int, or a 0-arg callable returning it -- e.g. the value
            ``ControlChannel.begin_scan`` produced for this scan).
        seq_config: the :class:`SeqConfig`; ``seq_config.G.seq_id`` is read for the per-shot
            ``seq_id`` BEFORE the run loop bumps it (run_seq.py runs ``post_cb`` then
            ``_bump_seq_id``), matching MATLAB's ``server_post_run`` reading ``s1.G.seq_id``.

    The returned callback has the ``post_cb(cur_seq_num, arg0)`` signature run_scan_group uses
    (both args ignored here; routing comes from scan_id + seq_id).
    """
    def post_cb(cur_seq_num, arg0):  # noqa: ARG001 - run_scan_group's post_cb signature
        sid = scan_id() if callable(scan_id) else scan_id
        try:
            seq_id = int(seq_config.G.seq_id(1))
        except Exception:  # noqa: BLE001 - absent field -> default to 1
            seq_id = 1
        store_shot_frames(camera, server, num_images, sid, seq_id, timeout=timeout)

    return post_cb
