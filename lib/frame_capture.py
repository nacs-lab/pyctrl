"""frame_capture.py -- the default per-shot camera capture (run-loop ACQUIRE side).

The pyctrl counterpart of MATLAB ``server_post_run`` for non-rearrangement scans. In scenario 3
the FPGA pulses the Orca trigger (``FPGA1/TTL54``, rising edge) at each imaging step, so after a
shot the camera buffer holds ``num_images`` frames. This module's job is the **acquire** half:
read those frames off the buffer (synchronously -- buffer-timing sensitive, and cheap). The
**persist** half (encode + ``store_imgs`` + ``seq_finish``) is owned by ``ExptServer`` and runs on
its single FIFO worker, so the ~80 ms save overlaps the next shot's hardware instead of adding to
the end-of-shot dead time. See ``pyctrl/docs/callbacks.md`` for the acquire-vs-persist split.

Two pieces:
  * :func:`read_shot_frames` -- read exactly ``num_images`` frames (or ``None`` on a short read,
    so the caller drops the shot -- mirrors MATLAB's ``nFrames`` guard). NO-HARDWARE-testable.
  * :func:`make_capture_post_cb` -- the ``run_scan_group`` ``post_cb`` (per-shot): read frames,
    then hand them to ``server.publish_shot(..., async_=...)``. ``scan_id`` from a holder, ``seq_id``
    from ``seq_config.G.seq_id`` (read BEFORE the loop bumps it). ``async_`` (default True) offloads
    the persist to the worker; ``async_=False`` (the ``YB_ASYNC_FRAME_SAVE=0`` kill-switch) runs it
    inline -- the pre-async behaviour, for an A/B or rollback.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time

import run_timing


def read_shot_frames(camera, num_images, *, timeout=10.0, sleep=time.sleep,
                     clock=time.monotonic):
    """Read exactly ``num_images`` frames off ``camera``; return the list, or ``None`` on a short
    read (the caller then drops the shot, so the ``.h5`` never holds a partial sequence -- the
    MATLAB ``server_post_run`` nFrames guard). Reads (consumes) any frames that did arrive.

    ``camera`` exposes ``read_frames() -> list_of_ndarray`` (already armed for external trigger by
    the caller). NO-HARDWARE-testable with a scripted fake camera.
    """
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
        return None
    return collected[:num_images]


def make_capture_post_cb(camera, server, num_images, scan_id, seq_config, *,
                         async_=True, timeout=10.0):
    """Build a ``run_scan_group`` ``post_cb`` that reads one shot's frames and publishes them.

    Args:
        camera: an :class:`orca_camera.OrcaCamera` (or fake) -- armed for external trigger.
        server: an ExptServer-like hub exposing ``publish_shot(frames, scan_id, seq_id, async_=)``.
        num_images / timeout: forwarded to :func:`read_shot_frames`.
        scan_id: the scan id (an int, or a 0-arg callable returning it).
        seq_config: the :class:`SeqConfig`; ``seq_config.G.seq_id`` is read BEFORE the run loop
            bumps it (run_seq.py runs ``post_cb`` then ``_bump_seq_id``), matching MATLAB's
            ``server_post_run`` reading ``s1.G.seq_id``.
        async_: True (default) -> persist on the ExptServer worker (overlaps the next shot's
            hardware); False -> persist inline (kill-switch / A/B "before").

    The returned ``post_cb(cur_seq_num, arg0)`` reads the frames synchronously (timed as the
    ``cam_read`` sub-stage) then hands them to ``publish_shot``; a short read drops the shot
    (nothing is staged -- ``publish_shot`` is per-shot atomic, so no orphan in ``temp_imgs``).
    """
    def post_cb(cur_seq_num, arg0):  # noqa: ARG001 - run_scan_group's post_cb signature
        sid = scan_id() if callable(scan_id) else scan_id
        try:
            seq_id = int(seq_config.G.seq_id(1))
        except Exception:  # noqa: BLE001 - absent field -> default to 1
            seq_id = 1
        with run_timing.substage("cam_read"):
            frames = read_shot_frames(camera, num_images, timeout=timeout)
        if frames is None:
            return                       # short read -> drop this shot (nothing staged)
        server.publish_shot(frames, sid, seq_id, async_=async_)

    return post_cb
