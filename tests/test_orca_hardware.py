"""Phase-5 Orca camera LIVE checks (needs_hardware -- excluded by default).

Run only when the camera is free of MATLAB (DCAM is one-handle-per-camera) and pylablib is
installed (the .venv-engine has it):

    cd pyctrl; .venv-engine/Scripts/python -m pytest -m needs_hardware tests/test_orca_hardware.py

Read-only connectivity + a single INTERNAL-trigger snap (no FPGA, no external trigger). The
FPGA-triggered (TTL54) capture is deliberately NOT here -- it drives the FPGA via set_chns and
belongs in a gated downtime run, not the test suite.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_hardware


@pytest.fixture
def cam():
    from orca_camera import OrcaCamera
    try:
        c = OrcaCamera()
    except Exception as e:  # noqa: BLE001 - no camera / held by MATLAB
        pytest.skip("Orca not available (held by MATLAB / powered off?): %s" % e)
    yield c
    c.close()


def test_connectivity_readonly(cam):
    """Model + cooled sensor + sane exposure/ROI (read-only)."""
    t = cam.get_temperature()
    assert -50.0 < t < 40.0, "implausible sensor temperature %r" % t
    assert cam.get_exposure() > 0
    x, y, w, h = cam.current_roi()
    assert w > 0 and h > 0 and w <= 4096 and h <= 2304


def test_internal_snap_and_store_format(cam):
    """One internal-trigger frame; to_store_array round-trips (the store_imgs wire format)."""
    from orca_camera import to_store_array
    cam.set_roi([0, 0, 128, 128])
    cam.set_exposure(0.01)
    frame = np.asarray(cam.snap())                 # internal/software trigger -- no FPGA
    assert frame.shape == (128, 128)
    arr = to_store_array(frame)
    s1, s2, s3 = int(arr[0]), int(arr[1]), int(arr[2])
    assert (s1, s2, s3) == (128, 128, 1)
    assert arr.size == 3 + s1 * s2 * s3
    recon = arr[3:].reshape(s1, s2, s3, order="F")[:, :, 0]
    assert np.array_equal(recon, frame)


def test_start_stop_video_freerun(cam):
    """MATLAB start(vid)/stop(vid) parity via the internal free-run trigger (NO FPGA).

    Frames accumulate in the buffer while armed (FramesAvailable climbs), read/flush drain
    them, stop halts. No external trigger -> no TTL -> no FPGA.
    """
    import time
    cam.set_roi([0, 0, 256, 256])
    cam.set_exposure(0.003)
    cam.start_video(external=False, nframes=64)    # internal free-run
    try:
        assert cam.is_running() is True
        time.sleep(0.15)
        assert cam.frames_available() > 0          # buffer filling
        got = cam.read_frames()
        assert got and np.asarray(got[0]).shape == (256, 256)
        cam.flush()
    finally:
        cam.stop_video()
    assert cam.is_running() is False


def test_disconnect_reconnect(cam):
    """Release the DCAM handle and reopen it (re-init parity)."""
    cam.disconnect()
    assert cam.connected is False
    cam.reconnect()
    assert cam.connected is True
    assert -50.0 < cam.get_temperature() < 40.0
