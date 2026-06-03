"""Phase-5 orca_camera: the pylablib Orca wrapper, against a FAKE DCAM backend.

NO-HARDWARE: pylablib is never imported -- a fake DCAMCamera is injected. Covers the
handle_camera_cmd interface (init/apply_settings/current_roi/close), the [x,y,w,h] <-> pylablib
(hstart,hend,vstart,vend) ROI conversion, exposure/temperature, trigger modes, snap, and the
store_imgs column-major shaping (to_store_array) -- verified by reconstructing the frame with the
exact reshape yb_analysis _process_imgs uses. Also drives runner.handle_camera_cmd end-to-end
with a real OrcaCamera over the fake.
"""

import numpy as np
import pytest

import runner
from orca_camera import DEFAULT_ROI, OrcaCamera, to_store_array

pytestmark = pytest.mark.no_hardware


class FakeDCAM:
    """Mimics the subset of pylablib DCAMCamera that OrcaCamera uses."""

    def __init__(self):
        self._exposure = 0.03
        self._roi = (0, 4096, 0, 2304)          # (hstart, hend, vstart, vend)
        self._attrs = {"sensor_temperature": -20.0, "sensor_cooler_status": "ready",
                       "sensor_cooler": "max", "trigger_polarity": "negative"}
        self.trigger = None
        self.acquiring = False
        self.closed = False
        self.opened = 1                          # count of opens (for reconnect)
        self.setup_mode = None
        self._snap_frame = np.arange(6, dtype=np.uint16).reshape(2, 3)
        self._pending = [self._snap_frame, self._snap_frame]   # frames waiting to be read

    @property
    def trigger_polarity(self):
        return self._attrs["trigger_polarity"]

    def set_attribute_value(self, name, value):
        self._attrs[name] = value

    def get_exposure(self):
        return self._exposure

    def set_exposure(self, t):
        self._exposure = float(t)

    def get_roi(self):
        return self._roi

    def set_roi(self, hstart, hend, vstart, vend):
        self._roi = (hstart, hend, vstart, vend)

    def get_attribute_value(self, name):
        return self._attrs[name]

    def set_trigger_mode(self, mode):
        self.trigger = mode

    def snap(self):
        return self._snap_frame

    def setup_acquisition(self, mode=None, nframes=None):
        self.setup_mode = mode
        self._nframes = nframes

    def start_acquisition(self):
        self.acquiring = True

    def acquisition_in_progress(self):
        return self.acquiring

    def get_frames_status(self):
        from collections import namedtuple
        S = namedtuple("S", ["acquired", "unread", "skipped", "buffer_size"])
        return S(len(self._pending), len(self._pending), 0, 256)

    def read_multiple_images(self):
        out, self._pending = self._pending, []
        return out

    def stop_acquisition(self):
        self.acquiring = False

    def close(self):
        self.closed = True


def _cam(**kw):
    fake = FakeDCAM()
    cam = OrcaCamera(cam=fake, open_cam=lambda idx: _reopen(fake), **kw)
    return cam, fake


def _reopen(fake):
    """Reconnect factory for the fake: mark reopened + reset state."""
    fake.closed = False
    fake.opened += 1
    fake.acquiring = False
    return fake


# --------------------------------------------------------------------------- #
# ROI / exposure / temperature
# --------------------------------------------------------------------------- #
class TestSettings:
    def test_get_roi_xywh(self):
        cam, fake = _cam()
        fake._roi = (100, 612, 50, 562)         # hstart,hend,vstart,vend
        assert cam.get_roi() == [100, 50, 512, 512]   # [x, y, w, h]

    def test_set_roi_roundtrip(self):
        cam, fake = _cam()
        cam.set_roi([10, 20, 200, 100])
        assert fake._roi == (10, 210, 20, 120)        # hstart,hend,vstart,vend
        assert cam.get_roi() == [10, 20, 200, 100]

    def test_exposure(self):
        cam, fake = _cam()
        assert cam.set_exposure(0.05) == 0.05
        assert cam.get_exposure() == 0.05

    def test_temperature_and_cooler(self):
        cam, _ = _cam()
        assert cam.get_temperature() == -20.0
        assert cam.get_cooler_status() == "ready"

    def test_init_applies_and_reports(self):
        cam, _ = _cam()
        roi, exp = cam.init([0, 0, 256, 256], exposure=0.02)
        assert roi == [0, 0, 256, 256] and exp == 0.02

    def test_constructor_applies_initial(self):
        fake = FakeDCAM()
        cam = OrcaCamera(roi=[0, 0, 128, 64], exposure=0.01, cam=fake)
        assert cam.get_roi() == [0, 0, 128, 64] and cam.get_exposure() == 0.01


# --------------------------------------------------------------------------- #
# trigger + capture + close
# --------------------------------------------------------------------------- #
class TestCapture:
    def test_snap_uses_internal_trigger(self):
        cam, fake = _cam()
        frame = cam.snap()
        assert fake.trigger == "int"
        assert frame.shape == (2, 3)

    def test_external_acquisition_lifecycle(self):
        cam, fake = _cam()
        cam.start_acquisition(nframes=2)
        assert fake.trigger == "ext" and fake.acquiring is True
        assert fake.trigger_polarity == "positive"      # rising-edge forced (pylablib defaults negative)
        frames = cam.read_frames()
        assert len(frames) == 2
        cam.stop_acquisition()
        assert fake.acquiring is False

    def test_apply_settings_stops_then_leaves_stopped(self):
        cam, fake = _cam()
        cam.start_acquisition()
        roi, _ = cam.apply_settings([0, 0, 64, 64])
        assert fake.acquiring is False and roi == [0, 0, 64, 64]

    def test_close_releases_and_is_idempotent(self):
        cam, fake = _cam()
        cam.close()
        assert fake.closed is True
        cam.close()                              # second close is a no-op

    def test_close_stops_acquisition_first(self):
        cam, fake = _cam()
        cam.start_acquisition()
        cam.close()
        assert fake.acquiring is False and fake.closed is True


# --------------------------------------------------------------------------- #
# MATLAB IMAQ-parity arm/stop: start_video / stop_video / FramesAvailable / flush
# --------------------------------------------------------------------------- #
class TestVideoArmStop:
    def test_start_video_external_sets_polarity_and_runs(self):
        cam, fake = _cam()
        cam.start_video(nframes=64)
        assert fake.trigger == "ext" and fake.trigger_polarity == "positive"
        assert fake.setup_mode == "sequence" and fake._nframes == 64
        assert cam.is_running() is True

    def test_start_video_internal_freerun(self):
        cam, fake = _cam()
        cam.start_video(external=False)
        assert fake.trigger == "int"             # free-run, no FPGA trigger

    def test_stop_video(self):
        cam, fake = _cam()
        cam.start_video()
        cam.stop_video()
        assert fake.acquiring is False and cam.is_running() is False

    def test_frames_available_and_flush(self):
        cam, _ = _cam()
        assert cam.frames_available() == 2       # two seeded frames waiting
        assert cam.flush() == 2                  # flushdata drops them
        assert cam.frames_available() == 0


# --------------------------------------------------------------------------- #
# disconnect / reconnect
# --------------------------------------------------------------------------- #
class TestDisconnectReconnect:
    def test_disconnect_then_reconnect(self):
        cam, fake = _cam()
        assert cam.connected is True
        cam.disconnect()
        assert cam.connected is False and fake.closed is True
        cam.reconnect()
        assert cam.connected is True and fake.opened == 2

    def test_reconnect_is_noop_when_connected(self):
        cam, fake = _cam()
        cam.reconnect()
        assert fake.opened == 1                  # still connected -> no reopen


# --------------------------------------------------------------------------- #
# cooling
# --------------------------------------------------------------------------- #
class TestCooling:
    def test_get_cooler(self):
        cam, _ = _cam()
        assert cam.get_cooler() == "max"

    def test_set_cooler(self):
        cam, fake = _cam()
        assert cam.set_cooler("on") == "on"
        assert fake.get_attribute_value("sensor_cooler") == "on"


# --------------------------------------------------------------------------- #
# to_store_array -- column-major + 3-element shape prefix
# --------------------------------------------------------------------------- #
class TestStoreArray:
    def test_2d_frame_shape_prefix_and_colmajor(self):
        frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)   # (H=2, W=3)
        arr = to_store_array(frame)
        assert arr[0] == 2 and arr[1] == 3 and arr[2] == 1          # s1, s2, s3
        # Reconstruct exactly as yb_analysis _process_imgs does.
        s1, s2, s3 = int(arr[0]), int(arr[1]), int(arr[2])
        recon = arr[3:].reshape(s1, s2, s3, order="F")[:, :, 0]
        assert np.array_equal(recon, frame)

    def test_pixels_are_column_major(self):
        frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        arr = to_store_array(frame)
        # Fortran order of [[1,2,3],[4,5,6]] is column-by-column: 1,4,2,5,3,6.
        assert list(arr[3:]) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    def test_rejects_bad_ndim(self):
        with pytest.raises(ValueError):
            to_store_array(np.zeros((2, 2, 2, 2)))


# --------------------------------------------------------------------------- #
# end-to-end via runner.handle_camera_cmd (real OrcaCamera over the fake)
# --------------------------------------------------------------------------- #
class _CmdServer:
    def __init__(self, cmd):
        self._cmd = cmd
        self.results = []

    def get_camera_cmd(self):
        c, self._cmd = self._cmd, None
        return c

    def set_camera_result(self, connected, roi, error="", exposure_time=None):
        self.results.append((connected, list(roi), error, exposure_time))


class TestRunnerIntegration:
    def test_init_cmd_reports_actuals(self):
        cam, _ = _cam()
        srv = _CmdServer({"cmd": "init", "roi": [0, 0, 256, 256], "exposure_time": 0.02})
        runner.handle_camera_cmd(srv, cam)
        assert srv.results == [(True, [0, 0, 256, 256], "", 0.02)]

    def test_close_cmd_releases(self):
        cam, fake = _cam()
        srv = _CmdServer({"cmd": "close"})
        runner.handle_camera_cmd(srv, cam)
        assert fake.closed is True
        assert srv.results == [(False, [0, 0, 0, 0], "", None)]
