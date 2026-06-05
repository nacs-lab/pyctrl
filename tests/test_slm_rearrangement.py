"""NO-HARDWARE tests for the SLM rearrangement scan plumbing.

Covers the pure logic of the new pieces with fakes -- never touches a real SLM server, camera, or
the engine:
  * slm_client       -- HTTP body shaping + the server-side-blocking acquire retry/raise.
  * slm_scan_session -- the scan-long lock state machine (begin / keepalive / ensure_held /
                        pause / resume / mandatory-acquire-raises).
  * rearrange_runtime -- collect_kwargs / translate_zernike_zN, grab_one_frame, the pause/resume
                        hooks, and the ported atom detector (sparse matvec bits).
"""

import pytest

from devices.slm.slm_client import SlmClient, SlmHTTPError, _build_setup_body, _encode_bits
from devices.slm import SlmScanSession, SlmLockUnavailable
import rearrange_runtime

pytestmark = pytest.mark.no_hardware


# =========================================================================== #
# fakes
# =========================================================================== #
class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Stand-in for a requests.Session: records calls, returns scripted responses per path."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = responses or {}

    def _resp(self, url):
        path = url.split("8551", 1)[-1] if "8551" in url else url
        r = self._responses.get(path)
        if isinstance(r, list):
            return r.pop(0) if r else FakeResp(200, {})
        return r if r is not None else FakeResp(200, {})

    def post(self, url, json=None, headers=None, timeout=None, verify=None):
        self.calls.append(("POST", url, json, headers))
        return self._resp(url)

    def get(self, url, headers=None, timeout=None, verify=None):
        self.calls.append(("GET", url, None, headers))
        return self._resp(url)


def _mkclock(start=0.0, step=1.0):
    t = [start]

    def clock():
        v = t[0]
        t[0] += step
        return v
    return clock


# =========================================================================== #
# slm_client: body shaping
# =========================================================================== #
def test_setup_body_phase_filepath_and_extras_merge():
    body = _build_setup_body({
        "model_filename": "m.pth",
        "initial_phase": "phase/a.pt",
        "final_phase": "phase/b.pt",
        "nsteps": 50,
        "reset_params": True,
        "extras": {"grid_rotation": 90, "z4": -4, "skip": None},
        "ignored_none": None,
    })
    assert body["initial_phase_filepath"] == "phase/a.pt"
    assert body["final_phase_filepath"] == "phase/b.pt"
    assert body["model_filename"] == "m.pth"
    assert body["nsteps"] == 50 and body["reset_params"] is True
    assert body["grid_rotation"] == 90 and body["z4"] == -4   # extras merged top-level
    assert "extras" not in body and "skip" not in body and "ignored_none" not in body


def test_encode_bits_string_logical_and_index_list():
    assert _encode_bits("0101") == "0101"
    assert _encode_bits([0, 1, 1, 0]) == "0110"
    assert _encode_bits([3, 17, 42]) == [3, 17, 42]          # any value > 1 -> index list
    assert _encode_bits({"indices": [1, 2], "n": 9}) == {"indices": [1, 2], "n": 9}


def test_rearrange_body_stamps_runid_and_bits():
    sess = FakeSession()
    c = SlmClient(session=sess, client_id="cid")
    c.rearrange("0110", scan_id="20260605120000", seq_id=7)
    _, url, body, headers = sess.calls[-1]
    assert url.endswith("/slm/rearrange")
    assert body == {"bits": "0110", "scan_id": "20260605120000", "seq_id": 7}
    assert headers["X-Client-Id"] == "cid"


def test_write_loading_phase_body():
    sess = FakeSession()
    c = SlmClient(session=sess)
    c.write_loading_phase("phase/x.pt", [0, 0, 0, 0, -5], name="x",
                          legacy_zerniked=True, baked_zernike=[0, 0, 0, 0, -4])
    _, url, body, _ = sess.calls[-1]
    assert url.endswith("/slm/write_loading_phase")
    assert body["phase_filepath"] == "phase/x.pt"
    assert body["loading_zernike"] == [0.0, 0.0, 0.0, 0.0, -5.0]
    assert body["baked_zernike"] == [0.0, 0.0, 0.0, 0.0, -4.0]
    assert body["legacy_zerniked"] is True and body["name"] == "x"


# =========================================================================== #
# slm_client: acquire_lock blocking semantics
# =========================================================================== #
def test_acquire_lock_retries_on_423_then_succeeds():
    sess = FakeSession({"/lock/acquire": [FakeResp(423, {"detail": "busy"}),
                                          FakeResp(200, {"ok": True})]})
    c = SlmClient(session=sess)
    r = c.acquire_lock("slm", "run", block_timeout=30,
                       clock=_mkclock(step=0.0), sleep=lambda _s: None)
    assert r == {"ok": True}
    assert len(sess.calls) == 2
    # the request asks the server to block (block=True + a positive budget).
    _, _, body, _ = sess.calls[0]
    assert body["device"] == "slm" and body["block"] is True and body["block_timeout_s"] > 0


def test_acquire_lock_raises_after_deadline():
    sess = FakeSession({"/lock/acquire": [FakeResp(423, {"detail": "busy"})]})
    c = SlmClient(session=sess)
    with pytest.raises(SlmHTTPError) as ei:
        c.acquire_lock("slm", "run", block_timeout=0,
                       clock=_mkclock(step=1.0), sleep=lambda _s: None)
    assert ei.value.status == 423


# =========================================================================== #
# slm_scan_session: state machine
# =========================================================================== #
class FakeClient:
    def __init__(self, acquire_fail=False):
        self.log = []
        self.acquire_fail = acquire_fail

    def acquire_lock(self, device, description="", timeout_s=60, block_timeout=30):
        self.log.append(("acquire", device, timeout_s, block_timeout))
        if self.acquire_fail:
            raise SlmHTTPError(423, "busy")

    def release_lock(self, device="all"):
        self.log.append(("release", device))

    def heartbeat(self, device="all"):
        self.log.append(("heartbeat", device))

    def write_loading_phase(self, phase_path, loading_zernike=None, name=None,
                            legacy_zerniked=False, baked_zernike=None):
        self.log.append(("write", phase_path))


class StepClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _session(client, clk):
    s = SlmScanSession(client, lease_s=10.0, acquire_block_s=5.0, clock=clk)
    s.set_loading_pattern("33x33", "phase/33.pt", [0, 0, 0, 0, -5])
    return s


def test_begin_acquires_and_writes_once():
    c = FakeClient()
    s = _session(c, StepClock())
    s.begin()
    assert ("acquire", "slm", 10.0, 5.0) in c.log
    assert ("write", "phase/33.pt") in c.log
    # begin twice with the same pattern -> no second write (write-on-change).
    c.log.clear()
    s._last_ok_t = 0.0   # keep "held & fresh"
    s.begin()
    assert ("write", "phase/33.pt") not in c.log


def test_begin_raises_when_lock_unavailable():
    c = FakeClient(acquire_fail=True)
    s = _session(c, StepClock())
    with pytest.raises(SlmLockUnavailable):
        s.begin()
    assert s.is_held() is False


def test_keepalive_is_single_heartbeat_no_write():
    c = FakeClient()
    s = _session(c, StepClock())
    s.begin()
    c.log.clear()
    s.keepalive()
    assert c.log == [("heartbeat", "slm")]          # exactly one call, no phase write


def test_ensure_held_noop_when_fresh_regrab_when_lapsed():
    c = FakeClient()
    clk = StepClock()
    s = _session(c, clk)
    s.begin()
    c.log.clear()
    clk.t = 5.0                                     # within the 10 s lease
    s.ensure_held()
    assert c.log == []                              # NO SLM comm on the fresh path
    clk.t = 25.0                                    # lease lapsed
    s.ensure_held()
    assert ("acquire", "slm", 10.0, 5.0) in c.log   # regrab
    assert ("write", "phase/33.pt") in c.log        # + rewrite the WGS phase


def test_pause_drops_and_resume_regrabs():
    c = FakeClient()
    s = _session(c, StepClock())
    s.begin()
    c.log.clear()
    s.on_pause()
    assert c.log == [("release", "slm")] and s.is_held() is False
    # keepalive after a drop is a no-op (not held).
    s.keepalive()
    assert c.log == [("release", "slm")]
    c.log.clear()
    s.on_resume()
    assert ("acquire", "slm", 10.0, 5.0) in c.log and ("write", "phase/33.pt") in c.log
    assert s.is_held() is True


# =========================================================================== #
# rearrange_runtime: kwargs helpers
# =========================================================================== #
def test_collect_kwargs_leaves_extras_and_skips_nested():
    out = rearrange_runtime.collect_kwargs({
        "nsteps": 50, "protocol": "rearrange",
        "extras": {"z4": -4, "pattern": "every-other"},
        "nested_namespace": {"a": 1},
    })
    assert out["nsteps"] == 50 and out["protocol"] == "rearrange"
    assert out["extras"] == {"z4": -4, "pattern": "every-other"}
    assert "nested_namespace" not in out


def test_translate_zernike_zN_bundles_into_coeffs():
    out = rearrange_runtime.translate_zernike_zN(
        {"extras": {"z4": -4, "z2": 1.5, "pattern": "x"}})
    ex = out["extras"]
    assert ex["zernike_coeffs"] == [0.0, 0.0, 1.5, 0.0, -4.0]
    assert "z4" not in ex and "z2" not in ex and ex["pattern"] == "x"


def test_translate_zernike_respects_explicit_coeffs():
    out = rearrange_runtime.translate_zernike_zN(
        {"extras": {"z4": -4, "zernike_coeffs": [1, 2, 3]}})
    assert out["extras"]["zernike_coeffs"] == [1, 2, 3]   # explicit wins


# =========================================================================== #
# rearrange_runtime: grab_one_frame
# =========================================================================== #
class FakeCam:
    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.i = 0

    def read_frames(self):
        if self.i < len(self.schedule):
            r = self.schedule[self.i]
            self.i += 1
            return r
        return []


def test_grab_one_frame_exactly_one():
    cam = FakeCam([[], ["IMG"]])
    img, ok = rearrange_runtime.grab_one_frame(
        cam, timeout=0.1, sleep=lambda _s: None, clock=_mkclock(step=0.02))
    assert ok is True and img == "IMG"


def test_grab_one_frame_timeout_returns_false():
    cam = FakeCam([[]])
    img, ok = rearrange_runtime.grab_one_frame(
        cam, timeout=0.1, sleep=lambda _s: None, clock=_mkclock(step=0.02))
    assert ok is False and img is None


def test_grab_one_frame_surplus_is_rejected():
    cam = FakeCam([["A", "B"]])                     # two stale frames -> not exactly one
    img, ok = rearrange_runtime.grab_one_frame(
        cam, timeout=0.1, sleep=lambda _s: None, clock=_mkclock(step=0.02))
    assert ok is False and img is None


# =========================================================================== #
# rearrange_runtime: pause/resume hooks route to the active session
# =========================================================================== #
class HookSpySession:
    def __init__(self):
        self.events = []

    def on_pause(self):
        self.events.append("pause")

    def on_resume(self):
        self.events.append("resume")


def test_pause_resume_hooks_route_to_session(monkeypatch):
    spy = HookSpySession()
    ctx = rearrange_runtime.ScanContext(session=spy, camera=None, server=None,
                                        client=None, scan_id=1)
    rearrange_runtime.set_context(ctx)
    try:
        rearrange_runtime.on_pause()
        rearrange_runtime.on_resume()
        assert spy.events == ["pause", "resume"]
    finally:
        rearrange_runtime.clear_context()
    # no active context -> hooks are silent no-ops
    rearrange_runtime.on_pause()
    rearrange_runtime.on_resume()
    assert spy.events == ["pause", "resume"]


# =========================================================================== #
# rearrange_runtime: the ported atom detector (sparse matvec bits)
# =========================================================================== #
def test_detector_bits_day_folder(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    import time
    from scipy.io import savemat
    from rearrange_runtime import _Detector

    # Real day-folder calibration: two sites, one low threshold (fires) + one huge (never fires).
    day = tmp_path / time.strftime("%Y%m%d")
    day.mkdir()
    (day / "gridLocations.txt").write_text("Y\tX\n10\t10\n10\t30\n")
    savemat(str(day / "threshold.mat"), {"thresholds": np.array([5.0, 1.0e9])})

    img = np.zeros((40, 40))
    img[5:14, 5:14] = 100.0          # light the 9x9 box around site 0 (Y=10, X=10), 1-based

    det = _Detector(str(tmp_path))   # pattern_name=None -> day-folder source
    assert det.bits(img) == "10"
