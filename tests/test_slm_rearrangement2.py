"""NO-HARDWARE tests for the TWO-ROUND SLM rearrangement seq (RearrangeCommSeq2) callbacks.

Complements test_slm_rearrangement.py (single-round). Focus: the three-frame / two-round runtime
and its ABORT / FRAME-ALIGNMENT invariant -- a shot either stages ALL THREE frames and publishes
them with ONE finish_shot, or it is cancelled and NO partial triple is persisted, and no produced
frame is ever left buffered to shift img1/img2/img3 by one on the next shot.

Never touches a real SLM server, camera, or the engine.
"""

import pytest

import rearrange_runtime

pytestmark = pytest.mark.no_hardware


# =========================================================================== #
# fakes
# =========================================================================== #
class FakeImgServer:
    """Sync stand-in for the ExptServer persister. stage_frame stages; finish_shot publishes the
    staged frames as one shot; cancel_shot drops the staged set. Records shot errors + ok."""

    def __init__(self):
        self.temp = []
        self.finished = []      # list of published shots (each a list of (frame, scan_id, seq_id))
        self.cancelled = 0
        self.errors = []
        self.oks = 0

    def stage_frame(self, frame, scan_id=-1, seq_id=-1, *, async_=True):
        self.temp.append((frame, scan_id, seq_id))

    def finish_shot(self, *, async_=True):
        self.finished.append(list(self.temp))
        self.temp = []

    def cancel_shot(self, *, async_=True):
        self.cancelled += 1
        self.temp = []

    def record_shot_error(self, message, scan_id=None, seq_id=None, kind=None):
        self.errors.append((kind, seq_id, message))

    def record_shot_ok(self):
        self.oks += 1


class FakeReClient:
    """Records rearrange / update / lock calls; scriptable rearrange results / raises per round."""

    def __init__(self, results=None, raises=None):
        self.calls = []
        # results[i] / raises[i] apply to the i-th rearrange() call (0 = round 1, 1 = round 2).
        self._results = list(results) if results is not None else []
        self._raises = list(raises) if raises is not None else []
        self._i = 0

    def rearrange(self, probs, **kw):
        i = self._i
        self._i += 1
        self.calls.append(("rearrange", list(probs), kw))
        if i < len(self._raises) and self._raises[i]:
            raise RuntimeError("rearrange boom round %d" % (i + 1))
        if i < len(self._results) and self._results[i] is not None:
            return self._results[i]
        return {"ok": True}

    def update_rearrange(self, results, **kw):
        self.calls.append(("update", results, kw))

    def cancel_last_shot(self, **kw):
        self.calls.append(("cancel_last", kw))

    def release_lock(self, device="all"):
        self.calls.append(("release", device))


class MultiFrameCam:
    """Yields the scripted frames one per grab (read_frames returns [frame] then [] when drained).
    current_roi feeds the detector roi_provider (unused here -- detect_* are monkeypatched)."""

    def __init__(self, frames):
        self._frames = list(frames)

    def read_frames(self):
        return [self._frames.pop(0)] if self._frames else []

    def current_roi(self):
        return [0, 0, 8, 8]


def _ctx(server, client, cam, *, frame_patterns=None, probs_by_pattern=None, bits_by_pattern=None):
    """A ScanContext with the per-pattern detectors monkeypatched so a test controls exactly what
    each frame detects (keyed by the pattern name passed to detect_*_for)."""
    ctx = rearrange_runtime.ScanContext(
        session=_KeepaliveSession(), camera=cam, server=server, client=client,
        scan_id="20260630120000", n_rounds=2, frame_patterns=frame_patterns)
    pbp = probs_by_pattern or {}
    bbp = bits_by_pattern or {}
    ctx.detect_probs_for = lambda pat, _img: list(pbp.get(pat, [1.0]))
    ctx.detect_bits_for = lambda pat, _img: bbp.get(pat, "1")
    return ctx


class _KeepaliveSession:
    def __init__(self):
        self.keepalives = 0

    def keepalive(self):
        self.keepalives += 1

    def ensure_held(self):
        pass


def _s1(lock_ok=True, img1_ok=False, img2_ok=False):
    from dyn_props import DynProps
    s1 = type("S1", (), {})()
    s1.G = DynProps({})
    s1.G.rearrange_lock_ok = lock_ok
    s1.G.rearrange_img1_ok = img1_ok
    s1.G.rearrange_img2_ok = img2_ok
    s1.G.seq_id = 7
    return s1


# =========================================================================== #
# happy path: 3 frames -> ONE published triple, 2 rearranges, compute released
# =========================================================================== #
def test_two_round_happy_path_stages_aligned_triple():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]),
               frame_patterns=["load", "mid", "final"],
               bits_by_pattern={"final": "101"})
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)                 # img1 -> rearrange round 1 -> stage img1
        assert s1.G.rearrange_img1_ok(False) is True
        R.hand_over_slm_2(s1)               # img2 -> rearrange round 2 -> stage img2
        assert s1.G.rearrange_img2_ok(False) is True
        R.post_run(s1)                      # img3 -> update -> stage img3 -> finish
    finally:
        rearrange_runtime.clear_context()
    # exactly one published shot, three frames, all real scan_id, in order img1,img2,img3.
    assert len(server.finished) == 1
    triple = server.finished[0]
    assert len(triple) == 3
    assert [int(f[0, 0]) for (f, _sid, _sq) in triple] == [1, 2, 3]
    assert all(sid == "20260630120000" for (_f, sid, _sq) in triple)
    assert server.cancelled == 0
    # two rearrange calls (round 1 + round 2) + one update_rearrange (final) + compute released.
    assert sum(1 for c in client.calls if c[0] == "rearrange") == 2
    assert any(c[0] == "update" for c in client.calls)
    assert ("release", "compute") in client.calls
    assert ctx.session.keepalives == 1
    assert server.oks == 1


def test_two_round_detects_each_frame_with_its_own_pattern():
    """img1 detected against LOADING, img2 against MIDDLE, img3 against FINAL -- the per-pattern
    (independent-grid) requirement."""
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    seen = []
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]),
               frame_patterns=["load", "mid", "final"])
    ctx.detect_probs_for = lambda pat, _img: (seen.append(("probs", pat)) or [1.0])
    ctx.detect_bits_for = lambda pat, _img: (seen.append(("bits", pat)) or "1")
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)
        R.hand_over_slm_2(s1)
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()
    assert seen == [("probs", "load"), ("probs", "mid"), ("bits", "final")]


# =========================================================================== #
# abort: round-1 rearrange fails -> NO partial persist, img2+img3 drained, display-only
# =========================================================================== #
def test_round1_failure_drains_all_frames_no_persist():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    cam = MultiFrameCam([f1, f2, f3])
    server = FakeImgServer()
    client = FakeReClient(raises=[True])       # round 1 raises
    ctx = _ctx(server, client, cam, frame_patterns=["load", "mid", "final"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)                     # round 1 raises -> stage img1 then drop it
        assert s1.G.rearrange_img1_ok(False) is False
        assert any(c[0] == "cancel_last" for c in client.calls)   # phantom ledger row dropped
        R.hand_over_slm_2(s1)                   # img1 not ok -> DRAIN img2, cancel, no round 2
        assert s1.G.rearrange_img2_ok(False) is False
        assert sum(1 for c in client.calls if c[0] == "rearrange") == 1   # round 2 never ran
        R.post_run(s1)                          # DRAIN img3, publish display-only, no persist
    finally:
        rearrange_runtime.clear_context()
    # every produced frame consumed -> camera fully drained (no straggler for the next shot).
    assert cam.read_frames() == []
    # nothing persisted as a real triple; the display re-publish is under the failing sentinel.
    assert len(server.finished) == 1
    published = server.finished[0]
    assert all(sid == FAILING_DISPLAY_SCAN_ID for (_f, sid, _sq) in published)
    # img1, img2, img3 all captured -> all three flashed for display.
    assert [int(f[0, 0]) for (f, _sid, _sq) in published] == [1, 2, 3]
    assert ("release", "compute") in client.calls
    assert ctx.session.keepalives == 1


# =========================================================================== #
# abort: round-2 rearrange fails -> img1 dropped, img3 drained, display-only, no persist
# =========================================================================== #
def test_round2_failure_no_offby1_no_partial_persist():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    cam = MultiFrameCam([f1, f2, f3])
    server = FakeImgServer()
    client = FakeReClient(raises=[False, True])   # round 1 ok, round 2 raises
    ctx = _ctx(server, client, cam, frame_patterns=["load", "mid", "final"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)                        # round 1 ok -> img1 staged (real)
        assert s1.G.rearrange_img1_ok(False) is True
        R.hand_over_slm_2(s1)                      # round 2 raises -> img2 staged then dropped
        assert s1.G.rearrange_img2_ok(False) is False
        R.post_run(s1)                             # DRAIN img3, display-only, no persist
    finally:
        rearrange_runtime.clear_context()
    assert cam.read_frames() == []                 # no leftover produced frame
    assert len(server.finished) == 1
    published = server.finished[0]
    assert all(sid == FAILING_DISPLAY_SCAN_ID for (_f, sid, _sq) in published)
    assert [int(f[0, 0]) for (f, _sid, _sq) in published] == [1, 2, 3]
    # round 2 committed server-side before the downstream failure -> phantom ledger row dropped.
    assert any(c[0] == "cancel_last" for c in client.calls)


# =========================================================================== #
# abort: both rounds ok but img3 lost -> display img1/img2, NO persist, frame stream stays sane
# =========================================================================== #
def test_img3_lost_publishes_img1_img2_display_only():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1, f2 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2))
    cam = MultiFrameCam([f1, f2])                  # only two frames ever arrive (img3 lost)
    server = FakeImgServer()
    client = FakeReClient()                        # both rounds ok
    ctx = _ctx(server, client, cam, frame_patterns=["load", "mid", "final"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)
        R.hand_over_slm_2(s1)
        assert s1.G.rearrange_img1_ok(False) is True
        assert s1.G.rearrange_img2_ok(False) is True
        R.post_run(s1)                             # img3 grab fails -> display img1/img2
    finally:
        rearrange_runtime.clear_context()
    assert len(server.finished) == 1
    published = server.finished[0]
    assert len(published) == 2                      # only img1 + img2 (img3 == no data)
    assert all(sid == FAILING_DISPLAY_SCAN_ID for (_f, sid, _sq) in published)
    assert any(kind == "frame_timeout" for (kind, _sq, _m) in server.errors)


# =========================================================================== #
# abort: img1 grab fails -> cancel, no rearrange; downstream rounds also cancel cleanly
# =========================================================================== #
def test_img1_unavailable_cancels_whole_shot():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    cam = MultiFrameCam([])                         # never yields img1
    server = FakeImgServer()
    client = FakeReClient()
    ctx = _ctx(server, client, cam, frame_patterns=["load", "mid", "final"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=True)
        R.hand_over_slm(s1)                         # img1 grab fails -> cancel, no rearrange
        assert not any(c[0] == "rearrange" for c in client.calls)
        R.hand_over_slm_2(s1)                       # img1 not ok -> drain (nothing) + cancel
        R.post_run(s1)                              # nothing captured -> no persist
    finally:
        rearrange_runtime.clear_context()
    assert server.cancelled >= 1
    assert not any(f for f in server.finished if any(sid == "20260630120000"
                   for (_x, sid, _y) in f))         # never persisted a real triple


# =========================================================================== #
# lock miss: pre_run couldn't get compute -> callbacks are inert (no rearrange, no persist)
# =========================================================================== #
def test_no_compute_lock_callbacks_are_inert():
    np = pytest.importorskip("numpy")
    import RearrangeCommSeq2 as R
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    server = FakeImgServer()
    client = FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]),
               frame_patterns=["load", "mid", "final"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lock_ok=False)                     # compute lock not held
        R.hand_over_slm(s1)
        R.hand_over_slm_2(s1)
        assert not any(c[0] == "rearrange" for c in client.calls)
    finally:
        rearrange_runtime.clear_context()


# =========================================================================== #
# per-pattern detector cache (rearrange_runtime): independent grids per pattern
# =========================================================================== #
def test_detector_cache_independent_per_pattern(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    import time
    from scipy.io import savemat

    # Day-folder calibration (the fallback source, pattern registry absent in tmp): two sites.
    day = tmp_path / time.strftime("%Y%m%d")
    day.mkdir()
    (day / "gridLocations.txt").write_text("Y\tX\n10\t10\n10\t30\n")
    savemat(str(day / "threshold.mat"), {"thresholds": np.array([5.0, 1.0e9])})

    ctx = rearrange_runtime.ScanContext(
        session=None, camera=None, server=None, client=None, scan_id="1",
        n_rounds=2, calib_root=str(tmp_path))
    # Distinct pattern names -> distinct cached detector objects (independent grids).
    d_load = ctx.detector_for("load")
    d_mid = ctx.detector_for("mid")
    assert d_load is not d_mid
    assert ctx.detector_for("load") is d_load       # cached (same object on re-request)
    # None -> the frame-0 detector.
    assert ctx.detector_for(None) is ctx._detector
