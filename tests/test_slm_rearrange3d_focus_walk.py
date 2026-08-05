"""NO-HARDWARE tests for the 3-D focus-walk rearrangement seq (Rearrange3DFocusWalkCommSeq).

Sibling of test_slm_rearrangement2.py (whose fakes this file mirrors). What is specific here and
worth pinning:

  * the COMPOSED bit vectors: the walk call posts [back | zeros] and the 3-D call posts
    [back | front] over the server's 242-site (2 x 121) init_grid, with the back half measured
    BEFORE the axial walk and the front half after it;
  * the FINAL report pads the emptied back layer: ``"0"*121 + img3 bits``;
  * the SECOND per-shot setup_rearrangement (protocol rearrange + the WGS bookend re-write at the
    post-walk carrier) is pushed mid-shot, between img2 and the 3-D rearrange call;
  * walk-only mode: ONE server call (the walk), no second setup, no update_rearrange, and the
    triple still publishes (that is what bootstraps a new array's thresholds);
  * the frame-alignment invariant: a failing shot drains every produced frame and persists nothing.

Never touches a real SLM server, camera, or the engine.
"""

import pytest

import rearrange_runtime

pytestmark = pytest.mark.no_hardware

N = 4                       # sites per layer in these tests (production: 121)
BACK = [1.0, 0.0, 1.0, 0.0]
FRONT = [0.0, 1.0, 1.0, 0.0]
POST_WALK_Z4 = -17.531328320802004


# =========================================================================== #
# fakes
# =========================================================================== #
class FakeImgServer:
    def __init__(self):
        self.temp = []
        self.finished = []
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
    """Records rearrange / setup / update / lock calls; scriptable raises per rearrange call."""

    def __init__(self, raises=None):
        self.calls = []
        self._raises = list(raises) if raises is not None else []
        self._i = 0

    def rearrange(self, probs, **kw):
        i = self._i
        self._i += 1
        self.calls.append(("rearrange", [float(p) for p in probs], kw))
        if i < len(self._raises) and self._raises[i]:
            raise RuntimeError("rearrange boom call %d" % (i + 1))
        return {"ok": True}

    def setup_rearrangement(self, **kw):
        self.calls.append(("setup", kw))

    def update_rearrange(self, results, **kw):
        self.calls.append(("update", results, kw))

    def cancel_last_shot(self, **kw):
        self.calls.append(("cancel_last", kw))

    def release_lock(self, device="all"):
        self.calls.append(("release", device))


class MultiFrameCam:
    def __init__(self, frames):
        self._frames = list(frames)

    def read_frames(self):
        return [self._frames.pop(0)] if self._frames else []

    def current_roi(self):
        return [0, 0, 8, 8]


class _KeepaliveSession:
    def __init__(self):
        self.keepalives = 0

    def keepalive(self):
        self.keepalives += 1

    def ensure_held(self):
        pass


PAT = "2x11x11_5um_z20um_layer"


def _ctx(server, client, cam, *, probs=None, bits="1101"):
    """A ScanContext whose per-pattern detection is scripted per FRAME (all three frames declare
    the SAME single-plane pattern, so the script is a per-call queue, not a per-pattern map)."""
    ctx = rearrange_runtime.ScanContext(
        session=_KeepaliveSession(), camera=cam, server=server, client=client,
        scan_id="20260729120000", n_rounds=1, frame_patterns=[PAT, PAT, PAT],
        loading_defocus=7.531328320802004)
    queue = list(probs if probs is not None else [BACK, FRONT])
    seen = []

    def _probs(pat, _img):
        seen.append(pat)
        return list(queue.pop(0)) if queue else []

    ctx.detect_probs_for = _probs
    ctx.detect_bits_for = lambda pat, _img: bits
    ctx.detected_patterns = seen
    return ctx


def _s1(*, lock_ok=True, walk_only=False, n_per_plane=N, kwargs2=None):
    from dyn_props import DynProps
    s1 = type("S1", (), {})()
    s1.G = DynProps({})
    s1.G.rearrange_lock_ok = lock_ok
    s1.G.focus_walk_ok = False
    s1.G.rearrange_front_ok = False
    s1.G.seq_id = 7
    rk = {"extras": {}}
    if walk_only:
        rk["extras"]["walk_only"] = True
    rk2 = kwargs2 if kwargs2 is not None else {
        "protocol": "rearrange", "nsteps": 40, "step_period_ms": 0.696,
        "final_phase": "phase/2x11x11_5um_z20um.pt", "skip_grid_derive": True,
        "extras": {"loading_zernike": [0, 0, 0, 0, POST_WALK_Z4], "z4": POST_WALK_Z4,
                   "pattern": "front_layer"}}
    s1.C = DynProps({"rearrange_kwargs": rk, "rearrange_kwargs2": rk2,
                     "focus_walk": {"n_per_plane": n_per_plane}})
    return s1


def _frames(np, n=3):
    return [np.full((8, 8), k, dtype=np.uint16) for k in range(1, n + 1)]


# =========================================================================== #
# happy path
# =========================================================================== #
def test_full_flow_composes_both_layers_and_publishes_one_triple():
    np = pytest.importorskip("numpy")
    import Rearrange3DFocusWalkCommSeq as R
    f1, f2, f3 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]))
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1()
        R.walk_to_front(s1)                 # img1 -> back probs -> the axial walk
        assert s1.G.focus_walk_ok(False) is True
        R.rearrange_to_front(s1)            # img2 -> front probs -> setup#2 -> 3-D rearrange
        assert s1.G.rearrange_front_ok(False) is True
        R.post_run(s1)                      # img3 -> update -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    # ONE published shot of THREE frames, in order, under the real scan id.
    assert len(server.finished) == 1
    triple = server.finished[0]
    assert [int(f[0, 0]) for (f, _s, _q) in triple] == [1, 2, 3]
    assert all(sid == "20260729120000" for (_f, sid, _q) in triple)
    assert server.cancelled == 0
    assert server.oks == 1

    # the two server rounds, with the COMPOSED vectors.
    rounds = [c for c in client.calls if c[0] == "rearrange"]
    assert len(rounds) == 2
    assert rounds[0][1] == BACK + [0.0] * N          # walk: back half only
    assert rounds[1][1] == BACK + FRONT              # 3-D: both layers
    # every frame detected against the single-plane pattern.
    assert ctx.detected_patterns == [PAT, PAT]

    # the mid-shot setup #2 ran BETWEEN the two rounds, and carries the post-walk bookend carrier.
    kinds = [c[0] for c in client.calls]
    assert kinds.index("setup") > kinds.index("rearrange")
    assert kinds.index("setup") < len(kinds) - 1
    setup = [c for c in client.calls if c[0] == "setup"][0][1]
    assert setup["protocol"] == "rearrange"
    assert setup["skip_grid_derive"] is True
    assert setup["final_phase"] == "phase/2x11x11_5um_z20um.pt"
    assert setup["extras"]["loading_zernike"] == [0, 0, 0, 0, POST_WALK_Z4]
    # extras.z4 is folded into the server's bundled model-Zernike vector (translate_zernike_zN),
    # so the model frames and the re-written bookend end up on the SAME post-walk plane.
    assert setup["extras"]["zernike_coeffs"] == [0.0, 0.0, 0.0, 0.0, POST_WALK_Z4]
    assert "z4" not in setup["extras"]
    assert setup["client_scan_id"] == "20260729120000"

    # the final report pads the emptied BACK layer.
    upd = [c for c in client.calls if c[0] == "update"][0]
    assert upd[1] == "0000" + "1101"
    assert ("release", "compute") in client.calls
    assert ctx.session.keepalives == 1


# =========================================================================== #
# walk-only (bootstrap / walk-survival mode)
# =========================================================================== #
def test_walk_only_does_one_call_no_setup_no_update_but_publishes():
    np = pytest.importorskip("numpy")
    import Rearrange3DFocusWalkCommSeq as R
    f1, f2, f3 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    # No detection at all (un-bootstrapped thresholds) -- walk-only must not care.
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]), probs=[[], []])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(walk_only=True)
        R.walk_to_front(s1)
        R.rearrange_to_front(s1)
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()

    assert len(server.finished) == 1 and len(server.finished[0]) == 3
    assert server.cancelled == 0
    rounds = [c for c in client.calls if c[0] == "rearrange"]
    assert len(rounds) == 1                      # the walk only
    assert rounds[0][1] == [0.0] * (2 * N)       # sized from focus_walk.n_per_plane
    assert not [c for c in client.calls if c[0] == "setup"]
    assert not [c for c in client.calls if c[0] == "update"]
    assert not server.errors                     # walk-only must not flag the shot failing


# =========================================================================== #
# abort / frame alignment
# =========================================================================== #
def test_walk_failure_drains_every_frame_and_persists_nothing():
    np = pytest.importorskip("numpy")
    import Rearrange3DFocusWalkCommSeq as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1, f2, f3 = _frames(np)
    cam = MultiFrameCam([f1, f2, f3])
    server, client = FakeImgServer(), FakeReClient(raises=[True])   # the walk call raises
    ctx = _ctx(server, client, cam)
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1()
        R.walk_to_front(s1)
        assert s1.G.focus_walk_ok(False) is False
        R.rearrange_to_front(s1)
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()

    # nothing persisted, every produced frame consumed (the camera is drained).
    assert not [sh for sh in server.finished
                if all(sid != FAILING_DISPLAY_SCAN_ID for (_f, sid, _q) in sh)]
    assert cam.read_frames() == []
    assert server.cancelled >= 1
    assert any(k == "rearrange" for (k, *_r) in client.calls)
    assert not [c for c in client.calls if c[0] == "setup"]      # no 3-D setup after a failed walk
    assert ("release", "compute") in client.calls                # lock always released


def test_missing_probs_walks_but_does_not_score_the_shot():
    """Un-bootstrapped thresholds in FULL mode: the walk still runs (frames #2/#3 must be in focus
    and the frame stream must stay aligned), the shot is flagged + not persisted, no 3-D call."""
    np = pytest.importorskip("numpy")
    import Rearrange3DFocusWalkCommSeq as R
    f1, f2, f3 = _frames(np)
    cam = MultiFrameCam([f1, f2, f3])
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, cam, probs=[[], []])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1()
        R.walk_to_front(s1)
        R.rearrange_to_front(s1)
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()

    rounds = [c for c in client.calls if c[0] == "rearrange"]
    assert len(rounds) == 1 and rounds[0][1] == [0.0] * (2 * N)   # the walk ran
    assert s1.G.focus_walk_ok(False) is False                     # but the shot is not scored
    assert not [c for c in client.calls if c[0] == "setup"]
    assert cam.read_frames() == []                                # all frames consumed
    assert any(kind == "detect" for (kind, _sq, _m) in server.errors)


def test_build_and_serialize_the_seq():
    """The BUILD path (what serialize() sees) works for both modes over the real expConfig."""
    import os
    import sys

    import seq_manager
    from conftest import _TESTS_DIR
    from exp_seq import ExpSeq
    from seq_config import SeqConfig
    sys.path.insert(0, os.path.join(os.path.dirname(_TESTS_DIR), "YbScans"))
    import SLMRearrangement3DFocusWalkScan as S
    from Rearrange3DFocusWalkCommSeq import Rearrange3DFocusWalkCommSeq

    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    try:
        for walk_only in (False, True):
            seq_name, g = S.build(walk_only=walk_only)
            assert seq_name == "Rearrange3DFocusWalkCommSeq"
            blob = Rearrange3DFocusWalkCommSeq(ExpSeq(g.getseq(1))).serialize()
            assert len(blob) > 1000
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()
