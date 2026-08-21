"""NO-HARDWARE tests for the two-layer assembly seq (TwoLayerLiftCommSeq).

Sibling of test_slm_rearrange3d_focus_walk.py (whose fakes this file mirrors). What is specific
here and worth pinning:

  * FOUR frames, THREE rounds, THREE per-shot setups -- and each of the last two setups lands
    BETWEEN its frame's detection and its rearrange call (that ordering is the whole reason those
    rounds cannot use the shared ``rearrange_round``);
  * every round posts MEASURED per-site probabilities over the 1068-site loading grid, including
    the grating round (whose bits are ignored but still length-checked);
  * the final frame is published WITHOUT ``update_rearrange`` (img4 is on the far-layer grid,
    which is not the server's init_grid);
  * the frame-alignment invariant: a failing shot drains every produced frame and persists nothing
    under the real scan id.

Never touches a real SLM server, camera, or the engine.
"""

import pytest

import rearrange_runtime

pytestmark = pytest.mark.no_hardware

N_LOAD = 6                  # loading sites in these tests (production: 1068)
LOAD_PROBS = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
MID_PROBS = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
NEAR_PROBS = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
Z4 = -5.0
FULL_PHASE = "phase/fb11_2layer_full_z20.pt"

LOAD_PAT, FAR_PAT = "33x33_feedback11", "fb11_2layer_far"


# =========================================================================== #
# fakes (same shapes as test_slm_rearrange3d_focus_walk.py)
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


def _ctx(server, client, cam, *, probs=None, mid_image=True):
    pats = ([LOAD_PAT, LOAD_PAT, LOAD_PAT, FAR_PAT] if mid_image
            else [LOAD_PAT, LOAD_PAT, FAR_PAT])
    ctx = rearrange_runtime.ScanContext(
        session=_KeepaliveSession(), camera=cam, server=server, client=client,
        scan_id="20260807120000", n_rounds=1,
        frame_patterns=pats, loading_defocus=Z4)
    if probs is None:
        probs = ([LOAD_PROBS, MID_PROBS, NEAR_PROBS] if mid_image
                 else [LOAD_PROBS, NEAR_PROBS])
    queue = list(probs)
    seen = []

    def _probs(pat, _img):
        seen.append(pat)
        return list(queue.pop(0)) if queue else []

    ctx.detect_probs_for = _probs
    ctx.detect_bits_for = lambda pat, _img: "101000"
    ctx.detected_patterns = seen
    return ctx


def _s1(*, lock_ok=True, kwargs2=None, kwargs3=None, mid_image=True, mid_bits=None,
        do_walk=True, lift_image=True):
    from dyn_props import DynProps
    s1 = type("S1", (), {})()
    s1.G = DynProps({})
    s1.G.rearrange_lock_ok = lock_ok
    s1.G.two_layer_compact_ok = False
    s1.G.two_layer_lift_ok = False
    s1.G.two_layer_walk_ok = False
    s1.G.seq_id = 7
    rk2 = kwargs2 if kwargs2 is not None else {
        "protocol": "rearrange", "nsteps": 20, "step_period_ms": 0.696,
        "initial_phase": "phase/33x33_feedback11.pt", "final_phase": FULL_PHASE,
        "extras": {"target_grid_planes_z_rad": [0.0, 25.08], "pattern": None,
                   "target_bits": [1, 0, 1, 1, 0, 1], "wgs3d_warm": True,
                   "depth_piston_corr": -0.5, "z4": Z4}}
    rk3 = kwargs3 if kwargs3 is not None else {
        "protocol": "pingponggrating", "nsteps": 25, "step_period_ms": 0.696,
        "initial_phase": FULL_PHASE, "skip_grid_derive": True,
        "extras": {"depth": True, "true_defocus": True, "step_size": -0.8,
                   "return_trip": False, "depth_piston_corr": 0.4, "z4": Z4}}
    two_layer = {}
    if mid_bits is not None:
        two_layer["mid_bits"] = list(mid_bits)
    s1.C = DynProps({"rearrange_kwargs": {"extras": {"mid_image": 1 if mid_image else 0,
                                                     "lift_image": 1 if lift_image else 0,
                                                     "do_walk": 1 if do_walk else 0}},
                     "rearrange_kwargs2": rk2, "rearrange_kwargs3": rk3,
                     "two_layer": two_layer})
    return s1


def _frames(np, n=4):
    return [np.full((8, 8), k, dtype=np.uint16) for k in range(1, n + 1)]


# =========================================================================== #
# happy path
# =========================================================================== #
def test_full_flow_fires_three_rounds_and_publishes_one_quad():
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    frames = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam(frames))
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1()
        T.handoff_0(s1)                     # img1 -> round 0 (sticky setup from pre_run)
        assert s1.G.two_layer_compact_ok(False) is True
        T.handoff_1(s1)                     # img2 -> setup#2 -> round 1
        assert s1.G.two_layer_lift_ok(False) is True
        T.handoff_2(s1)                     # img3 -> setup#3 -> round 2
        assert s1.G.two_layer_walk_ok(False) is True
        T.post_run(s1)                      # img4 -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    # ONE published shot of FOUR frames, in order, under the real scan id.
    assert len(server.finished) == 1
    quad = server.finished[0]
    assert [int(f[0, 0]) for (f, _s, _q) in quad] == [1, 2, 3, 4]
    assert all(str(sid) == "20260807120000" for (_f, sid, _q) in quad)
    assert server.cancelled == 0
    assert server.oks == 1

    # every round posts the MEASURED probabilities of the frame in front of it.
    rounds = [c for c in client.calls if c[0] == "rearrange"]
    assert [r[1] for r in rounds] == [LOAD_PROBS, MID_PROBS, NEAR_PROBS]

    # two mid-shot setups, each between its frame's detection and its rearrange call.
    kinds = [c[0] for c in client.calls if c[0] in ("rearrange", "setup")]
    assert kinds == ["rearrange", "setup", "rearrange", "setup", "rearrange"]
    setups = [c[1] for c in client.calls if c[0] == "setup"]
    assert setups[0]["final_phase"] == FULL_PHASE
    assert setups[0]["extras"]["pattern"] is None          # must clear round 0's every_other
    assert setups[0]["extras"]["target_bits"] == [1, 0, 1, 1, 0, 1]
    assert setups[1]["protocol"] == "pingponggrating"
    assert setups[1]["initial_phase"] == FULL_PHASE        # the grating rides the two-layer WGS
    assert setups[1]["skip_grid_derive"] is True

    # img4 is NOT reported to the server (far-layer grid != init_grid).
    assert [c[0] for c in client.calls].count("update") == 0

    # frames 1-3 detect on the loading pattern; frame 4 is not detected at all (update=False).
    assert ctx.detected_patterns == [LOAD_PAT, LOAD_PAT, LOAD_PAT]


def test_no_lift_image_measures_the_axially_moved_atoms():
    """``lift_image = 0``: 3 frames (load / compacted / far).  The lift and the walk run back to
    back in ONE handoff.  The lift is still scored on MEASURED probabilities (the compacted frame
    is right in front of it); the trailing walk gets the declared vector, which it ignores.
    Survival then follows the atoms that went UP and came back to the camera plane."""
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    declared = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam(_frames(np, 3)),
               probs=[LOAD_PROBS, MID_PROBS])
    ctx.frame_patterns = [LOAD_PAT, LOAD_PAT, FAR_PAT]
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(lift_image=False, mid_bits=declared)
        T.handoff_0(s1)                     # img1 -> round 0
        T.handoff_1(s1)                     # img2 -> round 1 (declared) -> round 2 (measured)
        assert s1.G.two_layer_lift_ok(False) is True
        assert s1.G.two_layer_walk_ok(False) is True
        T.post_run(s1)                      # img3 (FAR layer) -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    assert len(server.finished) == 1
    assert [int(f[0, 0]) for (f, _s, _q) in server.finished[0]] == [1, 2, 3]
    assert server.cancelled == 0
    # The LIFT is still MEASURED: the compacted frame sits directly in front of it, so it is the
    # FIRST round of this handoff and gets that frame's probabilities.  Only the trailing round --
    # the walk, whose bits pingponggrating ignores anyway -- falls back to the declared vector.
    assert [c[1] for c in client.calls if c[0] == "rearrange"] == [LOAD_PROBS, MID_PROBS, declared]
    # both mid-shot setups still land, in order, inside the single handoff
    setups = [c[1] for c in client.calls if c[0] == "setup"]
    assert [x["protocol"] for x in setups] == ["rearrange", "pingponggrating"]
    # only TWO frames are detected -- there is no near-layer frame
    assert ctx.detected_patterns == [LOAD_PAT, LOAD_PAT]


def test_no_walk_ends_on_the_near_layer_frame():
    """``do_walk = 0``: two rounds, three frames (load / compacted / near), and the shot publishes
    WITHOUT the walk's ok-flag being set -- the far layer is assembled and simply never imaged."""
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam(_frames(np, 3)),
               probs=[LOAD_PROBS, MID_PROBS])
    ctx.frame_patterns = [LOAD_PAT, LOAD_PAT, LOAD_PAT]
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(do_walk=False)
        T.handoff_0(s1)                       # img1 -> round 0
        T.handoff_1(s1)                     # img2 -> setup#2 -> round 1
        assert s1.G.two_layer_lift_ok(False) is True
        T.post_run(s1)                      # img3 (NEAR layer) -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    # published even though two_layer_walk_ok was never set
    assert s1.G.two_layer_walk_ok(False) is False
    assert len(server.finished) == 1
    assert [int(f[0, 0]) for (f, _s, _q) in server.finished[0]] == [1, 2, 3]
    assert server.cancelled == 0
    # exactly TWO rounds and ONE setup -- the grating never runs
    assert [c[0] for c in client.calls if c[0] in ("rearrange", "setup")] == \
        ["rearrange", "setup", "rearrange"]
    assert [c[0] for c in client.calls].count("update") == 0


def test_no_mid_image_runs_both_rounds_on_one_frame():
    """THE DEMO SHAPE: 3 frames.  Round 0 fires on the loading frame's MEASURED probabilities and
    round 1 immediately after on the scan's DECLARED vector -- no camera frame between them."""
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    declared = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam(_frames(np, 3)), mid_image=False)
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(mid_image=False, mid_bits=declared)
        T.handoff_0(s1)                     # img1 -> round 0 -> setup#2 -> round 1
        assert s1.G.two_layer_compact_ok(False) is True
        assert s1.G.two_layer_lift_ok(False) is True
        T.handoff_1(s1)                     # img2 (NEAR layer) -> setup#3 -> round 2
        assert s1.G.two_layer_walk_ok(False) is True
        T.post_run(s1)                      # img3 (FAR layer) -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    assert len(server.finished) == 1
    assert [int(f[0, 0]) for (f, _s, _q) in server.finished[0]] == [1, 2, 3]
    assert server.cancelled == 0
    rounds = [c[1] for c in client.calls if c[0] == "rearrange"]
    assert rounds == [LOAD_PROBS, declared, NEAR_PROBS]
    # only TWO frames are ever detected: the loading one and the near-layer one.
    assert ctx.detected_patterns == [LOAD_PAT, LOAD_PAT]


def test_no_mid_image_without_declared_bits_cancels():
    """Round 1 has no frame in front of it in the 3-frame shape, so a missing declaration is a
    loud failing shot -- never a guess."""
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    server, client = FakeImgServer(), FakeReClient()
    rearrange_runtime.set_context(
        _ctx(server, client, MultiFrameCam(_frames(np, 3)), mid_image=False))
    try:
        s1 = _s1(mid_image=False, mid_bits=None)
        T.handoff_0(s1)
    finally:
        rearrange_runtime.clear_context()
    assert s1.G.two_layer_lift_ok(False) is False
    assert server.cancelled == 1
    # round 0 still fired (it is measured and valid); round 1 did not, and no setup was pushed.
    assert [c[0] for c in client.calls] == ["rearrange"]


def test_build_serializes_and_is_repeatable_in_both_shapes():
    """The BUILD path is RearrangeCommSeq2's with one optional extra basic sequence: with
    ``mid_image`` off it must be byte-identical to that 3-bseq seq, and with it on it must be
    longer.  Both must be repeatable."""
    import seq_manager
    from exp_seq import ExpSeq
    from seq_config import SeqConfig
    import RearrangeCommSeq2 as R
    import TwoLayerLiftCommSeq as T

    def _build(mid):
        s = ExpSeq()
        s.C.rearrange_kwargs.extras.mid_image = 1 if mid else 0
        return T.TwoLayerLiftCommSeq(s).serialize()

    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    try:
        four, four_again = _build(True), _build(True)
        three, three_again = _build(False), _build(False)
        rcs2 = R.RearrangeCommSeq2(ExpSeq()).serialize()
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()
    assert four == four_again and three == three_again, "build not repeatable"
    assert three == rcs2, "3-frame shape must match RearrangeCommSeq2 byte for byte"
    assert len(four) > len(three)


def test_missing_setup_declaration_cancels_instead_of_rearranging():
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    server, client = FakeImgServer(), FakeReClient()
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam(_frames(np))))
    try:
        s1 = _s1(kwargs2={})
        T.handoff_0(s1)
        T.handoff_1(s1)
    finally:
        rearrange_runtime.clear_context()
    assert s1.G.two_layer_lift_ok(False) is False
    assert server.cancelled == 1
    # round 0 still fired; the lift did not, and no setup was pushed.
    assert [c[0] for c in client.calls] == ["rearrange"]


def test_failed_round_drains_every_frame_and_persists_nothing():
    np = pytest.importorskip("numpy")
    import TwoLayerLiftCommSeq as T
    server = FakeImgServer()
    client = FakeReClient(raises=[False, True])       # the lift blows up
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam(_frames(np))))
    try:
        s1 = _s1()
        T.handoff_0(s1)
        T.handoff_1(s1)
        assert s1.G.two_layer_lift_ok(False) is False
        T.handoff_2(s1)                                # must still CONSUME img3
        T.post_run(s1)                                 # must still CONSUME img4
    finally:
        rearrange_runtime.clear_context()

    # Nothing persisted under the REAL scan id; the captured frames come back only under the
    # FAILING sentinel (negative scan id), which is display-only.
    assert all(int(sid) < 0 for pub in server.finished for (_f, sid, _q) in pub)
    assert server.cancelled >= 1
    # the grating setup is never pushed on a dead shot
    assert [c[0] for c in client.calls].count("setup") == 1
