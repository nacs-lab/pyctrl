"""NO-HARDWARE tests for the 3-D sphere assembly + focus-stack seq
(``RearrangeSphereFocusStackCommSeq``) and its scan.

Sibling of test_two_layer_lift.py / test_slm_rearrange3d_focus_walk.py (whose fakes this file
mirrors).  What is specific here and worth pinning:

  * the shot fires the server TWICE from ONE callback -- the 3-D assembly, then the phase-only
    axial walk -- with the walk's ``setup_rearrangement`` pushed BETWEEN them;
  * that walk setup swaps ``initial_phase`` to the SPHERE (``skip_grid_derive=True``): the grating
    ramps on the cached WGS initial phase, so without the swap it would translate the loading
    array and every atom would be lost;
  * setup #1 swaps it BACK to the loading phase, which is what makes shot N+1 load at all;
  * img2 is NOT reported to the server (a 50-site grid against a 1068-site init_grid);
  * assemble-only mode: one server call, no walk setup, and the pair still publishes (that is
    what bootstraps the sphere pattern's thresholds);
  * the frame-alignment invariant: a failing shot drains every produced frame and persists
    nothing;
  * the trap-servo ramp is in the BUILD (and only when the scan asks for it).

Never touches a real SLM server, camera, or the engine.
"""

import pytest

import rearrange_runtime

pytestmark = pytest.mark.no_hardware

N_LOAD = 6                                   # loading sites in these tests (production: 1068)
PROBS = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
LOAD_PAT = "33x33_feedback11"
SPHERE_PAT = "sphere50_r20um"
SPHERE_PHASE = "phase/sphere50_r20um.pt"
LOAD_PHASE = "phase/33x33_feedback11.pt"


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


def _ctx(server, client, cam, *, probs=None, bits="101101"):
    ctx = rearrange_runtime.ScanContext(
        session=_KeepaliveSession(), camera=cam, server=server, client=client,
        scan_id="20260807120000", n_rounds=1,
        frame_patterns=[LOAD_PAT, SPHERE_PAT], loading_defocus=-4.0)
    queue = list(probs if probs is not None else [PROBS])
    seen = []

    def _probs(pat, _img):
        seen.append(pat)
        return list(queue.pop(0)) if queue else []

    ctx.detect_probs_for = _probs
    ctx.detect_bits_for = lambda pat, _img: bits
    ctx.detected_patterns = seen
    return ctx


def _s1(*, lock_ok=True, assemble_only=False, kwargs2=None):
    from dyn_props import DynProps
    s1 = type("S1", (), {})()
    s1.G = DynProps({})
    s1.G.rearrange_lock_ok = lock_ok
    s1.G.sphere_ok = False
    s1.G.seq_id = 11
    rk = {"extras": {}}
    if assemble_only:
        rk["extras"]["assemble_only"] = True
    rk2 = kwargs2 if kwargs2 is not None else {
        "protocol": "pingponggrating", "nsteps": 50, "step_period_ms": 0.696,
        "initial_phase": SPHERE_PHASE, "skip_grid_derive": True,
        "extras": {"depth": True, "step_size": 0.5012, "return_trip": False,
                   "depth_piston_corr": 0.441, "precompute": True}}
    s1.C = DynProps({"rearrange_kwargs": rk, "rearrange_kwargs2": rk2,
                     "sphere": {"servo": 0.35, "servo_ramp": 10e-3, "n_sites": 50}})
    return s1


def _frames(np, n=2):
    return [np.full((8, 8), k, dtype=np.uint16) for k in range(1, n + 1)]


def _persisted(server):
    """The shots that actually PERSIST: a failing shot is re-published under the negative
    display-only sentinel scan id (``publish_failed_shot``), which the lab's show-without-persist
    path drops -- so "nothing was kept" means "nothing under the real scan id"."""
    return [shot for shot in server.finished
            if any(sid == "20260807120000" for (_f, sid, _q) in shot)]


# =========================================================================== #
# happy path
# =========================================================================== #
def test_full_flow_assembles_then_walks_and_publishes_one_pair():
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2]))
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1()
        R.assemble_and_walk(s1)             # img1 -> probs -> assemble -> setup#2 -> walk
        assert s1.G.sphere_ok(False) is True
        R.post_run(s1)                      # img2 -> stage -> finish
    finally:
        rearrange_runtime.clear_context()

    # ONE published shot of TWO frames, in order, under the real scan id.
    assert len(server.finished) == 1
    pair = server.finished[0]
    assert [int(f[0, 0]) for (f, _s, _q) in pair] == [1, 2]
    assert all(sid == "20260807120000" for (_f, sid, _q) in pair)
    assert server.cancelled == 0
    assert server.oks == 1

    # two server rounds, both carrying the SAME loading-occupancy vector (the walk ignores the
    # bits but the server still length-checks them against init_grid).
    rounds = [c for c in client.calls if c[0] == "rearrange"]
    assert len(rounds) == 2
    assert rounds[0][1] == PROBS
    assert rounds[1][1] == PROBS

    # the walk setup sits BETWEEN the two rounds and re-points initial_phase at the SPHERE.
    kinds = [c[0] for c in client.calls]
    assert kinds.index("setup") == 1
    setup = [c for c in client.calls if c[0] == "setup"][0][1]
    assert setup["protocol"] == "pingponggrating"
    assert setup["initial_phase"] == SPHERE_PHASE
    assert setup["skip_grid_derive"] is True
    assert setup["extras"]["depth"] is True
    assert setup["extras"]["return_trip"] is False
    assert setup["client_scan_id"] == "20260807120000"

    # img2 is never posted back (50-site grid vs 1068-site init_grid).
    assert kinds.count("update") == 0
    # only the LOADING pattern is detected in the shot path.
    assert ctx.detected_patterns == [LOAD_PAT]
    assert ("release", "compute") in client.calls
    assert ctx.session.keepalives == 1


# =========================================================================== #
# assemble-only (bootstrap mode)
# =========================================================================== #
def test_assemble_only_does_one_call_no_walk_setup_but_publishes():
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam([f1, f2])))
    try:
        s1 = _s1(assemble_only=True)
        R.assemble_and_walk(s1)
        assert s1.G.sphere_ok(False) is True
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()

    kinds = [c[0] for c in client.calls]
    assert kinds.count("rearrange") == 1        # the assembly only
    assert kinds.count("setup") == 0            # no walk setup
    assert kinds.count("update") == 0
    assert len(server.finished) == 1            # the pair still publishes -> thresholds can fit
    assert server.cancelled == 0


# =========================================================================== #
# failure paths
# =========================================================================== #
def test_failed_assembly_cancels_and_persists_nothing():
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient(raises=[True])
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam([f1, f2])))
    try:
        s1 = _s1()
        R.assemble_and_walk(s1)
        assert s1.G.sphere_ok(False) is False
        R.post_run(s1)                          # still drains img2
    finally:
        rearrange_runtime.clear_context()

    assert _persisted(server) == []             # nothing persisted under the real scan id
    assert server.cancelled >= 1
    kinds = [c[0] for c in client.calls]
    assert kinds.count("setup") == 0            # never reached the walk
    assert "cancel_last" in kinds               # the phantom server row was dropped


def test_failed_walk_cancels_after_a_successful_assembly():
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient(raises=[False, True])
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam([f1, f2])))
    try:
        s1 = _s1()
        R.assemble_and_walk(s1)
        assert s1.G.sphere_ok(False) is False
        R.post_run(s1)
    finally:
        rearrange_runtime.clear_context()

    assert _persisted(server) == []
    assert [c[0] for c in client.calls].count("rearrange") == 2


def test_no_loading_probs_means_no_server_call_at_all():
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam([f1, f2]), probs=[[]]))
    try:
        s1 = _s1()
        R.assemble_and_walk(s1)
    finally:
        rearrange_runtime.clear_context()

    assert client.calls == []                   # nothing to assemble -> nothing commanded
    assert server.cancelled == 1
    assert any(kind == "detect" for (kind, _sid, _msg) in server.errors)


def test_missing_kwargs2_is_a_hard_error_not_a_silent_loading_array_walk():
    """An empty ``rearrange_kwargs2`` must NOT fall through to a walk: the grating would still be
    pointed at the LOADING phase and would drag the just-assembled atoms nowhere useful."""
    np = pytest.importorskip("numpy")
    import RearrangeSphereFocusStackCommSeq as R
    f1, f2 = _frames(np)
    server, client = FakeImgServer(), FakeReClient()
    rearrange_runtime.set_context(_ctx(server, client, MultiFrameCam([f1, f2])))
    try:
        s1 = _s1(kwargs2={})
        R.assemble_and_walk(s1)
        assert s1.G.sphere_ok(False) is False
    finally:
        rearrange_runtime.clear_context()

    assert [c[0] for c in client.calls].count("rearrange") == 1   # the assembly, then stop
    assert server.cancelled == 1


# =========================================================================== #
# build path
# =========================================================================== #
def _build(servo, ramp_s=10e-3):
    """Serialize the seq with the scan-param override a ScanGroup point produces."""
    import seq_manager
    from exp_seq import ExpSeq
    from seq_config import SeqConfig
    import RearrangeSphereFocusStackCommSeq as R

    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    try:
        c_ovr = {"sphere": {"servo": servo, "servo_ramp": ramp_s}}
        return R.RearrangeSphereFocusStackCommSeq(ExpSeq(c_ovr)).serialize()
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()


def test_build_serializes_and_the_servo_ramp_is_opt_in():
    """The seq must build over the real config, and the trap-servo ramp must be present ONLY when
    the scan declares ``sphere.servo > 0`` -- a scan that wants the loading setpoint straight
    through gets the ramp-free bytes."""
    with_ramp = _build(0.35)
    without = _build(0.0)
    assert len(with_ramp) > 0 and len(without) > 0
    assert with_ramp != without, "sphere.servo > 0 must add the VSLMservo ramp to the build"
