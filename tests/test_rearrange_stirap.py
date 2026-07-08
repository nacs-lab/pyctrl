"""NO-HARDWARE tests for the RearrangeSTIRAPSeq hybrid callbacks.

The hybrid "rearrange, then science" per-shot flow over the shared rearrange_callbacks
machinery, with fakes (mirrors test_slm_rearrangement2.py):
  * verify ON  (3 frames): img1 -> rearrange; img2 -> update_rearrange (verify_frame);
    img3 staged + finished with update=False (post-pushout frame never feeds the server).
  * verify OFF (2 frames): img1 -> rearrange; img2 staged + finished, NO update_rearrange
    at all.
  * failing round: the verify frame is still DRAINED, the shot cancels, and the captured
    frames re-publish under the failing display sentinel (no persist).
"""

import pytest

import rearrange_runtime

pytestmark = pytest.mark.no_hardware


# =========================================================================== #
# fakes (mirroring test_slm_rearrangement2.py)
# =========================================================================== #
class FakeImgServer:
    def __init__(self):
        self.temp = []
        self.finished = []
        self.cancelled = 0
        self.oks = 0

    def stage_frame(self, frame, scan_id=-1, seq_id=-1, *, async_=True):
        self.temp.append((frame, scan_id, seq_id))

    def finish_shot(self, *, async_=True):
        self.finished.append(list(self.temp))
        self.temp = []

    def cancel_shot(self, *, async_=True):
        self.cancelled += 1
        self.temp = []

    def record_shot_ok(self):
        self.oks += 1


class FakeReClient:
    def __init__(self, raise_rearrange=False):
        self.calls = []
        self._raise = raise_rearrange

    def rearrange(self, probs, **kw):
        self.calls.append(("rearrange", list(probs), kw))
        if self._raise:
            raise RuntimeError("rearrange boom")
        return {"ok": True}

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


def _ctx(server, client, cam, frame_patterns):
    ctx = rearrange_runtime.ScanContext(
        session=_KeepaliveSession(), camera=cam, server=server, client=client,
        scan_id="20260708120000", n_rounds=1, frame_patterns=frame_patterns)
    ctx.detect_probs_for = lambda pat, _img: [1.0, 0.0, 1.0]
    ctx.detect_bits_for = lambda pat, _img: "101"
    return ctx


def _s1(verify, lock_ok=True):
    from dyn_props import DynProps
    s1 = type("S1", (), {})()
    s1.G = DynProps({})
    s1.G.rearrange_lock_ok = lock_ok
    s1.G.rearrange_img1_ok = False
    s1.G.rearrange_img2_ok = False
    s1.G.seq_id = 7
    # s1.C.rearrange_kwargs.extras.verifyImage(default) -> the layout flag (DynProps-call style).
    extras = type("E", (), {})()
    extras.verifyImage = lambda default=True, _v=verify: _v
    rk = type("RK", (), {})()
    rk.extras = extras
    s1.C = type("C", (), {})()
    s1.C.rearrange_kwargs = rk
    return s1


# =========================================================================== #
# verify ON: 3 frames -> one persisted triple; update on img2 ONLY
# =========================================================================== #
def test_verify_on_happy_path_triple_update_on_img2_only():
    np = pytest.importorskip("numpy")
    import RearrangeSTIRAPSeq as R
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]), ["load", "target", "target"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(verify=True)
        R.hand_over_slm(s1)                 # img1 -> rearrange -> stage
        assert s1.G.rearrange_img1_ok(False) is True
        R.verify_and_report(s1)             # img2 -> update_rearrange -> stage
        assert s1.G.rearrange_img2_ok(False) is True
        R.post_run(s1)                      # img3 -> stage -> finish (update=False)
    finally:
        rearrange_runtime.clear_context()
    assert len(server.finished) == 1
    triple = server.finished[0]
    assert [int(f[0, 0]) for (f, _sid, _sq) in triple] == [1, 2, 3]
    assert all(sid == "20260708120000" for (_f, sid, _sq) in triple)
    assert server.cancelled == 0
    assert sum(1 for c in client.calls if c[0] == "rearrange") == 1
    # exactly ONE update_rearrange -- the img2 verify frame; img3 (post-pushout) never updates.
    assert sum(1 for c in client.calls if c[0] == "update") == 1
    assert ("release", "compute") in client.calls
    assert ctx.session.keepalives == 1
    assert server.oks == 1


# =========================================================================== #
# verify OFF: 2 frames -> one persisted pair; NO update_rearrange at all
# =========================================================================== #
def test_verify_off_happy_path_pair_no_update():
    np = pytest.importorskip("numpy")
    import RearrangeSTIRAPSeq as R
    f1, f2 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2))
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1, f2]), ["load", "target"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(verify=False)
        R.hand_over_slm(s1)
        assert s1.G.rearrange_img1_ok(False) is True
        R.post_run(s1)                      # img2 -> stage -> finish (final frame)
    finally:
        rearrange_runtime.clear_context()
    assert len(server.finished) == 1
    pair = server.finished[0]
    assert [int(f[0, 0]) for (f, _sid, _sq) in pair] == [1, 2]
    assert server.cancelled == 0
    # post-pushout survival frame never feeds the server's rearrange stats.
    assert not any(c[0] == "update" for c in client.calls)
    assert ("release", "compute") in client.calls
    assert server.oks == 1


# =========================================================================== #
# verify ON, rearrange fails: drain everything, display-only re-publish, no persist
# =========================================================================== #
def test_verify_on_rearrange_failure_drains_and_publishes_display_only():
    np = pytest.importorskip("numpy")
    import RearrangeSTIRAPSeq as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1, f2, f3 = (np.full((8, 8), k, dtype=np.uint16) for k in (1, 2, 3))
    server, client = FakeImgServer(), FakeReClient(raise_rearrange=True)
    ctx = _ctx(server, client, MultiFrameCam([f1, f2, f3]), ["load", "target", "target"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(verify=True)
        R.hand_over_slm(s1)                 # rearrange raises -> stage img1 then drop
        assert s1.G.rearrange_img1_ok(False) is False
        assert any(c[0] == "cancel_last" for c in client.calls)
        R.verify_and_report(s1)             # prior failed -> DRAIN img2 + cancel, NO update
        assert s1.G.rearrange_img2_ok(False) is False
        R.post_run(s1)                      # DRAIN img3, display-only re-publish
    finally:
        rearrange_runtime.clear_context()
    assert server.cancelled >= 2
    assert not any(c[0] == "update" for c in client.calls)
    # all three captured frames flash by on the live view, under the failing sentinel.
    assert len(server.finished) == 1
    shown = server.finished[0]
    assert [int(f[0, 0]) for (f, _sid, _sq) in shown] == [1, 2, 3]
    assert all(sid == FAILING_DISPLAY_SCAN_ID for (_f, sid, _sq) in shown)
    assert server.oks == 0
    assert ("release", "compute") in client.calls   # finally path still releases


# =========================================================================== #
# verify ON, img2 grab lost: shot cancels; no half set persists
# =========================================================================== #
def test_verify_on_img2_lost_cancels_no_persist():
    np = pytest.importorskip("numpy")
    import RearrangeSTIRAPSeq as R
    from rearrange_runtime import FAILING_DISPLAY_SCAN_ID
    f1 = np.full((8, 8), 1, dtype=np.uint16)
    server, client = FakeImgServer(), FakeReClient()
    ctx = _ctx(server, client, MultiFrameCam([f1]), ["load", "target", "target"])
    rearrange_runtime.set_context(ctx)
    try:
        s1 = _s1(verify=True)
        R.hand_over_slm(s1)                 # img1 ok, rearrange ok
        R.verify_and_report(s1)             # img2 grab fails -> cancel
        assert s1.G.rearrange_img2_ok(False) is False
        R.post_run(s1)                      # img3 also absent -> display-only img1
    finally:
        rearrange_runtime.clear_context()
    assert server.cancelled >= 1
    # only img1 was captured; it re-publishes display-only (never persisted).
    assert len(server.finished) == 1
    shown = server.finished[0]
    assert len(shown) == 1
    assert shown[0][1] == FAILING_DISPLAY_SCAN_ID
    assert server.oks == 0
