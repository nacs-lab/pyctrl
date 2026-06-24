"""Phase-5 control_channel: the ZMQ SeqRequest gate that replaces the memmap CheckPauseAbort.

NO-HARDWARE: a fake control source models ExptServer's SeqRequest/State transitions, and a
scripted ``sleep`` advances the request timeline deterministically (standing in for the
ExptServer worker thread, which updates the request from incoming ZMQ pause/abort/resume
while the run loop spins). Asserts CheckPauseAbort parity + the single-clear-point /
abort-sticky / request-vs-reached must-fixes.

The live two-process coherency test (a real ExptServer + a separate control writer) is
item-7 territory per PYTHON_FRONTEND_PLAN.md Phase 5.
"""

import pytest

from control_channel import ControlChannel, SeqRequest

pytestmark = pytest.mark.no_hardware


class FakeControlSource:
    """Faithful stand-in for ExptServer's control surface (seq_req + State + ack)."""

    def __init__(self):
        self.req = SeqRequest.NoRequest
        self.state = "Init"               # Init / Running / Paused (ExptServer.State)
        self.ack_log = []                 # [True, False, ...] reached-paused transitions
        self.started = 0                  # number of start_scan() calls

    # -- run-loop-facing -- #
    def check_request(self):
        return int(self.req)

    def start_scan(self):
        self.state = "Running"
        self.req = SeqRequest.NoRequest   # clears Pause/NoRequest (single clear-point)
        self.started += 1
        return 1000 + self.started        # a scan_id

    def ack_paused(self, on):
        self.ack_log.append(bool(on))

    # -- external (GUI / ZMQ) request setters -- #
    def pause_seq(self):
        if self.state == "Running":
            self.state = "Paused"
            self.req = SeqRequest.Pause

    def abort_seq(self):
        if self.state in ("Running", "Paused"):
            self.state = "Init"
            self.req = SeqRequest.Abort

    def resume(self):                     # ExptServer.start_seq_serv
        if self.state == "Paused":
            self.req = SeqRequest.NoRequest
            self.state = "Running"


class MinimalSource:
    """A source WITHOUT ack_paused -- coarse-status v1 ExptServer; must still park."""

    def __init__(self, script):
        self._script = list(script)       # successive check_request() return values
        self._i = 0

    def check_request(self):
        v = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return int(v)

    def start_scan(self):
        return 1


class ScriptedSleep:
    """A sleep that fires ``action`` once it has been called ``after_n`` times, and raises
    if the spin runs away (so a non-terminating gate fails loudly instead of hanging)."""

    def __init__(self, after_n, action, cap=1000):
        self.calls = 0
        self.after_n = after_n
        self.action = action
        self.cap = cap

    def __call__(self, _dt):
        self.calls += 1
        if self.calls > self.cap:
            raise RuntimeError("control gate spin did not terminate")
        if self.calls == self.after_n:
            self.action()


# --------------------------------------------------------------------------- #
# the gate: proceed / abort / park
# --------------------------------------------------------------------------- #
class TestCheckPauseAbort:
    def test_no_request_proceeds(self):
        src = FakeControlSource()
        cc = ControlChannel(src, sleep=_no_sleep)
        assert cc.check_pause_abort() is False
        assert src.ack_log == []           # never parked

    def test_abort_returns_true_without_parking(self):
        src = FakeControlSource()
        src.start_scan()
        src.abort_seq()
        cc = ControlChannel(src, sleep=_forbid_sleep)
        assert cc.check_pause_abort() is True
        assert src.ack_log == []           # abort precedes pause -- no park, no ack

    def test_pause_then_resume_proceeds(self):
        src = FakeControlSource()
        src.start_scan()
        src.pause_seq()
        sleeper = ScriptedSleep(2, src.resume)
        cc = ControlChannel(src, poll_interval=0.01, sleep=sleeper)
        assert cc.check_pause_abort() is False
        assert sleeper.calls == 2          # actually parked / spun before resuming
        assert src.ack_log == [True, False]  # reached-paused ack raised then cleared

    def test_pause_then_abort_returns_true(self):
        src = FakeControlSource()
        src.start_scan()
        src.pause_seq()
        sleeper = ScriptedSleep(2, src.abort_seq)
        cc = ControlChannel(src, poll_interval=0.01, sleep=sleeper)
        assert cc.check_pause_abort() is True   # abort during pause wins
        assert src.ack_log == [True, False]     # ack cleared even on the abort path

    def test_parks_without_ack_hook(self):
        # MinimalSource has no ack_paused: Pause, Pause, then NoRequest -> proceeds, no crash.
        src = MinimalSource([SeqRequest.Pause, SeqRequest.Pause, SeqRequest.NoRequest])
        sleeper = ScriptedSleep(99, lambda: None)   # never needs to fire; script resumes
        cc = ControlChannel(src, poll_interval=0.0, sleep=sleeper)
        assert cc.check_pause_abort() is False
        assert sleeper.calls >= 1                   # it did spin at least once


# --------------------------------------------------------------------------- #
# scan boundary: single clear-point (clear-at-job-start)
# --------------------------------------------------------------------------- #
class TestBeginScan:
    def test_begin_scan_starts_when_idle(self):
        src = FakeControlSource()
        cc = ControlChannel(src, sleep=_no_sleep)
        sid = cc.begin_scan()
        assert sid == 1001 and src.started == 1
        assert src.req == SeqRequest.NoRequest and src.state == "Running"

    def test_begin_scan_clears_pending_pause(self):
        # A leftover Pause at scan start is cleared by start_scan (single clear-point).
        src = FakeControlSource()
        src.start_scan()
        src.pause_seq()
        cc = ControlChannel(src, sleep=_no_sleep)
        assert cc.begin_scan() is not None
        assert src.req == SeqRequest.NoRequest

    def test_begin_scan_clears_stale_abort(self):
        # clear-at-job-start: a stale abort left from a PRIOR scan is cleared by start_scan so
        # a newly-submitted scan runs (bug-runjob-stale-abortrunseq -- abort stops the current
        # scan, not future ones). The mid-run check_pause_abort gate still honors a fresh abort.
        src = FakeControlSource()
        src.start_scan()
        src.abort_seq()
        started_before = src.started
        cc = ControlChannel(src, sleep=_no_sleep)
        assert cc.begin_scan() is not None          # starts despite the stale abort
        assert src.started == started_before + 1    # start_scan WAS called
        assert src.req == SeqRequest.NoRequest       # stale abort cleared

    def test_aborting_query(self):
        src = FakeControlSource()
        cc = ControlChannel(src, sleep=_no_sleep)
        assert cc.aborting() is False
        src.start_scan()
        src.abort_seq()
        assert cc.aborting() is True


def _no_sleep(_dt):
    pass


def _forbid_sleep(_dt):
    raise AssertionError("gate slept on a non-pause request")


# --------------------------------------------------------------------------- #
# should_yield -- the background-lane yield predicate (NOT a SeqRequest)
# --------------------------------------------------------------------------- #
class _FgSource:
    """A control source exposing has_foreground_work (+ a SeqRequest we assert stays clean)."""

    def __init__(self, fg):
        self._fg = fg
        self.req = SeqRequest.NoRequest
        self.checked = 0

    def check_request(self):
        self.checked += 1
        return int(self.req)

    def has_foreground_work(self):
        return self._fg

    def start_scan(self):
        return 1


class TestShouldYield:
    def test_foreground_control_never_yields(self):
        # is_background=False -> should_yield is always False, even with foreground work queued.
        cc = ControlChannel(_FgSource(fg=True), is_background=False)
        assert cc.should_yield() is False

    def test_background_yields_when_foreground_queued(self):
        cc = ControlChannel(_FgSource(fg=True), is_background=True)
        assert cc.should_yield() is True

    def test_background_does_not_yield_without_foreground(self):
        cc = ControlChannel(_FgSource(fg=False), is_background=True)
        assert cc.should_yield() is False

    def test_source_without_predicate_never_yields(self):
        # getattr-guard: an older source lacking has_foreground_work -> no yield (no crash).
        class _NoPred:
            def check_request(self):
                return int(SeqRequest.NoRequest)

        cc = ControlChannel(_NoPred(), is_background=True)
        assert cc.should_yield() is False

    def test_should_yield_never_touches_seq_request(self):
        # Cardinal rule: yielding sets NO control flag, so the incoming foreground scan's
        # begin_scan sees a clean slate (NoRequest). should_yield must not even read it.
        src = _FgSource(fg=True)
        cc = ControlChannel(src, is_background=True)
        cc.should_yield()
        assert src.checked == 0
        assert src.req == SeqRequest.NoRequest

    def test_predicate_failure_is_swallowed(self):
        class _Boom:
            def has_foreground_work(self):
                raise RuntimeError("boom")

        cc = ControlChannel(_Boom(), is_background=True)
        assert cc.should_yield() is False
