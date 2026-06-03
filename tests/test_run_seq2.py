"""Phase-5 run_seq2: per-shot engine execution (run_real / run_bseq) call-order.

NO-HARDWARE: a fake ``pyseq`` (scripted ``post_run`` next_idx walk + scripted ``wait``
poll-loop) and a fake ``nidaq`` drive run_real/run_bseq with the engine never loaded. Every
engine call and callback appends to one ordered log, and per-bseq callbacks are TAGGED with
their bseq, so a misrouted branch or a misplaced callback fails loudly. Covers the verified
call order, the NI arm-before-start / wait-after, the per-bseq None-guard, the wait
poll-loop, the error path (reset_globals + re-raise), and the tail wall-clock pause.
"""

import pytest

import run_seq2

pytestmark = pytest.mark.no_hardware


class FakePyseq:
    def __init__(self, log, next_idxs, wait_returns=None, ni_data_by_call=None):
        self._log = log
        self._next = list(next_idxs)         # successive post_run() returns
        self._wait = list(wait_returns or [True])
        self._ni = list(ni_data_by_call or [])  # successive get_nidaq_data() returns
        self._wait_i = 0
        self._ni_i = 0

    def init_run(self):
        self._log.append("init_run")

    def pre_run(self):
        self._log.append("pre_run")

    def get_nidaq_data(self, name):
        self._log.append("get_nidaq_data:%s" % name)
        v = self._ni[self._ni_i] if self._ni_i < len(self._ni) else None
        self._ni_i += 1
        return v

    def start(self):
        self._log.append("start")

    def wait(self, timeout_ms):
        # cycle through the scripted poll results; default True (single poll)
        v = self._wait[min(self._wait_i, len(self._wait) - 1)]
        self._wait_i += 1
        self._log.append("wait")
        return v

    def cur_bseq_length(self):
        self._log.append("cur_bseq_length")
        return 0

    def post_run(self):
        self._log.append("post_run")
        return self._next.pop(0)


class FakeNidaq:
    def __init__(self, log):
        self._log = log
        self.run_calls = []

    def run(self, channels, clocks, triggers, data):
        self._log.append("nidaq.run")
        self.run_calls.append((channels, clocks, triggers, data))

    def wait(self):
        self._log.append("nidaq.wait")


class _C:
    pass


class FakeBSeq:
    def __init__(self, log, tag):
        self.before_bseq_cbs = [_cb(log, "before_bseq:%s" % tag)]
        self.after_bseq_cbs = [_cb(log, "after_bseq:%s" % tag)]
        self.after_branch_cbs = [_cb(log, "after_branch:%s" % tag)]


class FakeSeq:
    """A generated runnable seq. ``self`` is bseq 1; ``basic_seqs`` are bseqs 2.."""

    def __init__(self, log, next_idxs, n_basic=0, ni_channels=(), wait_returns=None,
                 ni_data_by_call=None, time_scale=1e12):
        self.C = _C()
        self.pyseq = FakePyseq(log, next_idxs, wait_returns, ni_data_by_call)
        # bseq-1 (root) callback lists live on the seq itself
        self.before_bseq_cbs = [_cb(log, "before_bseq:1")]
        self.after_bseq_cbs = [_cb(log, "after_bseq:1")]
        self.after_branch_cbs = [_cb(log, "after_branch:1")]
        self.basic_seqs = [FakeBSeq(log, str(i + 2)) for i in range(n_basic)]
        self.before_start_cbs = [_cb(log, "before_start")]
        self.after_end_cbs = [_cb(log, "after_end")]
        self.ni_channels = list(ni_channels)
        self.config = type("cfg", (), {"ni_clocks": {"Dev1": "PFI0"},
                                       "ni_start": {"Dev1": "PFI1"}})()
        self.time_scale = time_scale
        self._reset_log = log

    def reset_globals(self, persist):
        self._reset_log.append("reset_globals:%s" % persist)


def _cb(log, tag):
    def _fn(seq):
        log.append(tag)
    return _fn


# --------------------------------------------------------------------------- #
# single bseq, no NI
# --------------------------------------------------------------------------- #
def test_single_bseq_no_ni_call_order():
    log = []
    seq = FakeSeq(log, next_idxs=[0])
    run_seq2.run_real(seq, clock=_clock([0.0, 0.0]), sleep=_no_sleep)
    assert log == [
        "before_start", "init_run",
        "before_bseq:1", "pre_run", "start", "wait", "after_bseq:1",
        "cur_bseq_length", "post_run", "after_branch:1",
        "after_end", "reset_globals:False",
    ]


# --------------------------------------------------------------------------- #
# multi-bseq branch walk 1 -> 2 -> 1 -> 0 (per-bseq callback routing)
# --------------------------------------------------------------------------- #
def test_branch_walk_routes_per_bseq_callbacks():
    log = []
    seq = FakeSeq(log, next_idxs=[2, 1, 0], n_basic=1)
    run_seq2.run_real(seq, clock=_clock([0.0] * 8), sleep=_no_sleep)
    # before_start/after_end are root-level (once); bseq tags follow the 1,2,1 walk.
    assert log[0] == "before_start" and log[1] == "init_run"
    assert log[-2] == "after_end" and log[-1] == "reset_globals:False"
    befores = [x for x in log if x.startswith("before_bseq")]
    branches = [x for x in log if x.startswith("after_branch")]
    assert befores == ["before_bseq:1", "before_bseq:2", "before_bseq:1"]
    assert branches == ["after_branch:1", "after_branch:2", "after_branch:1"]


# --------------------------------------------------------------------------- #
# NI DAQ: arm before start, wait after
# --------------------------------------------------------------------------- #
def test_ni_armed_runs_before_start_and_waits_after():
    log = []
    nidaq = FakeNidaq(log)
    # 2 channels, 3 samples -> flat channel-major length 6
    seq = FakeSeq(log, next_idxs=[0], ni_channels=["a", "b"],
                  ni_data_by_call=[[1, 2, 3, 4, 5, 6]])
    run_seq2.run_real(seq, nidaq=nidaq, clock=_clock([0.0, 0.0]), sleep=_no_sleep)
    # get_nidaq_data -> nidaq.run BEFORE start; nidaq.wait AFTER the wait poll-loop.
    assert log.index("nidaq.run") < log.index("start")
    assert log.index("start") < log.index("wait") < log.index("nidaq.wait")
    # reshape: [nsamps=3, nchns=2], column-major -> col0 = ch0 = [1,2,3]
    _, _, _, data = nidaq.run_calls[0]
    assert _shape(data) == (3, 2)
    assert _col(data, 0) == [1.0, 2.0, 3.0] and _col(data, 1) == [4.0, 5.0, 6.0]


def test_ni_none_guard_skips_arm_and_wait():
    # ni_channels non-empty but this bseq has no analog data -> get_nidaq_data returns None
    # -> NEITHER nidaq.run NOR nidaq.wait fire (physically correct, no stale clock).
    log = []
    nidaq = FakeNidaq(log)
    seq = FakeSeq(log, next_idxs=[0], ni_channels=["a"], ni_data_by_call=[None])
    run_seq2.run_real(seq, nidaq=nidaq, clock=_clock([0.0, 0.0]), sleep=_no_sleep)
    assert "get_nidaq_data:NiDAQ" in log
    assert "nidaq.run" not in log and "nidaq.wait" not in log


# --------------------------------------------------------------------------- #
# wait poll-loop
# --------------------------------------------------------------------------- #
def test_wait_poll_loop_spins_until_true():
    log = []
    seq = FakeSeq(log, next_idxs=[0], wait_returns=[False, False, True])
    run_seq2.run_real(seq, clock=_clock([0.0, 0.0]), sleep=_no_sleep)
    assert log.count("wait") == 3            # polled until wait() returned True


# --------------------------------------------------------------------------- #
# error path
# --------------------------------------------------------------------------- #
def test_error_path_resets_globals_and_reraises():
    log = []
    seq = FakeSeq(log, next_idxs=[0])

    def boom(seq):
        raise RuntimeError("bseq blew up")

    seq.after_bseq_cbs = [boom]              # raise mid-bseq
    with pytest.raises(RuntimeError, match="blew up"):
        run_seq2.run_real(seq, clock=_clock([0.0, 0.0]), sleep=_no_sleep)
    assert log.count("reset_globals:False") == 1   # reset exactly once (catch path)
    assert "after_end" not in log                  # after_end never reached on error


# --------------------------------------------------------------------------- #
# tail wall-clock pause
# --------------------------------------------------------------------------- #
def test_tail_pause_waits_remaining_time():
    log = []
    slept = []
    seq = FakeSeq(log, next_idxs=[0], time_scale=1e12)
    # cur_bseq_length returns 0 -> bseq_len/time_scale == 0 -> end_after = start_t - 50ms.
    # Make end_after > end_t by scripting the clock: start_t reading = 100.0 (in the loop),
    # then end_t reading = 100.0; end_after = 100.0 + (5e10/1e12) - 0.05 = 100.0.
    seq.pyseq.cur_bseq_length = lambda: 5e10   # 0.05 s of length
    clock = _clock([0.0, 100.0, 100.0])        # prologue, in-loop start_t, end_t
    run_seq2.run_real(seq, clock=clock, sleep=lambda dt: slept.append(dt))
    # end_after = 100.0 + 0.05 - 0.05 = 100.0; end_t = 100.0 -> not < end_after -> no sleep.
    assert slept == []

    slept2 = []
    seq2 = FakeSeq([], next_idxs=[0], time_scale=1e12)
    seq2.pyseq.cur_bseq_length = lambda: 1e11  # 0.1 s length
    clock2 = _clock([0.0, 100.0, 100.0])
    run_seq2.run_real(seq2, clock=clock2, sleep=lambda dt: slept2.append(dt))
    # end_after = 100.0 + 0.1 - 0.05 = 100.05; end_t = 100.0 -> sleep ~0.05 s
    assert len(slept2) == 1 and abs(slept2[0] - 0.05) < 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _no_sleep(_dt):
    pass


def _clock(values):
    it = iter(values)
    last = [0.0]

    def _c():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return _c


def _shape(data):
    try:
        return tuple(data.shape)
    except AttributeError:
        return (len(data), len(data[0]))


def _col(data, j):
    try:
        return [float(x) for x in data[:, j]]
    except TypeError:
        return [float(row[j]) for row in data]
