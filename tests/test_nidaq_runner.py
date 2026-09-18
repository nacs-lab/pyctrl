"""nidaq_runner: the transpose (#1 silent-bug risk) + structural session caching.

NO-HARDWARE: the device hooks (_build_task / _write_and_start / _wait_task / _close_task /
_generated_count) are stubbed, so the pure transpose, the cache-invalidation DECISION and the
DAQmx -200018 clock-burst triage are exercised without nidaqmx or the card. The live
card-listen / clock-out is NEEDS-HARDWARE (maintenance window).
"""

import pytest

import devices.nidaq.nidaq_runner as nidaq_runner
from devices.nidaq import NiDAQRunner

pytestmark = pytest.mark.no_hardware

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def _clean_cache():
    NiDAQRunner.reset_cache()
    yield
    NiDAQRunner.reset_cache()


# --------------------------------------------------------------------------- #
# the transpose
# --------------------------------------------------------------------------- #
def test_to_channel_major_transposes():
    # sample-major [nsamps=3, nchns=2]: col j is channel j -> channel-major [2, 3].
    sample_major = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    cm = nidaq_runner._to_channel_major(sample_major)
    assert cm.shape == (2, 3)
    assert list(cm[0]) == [1.0, 2.0, 3.0]      # channel 0 = all its samples
    assert list(cm[1]) == [4.0, 5.0, 6.0]


def test_to_channel_major_single_channel():
    # sample-major [nsamps=3, nchns=1] -> channel-major [1, 3] (the squeeze happens at write).
    cm = nidaq_runner._to_channel_major(np.array([[1.0], [2.0], [3.0]]))
    assert cm.shape == (1, 3)
    assert list(cm[0]) == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# _write_and_start: single-channel squeeze (DaqError -200524 regression, PicoMotor308)
# --------------------------------------------------------------------------- #
class _RecordTask:
    """Captures what task.write() received; fakes the nidaqmx Task surface _write_and_start uses.

    Also counts the DAQmx state transitions the per-arm fast path is about: how often the
    sample-clock timing was (re)configured and how often the task was explicitly COMMITTED.
    """
    def __init__(self):
        self.written = None
        self.cfg_calls = []     # the samps_per_chan of each cfg_samp_clk_timing call
        self.commits = 0
        self.stops = 0
        self.starts = 0

        class _Timing:
            def __init__(self, owner):
                self._owner = owner

            def cfg_samp_clk_timing(self, *a, **k):
                self._owner.cfg_calls.append(k.get("samps_per_chan"))
        self.timing = _Timing(self)

    def control(self, mode):
        self.commits += 1

    def stop(self):
        self.stops += 1

    def write(self, data, auto_start=False):
        self.written = data

    def start(self):
        self.starts += 1


@pytest.fixture
def fake_nidaqmx(monkeypatch):
    """Stub nidaqmx.constants so _write_and_start's import works without the package."""
    import sys
    import types
    mod = types.ModuleType("nidaqmx")
    consts = types.ModuleType("nidaqmx.constants")
    consts.AcquisitionType = types.SimpleNamespace(FINITE=object())
    consts.Edge = types.SimpleNamespace(RISING=object())
    consts.TaskMode = types.SimpleNamespace(TASK_COMMIT=object())
    mod.constants = consts
    monkeypatch.setitem(sys.modules, "nidaqmx", mod)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", consts)


def _new_task():
    task = _RecordTask()
    nidaq_runner._TASK_META[task] = (nidaq_runner._RATE, "/Dev1/PFI0", True)   # (rate, clk, preload)
    return task


def _write(samples):
    task = _new_task()
    nidaq_runner._write_and_start(task, samples)
    return task.written


def test_write_single_channel_is_1d(fake_nidaqmx):
    # (1, nsamps) 2-D must be squeezed to 1-D, else nidaqmx raises -200524.
    written = _write(np.array([[0.0, 0.0, 0.0]]))   # 1 channel, 3 samples
    assert np.asarray(written).ndim == 1
    assert list(written) == [0.0, 0.0, 0.0]


def test_write_multi_channel_stays_2d(fake_nidaqmx):
    written = _write(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))   # 2 channels
    assert np.asarray(written).shape == (2, 3)


# --------------------------------------------------------------------------- #
# _write_and_start: the timing-reconfig cache + explicit COMMIT (the ni_arm fast path).
# Reconfiguring the sample clock un-commits the task, forcing start() to redo
# verify/reserve/commit on every arm; guarding it on the sample count keeps the task
# COMMITTED across the per-shot stop -> write -> start.
# --------------------------------------------------------------------------- #
def test_first_arm_configures_timing_and_commits(fake_nidaqmx):
    task = _new_task()
    nidaq_runner._write_and_start(task, np.zeros((2, 5)))
    assert task.cfg_calls == [5]
    assert task.commits == 1
    assert (task.stops, task.starts) == (1, 1)


def test_repeat_arm_same_nsamps_skips_reconfig_and_recommit(fake_nidaqmx):
    task = _new_task()
    for _ in range(3):
        nidaq_runner._write_and_start(task, np.zeros((2, 5)))
    assert task.cfg_calls == [5]            # configured ONCE, not three times
    assert task.commits == 1                # and committed once -- start() stays the fast path
    assert (task.stops, task.starts) == (3, 3)   # every arm still stops + starts


def test_changed_nsamps_reconfigures_and_recommits(fake_nidaqmx):
    task = _new_task()
    nidaq_runner._write_and_start(task, np.zeros((2, 5)))
    nidaq_runner._write_and_start(task, np.zeros((2, 7)))   # different bseq length
    nidaq_runner._write_and_start(task, np.zeros((2, 7)))   # ... then repeats -> cached again
    assert task.cfg_calls == [5, 7]
    assert task.commits == 2


def test_stop_task_forgets_armed_count(fake_nidaqmx):
    """Every error path stops the task; DAQmx may have reset its state, so the next arm must
    reconfigure rather than trust the cache."""
    task = _new_task()
    nidaq_runner._write_and_start(task, np.zeros((2, 5)))
    nidaq_runner._stop_task(task)                            # the run()/wait() error path
    nidaq_runner._write_and_start(task, np.zeros((2, 5)))    # same nsamps, but must reconfigure
    assert task.cfg_calls == [5, 5]
    assert task.commits == 2


def test_commit_failure_is_survivable(fake_nidaqmx, monkeypatch):
    """A card/nidaqmx that refuses TASK_COMMIT falls back to the old implicit-commit path:
    slower, never wrong -- and the arm still completes."""
    task = _new_task()

    def boom(mode):
        raise RuntimeError("TaskControl unsupported")
    monkeypatch.setattr(task, "control", boom)
    nidaq_runner._write_and_start(task, np.zeros((2, 5)))
    assert task.cfg_calls == [5]
    assert (task.stops, task.starts) == (1, 1)
    assert np.asarray(task.written).shape == (2, 5)


# --------------------------------------------------------------------------- #
# structural equality helpers
# --------------------------------------------------------------------------- #
class TestStructuralEquality:
    def test_channels_equal_by_dev_chn_in_order(self):
        a = [{"dev": "Dev1", "chn": 0}, {"dev": "Dev1", "chn": 1}]
        b = [{"dev": "Dev1", "chn": 0}, {"dev": "Dev1", "chn": 1}]
        assert nidaq_runner._channels_equal(a, b)

    def test_channels_differ_on_order(self):
        a = [{"dev": "Dev1", "chn": 0}, {"dev": "Dev1", "chn": 1}]
        b = [{"dev": "Dev1", "chn": 1}, {"dev": "Dev1", "chn": 0}]
        assert not nidaq_runner._channels_equal(a, b)   # add-order is load-bearing

    def test_channels_differ_on_chn_and_length(self):
        a = [{"dev": "Dev1", "chn": 0}]
        assert not nidaq_runner._channels_equal(a, [{"dev": "Dev1", "chn": 2}])
        assert not nidaq_runner._channels_equal(a, a + [{"dev": "Dev1", "chn": 1}])

    def test_map_equal(self):
        assert nidaq_runner._map_equal({"Dev1": "PFI0"}, {"Dev1": "PFI0"})
        assert not nidaq_runner._map_equal({"Dev1": "PFI0"}, {"Dev1": "PFI1"})


# --------------------------------------------------------------------------- #
# session cache decision (device hooks stubbed)
# --------------------------------------------------------------------------- #
class FakeTask:
    def __init__(self):
        self.writes = []
        self.started = 0
        self.stopped = 0        # stops performed INSIDE the stubbed _write_and_start (re-arm)
        self.stops = 0          # real _stop_task(task) calls (error / swallow paths)
        self.closed = False

    def stop(self):
        # _stop_task is NOT stubbed: it calls task.stop() for real and swallows failures, so
        # without this method every "did it stop the task?" assertion would silently pass.
        self.stops += 1


@pytest.fixture
def stub_device(monkeypatch):
    builds = []

    def fake_build(channels, clocks, triggers, rate):
        t = FakeTask()
        builds.append((t, list(channels), dict(clocks), dict(triggers), rate))
        return t

    def fake_write_and_start(task, samples):
        task.stopped += 1
        task.writes.append(samples)
        task.started += 1

    monkeypatch.setattr(nidaq_runner, "_build_task", fake_build)
    monkeypatch.setattr(nidaq_runner, "_write_and_start", fake_write_and_start)
    monkeypatch.setattr(nidaq_runner, "_wait_task", lambda task: None)
    monkeypatch.setattr(nidaq_runner, "_close_task",
                        lambda task: setattr(task, "closed", True) if task else None)
    return builds


_CH = [{"dev": "Dev1", "chn": 0}, {"dev": "Dev1", "chn": 1}]
_CLK = {"Dev1": "PFI0"}
_TRG = {"Dev1": "PFI1"}
_DATA = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])   # [3 samps, 2 chns]


def test_first_run_builds_once_and_writes_channel_major(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert len(stub_device) == 1
    task = stub_device[0][0]
    assert task.started == 1 and task.stopped == 1     # STOP before write (FINITE re-arm)
    written = task.writes[0]
    assert written.shape == (2, 3)                     # channel-major
    assert list(written[0]) == [1.0, 2.0, 3.0]


def test_reuse_after_wait_does_not_rebuild(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    NiDAQRunner.wait()                                 # clears cache_in_use
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert len(stub_device) == 1                       # same session reused

def test_rebuild_when_run_without_wait(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)            # no wait() -> cache_in_use still set
    assert len(stub_device) == 2                       # rebuilt (can't reuse a busy session)


def test_rebuild_on_changed_channels(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    NiDAQRunner.wait()
    NiDAQRunner.run(_CH + [{"dev": "Dev1", "chn": 2}], _CLK, _TRG,
                    np.zeros((3, 3)))
    assert len(stub_device) == 2


def test_rebuild_on_changed_clocks_or_triggers(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    NiDAQRunner.wait()
    NiDAQRunner.run(_CH, {"Dev1": "PFI3"}, _TRG, _DATA)   # different clock source
    assert len(stub_device) == 2


def test_wait_clears_cache_in_use(stub_device):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert NiDAQRunner._cache_in_use is True
    NiDAQRunner.wait()
    assert NiDAQRunner._cache_in_use is False


def test_error_on_write_stops_task_and_forces_rebuild(stub_device, monkeypatch):
    def boom(task, samples):
        task.stopped += 1
        raise RuntimeError("write failed")

    monkeypatch.setattr(nidaq_runner, "_write_and_start", boom)
    with pytest.raises(RuntimeError, match="write failed"):
        NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    # cache_in_use stayed True (no wait) -> the next run rebuilds the session.
    monkeypatch.setattr(nidaq_runner, "_write_and_start",
                        lambda task, samples: task.writes.append(samples))
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert len(stub_device) == 2


# --------------------------------------------------------------------------- #
# DAQmx -200018 triage: the post-sequence PFI0 clock burst (molecube2
# controller.cpp:716-736) completes the finite task and then keeps strobing it, so a -200018
# can fire with ALL real samples already converted -- cosmetic, must not kill the scan.
# --------------------------------------------------------------------------- #
class _FakeDaqError(Exception):
    """Stand-in for nidaqmx.DaqError: carries .error_code like the real one."""
    def __init__(self, code=-200018,
                 msg="DAC conversion attempted before data to be converted was available"):
        super().__init__(msg)
        self.error_code = code


class _MsgOnlyError(Exception):
    """A DAQmx-ish error with NO error_code attribute -- only the message carries the code."""


def _raise_underflow(*a, **k):
    raise _FakeDaqError()


def _stub_generated_count(monkeypatch, values):
    """Stub ``_generated_count`` with a STATEFUL sequence (the last value repeats).

    The counter is now read TWICE per shot: once in run() right after the start (the delta
    base) and once in wait(). The base is deliberately NONZERO here (1000) -- a cumulative
    ``total_samp_per_chan_generated`` mistaken for a fresh per-start count would make the gate
    pass on any shot and swallow GENUINE underflows.
    """
    seq = list(values)
    calls = []

    def fake(task):
        calls.append(task)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(nidaq_runner, "_generated_count", fake)
    return calls


def test_wait_swallows_burst_race_underflow(stub_device, monkeypatch):
    calls = _stub_generated_count(monkeypatch, [1000, 1003])   # base, base + nsamps
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)            # 3 samples per channel
    assert NiDAQRunner._last_nsamps == 3
    assert NiDAQRunner._gen_base == 1000               # snapshot taken at arm time
    task = stub_device[0][0]
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    NiDAQRunner.wait()                                 # cosmetic -> must NOT raise
    assert len(calls) == 2                             # arm-time base + wait-time count
    assert task.stops == 1                             # errored task was stopped (_stop_task)
    assert NiDAQRunner.underflows_swallowed == 1
    # Contract: the stopped task is REUSED. Keeping cache_in_use True forced a ~0.7 s channel
    # rebuild at the next bseq arm -- mid-shot in a multi-bseq rearrangement shot.
    assert NiDAQRunner._cache_in_use is False
    monkeypatch.setattr(nidaq_runner, "_wait_task", lambda task: None)
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert len(stub_device) == 1                       # no mid-shot rebuild; same task re-armed


def test_wait_reraises_genuine_underflow_short_count(stub_device, monkeypatch):
    _stub_generated_count(monkeypatch, [1000, 1002])    # delta 2 < nsamps 3 -> a sample lost
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    task = stub_device[0][0]
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    with pytest.raises(_FakeDaqError):
        NiDAQRunner.wait()
    # Errored Task must be stopped anyway: an open+errored Task keeps the AO channels DAQmx
    # RESERVED, so out-of-band dashboard DC sets would hit -50103 until the idle release.
    assert task.stops == 1
    assert NiDAQRunner.underflows_swallowed == 0
    assert NiDAQRunner._cache_in_use is True            # rebuild at the next arm


def test_wait_reraises_when_count_unavailable(stub_device, monkeypatch):
    _stub_generated_count(monkeypatch, [1000, None])    # wait-time count unqueryable
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    with pytest.raises(_FakeDaqError):
        NiDAQRunner.wait()


def test_wait_reraises_when_base_unavailable(stub_device, monkeypatch):
    # Arm-time base unreadable -> the delta is unknowable, so even a huge count cannot verify
    # the shot: strict None handling must fall through to the re-raise.
    _stub_generated_count(monkeypatch, [None, 10 ** 9])
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert NiDAQRunner._gen_base is None
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    with pytest.raises(_FakeDaqError):
        NiDAQRunner.wait()


def test_wait_reraises_when_fifo_preload_inactive(stub_device, monkeypatch):
    # preload_ok=False (card/nidaqmx rejected ao_use_only_on_brd_mem -> streaming fallback):
    # a GENUINE mid-sequence host-feed underflow can still end with generated >= nsamps, so
    # the "output was complete" argument does not hold and the gate must NOT swallow.
    _stub_generated_count(monkeypatch, [1000, 1003])    # delta 3 == nsamps -> gate would pass
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    fake_task = NiDAQRunner._session
    nidaq_runner._TASK_META[fake_task] = (nidaq_runner._RATE, "/Dev1/PFI0", False)
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    with pytest.raises(_FakeDaqError):
        NiDAQRunner.wait()
    assert NiDAQRunner.underflows_swallowed == 0


def test_delta_gate_realigns_per_arm_with_changing_nsamps(stub_device, monkeypatch):
    # Consecutive shots of DIFFERENT length on the SAME cached task: _last_nsamps AND _gen_base
    # must both re-latch at each arm, so shot 2's 4-sample delta is short of its 5 samples even
    # though it exceeds shot 1's 3.
    _stub_generated_count(monkeypatch, [1000, 1003, 1007])
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)             # 3 samples
    NiDAQRunner.wait()                                  # clean -> task reused
    assert len(stub_device) == 1
    NiDAQRunner.run(_CH, _CLK, _TRG, np.zeros((5, 2)))   # 5 samples
    assert NiDAQRunner._last_nsamps == 5 and NiDAQRunner._gen_base == 1003
    monkeypatch.setattr(nidaq_runner, "_wait_task", _raise_underflow)
    with pytest.raises(_FakeDaqError):                  # delta 4 < 5
        NiDAQRunner.wait()
    assert NiDAQRunner.underflows_swallowed == 0


def test_wait_matches_underflow_by_message_only(stub_device, monkeypatch):
    # No error_code attribute at all (some wrappers re-raise plain Exceptions): the code must
    # still be recognised from the message text and triaged by the same gate.
    _stub_generated_count(monkeypatch, [1000, 1003])
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)

    def boom(task):
        raise _MsgOnlyError("NI-DAQmx generation failed. Status Code: -200018")

    monkeypatch.setattr(nidaq_runner, "_wait_task", boom)
    NiDAQRunner.wait()                                  # recognised -> swallowed
    assert NiDAQRunner.underflows_swallowed == 1
    assert NiDAQRunner._cache_in_use is False


def test_wait_reraises_non_underflow(stub_device, monkeypatch):
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)

    def boom(task):
        raise RuntimeError("boom")

    monkeypatch.setattr(nidaq_runner, "_wait_task", boom)
    with pytest.raises(RuntimeError, match="boom"):
        NiDAQRunner.wait()


def test_run_retries_once_on_stale_underflow_at_arm(stub_device, monkeypatch):
    # A -200018 latched by the PREVIOUS shot's burst surfaces at the new shot's first DAQmx
    # call (task.stop() inside _write_and_start) -> rebuild the task and re-arm once.
    calls = []

    def flaky(task, samples):
        calls.append(task)
        if len(calls) == 1:
            raise _FakeDaqError()
        task.writes.append(samples)
        task.started += 1

    monkeypatch.setattr(nidaq_runner, "_write_and_start", flaky)
    NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)            # must NOT raise
    assert len(calls) == 2                             # armed again after the rebuild
    assert calls[0] is not calls[1]                    # on a FRESH task
    assert len(stub_device) == 2                       # initial build + rebuild
    assert stub_device[0][0].closed is True            # the errored task was closed


def test_run_raises_after_second_stale_underflow(stub_device, monkeypatch):
    monkeypatch.setattr(nidaq_runner, "_write_and_start", _raise_underflow)
    with pytest.raises(_FakeDaqError):
        NiDAQRunner.run(_CH, _CLK, _TRG, _DATA)
    assert len(stub_device) == 2                       # retried exactly once, then gave up
    assert NiDAQRunner._cache_in_use is True           # never waited -> next arm rebuilds again
    assert stub_device[1][0].stops == 1                # the failed retry task was stopped
