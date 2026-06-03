"""Phase-5 nidaq_runner: the transpose (#1 silent-bug risk) + structural session caching.

NO-HARDWARE: the device hooks (_build_task / _write_and_start / _wait_task / _close_task) are
stubbed, so the pure transpose and the cache-invalidation DECISION are exercised without
nidaqmx or the card. The live card-listen / clock-out is NEEDS-HARDWARE (maintenance window).
"""

import pytest

import nidaq_runner
from nidaq_runner import NiDAQRunner

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
        self.stopped = 0
        self.closed = False


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
