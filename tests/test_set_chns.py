"""test_set_chns.py -- set_chns NI + FPGA routing / clocking-hold logic (no_hardware).

The manual set-channel operator tool (YbExptCtrl/set_chns.py) builds a one-shot ExpSeq and runs
it through the engine. These tests exercise the BUILD path only: generate() (the one
engine-needing call) is stubbed, so they prove channel resolution (NI V*/Dev1 -> NiDAQ/Dev1/N,
FPGA passthrough), the NI clocking hold (added only when an NI channel is present; opt-out via
ni_hold=0 for MATLAB-exact parity), and the varargin parsing -- all engine-free.
"""

import pytest

import seq_manager
import set_chns as sc
from exp_seq import ExpSeq
from mat_utils import mat_round
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_TICK = 1_000_000_000_000  # 1 ps tick (engine-free); matches the real engine's rate scale.


@pytest.fixture(autouse=True)
def _engine_free(monkeypatch):
    # Real config so V* aliases resolve to NiDAQ/Dev1/N; fixed tick so the build is engine-free.
    seq_manager.override_tick_per_sec(_TICK)
    SeqConfig.reset()
    SeqConfig.load_real()
    # generate() is the only engine-needing call in build_set_chns -> stub it to a no-op so the
    # build path (channel resolution + hold) runs without libnacs.
    monkeypatch.setattr(ExpSeq, "generate", lambda self, preserve=False: self)
    yield
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


def _hold_ticks(seconds):
    return mat_round(seconds * _TICK)   # mirrors wait()'s mat_round(t * time_scale)


class TestParsePairs:
    def test_flat(self):
        assert sc._parse_pairs(("FPGA1/TTL31", 1, "VElectrode1", 2.0)) == \
            [("FPGA1/TTL31", 1.0), ("VElectrode1", 2.0)]

    def test_pairs_list(self):
        assert sc._parse_pairs(([("FPGA1/TTL31", 1), ("VElectrode1", 2.0)],)) == \
            [("FPGA1/TTL31", 1.0), ("VElectrode1", 2.0)]

    def test_inline_tuples_flat(self):
        assert sc._parse_pairs((("FPGA1/TTL31", 1), ("VElectrode1", 2.0))) == \
            [("FPGA1/TTL31", 1.0), ("VElectrode1", 2.0)]

    def test_missing_value_raises(self):
        with pytest.raises(ValueError):
            sc._parse_pairs(("FPGA1/TTL31",))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sc._parse_pairs(())


class TestNiResolution:
    def test_ni_alias_translates(self):
        s = sc.build_set_chns("VElectrode1", 2.0)
        assert "NiDAQ/Dev1/12" in s.channel_names   # expConfig: VElectrode1 -> Dev1/12

    def test_ni_backend_name_translates(self):
        s = sc.build_set_chns("Dev1/7", -1.5)
        assert "NiDAQ/Dev1/7" in s.channel_names


class TestClockingHold:
    def test_ni_present_appends_default_hold(self):
        s = sc.build_set_chns("VElectrode1", 2.0)
        assert s.cur_seq_time.get_val() == _hold_ticks(1e-3)   # default ni_hold = 1 ms

    def test_fpga_only_stays_zero_length(self):
        s = sc.build_set_chns("FPGA1/TTL31", 1)
        assert not any(n.startswith("NiDAQ") for n in s.channel_names)
        assert s.cur_seq_time.get_val() == 0                   # no NI -> no hold added

    def test_ni_hold_zero_skips_hold(self):
        s = sc.build_set_chns("VElectrode1", 2.0, ni_hold=0)
        assert s.cur_seq_time.get_val() == 0                   # MATLAB-exact parity (NI may not latch)

    def test_custom_ni_hold(self):
        s = sc.build_set_chns("VElectrode1", 2.0, ni_hold=5e-3)
        assert s.cur_seq_time.get_val() == _hold_ticks(5e-3)

    def test_mixed_fpga_and_ni_holds(self):
        s = sc.build_set_chns("FPGA1/TTL31", 1, "VElectrode1", 2.0)
        assert "FPGA1/TTL31" in s.channel_names
        assert "NiDAQ/Dev1/12" in s.channel_names
        assert s.cur_seq_time.get_val() == _hold_ticks(1e-3)   # NI present -> hold


def test_set_chns_builds_then_runs(monkeypatch):
    """set_chns = build + run_real; assert it builds the same seq and hands it to run_real."""
    ran = {}
    monkeypatch.setattr("run_seq2.run_real", lambda s, **kw: ran.setdefault("seq", s))
    s = sc.set_chns("VElectrode1", 2.0)
    assert ran["seq"] is s
    assert "NiDAQ/Dev1/12" in s.channel_names
    assert s.cur_seq_time.get_val() == _hold_ticks(1e-3)
