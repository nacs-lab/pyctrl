"""test_line_trigger.py -- 60 Hz AC-line trigger: serialize byte format + runner config resolver.

The line trigger is ``ExpSeq.enable_global_wait_trigger(device, channel, raise_, timeout)`` ->
a version-2 ``ZYNQZYNQ`` backend block (``trig_type``/``chn``/``timeout_ns``) -> libnacs emits a
``WaitTrigger`` bytecode op so the FPGA waits for the line edge before each basic sequence.

Two halves, both NO-HARDWARE (pure byte math + a config resolver; no engine):
  * the SERIALIZED byte format (THE ONE RULE: matches MATLAB ``serializeTriggerData`` /
    ``collectBackendData`` -- trig_type 2=raise/1=lower, chn, int64 timeout_ns), and
  * the runner's resolver ``runner._line_trigger_config`` (expConfig ``consts['LineTrigger']`` +
    per-scan ``runp().LineTrigger*`` overrides, conservative-off fallback, skip-when-no-channel).
"""

import struct

import pytest

import seq_manager
import runner
from exp_seq import ExpSeq

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)


# --------------------------------------------------------------------------- #
# serialize: the version-2 ZYNQZYNQ trigger payload (byte-identical to MATLAB)
# --------------------------------------------------------------------------- #
def test_trigger_data_rising_edge():
    s = ExpSeq()
    s.enable_global_wait_trigger("FPGA1", 14, True, 0.02)
    # trig_type=2 (Raise), chn=14, timeout_ns=int64(0.02*1e9)=20_000_000
    assert s.serialize_trigger_data() == bytes([2, 14]) + struct.pack("<q", 20_000_000)


def test_trigger_data_falling_edge():
    s = ExpSeq()
    s.enable_global_wait_trigger("FPGA1", 7, False, 0.05)
    # trig_type=1 (Lower), chn=7, timeout_ns=50_000_000
    assert s.serialize_trigger_data() == bytes([1, 7]) + struct.pack("<q", 50_000_000)


def test_double_enable_raises():
    s = ExpSeq()
    s.enable_global_wait_trigger("FPGA1", 14, True, 0.02)
    with pytest.raises(ValueError):
        s.enable_global_wait_trigger("FPGA1", 14, True, 0.02)


def test_serialize_emits_trigger_block():
    """A trigger-enabled sequence carries the version-2 ZYNQZYNQ block + the trigger payload."""
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)
    s.enable_global_wait_trigger("FPGA1", 14, True, 0.02)
    blob = bytes(s.serialize())
    payload = bytes([2, 14]) + struct.pack("<q", 20_000_000)
    assert b"ZYNQZYNQ" in blob
    assert payload in blob


def test_serialize_without_trigger_has_no_payload():
    """The trigger is byte-inert unless enabled: the SAME sequence without it lacks the payload."""
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)
    blob = bytes(s.serialize())
    payload = bytes([2, 14]) + struct.pack("<q", 20_000_000)
    assert payload not in blob


# --------------------------------------------------------------------------- #
# runner._line_trigger_config: consts source of truth + runp() overrides
# --------------------------------------------------------------------------- #
_MISSING = object()


class _RunP:
    """DynProps-fallback fake: ``getattr(rp, name)(default)`` -> the set value or the default."""

    def __init__(self, **vals):
        self._vals = vals

    def __getattr__(self, name):
        v = self._vals.get(name, _MISSING)
        return lambda default=None: (default if v is _MISSING else v)


class _ScanGroup:
    def __init__(self, **runp_vals):
        self._rp = _RunP(**runp_vals)

    def runp(self):
        return self._rp


class _SeqCfg:
    def __init__(self, line_trigger=None):
        self.consts = {} if line_trigger is None else {"LineTrigger": line_trigger}


def test_resolver_disabled_returns_none():
    cfg = _SeqCfg({"Enable": False, "Device": "FPGA1", "Channel": 14,
                   "Raise": True, "Timeout": 0.02})
    assert runner._line_trigger_config(_ScanGroup(), cfg) is None


def test_resolver_enabled_with_channel():
    cfg = _SeqCfg({"Enable": True, "Device": "FPGA1", "Channel": 14,
                   "Raise": True, "Timeout": 0.02})
    got = runner._line_trigger_config(_ScanGroup(), cfg)
    assert got == {"device": "FPGA1", "channel": 14, "raise_": True, "timeout": 0.02}


def test_resolver_enabled_no_channel_skips_and_logs():
    cfg = _SeqCfg({"Enable": True, "Device": "FPGA1", "Channel": None,
                   "Raise": True, "Timeout": 0.02})
    logs = []
    assert runner._line_trigger_config(_ScanGroup(), cfg, log=logs.append) is None
    assert logs and "no input channel" in logs[0]


def test_resolver_runp_overrides_consts():
    # consts disabled, but the scan turns it on + sets channel/edge/timeout via runp().
    cfg = _SeqCfg({"Enable": False, "Device": "FPGA1", "Channel": None,
                   "Raise": True, "Timeout": 0.02})
    sg = _ScanGroup(LineTriggerEnable=True, LineTriggerChannel=22,
                    LineTriggerRaise=False, LineTriggerTimeout=0.03)
    got = runner._line_trigger_config(sg, cfg)
    assert got == {"device": "FPGA1", "channel": 22, "raise_": False, "timeout": 0.03}


def test_resolver_consts_absent_falls_back_off():
    # No LineTrigger subtree at all -> conservative fallback (Enable=False) -> None.
    assert runner._line_trigger_config(_ScanGroup(), _SeqCfg()) is None
