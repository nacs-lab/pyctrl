"""test_ttl_manager.py -- per-TTL-channel hardware timing manager: serialize byte format +
engine_run config resolver.

A TTL manager is ``ExpSeq.add_ttl_mgr(chn, off_delay, on_delay, skip_time, min_time, off_val)``
-> a per-device record inside the version-1 (or version-2 when a trigger is also set) ``ZYNQZYNQ``
backend block -> libnacs (zynq/bc_gen.cpp) shifts every edge on that channel (on_delay/off_delay
FIRE THE EDGE EARLIER, skip_time drops short off-intervals, min_time extends short on-times).

Two halves, both NO-HARDWARE (pure byte math + a config resolver; no engine):
  * the SERIALIZED record layout (THE ONE RULE: matches MATLAB ExpSeq collectBackendData --
    [cid:4B LE][off_delay:8B LE][on_delay:8B LE][skip:8B LE][min:8B LE][off_val:1B]), the per-
    device count byte, and the used-channel gating (a manager on an UNUSED channel is dropped), and
  * the resolver ``engine_run._ttl_managers_config`` (expConfig ``consts['TTLManagers']`` +
    per-scan ``runp().TTLManagers`` overrides, all-zero entries dropped, arg-order off/on swap).
"""

import struct

import pytest

import seq_manager
import engine_run
from exp_seq import ExpSeq

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)     # 1 tick = 1 ms -> on_delay 0.02 s = 20 ticks
    yield
    seq_manager.override_tick_per_sec(0)


# --------------------------------------------------------------------------- #
# serialize: the per-device ZYNQZYNQ TTL-manager record (byte-identical to MATLAB)
# --------------------------------------------------------------------------- #
def test_manager_record_on_used_channel_in_blob():
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)          # USE the channel (assigns a cid)
    # off_delay=0, on_delay=0.02 s -> 20 ticks; skip/min 0; off_val False
    s.add_ttl_mgr("FPGA1/TTL2", 0.0, 0.02, 0.0, 0.0, False)
    blob = bytes(s.serialize())
    assert b"ZYNQZYNQ" in blob
    # the delay fields (off=0, on=20, skip=0, min=0 ticks) must appear in the blob
    assert struct.pack("<qqqq", 0, 20, 0, 0) in blob


def test_manager_serializes_full_record_and_count():
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)
    s.add_ttl_mgr("FPGA1/TTL2", 0.003, 0.02, 0.001, 0.005, True)   # 3,20,1,5 ticks; off_val=1
    recs = s.collect_backend_data()
    assert len(recs) == 1
    dev = recs[0]
    # layout: "FPGA1\0" + int32 len + "ZYNQZYNQ" + ver(1B) + count(1B) + record...
    name, _, rest = dev.partition(b"\x00")
    assert name == b"FPGA1"
    body = rest[4:]                              # strip the <i length prefix
    assert body[:8] == b"ZYNQZYNQ"
    ver, count = body[8], body[9]
    assert ver == 1 and count == 1               # version-1 (no trigger), one manager
    rec = body[10:]
    off_t, on_t, skip_t, min_t = struct.unpack("<qqqq", rec[4:36])   # skip the 4B cid
    assert (off_t, on_t, skip_t, min_t) == (3, 20, 1, 5)
    assert rec[36] == 1                          # off_val


def test_manager_on_unused_channel_is_dropped():
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)           # TTL2 used, TTL9 NOT used
    s.add_ttl_mgr("FPGA1/TTL9", 0.0, 0.02, 0.0, 0.0, False)
    assert s.collect_backend_data() == []        # dropped: never assigned a cid


def test_no_manager_no_backend_data():
    s = ExpSeq()
    s.add_step(1).add("FPGA1/TTL2", 1)
    assert s.collect_backend_data() == []        # byte-inert when none registered


def test_add_ttl_mgr_rejects_negative():
    s = ExpSeq()
    with pytest.raises(ValueError):
        s.add_ttl_mgr("FPGA1/TTL2", -1e-6, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# engine_run._ttl_managers_config: consts source of truth + runp() overrides
# --------------------------------------------------------------------------- #
_MISSING = object()


class _RunP:
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
    def __init__(self, ttl_managers=None):
        self.consts = {} if ttl_managers is None else {"TTLManagers": ttl_managers}


def test_resolver_reads_consts_and_swaps_to_add_ttl_mgr_order():
    cfg = _SeqCfg({"TTL556RydAWG": {"on_delay": 1.5e-6, "off_delay": 0.0,
                                    "skip_time": 0.0, "min_time": 0.0, "off_val": False}})
    got = engine_run._ttl_managers_config(_ScanGroup(), cfg)
    # tuple order is (chn, off_delay, on_delay, skip, min, off_val) -- ready for add_ttl_mgr
    assert got == [("TTL556RydAWG", 0.0, 1.5e-6, 0.0, 0.0, False)]


def test_resolver_drops_all_zero_entries():
    cfg = _SeqCfg({"TTL308RydAWG": {"on_delay": 0.0, "off_delay": 0.0,
                                    "skip_time": 0.0, "min_time": 0.0, "off_val": True}})
    assert engine_run._ttl_managers_config(_ScanGroup(), cfg) == []   # no-op -> not emitted


def test_resolver_runp_overrides_consts_per_field():
    cfg = _SeqCfg({"TTL556RydAWG": {"on_delay": 1.5e-6, "off_delay": 0.0,
                                    "skip_time": 0.0, "min_time": 0.0, "off_val": False}})
    sg = _ScanGroup(TTLManagers={"TTL556RydAWG": {"on_delay": 2.0e-6}})   # bump on_delay only
    got = engine_run._ttl_managers_config(sg, cfg)
    assert got == [("TTL556RydAWG", 0.0, 2.0e-6, 0.0, 0.0, False)]


def test_resolver_consts_absent_returns_empty():
    assert engine_run._ttl_managers_config(_ScanGroup(), _SeqCfg()) == []
