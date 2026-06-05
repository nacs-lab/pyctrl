"""Phase-5 scan_export + ybStartScan: build a ScanGroup imperatively, export -> descriptor.

NO-HARDWARE. The headline check is the ROUND TRIP: a ScanGroup built field-by-field, exported
with ``scangroup_to_descriptor``, then rebuilt with ``dispatch_descriptor``, must enumerate the
SAME points (``nseq`` + per-point ``getseq``) -- i.e. the exporter is a faithful inverse of the
dispatcher (Option A's one-payload-format guarantee). Plus unit cover for the value/seq/opts/runp
encoders and ybStartScan's wiring (submit injected -- no socket bound).
"""

import json

import pytest

from dispatch_descriptor import dispatch_descriptor
from scan_export import (linspace, logspace, matlab_colon, scangroup_to_descriptor,
                         _encode_value, _seq_name)
from scan_group import ScanGroup
from yb_start_scan import ybStartScan

pytestmark = pytest.mark.no_hardware


def _fake_resolver(name):
    """A seq/handle resolver that returns the name (no real import needed)."""
    return name


def _enumerate(group):
    """All points of a group as a list of (nested) param dicts."""
    return [group.getseq(n) for n in range(1, group.nseq() + 1)]


# --------------------------------------------------------------------------- #
# matlab_colon: bit-identical reproduction of MATLAB's colon operator
# --------------------------------------------------------------------------- #
class TestMatlabColon:
    def test_integer_step_exact(self):
        # Integer-valued progressions are exact regardless of algorithm.
        assert matlab_colon(220.0, 35.0, 360.0) == [220.0, 255.0, 290.0, 325.0, 360.0]
        assert matlab_colon(0.0, 1.0, 4.0) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_count_and_endpoints(self):
        f = matlab_colon(103.5, 0.1, 106.5)          # Spectrum556 sweep
        assert len(f) == 31 and f[0] == 103.5 and f[-1] == 106.5
        v = matlab_colon(2.0, 0.6, 9.0)              # BlueLAC sweep base (12 pts; does NOT reach 9)
        assert len(v) == 12 and v[0] == 2.0 and v[-1] == 8.6

    def test_symmetric_differs_from_naive_at_byte_critical_point(self):
        # THE reason matlab_colon exists: MATLAB's colon is NOT start + k*step. The naive sum
        # lands exactly on 6.2 at index 7 of 2:0.6:9; MATLAB (and matlab_colon) yield one ULP
        # below. A swept value serializes as a raw float64, so this 1-ULP gap is byte-visible.
        v = matlab_colon(2.0, 0.6, 9.0)
        naive = [2.0 + 0.6 * k for k in range(12)]
        assert v != naive
        assert v[7] * 1e6 == 6199999.999999999       # MATLAB R2023a-exact (verified bit-for-bit)
        assert v[7] * 1e6 != 6200000.0               # what the naive a+k*step gives

    def test_edge_cases(self):
        assert matlab_colon(5.0, 1.0, 5.0) == [5.0]   # single point
        assert matlab_colon(5.0, 1.0, 4.0) == []      # empty (start past stop, positive step)
        with pytest.raises(ValueError):
            matlab_colon(0.0, 0.0, 1.0)               # zero step


# --------------------------------------------------------------------------- #
# round trip: g -> descriptor -> dispatch -> same enumeration
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_fixed_params_only(self):
        g = ScanGroup()
        g().Cooling.Detuning = 25e6
        g().Pushout.Green.Power = 0.5
        desc = scangroup_to_descriptor(g, "MySeq")
        rebuilt = dispatch_descriptor(desc, seq_resolver=_fake_resolver).scangroup
        assert rebuilt.nseq() == g.nseq() == 1
        assert _enumerate(rebuilt) == _enumerate(g)
        assert desc["seq"] == "MySeq"

    def test_one_d_sweep(self):
        g = ScanGroup()
        g().Cooling.Detuning = 25e6
        g(1).Pushout.Green.Freq.scan(linspace(105e6, 107e6, 23))
        desc = scangroup_to_descriptor(g, "Spectrum556Seq")
        # The sweep is exported as an explicit values array on dim 1.
        assert desc["params"]["Pushout.Green.Freq"]["scan"] == 1
        assert len(desc["params"]["Pushout.Green.Freq"]["values"]) == 23
        rebuilt = dispatch_descriptor(desc, seq_resolver=_fake_resolver).scangroup
        assert rebuilt.nseq() == g.nseq() == 23
        assert _enumerate(rebuilt) == _enumerate(g)

    def test_two_d_sweep(self):
        g = ScanGroup()
        g(1).A.x.scan(1, [1.0, 2.0, 3.0])
        g(1).B.y.scan(2, [10.0, 20.0])
        desc = scangroup_to_descriptor(g, "Seq2D")
        assert desc["params"]["A.x"]["scan"] == 1
        assert desc["params"]["B.y"]["scan"] == 2
        rebuilt = dispatch_descriptor(desc, seq_resolver=_fake_resolver).scangroup
        assert rebuilt.nseq() == g.nseq() == 6          # 3 x 2
        assert _enumerate(rebuilt) == _enumerate(g)

    def test_runp_exported(self):
        g = ScanGroup()
        g().X = 1.0
        g.runp().NumPerGroup = 200
        g.runp().Scramble = True
        desc = scangroup_to_descriptor(g, "S")
        assert desc["runp"]["NumPerGroup"] == 200.0
        assert desc["runp"]["Scramble"] is True

    def test_callable_seq_resolves_to_name(self):
        def CoolingSeq(s):
            return s

        g = ScanGroup()
        g().X = 1.0
        desc = scangroup_to_descriptor(g, CoolingSeq)
        assert desc["seq"] == "CoolingSeq"


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
class TestEncoders:
    def test_int_becomes_float(self):
        assert _encode_value(5) == 5.0 and isinstance(_encode_value(5), float)

    def test_bool_stays_bool(self):
        assert _encode_value(True) is True               # not coerced to 1.0

    def test_string_and_none_pass_through(self):
        assert _encode_value("hi") == "hi"
        assert _encode_value(None) is None

    def test_list_encoded_elementwise(self):
        assert _encode_value([1, 2, 3]) == [1.0, 2.0, 3.0]

    def test_callable_becomes_handle(self):
        def Foo():
            pass

        assert _encode_value(Foo) == {"@": "Foo"}

    def test_lambda_handle_rejected(self):
        with pytest.raises(ValueError):
            _encode_value(lambda: None)                   # "<lambda>" is not an identifier

    def test_seq_name_validation(self):
        assert _seq_name("Abc_1") == "Abc_1"
        with pytest.raises(ValueError):
            _seq_name("1bad")
        with pytest.raises(TypeError):
            _seq_name(123)


# --------------------------------------------------------------------------- #
# MATLAB endpoint semantics in the sweep builders
# --------------------------------------------------------------------------- #
class TestSweepBuilders:
    def test_linspace_endpoints(self):
        assert linspace(0, 10, 11)[0] == 0 and linspace(0, 10, 11)[-1] == 10
        assert len(linspace(0, 10, 11)) == 11

    def test_linspace_n1_is_stop(self):
        assert linspace(2, 9, 1) == [9.0]                 # STOP endpoint, not start

    def test_logspace(self):
        assert logspace(0, 2, 3) == [1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
class TestGuards:
    def test_multi_group_rejected(self):
        g = ScanGroup()
        g.new_empty()                                     # scan 1
        g.new_empty()                                     # scan 2 -> groupsize 2
        with pytest.raises(ValueError):
            scangroup_to_descriptor(g, "S")

    def test_non_scangroup_rejected(self):
        with pytest.raises(TypeError):
            scangroup_to_descriptor({"not": "a group"}, "S")


# --------------------------------------------------------------------------- #
# ybStartScan -- wiring (submit injected; no socket)
# --------------------------------------------------------------------------- #
class TestYbStartScan:
    def test_builds_descriptor_and_submits(self):
        g = ScanGroup()
        g().Cooling.Detuning = 25e6
        g(1).Pushout.Green.Freq.scan(linspace(105e6, 107e6, 5))
        captured = {}

        def fake_submit(desc_json, label):
            captured["desc"] = json.loads(desc_json)
            captured["label"] = label
            return 77

        rid = ybStartScan("Spectrum556Seq", g, rep=5, submit=fake_submit)
        assert rid == 77
        assert captured["label"] == "Spectrum556Seq"     # defaults to seq name
        d = captured["desc"]
        assert d["seq"] == "Spectrum556Seq"
        assert d["params"]["Pushout.Green.Freq"]["scan"] == 1
        # opts round-trip: rep exported as a [key, value] pair.
        assert ["rep", 5.0] in d["opts"]
        # and the descriptor dispatches back to the same enumeration.
        rebuilt = dispatch_descriptor(d, seq_resolver=_fake_resolver).scangroup
        assert _enumerate(rebuilt) == _enumerate(g)

    def test_callable_opt_becomes_handle(self):
        def my_pre_cb(n, a):
            pass

        g = ScanGroup()
        g().X = 1.0
        captured = {}

        def fake_submit(dj, lb):
            captured["desc"] = json.loads(dj)
            return 1

        ybStartScan("S", g, pre_cb=my_pre_cb, submit=fake_submit)
        assert ["pre_cb", {"@": "my_pre_cb"}] in captured["desc"]["opts"]

    def test_explicit_label(self):
        g = ScanGroup()
        g().X = 1.0
        seen = {}

        def fake_submit(dj, lb):
            seen["lb"] = lb
            return 1

        ybStartScan("S", g, label="my run", submit=fake_submit)
        assert seen["lb"] == "my run"
