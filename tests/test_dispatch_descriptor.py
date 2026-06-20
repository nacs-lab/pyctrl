"""Phase-5 dispatch_descriptor: descriptor JSON -> ScanGroup + resolved seq.

Two layers (PYTHON_FRONTEND_PLAN.md Phase 5, finding E + the L1/L2 byte-oracle plan):

  * **Unit / trap tests** (NO config, fake seq resolver): the sweep + collapse + coercion
    + resolution rules a naive port gets wrong -- linspace ``n==1`` STOP endpoint,
    single-element numeric collapse, sweep-before-handle ordering, ``bad_object_value``,
    ``auto`` reserved, JSON-int -> float coercion, NumPerGroup auto-derive, opts.

  * **L2 per-point BYTE oracle** (real expConfig + tick 1e12): author descriptor fixtures
    that reproduce the EXISTING committed W6 scans (test_scan_point_oracle.py's
    ``build_spectrum399`` / ``build_imaging_hist``), run them THROUGH dispatch_descriptor,
    and assert each point's ``serialize()`` is byte-identical to the committed
    ``reference_scan_point/scan_point_reference.json`` (MATLAB ground truth). This proves
    the descriptor -> ScanGroup -> per-point bytes path end to end with no new capture --
    and confirms the JSON-int -> float coercion matches MATLAB ``jsondecode`` -> double.

NO-HARDWARE throughout: the engine is never loaded; serialize() never runs the deferred
camera/AWG/server callbacks.
"""

import json
import os

import pytest

import compare_bytes
import seq_manager
from conftest import _TESTS_DIR
from dispatch_descriptor import (
    DispatchResult, NotMigratedError, dispatch_descriptor, _linspace, _logspace)
from exp_seq import ExpSeq
from scan_group import ScanGroup
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware


# A resolver that fabricates a stub seq for any name, so trap tests need not import the
# real YbSeqs (and can't fail on migration status).
def _fake_resolver(name):
    def _stub(s):
        return s
    _stub.__name__ = name
    return _stub


def _dispatch(desc):
    return dispatch_descriptor(desc, seq_resolver=_fake_resolver)


# --------------------------------------------------------------------------- #
# linspace / logspace endpoint math (the n==1 STOP trap)
# --------------------------------------------------------------------------- #
class TestLinspaceLogspace:
    def test_linspace_multi(self):
        assert _linspace(220e6, 360e6, 5) == [220e6, 255e6, 290e6, 325e6, 360e6]

    def test_linspace_n1_is_stop_not_start(self):
        # MATLAB linspace(a,b,1) == b (the STOP); NumPy returns a (the START).
        assert _linspace(10.0, 99.0, 1) == [99.0]

    def test_linspace_n0_empty(self):
        assert _linspace(1.0, 2.0, 0) == []

    def test_linspace_endpoint_exact(self):
        # The stop endpoint is pinned exactly (no float drift).
        assert _linspace(0.0, 1.0, 11)[-1] == 1.0

    def test_logspace_n1_is_ten_pow_stop(self):
        assert _logspace(1.0, 3.0, 1) == [1000.0]

    def test_logspace_multi(self):
        assert _logspace(0.0, 2.0, 3) == [1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
# sweep parsing + single-element collapse
# --------------------------------------------------------------------------- #
class TestSweeps:
    def test_values_sweep_builds_axis(self):
        res = _dispatch({"seq": "S", "params": {
            "A.B": {"scan": 1, "values": [1, 2, 3]}}})
        assert res.scangroup.nseq() == 3

    def test_linspace_sweep_builds_axis(self):
        res = _dispatch({"seq": "S", "params": {
            "A.B": {"scan": 1, "linspace": [0, 10, 6]}}})
        assert res.scangroup.nseq() == 6

    def test_single_element_numeric_sweep_collapses_to_fixed(self):
        # [5] is not an array (ScanGroup._isarray) -> fixed param -> nseq 1, value readable.
        res = _dispatch({"seq": "S", "params": {
            "A.B": {"scan": 1, "values": [5]}}})
        g = res.scangroup
        assert g.nseq() == 1
        assert g().A.B() == 5.0

    def test_linspace_n1_collapses_to_stop_endpoint(self):
        # linspace(2,9,1) -> [9] -> collapses to the fixed value 9 (STOP), never 2 (START).
        res = _dispatch({"seq": "S", "params": {
            "A.B": {"scan": 1, "linspace": [2, 9, 1]}}})
        g = res.scangroup
        assert g.nseq() == 1
        assert g().A.B() == 9.0

    def test_2d_sweep_point_count(self):
        res = _dispatch({"seq": "S", "params": {
            "X.Freq": {"scan": 1, "values": [-5e6, 0.0]},
            "X.Amp": {"scan": 2, "values": [0.2, 0.3]}}})
        assert res.scangroup.nseq() == 4

    def test_sweep_dim_must_be_positive_integer(self):
        with pytest.raises(ValueError, match="positive integer"):
            _dispatch({"seq": "S", "params": {"A": {"scan": 0, "values": [1, 2]}}})
        with pytest.raises(ValueError, match="positive integer"):
            _dispatch({"seq": "S", "params": {"A": {"scan": 1.5, "values": [1, 2]}}})

    def test_sweep_exactly_one_kind(self):
        with pytest.raises(ValueError, match="exactly one"):
            _dispatch({"seq": "S", "params": {
                "A": {"scan": 1, "linspace": [0, 1, 2], "values": [1, 2]}}})
        with pytest.raises(ValueError, match="exactly one"):
            _dispatch({"seq": "S", "params": {"A": {"scan": 1}}})

    def test_linspace_needs_three_elements(self):
        with pytest.raises(ValueError, match="linspace must be"):
            _dispatch({"seq": "S", "params": {"A": {"scan": 1, "linspace": [0, 1]}}})


# --------------------------------------------------------------------------- #
# value decoding: coercion, handles, bad objects
# --------------------------------------------------------------------------- #
class TestValueDecoding:
    def test_json_int_param_coerced_to_float(self):
        # jsondecode yields double; a Python int would tag ARG_CONST_INT32 (byte divergence).
        res = _dispatch({"seq": "S", "params": {"A.B": 1}})
        val = res.scangroup().A.B()
        assert val == 1.0 and isinstance(val, float)

    def test_bool_param_stays_bool(self):
        res = _dispatch({"seq": "S", "params": {"A.B": True}})
        val = res.scangroup().A.B()
        assert val is True

    def test_string_param_passthrough(self):
        res = _dispatch({"seq": "S", "params": {"A.B": "hello"}})
        assert res.scangroup().A.B() == "hello"

    def test_numeric_array_param_coerced(self):
        res = _dispatch({"seq": "S", "params": {"A.B": [1, 2, 3]}})
        assert res.scangroup().A.B() == [1.0, 2.0, 3.0]

    def test_handle_value_resolves_via_resolver(self):
        res = _dispatch({"seq": "S", "params": {"A.B": {"@": "MyFunc"}}})
        fn = res.scangroup().A.B()
        assert callable(fn) and fn.__name__ == "MyFunc"

    def test_bad_object_value_rejected(self):
        # A dict that is neither a sweep ({"scan":...}) nor a handle ({"@":...}).
        with pytest.raises(ValueError, match="unrecognized object value"):
            _dispatch({"seq": "S", "params": {"A.B": {"nope": 1}}})

    def test_sweep_checked_before_handle(self):
        # A dict carrying BOTH scan and @ is a sweep (sweep-check runs first); the @ is
        # ignored. Mirrors dispatch_descriptor.m running parse_sweep before decode_value.
        res = _dispatch({"seq": "S", "params": {
            "A.B": {"scan": 1, "values": [1, 2], "@": "Ignored"}}})
        assert res.scangroup.nseq() == 2


# --------------------------------------------------------------------------- #
# seq resolution
# --------------------------------------------------------------------------- #
class TestSeqResolution:
    def test_string_seq(self):
        res = _dispatch({"seq": "CoolingSeq"})
        assert res.seq_name == "CoolingSeq" and res.seq.__name__ == "CoolingSeq"

    def test_handle_seq(self):
        res = _dispatch({"seq": {"@": "RearrangeCommSeq2"}})
        assert res.seq_name == "RearrangeCommSeq2"

    def test_auto_reserved(self):
        with pytest.raises(ValueError, match="auto.*reserved"):
            _dispatch({"seq": "auto"})

    def test_missing_seq(self):
        with pytest.raises(ValueError, match="must set"):
            _dispatch({"params": {"A": 1}})

    def test_bad_seq_name(self):
        with pytest.raises(ValueError, match="valid identifier"):
            _dispatch({"seq": "123 not ident"})

    def test_not_migrated_seq_raises(self):
        # Real resolver (import-by-convention): a bogus name is not importable.
        with pytest.raises(NotMigratedError):
            dispatch_descriptor({"seq": "ThisSeqDoesNotExistAnywhere999"})

    def test_label_falls_back_to_seq_name(self):
        assert _dispatch({"seq": "S"}).label == "S"
        assert _dispatch({"seq": "S", "label": "My scan"}).label == "My scan"


# --------------------------------------------------------------------------- #
# runp + opts + input forms
# --------------------------------------------------------------------------- #
class TestRunpOptsInput:
    def test_runp_assignment(self):
        res = _dispatch({"seq": "S", "runp": {"NumImages": 2, "Scramble": 1}})
        runp = res.scangroup.runp()
        assert runp.NumImages() == 2.0
        assert runp.Scramble() == 1.0

    def test_numpergroup_autoderived_when_absent(self):
        # nseq==3 -> max(3*20, 100) == 100.
        res = _dispatch({"seq": "S", "params": {"A": {"scan": 1, "values": [1, 2, 3]}}})
        assert res.scangroup.runp().NumPerGroup() == 100.0

    def test_numpergroup_autoderived_scales_with_nseq(self):
        vals = list(range(1, 11))   # 10 points -> max(10*20, 100) == 200
        res = _dispatch({"seq": "S", "params": {"A": {"scan": 1, "values": vals}}})
        assert res.scangroup.runp().NumPerGroup() == 200.0

    def test_numpergroup_explicit_kept(self):
        res = _dispatch({"seq": "S", "runp": {"NumPerGroup": 9999}})
        assert res.scangroup.runp().NumPerGroup() == 9999.0

    def test_numpergroup_zero_sentinel_rederived(self):
        res = _dispatch({"seq": "S", "runp": {"NumPerGroup": 0}})
        assert res.scangroup.runp().NumPerGroup() == 100.0

    def test_opts_unpacked_in_order(self):
        res = _dispatch({"seq": "S", "opts": [["random", True], ["tstartwait", 0.1]]})
        assert res.opts == [("random", True), ("tstartwait", 0.1)]

    def test_opts_handle_value_resolved(self):
        res = _dispatch({"seq": "S", "opts": [["pre_cb", {"@": "MyCb"}]]})
        (key, val), = res.opts
        assert key == "pre_cb" and callable(val) and val.__name__ == "MyCb"

    def test_json_string_input(self):
        res = _dispatch(json.dumps({"seq": "S", "params": {"A.B": 2}}))
        assert res.scangroup().A.B() == 2.0

    def test_dict_and_string_equivalent(self):
        d = {"seq": "S", "params": {"A.B": {"scan": 1, "values": [1, 2, 3]}}}
        assert _dispatch(d).scangroup.nseq() == _dispatch(json.dumps(d)).scangroup.nseq()


# =========================================================================== #
# L2 per-point BYTE oracle -- reproduce the committed W6 scans via descriptors
# =========================================================================== #
_REF = os.path.join(_TESTS_DIR, "reference_scan_point", "scan_point_reference.json")


def _reference():
    if not os.path.exists(_REF):
        return {}
    with open(_REF) as f:
        return json.load(f)


_REF_DATA = _reference()
_needs_ref = pytest.mark.skipif(
    not _REF_DATA, reason="no per-point capture (run tools/capture_scan_point_reference.m)")


# Descriptor fixtures that reproduce test_scan_point_oracle.py's hand-built scans EXACTLY
# (same swept values, same fixed params, same seq). linspace exercises the n>1 endpoint
# math; the other axes use explicit values. JSON ints (SLM.VServo, Pushout.*.Amp 0) are
# float-coerced -- which is what MATLAB jsondecode does, so they match the MATLAB reference.
_DESC = {
    "spectrum399": {
        "seq": "PushoutSurvival399Seq",
        "params": {
            "Pushout.Blue.Amp1": 0.25,
            "Pushout.Blue.Freq": {"scan": 1, "linspace": [220e6, 360e6, 5]},
            "Pushout.Time": 10e-3,
        },
        "runp": {"NumPerGroup": 10000, "NumImages": 2, "Scramble": 1},
    },
    "imaging_hist": {
        "seq": "PushoutSurvivalSeq",
        "params": {
            "Imag399.ExposureTime": 100e-3,
            "SLM.VServo": 1,
            "Imag399.FreqDetuning": {"scan": 1, "values": [-5e6, 0]},
            "Imag399.Amp": {"scan": 2, "values": [0.2, 0.3]},
            "Pushout.Green.Amp": 0,
            "Pushout.Blue.Amp1": 0,
            "Pushout.Time": 10e-3,
        },
        "runp": {"NumImages": 2, "Scramble": 1},
    },
}


@pytest.fixture
def real_config():
    """Real expConfig + production tick rate; reset both in teardown (process singletons)."""
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    yield
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


@_needs_ref
@pytest.mark.parametrize("name", sorted(_DESC))
def test_dispatch_per_point_bytes_match_matlab(real_config, name):
    ref = _REF_DATA[name]
    res = dispatch_descriptor(_DESC[name])      # real import-by-convention resolver
    g = res.scangroup

    # L1 (structural): the descriptor expanded to the same point count as MATLAB.
    assert res.seq_name == ref["seq"]
    assert g.nseq() == ref["nseq"], "%s: nseq %d != %d" % (name, g.nseq(), ref["nseq"])

    want_hex = ref["points"]
    seen = set()
    for n in range(1, g.nseq() + 1):
        params = g.getseq(n)
        got = res.seq(ExpSeq(params)).serialize()
        want = bytes.fromhex(want_hex[n - 1])
        if got != want:
            d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
            raise AssertionError(
                "%s point %d/%d: %d bytes vs reference %d; first diff at %s"
                % (name, n, g.nseq(), len(got), len(want), d))
        seen.add(got)
    assert len(seen) > 1, "%s: expected the scan to vary the bytes across points" % name
