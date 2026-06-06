"""test_provenance.py -- param<->channel provenance capture (SeqPlotter Task 4c).

NO-HARDWARE. Covers lib/provenance.py (TaggedFloat value-flow + global-dep walk +
ProvenanceSession), the inert-when-unset guarantee (byte path unaffected), and the
tools/provenance_scan.py per-point capture + xref.json writer. The producer feeds
yb_analysis/sequence/xref.py (the reader, in the superproject), whose by_file entry
shape -- {param_to_channels, channel_to_params} -- is asserted here.
"""

import json

import numpy as np
import pytest

import seq_manager
import provenance
import provenance_scan
from dyn_props import DynProps
from exp_seq import ExpSeq
from seq_val import SeqVal

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)
    provenance.end()                 # never leak an active session across tests


# --------------------------------------------------------------------------- #
# TaggedFloat: numeric behaviour + tag propagation through arithmetic.
# --------------------------------------------------------------------------- #
def test_taggedfloat_is_a_float():
    t = provenance.TaggedFloat(2.5, {"A"})
    assert isinstance(t, float)
    assert float(t) == 2.5
    assert t._prov == frozenset({"A"})


def test_taggedfloat_arithmetic_unions_provenance():
    a = provenance.TaggedFloat(2.0, {"A"})
    b = provenance.TaggedFloat(3.0, {"B"})
    for res in (a + b, a - b, a * b, a / b, a ** b):
        assert isinstance(res, provenance.TaggedFloat)
        assert res._prov == frozenset({"A", "B"})
    # mixing with a plain number keeps only the tagged operand's provenance
    assert (a + 5)._prov == frozenset({"A"})
    assert (5 + a)._prov == frozenset({"A"})         # __radd__
    assert (10 - a)._prov == frozenset({"A"})        # __rsub__
    assert (-a)._prov == frozenset({"A"})
    assert abs(provenance.TaggedFloat(-1.0, {"A"}))._prov == frozenset({"A"})


def test_taggedfloat_defers_to_seqval():
    # tagged * SeqVal must NOT be folded by TaggedFloat -> SeqVal stores the tag as an arg.
    s = ExpSeq()
    g = s.new_global()
    a = provenance.TaggedFloat(2.0, {"A"})
    node = a * g
    assert isinstance(node, SeqVal)
    assert any(isinstance(x, provenance.TaggedFloat) for x in node.args)


def test_taggedfloat_integral_index():
    n = provenance.TaggedFloat(3.0, {"N"})
    assert list(range(n)) == [0, 1, 2]
    assert [10, 20, 30][n - 1] == 30                 # n-1 is TaggedFloat(2.0) -> __index__
    with pytest.raises(TypeError):
        [0][provenance.TaggedFloat(0.5)]             # non-integral index


def test_tag_value_only_numbers():
    tv = provenance.tag_value
    assert isinstance(tv(3, "P"), provenance.TaggedFloat)
    assert isinstance(tv(np.int32(3), "P"), provenance.TaggedFloat)
    assert tv(True, "P") is True                     # bools are logical, not traced
    d = {"x": 1}
    assert tv(d, "P") is d                            # structs pass through
    # an already-tagged value unions
    assert tv(provenance.TaggedFloat(1.0, {"A"}), "B")._prov == frozenset({"A", "B"})


# --------------------------------------------------------------------------- #
# Inertness: with NO session the hooks are pass-throughs and the byte path is untouched.
# --------------------------------------------------------------------------- #
def _build_seq(c_ovr=None):
    s = ExpSeq(c_ovr) if c_ovr else ExpSeq()
    s.add_step(1).add("Device1/CH1", 4)
    s.add_step(0.5).add("Device2/CH3", lambda t: t * 2)
    return s


def test_on_access_inert_returns_value_unchanged():
    assert provenance._session is None
    v = object()
    assert provenance.on_access(DynProps({}), ("p",), v) is v
    assert provenance.on_pulse(None, 1, 7, 0.0, 0.0) is None  # no crash, no session


def test_serialize_identical_without_session():
    a = _build_seq().serialize()
    b = _build_seq().serialize()
    assert a == b


def test_serialize_identical_under_active_session():
    """A float-consts build serializes IDENTICALLY with provenance active (TaggedFloat is
    byte-safe): the tag rides on a float subclass that serializes as float64 like the bare
    value would. This is the guard that the offline pass can't corrupt a byte comparison."""
    c = {"Amp": 4.0, "Wait": 0.5}
    plain = ExpSeq(c)
    plain.add_step(plain.C.Wait()).add("Device1/CH1", plain.C.Amp())
    want = plain.serialize()

    tagged = ExpSeq(c)
    with provenance.capture(consts_dp=tagged.C):
        tagged.add_step(tagged.C.Wait()).add("Device1/CH1", tagged.C.Amp())
    assert tagged.serialize() == want


# --------------------------------------------------------------------------- #
# ProvenanceSession: namespace prefixes + the two-layer capture.
# --------------------------------------------------------------------------- #
def test_session_namespace_prefix():
    sess = provenance.ProvenanceSession()
    consts = DynProps({"X": 1.0})
    globs = DynProps({"Y": 2.0})
    sess.register(consts, "")
    sess.register(globs, "G.")
    assert sess.wrap(consts, ("X",), 1.0)._prov == frozenset({"X"})
    assert sess.wrap(globs, ("Y",), 2.0)._prov == frozenset({"G.Y"})
    # an UNREGISTERED DynProps (e.g. a fresh Consts()) defaults to the bare namespace
    other = DynProps({"Z": 3.0})
    assert sess.wrap(other, ("Z",), 3.0)._prov == frozenset({"Z"})


def test_value_flow_capture_matches_reader_example():
    """Reproduces yb_analysis/sequence/xref.py's documented example exactly."""
    s = ExpSeq({"Init": {"EOM616": {"Freq": 100.0}}})
    with provenance.capture(consts_dp=s.C, globals_dp=s.G) as sess:
        s.add_step(1).add("FreqEOM616", s.C.Init.EOM616.Freq())
    res = sess.result()
    assert res["param_to_channels"] == {"Init.EOM616.Freq": ["FreqEOM616"]}
    assert res["channel_to_params"] == {"FreqEOM616": ["Init.EOM616.Freq"]}


def test_capture_value_flow_through_callable_and_expression():
    s = ExpSeq({"A": 4.0, "B": 2.0})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", lambda t: s.C.A() * t + s.C.B())
    res = sess.result()
    # both A and B reach CH1 (A via the *t term, B via the + term)
    assert set(res["channel_to_params"]["Device1/CH1"]) == {"A", "B"}


def test_per_pulse_provenance_is_segment_specific():
    """Two pulses on ONE channel from DIFFERENT params -> each pulse maps to its OWN param
    (the fix for whole-channel coarseness). pulses + param_to_pids invert cleanly; the pulse
    id == the .seq's per-point pid, so the viewer maps a clicked point to just its segment."""
    s = ExpSeq({"A": 1.0, "B": 2.0})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", s.C.A())   # segment 1 <- A
        s.add_step(1).add("Device1/CH1", s.C.B())   # segment 2 <- B
    res = sess.result()
    # aggregate: the channel is fed by both params...
    assert set(res["channel_to_params"]["Device1/CH1"]) == {"A", "B"}
    # ...but per-pulse, each segment carries its OWN single param
    pulses = res["pulses"]
    assert len(pulses) == 2
    assert all(e["channel"] == "Device1/CH1" for e in pulses.values())
    assert {tuple(e["params"]) for e in pulses.values()} == {("A",), ("B",)}
    # param_to_pids inverts: each param -> exactly one pulse id present in `pulses`
    pids_a, pids_b = res["param_to_pids"]["A"], res["param_to_pids"]["B"]
    assert len(pids_a) == 1 and len(pids_b) == 1
    assert pulses[str(pids_a[0])]["params"] == ["A"]
    assert pulses[str(pids_b[0])]["params"] == ["B"]


def test_global_dep_layer_records_runtime_global():
    s = ExpSeq({"Foo": 3.0})
    g = s.new_global()                              # runtime global -> g(0)
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("AmpThing", g + s.C.Foo())
    res = sess.result()
    assert set(res["channel_to_params"]["AmpThing"]) == {"Foo", "g(0)"}


def test_disabled_pulse_contributes_nothing():
    s = ExpSeq({"A": 1.0})
    with provenance.capture(consts_dp=s.C) as sess:
        # conditional(False) -> the step is gated off; its add() early-returns before on_pulse
        s.conditional(False).add_step(1).add("Device1/CH1", s.C.A())
    assert sess.result()["channel_to_params"] == {}


def test_control_flow_dependent_edges_differ_per_point():
    def build(amp):
        s = ExpSeq({"Amp": amp})
        with provenance.capture(consts_dp=s.C) as sess:
            if s.C.Amp() > 5:                      # value-dependent branch
                s.add_step(1).add("HighChan", s.C.Amp())
            else:
                s.add_step(1).add("LowChan", s.C.Amp())
        return sess.result()["param_to_channels"]["Amp"]
    assert build(10.0) == ["HighChan"]
    assert build(1.0) == ["LowChan"]


def test_decorate_channel_name():
    d = provenance._decorate_channel
    assert d("FreqEOM616", None) == "FreqEOM616"
    assert d("nom", {"nom": ["nom"]}) == "nom"
    assert d("nom", {"nom": ["alias", "nom"]}) == "alias (nom)"


# --------------------------------------------------------------------------- #
# tools/provenance_scan.py: per-point capture + xref.json writer (reader format).
# --------------------------------------------------------------------------- #
def test_capture_point_xref():
    res = provenance_scan.capture_point_xref(
        lambda s: s.add_step(1).add("FreqEOM616", s.C.Init.EOM616.Freq()),
        {"Init": {"EOM616": {"Freq": 50.0}}})
    assert res["param_to_channels"] == {"Init.EOM616.Freq": ["FreqEOM616"]}


def test_write_xref_json_roundtrip(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    entry = {"param_to_channels": {"A": ["CH1"]}, "channel_to_params": {"CH1": ["A"]}}
    path = provenance_scan.write_xref_json(seq_dir, {"point_00001__seqid_1.seq": entry},
                                           scan_id="20250619170317")
    assert path is not None
    doc = json.loads(open(path, encoding="utf-8").read())
    assert doc["scan_id"] == "20250619170317"
    assert doc["by_file"]["point_00001__seqid_1.seq"] == entry


def test_write_xref_json_skips_empty(tmp_path):
    assert provenance_scan.write_xref_json(str(tmp_path / "sequence"), {}) is None
