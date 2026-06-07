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


def test_per_pulse_formula_with_param_names():
    """A value that folds to a constant still carries a param-named formula (the user-facing
    'how is this number derived' hint), even though serialize() sees only the folded float."""
    s = ExpSeq({"Res": 3.0, "Det": -1.0, "Gain": 2.0})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("FreqX", s.C.Res() + s.C.Det())
        s.add_step(1).add("AmpX", s.C.Gain() * 2)
    pulses = sess.result()["pulses"]
    exprs = {tuple(e["params"]): e.get("expr") for e in pulses.values()}
    assert exprs[("Det", "Res")] == "Res + Det"
    assert exprs[("Gain",)] == "Gain * 2"


def test_backtrace_captured_per_pulse():
    """B3: each pulse records the source backtrace of its ``.add`` call (offline capture),
    keyed by the same pid as ``pulses``; the leaf frame is the user's call site, NOT the
    framework (proving the lib/ frames were stripped)."""
    s = ExpSeq({"A": 1.0})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", s.C.A())      # <- this line is the captured leaf
    res = sess.result()
    bts = res["backtraces"]
    assert len(bts) == 1
    (pid, frames), = bts.items()
    assert pid in res["pulses"]                         # same pid space as the pulse map
    leaf = frames[0]
    assert leaf["file"].endswith("test_provenance.py")  # the user call site, not lib/
    assert leaf["name"] == "test_backtrace_captured_per_pulse"
    assert leaf["line"] > 0


def test_backtrace_covers_param_less_pulse():
    """A constant pulse has NO param provenance (absent from ``pulses``) but STILL records a
    source backtrace -- captured before the param early-return, so clicking any pulse works."""
    s = ExpSeq()
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", 4)            # constant -> no params
    res = sess.result()
    assert res["pulses"] == {}                          # no param-bearing pulse recorded
    assert len(res["backtraces"]) == 1                  # ...but its source IS
    (_pid, frames), = res["backtraces"].items()
    assert frames[0]["name"] == "test_backtrace_covers_param_less_pulse"


def test_backtrace_capture_can_be_disabled():
    """``capture_bt=False`` skips the per-pulse stack walk (the xref is otherwise unchanged)."""
    s = ExpSeq({"A": 1.0})
    with provenance.capture(consts_dp=s.C, capture_bt=False) as sess:
        s.add_step(1).add("Device1/CH1", s.C.A())
    res = sess.result()
    assert res["backtraces"] == {}
    assert "A" in res["param_to_channels"]              # rest of the capture still works


def test_formula_cleanup_simplifies_ramp_noise():
    """The renderer collapses the tick<->second round-trip and names the pulse args, so a
    ramp reads cleanly (arg(0)->t, arg(1)->from, (X*N)/N -> X)."""
    c = provenance._cleanup_formula
    assert c("(GreenMOT.BFieldRampTime * 1000000000000) / 1000000000000") == \
        "GreenMOT.BFieldRampTime"
    assert c("arg(0) / 1000000000000") == "t"
    assert c("(arg(0) / 1000000000000)") == "t"
    assert c("arg(1)") == "from"
    # a genuine (10 * X) / 7 is NOT a round-trip (different constants) -> preserved
    assert c("(10 * GreenMOT.BiasCoilCurrent.X) / 7") == \
        "(10 * GreenMOT.BiasCoilCurrent.X) / 7"


def test_global_dep_layer_records_runtime_global():
    s = ExpSeq({"Foo": 3.0})
    g = s.new_global()                              # runtime global -> g(0)
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("AmpThing", g + s.C.Foo())
    res = sess.result()
    assert set(res["channel_to_params"]["AmpThing"]) == {"Foo", "g(0)"}


def test_wait_time_region_capture():
    """A param-driven wait maps to a time-axis region [t0, t1] (no channel output); a plain
    constant wait contributes nothing. (Fixture tick_per_sec=1000; ms = ticks * 1e-9.)"""
    s = ExpSeq({"LoadTime": 0.5})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", 4)        # advance to t=1000 ticks first
        s.wait(s.C.LoadTime())                      # 0.5 -> +500 ticks, tagged LoadTime
    tr = sess.result()["time_regions"]
    assert list(tr.keys()) == ["LoadTime"]
    (t0, t1), = tr["LoadTime"]
    assert t0 == pytest.approx(1000 * 1e-9)
    assert t1 == pytest.approx(1500 * 1e-9)

    s2 = ExpSeq()
    with provenance.capture(consts_dp=s2.C) as sess2:
        s2.add_step(1).add("Device1/CH1", 4)
        s2.wait(0.3)                                # constant wait -> no provenance
    assert sess2.result()["time_regions"] == {}


def test_wait_time_region_absolute_in_subsequence():
    """A wait built inside an OFFSET sub-sequence must land in ABSOLUTE time, not the
    sub-sequence's local frame. Regression for the bug where GreenMOT.CoolDown.HoldTime
    showed at ~65ms (local) instead of ~316ms (absolute). (tick_per_sec=1000; ms=ticks*1e-9.)"""
    s = ExpSeq({"Hold": 0.4})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", 4)        # parent advances to t=1000 ticks
        def sub(ss):
            ss.add_step(0.2).add("Device1/CH1", 5)  # +200 ticks LOCAL to the sub-seq
            ss.wait(s.C.Hold())                     # +400 ticks, tagged Hold
        s.add_step(sub)                            # sub-seq placed at parent t=1000 (offset)
    (t0, t1), = sess.result()["time_regions"]["Hold"]
    # LOCAL would be [200, 600] ticks; ABSOLUTE = offset(1000) + [200, 600] = [1200, 1600].
    assert t0 == pytest.approx(1200 * 1e-9)
    assert t1 == pytest.approx(1600 * 1e-9)


def test_pending_globals_zero_when_all_bands_placed():
    """A scan whose bands resolve WITHOUT globals reports pending_globals == 0 (nothing is
    waiting on the run's globals)."""
    s = ExpSeq({"LoadTime": 0.5})
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(1).add("Device1/CH1", 4)
        s.wait(s.C.LoadTime())                         # placed without any global
    res = sess.result()
    assert res["time_regions"]                         # the band IS placed
    assert res["pending_globals"] == 0


def test_pending_globals_counts_global_dependent_band():
    """A wait whose absolute position depends on a runtime global can't be placed without the
    run's globals -> it's skipped AND counted; supplying the global resolves it (count -> 0)."""
    s = ExpSeq({"Hold": 0.4})
    g = s.new_global()
    with provenance.capture(consts_dp=s.C, globals_dp=s.G) as sess:
        s.add_step(g * 0.001).add("Device1/CH1", 4)    # GLOBAL-dependent step -> offset needs g
        def sub(ss):
            ss.wait(s.C.Hold())                         # wait inside the offset sub-seq
        s.add_step(sub)
    # no globals supplied -> the Hold band can't be placed; counted as pending
    res = sess.result()
    assert res["pending_globals"] >= 1
    assert "Hold" not in res["time_regions"]
    # supplying the global resolves the offset -> band placed, nothing pending
    res2 = sess.result(globals_map={0: 1000.0})
    assert res2["pending_globals"] == 0
    assert "Hold" in res2["time_regions"]


def test_eval_num_substitutes_globals():
    """The numeric SeqVal evaluator substitutes captured globals (used to place
    global-dependent sub-sequence offsets) and bails when a global is unavailable."""
    s = ExpSeq()
    g = s.new_global()                              # -> SeqVal head H_GLOBAL, id 0
    expr = (g - 2.5207e8) * 2.0 + 100              # arithmetic over a runtime global
    assert provenance._eval_num(expr, {0: 2.5207e8}) == pytest.approx(100.0)
    assert provenance._eval_num(expr, None) is None       # no globals -> unresolved
    assert provenance._eval_num(expr, {}) is None         # global 0 missing -> unresolved
    assert provenance._eval_num(5.0, None) == pytest.approx(5.0)   # plain number


def test_step_boundaries_captured():
    """Top-level steps are recorded as labeled [start, end] absolute-ms spans (the phase
    ruler). Nested steps are NOT recorded. (tick_per_sec=1000; ms = ticks * 1e-9.)"""
    s = ExpSeq({"A": 1.0})
    def PhaseOne(ss):
        ss.add_step(0.3).add("Device1/CH1", 4)              # nested step (not a phase)
    def PhaseTwo(ss):
        ss.add_step(0.2).add("Device1/CH1", 5)
    with provenance.capture(consts_dp=s.C) as sess:
        s.add_step(PhaseOne)                                # top-level -> [0, 300] ticks
        s.add_step(PhaseTwo)                                # top-level -> [300, 500] ticks
    steps = sess.result()["steps"]
    got = [(st["label"], round(st["t0"] / 1e-9), round(st["t1"] / 1e-9)) for st in steps]
    assert got == [("PhaseOne", 0, 300), ("PhaseTwo", 300, 500)]


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
