"""Phase-4 W4 -- ScanGroup expansion-equality oracle (the headline cross-check).

NO-HARDWARE: reads a committed JSON reference; never loads the engine or MATLAB.

This is the key Phase-4 verification: build a battery of ScanGroups in Python, expand
every point with ``getseq(n)``, and assert the ordered list matches -- element for element,
field for field -- the ground truth captured from REAL MATLAB. Where the W3 tests check the
column-major expansion against *hand-written* expectations, this checks it against MATLAB
itself, so a column/row-major flip, an off-by-one, or a base-merge divergence shows up here.

Ground truth: ``tests/reference_scangroup/scangroup_reference.json``, produced engine-free by
``tools/capture_scangroup_reference.m`` over ``tools/scangroup_list.m`` (run once in a fresh
headless ``matlab -batch``; ScanGroup touches no SeqManager/SeqConfig/engine, so it is the
most engine-free capture in the suite). The Python ``BATTERY`` below MUST stay byte-for-byte
equivalent to the MATLAB builders in ``scangroup_list.m`` -- the only intended difference is
the scan-axis syntax (MATLAB ``.scan(dim) = vals`` vs Python ``.scan(dim, vals)``).

The comparison is field-order sensitive: MATLAB ``jsonencode`` preserves struct insertion
order and Python dicts preserve insertion order, and the merge/expansion algorithm produces
the same order in both -- so order equality is a faithful (and strict) extra check on top of
value equality. (Field order is byte-irrelevant for the FPGA blob, but matching it confirms
the algorithm is a faithful transliteration.)
"""

import json
import math
import os

import pytest

from scan_group import ScanGroup

pytestmark = pytest.mark.no_hardware

_REF_PATH = os.path.join(os.path.dirname(__file__),
                         "reference_scangroup", "scangroup_reference.json")


# --------------------------------------------------------------------------- #
# The Python battery -- a 1:1 twin of tools/scangroup_list.m.
# --------------------------------------------------------------------------- #
def build_fixed_only():
    g = ScanGroup()
    g().a = 1
    g().b.c = 2
    g().s = "hello"
    return g


def build_scan_1d():
    g = ScanGroup()
    g().amp = 0.5
    g().freq.scan(1, [10, 20, 30, 40])
    return g


def build_scan_2d():
    g = ScanGroup()
    g().fixed = 7
    g().c.scan(1, [1, 2, 3])
    g().d.scan(2, [10, 20])
    g.runp().NumImages = 2
    g.runp().NumPerGroup = 16
    g.runp().Scramble = 1
    return g


def build_awg_like():
    g = ScanGroup()
    g().AWG.AWG556.pulse_width_us.scan(1, [1, 2, 3, 4])
    g().AWG.AWG556.carrier_freq_MHz.scan(2, [100, 110])
    g().Pushout.delay = 1.3e-6
    return g


def build_mixed_float_1d():
    g = ScanGroup()
    g().t.scan(1, [0.1, 0.2, 0.3, 0.4, 0.5])
    g().n = 16
    return g


def build_two_scan_basemerge():
    g = ScanGroup()
    g().a = 1
    g().b = 2
    g().c.scan(1, [1, 2, 3])
    g(1).c = 3                 # scan 1 fixes c -> shadows the base scan axis
    g().d.scan(2, [1, 2])
    g(2).d = 0                 # scan 2 fixes d -> shadows the base scan axis
    g(2).k.a.b.c = 2           # nested fixed param on scan 2
    return g


# --- W5 production shapes (twins of scangroup_list.m's real-YbScans builders) - #
def build_spectrum399_like():
    g = ScanGroup()
    g().Pushout.Blue.Amp = 0.25
    g().Pushout.Blue.Freq.scan(1, [v * 1e6 for v in range(220, 361, 20)])  # 8 points
    g().Pushout.Time = 10e-3
    g.runp().NumPerGroup = 10000
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    g.runp().isGrid2 = 0
    g.runp().isInit = 0
    g.runp().isHC = 0
    g().scanname = "spectrum399_like"
    return g


def build_imaging_hist_like():
    g = ScanGroup()
    g().Imag399.ExposureTime = 100e-3
    g().SLM.VServo = 1
    g().Imag399.FreqDetuning.scan(1, [-5 * 1e6, 0 * 1e6])
    g().Imag399.Amp.scan(2, [0.2, 0.3])
    g().Pushout.Green.Amp = 0
    g().Pushout.Blue.Amp = 0
    g().Pushout.Time = 10e-3
    g.runp().NumPerGroup = 2 * 2 * 100
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    g.runp().isInit = 1
    g.runp().isHC = 0
    g().scanname = "imaging_hist_like"
    return g


def build_stirap_awg_like():
    g = ScanGroup()
    g().Imag399.ExposureTime = 100e-3
    g().BlueMOT.LoadingTime = 0.5
    g().AWG.AWG556.carrier_freq_MHz = 142.87
    g().AWG.AWG556.pulse_width_us = 4
    g().AWG.AWG556.steepness = 4
    g().AWG.AWG556.max_amplitude_vpp = 11
    g().AWG.AWG556.amplitude_scale = 1
    g().AWG.AWG308.carrier_freq_MHz = 200
    g().AWG.AWG308.pulse_width_us = 4
    g().AWG.AWG308.steepness = 4
    g().AWG.AWG308.max_amplitude_vpp = 6.5
    g().AWG.AWG308.amplitude_scale = 1
    g.runp().AWGs = ["AWG556", "AWG308"]
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.VRydTrap = 0.1
    g().Pushout.STIRAP.delay = 1.1e-6
    g().Pushout.STIRAP.ifReverse = True
    g().Pushout.STIRAP.reverse_delay = 1.1e-6
    g().Pushout.STIRAP.gap.scan(1, [v * 1e-9 for v in range(100, 801, 100)])  # 8 points
    g().Pushout.STIRAP.waitTime = 0.5e-6
    g.runp().NumPerGroup = 100000
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    g.runp().isInit = 0
    g.runp().isHC = 0
    g().scanname = "stirap_awg_like"
    return g


BATTERY = {
    "fixed_only": build_fixed_only,
    "scan_1d": build_scan_1d,
    "scan_2d": build_scan_2d,
    "awg_like": build_awg_like,
    "mixed_float_1d": build_mixed_float_1d,
    "two_scan_basemerge": build_two_scan_basemerge,
    "spectrum399_like": build_spectrum399_like,
    "imaging_hist_like": build_imaging_hist_like,
    "stirap_awg_like": build_stirap_awg_like,
}


@pytest.fixture(scope="module")
def reference():
    if not os.path.exists(_REF_PATH):
        pytest.skip("scangroup_reference.json missing; run "
                    "tools/capture_scangroup_reference.m in a headless MATLAB")
    with open(_REF_PATH, "r") as f:
        return json.load(f)


def _assert_equal(p, e, path):
    """Recursive, field-ORDER-sensitive, numeric-tolerant structural equality."""
    if isinstance(e, dict):
        assert isinstance(p, dict), "%s: expected dict, got %r" % (path, type(p))
        assert list(p.keys()) == list(e.keys()), (
            "%s: field order/set differs\n  python:  %s\n  matlab:  %s"
            % (path, list(p.keys()), list(e.keys())))
        for k in e:
            _assert_equal(p[k], e[k], "%s.%s" % (path, k))
    elif isinstance(e, list):
        assert isinstance(p, (list, tuple)), "%s: expected list, got %r" % (path, type(p))
        assert len(p) == len(e), "%s: length %d != %d" % (path, len(p), len(e))
        for i, (pp, ee) in enumerate(zip(p, e)):
            _assert_equal(pp, ee, "%s[%d]" % (path, i))
    elif isinstance(e, bool):
        assert p == e, "%s: %r != %r" % (path, p, e)
    elif isinstance(e, (int, float)):
        assert isinstance(p, (int, float)) and not isinstance(p, bool), (
            "%s: expected number, got %r" % (path, p))
        assert math.isclose(p, e, rel_tol=1e-12, abs_tol=1e-15), (
            "%s: %r != %r" % (path, p, e))
    else:
        assert p == e, "%s: %r != %r" % (path, p, e)


def test_battery_matches_reference_keys(reference):
    # Guard against the Python battery and the MATLAB list drifting apart.
    assert set(BATTERY) == set(reference)


@pytest.mark.parametrize("name", sorted(BATTERY))
def test_expansion_matches_matlab(name, reference):
    ref = reference[name]
    g = BATTERY[name]()

    assert g.groupsize() == ref["groupsize"], "%s: groupsize" % name
    assert g.nseq() == ref["nseq"], "%s: nseq" % name

    py_seqs = [g.getseq(n) for n in range(1, g.nseq() + 1)]
    assert len(py_seqs) == len(ref["seqs"]), "%s: seq count" % name
    for n, (p, e) in enumerate(zip(py_seqs, ref["seqs"]), start=1):
        _assert_equal(p, e, "%s.getseq(%d)" % (name, n))

    _assert_equal(g.dump()["runparam"], ref["runp"], "%s.runp" % name)
