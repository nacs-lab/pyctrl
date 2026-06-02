"""Phase-4 W6 -- per-point BYTE oracle over REAL YbScans (links Phase 4 back to 1-3).

NO-HARDWARE: real expConfig (SeqConfig.load_real) + tick_per_sec = 1e12; the engine is
never loaded and serialize() never runs the deferred camera/AWG/server/MemoryMap callbacks.

This is the production-sufficient milestone: drive a REAL (scan, sequence) pair exactly as
RunScans/runSeq2 does per shot --

    params = g.getseq(n);  s = ExpSeq(params);  seqfn(s);  bytes = s.serialize()

-- for every point of the scan, and assert each point is BYTE-IDENTICAL to MATLAB. Where
test_ybseqs_build.py proved the no-param build (ExpSeq()) of these sequences, this proves the
SCAN seam: ExpSeq(getseq(n)) merges the per-point params onto consts (top-level field
replacement), the sequence reads the swept leaf, and the bytes still match -- end to end
across ScanGroup (Phase 4) -> ExpSeq/serialize (Phases 1-2) -> config/DynProps (Phase 3).

Pairs (mirrors matlab_new/YbScans/*; builders are twins of tools/scan_point_list.m):
  * spectrum399  : FreqPushOut399Scan (1-D scan of Pushout.Blue.Freq) -> PushoutSurvival399Seq
  * imaging_hist : imagingScan (2-D Imag399.FreqDetuning x Imag399.Amp)  -> PushoutSurvivalSeq

Ground truth: tests/reference_scan_point/scan_point_reference.json from
tools/capture_scan_point_reference.m (headless matlab -batch). The JSON is committed; the
default run needs no MATLAB.
"""

import json
import os

import pytest

import compare_bytes
import seq_manager
from conftest import _TESTS_DIR
from exp_seq import ExpSeq
from scan_group import ScanGroup
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_REF = os.path.join(_TESTS_DIR, "reference_scan_point", "scan_point_reference.json")


# --------------------------------------------------------------------------- #
# Twin builders of tools/scan_point_list.m (same scanned values; metadata +
# in-body hardware calls omitted -- neither affects the serialized bytes).
# --------------------------------------------------------------------------- #
def build_spectrum399():
    g = ScanGroup()
    g().Pushout.Blue.Amp = 0.25
    g().Pushout.Blue.Freq.scan(1, [v * 1e6 for v in range(220, 361, 35)])  # 5 points
    g().Pushout.Time = 10e-3
    g.runp().NumPerGroup = 10000
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    return g


def build_imaging_hist():
    g = ScanGroup()
    g().Imag399.ExposureTime = 100e-3
    g().SLM.VServo = 1
    g().Imag399.FreqDetuning.scan(1, [-5 * 1e6, 0 * 1e6])
    g().Imag399.Amp.scan(2, [0.2, 0.3])
    g().Pushout.Green.Amp = 0
    g().Pushout.Blue.Amp = 0
    g().Pushout.Time = 10e-3
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    return g


PAIRS = {
    "spectrum399": (build_spectrum399, "PushoutSurvival399Seq"),
    "imaging_hist": (build_imaging_hist, "PushoutSurvivalSeq"),
}


def _reference():
    if not os.path.exists(_REF):
        return {}
    with open(_REF) as f:
        return json.load(f)


_REF_DATA = _reference()
_needs_ref = pytest.mark.skipif(
    not _REF_DATA, reason="no per-point capture (run tools/capture_scan_point_reference.m)")


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
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_per_point_bytes_match_matlab(real_config, name):
    build, seqname = PAIRS[name]
    ref = _REF_DATA[name]
    assert ref["seq"] == seqname
    g = build()
    assert g.nseq() == ref["nseq"], "%s: nseq %d != %d" % (name, g.nseq(), ref["nseq"])

    mod = __import__(seqname)
    seqfn = getattr(mod, seqname)
    want_hex = ref["points"]

    seen = set()
    for n in range(1, g.nseq() + 1):
        params = g.getseq(n)
        got = seqfn(ExpSeq(params)).serialize()
        want = bytes.fromhex(want_hex[n - 1])
        if got != want:
            d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
            raise AssertionError(
                "%s point %d/%d: %d bytes vs reference %d; first diff at %s"
                % (name, n, g.nseq(), len(got), len(want), d))
        seen.add(got)

    # Sanity: the scan actually drives per-point byte variation (not a constant sequence).
    assert len(seen) > 1, "%s: expected the scan to vary the bytes across points" % name
