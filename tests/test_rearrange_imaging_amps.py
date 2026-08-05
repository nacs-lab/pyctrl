"""test_rearrange_imaging_amps.py -- the per-frame 399 imaging knobs (DDS AMPS + 399 PULSE
LENGTH) on RearrangeCommSeq2 (2-round) and RearrangeCommSeq (single-round).

The knob (``rearrange_kwargs.extras.MidImgAmp1/2`` + ``FinImgAmp1/2``, 2026-07-27; the
single-round twin ``InitImgAmp1/2`` + ``FinImgAmp1/2``, 2026-07-28) lets a scan
sweep individual frames' 399 DDS amplitudes INDEPENDENTLY, which a plain
``g().Imag399.Amp1`` cannot do (scan params beat ByPattern in every bseq). See the policy block in
YbSeqs/RearrangeCommSeq2.py and yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md.

2026-07-28 the single-round seq gained the matching per-frame 399 PULSE-LENGTH extras
``InitImgExposure`` / ``FinImgExposure`` (seconds, same ``< 0`` = absent sentinel). They exist
because the FINAL image -- whose occupancy IS the fill metric -- is already at MAXIMUM available
399 power (amps 1/1, AOM flat over 0.5-1.0, PID setpoints pinned and held from the root
BlueMOTStep), so TIME is the last brightness lever there; the extra heating is free on the last
image of a shot. The CAMERA window is a separate global setting the runner syncs once per scan
(camera_runtime.sync_camera_exposure), so these move the 399 pulse INSIDE a fixed window.

What is pinned here:
  1. AMP 1.0 == NO OVERRIDE -- setting the extras to 1.0 (the ByPattern/base value for all three
     patterns) is byte-identical to not setting them at all. Combined with (4) below this is what
     makes the default build provably unchanged: with no extras the seq takes the untouched
     ``Imag399Step`` call, and even the amp-step path at the sentinel emits the same bytes.
  2. THE OVERRIDE BITES, PER FRAME -- a mid-only override changes the bytes; a final-only
     override changes the bytes; and mid-only != final-only (they touch different bseqs).
  3. Every cell of the shipped sweep grid is a distinct sequence.
  4. Imag399AmpStep with no override is byte-identical to Imag399Step (the two bodies are copies;
     this is the guard against them drifting apart).

NOT pinned here, on purpose: equality against ``tests/reference_ybseqs/ybseqs_reference.json``.
That MATLAB capture is STALE for the whole Rearrange family -- untouched ``RearrangeCommSeq``
fails it identically (9935 B vs 8028 B, nodes 165 vs 107) because expConfig has moved on since the
capture; see yb_skills/memory/bug-pyctrl-byte-capture-stale-vs-expconfig.md and
test_ybseqs_build.py. Invariance of the default build was instead verified directly, by building
``git show HEAD:YbSeqs/RearrangeCommSeq2.py`` alongside the working tree over the real expConfig
(bare: 9935 B both, identical; with the production 3-pattern ByPattern tags: 10243 B both,
identical) -- 2026-07-27. Same check for the SINGLE-round ``RearrangeCommSeq`` (bare: 8991 B both,
identical; with the production 2-pattern tags: 9279 B both, identical) -- 2026-07-28, re-run with
the SAME numbers after the ExposureTime knob landed.

NO-HARDWARE: real expConfig + tick_per_sec 1e12, engine never loaded.
"""

import os

import pytest

import compare_bytes
import seq_manager
from conftest import _TESTS_DIR
from exp_seq import ExpSeq
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def real_config():
    """Real expConfig + production tick rate; reset both in teardown (process singletons)."""
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    yield
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


def _build(c_ovr=None):
    """Serialize RearrangeCommSeq2 with the given scan-param override (the c_ovr a ScanGroup
    point produces)."""
    from RearrangeCommSeq2 import RearrangeCommSeq2
    return RearrangeCommSeq2(ExpSeq(c_ovr)).serialize()


def _why(got, want):
    """A readable first-difference for an assertion message (raw bytes stay the assertion)."""
    try:
        return compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
    except Exception as e:  # noqa: BLE001 - the message is best-effort
        return "undecodable (%s)" % e


def _extras(**kw):
    """A scan-point c_ovr carrying only ``rearrange_kwargs.extras`` leaves, plus the pattern tags
    the production scan always sets (so ByPattern is exercised, not bypassed)."""
    ex = {"initial_pattern": "tri_3013_camfb",
          "middle_pattern": "kagome_res_2198",
          "final_pattern": "kagome_2078_camfb",
          "n_rounds": 2, "ifEnhanced": True}
    ex.update(kw)
    return {"rearrange_kwargs": {"extras": ex}}


# --- 1/2/3. override semantics ------------------------------------------------------- #

def test_amp_one_is_byte_identical_to_no_override():
    """1.0 is the ByPattern/base value for all three patterns' Imag399.Amp1/Amp2, so the
    baseline cell of the sweep must serialize exactly like production."""
    base = _build(_extras())
    ones = _build(_extras(MidImgAmp1=1.0, MidImgAmp2=1.0, FinImgAmp1=1.0, FinImgAmp2=1.0))
    assert ones == base, (
        "amps 1.0 no longer reproduce ByPattern -- someone changed "
        "ByPattern[kagome_res_2198 | kagome_2078_camfb].Imag399.Amp1/Amp2 away from 1. "
        "SLMRearrangeImagingOptScan's baseline cell is then NOT production; update its grid "
        "(and this test) to the new default. First diff: %s" % _why(ones, base))


def test_mid_override_changes_bytes():
    base = _build(_extras())
    mid = _build(_extras(MidImgAmp1=0.3, MidImgAmp2=0.3))
    assert mid != base


def test_final_override_changes_bytes():
    base = _build(_extras())
    fin = _build(_extras(FinImgAmp1=0.25, FinImgAmp2=0.25))
    assert fin != base


def test_mid_and_final_overrides_are_independent():
    """Same amp value on the middle vs the final frame must produce DIFFERENT bytes -- proof the
    knob lands in one bseq each and not on all three frames."""
    mid = _build(_extras(MidImgAmp1=0.3, MidImgAmp2=0.3))
    fin = _build(_extras(FinImgAmp1=0.3, FinImgAmp2=0.3))
    assert mid != fin


def test_single_amp_override_leaves_the_other_beam_on_bypattern():
    """Setting only Amp1 must still take the override path (a1 >= 0) with Amp2 falling back to
    ByPattern -- i.e. it differs from the baseline but equals an explicit Amp2=1.0."""
    only1 = _build(_extras(MidImgAmp1=0.3))
    both = _build(_extras(MidImgAmp1=0.3, MidImgAmp2=1.0))
    assert only1 == both, _why(only1, both)
    assert only1 != _build(_extras())


def test_each_grid_point_is_distinct():
    """Every cell of the shipped sweep grid must be a physically distinct sequence EXCEPT the
    optically-flat 1.0/0.5 duplicates, which are intentionally different DDS values (so still
    distinct bytes). Guards against a grid edit that accidentally repeats a cell."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(_TESTS_DIR), "YbScans"))
    import SLMRearrangeImagingOptScan as S
    seen = {}
    for mp in S.MID_AMP_PAIRS:
        for fp in S.FIN_AMP_PAIRS:
            b = _build(_extras(MidImgAmp1=mp[0], MidImgAmp2=mp[1],
                               FinImgAmp1=fp[0], FinImgAmp2=fp[1]))
            key = b.hex()
            assert key not in seen, "duplicate grid cell: %s vs %s" % ((mp, fp), seen[key])
            seen[key] = (mp, fp)
    assert len(seen) == len(S.MID_AMP_PAIRS) * len(S.FIN_AMP_PAIRS)


# --- 3b. the SINGLE-round seq (RearrangeCommSeq): InitImgAmp / FinImgAmp ------------- #
# Same mechanism, two frames: img1 = INITIAL (loading, tri_3013_camfb), img2 = FINAL
# (kagome_2078_camfb). ``InitImgAmp*`` is the campaign knob (dim frame 0 -> less 399 heating ->
# more surviving atoms handed to rearrangement); ``FinImgAmp*`` reuses the 2-round name.

def _build1(c_ovr=None):
    from RearrangeCommSeq import RearrangeCommSeq
    return RearrangeCommSeq(ExpSeq(c_ovr)).serialize()


def _extras1(**kw):
    """A single-round scan point: the two pattern tags the production scan always sets."""
    ex = {"initial_pattern": "tri_3013_camfb",
          "final_pattern": "kagome_2078_camfb",
          "n_rounds": 1, "ifEnhanced": True}
    ex.update(kw)
    return {"rearrange_kwargs": {"extras": ex}}


def test_single_round_amp_one_is_byte_identical_to_no_override():
    base = _build1(_extras1())
    ones = _build1(_extras1(InitImgAmp1=1.0, InitImgAmp2=1.0, FinImgAmp1=1.0, FinImgAmp2=1.0))
    assert ones == base, (
        "amps 1.0 no longer reproduce ByPattern for the single-round chain -- someone changed "
        "ByPattern[tri_3013_camfb | kagome_2078_camfb].Imag399.Amp1/Amp2 away from 1. "
        "First diff: %s" % _why(ones, base))


def test_single_round_init_override_changes_bytes():
    """The frame-0 knob the imaging-survival sweep uses (WarmWGSKagomeStepSweep YB_INITAMP*)."""
    assert _build1(_extras1(InitImgAmp1=0.35, InitImgAmp2=0.35)) != _build1(_extras1())


def test_single_round_final_override_changes_bytes():
    assert _build1(_extras1(FinImgAmp1=0.25, FinImgAmp2=0.25)) != _build1(_extras1())


def test_single_round_init_and_final_overrides_are_independent():
    """Same amp on img1 vs img2 must give DIFFERENT bytes -- proof each lands in one bseq."""
    assert (_build1(_extras1(InitImgAmp1=0.3, InitImgAmp2=0.3))
            != _build1(_extras1(FinImgAmp1=0.3, FinImgAmp2=0.3)))


def test_single_round_amp_sweep_cells_are_distinct():
    """Every cell of the WarmWGSKagomeStepSweep frame-0 ladder is a distinct sequence."""
    seen = {}
    for a in (0.5, 0.35, 0.25, 0.175, 0.12):
        b = _build1(_extras1(InitImgAmp1=a, InitImgAmp2=a)).hex()
        assert b not in seen, "duplicate cell: %g vs %g" % (a, seen[b])
        seen[b] = a
    assert len(seen) == 5


# --- 3c. the SINGLE-round 399 PULSE-LENGTH knob: InitImgExposure / FinImgExposure ----- #
# The campaign knob for the FINAL image (its amps are railed at 1/1 and the PID setpoints are
# pinned, so TIME is all that is left). Both production patterns resolve
# Imag399.ExposureTime -> ByPattern[<pattern>].Orca.ExposureTime = 0.1 s via
# expConfig_helper.apply_cross_refs, which is why 0.1 is the no-op value below.

def test_single_round_exposure_default_value_is_byte_identical_to_no_override():
    """0.1 s is the resolved ExposureTime for BOTH production patterns (cross-ref to their
    ByPattern Orca.ExposureTime), so the baseline cell of an exposure sweep must serialize
    exactly like production. Fails loudly if either pattern's Orca.ExposureTime moves."""
    base = _build1(_extras1())
    same = _build1(_extras1(InitImgExposure=0.1, FinImgExposure=0.1))
    assert same == base, (
        "0.1 s no longer reproduces the resolved Imag399.ExposureTime -- someone changed "
        "ByPattern[tri_3013_camfb | kagome_2078_camfb].Orca.ExposureTime away from 0.1. "
        "First diff: %s" % _why(same, base))


def test_single_round_exposure_override_changes_bytes():
    """A longer 399 pulse on ONE frame must move the bytes (both directions of the knob)."""
    base = _build1(_extras1())
    assert _build1(_extras1(FinImgExposure=0.2)) != base
    assert _build1(_extras1(InitImgExposure=0.2)) != base


def test_single_round_exposure_overrides_are_independent():
    """Same pulse length on img1 vs img2 must give DIFFERENT bytes -- proof each lands in one
    bseq (img1 in the root bseq, img2 in the rearrange bseq)."""
    assert _build1(_extras1(InitImgExposure=0.2)) != _build1(_extras1(FinImgExposure=0.2))


def test_single_round_exposure_composes_with_amps():
    """Exposure and amps are independent knobs on the same frame: setting both must differ from
    setting either alone (guards the shared Imag399AmpStep call from dropping an argument)."""
    both = _build1(_extras1(FinImgExposure=0.2, FinImgAmp1=0.5, FinImgAmp2=0.5))
    assert both != _build1(_extras1(FinImgExposure=0.2))
    assert both != _build1(_extras1(FinImgAmp1=0.5, FinImgAmp2=0.5))


def test_single_round_exposure_sweep_cells_are_distinct():
    """Every cell of the WarmWGSKagomeStepSweep YB_FINEXP_SWEEP ladder is a distinct sequence."""
    seen = {}
    for t in (0.1, 0.15, 0.2, 0.3):
        b = _build1(_extras1(FinImgExposure=t)).hex()
        assert b not in seen, "duplicate cell: %g vs %g" % (t, seen[b])
        seen[b] = t
    assert len(seen) == 4


# --- 4. the copied step body has not drifted ----------------------------------------- #

def _one_step(step, *extra):
    s = ExpSeq()
    s.add_step(step, s.C.Imag399, *extra)
    return s.serialize()


def test_imag399_amp_step_default_equals_imag399_step():
    """Imag399AmpStep(a1=-1, a2=-1, t_exp=-1) must be byte-identical to Imag399Step -- the guard
    against the two (deliberately copied) bodies diverging. Both the 2-arg call (what an
    amps-only build emits) and the explicit 3-arg sentinel are pinned."""
    from Imag399AmpStep import Imag399AmpStep
    from Imag399Step import Imag399Step

    ref = _one_step(Imag399Step)
    for extra in ((-1.0, -1.0), (-1.0, -1.0, -1.0)):
        got = _one_step(Imag399AmpStep, *extra)
        assert got == ref, "%r: %s" % (extra, _why(got, ref))


def test_imag399_amp_step_exposure_arg_is_additive():
    """The t_exp argument must not disturb the amps-only path: an explicit -1 sentinel equals
    omitting it, and a real value bites."""
    from Imag399AmpStep import Imag399AmpStep
    from Imag399Step import Imag399Step

    amps_only = _one_step(Imag399AmpStep, 0.35, 0.35)
    assert _one_step(Imag399AmpStep, 0.35, 0.35, -1.0) == amps_only, _why(
        _one_step(Imag399AmpStep, 0.35, 0.35, -1.0), amps_only)
    assert _one_step(Imag399AmpStep, -1.0, -1.0, 0.2) != _one_step(Imag399Step)
