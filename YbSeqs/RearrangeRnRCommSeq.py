"""RearrangeRnRCommSeq.py -- single-round SLM rearrangement + a release-and-recapture test.

This is ``RearrangeCommSeq`` (the single-round img1 -> rearrange -> img2 handoff) with ONE
insertion: a :func:`RearrangeRnRStep` dropped into the second basic sequence, AFTER the
post-rearrange cooling and BEFORE the final image. So the per-shot flow is:

    img1 (Imag399 #1) -> rearrange(probs) [writes the target pattern on the SLM]
                      -> Cool556 -> RELEASE-AND-RECAPTURE -> img2 (Imag399 #2)

The headline observable is survival across the release-and-recapture, measured on the freshly
REARRANGED array. ``NumImages == 2`` (img1 = load, img2 = post-R&R) exactly as the parent seq.

Baseline via zero release time: :func:`RearrangeRnRStep` SKIPS itself entirely when
``ReleaseRecapture.Time == 0`` (adds no bytes -- no trap drop, no scope trig, no AOM settle), so
a ``ReleaseRecapture.Time`` sweep that includes 0 gets a true "rearrange, then image, trap never
perturbed" control point for free. See YbScans/RearrangeRnRScan.py + YbSteps/RearrangeRnRStep.py.

Everything else is IDENTICAL to RearrangeCommSeq and DELIBERATELY so -- same
``@seq_capabilities(owns_frames=True)``, same per-bseq pattern tagging (initial_pattern on bseq1,
final_pattern on bseq2), and the SAME shared ``rearrange_callbacks`` machinery
(``pre_run`` / ``hand_over_slm`` / ``post_run``). The R&R step adds NO camera frame, so the final
image is still frame index 1 and ``finalize(..., final_frame_idx=1)`` is unchanged. ``serialize()``
never runs the deferred callbacks, so the BUILD path stays a faithful transliteration.

This is a NEW seq (RearrangeCommSeq is left untouched): the only structural change is the R&R
insertion in bseq2 + swapping ReleaseRecaptureStep's recapture window in between the rearrange and
the second image.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from RearrangeRnRStep import RearrangeRnRStep
from SLMStep import SLMStep

import rearrange_callbacks
from seq_capability import seq_capabilities


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoff)
def RearrangeRnRCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 images the INITIAL (dense load)
    # pattern, bseq2 (below) the FINAL (rearranged target); each bseq's cooling/imaging/VSLMServo
    # resolve from ByPattern[that pattern]. Names from rearrange_kwargs.extras.initial_pattern /
    # final_pattern (set by the scan); absent -> scan-default / inherit. Tag s HERE, before its
    # steps build, so they pick it up. No-op when ByPattern is empty.
    _init_pat = s.C.rearrange_kwargs.extras.initial_pattern("")
    if _init_pat:
        s.set_pattern(_init_pat)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    # LAC vs BlueLAC chosen at build time; default (no rearrange_kwargs) -> LAC.
    ifEnhanced = s.C.rearrange_kwargs.extras.ifEnhanced(False)
    if ifEnhanced:
        s.add_step(BlueLACStep, s.C.LAC)
    else:
        s.add_step(LACStep, s.C.LAC)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # Second part: SLM-rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    # bseq2 -> the FINAL (rearranged target) pattern (see the initial_pattern note above).
    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> bits -> rearrange

    s2.add_step(SLMStep, s.C.SLM)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # Release-and-recapture on the REARRANGED array (this is the test). Skips itself entirely
    # when ReleaseRecapture.Time == 0 -> that scan point is the "held in traps" baseline (no trap
    # perturbation between rearrange and the second image). Sits AFTER the post-rearrange cooling
    # and BEFORE the second image; adds no camera frame.
    s2.add_step(RearrangeRnRStep, s.C.ReleaseRecapture)

    # Second Imag399.
    s2.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img2 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# Thin wrappers over the shared rearrange_callbacks machinery -- identical to RearrangeCommSeq
# (the R&R insertion adds no frame, so the frame indices are unchanged). These names stay public
# (the seq registration above + the tests import them from THIS module).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement (sticky -- no reset_params), and reload_rearrange."""
    rearrange_callbacks.pre_run(s1, flags=("rearrange_img1_ok",),
                                lock_desc="rearrange compute")


def hand_over_slm(s1):
    """Read img1 (Imag399 #1), detect probs, rearrange(probs), and store img1. Records shot
    health here (record_ok) -- the single-round seq's live health signal is the rearrange
    result, not the final publish."""
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", record_ok=True)


def post_run(s1):
    """Read img2 (Imag399 #2), update_rearrange(bits2), store img2 + finish; release the compute
    lock and keepalive the scan-long slm lock. img2 detects with the frame-0 detector
    (server-grid-anchored) -- the single-round arrays share one site order."""
    rearrange_callbacks.finalize(
        s1, round_flags=("rearrange_img1_ok",), final_frame_idx=1, tag="post_run")
