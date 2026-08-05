"""RearrangeRnRHeatCommSeq.py -- RearrangeRnRCommSeq with an INDEPENDENT post-motion cooling
group, for rearrangement-heating measurements (release-and-recapture thermometry).

Identical to :mod:`RearrangeRnRCommSeq` (img1 -> rearrange -> cool -> R&R -> img2) with ONE
change: the bseq2 cooling step -- the one between the rearrangement motion and the release
window -- reads its OWN config group ``s.C.PostRearrCool`` instead of the shared
``s.C.Cool556``. Cool556hXStep's fallbacks still point at ``Consts().Cool556`` (pattern-
overlaid), so an UNSET ``PostRearrCool`` is byte-identical to the parent seq; a scan can turn
the post-motion cooling off (``g().PostRearrCool.X.Amp = 0`` + ``h.Amp = 0``, and/or a short
``PostRearrCool.Time``) WITHOUT touching bseq1's pre-image / pre-motion cooling.

Why: release-and-recapture measures the temperature at release. For a heating measurement the
right structure is  cool (defined cold state, same as the baseline RNR scan) -> MOTION ->
release IMMEDIATELY.  The parent seq's shared Cool556 group cannot express "cool before the
motion but not after it"; this variant can:

    bseq1: ... img1 -> Cool556 (s.C.Cool556, kept = baseline RNR's prepared state)
    bseq2: [rearrange motion] -> PostRearrCool (amps 0 => short dark hold) -> R&R -> img2

Everything else -- the t=0 R&R skip baseline, per-bseq pattern tagging, the shared
rearrange_callbacks machinery, NumImages == 2 -- is unchanged from RearrangeRnRCommSeq.
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
def RearrangeRnRHeatCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload

    # Per-bseq SLM pattern (expConfig ByPattern overlay) -- see RearrangeRnRCommSeq.
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

    # Pre-motion cooling. Reads its OWN group ``s.C.PreMotionCool``; UNSET it falls back to
    # Consts().Cool556 (pattern-overlaid) exactly like s.C.Cool556 would, so the default build is
    # byte-identical to before -- the atoms still enter the motion in the same prepared cold state
    # the baseline RNR scan measures.
    #
    # Why it needs its own group (2026-07-31): the release-recapture fit brackets the temperature
    # between "the motion loss is energy-BLIND" (survivors still thermal) and "the loss is pure
    # EVAPORATION" (survivors truncated, parent much hotter) -- a factor ~2 at the above-cliff
    # points. The two are told apart by how the motion loss responds to the INITIAL temperature:
    # evaporation over a barrier is steeply sensitive to it, an impulsive mechanical ejection is
    # nearly blind to it. That needs the pre-motion cooling spoiled on purpose -- and bseq1 uses
    # s.C.Cool556 TWICE (once before img1, once here), so touching the shared group would also
    # degrade img1 detection and confound the very survival being measured.
    s.add_step(Cool556hXStep, s.C.PreMotionCool)

    # Leave the cooling light on a little during rearrangement (default OFF: RearrCoolAmp 0).
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # Second part: SLM-rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    # bseq2 -> the FINAL (rearranged target) pattern.
    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> bits -> rearrange (the MOTION happens here)

    s2.add_step(SLMStep, s.C.SLM)

    # POST-MOTION cooling -- the ONE deliberate difference from RearrangeRnRCommSeq: its own
    # config group. Unset -> falls back to Consts().Cool556 (pattern-overlaid) = byte-identical
    # to the parent seq. For heating measurements set g().PostRearrCool.X.Amp = 0 + h.Amp = 0
    # (+ a short PostRearrCool.Time) so the release probes the post-motion temperature directly.
    s2.add_step(Cool556hXStep, s.C.PostRearrCool)

    # Release-and-recapture on the REARRANGED array (skips itself at Time == 0 -> the "held in
    # traps" baseline). Sits right after the (optionally disabled) post-motion cooling.
    s2.add_step(RearrangeRnRStep, s.C.ReleaseRecapture)

    # Second Imag399.
    s2.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img2 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks -- identical to RearrangeRnRCommSeq (thin wrappers over the
# shared rearrange_callbacks machinery; the R&R + cooling changes add no camera frame).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement (sticky -- no reset_params), and reload_rearrange."""
    rearrange_callbacks.pre_run(s1, flags=("rearrange_img1_ok",),
                                lock_desc="rearrange compute")


def hand_over_slm(s1):
    """Read img1 (Imag399 #1), detect probs, rearrange(probs), and store img1."""
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", record_ok=True)


def post_run(s1):
    """Read img2 (Imag399 #2), update_rearrange(bits2), store img2 + finish; release the compute
    lock and keepalive the scan-long slm lock."""
    rearrange_callbacks.finalize(
        s1, round_flags=("rearrange_img1_ok",), final_frame_idx=1, tag="post_run")
