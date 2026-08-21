"""RearrangeSTIRAPSeq.py -- hybrid seq: SLM rearrangement, then STIRAP push-out survival.

The "rearrange, then science" composition: the RearrangeCommSeq loading + handoff prologue
(img1 on the LOADING pattern -> rearrange to the TARGET pattern) followed by the
PushoutSurvivalAWGSeq science block (Cool556 -> STIRAPPushoutStep via the Siglent AWGs ->
final Imag399). Survival is measured on a rearranged (defect-filled) array instead of the
Poisson-loaded one.

Two frame layouts, chosen at BUILD time by ``rearrange_kwargs.extras.verifyImage`` (a per-scan
constant -- NOT sweepable, it changes the bseq structure + NumImages):

  verifyImage = True  (default; scan sets NumImages = 3):
      bseq1: load + Imag399 #1 (LOADING pattern)
        -- hand_over_slm: img1 -> rearrange --
      bseq2: SLM + Cool556hX + Imag399 #2 (TARGET pattern)      <- VERIFY frame
        -- verify_and_report: img2 -> update_rearrange --
      bseq3: Cool556 + STIRAPPushout + Imag399 #3 + Init         <- survival frame
    img1->img2 gives the rearrangement fidelity, img2->img3 the clean science survival
    (normalized against the VERIFIED occupancy).

  verifyImage = False (scan sets NumImages = 2):
      bseq1: load + Imag399 #1
        -- hand_over_slm: img1 -> rearrange --
      bseq2: SLM + Cool556hX + Cool556 + STIRAPPushout + Imag399 #2 + Init
    One cool+image block shorter, but img1->img2 conflates a failed move with a push-out
    (empty site is ambiguous), and the server gets NO update_rearrange (see below).

``update_rearrange`` policy: the final frame is POST-PUSHOUT -- its occupancy reflects the
science, not the rearrangement -- so it must never feed the SLM server's rearrange statistics
(finalize(update=False) in both layouts). With verifyImage on, the server gets its result
frame from img2 via :func:`rearrange_callbacks.verify_frame` instead.

Like PushoutSurvivalAWGSeq: opens with the 616-EOM slow ramp driven by a sequence global
(``register_eom616_persistence`` injects the last run's frequency -> ~20 ms ramp), and the
Siglent AWG recall is RUNNER-SIDE (the scan lists ``runp().AWGs``; ``STIRAPPushoutStep``
drives the FPGA gate TTLs). Like RearrangeCommSeq: ``@seq_capabilities(owns_frames=True)`` --
the callbacks grab + stage EVERY camera frame; the runner adds no capture post_cb.

BYTE-CRITICAL: the literal ``3.0`` in the EOM ramp time is a SeqVal operand; a bare ``3``
would serialize as INT32 (see PushoutSurvivalAWGSeq).
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from ramp_to import ramp_to
from RearrangeCool556hXStep import RearrangeCool556hXStep
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep
from STIRAPPushoutStep import STIRAPPushoutStep
from STIRAPHighFieldPushoutStep import STIRAPHighFieldPushoutStep
from ReleaseRecaptureStep import ReleaseRecaptureStep

import rearrange_callbacks
from seq_capability import seq_capabilities


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence
def RearrangeSTIRAPSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    # rearrange_img2_ok is only meaningful with verifyImage on; harmless otherwise.
    s.G.rearrange_img1_ok = False
    s.G.rearrange_img2_ok = False
    s.G.rearrange_lock_ok = False

    # Build-time frame-layout switch (per-scan constant; the scan derives NumImages +
    # imagePatternsJson from the same flag).
    verify = bool(s.C.rearrange_kwargs.extras.verifyImage(True))

    # Initialising 616EOM to its old value from last run (via a sequence global).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    # Slow EOM ramp. 3.0 (not 3): SeqVal operand -> must be FLOAT64.
    time = abs((Freq_EOM616 - freq616global) * 2e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))
    register_eom616_persistence(s, freq616global, Freq_EOM616)

    s.reg_before_start(pre_run)        # slm lock, compute lock, per-shot setup + reload

    # ---- bseq1: load + img1 (LOADING pattern) -- the RearrangeCommSeq prologue ---------- #
    _init_pat = s.C.rearrange_kwargs.extras.initial_pattern("")
    if _init_pat:
        s.set_pattern(_init_pat)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    ifEnhanced = s.C.rearrange_kwargs.extras.ifEnhanced(False)
    if ifEnhanced:
        s.add_step(BlueLACStep, s.C.LAC)
    else:
        s.add_step(LACStep, s.C.LAC)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # First Imag399 (img1, LOADING pattern).
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # ---- bseq2: post-rearrangement (TARGET pattern) ------------------------------------- #
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> probs -> rearrange (loading -> target)

    s2.add_step(SLMStep, s.C.SLM)
    s2.add_step(Cool556hXStep, s.C.Cool556)

    if verify:
        # NI-DAQ minimum-buffer guard: DAQmx FINITE AO rejects a 1-sample buffer
        # (SampQuant_SampPerChan min 2, error -200077). Without this, the verify bseq's only
        # NI update time is SLMStep at t0 (Cool556hX/Imag399 touch only TTL+DDS) -> 1 sample.
        # Re-asserting one V* channel at its current value AFTER the cooling step adds a second
        # update time -> >= 2 samples. Physical no-op (same trick as RearrangeCommSeq2's
        # keep-alive, at a second time point).
        s2.add('VMOTCoil', 0)
        # Second Imag399 (img2, TARGET pattern -- the VERIFY frame).
        s2.add_step(Imag399Step, s.C.Imag399)

        # ---- bseq3: science on the verified array (TARGET pattern) --------------------- #
        s3 = s.new_basic_seq()
        s2.cond_branch(True, s3)
        if _final_pat:
            s3.set_pattern(_final_pat)

        s3.reg_before_bseq(verify_and_report)   # img2 -> bits -> update_rearrange

        # NI-DAQ keep-alive: reassert one V* channel so libnacs emits non-None NI data even
        # if the science steps' NI activity were trimmed. NO InitStep between bseqs (that
        # zeroes VSLMservo and loses the atoms). Physical no-op.
        s3.add('VMOTCoil', 0)
        science = s3
    else:
        science = s2

    # ---- science block (mirrors PushoutSurvivalAWGSeq: Cool556 -> STIRAP -> Imag399) ---- #
    # 2026-08-06: the recool now has its OWN config block (RearrangeCool556) so it can be optimized
    science.add_step(Cool556hXStep, s.C.Cool556) #add_step(RearrangeCool556hXStep, s.C.RearrangeCool556)
    # 2026-08-06 A/B: RNR in place of the STIRAP push-out, to test whether the RNR-level survival
    # science_step = str(s.C.rearrange_kwargs.extras.scienceStep("rnr")).lower()
    # default to be stirap
    Bfield = s.C.Pushout.BiasCoilCurrent.Ryd(Consts().Pushout.BiasCoilCurrent.Ryd)
    if Bfield < 31:
        science.add_step(STIRAPPushoutStep, s.C.Pushout) # Low field STIRAP pushout
    elif 50 <= Bfield <= 80:
        science.add_step(STIRAPHighFieldPushoutStep, s.C.Pushout) # High field STIRAP pushout
    else:
        raise ValueError(f'RearrangeSTIRAPSeq: BiasCoilCurrent.Ryd={Bfield} G is not in the valid range for STIRAP push-out.')
    
    # if science_step == "rnr":
    #     science.add_step(ReleaseRecaptureStep, s.C.ReleaseRecapture)
        
    science.add_step(Cool556hXStep, s.C.Cool556)
    # Final Imag399 (the survival frame: img3 with verify, img2 without).
    science.add_step(Imag399Step, s.C.Imag399)

    science.wait(0.1)
    science.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # final frame -> stage + finish; release compute; keepalive
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# Thin wrappers over the shared rearrange_callbacks machinery.
# =========================================================================== #
def _verify_on(s1):
    """The BUILD-time verifyImage flag, re-read at run time (same source: s1.C)."""
    try:
        return bool(s1.C.rearrange_kwargs.extras.verifyImage(True))
    except Exception:  # noqa: BLE001 - absent config -> the default layout
        return True


def pre_run(s1):
    """Slm lock (ensure_held), per-shot compute lock, sticky setup_rearrangement, reload."""
    flags = (("rearrange_img1_ok", "rearrange_img2_ok") if _verify_on(s1)
             else ("rearrange_img1_ok",))
    rearrange_callbacks.pre_run(s1, flags=flags, lock_desc="rearrange compute (STIRAP)")


def hand_over_slm(s1):
    """Read img1 (LOADING pattern), detect probs, rearrange(probs), stage img1."""
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", use_frame_pattern=True)


def verify_and_report(s1):
    """verifyImage layout only: read img2 (TARGET pattern), update_rearrange(bits2), stage
    img2. The server's rearrange statistics get their result frame HERE -- the final frame is
    post-pushout and never feeds them."""
    rearrange_callbacks.verify_frame(
        s1, 1, ok_flag="rearrange_img2_ok", prior_flags=("rearrange_img1_ok",),
        tag="verify_and_report")


def post_run(s1):
    """Read the FINAL (post-pushout survival) frame, stage + finish the shot; release the
    compute lock and keepalive the scan-long slm lock. update=False ALWAYS: the final frame's
    occupancy reflects the push-out, not the rearrangement."""
    if _verify_on(s1):
        rearrange_callbacks.finalize(
            s1, round_flags=("rearrange_img1_ok", "rearrange_img2_ok"), final_frame_idx=2,
            tag="post_run", update=False, record_ok=True)
    else:
        rearrange_callbacks.finalize(
            s1, round_flags=("rearrange_img1_ok",), final_frame_idx=1,
            tag="post_run", update=False, record_ok=True)
