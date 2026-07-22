"""RearrangeCommSeq2Dev.py -- DEV sandbox copy of RearrangeCommSeq2 (two-round rearrangement).

Full copy of the base seq body (RearrangeCommSeq2) with two DEV additions, both OPT-IN so the
default build is physics-identical to the base:

1. Min-load shot gates (per-shot abort via the normal failing-shot path):
   * ``extras.MinLoadAtoms`` (default 0 = off): fewer LOADING-frame atoms than this -> skip the
     shot (frame drained + cancel_shot, no SLM playback); the sequence still runs so the array /
     thermalization stays in steady state.
   * ``extras.MinMidAtoms`` (default 0 = off): same gate on the MIDDLE frame before round 2.

2. Per-bseq 399 imaging-PID RELOCK (``extras.RelockPIDs`` = 1, default 0 = off): the PID lock is
   engaged ONCE in the root BlueMOTStep and then held, so in the 2-round context the MIDDLE and
   FINAL images run at the LOADING pattern's held power -- the per-pattern
   ``BlueMOT.Img1/Img2PIDSet`` never applied (discovered 2026-07-19). With RelockPIDs on, each
   rearrange bseq re-engages the lock (mirrors BlueMOTStep's engage block: amps 1 + shutters open
   + PIDMode 1 + setpoints, settle ``extras.RelockTime`` (default 15 ms), then hold + all off)
   BEFORE its Cool556 + image, at setpoints resolved per-bseq:
     middle:  ``extras.MidImg1PIDSet`` / ``extras.MidImg2PIDSet``  (sweepable; default -1 ->
              the bseq's own ``s.C.BlueMOT.Img1/Img2PIDSet``, i.e. ByPattern[middle_pattern])
     final:   ``extras.FinImg1PIDSet`` / ``extras.FinImg2PIDSet``  (same, ByPattern[final])
   Cost: atoms see ~RelockTime of 399 light at the new setpoint before the recool + image.

Production scans keep using RearrangeCommSeq2 (untouched); only SLMRearrangementScanDev submits
THIS seq. 2026-07-19 kagome >=97.5% campaign.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep

import rearrange_callbacks
from seq_capability import seq_capabilities


def _extras_num(s, name, default):
    try:
        return float(getattr(s.C.rearrange_kwargs.extras, name)(default))
    except Exception:  # noqa: BLE001 - absent/odd extras -> default
        return float(default)


def Relock399PIDStep(s, g, ovr1, ovr2, t_relock):
    """Re-engage the 399 Img1/Img2 imaging-power PID at new setpoints, then hold.

    A CUSTOM STEP (so the per-bseq ByPattern overlay applies to the ``g`` reads -- bare adds
    between steps would resolve the wrong pattern). ``g = s.C.BlueMOT``; the setpoints are the
    sweepable extras overrides ``ovr1/ovr2`` when >= 0, else this bseq's per-pattern
    ``BlueMOT.Img1/Img2PIDSet``. Mirrors BlueMOTStep's engage block (amps 1 + shutters open +
    PIDMode 1 + setpoints), holds ``t_relock`` for the servo to settle -- the atoms see that
    much 399 light at the NEW setpoint -- then freezes the lock (PIDMode 0) and shuts light +
    shutters. The following Imag399Step images at the newly held power."""
    set1 = ovr1 if ovr1 >= 0 else g.Img1PIDSet(Consts().BlueMOT.Img1PIDSet)
    set2 = ovr2 if ovr2 >= 0 else g.Img2PIDSet(Consts().BlueMOT.Img2PIDSet)
    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = Consts().Imag399.FreqDetuning
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    # Light + shutters FIRST, engage the lock only once the PD actually sees light. Engaging
    # against a dark PD (shutter still opening) rails the integrator -> light burst when the
    # shutter opens (run 20260719_170x: mid counts collapsed 2100 -> ~900). BlueMOTStep gets
    # away with the simultaneous engage because it settles over ~1 s of MOT loading.
    (s.add('FreqAbsImag', Freq_Imag399)
        .add('Freq399Imag2', Freq_Imag399)
        .add('AmpAbsImag', 1)
        .add('Amp399Imag2', 1)
        .add('TTL399AbsImagShutter', 1)
        .add('TTL399Imag2Shutter', 1)
        .add('VImg1PIDSet', set1)
        .add('VImg2PIDSet', set2))
    s.wait(4e-3)             # shutter fully open (PD lit at the held power)
    s.add('TTL399IMG1PIDMode', 1)    # engage on a lit PD
    s.add('TTL399IMG2PIDMode', 1)
    s.wait(t_relock)         # servo settle at the new setpoint (399 light on the atoms)
    s.add('TTL399IMG1PIDMode', 0)   # freeze -> hold
    s.add('TTL399IMG2PIDMode', 0)
    (s.add('AmpAbsImag', 0)
        .add('Amp399Imag2', 0)
        .add('TTL399AbsImagShutter', 0)
        .add('TTL399Imag2Shutter', 0))


def Imag399DevStep(s, g, a1, a2):
    """Imag399Step with the 399 DDS amplitudes overridable per-round (dose control with the PID
    HELD): the PID output stays at the root-locked power; ``AmpAbsImag``/``Amp399Imag2`` (default
    1.0) attenuate below the AOM on top of it. ``a1/a2`` >= 0 override ``g.Amp1/Amp2`` for THIS
    image only; both DDS amps are set back inside the step's turn-off (identical to the base
    step), so nothing leaks to the next image. NOTE: DDS amp -> optical power is NONLINEAR (AOM);
    sweep empirically, don't compute the ratio. Everything else byte-mirrors Imag399Step."""
    t_Imag399 = g.ExposureTime(Consts().Imag399.ExposureTime)

    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = g.FreqDetuning(Consts().Imag399.FreqDetuning)
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    Amp_Imag399_1 = a1 if a1 >= 0 else g.Amp1(Consts().Imag399.Amp1)
    Amp_Imag399_2 = a2 if a2 >= 0 else g.Amp2(Consts().Imag399.Amp2)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq

    Freq_Cool556DetuningX = g.Cool556.X.FreqDetuning(Consts().Imag399.Cool556.X.FreqDetuning)
    Freq_Cool556Detuningh = g.Cool556.h.FreqDetuning(Consts().Imag399.Cool556.h.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Freq_Cool556h = Freq_Resonance556mj0Freq + Freq_Cool556Detuningh

    Amp_Cool556X = g.Cool556.X.Amp(Consts().Imag399.Cool556.X.Amp)
    Amp_Cool556h = g.Cool556.h.Amp(Consts().Imag399.Cool556.h.Amp)

    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    s.add('TTL556RydbergShutter', 0)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', Amp_Cool556h)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', Amp_Imag399_1)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', Amp_Imag399_2)
    s.add_step(100e-6).add('TTLOrcaTrig', 1)

    s.add('TTLOrcaTrig', 0)

    s.wait(t_Imag399)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', 0)

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', 0)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', 0)

    s.add('TTL399AbsImagShutter', 0)
    s.add('TTL399Imag2Shutter', 0)
    s.wait(3e-3)  # wait for the shutter


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def RearrangeCommSeq2Dev(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_img2_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload, n_rounds=2

    relock = bool(_extras_num(s, "RelockPIDs", 0))
    t_relock = _extras_num(s, "RelockTime", 15e-3)

    # Per-bseq SLM pattern (expConfig ByPattern overlay) -- same as the base seq.
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

    # Round 1: SLM rearrangement basic sequence (always entered). Imaged at the MIDDLE pattern.
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _mid_pat = s.C.rearrange_kwargs.extras.middle_pattern("")
    if _mid_pat:
        s2.set_pattern(_mid_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> rearrange round 1 (loading -> middle)

    # NI-DAQ keep-alive: reassert one V* channel (VMOTCoil at its current 0) so libnacs emits
    # non-None NI data for this bseq. A physical no-op. (See the base seq's note.)
    s2.add('VMOTCoil', 0)

    if relock:
        s2.add_step(Relock399PIDStep, s.C.BlueMOT,
                    _extras_num(s, "MidImg1PIDSet", -1.0),
                    _extras_num(s, "MidImg2PIDSet", -1.0), t_relock)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # NI-DAQ minimum-buffer guard (>= 2 update times; see the base seq's note).
    s2.add('VMOTCoil', 0)

    # Second Imag399 (img2, MIDDLE pattern). Dose overridable via extras.MidImgAmp1/2 (DDS
    # attenuation on top of the HELD PID power; -1 = per-pattern/base default = 1.0).
    s2.add_step(Imag399DevStep, s.C.Imag399,
                _extras_num(s, "MidImgAmp1", -1.0), _extras_num(s, "MidImgAmp2", -1.0))

    # Round 2: second SLM-rearrangement basic sequence. Imaged at the FINAL pattern.
    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s3.set_pattern(_final_pat)

    s3.reg_before_bseq(hand_over_slm_2)   # img2 -> rearrange round 2 (middle -> final)

    # NI-DAQ keep-alive (same reason as s2). ASSUME-WRITTEN: no SLMStep / no phase write.
    s3.add('VMOTCoil', 0)

    if relock:
        s3.add_step(Relock399PIDStep, s.C.BlueMOT,
                    _extras_num(s, "FinImg1PIDSet", -1.0),
                    _extras_num(s, "FinImg2PIDSet", -1.0), t_relock)

    s3.add_step(Cool556hXStep, s.C.Cool556)

    # NI-DAQ minimum-buffer guard (same as s2).
    s3.add('VMOTCoil', 0)

    # Third Imag399 (img3, FINAL pattern). Dose overridable via extras.FinImgAmp1/2.
    s3.add_step(Imag399DevStep, s.C.Imag399,
                _extras_num(s, "FinImgAmp1", -1.0), _extras_num(s, "FinImgAmp2", -1.0))

    # Initialisation again (shut down for safety).
    s3.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img3 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks -- gated wrappers over the shared machinery.
# =========================================================================== #
def _extras_int(s1, name):
    """extras.<name> as int with 0 (= gate off) on absence/any resolution error."""
    try:
        return int(getattr(s1.C.rearrange_kwargs.extras, name)(0))
    except Exception:  # noqa: BLE001 - absent/odd extras -> gate off
        return 0


def pre_run(s1):
    rearrange_callbacks.pre_run(
        s1, flags=("rearrange_img1_ok", "rearrange_img2_ok"),
        lock_desc="rearrange compute (2 rounds)", force_n_rounds=2)


def hand_over_slm(s1):
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", use_frame_pattern=True,
        min_load=_extras_int(s1, "MinLoadAtoms"))


def hand_over_slm_2(s1):
    rearrange_callbacks.rearrange_round(
        s1, 1, ok_flag="rearrange_img2_ok", prior_flag="rearrange_img1_ok",
        tag="hand_over_slm_2", use_frame_pattern=True,
        min_load=_extras_int(s1, "MinMidAtoms"))


def post_run(s1):
    rearrange_callbacks.finalize(
        s1, round_flags=("rearrange_img1_ok", "rearrange_img2_ok"), final_frame_idx=2,
        tag="post_run", use_frame_pattern=True, record_ok=True)
