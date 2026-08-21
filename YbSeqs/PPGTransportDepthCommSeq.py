"""PPGTransportDepthCommSeq.py -- RearrangeCommSeq that RAISES THE TRAP DEPTH during transport.

THE IDEA (user, 2026-08-10).  Every cooling attempt so far (jobs 641/646/649/652) failed to add
energy-removal during the move.  This attacks the same loss from the other side: instead of
cooling the atom, make the trap it is riding in DEEPER, so the same per-step displacement is a
smaller fraction of the trap and the atom is further from the escape threshold.  The transit
traps are the shallow ones (the blaze-grating transit frames are far less efficient than the WGS
endpoint), which is exactly the regime this addresses.

SHAPE OF THE SHOT (differences from ``RearrangeCommSeq`` marked ``<<``)::

    bseq1:  ... Imag399 (img1) -> Cool556
            << ramp VSLMservo  Init.VSLMServo -> extras.TransportVServo   over RampMs (smootherstep)
            -- handoff: the FPGA HOLDS that voltage, so the whole rearrange window
               (detect + compute + the SLM playback that IS the transport) runs DEEP --
    bseq2:  << wait extras.DepthHoldMs at the deep setting  (let the array settle before touching it)
            << ramp VSLMservo  back to Init.VSLMServo       over RampMs
            SLMStep -> Cool556 -> Imag399 (img2) -> InitStep

ORDERING IS LOAD-BEARING.  ``SLMStep`` writes ``VSLMservo = SLM.VServo`` (cross-referenced to
``Init.VSLMServo``) at t=0 of its step, so the stock bseq2 would SNAP the depth back to normal
before anything else -- the hold and the ramp-down must come BEFORE ``SLMStep``, which is why
bseq2 is reordered here rather than just having a ramp appended.  ``AmpSLM`` is untouched by this
seq: bseq1's ``SLMStep`` already turned the AOM on and the FPGA holds it across the handoff, so
only the servo setpoint moves.

RAMPS ARE ``ramp_to`` (quintic smootherstep, C2 at both ends -- zero slope AND zero acceleration
at the joins), over ``extras.DepthRampMs`` = 5 ms by default.  The radial trap period is ~12 us
(85.7 kHz), so 5 ms is ~400 trap periods and deeply adiabatic; the knob exists so that can be
checked rather than assumed.

CONTROL.  ``extras.TransportVServo`` defaults to ``Init.VSLMServo``, i.e. the ramps and the hold
still happen but they move the depth NOWHERE.  That cell is the matched-timing control: it pays
the same ~20 ms of extra sequence and the same two ramps as every other cell, so a difference
between it and a deeper cell is depth, not timing.  Absent the extra entirely, the emitted bytes
still differ from ``RearrangeCommSeq`` (the ramps and the hold are real pulses), so this seq is
NOT byte-identical to its parent by design -- unlike ``PPGTransportCoolCommSeq``.

Used by ``YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode depth``.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399AmpStep import Imag399AmpStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from ramp_to import ramp_to
from SLMStep import SLMStep

import rearrange_callbacks
from seq_capability import seq_capabilities


# --------------------------------------------------------------------------- #
# Per-frame 399 imaging brightness -- verbatim copies of RearrangeCommSeq's helpers.
# --------------------------------------------------------------------------- #
def _extras_num(s, name, default):
    """``rearrange_kwargs.extras.<name>`` as a float; ``default`` when absent/unresolvable."""
    try:
        return float(getattr(s.C.rearrange_kwargs.extras, name)(default))
    except Exception:  # noqa: BLE001 - absent/odd extras -> default
        return float(default)


def _img_amps(s, n1, n2):
    """The (Amp1, Amp2) override for ONE image, or ``None`` when the scan set NEITHER extra."""
    a1 = _extras_num(s, n1, -1.0)
    a2 = _extras_num(s, n2, -1.0)
    return None if (a1 < 0 and a2 < 0) else (a1, a2)


def _img_exposure(s, name):
    """The 399 PULSE length (s) override for ONE image, or ``None`` when the extra is unset."""
    t = _extras_num(s, name, -1.0)
    return None if t < 0 else t


def _add_imag399(sb, g, amps, exposure=None):
    """The EXACT pre-existing ``Imag399Step`` call when neither override is given, else the twin."""
    if amps is None and exposure is None:
        return sb.add_step(Imag399Step, g)
    a1, a2 = amps if amps is not None else (-1.0, -1.0)
    return sb.add_step(Imag399AmpStep, g, a1, a2,
                       -1.0 if exposure is None else exposure)


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoff)
def PPGTransportDepthCommSeq(s):
    s.G.rearrange_img1_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)

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

    _add_imag399(s, s.C.Imag399, _img_amps(s, "InitImgAmp1", "InitImgAmp2"),
                 _img_exposure(s, "InitImgExposure"))

    s.add_step(Cool556hXStep, s.C.Cool556)

    # ---- transit molasses (unchanged from RearrangeCommSeq; defaults to OFF) ----
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)
    s.add('Freq556MOTX', Freq_Resonance556mj0Freq + Freq_Cool556Detuning)
    s.add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Resonance556mj0Freq + Freq_Cool556Detuning)
    s.add('Amp556RydbergMOTh', Amp_Cool556)

    # ---- RAISE THE TRAP DEPTH for the transport, adiabatically ----
    # The FPGA holds this across the handoff, so the whole rearrange window runs at the new depth.
    #
    # V_normal MUST come from ``s.C.Init`` (the PATTERN-OVERLAID value), not from ``Consts()``.
    # ``Consts().Init.VSLMServo`` is the BASE config, 3.7, while ByPattern["33x33_feedback11"]
    # sets 1.9 -- so reading the raw Consts would ramp the servo to 3.7 (about double this
    # array's depth) right before img2, and would make the "no change" control cell a secret
    # 1.9 -> 3.7 depth ramp. ``SLMStep`` resolves the same overlaid value via its
    # ``SLM.VServo`` cross-reference, so this is also what bseq2 snaps back to.
    # (Read in bseq1 scope = the INITIAL pattern's value; this campaign uses the same pattern for
    # both frames, so it is also the final one. A scan with different initial/final patterns would
    # need the ramp-down target re-read under the final pattern.)
    V_normal = s.C.Init.VSLMServo(Consts().Init.VSLMServo)
    V_transport = s.C.rearrange_kwargs.extras.TransportVServo(V_normal)
    t_ramp = s.C.rearrange_kwargs.extras.DepthRampMs(5.0) * 1e-3
    t_hold = s.C.rearrange_kwargs.extras.DepthHoldMs(10.0) * 1e-3

    s.add_step(t_ramp).add('VSLMservo', ramp_to(V_transport))

    # Second part: SLM-rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> bits -> rearrange

    # Settle at the deep setting BEFORE touching anything, then come back down -- both must
    # precede SLMStep, which would otherwise snap VSLMservo back to Init.VSLMServo at t=0.
    s2.wait(t_hold)
    s2.add_step(t_ramp).add('VSLMservo', ramp_to(V_normal))

    s2.add_step(SLMStep, s.C.SLM)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    _add_imag399(s2, s.C.Imag399, _img_amps(s, "FinImgAmp1", "FinImgAmp2"),
                 _img_exposure(s, "FinImgExposure"))

    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)
    return s


# =========================================================================== #
# Deferred per-shot callbacks -- identical to RearrangeCommSeq's.
# =========================================================================== #
def pre_run(s1):
    rearrange_callbacks.pre_run(s1, flags=("rearrange_img1_ok",),
                                lock_desc="rearrange compute")


def hand_over_slm(s1):
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", record_ok=True)


def post_run(s1):
    rearrange_callbacks.finalize(
        s1, round_flags=("rearrange_img1_ok",), final_frame_idx=1, tag="post_run")
