"""PPGTransportCoolCommSeq.py -- RearrangeCommSeq with PER-BEAM transit cooling.

WHY THIS EXISTS (2026-08-10).  ``RearrangeCommSeq`` drives the transit molasses from a SINGLE
pair of knobs::

    Freq556MOTX / Freq556RydbergMOTh <- Resonance556mj0Freq + extras.RearrCoolDet
    Amp556MOTX  / Amp556RydbergMOTh  <- extras.RearrCoolAmp

i.e. the X beams (X1+X2) and the H beam are forced to the same detuning and the same amplitude.
Jobs 641/646 scanned that pair over det -0.35..+0.70 MHz and amp 0.005..0.25 and found a clean
harm resonance but NO cooling benefit.  One mundane explanation is geometric: the 556 recool is
a two-channel molasses (``556MOTX`` = X1+X2, diagonal in a plane containing the tweezer axis;
``556RydbergMOTh`` = the horizontal MOT-H beam), and a beam with little projection on the
transport axis cannot damp the degree of freedom transport heats.  Driving both beams together
averages any such anisotropy away -- so it cannot be tested with the single-knob seq.

This seq is ``RearrangeCommSeq`` with that one block replaced by a per-beam version::

    extras.RearrCoolAmpX / RearrCoolDetX   -> Amp556MOTX / Freq556MOTX
    extras.RearrCoolAmpH / RearrCoolDetH   -> Amp556RydbergMOTh / Freq556RydbergMOTh

Each defaults to the CORRESPONDING single-beam extra (``RearrCoolAmp`` / ``RearrCoolDet``, which
themselves default to 0 / 0.13 MHz), so with none of the per-beam keys set this seq emits exactly
what ``RearrangeCommSeq`` emits -- same channels, same order, same values.  Everything else in
the file (steps, bseq structure, imaging overrides, callbacks) is a verbatim copy of
``RearrangeCommSeq``; the production seq is not touched, and no default moves.

Used by ``YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode dir``.
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
from SLMStep import SLMStep

import rearrange_callbacks
from seq_capability import seq_capabilities


# --------------------------------------------------------------------------- #
# Per-frame 399 imaging brightness -- verbatim copies of RearrangeCommSeq's helpers (that file
# documents the policy in full).  Copied rather than imported: nothing in YbSeqs has a seq -> seq
# import edge and this file does not introduce the first one.  Keep the two sets in lockstep.
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
    """The EXACT pre-existing ``Imag399Step`` call when neither override is given, else the
    overridable twin."""
    if amps is None and exposure is None:
        return sb.add_step(Imag399Step, g)
    a1, a2 = amps if amps is not None else (-1.0, -1.0)
    return sb.add_step(Imag399AmpStep, g, a1, a2,
                       -1.0 if exposure is None else exposure)


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoff)
def PPGTransportCoolCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload

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

    # ---- the transit molasses, PER BEAM (the only change vs RearrangeCommSeq) ----
    # The FPGA holds these DDS values across the bseq1 -> bseq2 handoff, so they are what the
    # atoms see for the whole host-side rearrange window (detect + compute + the SLM playback
    # that IS the transport). bseq2's Cool556hXStep turns them off again.
    #
    # Each per-beam key defaults to the shared RearrCool* value, which itself defaults to the
    # RearrangeCommSeq default -- so with no per-beam extras set this block emits exactly what
    # RearrangeCommSeq emits, on the same channels in the same order.
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Det_shared = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Amp_shared = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    Det_X = s.C.rearrange_kwargs.extras.RearrCoolDetX(Det_shared)
    Amp_X = s.C.rearrange_kwargs.extras.RearrCoolAmpX(Amp_shared)
    Det_H = s.C.rearrange_kwargs.extras.RearrCoolDetH(Det_shared)
    Amp_H = s.C.rearrange_kwargs.extras.RearrCoolAmpH(Amp_shared)

    s.add('Freq556MOTX', Freq_Resonance556mj0Freq + Det_X).add('Amp556MOTX', Amp_X)
    s.add('Freq556RydbergMOTh', Freq_Resonance556mj0Freq + Det_H).add('Amp556RydbergMOTh', Amp_H)

    # Second part: SLM-rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> bits -> rearrange

    s2.add_step(SLMStep, s.C.SLM)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    _add_imag399(s2, s.C.Imag399, _img_amps(s, "FinImgAmp1", "FinImgAmp2"),
                 _img_exposure(s, "FinImgExposure"))

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img2 -> update_rearrange; release compute; keepalive slm
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
