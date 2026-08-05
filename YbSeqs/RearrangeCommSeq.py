"""RearrangeCommSeq.py -- port of ``matlab_new/YbSeqs/RearrangeCommSeq.m`` (single-round).

The seq BUILD path (the byte-producing part) is the faithful transliteration: a nargin-1 seq with
a SECOND basic sequence for the SLM-rearrangement handoff. ``serialize()`` never runs the deferred
callbacks, so the build path is byte-identical to MATLAB.

The deferred callbacks (``pre_run`` / ``hand_over_slm`` / ``post_run``) carry the per-shot
rearrangement logic, mirroring RearrangeCommSeq.m. They are thin wrappers over the SHARED
:mod:`rearrange_callbacks` machinery (extracted from this file + RearrangeCommSeq2.py so hybrid
"rearrange, then science" seqs compose the same logic). They reach the camera / ExptServer /
scan-long SLM session through :mod:`rearrange_runtime` (the pyctrl analog of MATLAB's base
workspace). The scan-long ``slm`` HARDWARE lock + loading-phase write are owned by
:class:`SlmScanSession` for the WHOLE scan (set up by the runner at dequeue); per shot we take
only the ``compute`` (GPU) lock for the rearrange window.

Per-shot flow (mirrors the user spec / MATLAB):
  pre_run       -- ensure the scan-long slm lock is held; grab the compute lock (cancel+retry on
                   miss); per-shot setup_rearrangement (no reset_params -> sticky); reload_rearrange.
  hand_over_slm -- read img1, detect probs, rearrange(probs), stage img1 (between Imag399 #1 and #2).
  post_run      -- read img2, update_rearrange(bits2), stage img2 + seq_finish; release the compute
                   lock; keepalive the scan-long slm lock.

The one later addition to the BUILD path is the OPT-IN per-frame 399 imaging override -- amps
(``extras.InitImgAmp1/2`` for img1, ``FinImgAmp1/2`` for img2, 2026-07-28) and 399 PULSE LENGTH
(``extras.InitImgExposure`` / ``FinImgExposure``, seconds, 2026-07-28) -- documented in the policy
block below. Absent, the build is byte-identical to the port; that is pinned by
``tests/test_rearrange_imaging_amps.py``.
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
# Per-frame 399 imaging brightness (IMAGING-POWER POLICY, 2026-07-28)
#
# Same mechanism (and the same helper bodies) as RearrangeCommSeq2.py -- see the long policy
# block there; only the frame names differ. In one sentence: the 399 imaging-power PID locks
# ONCE at the ROOT BlueMOTStep (the initial pattern's BlueMOT.Img1/Img2PIDSet, 0.5/0.5) and
# HOLDS for the whole shot -- there is no time to re-PID mid-shot and a relock rails the
# integrator (yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md). So BOTH frames
# are taken at that one held optical power and their only per-frame brightness knob is the DDS
# amps of each frame's imaging beams. PID setpoints stay PINNED by policy (moving one would move
# the final image's power too and force a re-optimization of the rearranged array).
#
# ByPattern[<pattern>].Imag399.Amp1/Amp2 already give a per-frame STATIC value, but they cannot
# be SWEPT: a scan-level ``g().Imag399.Amp1`` wins over ByPattern in EVERY bseq (precedence
# base < ByPattern < scan g(), lib/expConfig_helper.py:79-94), so it would move both frames
# together. These optional extras are the per-frame swept knob:
#
#     rearrange_kwargs.extras.InitImgAmp1 / InitImgAmp2  -> img1 (INITIAL / loading) only
#     rearrange_kwargs.extras.FinImgAmp1  / FinImgAmp2   -> img2 (FINAL / rearranged) only
#
# FinImgAmp* deliberately reuses RearrangeCommSeq2's name for its own final image, so one
# analysis script covers both seqs. InitImgAmp* is the knob the single-round campaign actually
# wants: dimming img1 costs frame-0 detection confidence but reduces 399 heating, and a high
# ``prob_hungarian_beta`` absorbs the lost confidence by routing around low-p sites.
#
# They compose ON TOP of ByPattern: absent -> the frame's own ByPattern/base Amp1/Amp2 is used
# and the build is BYTE-IDENTICAL to the pre-knob seq (same step body, same callbacks, same
# provenance name); present -> that literal value replaces this frame's amp only.
#
# AOM knee: amps 0.5-1.0 are optically FLAT, only <= 0.5 attenuates (2026-07-16 R212).
#
# 399 PULSE LENGTH (2026-07-28) -- the knob the >= 99% single-round campaign needs. The FINAL
# image is already at MAXIMUM available 399 power (its DDS amps are 1/1 and the AOM is flat over
# 0.5-1.0) while the PID setpoints are pinned by the policy above, so brightness-by-power is
# EXHAUSTED for that frame. What is left is TIME:
#
#     rearrange_kwargs.extras.InitImgExposure   -> img1 (INITIAL / loading) 399 pulse, seconds
#     rearrange_kwargs.extras.FinImgExposure    -> img2 (FINAL / rearranged) 399 pulse, seconds
#
# absent -> the frame's own ``g.ExposureTime`` (expConfig cross-refs it to ``Orca.ExposureTime``,
# 0.1 s for both production patterns) and the pre-knob bytes. Two things to keep straight:
#   * the CAMERA exposure is a single global hardware setting the runner syncs ONCE per scan
#     (YbExptCtrl/camera_runtime.sync_camera_exposure), so this moves the 399 pulse INSIDE a fixed
#     window -- past ``Orca.ExposureTime`` the tail lands outside the frame and buys no photons.
#   * the extra 399 heating is HARMLESS on img2 (nothing needs the atoms after the final image)
#     but NOT on img1, whose atoms still have to survive into the rearrangement.
#
# NOTE the four helpers below are a deliberate copy of RearrangeCommSeq2.py's (no seq -> seq
# import edge; nothing else in YbSeqs has one). Keep the two sets in lockstep.
# --------------------------------------------------------------------------- #
def _extras_num(s, name, default):
    """``rearrange_kwargs.extras.<name>`` as a float; ``default`` when absent/unresolvable.

    Byte-inert: DynProps persists the default into ``s.C`` on a miss (lib/dyn_props.py:188)
    but no pulse is emitted from it, so serialize() is unaffected."""
    try:
        return float(getattr(s.C.rearrange_kwargs.extras, name)(default))
    except Exception:  # noqa: BLE001 - absent/odd extras -> default
        return float(default)


def _img_amps(s, n1, n2):
    """The (Amp1, Amp2) override for ONE image, or ``None`` when the scan set NEITHER extra.

    ``None`` is the load-bearing default: it routes the build back through the untouched
    ``Imag399Step`` call, so a scan that does not use this knob is byte-identical to before."""
    a1 = _extras_num(s, n1, -1.0)
    a2 = _extras_num(s, n2, -1.0)
    return None if (a1 < 0 and a2 < 0) else (a1, a2)


def _img_exposure(s, name):
    """The 399 PULSE length (s) override for ONE image, or ``None`` when the extra is unset.

    Same load-bearing ``None`` as :func:`_img_amps`: it routes the build back through the
    untouched ``Imag399Step`` call."""
    t = _extras_num(s, name, -1.0)
    return None if t < 0 else t


def _add_imag399(sb, g, amps, exposure=None):
    """Add this bseq's Imag399: the EXACT pre-existing ``Imag399Step`` call when NEITHER override
    is given, else the overridable twin (identical body, ``a1``/``a2`` replacing g.Amp1/2 and
    ``t_exp`` replacing g.ExposureTime). Each unset override passes its own ``-1`` sentinel, so
    an amps-only call is bit-for-bit what it was before ``exposure`` existed."""
    if amps is None and exposure is None:
        return sb.add_step(Imag399Step, g)
    a1, a2 = amps if amps is not None else (-1.0, -1.0)
    return sb.add_step(Imag399AmpStep, g, a1, a2,
                       -1.0 if exposure is None else exposure)


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoff)
def RearrangeCommSeq(s):
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

    # First Imag399 (img1, INITIAL / loading pattern). Brightness = ByPattern[initial].Imag399.
    # Amp1/Amp2 at the root-held PID power, optionally overridden per-frame (and swept) by
    # extras.InitImgAmp1/InitImgAmp2 (+ extras.InitImgExposure for the 399 pulse length) -- see
    # the policy block at the top of this file. Lengthening THIS pulse adds heating to atoms that
    # still have to survive into the rearrangement; that is the knob's cost here.
    _add_imag399(s, s.C.Imag399, _img_amps(s, "InitImgAmp1", "InitImgAmp2"),
                 _img_exposure(s, "InitImgExposure"))

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

    # Second Imag399 (img2, FINAL / rearranged pattern). Same per-frame knobs as img1, via
    # extras.FinImgAmp1/FinImgAmp2 + extras.FinImgExposure (same names RearrangeCommSeq2 uses for
    # ITS final image). This frame's occupancy IS the fill metric and its amps are already railed
    # at 1/1, so extras.FinImgExposure is the campaign's remaining brightness lever -- and the
    # extra heating it causes is free, since no atoms are needed after this image.
    _add_imag399(s2, s.C.Imag399, _img_amps(s, "FinImgAmp1", "FinImgAmp2"),
                 _img_exposure(s, "FinImgExposure"))

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img2 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# Thin wrappers over the shared rearrange_callbacks machinery -- these names stay public
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
