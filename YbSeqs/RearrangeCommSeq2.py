"""RearrangeCommSeq2.py -- two-round SLM-rearrangement variant of ``RearrangeCommSeq.py``.

Ground truth: the single-round ``RearrangeCommSeq.py`` (same lock/setup/detect/stage machinery,
now SHARED via :mod:`rearrange_callbacks`). This is the TWO-ROUND, THREE-PATTERN extension:

  Init -> MOT -> SLM -> GreenMOT -> LAC -> Imag399 (#1, LOADING pattern)
       -> Cool -> rearrange(round 1) -> Imag399 (#2, MIDDLE pattern)
       -> Cool -> rearrange(round 2) -> Imag399 (#3, FINAL pattern) -> Init

Three camera frames per shot. The scan (SLMRearrangementScan.py, two-round branch) declares three
patterns -- LOADING / MIDDLE / FINAL -- via ``rearrange_kwargs.extras.initial_pattern /
middle_pattern / final_pattern`` and a matching 3-entry ``runp().imagePatternsJson``. Each frame is
DETECTED with its OWN per-pattern registry grid + thresholds (independent site counts / orderings),
so the three patterns may be genuinely different arrays. The caller keeps each round's detected
site count in agreement with what the SLM server scores that round; a mismatch is surfaced (the
detector returns "" / [] on a stale/absent grid and the round is skipped), never a silent
off-by-one.

Pattern-write policy (per the user spec):
  * The LOADING (initial) phase is written once at scan start by :class:`SlmScanSession` (and
    re-written by ``ensure_held`` if the scan-long ``slm`` lock is ever lost). The atoms are then
    MOVED to the middle array by rearrange() round 1 and to the final array by round 2.
  * The MIDDLE and FINAL patterns are ASSUMED already on the SLM (produced by the rearrange calls);
    rounds 2 and 3 add NO SLMStep / no phase write -- they just cool + image as fast as possible.

Frame alignment / abort safety (the load-bearing invariant, enforced in rearrange_callbacks): a
shot either stages ALL THREE frames and publishes them with a single ``finish_shot``, or it is
cancelled and NO partial triple is persisted. Every callback that runs CONSUMES its frame even on
a failing shot (grab-and-drain) so a straggler can't shift img1/img2/img3 by one on the NEXT shot.
A failed round re-publishes the captured frames under the FAILING sentinel for LIVE DISPLAY ONLY.

The BUILD path is unchanged from the byte port (steps/branches/pattern tags); only the deferred
callbacks -- which ``serialize()`` never runs -- carry the runtime logic.
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


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def RearrangeCommSeq2(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_img2_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload, n_rounds=2

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 images the LOADING (dense load)
    # pattern; bseq2 the MIDDLE (round-1 target), bseq3 the FINAL (round-2 target). Each bseq's
    # cooling/imaging/VSLMServo resolve from ByPattern[that pattern]. Names from
    # rearrange_kwargs.extras.initial_pattern / middle_pattern / final_pattern (set by the scan);
    # absent -> scan-default / inherit. Tag each bseq HERE before its steps build. No-op when
    # ByPattern is empty.
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
    # non-None NI data for this bseq (Cool556/Imag399 touch only TTL+DDS). A physical no-op. See
    # the MATLAB original's note -- we must NOT InitStep between rounds (that zeroes VSLMservo and
    # loses the cooled atoms). ASSUME-WRITTEN policy: no SLMStep / no phase write here.
    s2.add('VMOTCoil', 0)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # Second Imag399 (img2, MIDDLE pattern).
    s2.add_step(Imag399Step, s.C.Imag399)

    # Round 2: second SLM-rearrangement basic sequence. Imaged at the FINAL pattern.
    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s3.set_pattern(_final_pat)

    s3.reg_before_bseq(hand_over_slm_2)   # img2 -> rearrange round 2 (middle -> final)

    # NI-DAQ keep-alive (same reason as s2). ASSUME-WRITTEN: no SLMStep / no phase write.
    s3.add('VMOTCoil', 0)

    s3.add_step(Cool556hXStep, s.C.Cool556)

    # Third Imag399 (img3, FINAL pattern).
    s3.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s3.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img3 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# Thin wrappers over the shared rearrange_callbacks machinery -- these names stay public
# (the seq registration above + the tests import them from THIS module).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement (sticky -- no reset_params, n_rounds forced to 2), and reload_rearrange."""
    rearrange_callbacks.pre_run(
        s1, flags=("rearrange_img1_ok", "rearrange_img2_ok"),
        lock_desc="rearrange compute (2 rounds)", force_n_rounds=2)


def hand_over_slm(s1):
    """Round 1: read img1 (Imag399 #1, LOADING pattern), detect probs, rearrange(round 1), stage
    img1. Sets rearrange_img1_ok on success."""
    rearrange_callbacks.rearrange_round(
        s1, 0, ok_flag="rearrange_img1_ok", tag="hand_over_slm", use_frame_pattern=True)


def hand_over_slm_2(s1):
    """Round 2: read img2 (Imag399 #2, MIDDLE pattern), detect probs, rearrange(round 2), stage
    img2. Sets rearrange_img2_ok on success.

    Frame alignment: img2 is PHYSICALLY produced whether or not round 1 succeeded. If round 1
    failed the shared round still CONSUMES img2 (grab-and-drain) so it can't straggle into the
    next shot's img1 -- then cancels. Never leave a produced frame buffered."""
    rearrange_callbacks.rearrange_round(
        s1, 1, ok_flag="rearrange_img2_ok", prior_flag="rearrange_img1_ok",
        tag="hand_over_slm_2", use_frame_pattern=True)


def post_run(s1):
    """Finalize: read img3 (Imag399 #3, FINAL pattern), update_rearrange(bits3), stage img3 +
    finish; release the compute lock and keepalive the scan-long slm lock. Publishes the full
    triple only when BOTH rounds succeeded, else re-publishes for DISPLAY ONLY (keeping the .h5
    in aligned img1/img2/img3 triples)."""
    rearrange_callbacks.finalize(
        s1, round_flags=("rearrange_img1_ok", "rearrange_img2_ok"), final_frame_idx=2,
        tag="post_run", use_frame_pattern=True, record_ok=True)
