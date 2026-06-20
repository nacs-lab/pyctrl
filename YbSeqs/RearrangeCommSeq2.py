"""RearrangeCommSeq2.py -- transliteration of ``matlab_new/YbSeqs/RearrangeCommSeq2.m``.

Two-round SLM-rearrangement variant: THREE basic sequences chained by branches
(s -> s2 -> s3). All hardware is in deferred callbacks (no-ops here). Same build-path
surface as RearrangeCommSeq. Note s2 reasserts ``VMOTCoil 0`` (a physical no-op) so the
bseq registers a V* event -- a runtime NI-DAQ concern, faithfully reproduced for bytes.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep

from seq_capability import seq_capabilities


def _noop(s1):
    pass


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def RearrangeCommSeq2(s):
    # Per-seq coordination flags.
    s.G.rearrange_img1_ok = False
    s.G.rearrange_img2_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(_noop)          # pre_run: connect, lock, prewarm, n_rounds=2 (deferred)

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 -> INITIAL pattern; the
    # post-rearrangement images (round 1 + round 2 below) -> FINAL. Names from
    # rearrange_kwargs.extras.initial_pattern / final_pattern (set by the scan); absent ->
    # scan-default / inherit. Tag s HERE before its steps build. No-op when ByPattern is empty.
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

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556Step, s.C.Cool556)

    # Leave the cooling light on a little during rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # Round 1: SLM rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    # Round 1 + round 2 images -> the FINAL/target pattern (see the initial_pattern note above).
    # [If round 1 ever needs a distinct intermediate pattern, declare and apply it here.]
    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(_noop)          # hand_over_slm (deferred)

    # NI-DAQ keep-alive: reassert one V* channel so libnacs emits non-None NI data.
    s2.add('VMOTCoil', 0)

    s2.add_step(Cool556Step, s.C.Cool556)

    # Second Imag399.
    s2.add_step(Imag399Step, s.C.Imag399)

    # Round 2: second SLM-rearrangement basic sequence.
    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)
    if _final_pat:
        s3.set_pattern(_final_pat)     # round-2 image also at the final/target pattern

    s3.reg_before_bseq(_noop)          # hand_over_slm_2 (deferred)

    s3.add_step(Cool556Step, s.C.Cool556)

    # Third Imag399.
    s3.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s3.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # post_run (deferred; camera + lock release)
    return s
