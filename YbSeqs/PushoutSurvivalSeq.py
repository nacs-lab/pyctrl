"""PushoutSurvivalSeq.py -- transliteration of
``matlab_new/YbSeqs/PushoutSurvivalSeq.m``.

nargin-1 seq. Opens with a build-time 616-EOM frequency ramp driven by a sequence
global (set from MemoryMap in the DEFERRED server_pre_run, which serialize() never runs,
so the global stays unassigned -- matching the engine-free capture). Then:
Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 -> Cool556 -> Pushout -> Imag399 ->
(wait) -> Init.

BYTE-CRITICAL: in ``abs((Freq_EOM616 - freq616global) * 20e-9 * 3.0) + 20e-3`` the literal
``3.0`` is a SeqVal operand (the expression involves the global), and the framework does
NOT float-coerce SeqVal operands -- so it must be a float to match MATLAB's double
(FLOAT64); a bare ``3`` would serialize as INT32 and diverge.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from PushoutStep import PushoutStep
from ramp_to import ramp_to
from SLMStep import SLMStep


def _noop(s1):
    pass


def PushoutSurvivalSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred; reads MemoryMap, sets global)

    # Initialising 616EOM to its old value from last run (via a sequence global).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    # Slow EOM ramp. 3.0 (not 3): SeqVal operand -> must be FLOAT64.
    time = abs((Freq_EOM616 - freq616global) * 20e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    # Cool556.
    s.add_step(Cool556Step, s.C.Cool556)

    # PushOut556 (and/or 308): shine 556 (and/or 308) light to push out atoms.
    s.add_step(PushoutStep, s.C.Pushout)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; MemoryMap + camera)

    # SeqPlotter dump is gated on s.C.debug, which defaults to 0 -> never taken.
    debug = s.C.debug(0)
    if debug:
        s.dump_output_to_file(100, 'DebugPushoutSurvival.seq', 'PushoutSurvival')

    return s
