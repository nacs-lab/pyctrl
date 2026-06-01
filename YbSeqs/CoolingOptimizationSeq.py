"""CoolingOptimizationSeq.py -- transliteration of
``matlab_new/YbSeqs/CoolingOptimizationSeq.m``.

nargin-1 seq. Modulates the SLM tweezer traps between two images:
Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 -> SLMTrapModulation -> Imag399 ->
(wait) -> Init. Server callbacks are deferred no-ops.
"""

from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep
from SLMTrapModulationStep import SLMTrapModulationStep


def _noop(s1):
    pass


def CoolingOptimizationSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    # SLM trap modulation.
    s.add_step(SLMTrapModulationStep, s.C.SLMTrapModulation)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
