"""TweezerLoadingSeq.py -- transliteration of ``matlab_new/YbSeqs/TweezerLoadingSeq.m``.

nargin-1 seq. Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 -> (wait) -> Init.

The regBeforeStart(@server_pre_run) / regAfterEnd(@server_post_run) callbacks drive the
camera/experiment-server, but only in DEFERRED callbacks that serialize() never runs, so
they are registered as no-ops (zero byte impact). MATLAB fprintf logging -> dropped.
"""

from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep


def _noop(s1):
    pass


def TweezerLoadingSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred; no build-time effect)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
