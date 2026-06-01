"""TweezerEnhancedLoadingSeq.py -- transliteration of
``matlab_new/YbSeqs/TweezerEnhancedLoadingSeq.m``.

nargin-1 seq. Init -> BlueMOT -> SLM -> GreenMOT -> BlueLAC(LAC cfg) -> Imag399 ->
(wait) -> Init. Server callbacks are deferred no-ops.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from SLMStep import SLMStep


def _noop(s1):
    pass


def TweezerEnhancedLoadingSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    # BlueLAC: blue-detuned light-assisted collisions in tweezers.
    s.add_step(BlueLACStep, s.C.LAC)

    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
