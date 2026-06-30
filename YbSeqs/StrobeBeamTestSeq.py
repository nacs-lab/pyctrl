"""StrobeBeamTestSeq.py -- minimal beam-response test for StrobeImag399Step (NO atoms).

Init -> StrobeImag399Step -> Init. There is NO MOT/SLM/LAC loading: this is purely to drive the
399/556 AOMs at the strobe timing so a photodiode + scope can measure the real optical response
(the AOM rise ~1 us vs the 800 ns command). The camera frame it produces is dark/incidental.

Use this for the FIRST scope characterization ("does the beam open in 800 ns; what is the on-fraction").
For a real strobe IMAGE with atoms, instead put StrobeImag399Step where Imag399Step normally goes in a
survival seq (e.g. Init->BlueMOT->SLM->GreenMOT->LAC->StrobeImag399 ... -> Imag399), which is a separate
seq; this one intentionally loads nothing.
"""

from InitStep import InitStep
from StrobeImag399Step import StrobeImag399Step


def _noop(s1):
    pass


def StrobeBeamTestSeq(s):
    s.reg_before_start(_noop)            # server_pre_run (deferred)

    s.add_step(InitStep, s.C.Init)
    s.add_step(StrobeImag399Step, s.C.Imag399)   # StrobeImag399Step reads g = s.C.Imag399
    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)               # server_post_run (deferred; camera -> server)
    return s
