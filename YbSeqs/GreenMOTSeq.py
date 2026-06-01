"""GreenMOTSeq.py -- transliteration of ``matlab_new/YbSeqs/GreenMOTSeq.m``.

nargin-1 seq. Wires: Init -> BlueMOT -> GreenMOT -> AbsImag -> (wait) -> Init.
"""

from AbsImagStep import AbsImagStep
from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from InitStep import InitStep


def GreenMOTSeq(s):
    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(AbsImagStep, s.C.AbsImag)
    s.wait(0.5)
    s.add_step(InitStep, s.C.Init)
    return s
