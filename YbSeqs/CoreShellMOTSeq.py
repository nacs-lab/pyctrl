"""CoreShellMOTSeq.py -- transliteration of ``matlab_new/YbSeqs/CoreShellMOTSeq.m``.

nargin-1 seq: takes a configured ``ExpSeq`` and returns it. Wires the step cone:
Init -> CoreShellMOT -> AbsImag -> (wait) -> Init.
"""

from AbsImagStep import AbsImagStep
from CoreShellMOTStep import CoreShellMOTStep
from InitStep import InitStep


def CoreShellMOTSeq(s):
    # Initialisation: 2D MOT & Zeeman slower on, shutters, AOMs/voltages to 0.
    s.add_step(InitStep, s.C.Init)

    # CoreShell MOT: handover and cool down.
    s.add_step(CoreShellMOTStep, s.C.CoreShellMOT)

    # Absorption imaging: 399nm light imaging on 556 MOT in free space.
    s.add_step(AbsImagStep, s.C.AbsImag)

    # Initialisation again (shut down for safety).
    s.wait(0.5)
    s.add_step(InitStep, s.C.Init)

    return s
