"""DummySeq.py -- transliteration of ``matlab_new/YbSeqs/DummySeq.m``.

nargin-0 seq: creates its own ExpSeq and returns it. Wires:
Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Init.

The MATLAB regBeforeStart(@server_pre_run) / regAfterEnd(@server_post_run) and the
Imag399 step are commented out in the source -> dropped here.
"""

from BlueMOTStep import BlueMOTStep
from exp_seq import ExpSeq
from GreenMOTStep import GreenMOTStep
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep


def DummySeq():
    s = ExpSeq()

    # Initialisation: 2D MOT & Zeeman slower on, shutters, AOMs/voltages to 0.
    s.add_step(InitStep, s.C.Init)

    # Blue MOT loading.
    s.add_step(BlueMOTStep, s.C.BlueMOT)

    # SLM tweezers on.
    s.add_step(SLMStep, s.C.SLM)

    # Green MOT: handover and cool down.
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    # LAC: turn off MOT, let cloud fly out, light-assisted collisions in tweezers.
    s.add_step(LACStep, s.C.LAC)

    # Initialisation again (shut down for safety).
    s.add_step(InitStep, s.C.Init)

    return s
