"""ReleaseRecaptureSeq.py -- transliteration of
``matlab_new/YbSeqs/ReleaseRecaptureSeq.m``.

nargin-1 seq. Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 ->
Cool556hX(Cool556 cfg) -> ReleaseRecapture -> Imag399 -> (wait) -> Init.
Server callbacks are deferred no-ops.
"""

from BlueMOTStep import BlueMOTStep
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from ReleaseRecaptureStep import ReleaseRecaptureStep
from SLMStep import SLMStep


def _noop(s1):
    pass


def ReleaseRecaptureSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    # Cool556 (the plain Cool556Step / Imag399heatingStep variants are commented out).
    s.add_step(Cool556hXStep, s.C.Cool556)

    # Release and recapture.
    s.add_step(ReleaseRecaptureStep, s.C.ReleaseRecapture)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
