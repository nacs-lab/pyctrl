"""ImagingSurvivalSeq.py -- transliteration of
``matlab_new/YbSeqs/ImagingSurvivalSeq.m``.

This is a weird sequence that does not follow the standard sequence convention. Tend to remove or consolidate.

nargin-1 seq with body code between the steps. Init -> BlueMOT -> SLM -> GreenMOT ->
LAC -> Imag399 -> (set cooling/imaging tones, wait t_Pushout) -> Imag399 -> (wait) -> Init.

Transliteration note: MATLAB auto-resolves a bare scalar-leaf read (``Consts().X.Y``) to
its value, but a pyctrl SubProps is lazy -- so every bare scalar-leaf read used as a value
gets an explicit ``()`` to force resolution (the steps avoid this by always using the
``g.X(Consts()...)`` default-call form). Server callbacks are deferred no-ops; disp dropped.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep


def _noop(s1):
    pass


def ImagingSurvivalSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    Freq_Imag399Detuning = Consts().Imag399.FreqDetuning()
    Freq_Resonance399 = Consts().Resonance399Freq()
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    Amp_Imag399_1 = Consts().Imag399.Amp1()   # beam 1 -> AmpAbsImag
    Amp_Imag399_2 = Consts().Imag399.Amp2()   # beam 2 -> Amp399Imag2

    Freq_Cool556Detuning = Consts().Imag399.Cool556.FreqDetuning()
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = Consts().Imag399.Cool556.Amp()

    s.add('FreqCatsEye', Freq_Cool556).add('AmpCatsEye', Amp_Cool556)
    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', Amp_Imag399_1)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', Amp_Imag399_2)

    g = s.C.Pushout
    t_Pushout = g.Time(Consts().Pushout.Time)
    s.wait(t_Pushout)

    # PushOut556 step is commented out in the source.

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
