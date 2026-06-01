"""Cool556hXStep.py -- transliteration of ``matlab_new/YbSteps/Cool556hXStep.m``.

``g = s.C.Cool556``. Power-balanced green (556nm) cooling on the X and h MOT beams:
set their frequencies (resonance + detuning) and amplitudes, hold for the cool time,
then turn the amplitudes off (frequencies held).

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by ``_resolve_pulse``,
so the bare ``0`` amplitudes serialize as ARG_CONST_FLOAT64 -- faithful to MATLAB's
doubles. No globals/measures here, so no explicit-float operand coercion is needed.
"""

from consts import Consts


def Cool556hXStep(s, g):
    t_Cool556 = g.Time(Consts().Cool556.Time)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq

    Freq_Cool556DetuningX = g.X.FreqDetuning(Consts().Cool556.X.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Amp_Cool556X = g.X.Amp(Consts().Cool556.X.Amp)

    Freq_Cool556Detuningh = g.h.FreqDetuning(Consts().Cool556.h.FreqDetuning)
    Freq_Cool556h = Freq_Resonance556mj0Freq + Freq_Cool556Detuningh
    Amp_Cool556h = g.h.Amp(Consts().Cool556.h.Amp)

    s.add('AmpAbsImag', 0)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', Amp_Cool556h)
    s.wait(t_Cool556)
    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', 0)
