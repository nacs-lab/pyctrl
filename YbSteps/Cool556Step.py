"""Cool556Step.py -- transliteration of ``matlab_new/YbSteps/Cool556Step.m``.

``g = s.C.Cool556``. Sub-Doppler 556nm cooling pulse: turn off absorption imaging,
pulse the 556 MOT beams on resonance-plus-detuning at the cooling amplitude for
``t_Cool556``, then turn the beams back off.

All arithmetic here is on concrete config floats (no globals/measures), so plain int
literals stay concrete Python floats and never become SeqVal operands -- no explicit
float coercion is needed. Pulse VALUES (2nd arg to ``add``) are float()-coerced by the
framework, so a bare ``0`` serializes as float64 -- faithful to MATLAB's doubles.
"""

from consts import Consts


def Cool556Step(s, g):
    t_Cool556 = g.Time(Consts().Cool556.Time)

    Freq_Cool556Detuning = g.FreqDetuning(Consts().Cool556.FreqDetuning)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = g.Amp(Consts().Cool556.Amp)

    s.add('AmpAbsImag', 0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)
    s.wait(t_Cool556)
    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', 0)
