"""Imag399Step.py -- transliteration of ``matlab_new/YbSteps/Imag399Step.m``.

``g = s.C.Imag399``. 399nm absorption imaging: open the abs-imaging shutter, close the
556 Rydberg shutter, set the 556 cooling beams (X & h) on resonance with the imaging
detunings, wait for the shutter, fire the 399 imaging beam and an Orca camera trigger,
expose, then turn everything back off.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Imag399.X.Y)``);
``Consts()`` default args are SubProps that DynProps resolves to a number.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by the framework, so
bare ints (``0``/``1``) serialize as float64 -- faithful to MATLAB's doubles. No globals
or measures are used, so no explicit-float coercion is needed.
"""

from consts import Consts


def Imag399Step(s, g):
    t_Imag399 = g.ExposureTime(Consts().Imag399.ExposureTime)

    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = g.FreqDetuning(Consts().Imag399.FreqDetuning)
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    Amp_Imag399 = g.Amp(Consts().Imag399.Amp)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq

    Freq_Cool556DetuningX = g.Cool556.X.FreqDetuning(Consts().Imag399.Cool556.X.FreqDetuning)
    Freq_Cool556Detuningh = g.Cool556.h.FreqDetuning(Consts().Imag399.Cool556.h.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Freq_Cool556h = Freq_Resonance556mj0Freq + Freq_Cool556Detuningh

    Amp_Cool556X = g.Cool556.X.Amp(Consts().Imag399.Cool556.X.Amp)
    Amp_Cool556h = g.Cool556.h.Amp(Consts().Imag399.Cool556.h.Amp)

    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL556RydbergShutter', 0)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', Amp_Cool556h)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', Amp_Imag399)
    s.add_step(100e-6).add('TTLOrcaTrig', 1)

    s.add('TTLOrcaTrig', 0)

    s.wait(t_Imag399)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', 0)

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', 0)

    s.add('TTL399AbsImagShutter', 0)
    s.wait(3e-3)  # wait for the shutter
