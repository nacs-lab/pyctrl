"""Imag399heatingStep.py -- transliteration of ``matlab_new/YbSteps/Imag399heatingStep.m``.

``g = s.C.Imag399heating``. Like Imag399Step but with NO camera trigger: fires the 399
imaging beam (same exposure, same cooling), but does NOT pulse TTLOrcaTrig. Used for
accumulating imaging-light exposure before the final triggered detection image.

Difference from Imag399Step:
  - No ``TTLOrcaTrig`` pulse.
  - X and h cooling beams share a single ``g.Cool556.FreqDetuning`` (not separate X/h
    detunings), defaulting to ``Consts().Imag399.Cool556.FreqDetuning``.
  - Config group is ``s.C.Imag399heating`` (separate from ``s.C.Imag399``).
"""

from consts import Consts


def Imag399heatingStep(s, g):
    t_Imag399 = g.ExposureTime(Consts().Imag399.ExposureTime)

    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = g.FreqDetuning(Consts().Imag399.FreqDetuning)
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    Amp_Imag399_1 = g.Amp1(Consts().Imag399.Amp1)   # beam 1 -> AmpAbsImag (369 fiber output)
    Amp_Imag399_2 = g.Amp2(Consts().Imag399.Amp2)   # beam 2 -> Amp399Imag2 (second imaging beam)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq
    Freq_Cool556Detuning = g.Cool556.FreqDetuning(Consts().Imag399.Cool556.FreqDetuning)
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning

    Amp_Cool556X = g.Cool556.X.Amp(Consts().Imag399.Cool556.X.Amp)
    Amp_Cool556h = g.Cool556.h.Amp(Consts().Imag399.Cool556.h.Amp)

    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    s.add('TTL556RydbergShutter', 0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556h)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', Amp_Imag399_1)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', Amp_Imag399_2)
    # No TTLOrcaTrig -- this is a "dark" exposure (no camera image taken).

    s.wait(t_Imag399)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', 0)

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', 0)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', 0)

    s.add('TTL399AbsImagShutter', 0)
    s.add('TTL399Imag2Shutter', 0)
    s.wait(3e-3)  # wait for the shutter
