"""PushouthXStep.py -- transliteration of ``matlab_new/YbSteps/PushouthXStep.m``.

``g = s.C.Pushout``. Pushout pulse with various beams with more flexibility than the pushoutstep, can make 556 h/X different: 
open the AbsImag/Imag2 shutters, set the 399 push-out beam (FreqAbsImag/AmpAbsImag) and the two 556 beams
(X and h), hold for the pushout time, then turn everything off, reset shutters and the
Ryd coil voltage.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``);
``Consts()`` default args are SubProps that DynProps eagerly resolves to a number.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by the framework, so
bare ints (``0``/``1``) serialize as float64 -- faithful to MATLAB's doubles. No
globals/measures here, so the explicit-float exception does not apply.
"""

from consts import Consts


def PushouthXStep(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    Freq_Pushout399 = g.Blue.Freq(Consts().Pushout.Blue.Freq)
    # Beam 1 (AmpAbsImag) <- g.Blue.Amp; beam 2 (Amp399Imag2) <- g.Blue.Amp2 -- read INDEPENDENTLY,
    # matching PushoutStep / Pushout399Step. (Both beams share g.Blue.Freq; there is no Blue.Freq2.)
    Amp1_Pushout399 = g.Blue.Amp(Consts().Pushout.Blue.Amp1)
    Amp2_Pushout399 = g.Blue.Amp2(Consts().Pushout.Blue.Amp2)

    Freq_Pushout556X = g.Green.X.Freq(Consts().Pushout.Green.Freq)
    Amp_Pushout556X = g.Green.X.Amp(Consts().Pushout.Green.Amp)
    Freq_Pushout556h = g.Green.h.Freq(Consts().Pushout.Green.Freq)
    Amp_Pushout556h = g.Green.h.Amp(Consts().Pushout.Green.Amp)

    # Turn on shutters
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    #s.add('TTL369Shutter', 1)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Pushout399).add('AmpAbsImag', Amp1_Pushout399)
    s.add('Freq399Imag2', Freq_Pushout399).add('Amp399Imag2', Amp2_Pushout399)

    s.add('Freq556MOTX', Freq_Pushout556X).add('Amp556MOTX', Amp_Pushout556X)
    s.add('Freq556RydbergMOTh', Freq_Pushout556h).add('Amp556RydbergMOTh', Amp_Pushout556h)

    s.wait(t_Pushout)

    s.add('TTLScopeTrig', 0)
    s.add('AmpAbsImag', 0)
    s.add('Amp399Imag2', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpAOM308', 0)
    s.add('TTL399AbsImagShutter', 0)
    s.add('TTL399Imag2Shutter', 0)
    s.add('TTL556MOTaShutter', 1).add('TTL556MOTbShutter', 1).add('TTL556MOTcShutter', 1)
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL369Shutter', 0)

    s.add('VRydCoil', 0)
