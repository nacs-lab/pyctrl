"""PushoutStep.py -- transliteration of ``matlab_new/YbSteps/PushoutStep.m``.

``g = s.C.Pushout``. Push out atoms with various beams, keep 556 X and h identical: open the AbsImag/Imag2
shutters, drive the 399 (AbsImag DDS or Imag2 DDS), 556 (MOTX / RydbergMOTh DDS) beams at the
pushout frequency/amplitude, hold for the pushout time, then turn everything back off
and reset shutters / coils.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``);
``Consts()`` default args are SubProps that DynProps resolves to a number.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced, so bare ints
(``0``/``1``) serialize as float64 -- faithful to MATLAB's doubles. No globals/measures
here, so no explicit-float coercion is needed.
"""

from consts import Consts


def PushoutStep(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    Freq_Pushout399 = g.Blue.Freq(Consts().Pushout.Blue.Freq)
    Amp1_Pushout399 = g.Blue.Amp1(Consts().Pushout.Blue.Amp1)
    # beam 2 (Amp399Imag2): own knob, default 0 (off) -> swept via Pushout.Blue.Amp2; freq tied
    Amp_Pushout399_2 = g.Blue.Amp2(Consts().Pushout.Blue.Amp2)
    Freq_Pushout556 = g.Green.Freq(Consts().Pushout.Green.Freq)
    Amp_Pushout556 = g.Green.Amp(Consts().Pushout.Green.Amp)
    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)

    Amp_Pushout308 = g.Ryd308.Amp(Consts().Pushout.Ryd308.Amp)

    # Using the 369 fiber output
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    # s.add('TTL369Shutter', 1)

    s.wait(3e-3)  # wait for the shutter

    # Using the 369 fiber output
    s.add('FreqAbsImag', Freq_Pushout399).add('AmpAbsImag', Amp1_Pushout399)
    s.add('Freq399Imag2', Freq_Pushout399).add('Amp399Imag2', Amp_Pushout399_2)

    # We previously use the MOT beams to do pushout
    s.add('Freq556MOTX', Freq_Pushout556).add('Amp556MOTX', Amp_Pushout556)
    s.add('Freq556RydbergMOTh', Freq_Pushout556).add('Amp556RydbergMOTh', Amp_Pushout556)

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
    s.add('TTL556MOTaShutter', 1).add('TTL556MOTbShutter', 1).add('TTL556MOTcShutter', 1) # Keep 556 MOT shutters open
    s.add('TTL556RydbergShutter', 0)
    #s.add('TTL369Shutter', 0)

    s.add('VRydCoil', 0)
