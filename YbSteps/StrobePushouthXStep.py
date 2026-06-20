"""StrobePushouthXStep.py -- strobe-imaging variant of PushouthXStep.

``g = s.C.Pushout``. Drives the IMAGING illumination (the two 399 imaging beams + the 556 X/h
cooling beams) during the push-out time slot, so the "push-out" acts as a controllable
strobe-imaging pulse between the two survival images. Every parameter DEFAULTS to its
imaging/cooling config value but is overridable per-scan via the ``Pushout.*`` params, so a scan
(StrobeImageScan) can sweep the strobe values:

  399 (two physical beams, mirrors Imag399Step):
    * freq      -> ``Blue.Freq``,  default = ``Resonance399Freq + Imag399.FreqDetuning`` (imaging line)
    * beam 1 amp -> ``Blue.Amp1``, default = ``Imag399.Amp1``  (AmpAbsImag)
    * beam 2 amp -> ``Blue.Amp2``, default = ``Imag399.Amp2``  (Amp399Imag2)
  556 cooling (X + h, the same channels Imag399Step cools with):
    * freq -> ``Green.{X,h}.Freq``, default = ``Resonance556mj0Freq + Imag399.Cool556.{X,h}.FreqDetuning``
    * amp  -> ``Green.{X,h}.Amp``,  default = ``Imag399.Cool556.{X,h}.Amp``

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(default)``). Pulse VALUES are
float()-coerced by the framework, so bare ints (``0``/``1``) serialize as float64. No
globals/measures here, so the explicit-float exception does not apply.
"""

from consts import Consts


def StrobePushouthXStep(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    # Two 399 strobe-imaging beams: default to the imaging line + the two imaging amplitudes.
    Freq_Pushout399 = g.Blue.Freq(Consts().Resonance399Freq + Consts().Imag399.FreqDetuning)
    Amp_Pushout399_1 = g.Blue.Amp1(Consts().Imag399.Amp1)   # beam 1 -> AmpAbsImag
    Amp_Pushout399_2 = g.Blue.Amp2(Consts().Imag399.Amp2)   # beam 2 -> Amp399Imag2

    # 556 cooling beams (X + h): default to the during-imaging cooling values (Imag399.Cool556).
    Freq_Resonance556mj0 = Consts().Resonance556mj0Freq
    Freq_Pushout556X = g.Green.X.Freq(Freq_Resonance556mj0 + Consts().Imag399.Cool556.X.FreqDetuning)
    Amp_Pushout556X = g.Green.X.Amp(Consts().Imag399.Cool556.X.Amp)
    Freq_Pushout556h = g.Green.h.Freq(Freq_Resonance556mj0 + Consts().Imag399.Cool556.h.FreqDetuning)
    Amp_Pushout556h = g.Green.h.Amp(Consts().Imag399.Cool556.h.Amp)

    # Using the 369 fiber output
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    s.add('TTL369Shutter', 1)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Pushout399).add('AmpAbsImag', Amp_Pushout399_1)
    s.add('Freq399Imag2', Freq_Pushout399).add('Amp399Imag2', Amp_Pushout399_2)

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
