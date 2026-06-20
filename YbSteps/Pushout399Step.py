"""Pushout399Step.py -- transliteration of ``matlab_new/YbSteps/Pushout399Step.m``.

``g = s.C.Pushout``. Pushes out atoms not in the target state: set bias/Ryd coil
voltages (current->voltage conversion), open the 399 absorption-imaging shutter, fire
the 399 (via AbsImag) + 556 MOT beams for the pushout time, then turn everything off,
restore the SLM servo, and zero the coils.

All arithmetic here is on concrete config floats (no globals/measures), so plain int
literals like ``5``/``10``/``100``/``7`` stay concrete Python floats -- they never
become SeqVal operands, so no explicit-float coercion is needed.
"""

from consts import Consts
from ramp_to import ramp_to


def Pushout399Step(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    Freq_Pushout399 = g.Blue.Freq(Consts().Pushout.Blue.Freq)
    Amp_Pushout399_1 = g.Blue.Amp1(0)
    Amp_Pushout399_2 = g.Blue.Amp2(0)
    Freq_Pushout556 = g.Green.Freq(Consts().Pushout.Green.Freq)
    Amp_Pushout556 = g.Green.Amp(0)
    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)

    I_RydCoil = g.BiasCoilCurrent.Ryd(0)
    V_RydCoil = 5 * I_RydCoil / 100
    s.add('VRydCoil', V_RydCoil)

    I_BiasCoilX = g.BiasCoilCurrent.X(0)
    V_BiasCoilX = 10 * I_BiasCoilX / 7
    s.add('VBiasCoilX', V_BiasCoilX)

    I_BiasCoilY = g.BiasCoilCurrent.Y(0)
    V_BiasCoilY = 10 * I_BiasCoilY / 7
    s.add('VBiasCoilY', V_BiasCoilY)

    s.wait(50e-3)

    # Ramp the tweezer down and wait
    s.add('TTLScopeTrig', 1)

    # Using the 369 fiber output
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)

    s.wait(3e-3)  # wait fot the shutter

    s.add('FreqAbsImag', Freq_Pushout399).add('AmpAbsImag', Amp_Pushout399_1)
    s.add('Freq399Imag2', Freq_Pushout399).add('Amp399Imag2', Amp_Pushout399_2)

    # We previously use the MOT beams to do pushout
    s.add('Freq556MOTX', Freq_Pushout556).add('Amp556MOTX', Amp_Pushout556)
    s.add('Freq556RydbergMOTh', Freq_Pushout556).add('Amp556RydbergMOTh', Amp_Pushout556)

    s.wait(t_Pushout)

    s.add('TTLScopeTrig', 0)

    s.add_step(1e-3).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))

    s.add('AmpAbsImag', 0)
    s.add('Amp399Imag2', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpAOM308', 0)
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)  # mirrors beam-1 shutter (set to 1, not 0, here -- pre-existing quirk)
    s.add('TTL556MOTaShutter', 1).add('TTL556MOTbShutter', 1).add('TTL556MOTcShutter', 1)
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL369Shutter', 0)

    s.add('VRydCoil', 0)
    s.add('VBiasCoilZ', 0)
    s.add('VBiasCoilX', 0)
    s.add('VBiasCoilY', 0)
    s.wait(50e-3)
