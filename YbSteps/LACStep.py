"""LACStep.py -- transliteration of ``matlab_new/YbSteps/LACStep.m``.

``g = s.C.LAC``. Loss-assisted collisions (LAC): turn off the MOT beams and coils
and wait for the cloud to fly out (dead time), then drive the 556 MOT beams on the
``mj=0`` resonance (plus a detuning) for a hold, and turn them off.

Reads resolve config with a ``Consts()`` fallback default; note the resonance read is
top-level (``Consts().Resonance556mj0Freq``), not under ``LAC``. All arithmetic is on
concrete config floats (no globals/measures), so no explicit-float coercion is needed.
"""

from consts import Consts


def LACStep(s, g):
    # step 1: turn off MOT and wait for it to fly out
    t_Dead = g.DeadTime(Consts().LAC.DeadTime)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpBlueMOT', 0)
    s.add('VMOTCoil', 0)
    s.add('VRydCoil', 0)
    s.add('VBiasCoilX', 0)
    s.add('VBiasCoilY', 0)
    s.add('VBiasCoilZ', 0)
    s.wait(t_Dead)

    # step 2: LAC with 556 MOT beams
    Freq_LACDetuning = g.FreqDetuning(Consts().LAC.FreqDetuning)
    Freq_Resonance556mj0Freq = g.Resonance556mj0Freq(Consts().Resonance556mj0Freq)
    Freq_LAC = Freq_Resonance556mj0Freq + Freq_LACDetuning
    Amp_LAC = g.Amp(Consts().LAC.Amp)
    t_LAC = g.Time(Consts().LAC.Time)

    s.add('Freq556MOTX', Freq_LAC).add('Amp556MOTX', Amp_LAC)
    s.add('Freq556RydbergMOTh', Freq_LAC).add('Amp556RydbergMOTh', Amp_LAC)
    s.wait(t_LAC)
    s.add('Freq556MOTX', Freq_LAC).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_LAC).add('Amp556RydbergMOTh', 0)
