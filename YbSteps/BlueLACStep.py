"""BlueLACStep.py -- transliteration of ``matlab_new/YbSteps/BlueLACStep.m``.

``g = s.C.LAC``. Light-assisted collisions to assure single atoms: turn off the MOT
and bias fields and wait for the cloud to fly out, set the Ryd bias coil current
(current->voltage conversion), then run a Blue LAC pulse, then a Red LAC pulse with
the 556 MOT beams.

All arithmetic here is on concrete config floats (no globals/measures), so the bare
``5``/``100`` literals stay concrete Python floats -- they never become SeqVal operands,
so no explicit-float coercion is needed. Pulse VALUES (2nd arg to ``add``) such as ``0``
are float()-coerced by the framework, so the bare ints serialize as float64.
"""

from consts import Consts


def BlueLACStep(s, g):
    # step 1: turn off MOT and wait for it to fly out
    t_Dead = g.BlueLAC.DeadTime(Consts().LAC.BlueLAC.DeadTime)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpBlueMOT', 0)
    s.add('VMOTCoil', 0)
    s.add('VBiasCoilX', 0)
    s.add('VBiasCoilY', 0)
    s.add('VBiasCoilZ', 0)

    I_RydCoil = g.BlueLAC.BiasCoilCurrent.Ryd(Consts().LAC.BlueLAC.BiasCoilCurrent.Ryd)
    V_RydCoil = 5 * I_RydCoil / 100
    s.add('VRydCoil', V_RydCoil)

    s.wait(t_Dead)

    # step 2: Blue LAC
    Freq_BlueResonance556mj0Freq = g.BlueLAC.Resonance556mj0Freq(Consts().LAC.BlueLAC.Resonance556mj0Freq)
    Freq_BlueLACDetuning = g.BlueLAC.FreqDetuning(Consts().LAC.BlueLAC.FreqDetuning)
    Freq_BlueLAC = Freq_BlueResonance556mj0Freq + Freq_BlueLACDetuning
    Amp_BlueLAC = g.BlueLAC.Amp(Consts().LAC.BlueLAC.Amp)
    t_BlueLAC = g.BlueLAC.Time(Consts().LAC.BlueLAC.Time)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq
    Freq_Cool556DetuningX = g.BlueLAC.X.FreqDetuning(Consts().LAC.BlueLAC.X.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Amp_Cool556X = g.BlueLAC.X.Amp(Consts().LAC.BlueLAC.X.Amp)

    s.add('Freq556RydbergMOTh', Freq_BlueLAC).add('Amp556RydbergMOTh', Amp_BlueLAC)
    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)

    s.wait(t_BlueLAC)

    s.add('Freq556RydbergMOTh', Freq_BlueLAC).add('Amp556RydbergMOTh', 0)
    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)

    s.wait(1e-3)

    # step 3: Red LAC with 556 MOT beams to assure single atoms
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
