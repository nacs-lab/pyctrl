"""CoreShellMOTStep.py -- transliteration of ``matlab_new/YbSteps/CoreShellMOTStep.m``.

``g = s.C.CoreShellMOT``. Switches magnetic fields (MOT + bias coils, current->voltage
conversion), turns on blue + green MOT beams, holds, then cools by ramping power and
detuning. Uses ``ramp_to`` (``rampTo``) for the field switch and the cool-down ramp.

All arithmetic here is on concrete config floats (no globals/measures), so plain int
literals like ``200``/``30``/``5`` stay concrete Python floats -- they never become
SeqVal operands, so no explicit-float coercion is needed.
"""

from consts import Consts
from ramp_to import ramp_to


def CoreShellMOTStep(s, g):
    # step 0: Switch magnetic fields
    # MOT coil current generating the anti-Helmholtz field
    BGradient = g.BFieldGradient(Consts().GreenMOT.BFieldGradient)
    I_MOTCoil = BGradient * 200 / 30

    # Bias coils current
    I_RydCoil = g.BiasCoilCurrent.Ryd(Consts().GreenMOT.BiasCoilCurrent.Ryd)
    I_BiasCoilX = g.BiasCoilCurrent.X(Consts().GreenMOT.BiasCoilCurrent.X)
    I_BiasCoilY = g.BiasCoilCurrent.Y(Consts().GreenMOT.BiasCoilCurrent.Y)
    I_BiasCoilZ = g.BiasCoilCurrent.Z(Consts().GreenMOT.BiasCoilCurrent.Z)

    # Convert current to control voltage
    V_MOTCoil = 5 * I_MOTCoil / 200
    V_RydCoil = 5 * I_RydCoil / 100
    V_BiasCoilX = 10 * I_BiasCoilX / 7
    V_BiasCoilY = 10 * I_BiasCoilY / 7
    V_BiasCoilZ = 10 * I_BiasCoilZ / 7

    # Switch magnetic field
    t_BFieldRamp = g.BFieldRampTime(Consts().GreenMOT.BFieldRampTime)
    (s.add_step(t_BFieldRamp)
        .add('VMOTCoil', ramp_to(V_MOTCoil))
        .add('VRydCoil', ramp_to(V_RydCoil))
        .add('VBiasCoilX', ramp_to(V_BiasCoilX))
        .add('VBiasCoilY', ramp_to(V_BiasCoilY))
        .add('VBiasCoilZ', ramp_to(V_BiasCoilZ)))

    # step 1: turn on blue MOT beams
    Freq_BlueMOTDetuning = g.BlueMOT.FreqDetuning(Consts().BlueMOT.FreqDetuning)
    Freq_Resonance399Freq = g.Resonance399Freq(Consts().Resonance399Freq)
    Freq_BlueMOT = Freq_Resonance399Freq + Freq_BlueMOTDetuning
    Amp_BlueMOT = g.BlueMOT.Amp(Consts().BlueMOT.Amp)
    s.add('FreqBlueMOT', Freq_BlueMOT).add('AmpBlueMOT', Amp_BlueMOT)

    # step 2: Turn on green MOT beams
    Freq_GreenMOTDetuning_PB = g.GreenMOT.PowerBroaden.FreqDetuning(
        Consts().GreenMOT.PowerBroaden.FreqDetuning)
    Freq_Resonance556mj0Freq = g.Resonance556mj0Freq(Consts().Resonance556mj0Freq)
    Freq_GreenMOT_PB = Freq_Resonance556mj0Freq + Freq_GreenMOTDetuning_PB
    Amp_GreenMOT_PB = g.GreenMOT.PowerBroaden.Amp(Consts().GreenMOT.PowerBroaden.Amp)
    s.add('Freq556MOTX', Freq_GreenMOT_PB).add('Amp556MOTX', Amp_GreenMOT_PB)
    s.add('Freq556RydbergMOTh', Freq_GreenMOT_PB).add('Amp556RydbergMOTh', Amp_GreenMOT_PB)

    t_MOTLoading = g.LoadingTime(Consts().BlueMOT.LoadingTime)
    s.wait(t_MOTLoading)

    # step 3: Turn off the blue MOT beams
    s.add('AmpBlueMOT', 0)
    # and the 2D MOT shutter to close the atomic flux
    s.add('TTL3992DMOTShutter', 0)
    s.add('Amp2DMOT', 0)

    # let the power broaden green MOT stabilize
    t_Handover = g.GreenMOT.PowerBroaden.HandoverTime(
        Consts().GreenMOT.PowerBroaden.HandoverTime)
    s.wait(t_Handover)

    # step 4: Cool down by ramping down the power and detuning
    t_Rampdown = g.GreenMOT.CoolDown.RampdownTime(Consts().GreenMOT.CoolDown.RampdownTime)
    Freq_GreenMOTDetuning_CD = g.GreenMOT.CoolDown.FreqDetuning(
        Consts().GreenMOT.CoolDown.FreqDetuning)
    Freq_GreenMOT_CD = Freq_Resonance556mj0Freq + Freq_GreenMOTDetuning_CD
    Amp_GreenMOT_CD = g.GreenMOT.CoolDown.Amp(Consts().GreenMOT.CoolDown.Amp)

    (s.add_step(t_Rampdown)
        .add('Freq556MOTX', ramp_to(Freq_GreenMOT_CD))
        .add('Amp556MOTX', ramp_to(Amp_GreenMOT_CD))
        .add('Freq556RydbergMOTh', ramp_to(Freq_GreenMOT_CD))
        .add('Amp556RydbergMOTh', ramp_to(Amp_GreenMOT_CD)))

    t_Hold = g.GreenMOT.CoolDown.HoldTime(Consts().GreenMOT.CoolDown.HoldTime)
    s.wait(t_Hold)
