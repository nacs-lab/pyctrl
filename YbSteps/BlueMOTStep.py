"""BlueMOTStep.py -- transliteration of ``matlab_new/YbSteps/BlueMOTStep.m``.

``g = s.C.BlueMOT``. Sets the magnetic field for the blue MOT (MOT coil +
bias coils, current->voltage conversion), turns on the blue MOT beams, then
holds for the loading time.

All arithmetic here is on concrete config floats (no globals/measures), so plain
int literals like ``200``/``30``/``5`` stay concrete Python floats -- they never
become SeqVal operands, so no explicit-float coercion is needed.
"""

from consts import Consts


def BlueMOTStep(s, g):
    # MOT coil current generating the anti-Helmholtz field
    BGradient = g.BFieldGradient(Consts().BlueMOT.BFieldGradient)
    I_MOTCoil = BGradient * 200 / 30

    # Bias coils current
    I_RydCoil = g.BiasCoilCurrent.Ryd(Consts().BlueMOT.BiasCoilCurrent.Ryd)
    I_BiasCoilX = g.BiasCoilCurrent.X(Consts().BlueMOT.BiasCoilCurrent.X)
    I_BiasCoilY = g.BiasCoilCurrent.Y(Consts().BlueMOT.BiasCoilCurrent.Y)
    I_BiasCoilZ = g.BiasCoilCurrent.Z(Consts().BlueMOT.BiasCoilCurrent.Z)
    Img1Amp = g.Img1PIDSet(Consts().BlueMOT.Img1PIDSet)
    Img2Amp = g.Img2PIDSet(Consts().BlueMOT.Img2PIDSet)
    
    # Convert current to control voltage
    V_MOTCoil = 5 * I_MOTCoil / 200
    V_RydCoil = 5 * I_RydCoil / 100
    V_BiasCoilX = 10 * I_BiasCoilX / 7
    V_BiasCoilY = 10 * I_BiasCoilY / 7
    V_BiasCoilZ = 10 * I_BiasCoilZ / 7

    # Set magnetic field for Blue MOT
    s.add('VMOTCoil', V_MOTCoil)
    s.add('VRydCoil', V_RydCoil)
    s.add('VBiasCoilX', V_BiasCoilX)
    s.add('VBiasCoilY', V_BiasCoilY)
    s.add('VBiasCoilZ', V_BiasCoilZ)
    
    #Set the 399 imaging parameters for the 399 Img1/Img2 PID lock
    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = Consts().Imag399.FreqDetuning
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    
    #Turn Imaging beam on for 399 Img1/Img2 PID lock
    (s.add('FreqAbsImag', Freq_Imag399)
        .add('Freq399Imag2', Freq_Imag399)
        .add('AmpAbsImag', 1) 
        .add('Amp399Imag2', 1)
        .add('TTL399AbsImagShutter', 1) # Getting ready for 399 Img1 PID
        .add('TTL399Imag2Shutter', 1) # Getting ready for 399 Img2 PID
        .add('TTL399IMG1PIDMode', 1) # Starting lock for 399 Img1 PID
        .add('TTL399IMG2PIDMode', 1) # Starting lock for 399 Img2 PID
        .add('VImg1PIDSet', Img1Amp) # Setting 399 Img1 PID
        .add('VImg2PIDSet', Img2Amp)) # Setting 399 Img2 PID
        
    # turn on blue MOT beams
    Freq_BlueMOTDetuning = g.FreqDetuning(Consts().BlueMOT.FreqDetuning)
    Freq_Resonance399Freq = g.Resonance399Freq(Consts().Resonance399Freq)
    Freq_BlueMOT = Freq_Resonance399Freq + Freq_BlueMOTDetuning
    Amp_BlueMOT = g.Amp(Consts().BlueMOT.Amp)
    s.add('FreqBlueMOT', Freq_BlueMOT).add('AmpBlueMOT', Amp_BlueMOT)

    # Proceed the CurTime to after LoadingTime
    t_BlueMOTLoading = g.LoadingTime(Consts().BlueMOT.LoadingTime)
    s.wait(t_BlueMOTLoading)
    
    s.add('TTL399IMG1PIDMode', 0) # Turning off lock for 399 Img1 PID
    s.add('TTL399IMG2PIDMode', 0) # Turning off lock for 399 Img2 PID
    
    
    
    (s.add('AmpAbsImag', 0) # Shutting down 399 Img1 PID
    .add('Amp399Imag2', 0) # Shutting down 399 Img2 PID
    .add('TTL399AbsImagShutter', 0) # Shutting down 399 Img1 PID
    .add('TTL399Imag2Shutter', 0)) # Shutting down 399 Img2 PID
     