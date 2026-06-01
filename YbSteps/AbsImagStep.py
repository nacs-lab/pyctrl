"""AbsImagStep.py -- transliteration of ``matlab_new/YbSteps/AbsImagStep.m``.

``g = s.C.AbsImag``. 399nm absorption imaging on the 556 MOT in free space: open the
imaging shutter, turn off the MOT, wait for time-of-flight, then take three exposures
(image / background / dark) with ThorCam triggers.
"""

from consts import Consts


def AbsImagStep(s, g):
    s.add('TTL399AbsImagShutter', 1)
    s.wait(5e-3)  # wait for the shutter to be fully open

    # step 1: turn off the MOT and wait for TOF
    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)
    s.add('AmpBlueMOT', 0)
    s.add('VMOTCoil', 0)
    s.add('VRydCoil', 0)
    s.add('VBiasCoilX', 0)
    s.add('VBiasCoilY', 0)
    s.add('VBiasCoilZ', 0)

    t_TOF = g.TOF(Consts().AbsImag.TOF)
    s.wait(t_TOF)

    # step 2: take 3 imagings
    t_Exposure = g.ExposureTime(Consts().AbsImag.ExposureTime)
    t_BetweenImags = g.BetweenImagsTime(Consts().AbsImag.BetweenImagsTime)
    Freq_AbsImag = g.Freq(Consts().AbsImag.Freq)
    Amp_AbsImag = g.Amp(Consts().AbsImag.Amp)

    # Abs Imaging
    s.add('AmpAbsImag', Amp_AbsImag).add('FreqAbsImag', Freq_AbsImag)
    s.add_step(t_Exposure).add('TTLThorCamTrig', 1)
    s.add('TTLThorCamTrig', 0)
    s.add('AmpAbsImag', 0)

    s.wait(t_BetweenImags)

    # Abs Imaging background
    s.add('AmpAbsImag', Amp_AbsImag)
    s.add_step(t_Exposure).add('TTLThorCamTrig', 1)
    s.add('TTLThorCamTrig', 0)
    s.add('AmpAbsImag', 0)

    s.wait(t_BetweenImags)

    # Dark Image
    s.add_step(t_Exposure).add('TTLThorCamTrig', 1)
    s.add('TTLThorCamTrig', 0)

    s.add('TTL399AbsImagShutter', 0)
    s.wait(3e-3)  # Just to proceed the CurTime
