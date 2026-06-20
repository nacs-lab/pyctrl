"""ImagingNumImgSeq.py -- sequence for scanning the number of imaging exposures.

Structure:
    Init -> BlueMOT -> SLM -> GreenMOT -> LAC
    -> Imag399Step (image 1, triggered)            -- confirm atom loaded
    -> N x Imag399heatingStep (no trigger)         -- "dark" exposures: 399 light, no camera
    -> Imag399Step (image 2, triggered)            -- check survival after N exposures
    -> wait -> Init

N is read from ``s.C.Imag399heating.NHeating`` at build time (a Python int), so each
scan point with a different NHeating compiles to a distinct sequence with a different
number of heating steps. NumImages = 2 per shot (two camera triggers).

Config groups:
    s.C.Imag399         -- settings for the two triggered imaging steps
    s.C.Imag399heating  -- settings for the dark heating steps + NHeating loop count
"""

from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from Imag399heatingStep import Imag399heatingStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep


def ImagingNumImgSeq(s):
    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399: confirm atom is loaded.
    s.add_step(Imag399Step, s.C.Imag399)

    # N dark exposures (399 light + 556 cooling, no camera trigger).
    # NHeating is a build-time integer that controls the number of steps.
    n_heating = round(s.C.Imag399heating.NHeating(0))
    for _ in range(n_heating):
        s.add_step(Imag399heatingStep, s.C.Imag399heating)

    # Second Imag399: check survival after the dark exposures.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    return s
