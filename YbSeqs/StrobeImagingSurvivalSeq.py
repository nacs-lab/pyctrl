"""StrobeImagingSurvivalSeq.py -- strobe-imaging variant of ImagingPushoutSurvivalSeq.

Identical structure to ImagingPushoutSurvivalSeq (image -> cool -> [push-out slot] -> cool ->
image; NumImages=2 => survival), except the middle "push-out" step is ``StrobePushouthXStep``:
a controllable strobe-imaging pulse (the two 399 imaging beams + the 556 X/h cooling) whose
parameters default to the imaging/cooling config but are scannable via the ``Pushout.*`` params.
Used by StrobeImageScan. See PushoutSurvivalSeq for the build-time EOM616 global ramp / 3.0 note.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from ramp_to import ramp_to
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep
from StrobePushouthXStep import StrobePushouthXStep


def StrobeImagingSurvivalSeq(s):
    # Initialising 616EOM to its old value from last run (via a sequence global).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    # Slow EOM ramp. 3.0 (not 3): SeqVal operand -> must be FLOAT64.
    time = abs((Freq_EOM616 - freq616global) * 20e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))

    register_eom616_persistence(s, freq616global, Freq_EOM616)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    # Cool556.
    s.add_step(Cool556hXStep, s.C.Cool556)

    # Strobe-imaging pulse (occupies the push-out slot; reads s.C.Pushout).
    s.add_step(StrobePushouthXStep, s.C.Pushout)

    # Cool556 again.
    s.add_step(Cool556hXStep, s.C.Cool556)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    return s
