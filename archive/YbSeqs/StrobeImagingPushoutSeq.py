"""StrobeImagingPushoutSeq.py -- strobe-imaging twin of ImagingPushoutSurvivalSeq.

Same skeleton as ImagingPushoutSurvivalSeq (EOM616 ramp persistence, Init->BlueMOT->SLM->GreenMOT->
LAC, Cool556hX before+after a PushouthX dose) BUT the two SURVIVAL IMAGES are StrobeImag399Step
(interleaved 399-image <-> 556-recool) instead of Imag399Step. So:

  img1 (Strobe)  -> Cool556hX -> PushouthX (dose: 0 = real survival, or long = amplify) -> Cool556hX
                 -> img2 (Strobe)

  * imaging FIDELITY comes from img1 (does strobe imaging discriminate atom vs empty?).
  * SURVIVAL comes from img1->img2 (does the atom survive strobe imaging?).

This is the seq behind the strobe-imaging optimization (StrobeImagingScan / tools/strobe_imaging_round.py),
mirroring the 2-beam imaging optimization on ImagingPushoutSurvivalSeq. Both strobe frames read the same
s.C.Imag399 config (incl. Imag399.Strobe.* timing + Imag399.Strobe.Cool556 recool tones), so a scan that
sweeps Imag399.Amp1/Amp2/FreqDetuning or Imag399.Strobe.Cool556.{X,h} moves BOTH images together.

NOTE on amplification: to amplify strobe-imaging loss, prefer raising Orca.ExposureTime (more strobe
cycles = more real strobe dose) over a long PushouthX pushout (which is CONTINUOUS, not strobed).
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from InitStep import InitStep
from LACStep import LACStep
from PushouthXStep import PushouthXStep
from ramp_to import ramp_to
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep
from StrobeImag399Step import StrobeImag399Step


def StrobeImagingPushoutSeq(s):
    # Initialising 616EOM to its old value from last run (via a sequence global) -- same as the
    # ImagingPushoutSurvivalSeq template (keeps the EOM ramp short; not serialized).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    time = abs((Freq_EOM616 - freq616global) * 20e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))
    register_eom616_persistence(s, freq616global, Freq_EOM616)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First strobe image.
    s.add_step(StrobeImag399Step, s.C.Imag399)

    # Cool556.
    s.add_step(Cool556hXStep, s.C.Cool556)

    # PushOut dose (hX variant): Pushout.Time = 0 for real survival, or long to amplify.
    s.add_step(PushouthXStep, s.C.Pushout)

    # Cool556 again.
    s.add_step(Cool556hXStep, s.C.Cool556)

    # Second strobe image (-> survival vs the strobe imaging condition).
    s.add_step(StrobeImag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    debug = s.C.debug(0)
    if debug:
        s.dump_output_to_file(100, 'DebugStrobeImagingPushout.seq', 'StrobeImagingPushout')

    return s
