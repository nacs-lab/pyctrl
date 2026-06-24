"""SLMTrapModulationSeq.py -- port of ``matlab_new/YbSeqs/SLMTrapModulationSeq.m``.

Modulates the SLM tweezer traps between two images -- the sequence behind a
parametric-heating trap-frequency measurement:

    Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 (img1)
         -> SLMTrapModulation -> Imag399 (img2) -> (wait) -> Init

Survival (img1 vs img2) dips at the parametric resonance ``f_mod = 2*f_trap``.

Server callbacks: the MATLAB ``regBeforeStart(@server_pre_run)`` /
``regAfterEnd(@server_post_run)`` become deferred no-ops here. The pyctrl run loop
attaches the per-shot frame-capture callback (``store_imgs`` / ``seq_finish``) for every
non-rearrange scan (runner.py make_capture_post_cb), so the seq need not deliver frames
itself -- same pattern as ImagingSurvivalSeq. There is intentionally NO 616-EOM ramp:
the MATLAB SLMTrapModulationSeq has none (it is not a Rydberg/clock sequence).
"""

from BlueMOTStep import BlueMOTStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep
from SLMTrapModulationStep import SLMTrapModulationStep


def _noop(s1):
    pass


def SLMTrapModulationSeq(s):
    s.reg_before_start(_noop)          # server_pre_run (deferred; no build-time effect)

    # Initialisation: 2D MOT & Zeeman slower on, shutters, AOMs/voltages to 0.
    s.add_step(InitStep, s.C.Init)

    # Blue MOT loading.
    s.add_step(BlueMOTStep, s.C.BlueMOT)

    # SLM tweezers on.
    s.add_step(SLMStep, s.C.SLM)

    # Green MOT: handover and cool down.
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    # LAC: turn off MOT, let cloud fly out, light-assisted collisions in tweezers.
    s.add_step(LACStep, s.C.LAC)

    # First Imag399 (image 1).
    s.add_step(Imag399Step, s.C.Imag399)

    # Modulate the SLM tweezer traps (parametric drive + trap-lowering flush).
    s.add_step(SLMTrapModulationStep, s.C.SLMTrapModulation)

    # Second Imag399 (image 2 -> survival vs modulation freq).
    s.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s.wait(0.2)
    s.add_step(InitStep, s.C.Init)

    s.reg_after_end(_noop)             # server_post_run (deferred; camera -> server)
    return s
