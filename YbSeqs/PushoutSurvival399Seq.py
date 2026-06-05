"""PushoutSurvival399Seq.py -- transliteration of
``matlab_new/YbSeqs/PushoutSurvival399Seq.m``.

Identical to PushoutSurvivalSeq except the pushout step is Pushout399Step. See that
module for the build-time EOM616 global ramp and the 3.0 SeqVal-operand byte note.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from Pushout399Step import Pushout399Step
from ramp_to import ramp_to
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep


def PushoutSurvival399Seq(s):
    # Initialising 616EOM to its old value from last run (via a sequence global).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    # Slow EOM ramp. 3.0 (not 3): SeqVal operand -> must be FLOAT64.
    time = abs((Freq_EOM616 - freq616global) * 20e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))

    # server_pre_run/server_post_run (MemoryMap-free): inject freq616global <- the last 616-EOM
    # frequency (persisted across shots/scans) BEFORE bc_gen, and persist this run's target
    # AFTER. Without it the ramp runs from 0 (~15 s/shot, ~60 MB bytecode). Not serialized.
    register_eom616_persistence(s, freq616global, Freq_EOM616)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)
    s.add_step(LACStep, s.C.LAC)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    # Cool556.
    s.add_step(Cool556Step, s.C.Cool556)

    # PushOut556 (and/or 308): shine 556 (and/or 308) light to push out atoms.
    s.add_step(Pushout399Step, s.C.Pushout)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    debug = s.C.debug(0)
    if debug:
        s.dump_output_to_file(100, 'DebugPushoutSurvival.seq', 'PushoutSurvival')

    return s
