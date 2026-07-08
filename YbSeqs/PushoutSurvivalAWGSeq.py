"""PushoutSurvivalAWGSeq.py -- transliteration of ``matlab_new/YbSeqs/PushoutSurvivalAWGSeq.m``.

nargin-1 seq. Identical to :func:`PushoutSurvivalSeq` except the push-out step is
:func:`STIRAPPushoutStep` (two-photon 556+308 STIRAP via the Siglent AWG) instead of
``PushoutStep``. Opens with the same 616-EOM frequency ramp driven by a sequence global; the
runtime ``register_eom616_persistence`` injects the last run's 616-EOM frequency before bc_gen so
the ramp stays ~20 ms (MemoryMap-free, byte-reproducible -- the deferred callback never runs in
serialize()). Then:
Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 -> Cool556 -> STIRAPPushout -> Imag399 ->
(wait) -> Init.

AWG / QICK wiring (differs from the .m's ``server_pre_run``):
  * **Siglent AWG recall is RUNNER-SIDE, not here.** The .m called ``AWGManager.recallForSeq`` in
    ``server_pre_run``; in pyctrl the run loop wires ``AWGManager.setup``/``recall_for_seq`` from
    ``runp().AWGs`` (``YbExptCtrl/runner.py``), so this seq adds no AWG call -- the scan just lists
    ``g.runp().AWGs = ['AWG556', 'AWG308']``. ``STIRAPPushoutStep`` drives the FPGA gate TTLs.
  * **QICK microwave upload is DEFERRED** (per the migration decision -- not ported yet). The .m's
    ``server_pre_run`` also uploaded a QICK Ramsey program + ``start_program``; that is omitted. The
    step still pulses ``TTLQickTrig`` for ``STIRAP.gap``, but with no QICK program loaded the
    trigger is inert (no microwave). Wire ``FPGAAWGManager`` later if the microwave is needed.

BYTE-CRITICAL: the literal ``3.0`` in the EOM ramp time is a SeqVal operand (the expression
involves the global); a bare ``3`` would serialize as INT32 and diverge from MATLAB's FLOAT64.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from ramp_to import ramp_to
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep
from STIRAPPushoutStep import STIRAPPushoutStep


def PushoutSurvivalAWGSeq(s):
    # Initialising 616EOM to its old value from last run (via a sequence global).
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq)
    freq616global = s.new_global()
    s.C.Init.EOM616.FreqOld = freq616global
    s.add('FreqEOM616', freq616global)
    # Slow EOM ramp. 3.0 (not 3): SeqVal operand -> must be FLOAT64.
    time = abs((Freq_EOM616 - freq616global) * 2e-9 * 3.0) + 20e-3
    s.add_step(time).add('FreqEOM616', ramp_to(Freq_EOM616))

    # MemoryMap-free EOM616 persistence: inject the last 616-EOM frequency before bc_gen, persist
    # this run's target after. Not serialized.
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

    # STIRAP push-out (two-photon 556 + 308 via the AWG).
    s.add_step(STIRAPPushoutStep, s.C.Pushout)

    # Second Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    # SeqPlotter dump is gated on s.C.debug, which defaults to 0 -> never taken.
    debug = s.C.debug(0)
    if debug:
        s.dump_output_to_file(100, 'DebugPushoutSurvival.seq', 'PushoutSurvival')

    return s
