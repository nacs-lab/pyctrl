"""PushoutDepthRampSeq.py -- 556 push-out spectroscopy AT A RAMPED TRAP DEPTH.

Purpose: calibrate VSLMservo -> trap depth directly, instead of assuming the servo is linear in
depth and extrapolating from a single anchor. Sweep ``DepthCal.VServo`` against
``Pushout.Green.Freq`` and the |mj|=1 dip position at each voltage gives the depth at that
voltage (the |mj|=1 light shift is depth-differential).

WHY THIS SEQ EXISTS. ``PushoutSurvivalSeq`` does the push-out at the LOADING depth, so it can
only ever measure the depth the array loads at. Here the trap is ramped to the test depth for
the push-out ONLY, then brought back before img2 -- so loading and imaging always happen at
``Init.VSLMServo`` where they are optimized, and only the spectroscopy sees the new depth:

    Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399 (img1)  [at Init.VSLMServo]
    << ramp VSLMservo -> DepthCal.VServo over DepthCal.RampMs >>
    Cool556 -> PushoutStep                                       [at the TEST depth]
    << ramp VSLMservo -> Init.VSLMServo over DepthCal.RampMs >>
    Imag399 (img2) -> Init                                       [back at Init.VSLMServo]

Ramps are ``ramp_to`` (quintic smootherstep, C2 at both joins) so the transfer is adiabatic:
the radial trap period is ~12 us, so even a few ms is hundreds of trap periods.

⚠ RAMP LENGTH IS CAPPED BY THE NI CARD, AND THE CAP IS PER BASIC SEQUENCE. The AO task preloads
the whole waveform into the 6738's 65,535-sample onboard FIFO, shared across the 21 Dev1 AO
channels, and an AO ramp is sampled at the full 400 kHz clock: 400 samples/ms/channel. That
leaves ~7.8 ms of TOTAL ramp time per basic sequence. BOTH ramps here are in ONE bseq, so each
must be ~3 ms -- 2 x 5 ms asked for 85,952 samples and failed with DAQmx -200341 (job 764,
2026-08-11). Contrast PPGTransportDepthCommSeq, which runs 2 x 7 ms only because its ramp-up and
ramp-down sit in DIFFERENT basic sequences and therefore get separate buffers. If longer ramps
are ever needed here, split this seq the same way rather than raising the ramp time.

⚠ THE POINT OF THE MEASUREMENT is the LOW-VOLTAGE end, where the 532 servo may sit below its
regulation floor and not deliver the commanded power (the 399 imaging PID has exactly that
pathology -- memory open-img1pid-servo-floor; PPGTransportCoolScan.py:127 flags 0-0.25 V on this
very knob). A commanded voltage is NOT a delivered depth, which is what this scan tests: if the
splitting stops tracking V below some voltage, that is the floor.

Cool556 is placed AFTER the ramp so the atoms are cooled in the trap they will be pushed in,
matching PushoutSurvivalSeq's Imag399 -> Cool556 -> Pushout ordering.
"""

from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556Step import Cool556Step
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from PushoutStep import PushoutStep
from ramp_to import ramp_to
from runtime_state import register_eom616_persistence
from SLMStep import SLMStep


def PushoutDepthRampSeq(s):
    # 616-EOM persistence: identical to PushoutSurvivalSeq (see its docstring for why the
    # literal 3.0 must be a float -- it is a SeqVal operand).
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

    # img1 at the LOADING depth (imaging W is optimized there).
    s.add_step(Imag399Step, s.C.Imag399)

    # ---- ramp to the test depth -------------------------------------------------------
    # V_normal from s.C.Init = the PATTERN-OVERLAID value (ByPattern), not raw Consts:
    # 17x17_20um sets 0.6 while the base is 3.7, so reading Consts would ramp to nearly 6x
    # the depth and would make the "no change" cell a hidden ramp. SLMStep resolves the same
    # overlaid value via its SLM.VServo cross-reference, which is what we return to.
    V_normal = s.C.Init.VSLMServo(Consts().Init.VSLMServo)
    V_test = s.C.DepthCal.VServo(V_normal)          # default = no change (matched-timing control)
    t_ramp = s.C.DepthCal.RampMs(5.0) * 1e-3

    s.add_step(t_ramp).add('VSLMservo', ramp_to(V_test))

    # Cool + push out AT THE TEST DEPTH.
    s.add_step(Cool556Step, s.C.Cool556)
    s.add_step(PushoutStep, s.C.Pushout)

    # ---- ramp back BEFORE img2, so the readout is always at the loading depth ----------
    s.add_step(t_ramp).add('VSLMservo', ramp_to(V_normal))

    s.add_step(Imag399Step, s.C.Imag399)

    s.wait(0.1)
    s.add_step(InitStep, s.C.Init)

    debug = s.C.debug(0)
    if debug:
        s.dump_output_to_file(100, 'DebugPushoutDepthRamp.seq', 'PushoutDepthRamp')

    return s
