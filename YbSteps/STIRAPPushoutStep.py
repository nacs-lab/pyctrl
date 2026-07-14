"""STIRAPPushoutStep.py -- transliteration of ``matlab_new/YbSteps/STIRAPPushoutStep.m``.

``g = s.C.Pushout``. STIRAP (two-photon 556 + 308) push-out step. Applies the Ryd bias field,
**switches the 556 + 308 AOMs from their DDS source to the Siglent AWG** (``TTL556RydAWGSwitch`` /
``TTL308RydAWGSwitch``), opens the Rydberg shutters, lowers the trap (``VSLMservo`` ramp to
``VRydTrap``), turns the trap fully off (``TTLSampleAndHold``/``AmpSLM``), then fires the **forward
STIRAP** pulse (308 gate, then 556 gate, overlapped via ``STIRAP.delay``), restores the trap,
pulses the QICK microwave (``TTLQickTrig`` for ``STIRAP.gap``), optionally fires the **reverse
STIRAP** (556 then 308, via ``STIRAP.reverse_delay``), and finally restores trap depth / shutters /
616 idle and zeroes the coil + the AWG switches.

AWG context: the 556/308 Gaussian pulses themselves are produced by the Siglent SDG6X AWGs
(out-of-band, NOT in the byte blob). This step only drives the FPGA TTLs that (a) switch the AOM
RF source to the AWG and (b) **gate** the AWG burst (``TTL556RydAWG`` / ``TTL308RydAWG`` -- gated
external trigger). The AWG waveform for this shot is pre-stored + selected by ``AWGManager``
(``ARWV NAME`` recall on fw >= 38R3, re-upload fallback otherwise) -- see ``devices/sigilent_awg``.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``).

Deviations from the .m (pyctrl-only): the four ``Freq/Amp_Pushout399`` + ``Freq/Amp_Pushout556``
reads are **computed-but-unused** in the .m (the body adds literal ``0`` to ``Amp556MOTX`` /
``Amp556RydbergMOTh``, never these), and pyctrl's config has ``Pushout.Blue.Amp1/Amp2`` (not
``Blue.Amp``), so those dead reads are dropped -- zero byte effect. ``Time_Pushout369`` is now a
LIVE read (2026-07-06): it sets the auto-ionization 369 pulse width (``Pushout.Time369``, default
2e-6 = the old hardcode; the .m's own 369-pushout block stays commented out). Bare TTL ``0``/``1``
+ the ``5*I/100`` coil math mirror ``RydbergPushoutStep`` (concrete config floats -> no
explicit-float coercion needed).
"""

from consts import Consts
from ramp_to import ramp_to


def STIRAPPushoutStep(s, g):

    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)
    Amp_Pushout369 = g.Amp369(0)
    Time_Pushout369 = g.Time369(0)   # auto-ionization 369 pulse width (was hardcoded 2us)
    PulseWidth556 = s.C.AWG.AWG556.pulse_width_us(1.55)  # lobe 1/e half-width (us)
    PulseName556 = s.C.AWG.AWG556.shape("rise_gaussian")
    Time_Delay = g.TimeDelay(2.2e-6)
    # Electrode ionization
    Vx = g.Vx(0)
    Vy = g.Vy(0)
    Vz = g.Vz(0)
    Vx_init = Consts().Init.Electrodes.Vx
    Vy_init = Consts().Init.Electrodes.Vy
    Vz_init = Consts().Init.Electrodes.Vz
    
    # Ramp the tweezer down and wait; set a B-field along Z.
    I_RydCoil = g.BiasCoilCurrent.Ryd(5)
    V_RydCoil = 5 * I_RydCoil / 100
    s.add('VRydCoil', V_RydCoil)

    # Switch the 556 + 308 AOMs from their DDS source to the AWG.
    s.add('TTL556RydAWGSwitch', 1)
    s.add('TTL308RydAWGSwitch', 1)

    # Get 369 ready.
    # s.add('Amp369', Amp_Pushout369)
    # s.add('TTL369Shutter', 1)

    # Turn on the 556 rydberg shutter, close the 556 MOTa shutter.
    s.add('TTL556RydbergShutter', 1)
    s.add('TTL556MOTaShutter', 0)

    # Wait until the coil current settles.
    s.wait(50e-3)

    # Change trap depth for Rydberg.
    V_RydTrap = g.VRydTrap(0.4)
    s.add_step(1e-3).add('VSLMservo', ramp_to(V_RydTrap))
    s.wait(3e-3)  # wait for the ramp to finish

    # Pre-lock the 308 cavity.
    s.add('AmpAOM616', 0)
    s.wait(1e-6)

    # Turn the tweezer off completely.
    #s.add('TTLSampleAndHold', 0).add('AmpSLM', 0)
    #s.wait(0.5e-6)


    # --- AWG STIRAP pulse (308 gate then 556 gate, overlapped via STIRAP.delay) ---
    s.add('TTL308RydAWG', 1).add('TTL556RydAWG', 1)
    s.wait(0.1e-6)
    s.add('TTL308RydAWG', 0).add('TTL556RydAWG', 0)
    
    if PulseName556 == "double_half_gaussian_inner":
        s.wait(PulseWidth556 * 6 * 1e-6 + Time_Delay)   # wait for the 556 pulse to finish
    else:
        s.wait(PulseWidth556 * 3 * 1e-6 + Time_Delay)   # wait for the 556 pulse to finish
    
    # --- forward STIRAP complete ---
    # Turn the trap back on.
    # s.add('AmpSLM', Amp_SLM).add('TTLSampleAndHold', 1)

    # Microwave Rabi (QICK), gated for STIRAP_gap.
    #s.add('TTLQickTrig', 1)
    #s.wait(STIRAP_gap)
    #s.add('TTLQickTrig', 0)

    # Back to the original trap depth: turn the trap back on.
    #s.add('AmpSLM', Amp_SLM).add('TTLSampleAndHold', 1)
    
    # auto-ionization (369 pulse width from Pushout.Time369; 0 -> zero-width pulse, no 369)
    # s.add('TTL369Switch', 1)
    
    # Electrode ionization: apply the Rydberg bias field for ionization. Trigger is the pulse
    (s.add_step(Time_Pushout369)
        .add('TTLScopeTrig', 1)
        .add('VElectrode1', +Vx + Vy - Vz)
        .add('VElectrode2', 0 + Vy - Vz)
        .add('VElectrode3', 0 + Vy + Vz)
        .add('VElectrode4', -Vx + Vy + Vz)
        .add('VElectrode5', 0 - Vy - Vz)
        .add('VElectrode6', -Vx - Vy - Vz)
        .add('VElectrode7', +Vx - Vy + Vz)
        .add('VElectrode8', 0 - Vy + Vz))

    # s.wait(Time_Pushout369)
    # s.add('TTL369Switch', 0)
    
    # Restore the electrode value to zero fileld
    (s.add_step(4e-6)
        .add('TTLScopeTrig', 0)
        .add('VElectrode1', +Vx_init + Vy_init - Vz_init)
        .add('VElectrode2', 0 + Vy_init - Vz_init)
        .add('VElectrode3', 0 + Vy_init + Vz_init)
        .add('VElectrode4', -Vx_init + Vy_init + Vz_init)
        .add('VElectrode5', 0 - Vy_init - Vz_init)
        .add('VElectrode6', -Vx_init - Vy_init - Vz_init)
        .add('VElectrode7', +Vx_init - Vy_init + Vz_init)
        .add('VElectrode8', 0 - Vy_init + Vz_init))
    

    s.add('AmpAbsImag', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpAOM308', 0)
    s.add('AmpAOM616', 0.12)
    # s.add('TTL369Shutter', 0)
    # s.add('Amp369', 0)

    # Ramp the tweezer up and wait.
    s.add_step(1e-3).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))

    # Switch the AOMs back from AWG to DDS, zero the coil.
    s.add('TTL556RydAWGSwitch', 0)
    s.add('TTL308RydAWGSwitch', 0)
    s.add('VRydCoil', 0)
    s.wait(50e-3)
