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
from devices.sigilent_awg.pulse_waveform import pulse_total_us


def STIRAPPushoutStep(s, g):

    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)
    Amp_Pushout369 = g.Amp369(0)
    Time_Pushout369 = g.Time369(0)   # auto-ionization 369 pulse width (was hardcoded 2us)
    
    # Full AWG playback window (s) per channel -- SHAPE-dependent, NOT always 3*pw. The gate must
    # stay high for the whole waveform: 3*pw for the half-Gaussian, but only pw for the cubic/
    # quintic splines (compact support), and pw + pad_time_us for a fall_quintic with a flat
    # pre-hold. Was a hardcoded ``PulseWidth*3``; now pulse_total_us(). ``pad_time_us`` must be
    # threaded in or the gate cuts the fall_quintic flat hold (pulse_total_us ignores it for the
    # other shapes). FORWARD fires on Ch1 (556 + 308), REVERSE on Ch2 -- each post-pulse settle
    # wait must cover BOTH beams of that stage (they overlap by the STIRAP delay), so all four
    # channel totals are needed. Each = the full playback window (shape-dependent).
    Total556Ch1 = pulse_total_us(s.C.AWG.AWG556.Ch1.shape("rise_quintic"),
                                 s.C.AWG.AWG556.Ch1.pulse_width_us(1.55),
                                 pad_time_us=s.C.AWG.AWG556.Ch1.pad_time_us(0)) * 1e-6
    Total308Ch1 = pulse_total_us(s.C.AWG.AWG308.Ch1.shape("rise_quintic"),
                                 s.C.AWG.AWG308.Ch1.pulse_width_us(1.55),
                                 pad_time_us=s.C.AWG.AWG308.Ch1.pad_time_us(0)) * 1e-6
    Total556Ch2 = pulse_total_us(s.C.AWG.AWG556.Ch2.shape("fall_quintic"),
                                 s.C.AWG.AWG556.Ch2.pulse_width_us(1.55),
                                 pad_time_us=s.C.AWG.AWG556.Ch2.pad_time_us(0)) * 1e-6
    Total308Ch2 = pulse_total_us(s.C.AWG.AWG308.Ch2.shape("fall_quintic"),
                                 s.C.AWG.AWG308.Ch2.pulse_width_us(1.55),
                                 pad_time_us=s.C.AWG.AWG308.Ch2.pad_time_us(0)) * 1e-6
    Forward_Delay = g.STIRAPDelay(0.5)  # fwd delay (us)
    Reverse_Delay = g.STIRAPReverseDelay(0.5)  # rev delay (us)
    IfReverse = g.IfReverse(0)  # 0: no reverse STIRAP, 1: do reverse STIRAP
    If_MW = g.IfMW(0)  # 1: fire the QICK microwave (TTLQickTrig) during the fwd->rev gap (spin
                       # echo / Ramsey / Rabi on the Rydberg state). The armed program (run loop,
                       # runp().QICK + g().QICK.*) fires on the edge and self-times; STIRAP_Gap must
                       # be >= qick_program_duration. Only valid with IfReverse (excite -> MW -> de-excite).
    Amp_AOM616Divert = Consts().AOM616Divert.Amp()
    
    IfPump = g.IfPump(0)
    Freq_Pump556 = g.Pump556Freq(143.3e6)
    Amp_Pump556 = g.Pump556Amp(0)
    Freq_Pump616 = g.Pump616Freq(234.444e6)
    Time_Pump = g.PumpTime(0)
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq) # For init value
    
    STIRAP_Gap = g.STIRAPGap(0.5)  # fwd->rev hold (us), SHARED by both beams
    SEL_C1, SEL_C2 = 0, 1  # For the channel switch 

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
    
    # Switch the 556 + 308 AWG channels to Ch1
    s.add('TTL556RydAWGChSwitch', SEL_C1)
    s.add('TTL308RydAWGChSwitch', SEL_C1)
    
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
    s.wait(3e-6)
    
    # Preset the DDS value for pump
    #s.add('Freq556RydbergMOTh', Freq_Pump556).add('Amp556RydbergMOTh', Amp_Pump556)
    #s.add('AmpAOM308', 0.4)
    
    # Turn the tweezer off completely. 2026-07-20: ENABLED for trap-off forward-STIRAP
    # optimization (removes trap light shift on the Rydberg transition during the pulse).
    #s.add('TTLSampleAndHold', 0).add('AmpSLM', 0)
    #s.wait(3e-6)

    # --- AWG STIRAP pulse (308 gate then 556 gate, overlapped via STIRAP.delay) ---
    if Forward_Delay > 0:
        s.add('TTL308RydAWG', 1)
        s.add('TTL308RydAWG', 0)
        
        s.wait(Forward_Delay)
        
        s.add('TTL556RydAWG', 1)
        s.add('TTL556RydAWG', 0)
        
        # Wait until BOTH forward pulses finish before switching to Ch2. Fired 308 @ t0, 556 @
        # t0+FD; from the 556 gate the 308 tail has (Total308Ch1 - FD) left, the 556 has Total556Ch1.
        s.wait(max(Total556Ch1, Total308Ch1 - Forward_Delay))
    else:
        s.add('TTL556RydAWG', 1)
        s.add('TTL556RydAWG', 0)

        s.wait(-Forward_Delay)

        s.add('TTL308RydAWG', 1)
        s.add('TTL308RydAWG', 0)

        # Fired 556 @ t0, 308 @ t0-FD (FD<=0 here); from the 308 gate the 556 tail has
        # (Total556Ch1 + FD) left, the 308 has Total308Ch1.
        s.wait(max(Total308Ch1, Total556Ch1 + Forward_Delay))
    
    
    # --- forward STIRAP complete ---
    
    # Turn the trap back on. 2026-07-20: ENABLED to pair with the trap-off block above.
    #s.add('AmpSLM', Amp_SLM).add('TTLSampleAndHold', 1)
    #s.wait(0.1e-6)

    # (QICK microwave now fires inside the fwd->rev gap below, gated on If_MW.)

    if IfReverse:
        s.add('TTL556RydAWGChSwitch', SEL_C2)
        s.add('TTL308RydAWGChSwitch', SEL_C2)
        # fwd->rev gap. When If_MW, pulse TTLQickTrig across it so the armed QICK program (spin
        # echo / Ramsey / Rabi) plays on the Rydberg state between forward and reverse STIRAP; the
        # board self-times, so STIRAP_Gap must be >= qick_program_duration. If_MW == 0 -> the bare
        # wait, byte-identical to before.
        if If_MW:
            s.add('TTLQickTrig', 1)
            s.wait(STIRAP_Gap)
            s.add('TTLQickTrig', 0)
        else:
            s.wait(STIRAP_Gap)
        
        # Turn the tweezer off completely.
        #s.add('TTLSampleAndHold', 0).add('AmpSLM', 0)
        #s.wait(0.5e-6)

        
        if Reverse_Delay > 0:
            # --- reverse STIRAP ---
            s.add('TTL556RydAWG', 1)
            s.add('TTL556RydAWG', 0)
            
            s.wait(Reverse_Delay)
            
            s.add('TTL308RydAWG', 1)
            s.add('TTL308RydAWG', 0)

            # Fired 556 @ t0, 308 @ t0+RD; from the 308 gate the 556 tail has (Total556Ch2 - RD)
            # left, the 308 has Total308Ch2.
            s.wait(max(Total308Ch2, Total556Ch2 - Reverse_Delay))
        else:
            s.add('TTL308RydAWG', 1)
            s.add('TTL308RydAWG', 0)

            s.wait(-Reverse_Delay)

            s.add('TTL556RydAWG', 1)
            s.add('TTL556RydAWG', 0)

            # Fired 308 @ t0, 556 @ t0-RD (RD<=0 here); from the 556 gate the 308 tail has
            # (Total308Ch2 + RD) left, the 556 has Total556Ch2.
            s.wait(max(Total556Ch2, Total308Ch2 + Reverse_Delay))
            
    elif IfPump:
        # --- incoherent pumping rydberg states down to ground states ---
        # Switch the 556 + 308 AOMs from their DDS source to the AWG.
        s.wait(STIRAP_Gap)
        
        # Ramp EOM616 freq in the background across the whole switch-on window.
        (s.add_step(Time_Pump)
            .add('TTL556RydAWGSwitch', 0)
            .add('TTL308RydAWGSwitch', 0))
        s.add('TTL556RydAWGSwitch', 1)
        s.add('TTL308RydAWGSwitch', 1)
        #s.add('Amp556RydbergMOTh', 0)
        #s.add('AmpAOM308', 0)
        
        
    # Back to the original trap depth: turn the trap back on.
    s.add('AmpSLM', Amp_SLM).add('TTLSampleAndHold', 1)
    
    # auto-ionization (369 pulse width from Pushout.Time369; 0 -> zero-width pulse, no 369)
    # s.add('TTL369Switch', 1)
    
    # Electrode ionization: apply the Rydberg bias field for ionization. Trigger is the pulse
    s.wait(0.1e-6)
    s.add('TTLScopeTrig', 1)
    (s.add_step(Time_Pushout369)
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
    s.add('TTLScopeTrig', 0)
    (s.add('VElectrode1', +Vx_init + Vy_init - Vz_init)
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
    s.add('AmpAOM616', Amp_AOM616Divert)
    s.add_step(Time_Pump).add("FreqEOM616", ramp_to(Freq_EOM616))

    # s.add('TTL369Shutter', 0)
    # s.add('Amp369', 0)

    # Ramp the tweezer up and wait.
    s.add_step(1e-3).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))

    # Switch the AOMs back from AWG to DDS, zero the coil.
    s.add('TTL556RydAWGSwitch', 0)
    s.add('TTL308RydAWGSwitch', 0)
    s.add('VRydCoil', 0)
    s.wait(50e-3)
