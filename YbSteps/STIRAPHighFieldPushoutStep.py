"""STIRAPHighFieldPushoutStep.py -- HIGH-FIELD (50-80 A on the Ryd bias coil) variant of
``STIRAPPushoutStep``.

``g = s.C.Pushout``. Same STIRAP (two-photon 556 + 308) push-out as ``STIRAPPushoutStep``: applies
the Ryd bias field, **switches the 556 + 308 AOMs from their DDS source to the Siglent AWG**
(``TTL556RydAWGSwitch`` / ``TTL308RydAWGSwitch``), lowers the trap (``VSLMservo`` ramp to
``VRydTrap``), turns the trap fully off (``TTLSampleAndHold`` / ``AmpSLM``), fires the **forward
STIRAP** pulse (308 gate, then 556 gate, overlapped via ``STIRAP.delay``), restores the trap, pulses
the QICK microwave (``TTLQickTrig`` for ``STIRAP.gap``), optionally fires the **reverse STIRAP** (556
then 308, via ``STIRAP.reverse_delay``), ionizes (electrodes / ``TTLIonizationSwitch5to8``), and
finally restores trap depth / shutters / 616 idle and zeroes the coil + the AWG switches.

HIGH-FIELD 556 PATH (the only difference from ``STIRAPPushoutStep``; mirrors what
``RydbergHighFieldPushoutStep`` does to ``RydbergPushoutStep``). Above ~50 G the Zeeman-shifted 556
single-photon resonance is out of reach of the first double-pass AOM alone, so the beam is routed
through a SECOND, SINGLE-PASS AOM on a separate high-field arm:
  * ``Freq556RydbergHF`` = 120e6 / ``Amp556RydbergHF`` = 0.9 -- parked at the low end of its range
    and near max amp, i.e. used as a static SWITCH (+120 MHz optical), not as a tuning element.
  * ``TTL556RydbergShutter`` is held **CLOSED (0)** for the whole step -- that shutter sits in the
    LOW-field arm, which must be blocked here. The high-field arm currently has **no shutter at
    all**: its only gates are the first AOM (AWG-driven) and ``Amp556RydbergHF``.
  * The first AOM still sets frequency + amplitude, but under STIRAP its RF comes from the **AWG**,
    not the DDS. So the -60 MHz high-field offset that the DDS scans apply to
    ``Pushout.Green.Freq`` (``HF_AOM_OFFSET_MHZ``, double-pass -> half of +120 MHz) must instead be
    applied SCAN-SIDE to ``g().AWG.AWG556.Ch1/Ch2.carrier_freq_MHz``. This step cannot do it: the
    AWG carrier is out-of-band config, not part of the byte blob.

Field bands are enforced by the seq, not here: ``RearrangeSTIRAPSeq`` sends ``Bfield < 31`` to
``STIRAPPushoutStep``, ``50 <= Bfield <= 80`` here, and raises in between (same thresholds as
``RydbergPushoutSurvivalSeq``).

AWG context: the 556/308 Gaussian pulses themselves are produced by the Siglent SDG6X AWGs
(out-of-band, NOT in the byte blob). This step only drives the FPGA TTLs that (a) switch the AOM
RF source to the AWG and (b) **gate** the AWG burst (``TTL556RydAWG`` / ``TTL308RydAWG`` -- gated
external trigger). The AWG waveform for this shot is pre-stored + selected by ``AWGManager``
(``ARWV NAME`` recall on fw >= 38R3, re-upload fallback otherwise) -- see ``devices/sigilent_awg``.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``).
"""

from consts import Consts
from ramp_to import ramp_to
from devices.sigilent_awg.pulse_waveform import pulse_total_us


def STIRAPHighFieldPushoutStep(s, g):

    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)
    Amp_SLM_gap = g.SLMAOMAmpGap(Consts().SLM.AOM.Amp)  # for the trap-on gap during the STIRAP pulse
    Amp_Pushout369 = g.Amp369(0)
    Time_ionization = g.TimeIonization(0)   # auto-ionization 369 pulse width (was hardcoded 2us)
    
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
    
    Forward_PadTime = g.STIRAPPadTime(0.0)  # fwd pad time (us)
    
    IfReverse = g.IfReverse(0)  # 0: no reverse STIRAP, 1: do reverse STIRAP
    If_MW = g.IfMW(0)  # 1: fire the QICK microwave (TTLQickTrig) during the fwd->rev gap (spin
                       # echo / Ramsey / Rabi on the Rydberg state). The armed program (run loop,
                       # runp().QICK + g().QICK.*) fires on the edge and self-times; STIRAP_Gap must
                       # be >= qick_program_duration. Only valid with IfReverse (excite -> MW -> de-excite).
    Amp_AOM616Divert = Consts().AOM616Divert.Amp()
    
    IfPump = g.IfPump(0)
    IfRecoveryIonization = g.IfRecoveryIonization(0)  
    Freq_Pump556 = g.Pump556Freq(143.3e6)
    Amp_Pump556 = g.Pump556Amp(0)
    Freq_Pump616 = g.Pump616Freq(234.444e6)
    Time_Pump = g.PumpTime(0)
    Freq_EOM616 = s.C.Init.EOM616.Freq(Consts().Init.EOM616.Freq) # For init value
    
    IonizationViaDAC = g.IonizationViaDAC(0)
    T_ionization_align = g.TIonizationAlign(0.5e-6)  # wait before ionization to align the trap with the ionization pulse
    STIRAP_Gap = g.STIRAPGap(0.5)  # fwd->rev hold (us), SHARED by both beams
    SEL_C1, SEL_C2 = 0, 1  # For the channel switch

    # 2026-08-06 GATE-PULSE CONTROL. IfGatePulses = 0 fires NO AWG gate pulses at all (no
    # TTL556RydAWG / TTL308RydAWG writes, forward or reverse) and replaces the whole pulse block with a
    # single clean trap chop whose OFF WINDOW IS EXACTLY ``STIRAPGap`` -- the RNR idiom.
    #
    # Why. After matching the trap-off budget properly, STIRAP is ~1.5x hotter than RNR plus a ~4-5%
    # fixed loss: Gaussian release-recapture fits give tau 42.3 us / S0 1.014 (plain RNR), tau 41.0 /
    # 1.015 (RNR step on the SAME rearranged array, 20260806_011851 -- so the atoms and the array are
    # innocent), but tau 34.5 / S0 0.955 for the STIRAP step (20260806_012221, t = 14.4 us fixed + gap).
    # The loss is a SHORT event, not the hold: a 10x change in coil-settle time (5 -> 50 ms) moved
    # survival by only -0.024 +- 0.021 (20260806_013234), and closing the Rydberg shutter or MOTb/MOTc
    # changed nothing. What remains that is STIRAP-exclusive, short and fixed-count is the four AWG gate
    # pulses. ``amplitude_scale = 0`` scales the stored WAVEFORM, but the AWG may still have a minimum
    # output amplitude, so each gate can emit a real pulse -- and 308 has NO shutter anywhere in this
    # step (it is gated only by AmpAOM308 and the AWG), which is exactly why blocking the 556 paths did
    # nothing. With IfGatePulses = 0 every other piece of STIRAP machinery still runs (coil, shutters,
    # AWG source switching, electrodes, depth ramp, ionization block), so if tau returns to ~41 us and
    # S0 to ~1.0 the gates were the cause; if not, the remaining suspects are the S&H ordering and the
    # depth-ramp transient. Default 1 = the normal science path.
    IfGatePulses = g.IfGatePulses(1)

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
    s.add('TTLMultimeterTrig', 0)
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
    
    # Turn off the low field 556 rydberg shutter, turn off the 556 MOT shutter. 
    # The high field 556 rydberg path currently has no shutter.
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL556MOTaShutter', 0)
    s.add('TTL556MOTbShutter', 0)
    s.add('TTL556MOTcShutter', 0)
    
    # Wait until the coil current settles. Scannable since 2026-08-06 (was a hardcoded 50 ms): any
    # EXPOSURE-PROPORTIONAL loss -- leak light on any path, background collisions -- scales with this
    # wait, while a fixed cost (the trap chop, the ionization pulse) does not. Safe to shorten in a
    # dark run: an unsettled 30 G field cannot shift a resonance that is not being driven.
    t_CoilSettle = g.CoilSettleTime(50e-3)
    s.wait(t_CoilSettle)
    
    # Change trap depth for Rydberg.
    V_RydTrap = g.VRydTrap(0.4)
    s.add_step(1e-3).add('VSLMservo', ramp_to(V_RydTrap))
    s.add('Freq556RydbergHF', 120e6).add('Amp556RydbergHF', 0.9)

    s.wait(3e-3)  # wait for the ramp to finish

    # Pre-lock the 308 cavity.
    s.add('AmpAOM616', 0)
    # The second double pass AOM is default at lower end and highest amp. It's used as a switch
    s.wait(2e-6)
    
    # Preset the DDS value for pump
    #s.add('Freq556RydbergMOTh', Freq_Pump556).add('Amp556RydbergMOTh', Amp_Pump556)
    #s.add('AmpAOM308', 0.4)
    
    # Turn the tweezer off completely. 2026-07-20: ENABLED for trap-off forward-STIRAP
    s.add('TTLSampleAndHold', 0)
    s.wait(1e-6)
    #s.add('AmpSLM', 0)

    if IfGatePulses == 1:
        # --- AWG STIRAP pulse (308 gate then 556 gate, overlapped via STIRAP.delay) ---
        if Forward_Delay > 0:
            s.add('TTL308RydAWG', 1)
            s.add('TTL308RydAWG', 0)
            
            # counting when the 308 is fired, after PatTime, turn off the trap
            def _trap_off(bs):
                bs.wait(Forward_PadTime - 0.5e-6) # 0.5us for the 532 AOM fall time
                bs.add('AmpSLM', 0)
            s.add_background(_trap_off)

            s.wait(Forward_Delay)

            s.add('TTL556RydAWG', 1)
            s.add('TTL556RydAWG', 0)
            
            # Three different ways to consider Forward STIRAP as finished
            # s.wait(max(Total556Ch1, Total308Ch1 -  Forward_Delay) - (Forward_PadTime - Forward_Delay))
            # s.wait(min(Total556Ch1, Total308Ch1 -  Forward_Delay) - (Forward_PadTime - Forward_Delay))
            s.wait(Total556Ch1)
            
        else:
            raise ValueError("Forward_Delay must be > 0 for the forward STIRAP pulse sequence.")

        # --- forward STIRAP complete --- 

        if IfReverse:
            # Turn the trap back on. but with a delay < 1.5us to compensate for the 532 AOM rise time
            def _trap_on_gap(bs):
                bs.wait(1e-6)
                bs.add('AmpSLM', Amp_SLM_gap)
            #if STIRAP_Gap > 1.0e-6:
            #s.add_background(_trap_on_gap)
            
            s.wait(0.8e-6)
            s.add('AmpSLM', Amp_SLM_gap)
            #s.wait(0.5e-6)  # wait for the AOM rise time
            
            def _strobe_trap(bs):
                it = STIRAP_Gap / 1.6e-6
                for i in range(int(it)):
                    bs.wait(0.8e-6)
                    bs.add('AmpSLM', 0)
                    bs.wait(0.8e-6)
                    bs.add('AmpSLM', Amp_SLM_gap)
            
            def _AWG_ch_switch(bs):
                bs.wait(1.5e-6)
                bs.add('TTL556RydAWGChSwitch', SEL_C2).add('TTL308RydAWGChSwitch', SEL_C2)
            #s.wait(1.5e-6) # The Sigilent AWG has a hard coded 1.5us delay between the pulse output and the trigger
            #s.add('TTL556RydAWGChSwitch', SEL_C2).add('TTL308RydAWGChSwitch', SEL_C2)
            s.add_background(_AWG_ch_switch)
            #s.add_background(_strobe_trap)

            if If_MW:
                def _qick_trigger(bs):
                    bs.wait(1.5e-6)  # This is for compensation of the AOM rise time
                    bs.add('TTLQickTrig', 1)
                    bs.wait(0.05e-6) # A short trigger
                    bs.add('TTLQickTrig', 0)
                s.add_background(_qick_trigger)
                s.wait(STIRAP_Gap)
            else:
                s.wait(STIRAP_Gap)
            
            s.wait(0.8e-6) # let the MW finish before turning the trap off
            
            s.add('AmpSLM', 0) # Turn the tweezer off again for reverse STIRAP

            
            if Reverse_Delay > 0:
                # --- reverse STIRAP ---
                s.add('TTL556RydAWG', 1)
                s.add('TTL556RydAWG', 0)
                
                s.wait(Reverse_Delay)
                
                s.add('TTL308RydAWG', 1)
                s.add('TTL308RydAWG', 0)

                # Fired 556 @ t0, 308 @ t0+RD; from the 308 gate the 556 tail has (Total556Ch2 - RD)
                # left, the 308 has Total308Ch2.
                # s.wait(max(Total308Ch2, Total556Ch2 - Reverse_Delay))
                s.wait(Total308Ch2)
            else:
                s.add('TTL308RydAWG', 1)
                s.add('TTL308RydAWG', 0)

                s.wait(-Reverse_Delay)

                s.add('TTL556RydAWG', 1)
                s.add('TTL556RydAWG', 0)

                # Fired 308 @ t0, 556 @ t0-RD (RD<=0 here); from the 556 gate the 308 tail has
                # (Total308Ch2 + RD) left, the 556 has Total556Ch2.
                # s.wait(max(Total556Ch2, Total308Ch2 + Reverse_Delay))
                s.wait(Total308Ch2 + Reverse_Delay)

        if IfPump:
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
    else:
        # IfGatePulses = 0: NO gate pulses. One clean chop, trap off for EXACTLY STIRAPGap -- the same
        # idiom as ReleaseRecaptureStep, so the swept time IS the trap-off time (no pulse-window or
        # pad overhead, and no reverse chop: the reverse block below is skipped too).
        s.add('AmpSLM', 0)
        s.wait(STIRAP_Gap)
        s.add('AmpSLM', Amp_SLM)
    
    
        
    # Turn the trap back on. but with a delay < 1.5us to compensate for the 532 AOM rise time
    def _trap_on(bs):
        bs.wait(1e-6)
        bs.add('AmpSLM', Amp_SLM) # Back to the original trap depth: turn the trap back on.
        bs.wait(3e-6)
        bs.add('TTLSampleAndHold', 1)
    
    s.add_background(_trap_on) # Ionization happens at the same time as trap is turning on
    
    # If we want to do recovery ionization, we want the trap to be on, wait for the gap, then do the ionization. If we don't want to do recovery ionization, we want to do the ionization without reverse STIRAP
    if IfRecoveryIonization:
           s.wait(STIRAP_Gap)
           
    s.wait(1.5e-6) # The Sigilent AWG has a hard coded 1.5us delay between the pulse output and the trigger

    # Electrode ionization: apply the Rydberg bias field for ionization. Trigger is the pulse
    if IonizationViaDAC:
        s.add('TTLScopeTrig', 1)
        
        (s.add_step(Time_ionization)
            .add('VElectrode1', +Vx + Vy - Vz)
            .add('VElectrode2', 0 + Vy - Vz)
            .add('VElectrode3', 0 + Vy + Vz)
            .add('VElectrode4', -Vx + Vy + Vz)
            .add('VElectrode5', 0 - Vy - Vz)
            .add('VElectrode6', -Vx - Vy - Vz)
            .add('VElectrode7', +Vx - Vy + Vz)
            .add('VElectrode8', 0 - Vy + Vz))
        
        s.add('TTLScopeTrig', 0)
        
        # Restore the electrode value to zero fileld
        (s.add('VElectrode1', +Vx_init + Vy_init - Vz_init)
            .add('VElectrode2', 0 + Vy_init - Vz_init)
            .add('VElectrode3', 0 + Vy_init + Vz_init)
            .add('VElectrode4', -Vx_init + Vy_init + Vz_init)
            .add('VElectrode5', 0 - Vy_init - Vz_init)
            .add('VElectrode6', -Vx_init - Vy_init - Vz_init)
            .add('VElectrode7', +Vx_init - Vy_init + Vz_init)
            .add('VElectrode8', 0 - Vy_init + Vz_init))
    else:
        #s.wait(0.5e-6) # A small alignment
        s.wait(T_ionization_align)
        s.add('TTLScopeTrig', 1).add('TTLIonizationSwitch5to8', 1)
        s.wait(Time_ionization)
        s.add('TTLIonizationSwitch5to8', 0).add('TTLScopeTrig', 0)
    
    # auto-ionization (369 pulse width from Pushout.Time369; 0 -> zero-width pulse, no 369)
    # s.add('TTL369Switch', 1)
    # s.wait(Time_Pushout369)
    # s.add('TTL369Switch', 0)
    
    s.add('AmpAbsImag', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)
    # Close the high-field arm's ONLY gate (it has no shutter) -- matches RydbergHighFieldPushoutStep.
    s.add('Amp556RydbergHF', 0)

    s.add('AmpAOM308', 0)
    s.add('AmpAOM616', Amp_AOM616Divert)
    
    # s.add_step(Time_Pump).add("FreqEOM616", ramp_to(Freq_EOM616))

    # s.add('TTL369Shutter', 0)
    # s.add('Amp369', 0)
    
    # Ramp the tweezer up and wait.
    s.add_step(1e-3).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))
    
    # Turn on the 556 MOTa shutter, close the 556 rydberg shutter.
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL556MOTaShutter', 1)
    s.add('TTL556MOTbShutter', 1)
    s.add('TTL556MOTcShutter', 1)
    
    # Switch the AOMs back from AWG to DDS
    s.add('TTL556RydAWGSwitch', 0)
    s.add('TTL308RydAWGSwitch', 0)
    # zero the coil and wait for it to settle
    s.add('VRydCoil', 0)
    s.add('TTLMultimeterTrig', 1)
    s.wait(50e-3)
    