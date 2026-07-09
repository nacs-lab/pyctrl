"""expConfig.py -- executable, actively-maintained twin of ``matlab_new/expConfig.m``.

pyctrl's PRODUCTION config source: channel aliases, channel default values, physical constants,
and AWG/Orca defaults. This replaces the frozen ``tests/reference/config_reference.json`` SNAPSHOT
for the live runner (the snapshot is demoted to the MATLAB-ground-truth reference for the drift
oracle), curing the silent staleness of the captured config -- the config is now CODE.

expConfig.py is pyctrl's live, executable config source -- the runner reloads it each job, and
calibration edits (e.g. the daily ``Resonance556mj0Freq`` update) land here directly. Its values
feed the serialize path, so ``tests/test_exp_config.py`` (the drift oracle) asserts this module
still resolves to the SAME config as the frozen MATLAB capture
(``tools/capture_config_reference.m``) -- a regression guard, not a live-MATLAB tracking
requirement. ``matlab_new/expConfig.m`` is the reference the snapshot came from; if a scan still
runs under MATLAB keep the change in sync there too, and re-capture the snapshot when you
deliberately change a value.

:func:`build_config` returns the raw dict ``SeqConfig`` consumes (same shape
``capture_config_reference.m`` emits). The long-lived runner reloads this module once per job
(``SeqConfig.load_real(reload=True)``), so edits go live without a restart -- only ``lib/`` (the
framework) and the expConfig snapshot capture still need a restart / re-capture.

Naming/units follow ``expConfig.m``: alias prefixes are load-bearing (``TTL*`` FPGA, ``V*`` NI DAQ,
``Freq*``/``Amp*`` DDS); defaults are Hz / s. Values mirror the ``.m`` literal forms for readability
(they resolve to the same IEEE-754 doubles); ints are float-coerced downstream by ``SeqConfig``.

The per-pattern overlay + const cross-ref logic lives in ``lib/expConfig_helper.py`` (this
module stays the pure data/config source); ``_consts`` calls it for the cross-refs.
"""

import expConfig_helper


def build_config():
    """Return the raw config dict (aliases / consts / defaults / NI wiring) for ``SeqConfig``."""
    channel_alias = _channel_alias()
    consts = _consts()
    default_vals = _default_vals(consts)
    return {
        "channel_alias_keys": list(channel_alias.keys()),
        "channel_alias_vals": list(channel_alias.values()),
        "consts": consts,
        "default_vals_keys": list(default_vals.keys()),
        "default_vals_vals": list(default_vals.values()),
        # NI DAQ external clock + start-trigger wiring (FPGA clock -> Dev1/PFI0;
        # FPGA TTL0 -> Dev1/PFI1). Run-loop NI arm only; omitted from the byte path.
        "ni_clocks_keys": ["Dev1"], "ni_clocks_vals": ["PFI0"],
        "ni_start_keys": ["Dev1"], "ni_start_vals": ["PFI1"],
    }


def _channel_alias():
    """Channel name -> backend path (``expConfig.m`` ``channelAlias``)."""
    a = {}
    a["Dev1"] = "NiDAQ/Dev1"                    # NIDAQ backend convention

    # ---- TTL channels (FPGA1) ----
    a["TTL556RydAWG"] = "FPGA1/TTL1"
    a["TTL399IMG1PIDMode"] = "FPGA1/TTL2"  # 1 for lock, 0 for hold
    a["TTLScopeTrig"] = "FPGA1/TTL3"
    a["TTL556RydbergShutter"] = "FPGA1/TTL4"
    a["TTL556MOTaShutter"] = "FPGA1/TTL5"
    a["TTL556MOTbShutter"] = "FPGA1/TTL6"
    a["TTL556MOTcShutter"] = "FPGA1/TTL7"
    a["TTL399AbsImagShutter"] = "FPGA1/TTL8"
    a["TTL399MOTShutter"] = "FPGA1/TTL9"
    a["TTL3992DMOTShutter"] = "FPGA1/TTL10"
    a["TTL369Shutter"] = "FPGA1/TTL11"
    a["TTL308RydAWG"] = "FPGA1/TTL12"
    a["TTLThorCamTrig"] = "FPGA1/TTL13"
    a["TTLQickTrig"] = "FPGA1/TTL14"
    a["TTL369Switch"] = "FPGA1/TTL16"
    a["TTL556RydAWGSwitch"] = "FPGA1/TTL17"
    a["TTL399Imag2Shutter"] = "FPGA1/TTL18"
    a["TTL399IMG2PIDMode"] = "FPGA1/TTL19"  # 1 for lock, 0 for hold
    a["TTL308RydAWGSwitch"] = "FPGA1/TTL55"
    a["TTLOrcaTrig"] = "FPGA1/TTL54"
    a["TTLSampleAndHold"] = "FPGA1/TTL15"
    a["TTLTrig"] = "FPGA1/TTL3"

    # ---- DDS channels (FPGA1) -- Name/FREQ and Name/AMP, no prefix ----
    a["Freq556RydbergMOTh"] = "FPGA1/DDS0/FREQ"
    a["Amp556RydbergMOTh"] = "FPGA1/DDS0/AMP"
    a["Freq556MOTX"] = "FPGA1/DDS1/FREQ"
    a["Amp556MOTX"] = "FPGA1/DDS1/AMP"
    a["FreqSLM"] = "FPGA1/DDS2/FREQ"
    a["AmpSLM"] = "FPGA1/DDS2/AMP"
    a["FreqAODs"] = "FPGA1/DDS3/FREQ"
    a["AmpAODs"] = "FPGA1/DDS3/AMP"
    a["FreqAOM308"] = "FPGA1/DDS4/FREQ"
    a["AmpAOM308"] = "FPGA1/DDS4/AMP"
    a["FreqAODv"] = "FPGA1/DDS5/FREQ"
    a["AmpAODv"] = "FPGA1/DDS5/AMP"
    a["FreqSLMmodulation"] = "FPGA1/DDS6/FREQ"
    a["AmpSLMmodulation"] = "FPGA1/DDS6/AMP"
    a["FreqEOM616"] = "FPGA1/DDS7/FREQ"
    a["AmpEOM616"] = "FPGA1/DDS7/AMP"
    a["FreqAOM616"] = "FPGA1/DDS8/FREQ"
    a["AmpAOM616"] = "FPGA1/DDS8/AMP"
    a["Freq369"] = "FPGA1/DDS12/FREQ"
    a["Amp369"] = "FPGA1/DDS12/AMP"
    a["Freq399Imag2"] = "FPGA1/DDS17/FREQ"
    a["Amp399Imag2"] = "FPGA1/DDS17/AMP"
    a["FreqAbsImag"] = "FPGA1/DDS18/FREQ"
    a["AmpAbsImag"] = "FPGA1/DDS18/AMP"
    a["FreqBlueMOT"] = "FPGA1/DDS19/FREQ"
    a["AmpBlueMOT"] = "FPGA1/DDS19/AMP"
    a["FreqZeeman"] = "FPGA1/DDS20/FREQ"
    a["AmpZeeman"] = "FPGA1/DDS20/AMP"
    a["Freq2DMOT"] = "FPGA1/DDS21/FREQ"
    a["Amp2DMOT"] = "FPGA1/DDS21/AMP"

    # ---- NI DAQ voltages (Dev1) ----
    a["VMOTCoil"] = "Dev1/0"
    a["VBiasCoilY"] = "Dev1/1"
    a["VBiasCoilX"] = "Dev1/3"
    a["VBiasCoilZ"] = "Dev1/4"
    a["VRydCoil"] = "Dev1/6"
    a["VSLMservo"] = "Dev1/8"
    a["VPicoMotor308h"] = "Dev1/10"
    a["VPicoMotor308v"] = "Dev1/11"
    a["VElectrode1"] = "Dev1/12"
    a["VElectrode2"] = "Dev1/13"
    a["VElectrode3"] = "Dev1/14"
    a["VElectrode4"] = "Dev1/15"
    a["VElectrode5"] = "Dev1/16"
    a["VElectrode6"] = "Dev1/17"
    a["VElectrode7"] = "Dev1/18"
    a["VElectrode8"] = "Dev1/19"
    a["VImg1PIDSet"] = "Dev1/21"
    a["VPicoMotor369h"] = "Dev1/22"
    a["VPicoMotor369v"] = "Dev1/23"
    a["VImg2PIDSet"] = "Dev1/24"
    return a


def _consts():
    """Physical constants / calibrated values (``expConfig.m`` ``consts``)."""
    c = {}
    # Data saving
    c["MatlabURL"] = "tcp://127.0.0.1:1408"
    c["PathPrefix"] = r"D:\OneDrive - Harvard University\Documents - Yb"

    # Orca Quest camera
    c["Orca"] = {"ROI": [1000, 100, 2100, 2100], "ExposureTime": 0.050004}

    # 556nm resonance (calibrate daily by spectroscopy; 3P1 mj=0 near-magic)
    c["Resonance556mj0Freq"] = 107.8861e6  # fit 2026-07-05 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.960, FWHM 58.0 kHz, 205 shots, scan 20260705112318, 33x33_feedback9); -2.5 kHz vs prior (within linewidth). was 107.8762e6 (07-03); 107.8560e6 (07-02); 107.8448e6 (06-30); 107.8499e6 (06-29); 107.8478e6 (06-28, NEW LUT); 107.8389e6 (06-26); 107.8199e6 (06-23, 33x33_feedback9); 107.7753e6 (06-12, 47x47_uniform); 107.7673e6 (06-11); 107.7677e6 (06-10); 107.7552e6 (06-09); 107.7531e6 (06-09); 107.7573e6 (06-09); 107.7503e6 (06-08); 107.735e6 (06-05); 107.717e6
    c["Resonance399Freq"] = 310e6              # not magic; changes with trap depth

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 252.07e6, "FreqOld": 252.07e6},
        "Electrodes": {"Vx": -0.042, "Vy": 0.0066, "Vz": 0.0059},  # 2026-06-28 DC-Stark E-field null via StarkV*Revival616Scan (per-axis 616-EOM revival vs electrode V, parabola vertex; each axis fit after the prior was nulled). Vx -0.0233->-0.042 (scan 20260628170653, R2=0.9994, a=1.637 MHz/V^2); Vy 0.0027->0.0066 (scan 20260628174713, R2=0.9999, a=1165 MHz/V^2); Vz 0.004859->0.0059 (scan 20260628182552, R2=0.9994, a=1000 MHz/V^2). Min-Stark 308 resonance (616-EOM) ~282.08 MHz at the null
        "VSLMServo": 3.7,                        # 112 sites at 6A at 30dB
    }

    # BlueMOT
    c["BlueMOT"] = {
        "BFieldGradient": 30,                  # G/cm maximum
        "BiasCoilCurrent": {"Ryd": 3, "X": 0.1, "Y": 0, "Z": 0},
        "FreqDetuning": -44e6,                 # fast-loading opt 2026-06-05; was -40e6
        "Amp": 0.6,
        "LoadingTime": 500e-3,                 # fast-loading opt 2026-06-05 (loading saturates ~0.21); was 500e-3
        "Img1PIDSet": 0.5,  
        "Img2PIDSet": 0.5
    }

    # GreenMOT
    c["GreenMOT"] = {
        "BFieldRampTime": 100e-6,              # blue->green B-field ramp
        "BFieldGradient": 3,
        # fast-loading opt 2026-06-05: X was 0.039, Y was 0.27
        # 2026-06-11 X-bias (MOT-position) LACScan 20260611112242: loading window
        # [0.036,0.040] A, rate peak 0.038 (0.557). x-gradient corr(load,x) flips
        # +0.27@0.038 -> -0.27@0.040 -> zero-crossing ~0.039 = MOT centered on the
        # array (flattest gradient = best uniformity), which is also the loading-
        # plateau center (drift-robust) with rate within ~2% of peak. X 0.040->0.039.
        # 2026-06-21 Y-bias (vertical MOT-position) sweep on 33x33_uniform (scan
        # 20260621_172514, _load_bias r11): corr(load,y) zero-crossing drifted UP to
        # Y~0.281; moved Y 0.268->0.280 to null the vertical gradient (corrY
        # -0.18@0.268 -> ~-0.05@0.280) at ~98% of peak load. X re-checked at the same
        # time (scan 20260621_172015): 0.039 still the 0.038-0.040 viable-window center.
        "BiasCoilCurrent": {"Ryd": 0, "X": 0.0388, "Y": 0.265, "Z": 0.18},
        # fast-loading opt 2026-06-05: HandoverTime was 30e-3
        "PowerBroaden": {"HandoverTime": 15e-3, "FreqDetuning": 0.7e6, "Amp": 0.8},
        # fast-loading opt 2026-06-05: HoldTime was 200e-3, Amp was 0.2
        "CoolDown": {"RampdownTime": 50e-3, "HoldTime": 200e-3,
                     "FreqDetuning": 0.35e6, "Amp": 0.25},
    }

    # Absorption imaging
    # There is only onebeam for AbsImag and it requires flipping a mirror mount
    c["AbsImag"] = {"TOF": 0, "ExposureTime": 50e-6, "BetweenImagsTime": 50e-3,
                    "Freq": 315e6, "Amp": 0.5}
    # SLM
    c["SLM"] = {
        "AOM": {"Freq": 120e6, "Amp": 0.55},
        "VServo": None,                        # cross-ref -> Init.VSLMServo (set below)
        "Modulation": {"Time": 10e-3, "Freq": 100e3, "Amp": 0},
        # Every-scan default loading pattern. RUNTIME-ONLY: consumed by the pyctrl runner
        # (_loading_defaults in YbExptCtrl/runner.py), never read by a sequence -> no
        # serialize() byte effect, so it does NOT touch the byte oracles. expConfig.py is the
        # live source of truth and hot-reloads per job, so the default array changes WITHOUT a
        # backend restart. The runner.py module constants (DEFAULT_LOADING_PATTERN_PHASE /
        # DEFAULT_LOADING_DEFOCUS / ALL_SCANS_LOAD_PATTERN) are now only the fallback when this
        # "Loading" key is absent (e.g. a bare JSON snapshot). After editing this, regenerate
        # the config drift oracle: ``python pyctrl/tools/capture_config_reference.py``.
        #   DefaultPhase        - server-side WGS phase written when a scan declares no pattern
        #   Defocus             - ANSI z4 loading defocus (rad); fixed loading plane (camera-set)
        #   AllScansLoadPattern - when True, EVERY no-pattern scan writes DefaultPhase + holds
        #                         the SLM lock + detects with that pattern's per-pattern thresholds
        "Loading": {
            "DefaultPhase": "phase/47x47_feedbackwarm3.pt",
            "Defocus": -5.0,
            "AllScansLoadPattern": True,
        },
    }

    # LAC
    c["LAC"] = {
        "FreqDetuning": 0.11e6, "Amp": 0.16, "Time": 20e-3, "DeadTime": 10e-3,
        "BlueLAC": {
            "FreqDetuning": -3.8e6, "Amp": 0.17, "Time": 500e-3, "DeadTime": 30e-3,
            "BiasCoilCurrent": {"Ryd": 0},
            "Resonance556mj0Freq": None,       # cross-ref -> Resonance556mj0Freq (set below)
            "X": {"FreqDetuning": 0.22 * 1e6, "Amp": 0.04},
        },
    }

    # Imag399
    c["Imag399"] = {
        "FreqDetuning": -5e6, "Amp1": 1, "Amp2": 1, # Now we use the VIMG1/2Set to control the imaging power
        "ExposureTime": None,                  # cross-ref -> Orca.ExposureTime (set below)
        "Cool556": {
            "FreqDetuning": 0.18e6, "Amp": 0.2,
            # cooling opt 2026-06-05 (CoolingScan, jointly converged at imaging amp 0.18):
            #   was X {0.12e6, 0.18}, h {0.14e6, 0.14}
            "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
            "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
        },
        # StrobeImag399Step: per super-cycle = BEAM 1 pulse -> BEAM 2 pulse -> 556 recool, within ONE
        # camera exposure (arXiv:2507.01011 scheme, our regime). The two COUNTER-PROPAGATING 399 imaging
        # beams (AmpAbsImag=img1, Amp399Imag2=img2) ALTERNATE -- each on for BeamPulseTime, the other off,
        # 556 off -- so a pair cancels net photon recoil ("400 ns pulses ... mitigate momentum transfer
        # from a single beam", here 1 us). Then a recool window (399 off, 556 on). period = 2*BeamPulseTime
        # + RecoolTime; n_cycles = round(Orca.ExposureTime/period) -> the train auto-fills the camera frame
        # (choose the collect window via Orca.ExposureTime, base or a ByPattern overlay).
        #
        # BeamPulseTime/RecoolTime set the LOOP structure -> FIXED scalars, NEVER a .scan() axis; sweep
        # them across ROUNDS (tools/strobe_imaging_round.py --pulse-time). BeamPulseTime 1 us ~ AOM rise
        # -> partial pulse; verify on scope (trigger on the first PD pulse -- no scope-sync TTL wired).
        # WARN: small RecoolTime -> many cycles -> big sequence; validate at short Orca.ExposureTime first.
        #
        # Cool556 here is a SEPARATE recool set from the during-imaging Imag399.Cool556: with the 399 OFF
        # during recool there is no light shift -> the optimum detuning/amp differs. Seeded from
        # Imag399.Cool556; retune via tools/strobe_imaging_round.py cool mode.
        # PulsesPerBurst (N): per super-cycle, fire N alternating beam1/beam2 SHORT pulses (the image
        # burst -- short pulses keep the per-beam momentum kick + heating low) THEN one recool window.
        # period = 2*N*BeamPulseTime + RecoolTime; duty = 2*N*BeamPulseTime/period. Raise N to raise duty
        # (more photons) WITHOUT lengthening the single-beam pulse. WARN: events ~ cycles*N -> a high-N,
        # short-pulse train over a 100 ms exposure is a LARGE sequence (~1e5 pulse events); watch the
        # engine. N/BeamPulseTime/RecoolTime are loop counts -> FIXED scalars, never a .scan() axis.
        "Strobe": {
            "BeamPulseTime": 1e-6, "RecoolTime": 5e-6, "PulsesPerBurst": 5,
            "Cool556": {
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
            },
        },
    }

    # Cool556
    c["Cool556"] = {
        "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
        # RNR cooling opt 2026-06-05 (CoolingScan_RNR, release-recapture 50us, interleaved X<->h
        # joint converged, survival 0.28->0.31): was X {0.11e6, 0.16}, h {0.11e6, 0.14}
        "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
        "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
    }

    # Pushout
    c["Pushout"] = {
        "Time": 10e-3,
        "Green": {"Freq": 118.1e6, "Amp": 0},
        "Blue": {"Freq": 320e6, "Amp1": 0, "Amp2": 0},  # Amp1 and Amp2 for pushout (matches Imag399.Amp1/Amp2)
        "Ryd308": {"Freq": 200e6, "Amp": 0},
        "Ionization": {"Amp": 0},              # 369 ionization-beam amp default (RydbergPushoutStep)
        "STIRAP": {"delay": 1e-6, "reverse_delay": 1e-6, "gap": 10e-6},
        "MRabi": {"Freq": 4000, "Gain": 0},
        "Ramsey": {"Phase": 0},
    }

    # 616 AOM diverted/idle amplitude (RydbergPushoutStep restores this after pushout)
    c["AOM616Divert"] = {"Amp": 0.11}

    # SLM trap modulation
    c["SLMTrapModulation"] = {
        "Time": 5e-3, "Freq": 100e3, "AmpFactor": 0.5,
        "lowerTrapDepth": {"Vservo": 0, "Time": 3e-3},
    }

    # Rearrangement
    c["SLMRearrange"] = {"Time": 100e-3}

    # AWG defaults (Siglent SDG6X)
    # shape: gaussian / rise_gaussian / fall_gaussian / rise_linear / fall_linear;
    # smooth_width_us: EXTRA half-cosine edge window (0 = sharp; ignored for gaussian).
    # pulse_width_us stays the MAIN window; total playback = pulse_width_us + smooth_width_us.
    # (devices/sigilent_awg/pulse_waveform.py; shape gallery: pyctrl/tmp/pulse_10_examples.png)
    c["AWG556"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCC900309::0::INSTR",
        "channel": "C1", "max_amplitude_vpp": 11, "num_points": 10000,
        "pulse_width_us": 4, "carrier_freq_MHz": 130.78, "steepness": 3.5,
        "amplitude_scale": 1.0, "shape": "gaussian", "smooth_width_us": 0.0,
    }
    c["AWG308"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCD801391::0::INSTR",
        "channel": "C1", "max_amplitude_vpp": 6, "num_points": 10000,
        "pulse_width_us": 4, "carrier_freq_MHz": 200, "steepness": 3.5,
        "amplitude_scale": 1.0, "shape": "gaussian", "smooth_width_us": 0.0,
    }

    # 60 Hz AC-line trigger. When enabled, the FPGA waits for an edge on a TTL INPUT line at the
    # start of each basic sequence (ExpSeq.enable_global_wait_trigger -> a version-2 ZYNQZYNQ
    # block -> libnacs emits a WaitTrigger bytecode op), so every shot begins at the same mains
    # phase (B-field stability). Consumed ONLY by the runner's compile_point
    # (YbExptCtrl/runner.py) -- it is never read by a sequence step, so it does NOT enter the
    # serialized bytes (the MATLAB byte oracles are unaffected; only this config snapshot needs
    # re-capturing). Per-scan override: runp().LineTriggerEnable / LineTriggerChannel /
    # LineTriggerRaise / LineTriggerTimeout.
    #   Channel: RAW FPGA TTL line number of the line-sync input (same numbering as the TTL
    #            outputs, e.g. FPGA1/TTL14 -> 14) -- NOT a channel alias. Must be 0..max_ttl_chn
    #            (config.yml), must NOT equal start_ttl_chn, and must NOT be driven as an output.
    #            None = unset -> the runner SKIPS enabling (and logs once) rather than guess a
    #            line; SET it to your physical line-sync input to activate for every scan.
    #   Raise:   True = wait for a rising edge, False = falling edge.
    #   Timeout: seconds. FPGA clock is 100 MHz and the bytecode timeout field is 24-bit, so the
    #            max is ~0.168 s; ~0.02 s = one 60 Hz period + margin (catches the next edge, then
    #            proceeds if the signal is absent -- it does not hang the shot).
    c["LineTrigger"] = {
        "Enable": False,
        "Device": "FPGA1",
        "Channel": None,                       # <-- SET to your line-sync FPGA TTL input line
        "Raise": True,                         # True = rising edge, False = falling edge
        "Timeout": 0.02,                       # seconds (~one 60 Hz period + margin)
    }

    # ---- per-pattern overrides (RUNTIME-ONLY; see the "Per-pattern config overlay" section) ----
    # Map an SLM loading-pattern NAME (the phase-file basename) to a SPARSE override of the leaves
    # above that differ for that array. Any leaf left unset falls back to the base value above,
    # then a swept/manual g() value wins over both. 47x47_uniform is SEEDED below with the CURRENT
    # base cooling/imaging defaults, so the overlay is byte-identical until you tune these from a
    # 47x47 scan -- then edit the numbers here. (Trap depth Init.VSLMServo is intentionally NOT
    # seeded -> 47x47 uses the base value; add "Init": {"VSLMServo": <v>} to tune it per array.)
    c["ByPattern"] = {
        "47x47_uniform": {
            # Initialize the SLM servo to 3.7
            "Init": {
                "VSLMServo": 3.5
            },
            # imaging (399) + cooling-during-imaging (556 X/h).
            # 556 X/h cooling-during-imaging re-optimized for 47x47_uniform 2026-06-11 (CoolingScan,
            # real imaging amp Blue 0.18 + 1 s hold, interleaved X<->h to the joint fixed point;
            # survival ~0.915). Only h.Amp moved (0.13 -> 0.14); X (det +0.16, amp 0.20) and h det
            # +0.16 confirmed. ExposureTime omitted (cross-ref to Orca.ExposureTime, re-resolved).
            # 2026-06-12: briefly tried Imag399.Amp 0.2 (larger histogram split, +14% SNR) but it
            # cost survival 0.905 -> ~0.80 even after re-tuning cooling (X->0.22, h->0.16) -- reverted
            # to 0.18 and these 0.18-optimum X/h values (full data in Notion 06/12).
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            # RNR / release-recapture cooling (556) -- re-optimized for 47x47_uniform 2026-06-11
            # (CoolingScan_RNR, 50us release, interleaved X<->h coordinate ascent to the joint
            # fixed point; survival ~0.29 at 50us, loading ~0.58, broad flat plateau).
            # Was X {0.135e6, 0.13}, h {0.13e6, 0.12} (seeded from the prior array).
            # 2026-06-12: h FreqDetuning 0.14e6 -> 0.12e6. Re-run at the more sensitive 30us
            # release (50us was washed out, spread ~0.06; 30us spread ~0.22, ~4x contrast);
            # interleaved X<->h converged, survival ~0.54 at 30us. X {0.13e6, 0.14} and h.Amp
            # 0.12 confirmed; only h det moved one fine step (0.14->0.12, ~4.6 SEM, ~0.02 gain).
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.13e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.12e6, "Amp": 0.12},
            },
            "LAC": {
                "FreqDetuning": 0.11e6, "Amp": 0.16, "Time": 10e-3, "DeadTime": 10e-3,
                "BlueLAC": {
                    "FreqDetuning": -3.8e6, "Amp": 0.17, "Time": 500e-3, "DeadTime": 30e-3,
                    "BiasCoilCurrent": {"Ryd": 0},
                    "Resonance556mj0Freq": None,       # cross-ref -> Resonance556mj0Freq (set below)
                    "X": {"FreqDetuning": 0.22 * 1e6, "Amp": 0.04},
                }
            }
        },
        "33x33_uniform": {
            # Initialize the SLM servo to 3.7
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},   # per-user override 2026-06-21 (base 10 ms); deep-merges, rest of LAC inherits base
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.10,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # feedback9 (2026-06-22): CLEAN final flatten = feedback6 flattened by its OWN combined-40rep map,
        # NO corner/near-DC boosts (those perturb the bulk speckle -> feedback8 bulk degraded to 3.99%).
        # This is the UNIFORM production array: split-half true depth CV 2.45% < 2.5% for ~1061 spots
        # (gentle, phase change 0.010 rad). The 7 optical-worst (3 near-DC + 4 extreme corners, r>0.94)
        # left natural -- they are under-illuminated by the corner optics and need realignment to load at
        # uniform depth (boosting fits them but breaks uniformity). Phase phase/33x33_feedback9.pt.
        "33x33_feedback9": {
            "Orca": {"ExposureTime": 0.035},  # 2026-07-08 kept 35 ms; after a bench beam-2 (399) power cut the imaging went separation-limited at the OLD amps (d' ~3), but higher Amp1+Amp2 recover separation without lengthening exposure (trap depth confirmed full ~413 uK via mj=1 dip scan 20260708210345). (Briefly ran 50 ms r526/r527 to diagnose; reverted -- 50 ms is GLOBAL via runner sync_camera_exposure.)
            "Init": {"VSLMServo": 1.9},
            
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                # 2026-07-01 re-baseline AFTER the two-beam 399 realignment (which RESTORED the 06-29
                #     power ceiling; dose/amp shifted ~1.6x so the old amps are void). Amp scan r400-r401
                #     (0 pushout): joint plateau Amp1 0.56 / Amp2 0.33 (heating turnover ~0.64). Cooling
                #     X + h both re-confirmed head-to-head (100-shot single-point pairs): X(0.158,0.267)
                #     kept vs fitted candidate (0.174,0.283) -- no gain; h(0.14,0.24) kept vs (0.14,0.195)
                #     -- no gain. Final 260-shot characterization: surv 0.9893 +/- 0.0003 (runs 0.986-0.991),
                #     per-site fid median 0.9928 / d' 4.17, spatially FLAT (<=0.13%/array) -- fid below the
                #     0.995 target; remaining lever = exposure 35->50 ms (not taken). NOTE mid-campaign the
                #     r400 DIM amp cells poisoned the threshold accumulator -> store refit + verify (see
                #     memory bug-threshold-dim-scan-contamination); numbers above are post-fix.
                # 2026-07-05 DAILY-CAL AMP RE-OPT (399 dose drifted UP ~2x again since 07-03/04: at the
                #     old 0.23/0.22 the daily-cal scans read surv 0.970 / fid med 0.9867 / d' 3.91; the
                #     r510 map put 0.23/0.22 at surv 0.75, deep past the heating cliff). Amp scans r510
                #     (7x5, 0.11-0.47 x 0.10-0.34) + r511 (low-Amp1 extension 0.05-0.17) -> interior
                #     plateau optimum Amp1 0.11 / Amp2 0.22. 100-shot single-point confirm (scan
                #     20260705_115052, 0 pushout, 35 ms): surv 0.9881 +/- 0.0004, per-site fid median
                #     0.9971, d' 4.88, matching the 07-04 baseline. Amp1 0.23 -> 0.11, Amp2 kept 0.22;
                #     cooling untouched (X 0.16e6/0.26, h 0.16e6/0.14).
                # 2026-07-03/04 RE-OPT (399 power drifted UP again since 07-01 -> the 07-01 amps 0.56/0.33
                #     over-dose/heat now; whole amp optimum dropped ~2.4x, the classic hardware-drift tell).
                #     Amp scan r1/r2 (0 pushout) -> plateau, Amp2>0.22 turns over -> Amp1 0.23 / Amp2 0.22.
                #     Cooling: 1s locate + 0-pushout confirm + X-amp push 0.26/0.29/0.32 -> X amp 0.267->0.26
                #     optimal (0.29/0.32 heat); h locate peak det 0.16/amp 0.14 (broad/flat) -> h 0.14e6/0.24
                #     -> 0.16e6/0.14 (what the confirm ran). 100-shot 0-pushout confirm (scan
                #     20260703185438, at 35 ms ByPattern exposure): real survival 0.952 -> 0.988 (~40 SEM),
                #     per-site fid median 0.990 -> 0.996, d' 3.87 -> 4.90, spatially flat. 2D amp map r6
                #     (20260704001324) confirmed the plateau. X det 0.158->0.16 (grid step, within noise).
                # --- prior 2026-07-01 re-baseline AFTER two-beam realignment: Amp1 0.56/Amp2 0.33,
                #     X(0.158,0.267) h(0.14,0.24); surv 0.9893, fid med 0.9928 / d' 4.17. Superseded by
                #     the 07-03/04 399 power drift + re-opt above.
                # --- prior 2026-06-29 DEGRADED interim (399 power drift): Amp1 1/Amp2 0.18, h(0.14,0.14),
                #     fid 0.986 / surv 0.957; superseded by the realignment + this re-baseline.
                # --- prior 2026-06-28 RECAL (post beam-1 fix): Amp1 0.24->0.52, Amp2 0.26->0.24; X(0.16,0.24),
                #     h(0.12,0.18); r10 confirm fid med 0.9934 / surv 0.986 / d' 4.17, flat.
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        
        # 2026-07-07: new array with 14.5um spacing for Rydberg
        "23x23_14p5um": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 3.2},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.22,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        
        # 2026-07-07: new array with 20um spacing for Rydberg
        "17x17_20um": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 0.6},
            "BlueMOT": {"LoadingTime": 300e-3},
            "GreenMOT": {"CoolDown": {"HoldTime": 150e-3}},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.22,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        
        # 2026-07-04: NEW two-layer bifocal array 2x11x11_5um (phase/2x11x11_5um.pt): 11x11 grid
        # duplicated at TWO axial planes z4 = +-2.7778 rad about the stack center, SAME xy for both
        # layers (pure bifocal stack -- camera boxes catch both layers; per-layer readout needs the
        # loading defocus moved to -5 +- 2.7778). Entry = EXACT COPY of 33x33_feedback9's current
        # params (user 07-04) except Init.VSLMServo 0.39 (242 traps vs 1068 -- proportionally less
        # total power). Per-pattern thresholds start fresh (seeded flat 202.5).
        "2x11x11_5um": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 0.39},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.23, "Amp2": 0.22,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.26},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },

        # 2026-07-02: NEW array tri_3013_camfb (3013-site triangular, camera-feedbacked, no spots near
        # DC -- nearest 120 knm px). Entry seeded as an EXACT COPY of 3270_tri's (runbook: cooling/
        # imaging carries over within an array family; VSLMServo copied verbatim, never tuned here).
        # Loading defocus -5 (user 07-02). Imaging re-optimization for THIS array tracked in
        # _feedback3013/CAMPAIGN_STATE.md; NOTE 07-01 399 realignment shifted dose ~1.6x, so
        # the inherited 06-30 amps are a starting point only. Orca.ExposureTime deliberately
        # NOT overridden (user 07-02: default exposure for this array, unlike 3270_tri's 100 ms).
        "tri_3013_camfb": {
            "Init": {"VSLMServo": 3.5},
            # 2026-07-02 LOADING optimization (campaign _feedback3013, phases 0-8c; verify scan
            # 20260702_045020): loading 0.42 -> 0.57 mean / 0.61 median. The BIG lever was LAC:
            # the inherited deep-trap LAC (25 ms, amp 0.16) was boiling ~40% of atoms out of this
            # array's ~250 uK traps -> (4 ms, 0.06). Also BFieldGradient 30->34 (+0.02, first
            # scan ever), CoolDown (0.35 MHz, 0.25)->(0.25 MHz, 0.28) (+0.09), bias Y 0.265->0.259
            # (nulls y-gradient), LoadingTime 0.9->0.4 s (flat 0.2-0.9). Blue capture (-44, 0.6),
            # PowerBroaden (0.7, 0.8), bias X 0.0385 confirmed at defaults. Pair check clean
            # (no 2-atom histogram peak; bright-img1 shots survive BETTER). Known cost: survival
            # 0.93 -> 0.84 from now-kept hot marginal atoms -> cooling/imaging re-opt follows.
            "BlueMOT": {"LoadingTime": 0.4, "BFieldGradient": 34},
            "GreenMOT": {
                "BiasCoilCurrent": {"Y": 0.259},
                "CoolDown": {"HoldTime": 0.3, "FreqDetuning": 0.25e6, "Amp": 0.28},
            },
            "LAC": {"Time": 4e-3, "Amp": 0.06},
            "Imag399": {
                # 2026-07-02 (i) amp re-center at default 50 ms, post-399-realignment: W was
                #     0.20/0.32, X(0.16,0.17)/h(0.20,0.17) (r502-r504: fid med 0.990, d' 4.03,
                #     load 0.45, surv med 0.936).
                # 2026-07-02 (ii) RE-OPT after the loading campaign (loading 0.42->0.60 keeps
                #     hot marginal atoms; survival dipped to 0.84): cooling X (0.19 MHz, 0.23) /
                #     h (0.18 MHz, 0.17) (r505/r506 0-pushout maps) + amps -> 0.26/0.28 (r507).
                #     r508 250-shot confirm: loading 0.578/0.604, survival 0.925/0.936, fid med
                #     0.983, d' 3.71, spatially flat (grad 2.6%), ALL 3013 sites fit.
                "FreqDetuning": -5e6, "Amp1": 0.26, "Amp2": 0.28,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.19e6, "Amp": 0.23},
                    "h": {"FreqDetuning": 0.18e6, "Amp": 0.17},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.14e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.14e6, "Amp": 0.12},
            },
        },
        "3270_tri": {
            "Orca": {"ExposureTime": 0.100},  # 2026-06-29: 35->50->75->100ms. Each step lifts survival ceiling (0.83->0.92->0.94) + fidelity/d' (limit was dim imaging, not cooling). 75ms W: amps 0.22/0.22, cool X(0.18,0.24)/h(0.18,0.18), surv ~0.94 fid 0.987 d'3.7. Pushing to 100ms (survival still < 99% gate).
            "Init": {"VSLMServo": 3.5},
            # 2026-07-01: per-pattern MOT loading raised 0.7 (global) -> 0.9 for this large 3270-site
            # array only, to load the wide array more fully (user request).
            "BlueMOT": {"LoadingTime": 0.9},
            "GreenMOT": {"CoolDown": {"HoldTime": 0.3}},  # 2026-07-01: per-pattern MOT loading raised 0.7 (global) -> 0.9 for this large 3270-site array only, to load the wide array more fully (user request).
            "LAC": {"Time": 25e-3},
            "Imag399": {
                # 2026-06-29 imaging optimization for 3270_tri (the prior values were copied from
                # 33x33_feedback9 -- never tuned for this array). Full campaign at 100 ms exposure
                # (raised 35->50->75->100 ms; each step lifted survival ceiling + d'/fidelity, the
                # limit being dim imaging not cooling) and at the re-found focus z4 = -1.2 (the array
                # was DEFOCUSED at -5; a z4 scan peaked at -1, then drifted to -1.2 -- thermal lensing,
                # separation dMu 7.2->4.9 over the session, so the loading defocus needs periodic re-scan).
                # Amp1 0.24->0.19, Amp2 0.26->0.14 (r237 amp scan at z4=-1; plateau, survival turns over
                # ~amp1 0.25+). X cooling 0.158/0.267 -> 0.18/0.21, h 0.14/0.24 -> 0.16/0.17 (0-pushout
                # confirms at 100 ms). FINAL per-site (r255, z4=-1.2, 100 shots): per-site survival 0.961,
                # per-site fidelity (analytic Gaussian-overlap) median 0.985 / d' 3.84. BELOW the 99%/99.5%
                # gate -- separation-limited by (a) thermal-lensing focus drift, (b) a chronic shallow
                # BOTTOM-LEFT trap region (BL survival 0.921 vs rest 0.964, X/Y MOT-position independent
                # -> trap-depth/SLM, needs trap-depth feedback). MOT left at default (X 0.0385/Y 0.265).
                # 2026-06-30 RE-OPTIMIZED on the now depth-BALANCED array (after the trap-depth feedback
                # campaign flattened CV 11%->1.54%). Focus re-scanned z4 -1.2 -> -1.5 (drifted, thermal
                # lensing). Amp1 0.19->0.28 (r3/r4 amp scan: fid plateaus + survival turns over >0.28; low
                # amp = detection-limited). Cooling re-opt (the win): X(0.18,0.21)->(0.16,0.17), h det
                # 0.16->0.20 amp 0.17 -- drove d' 3.60->4.59. The old shallow-BL d' deficit is GONE (depth
                # flat); fidelity is now globally separation-limited (cooling, not photons). FINAL per-site
                # (job 1140, z4=-1.5, 150 shots): fidelity median 0.9965 (72.7% sites >=0.995, GATE MET),
                # d' 4.59, survival mean 0.967 / median 0.988. Residual survival = scattered hi-d' loss +
                # mild BL 0.955 (hardware: 556/imaging-beam align or trap depth) + the 100 ms heating budget
                # (exposure-shorten deferred: it's a GLOBAL live-camera change, shared with rearrange).
                "FreqDetuning": -5e6, "Amp1": 0.28, "Amp2": 0.14,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.17},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.17},
                },
            },
            "Cool556": {
                # 2026-06-29 RNR (release-recapture) cooling re-opt at 30 us release (focus z4=-1.2):
                # X r260 det 0.16->0.14 (amp 0.14 kept), h r261 det 0.16->0.14 (amp 0.12 kept) -- both
                # detunings to 0.14 MHz peaked recapture survival (X 0.476 / h 0.502 @30us, +~0.03 ~8 SEM
                # = colder atoms). Loading flat ~0.58 (cooling-independent). Amps unchanged.
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.14e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.14e6, "Amp": 0.12},
            },
        },
    }

    # 2026-07-05: "2x11x11_5um_3d" = the 242-site (per-plane, no-dedup) DETECTION alias of the
    # bifocal array, used by the layer-isolation rearrangement runs (extras.initial_pattern keys
    # BOTH the rearrange detector calibration AND the per-bseq config overlay -> alias must exist
    # here or those runs fall back to base config). Same physical array -> same dict object.
    c["ByPattern"]["2x11x11_5um_3d"] = c["ByPattern"]["2x11x11_5um"]

    # ---- cross-references (mirror expConfig.m's const-to-const assignments) ----
    return expConfig_helper.apply_cross_refs(c)


def _default_vals(consts):
    """Per-channel default values (``expConfig.m`` ``defaultVals``; Hz / s).

    ``consts`` (the ``_consts()`` dict) is passed in so a default can cross-reference a
    constant (e.g. ``VSLMservo`` <- ``Init.VSLMServo``), mirroring expConfig.m's
    ``defaultVals(...) = consts....`` lines. Values are float-coerced downstream by
    ``SeqConfig`` (so an int constant like ``Init.VSLMServo`` becomes a float here too)."""
    d = {}
    # TTLs
    d["TTLThorCamTrig"] = 0
    d["TTLOrcaTrig"] = 0
    d["TTL556ImagingShutter"] = 0
    d["TTL556MOTaShutter"] = 1
    d["TTL556MOTbShutter"] = 1
    d["TTL556MOTcShutter"] = 1
    d["TTL399AbsImagShutter"] = 0
    d["TTL399Imag2Shutter"] = 0
    d["TTL399MOTShutter"] = 1
    d["TTL3992DMOTShutter"] = 1
    d["TTL556RydAWG"] = 0
    d["TTL308RydAWG"] = 0
    d["TTLScopeTrig"] = 0
    d["TTLQickTrig"] = 0
    d["TTLSampleAndHold"] = 1
    d["TTL399IMG1PIDMode"] = 0
    d["TTL399IMG2PIDMode"] = 0
    # DDS
    d["Freq556MOTX"] = 118e6
    d["Amp556MOTX"] = 0
    d["Freq556RydbergMOTh"] = 118e6
    d["Amp556RydbergMOTh"] = 0
    d["FreqBlueMOT"] = 270e6
    d["AmpBlueMOT"] = 0.85
    d["Freq369"] = 250e6
    d["Amp369"] = 0
    d["FreqAbsImag"] = 320e6
    d["AmpAbsImag"] = 0
    d["Freq399Imag2"] = 320e6
    d["Amp399Imag2"] = 0
    d["Freq2DMOT"] = 290e6
    d["Amp2DMOT"] = 1
    d["FreqZeeman"] = 273.5e6
    d["AmpZeeman"] = 0.6
    d["FreqSLM"] = 120e6
    d["AmpSLM"] = 0.4
    d["FreqAOM308"] = 200e6
    d["AmpAOM308"] = 0
    
    # NI DAQ
    d["VSLMservo"] = consts["Init"]["VSLMServo"]  # default SLM servo to its Init value
    d["VElectrode1"] = 0
    d["VElectrode2"] = 0
    d["VElectrode3"] = 0
    d["VElectrode4"] = 0
    d["VElectrode5"] = 0
    d["VElectrode6"] = 0
    d["VElectrode7"] = 0
    d["VElectrode8"] = 0
    d["VImg1PIDSet"] = consts["BlueMOT"]["Img1PIDSet"]
    d["VImg2PIDSet"] = consts["BlueMOT"]["Img2PIDSet"]
    # EOM616 (FreqEOM616Old is a MemoryMap runtime override in MATLAB; default here)
    d["FreqEOM616"] = 200e6
    d["AmpEOM616"] = 0.7
    d["FreqAOM616"] = 120e6
    d["AmpAOM616"] = consts["AOM616Divert"]["Amp"]  # default AOM616 to the divert amplitude
    d["FreqAODs"] = 80e6
    d["AmpAODs"] = 0
    d["FreqAODv"] = 89.3e6
    d["AmpAODv"] = 0.14
    return d
