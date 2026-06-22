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
    c["Resonance556mj0Freq"] = 107.7753e6  # fit 2026-06-12 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.966, FWHM 65.8 kHz, 218 shots, scan 20260612100157, 47x47_uniform array); +8.0 kHz vs prior (within linewidth). was 107.7673e6 (06-11); 107.7677e6 (06-10); 107.7552e6 (06-09); 107.7531e6 (06-09); 107.7573e6 (06-09); 107.7503e6 (06-08); 107.735e6 (06-05); 107.717e6
    c["Resonance399Freq"] = 310e6              # not magic; changes with trap depth

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 252.07e6, "FreqOld": 252.07e6},
        "Electrodes": {"Vx": -0.0233, "Vy": 0.0027, "Vz": 0.004859},
        "VSLMServo": 3.7,                        # 112 sites at 6A at 30dB
    }

    # BlueMOT
    c["BlueMOT"] = {
        "BFieldGradient": 30,                  # G/cm maximum
        "BiasCoilCurrent": {"Ryd": 3, "X": 0.1, "Y": 0, "Z": 0},
        "FreqDetuning": -44e6,                 # fast-loading opt 2026-06-05; was -40e6
        "Amp": 0.6,
        "LoadingTime": 500e-3,                 # fast-loading opt 2026-06-05 (loading saturates ~0.21); was 500e-3
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
        "BiasCoilCurrent": {"Ryd": 0, "X": 0.039, "Y": 0.280, "Z": 0.18},
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
        "FreqDetuning": 0.11e6, "Amp": 0.16, "Time": 10e-3, "DeadTime": 10e-3,
        "BlueLAC": {
            "FreqDetuning": -3.8e6, "Amp": 0.17, "Time": 500e-3, "DeadTime": 30e-3,
            "BiasCoilCurrent": {"Ryd": 0},
            "Resonance556mj0Freq": None,       # cross-ref -> Resonance556mj0Freq (set below)
            "X": {"FreqDetuning": 0.22 * 1e6, "Amp": 0.04},
        },
    }

    # Imag399
    c["Imag399"] = {
        "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
        "ExposureTime": None,                  # cross-ref -> Orca.ExposureTime (set below)
        "Cool556": {
            "FreqDetuning": 0.18e6, "Amp": 0.2,
            # cooling opt 2026-06-05 (CoolingScan, jointly converged at imaging amp 0.18):
            #   was X {0.12e6, 0.18}, h {0.14e6, 0.14}
            "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
            "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
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
    c["AWG556"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCC900309::0::INSTR",
        "channel": "C1", "max_amplitude_vpp": 11, "num_points": 10000,
        "pulse_width_us": 4, "carrier_freq_MHz": 130.78, "steepness": 3.5,
        "amplitude_scale": 1.0,
    }
    c["AWG308"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCD801391::0::INSTR",
        "channel": "C1", "max_amplitude_vpp": 6, "num_points": 10000,
        "pulse_width_us": 4, "carrier_freq_MHz": 200, "steepness": 3.5,
        "amplitude_scale": 1.0,
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
        # Two-layer 15x15 "diamond" array (phase/2x15x15_xyoffset_5um.pt): layers at +-2.5 um
        # = +-0.768 rad of the 2*rho^2-1 defocus map (1 rad ~ 3.26 um), 5 um xy offset, ~450
        # sites. Imaging (399) / cooling-during-imaging (556 X/h) / RNR cooling / VSLMServo are
        # an EXACT copy of 33x33_uniform (2026-06-12, per request -- same science camera + 556
        # tones). The midplane sits on the camera at the standard loading_defocus = -5 because
        # the phase is zernike-free / z=0 stack-centered (like a 2-D array's focal plane).
        # CAVEAT: ~450 sites vs 33x33's 1089 => the SAME VSLMServo = 1.9 gave a ~2.4x DEEPER
        # per-trap depth (the |mj|=1 light shift fell BELOW the push-out AOM band, traps far too
        # deep, and the copied-from-33x33 cooling/imaging amps were badly mismatched). VSLMServo
        # turned down 1.9 -> 1.0 (2026-06-13, per request) to bring per-trap depth back into the
        # 33x33-calibrated regime (loading re-checked fine: 0.59, CV 20%). Re-opt CONVERGED 2026-06-13:
        # imaging-cooling X det 0.16 -> 0.30 MHz (below); RNR Cool556 unchanged (insensitive at this
        # depth). Best detection box is 13 / maskSigma 3 (infid 6.3% -> 4.6%) but that is a GLOBAL
        # scan_prep setting, NOT applied here pending a per-pattern hook. Operating survival ~0.82-0.85
        # is the +-2.5 um OPTICAL-DEFOCUS limit (confirmed: unchanged 1 s vs 5 ms hold, and imaging-amp
        # 0.18-0.22 saturates) -- not cooling/hold/photons; a real fix needs per-layer focus.
        "2x15x15_xyoffset_5um": {
            "Init": {
                "VSLMServo": 1.0
            },
            # 2026-06-13 detection box matched to the defocused 2-layer PSF (offline sweep on scan
            # 20260613_174025): boxSize 9->13 / maskSigma 2->3 cut median infidelity 6.3% -> 4.6%
            # (SNR 2.40 -> 2.54). Read per-pattern by scan_prep._pattern_detection_box (gated --
            # other patterns keep the global 9/2). Inert as a sequence const (no step reads it).
            "boxSize": 13, "maskSigma": 3,
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-06-13 imaging-cooling re-opt at VSLMServo 1.0 (interleaved X<->h, Blue 0.18
                    # / 1 s stress hold): the deep-trap-shifted X cooling resonance moved X det
                    # 0.16 -> 0.30 MHz (det=0 is dead); h unchanged. Stress survival 0.80 -> 0.85,
                    # then a broad plateau (the residual cap is the +-2.5 um optical defocus, not cooling).
                    "X": {"FreqDetuning": 0.30e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            # Cool556 (release-recapture): re-checked 2026-06-13 (X & h 2-D at 30 & 50 us release) --
            # recapture survival is INSENSITIVE to the cooling detuning over 0.1-0.4 MHz at this depth
            # (broad ~0.36 plateau at 50 us), so the copied 33x33 seed is left UNCHANGED (only avoid amp>=0.20).
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        # Trap-depth feedback arrays (2026-06-12): warm-start WGS corrections of 33x33_uniform.
        # EXACT copies of the 33x33_uniform overlay (same array family => same SLM servo / cooling /
        # imaging). The load-bearing leaf is Init.VSLMServo = 1.9 (NOT the base 3.7): without this
        # per-pattern entry a renamed pattern falls back to base 3.7 -> traps far too deep -> the
        # |mj|=1 dip leaves the 104.2-107.2 sweep (seen on scan 20260612_182650, edge-pinned 104.14).
        "33x33_optimized_r1": {
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        "33x33_optimized_r2": {
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        "33x33_optimized_r2b": {
            # Step-1 retry: under-relaxed (gamma=0.4) WGS correction of 33x33_optimized_r1
            # (driving scan 20260613_014418). EXACT copy of the 33x33_uniform overlay (VSLMServo 1.9).
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        "33x33_probe": {
            # Step-2 transfer probe (one-off): +-10% random per-spot modulation of r1, to measure
            # whether realized depth tracks the per-spot target. EXACT copy of the 33x33_uniform
            # overlay (VSLMServo 1.9; NO cooling/imaging changes).
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        "33x33_optimized_d1": {
            # Doublet-fit + gentler-WGS restart from uniform (round 1, standard WGS / gradient kill).
            # EXACT copy of the 33x33_uniform overlay (VSLMServo 1.9).
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.18, "Amp2": 0.18,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.13},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.135e6, "Amp": 0.13},
                "h": {"FreqDetuning": 0.13e6, "Amp": 0.12},
            },
        },
        "47x47_feedbackwarm2": {
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
        "47x47_feedbackwarm3": {
            # Round-3 iterative WGS feedback array: warm-started from 47x47_feedbackwarm2.pt with
            # the full per-site trap-depth correction measured on warm2 (scan 20260612_170355).
            # Same cooling/imaging overlay as 47x47_feedbackwarm2 (same 47x47 array family).
            "Init": {
                "VSLMServo": 3.5
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.2, "Amp2": 0.2,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                },
            },
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
        "47x47_feedbackwarm4": {
            # Round-4 iterative WGS feedback array: warm-started from 47x47_feedbackwarm3.pt with the
            # full per-site correction measured on warm3 (scan 20260612_225239). Started as an exact
            # copy of the 47x47_feedbackwarm3 overlay (same 47x47 array family). VSLMServo 3.5.
            # 556 COOLING re-optimized 2026-06-20 for the TWO 399 imaging beams (Amp1/Amp2 = 0.30/0.20)
            # via interleaved X<->h coordinate ascent:
            #   imaging-during-imaging (Imag399.Cool556): X 0.16/0.20->0.20/0.24, h 0.16/0.14->0.20/0.22.
            #   release-recapture (Cool556 below): X 0.13/0.14->0.16/0.14, h 0.12/0.12->0.16/0.12 @30us.
            # IMAGING re-opt (det -4, Amp1 0.38, Amp2 0.17) was TESTED 2026-06-20 but REVERTED: clean
            # final-number measurement showed only fidelity 0.9845->0.9865 at EQUAL ~96% single-shot
            # survival (imaging loss is small vs fixed/detection loss), so not worth the change. Kept
            # 0.30/0.20 @ -5. Magnitude-along-3:2 sweep then explored separately (scans 20260620_~0145+).
            "Init": {
                "VSLMServo": 3.5
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.3, "Amp2": 0.2,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.20e6, "Amp": 0.24},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
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
        # New SLM array (phase/33x33_uniform_centered_level.pt -- a 33x33 "uniform"
        # array rotated to straight (lattice 0.5deg vs 33x33_uniform's 3.5deg) and
        # centered on the zeroth order (knm centroid 512,512). 1068 sites when
        # FFT-extracted at threshold 0.40 (0.30 leaks 8 spurious edge ghosts, 4
        # off-sensor -> detect_atom crash; per-pattern threshold 0.40 set in the
        # registry record). 2026-06-20 COOLING + IMAGING seeded from
        # 47x47_feedbackwarm4. Init.VSLMServo = 1.9 (CHOSEN 2026-06-20, kept after a
        # depth check): the array-average trap depth measured here is ~460 uK (mj=0 f0
        # 107.788 MHz vs mj=1 104.889 MHz, Delta_nu 5.77 MHz; scans 20260620_172908 /
        # _173141 at push-out 0.15). Going to 47x47's 3.5 would be ~850 uK and push the
        # mj=1 dip off the bottom of the push-out band, so 1.9 was kept. The mj=1 line
        # is broad -> large site-to-site depth spread (target of the per-site feedback).
        # cooling/imaging below = warm4 seed (cooling-X confirmed at 1.9: X 0.20/0.22,
        # survival ~0.84); being briefly confirmed. LAC inherited from base.
        "33x33_uniform_centered_level": {
            "Init": {
                "VSLMServo": 1.9
            },
            # imaging amps kept at warm4 seed 0.30/0.20 (StrobeImageScan 20260620_174758:
            # img1 fidelity ~0.975 saturated/flat -> already good). 556 imaging-cooling
            # briefly re-opt at 1.9 (cooling_img_round X 20260620_170130, h _173720):
            # detunings 0.20 MHz confirmed; X amp 0.24->0.22, h amp 0.22->0.18 (survival ~0.87).
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.3, "Amp2": 0.2,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.18},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # Trap-depth feedback round 1 of 33x33_uniform_centered_level: warm-start WGS
        # correction (phase change 1.07 rad) of the centered_level hologram, from the
        # per-site depths at scan 20260620_175714 (CV 23.1%, mean 468 uK). EXACT copy of
        # the centered_level overlay (same family: servo 1.9, same cooling/imaging).
        # 1068 sites @ registry threshold 0.40.
        "33x33_centered_level_fb1": {
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.3, "Amp2": 0.2,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.18},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # sinc-corrected + DC-free rebuild of the centered_level array (2026-06-20, by
        # user): a 1/sinc^2 radial pre-correction boosts the corner spots (the round-1
        # sinc envelope made corners ~0.47x center -> dim corners failed to fit), and the
        # near-zeroth-order spots are dropped (88-px DC gap) -> 1052 sites. Same overlay
        # as centered_level (servo 1.9, cooling X 0.22/h 0.18, imaging 0.30/0.20).
        # 1052 sites @ registry threshold 0.40. The proper starting array for the feedback.
        "33x33_uniform_centered_level_sinc_dcfree": {
            "Init": {
                "VSLMServo": 1.9
            },
            # IMAGING amps RAISED 0.30/0.20 -> 0.40/0.27 (2026-06-20) for the trap-depth
            # FEEDBACK phase: at 0.40/0.27 per-site imaging fidelity is median 1.000 with
            # only 1 dead site <90% (vs 5 at 0.30/0.20), so every site fits cleanly. This
            # lowers imaging survival, which DOES NOT matter for the mj=1 depth sweeps.
            # RESTORE to 0.30/0.20 (higher survival) after feedback converges. Sweep id=372.
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.4, "Amp2": 0.27,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.20e6, "Amp": 0.22},
                    "h": {"FreqDetuning": 0.20e6, "Amp": 0.18},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # Camera-feedback starting array for the ATOM (trap-depth) feedback campaign
        # (2026-06-21): phase/33x33_camfb_wrapper.pt -- the centered_level 33x33 array
        # camera-feedbacked via the new slmnet.experimental.slm_feedback wrapper
        # (square_array_ij center (2920,1850), n=33, pitch 90.9 px, dc_exclude 295 px ->
        # 1052 spots, same grid as the sinc_dcfree family). Camera-flat is NOT depth-flat,
        # so the atoms see a large site-to-site spread -> this is the seed the per-site
        # atom feedback corrects. IMAGING + COOLING OPTIMIZED on this array 2026-06-21
        # (scans 20260621_015938..022529, jobs 379-384): this camera-fb array has SHALLOWER/
        # more delicate traps than the slmnet arrays, so it images at MUCH lower 399 power --
        # Amp1 0.11 / Amp2 0.12 (0.30/0.20 and esp. 0.40/0.27 heated atoms out: loading
        # 0.56 @ 0.30 -> 0.29 @ 0.40). Cooling-during-imaging re-opt: X (det 0.16 MHz,
        # amp 0.26 -- higher X amp lifted pooled fidelity 0.977->0.99), h (det 0.20, amp 0.22).
        # Result: real 0-pushout survival ~0.975, per-site fidelity median 0.9975 / ALL 1052
        # spots >90% (min 0.901). Release-recapture Cool556 (below) left at the family seed
        # (not used by imaging; only by ReleaseRecaptureSeq).
        "33x33_camfb_wrapper": {
            "Init": {
                "VSLMServo": 1.9
            },
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.12,
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
        # Second camera-feedback array (2026-06-21): phase/33x33_centered_level_camfb_1068.pt
        # -- the centered_level 33x33 camera-feedbacked to 1068 sites (vs camfb_wrapper's 1052).
        # Same array family (servo 1.9, similar trap depth) -> imaging + cooling SEEDED from the
        # camfb_wrapper optimum W (amps 0.11/0.12, cool X 0.16/0.26, h 0.20/0.22); to be
        # confirmed + slightly adjusted on this array, then used for trap-depth feedback.
        "33x33_centered_level_camfb_1068": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},   # per-user override 2026-06-21 (base 10 ms); deep-merges, rest of LAC inherits base
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        # Feedback ROUND-1 correction of camfb_1068 via the slmsuite SLMFeedback (external_spot)
        # path (2026-06-21, knm-basis, no Fourier cal): flatten good sites + 2x brighten goal on
        # the dead site 591. EXACT copy of the camfb_1068 overlay (servo 1.9, imaging 0.11/0.10,
        # cooling X0.16/0.26 h0.20/0.22, LAC 20 ms) -- same array family, only the hologram differs.
        "33x33_camfb1068_fb1": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        # Trap-depth CONSISTENCY test (2026-06-21): 3 independent 8192-WGS, sinc^2-corrected
        # UNIFORM holograms on the SAME 33x33 centered grid (phase/33x33_consistency_s{1,2,3}.pt),
        # differing only in the random speckle seed. Measure per-site depth on each + correlate:
        # correlated -> systematic optics/atom-plane; uncorrelated -> hologram speckle. Cooling/
        # imaging/servo are an EXACT copy of 33x33_centered_level_camfb_1068 (per user: use the
        # center-level-fb array params). Only the hologram differs across the three.
        "33x33_consistency_s1": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        "33x33_consistency_s2": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        "33x33_consistency_s3": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        # s1 + SMOOTH-ONLY converged atom-depth correction (corrects only the deg<=3 optics bowl,
        # not per-site speckle; maxiter 8, realized/target 0.8%). Same camfb cooling/imaging.
        "33x33_consistency_s1_fb1s": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
        # s1 + ONE warm-start GS-step atom-depth correction (wgs_feedback --maxiter 2 from s1 FB0
        # depths). Same camfb_1068 cooling/imaging as the consistency arrays. Tests whether the
        # uniform-sinc-seed + warm-start atom feedback flattens depth (vs the camera-fb path).
        "33x33_consistency_s1_fb1": {
            "Init": {"VSLMServo": 1.9},
            "LAC": {"Time": 20e-3},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 0.11, "Amp2": 0.1,
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
    }

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
