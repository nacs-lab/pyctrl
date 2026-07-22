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
    a["TTL556RydAWGChSwitch"] = "FPGA1/TTL16"
    a["TTL556RydAWGSwitch"] = "FPGA1/TTL17"
    a["TTL399Imag2Shutter"] = "FPGA1/TTL18"
    a["TTL399IMG2PIDMode"] = "FPGA1/TTL19"  # 1 for lock, 0 for hold
    a["TTL308RydAWGSwitch"] = "FPGA1/TTL55"
    a["TTLOrcaTrig"] = "FPGA1/TTL54"
    a["TTLSampleAndHold"] = "FPGA1/TTL15"
    a["TTLTrig"] = "FPGA1/TTL3"
    a["TTL369Switch"] = "FPGA1/TTL44" # Unconnected be careful!!!
    a["TTL308RydAWGChSwitch"] = "FPGA1/TTL53"
    
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
    a["TestChannel"] = "Dev1/2"
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
    c["Resonance556mj0Freq"] = 107.9574e6  # fit 2026-07-22 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.959, FWHM 55.1 kHz, 209 shots, scan 20260722105037, 33x33_feedback11); +7.3 kHz vs prior. was 107.9501e6 (07-21, 33x33_feedback11); 107.9611e6 (07-20, 33x33_feedback11); 107.9253e6 (07-16, 33x33_feedback11); 107.9284e6 (07-14, 33x33_feedback11); 107.9054e6 (07-13, 33x33_feedback11); 107.8861e6 (07-05, 33x33_feedback9); 107.8762e6 (07-03); 107.8560e6 (07-02); 107.8448e6 (06-30); 107.8499e6 (06-29); 107.8478e6 (06-28, NEW LUT); 107.8389e6 (06-26); 107.8199e6 (06-23, 33x33_feedback9); 107.7753e6 (06-12, 47x47_uniform); 107.7673e6 (06-11); 107.7677e6 (06-10); 107.7552e6 (06-09); 107.7531e6 (06-09); 107.7573e6 (06-09); 107.7503e6 (06-08); 107.735e6 (06-05); 107.717e6
    c["Resonance399Freq"] = 310e6              # not magic; changes with trap depth

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 252.07e6, "FreqOld": 252.07e6},
        "Electrodes": {"Vx": -0.0175, "Vy": 0.0006, "Vz": 0.0092},  # 2026-07-16 full DC-Stark E-field re-null (new 616/308 line ~234 MHz; revival relocated from the old 282). Sequential Vx->Vy->Vz, each fit after the prior nulled; all parabola vertices INTERIOR, min-Stark centers agree 234.26/234.23/234.31 MHz. Vx -0.042->-0.0175 (20260716162632, R2=0.9999, a=1.816 MHz/V^2); Vy 0.0066->0.0006 (20260716171645, R2=0.9997, a=1215); Vz 0.0059->0.0092 (20260716173232, R2=0.9992, a=877). Scramble MUST be OFF for the 616-EOM sweep (random EOM jumps unlock 616). [prior 06-28: Vx -0.042/Vy 0.0066/Vz 0.0059; old min-Stark ~282.08 MHz]
        "VSLMServo": 3.7,                        # 112 sites at 6A at 30dB
    }

    # BlueMOT
    c["BlueMOT"] = {
        "BFieldGradient": 30,                  # G/cm maximum
        "BiasCoilCurrent": {"Ryd": 3, "X": 0.1, "Y": 0, "Z": 0},
        "FreqDetuning": -44e6,                 # fast-loading opt 2026-06-05; was -40e6
        "Amp": 0.6,
        "LoadingTime": 500e-3,                 # fast-loading opt 2026-06-05 (loading saturates ~0.21); was 500e-3
        "Img1PIDSet": 0.57,  
        "Img2PIDSet": 0.41
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
        # Every-scan default loading pattern. RUNTIME-ONLY: consumed by the pyctrl run loop
        # (_loading_defaults in YbExptCtrl/slm_runtime.py), never read by a sequence -> no
        # serialize() byte effect, so it does NOT touch the byte oracles. expConfig.py is the
        # live source of truth and hot-reloads per job, so the default array changes WITHOUT a
        # backend restart. The slm_runtime.py module constants (DEFAULT_LOADING_PATTERN_PHASE /
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
    c["AOM616Divert"] = {"Amp": 0.12}

    # SLM trap modulation
    c["SLMTrapModulation"] = {
        "Time": 5e-3, "Freq": 100e3, "AmpFactor": 0.5,
        "lowerTrapDepth": {"Vservo": 0, "Time": 3e-3},
    }

    # Rearrangement
    c["SLMRearrange"] = {"Time": 100e-3}

    # AWG defaults (Siglent SDG6X) -- TWO-CHANNEL SWITCH SCHEME.
    # Each box carries box-level hardware (resource_address, num_points, sample_rate_MHz) plus TWO
    # independent per-channel waveform dicts, Ch1 (-> SDG C1, "forward") and Ch2 (-> C2, "reverse").
    # AWGManager arms each channel as a single-cycle EXTERNALLY-triggered burst (TIME,1); an external
    # RF switch (e.g. TTL556RydAWGChSwitch = FPGA1/TTL16) selects which channel reaches the AOM, and
    # the 50 us STIRAP gap lives OUTSIDE the arb (FPGA/switch-controlled) so each short DDS arb keeps
    # the carrier well above Nyquist. Define pulses per channel in a scan, e.g.:
    #     g().AWG.AWG556.Ch1.shape = "rise_gaussian"
    #     g().AWG.AWG556.Ch1.pulse_width_us.scan(1, np.linspace(...))
    #     g().AWG.AWG556.Ch2.shape = "fall_gaussian"
    # Per-channel waveform fields (fall back to these defaults): shape / carrier_freq_MHz /
    # pulse_width_us / smooth_width_us / steepness / amplitude_scale / max_amplitude_vpp, plus
    # stirap_gap / f_delay / r_delay for double_half_gaussian_* and optional trig_delay_us (per-
    # channel burst DLAY, s->us; 0 = fire on the edge, switch does the timing).
    # shape gallery: pyctrl/tmp/pulse_10_examples.png ; pulse math: devices/sigilent_awg/pulse_waveform.py
    _AWG_CH_DEFAULTS_556 = {
        "carrier_freq_MHz": 143.4, "pulse_width_us": 1.437, "steepness": 3.5,
        "amplitude_scale": 1.0, "smooth_width_us": 0.0, "max_amplitude_vpp": 15,
        "trig_delay_us": 1.5,  # per-channel burst DLAY (us); >= the box's ~1.435us floor so it is
                               # HONORED (not clamped) -> deterministic edge->output latency. The
                               # seq must add this to its post-edge waits (edge + DLAY + 3*pw).
    }
    c["AWG556"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCC900309::0::INSTR",
        "num_points": 10000,
        "sample_rate_MHz": 2500,  # floor num_points to this MSa/s: keeps carrier Nyquist for short arbs
        "Ch1": dict(_AWG_CH_DEFAULTS_556, channel="C1", shape="rise_gaussian"),  # forward
        "Ch2": dict(_AWG_CH_DEFAULTS_556, channel="C2", shape="fall_gaussian"),  # reverse
    }
    _AWG_CH_DEFAULTS_308 = {
        "carrier_freq_MHz": 200, "pulse_width_us": 1.463, "steepness": 3.5,
        "amplitude_scale": 1.0, "smooth_width_us": 0.0, "max_amplitude_vpp": 5.5,
        "trig_delay_us": 1.5,  # honored burst DLAY (us); >= ~1.435us floor. See AWG556 note.
    }
    c["AWG308"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCD801391::0::INSTR",
        "num_points": 10000,
        "sample_rate_MHz": 2500,
        "Ch1": dict(_AWG_CH_DEFAULTS_308, channel="C1", shape="rise_gaussian"),  # forward
        "Ch2": dict(_AWG_CH_DEFAULTS_308, channel="C2", shape="fall_gaussian"),  # reverse
    }

    # Per-TTL-channel hardware timing managers (ExpSeq.add_ttl_mgr -> serialized into the byte blob
    # -> libnacs/FPGA shifts every edge on that channel). Consumed by the run loop's compile_point
    # (YbExptCtrl/engine_run.py), like LineTrigger; a channel absent here (or all-zero) adds NOTHING to
    # the bytes (byte-identical), and a manager only serializes if its channel is actually USED in
    # the sequence. Per-channel fields (all times in SECONDS):
    #   on_delay / off_delay -- fire the rising / falling edge THIS MUCH EARLIER (advance). The
    #     engine does t = max(t - on_delay, 0) (see libnacs zynq/bc_gen.cpp). Use this to
    #     PRE-COMPENSATE a downstream hardware latency so the effect lands at the nominal seq time.
    #   skip_time -- drop off-intervals shorter than this.  min_time -- stretch on-times to >= this.
    #   off_val   -- the channel's idle level (bool).
    # AWG gate use (2026-07-15): the 556/308 Rydberg AWG gates are EDGE triggers (NCYC=1); the box's
    # own trig_delay_us (1.5 us) fires the waveform 1.5 us AFTER the trigger. Setting on_delay =
    # trig_delay_us fires the gate 1.5 us early so the light lands at the nominal sequence time; the
    # step then drops the matching +DLAY from its post-gate wait. Set to 0 to disable (leave the
    # step's hand-added DLAY doing the accounting instead). Per-scan override: runp().TTLManagers.
    c["TTLManagers"] = {
        "TTL556RydAWG": {"on_delay": 1.5e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG556 trig_delay_us
        "TTL308RydAWG": {"on_delay": 1.5e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG308 trig_delay_us
        'TTL556RydAWGSwitch': {"on_delay": 1.05e-6, "off_delay": 1.05e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG556 trig_delay_us
        'TTL308RydAWGSwitch': {"on_delay": 0.85e-6, "off_delay": 0.85e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG308 trig_delay_us
        #'TTLSampleAndHold': {"on_delay": 4e-6, "off_delay": 0, "skip_time": 0.0,
        #                 "min_time": 0, "off_val": False},
    }

    # 60 Hz AC-line trigger. When enabled, the FPGA waits for an edge on a TTL INPUT line at the
    # start of each basic sequence (ExpSeq.enable_global_wait_trigger -> a version-2 ZYNQZYNQ
    # block -> libnacs emits a WaitTrigger bytecode op), so every shot begins at the same mains
    # phase (B-field stability). Consumed ONLY by the run loop's compile_point
    # (YbExptCtrl/engine_run.py) -- it is never read by a sequence step, so it does NOT enter the
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
        "Channel": 0,                          # trigger-input MUX index (WaitTrigger `chn:8`). 0 = TTL in 0 = TTL channel 24 (where the 60Hz is wired), 1 = TTL 52 / bd0-24, 2 = SMA04(FMC 1 la32p), 3 = SMA00 (FMC1 clock0p), clock SMA10 (FMC2 clock0p), TTLout24 SMA13 (FMC2 clock1n), TTLout52 SMA12 (FMC2 clock1pn)
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
        "tri_3013_camfb": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms COMMITTED 2026-07-16 evening (R216-R223).
            # The railed 399 servo (PD gain too low) leaves the floor light UNREGULATED -- it had
            # drifted hot by tonight: the 50 ms committed config re-measured 0.798 survival
            # (R223 _231231; overnight it was 0.955). At 100 ms x DDS amps 0.14/0.08 the
            # INSTANTANEOUS power drops ~2.7x (shallow traps boil at high intensity; photon count
            # ~matched) -> survival 0.9585, d' 3.85 (R222 _230657). 50 ms full-floor = 0.80. Until
            # the PD-gain fix, the attenuated long exposure is also the drift-safer config.
            "Init": {"VSLMServo": 3.5},
            # 2026-07-15 green-MOT POSITION scan on tri_3013_camfb @ committed LAC (0.22/0.10) + 1s
            # load: GreenMOT.BiasCoilCurrent X 0.037-0.041 x Y 0.24-0.28 (TweezerLoadingSeq). PEAK
            # loading 0.600 at X 0.0383 / Y 0.255; broad flat top X 0.038-0.040 x Y 0.245-0.265
            # (0.56-0.60). camfb loads ~2x better than v2 (0.60 vs 0.27) -- the ARRAY quality, not
            # position. Base 0.0388/0.265 already on the plateau; committed X 0.039 (mid-plateau) /
            # Y 0.255 (peak) as a PER-PATTERN overlay (base left global). scan _141456.
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},
            # 2026-07-15 LAC drive-plane opt: 2-D LAC.Amp x FreqDetuning (TweezerLoadingSeq,
            # loading-rate). Seed 0.2/0.11 sat on the detuning RISING EDGE (loading ~0.12).
            # Moving det up (~0.20 MHz) + amp down (0.10) MORE THAN DOUBLED loading -> broad flat
            # plateau ~0.25-0.29 across Amp 0.04-0.16 x det 0.14-0.30 (no knife-edge). Committed
            # central-plateau Amp 0.10 / det 0.22 MHz. BlueMOT.LoadingTime 1.0 s (per user; note:
            # 1s vs default gave NO real gain -- mean 0.253 vs 0.240, loading is NOT blue-delivery-
            # limited; ~0.28 ceiling is depth/geometry). scans _132218 / _132755 / _133549(@1s).
            # 2026-07-16 ENHANCED LOADING (blue-detuned LAC; BlueTweezerLoadingSeq): campaign found
            # BlueLAC det -3.0 MHz (base -3.8) at amp 0.17 / 0.5 s = the dose-ridge peak. Loading
            # 0.611 (normal LAC baseline) -> 0.71 self-thresholded (~2150/3013 atoms; target 2200).
            # Sharp det structure (-3.6 WORST 0.53, -3.0 best 0.70); amp x time = dose ridge, overdose
            # collapses (0.29/1.0s -> 0.15). Red clean-up 10-30 ms equivalent; 1 ms hurts. scans
            # _050131 (det) / _050549 (amp x time) / _051552 (red) / _051742 (60-shot confirm).
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3,   # red LAC unchanged
                    "BlueLAC": {"FreqDetuning": -3.0e6}},                  # enhanced-loading det
            # 2026-07-15 imaging-power opt (PID scheme; DDS Imag399.Amp1/Amp2=1). 2-D
            # Img1PIDSet x Img2PIDSet on ImagingPushoutSurvivalSeq @ 50 ms, 0-pushout, z4=-2.
            # Survival collapses as either setpoint rises; flat plateau at low Img2 (0.06-0.20),
            # optimum low corner. Picked Img1 0.40 / Img2 0.10 (mid-plateau, off the low edge):
            # survival ~0.34, d' ~3.0. d' ceiling is depth/loading-limited (~14% fill), NOT
            # imaging-power -- cooling / 532 depth are the next levers. (Was 1.3/0.25, copied
            # from feedback11 @ 25 ms.) scans 20260715_111735 (7x7) / _112905 (Img2-low ext).
            # 2026-07-18 POST-ND imaging-power re-opt (amps=1, 100 ms, 0-pushout): 1-D beam-1 (r1) +
            # beam-2 (r2) 0..3 -> 2-D (r3, edge-pinned low corner) -> low-corner refine (r4, 0.2..0.6)
            # -> 100-shot verify (r5, id 2768 _173159): Img1 0.3 / Img2 0.3 = survival 0.964, fidelity
            # 0.993 (pooled), d' 3.84, dist 6.4 -- clears the 99% fid / 95% surv gate. Servo regulates
            # at >=0.2 so 0.3 is a real setpoint (100 ms kept). Beam-2 carries d' (up to 5.06@2.4 solo);
            # more of either beam only heats (survival cliff above ~0.6/0.6). (Was 0.40/0.10 @ 50 ms.)
            # 2026-07-19 IN-SEQUENCE (2-round rearrange, defocus -4) 3013 img PID re-opt: 0.3 was badly
            # underpowered (d' 3.4 med, fid 0.980 -> corrupted loading counts). 0.5 = d' 4.67 med / 5.49
            # mean, fid 0.997 med (matches 33x33-level). 0.7/0.9 add brightness but median d'/fid plateau
            # + more img1 heating. Adopted 0.5/0.5. scans id 2804-2807 (_04xxxx).
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},  # LoadingTime 1.0 s per user 2026-07-15
            "Imag399": {
                # 2026-07-18: ND filters ADDED to the imaging path -> PIDSet is the light knob again
                # at higher (regulating) setpoints, so amps reverted to 1 (PID-servo convention,
                # gotcha-stale-dds-amps-pid-imaging). Re-optimizing Img1/Img2PIDSet at 100 ms.
                # [prev 2026-07-16 workaround: amps 0.14/0.08 -- servo railed, DDS amps were the only
                #  light knob; superseded by the ND filters.]
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 evening @ 100 ms/low dose: committed X/h RE-CONFIRMED drift-free
                    # (R220 _225554: X(0.16,0.20) 0.9619 beats (0.175,0.24); R221 _230133:
                    # h(0.16,0.18) 0.9561, more h amp HURTS -- unlike kagome). Unchanged below.
                    # 2026-07-15 imaging-cooling opt (Imag399.Cool556 X/h) interleaved X<->h
                    # coordinate ascent on ImagingPushoutSurvivalSeq, 0.1 s hold, z4=-2, at the
                    # committed PID imaging power (Img1 0.40 / Img2 0.10). det robustly ~0.16 MHz
                    # (0.22 always died); h amp 0.24->0.18 was the big win (survival ~0.16->0.30).
                    # Converged within cross-round shot-noise (peak bounced 0.16-0.30 at ~13% loading,
                    # optimum LOCATION stable). scans _114434 (X r2) / _114948 (h r3) / _115455 (X r4).
                    # 2026-07-15 RE-OPTIMIZED on tri_3013_camfb @ 0.60 loading (clean stats vs v2's
                    # ~13% noisy): interleaved X<->h, 0.1s hold, Img 0.40/0.10. det 0.16 robust both
                    # (0.10/0.22 die); X amp 0.23->0.20, h unchanged 0.18 (confirmed). survival
                    # 0.166(X r6) -> 0.239(h r7). scans _143721 (X) / _144235 (h).
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.20},  # camfb r6 (was 0.23 on v2)
                    "h": {"FreqDetuning": 0.16e6, "Amp": 0.18},  # camfb r7 confirmed (was 0.24 feedback11 copy)
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # 2026-07-16 NEW kagome arrays (names = trap count). SEEDED from tri_3013_camfb (the PID-servo
        # imaging scheme: DDS Imag399.Amp1/Amp2 stay at base=1, imaging power via BlueMOT.Img1/Img2PIDSet;
        # see gotcha-stale-dds-amps-pid-imaging). Same VSLMServo 3.5 as 3013 but HIGHER trap depth (per
        # user -> better imaging headroom). Loading phase phase/<name>.pt, z4=-2 baked. Imaging PID +
        # Cool556 X/h below are 3013's optima as a STARTING POINT -- being re-optimized 2026-07-16 (the
        # PID 2-D + Cool556 X<->h campaign). GreenMOT/LAC copied from camfb as loading seeds.
        "kagome_2078_camfb": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms 2026-07-18 (match 3013/2198 for the ND-filter PIDSet re-opt; was 0.05)
            # 2026-07-16 per-pattern detection mask (see kagome_res_2198 / open-imaging-psf-double-lobe)
            "boxSize": 13, "maskSigma": 3.5,
            # 2026-07-16 retro z4 sweep (-4..-1, scans r117-120): z4 = -2 CONFIRMED optimal
            # (d' 4.84 fid 0.9969 surv 0.9575; -3: 4.53, -1: 3.88) -- keep passing loading_defocus -2.
            # 2026-07-16 trap-depth feedback (amp scaling, _feedback_kagome/): CV 8.84 -> 5.50 -> 3.40
            # -> 2.98% in 3 rounds (f0 107.9428 MHz, ~314 uK, spread 1.79x -> 1.35x). KEEPER = fb3 =
            # live phase/kagome_2078_camfb.pt (+_r3.pt record; prev in phase_history/). 6 chronic
            # bad-survival sites (521,855,1083,1111,1192,1913; surv 0.56-0.86) x1.5-boosted -> 5/6
            # revived + fit ~280 uK; 521 still unfit (accepted). scans _041344/_042150/_0429xx/_044x.
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 2026-07-18 POST-ND imaging-power re-opt (amps=1, 100 ms, 0-pushout, z4=-2): low-corner
            # 2-D (r10) -> 100-shot verify (r11, id 2774 _181631) at Img1 0.4 / Img2 0.4. Common-mode-
            # normalized truth: survival 0.977, fidelity mean 0.998 / median 0.999 / worst-5% 0.990.
            # Fidelity clears 99.5%; survival ~1.3% under the 99% gate = trap-depth/loading-limited (NOT
            # imaging power -- power plateaus). Servo regulates at >=0.2 so 0.4 is a real setpoint.
            # [prev 0.40/0.10 @ 50 ms (workaround era).]
            # 2026-07-16 PID 2-D (Img1 0.40-1.00 x Img2 0.10-0.40, 0-pushout 50 ms): survival falls
            # with power on BOTH axes (heating cliff), pooled fidelity flat -> low corner optimal;
            # kept 3013's 0.40/0.10. Loading 0.68 (CV 0.05) after ~5 min thermalization. scan _011122.
            "BlueMOT": {"Img1PIDSet": 0.4, "Img2PIDSet": 0.4, "LoadingTime": 1.0},
            "Imag399": {
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 imaging-cooling re-opt on kagome_2078_camfb (deeper traps than 3013
                    # at same VSLMServo 3.5): interleaved X<->h @ 0-pushout 50 ms, PID 0.40/0.10.
                    # Cooling was THE lever (PID power was not): X amp 0.20->0.30 the big win, det
                    # 0.16->0.14; det >=0.22 collapses survival (0.4-0.6). X re-check at new h stable
                    # (fixed point, 2 rounds). h broad plateau, balanced (0.14, 0.22). HEAD-TO-HEAD
                    # 100-shot confirm vs 3013-seed: survival 0.869->0.955 (+8.6%, ~80 SEM), per-site
                    # fid med 0.981->0.990, d' med 3.31->3.84. scans _011814(X) _012234(Xexp)
                    # _012505(h) _012913(Xrecheck) _013119(W_new) _013425(seed).
                    "X": {"FreqDetuning": 0.14e6, "Amp": 0.30},
                    "h": {"FreqDetuning": 0.14e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        "kagome_res_2198": {
            "Orca": {"ExposureTime": 0.1},  # 100 ms COMMITTED 2026-07-16 evening (user): floor light +
            # double-lobe PSF leave d' ~4 at 50 ms; power lever dead (servo floor, R201 _102820) so
            # buy separation with TIME. Camera follows via runner sync_camera_exposure. 150-shot
            # confirm _220111 (with the light-down + cooling below): per-site surv 0.9815(3),
            # fid_med 0.9966, d' 4.51 -- vs 50 ms baseline _095945 0.9555/0.9877/3.94.
            # (Prev: 0.05; 75 ms test 07-16 REVERTED --
            # Img1PIDSet servo is FLOORED below ~0.6 V -- 0.25->0.60 setpoint gave only 1.23x light,
            # scan _034653 -- so the power-down chain couldn't be exercised; camera reverted too)
            # 2026-07-16 per-pattern detection mask (scan_prep hook): the imaging PSF is DOUBLE-LOBED
            # (2.2 px sep, see open-imaging-psf-double-lobe); (13, 3.5) captures both lobes, +7% d'
            # measured offline vs the default (9, 2.0). KEPT.
            "boxSize": 13, "maskSigma": 3.5,
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 2026-07-16 SURVIVAL-BIASED opt (goal: >=0.99 surv, fidelity tradeable -- rearrangement
            # mid-array). LOADING DEFOCUS z4 = -3 (NOT the inherited -2): z4 sweep -4..0 + refine
            # +-0.5 -> d' 3.7 -> 5.18, fid 0.988 -> 0.9984, surv 0.9365 -> 0.9463 (scans _014826..
            # _015616). Pass loading_defocus -3 on EVERY 2198 scan. PID-down 2-D at z4=-3 (_015808):
            # survival FLAT 0.93-0.97 over Img1 0.15-0.45 x Img2 0.04-0.12 (power not the lever at
            # this focus; ceiling depth/cooling-limited) -> picked low-dose best cell 0.25/0.10
            # (fid 0.9962, surv 0.9675). Trap-depth feedback + cooling re-opt = the path to 0.99.
            # 2026-07-18 POST-ND re-opt (amps=1, 100 ms, z4=-3): low-corner 2-D (r6) -> 100-shot verify
            # (r7, id 2770 _174249) at Img1 0.5 / Img2 0.5. Survival + fidelity RISE with Img2 (opposite
            # of 3013) and plateau at Img2>=0.5. Common-mode-normalized truth: survival 0.987, fidelity
            # mean 0.9996 / median 0.99995 / worst-5% 0.998. Cooling X<->h re-confirmed converged
            # (X 0.15/0.26~prior 0.15/0.27, h 0.15/0.22 unchanged; r8/r9 id 2771/2772) -> cooling NOT
            # the lever. Fidelity clears 99.5%; survival ~0.987 vs 0.99 = trap-depth-limited, not imaging.
            # [prev 0.25/0.10 (servo-floor workaround era).]
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},
            "Imag399": {
                # 2026-07-16 evening DELIBERATE post-servo light-down (WITH the 100 ms exposure
                # above): the Img1/Img2 PID servo is RAILED (PD gain too low, user-diagnosed --
                # open-img1pid-servo-floor) so setpoints cannot regulate down (R211 _213205: dist
                # flat over Img1 0.10-0.25 x Img2 0.04-0.10) and DDS amps only attenuate below the
                # AOM saturation knee ~0.5 (R212 _214049 flat 0.5-1.0). Amps 0.175/0.14 cut the
                # light ~2x (dist 10.4 -> 5.1) -> survival 0.972 -> 0.9815 at fid_med 0.9966
                # (R213/R214 maps _214853/_215457; 150-shot confirm _220111). This intentionally
                # violated the amps=1 PID convention (gotcha-stale-dds-amps-pid-imaging).
                # 2026-07-18: ND filters ADDED to the imaging path -> PIDSet regulates again, amps
                # reverted to 1; re-optimizing Img1/Img2PIDSet at 100 ms. [prev workaround 0.175/0.14.]
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # 2026-07-16 evening re-opt AT 100 ms, defocus -3 (X 2-D _210347, h 2-D _211211,
                    # drift-free h A/B _212630: amp 0.22 beats 0.18 by ~10 SEM; det 0.15~0.175).
                    # X optimum unchanged from the 50 ms maps (_105535); h amp moved up at the
                    # doubled dose. Knife-edge det ridge (~2-3%/25 kHz) -- cross-scan drift real,
                    # only in-scan comparisons decide.
                    # [50 ms history: overnight fb5 X/h (0.16, 0.30/0.22) confirm _033639
                    # 0.9483(6); daytime 50 ms maps _105535/_204925.]
                    "X": {"FreqDetuning": 0.15e6, "Amp": 0.27},
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
        },
        # 2026-07-19 kagome_res_2198 MOVED +50 knm-px (row) toward the zeroth order to deepen the traps:
        # mean depth 292 -> 322 uK (+10%, mj1 dip shift 1.895 -> 2.087 MHz, id 2780), all 2198 spots still
        # load (a +75 variant dropped ~8 near-DC spots to the physical DC hole -> rejected). CV rose
        # 4.4 -> 6.9% (mostly speckle from the grating re-projection; ~12% radial) -- trap-depth feedback
        # to re-flatten is deferred. Loading hologram phase/kagome_res_2198_closer.pt, z4=-3 (pass
        # loading_defocus -3). Imaging/cooling/PID/mask SEEDED from kagome_res_2198 (being re-optimized
        # 2026-07-19 now that the array is deeper). Registry+detection key = this basename.
        "kagome_res_2198_closer": {
            "Orca": {"ExposureTime": 0.1},          # 100 ms. (2026-07-19 tested 50 ms r19: fidelity dropped 0.999->0.989, d' 4.2->4.1, no survival win -- fewer photons cost separation faster than shorter dose helped. Reverted.)
            "boxSize": 13, "maskSigma": 3.5,        # double-lobe PSF mask (see open-imaging-psf-double-lobe)
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},  # seed from 2198; re-opt 07-19
            "Imag399": {
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.15e6, "Amp": 0.27},   # seed from 2198
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.22},
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                "X": {"FreqDetuning": 0.16e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.16e6, "Amp": 0.12},
            },
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

        # 2026-07-10: feedback9b re-optimization family. The optical path was MOVED, which drifted the
        # atom-plane trap-depth CV of 33x33_feedback9 back UP to 7.05% (edge-hot radial bowl + fresh
        # speckle; f0 re-measured 107.8947 MHz). Re-flattening by amplitude scaling (speckle-preserving),
        # warm-started from the deployed fb9 each round. Entry = EXACT COPY of 33x33_feedback9 (same
        # VSLMServo / LAC / imaging+cooling -- depth-only campaign, cooling/imaging already optimized).
        # Phase phase/33x33_feedback9b_r1.pt. Per-round r<N> reuses this same overlay.
        "33x33_feedback9b_r1": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
                # EXACT structural copy of 33x33_feedback9: top-level Amp1/Amp2 inherit base (=1, power
                # is set via VIMG1/2Set); only the 556 cooling-during-imaging is overlaid.
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
        # 2026-07-10: 33x33_feedback11 = PRODUCTION successor to feedback10 (one more amp-scale round r4,
        # measured with a DENSE mj=1 sweep 104.8-105.7/0.05). Measured CV 1.54% (1065/1068 sites -- dense
        # sweep recovered the marginal edge sites feedback10's coarse sweep dropped); split-half TRUE CV
        # ~2.2% (residual is REAL corner under-illumination [optical, needs a beam realign] + speckle floor
        # -- amp-scale actuator exhausted, Dphi flat 0.007 rad, transfer +0.27). IS the r4 keeper phase
        # (identical .pt, sha df7dde588a54). Overlay = EXACT COPY of 33x33_feedback9. See CAMPAIGN_STATE_fb9b.md.
        "33x33_feedback11": {
            "Orca": {"ExposureTime": 0.050},  # 2026-07-16 25->50 ms (imaging fidelity low @ 25 ms; longer collect for separation). Re-optimizing PID setpoints + Cool556 X/h at 50 ms. sync_camera_exposure drives the live camera from here. (was 0.025 @ 07-14; 35 ms opt W 1.0/0.35, 25 ms opt W 1.3/0.25.)
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            # 2026-07-14 imaging optimization (399 now PID-servoed via Img1/Img2PIDSet;
            # DDS Imag399.Amp1/Amp2 held at 1). Beam-isolation tests showed both 399
            # beams image (beam 2 cleaner, peak d' 4.24; beam 1 saturates ~3.5-4); the
            # d' cap was COOLING-limited, not 399-power. Re-optimized at 25 ms exposure
            # (was 35 ms): 2-D setpoint map -> W Img1 1.3 / Img2 0.25 (higher Img1 recovers
            # the separation lost to the shorter dose; Img2 heating cliff sits higher at
            # 25 ms). Retuned Imag399.Cool556 (0-pushout X-then-h) -> per-site fidelity
            # 0.9995, d' 5.56, survival 0.992, spatially flat (confirm data_20260714_194013,
            # 100 shots). 25 ms matches/beats 35 ms at a gentler dose. Notion 07/14.
            # (35 ms optimum was W 1.0/0.35, X 0.14/0.28, h 0.14/0.20.)
            # 2026-07-16 25->50 ms re-optimization (imaging fidelity was LOW at 25 ms).
            # Doubling the collect window fixed fidelity to ~0.998 and dropped the imaging
            # power needed (25 ms 1.3/0.25 -> 50 ms 0.8/0.15; fidelity now FLAT vs setpoint,
            # so d' is cooling/depth-limited, not 399-power). R1 pidset 2-D (data_20260716
            # _130147) -> 0.8/0.15; R2 Cool556.X (data_..130525) d' 4.4->4.8; R3 Cool556.h
            # (data_..130933) -> h det 0.16->0.14, amp 0.24->0.20. Single-point 100-shot verify
            # data_20260716_131428: fidelity 0.9981, d' 4.14, survival 0.9922, load 0.58.
            # 2026-07-18 ND-FILTER RECAL. Added ND at the END of each 399 imaging path,
            # DOWNSTREAM of the imaging-PID photodiode: Img1 ND=0.3 (~50.1% T), Img2 ND=0.7
            # (~20% T). The PD reads pre-ND power, so the setpoints must rise ~1/T to restore the
            # atom-plane intensity. Naive 1/T predicted 0.8/0.15 -> ~1.6/0.75; the measured optimum
            # was HIGHER (servo gain-compressed): R1 PID map (Img1 1.0-2.2 x Img2 0.45-1.05,
            # data_20260718_112826) rose monotonically to the corner; R2 (2.2-3.4 x 1.05-1.65,
            # _113416) went DEAD FLAT -> the servo SATURATES above ~2.2/1.05 (mirror of the <0.6 V
            # low floor, open-img1pid-servo-floor). Rise was done by Img1 ~1.3. Picked COMFORTABLE
            # mid-range 2.0/1.0 (well above the 0.6 floor, below the 2.2 rail, servo actively
            # regulating; fid ~0.997/surv ~0.99, tied with the corner). d' flat vs power = cooling-
            # limited, so re-tuned Cool556 at 2.0/1.0: R3 X (_113829) det 0.14->0.16 amp 0.25
            # (plateau det 0.16-0.20 x amp 0.22-0.28); R4 h (_114920) det 0.14->0.18 amp 0.20->0.18
            # (det-0.18 column best, low-amp safe). 100-shot VERIFY (data_20260718_120029): fidelity
            # 0.9980, d' 4.32, survival 0.9909, load 0.59 -- at parity with the pre-ND 50 ms baseline
            # (fid 0.9981/d' 4.14/surv 0.9922). DDS Imag399.Amp1/Amp2 stay at base=1 (PID scheme).
            "BlueMOT": {"Img1PIDSet": 2.0, "Img2PIDSet": 1.0},  # ND recal 2026-07-18 (was 0.8/0.15 pre-ND @50ms)
            "Imag399": {
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.16e6, "Amp": 0.25},  # ND recal 2026-07-18 (was 0.14e6/0.24)
                    "h": {"FreqDetuning": 0.18e6, "Amp": 0.18},  # ND recal 2026-07-18 (was 0.14e6/0.20)
                },
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                # 2026-07-20 RNR re-optimization (30 us release, interleaved X<->h coordinate ascent,
                # 4 rounds, data_20260720_174455/174908/175316/175704). The RNR Cool556 block predated
                # the 07-16/18 50 ms + ND re-optimizations (was X 0.16/0.14, h 0.16/0.12 -- detuning too
                # high). Converged to a symmetric fixed point det 0.12 MHz / amp 0.14 on BOTH beams
                # (X and h coupled; total 556 power is the lever). Peak recapture survival ~0.59 @ 30 us
                # (loading flat ~0.58 across the grid = real cooling signal). Plateau det 0.10-0.14 x
                # amp 0.12-0.18 flat within SEM. Notion 07/20.
                "X": {"FreqDetuning": 0.12e6, "Amp": 0.14},  # RNR re-opt 2026-07-20 (was 0.16e6/0.14)
                "h": {"FreqDetuning": 0.12e6, "Amp": 0.14},  # RNR re-opt 2026-07-20 (was 0.16e6/0.12)
            },
        },
        # 2026-07-10: 33x33_feedback10 = PRODUCTION successor to 33x33_feedback9 after the optical-path
        # move. It IS the 33x33_feedback9b_r3 keeper phase (identical .pt, sha e5995eaab4a7), renamed for
        # production. Amplitude-scaling depth re-flatten restored CV 7.05% (post-move) -> 2.22% (split-half
        # true ~1.5%) in 3 rounds. Overlay = EXACT COPY of 33x33_feedback9 (depth-only campaign; cooling/
        # imaging unchanged, already optimized). Phase phase/33x33_feedback10.pt. See CAMPAIGN_STATE_fb9b.md.
        "33x33_feedback10": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
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
        # 33x33_feedback9b round 4 (warm from r3, push for <2%). Same overlay (exact copy of 33x33_feedback9).
        "33x33_feedback9b_r4": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
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
        # 33x33_feedback9b round 3 (warm from r2). Same overlay (exact copy of 33x33_feedback9).
        "33x33_feedback9b_r3": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
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
        # 33x33_feedback9b round 2 (warm from r1). Same overlay (exact copy of 33x33_feedback9).
        "33x33_feedback9b_r2": {
            "Orca": {"ExposureTime": 0.035},
            "Init": {"VSLMServo": 1.9},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "Imag399": {
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
            # 2026-07-14: migrated to the PID-servo imaging scheme (399 power via
            # BlueMOT.Img1/Img2PIDSet, DDS Imag399.Amp1/Amp2 held at 1 -- the stale legacy
            # 0.23/0.22 DDS amps starved the imaging light ~4x under the new scheme and were
            # the likely cause of the back2um d'~2.8). Setpoint seed = feedback11's 35 ms
            # optimum (1.0/0.35); to be re-optimized per-array.
            "BlueMOT": {"Img1PIDSet": 1.0, "Img2PIDSet": 0.35},
            "Imag399": {
                "FreqDetuning": -5e6, "Amp1": 1, "Amp2": 1,
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

    # 2026-07-14: "2x11x11_5um_back2um" = the NEW bifocal array (phase/2x11x11_5um_back2um.pt) whose
    # two 11x11 layers focus at z4 = -5 +- 2.7778 (same axial offset as 2x11x11_5um) but are OFFSET
    # in xy by ~4.6 knm px (front/back layers laterally separated, NOT xy-coincident). Because the
    # layers are xy-separated they are BOTH read from ONE image at ONE loading defocus via a 242-site
    # NO-DEDUP detection grid (dedup would merge the 4.6-px pairs). Layer-focus crossover measured
    # 2026-07-14: FRONT peaks z4 ~-3.5, BACK ~<-9, equal-focus point -6.5 (NOT the geometric -5).
    # Params seeded from 2x11x11_5um then re-tuned FOR THIS PATTERN ONLY (own deep copy, NOT the
    # shared-dict alias the _3d name uses -- edits here must not leak to 2x11x11_5um):
    #   - Orca.ExposureTime 0.035 -> 0.050 (2026-07-14: at -6.5 crossover fid ~0.92/d'~2.8/surv
    #     ~0.64 photon-limited; more exposure to buy separation. Remember the LIVE camera is a
    #     separate global setting -- camera_apply_settings.)
    #   - 07-14 optimization campaign at 50 ms / defocus -6.5 (setpoint map 20260714_232556,
    #     X-cool 233319, h-cool 234018, 150-shot confirm 234903): W = Img1 0.625 / Img2 0.35
    #     (plateau Img1 0.6-1.1 x Img2 0.15-0.35; heating cliff at max dose), X (0.14 MHz, 0.26)
    #     (inherited was near-optimal), h (0.13 MHz, 0.20) (inherited amp 0.14 was WEAK -- the
    #     big cooling gain). Confirm: fid 0.940/0.948, d' 3.08/3.13, surv 0.759/0.789 F/B
    #     (from 0.929/2.92/0.694 at the old scheme). Still photon-limited.
    #   - Init.VSLMServo: step-2 bump 0.39 -> 0.5 tested 2026-07-15 (baseline 20260715_001135,
    #     X re-opt 001425 + edge 002404 (det 0.02 catastrophic 0.41; optimum X (0.10,0.26)),
    #     h re-opt 002630 (flat), final confirm 20260715_003338: F 0.720/B 0.779) -- NO GAIN vs
    #     0.39 (F 0.759/B 0.789, confirm 20260714_234903), front slightly worse -> REVERTED to
    #     0.39 with the 0.39-optimal cooling below. Depth is not the remaining-loss lever.
    import copy as _copy
    c["ByPattern"]["2x11x11_5um_back2um"] = _copy.deepcopy(c["ByPattern"]["2x11x11_5um"])
    _b2 = c["ByPattern"]["2x11x11_5um_back2um"]
    _b2["Orca"]["ExposureTime"] = 0.050
    _b2["Init"]["VSLMServo"] = 0.39
    _b2["BlueMOT"]["Img1PIDSet"] = 0.625
    _b2["BlueMOT"]["Img2PIDSet"] = 0.35
    _b2["Imag399"]["Cool556"]["X"] = {"FreqDetuning": 0.14e6, "Amp": 0.26}
    _b2["Imag399"]["Cool556"]["h"] = {"FreqDetuning": 0.13e6, "Amp": 0.20}

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
