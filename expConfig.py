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

COMMENT RULE (also YbScans/): keep a comment only if editing the line it sits on forces you to read it.
KEEP    the value and the scan id that set it -- current entry only, wrapped at 100 cols.
NEVER   rules, procedures, or claims about other files/subsystems: those belong in yb_skills.
HISTORY is git: commit expConfig.py (+ config_reference.json if a value moved) after daily calibration.
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
    a["TTLIonizationSwitch5to8"] = "FPGA1/TTL48"
    a["TTLMultimeterTrig"] = "FPGA1/TTL25"
    
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
    a["Freq556RydbergHF"] = "FPGA1/DDS9/FREQ"
    a["Amp556RydbergHF"] = "FPGA1/DDS9/AMP"
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
    a["VIonizationSet5to8"] = "Dev1/2"
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

    # 556nm resonance (calibrate daily by spectroscopy; 3P1 mj=0 near-magic).
    # 2026-09-17 scan 20260917091449 -- FWHM 40.0 kHz, R^2 0.940, 205 shots.
    c["Resonance556mj0Freq"] = 108.229875e6
    # 307.6932e6 = the 2026-09-15 fit (scan 20260915195725). Both 09-16 chase writes were REVERTED.
    # *** DO NOT CHASE THIS LINE *** -- shared reference for 4 subsystems; chasing it emptied the array
    # (scans 20260916104558, 20260916130300, 20260916131739). Mechanism + rule -> yb_skills. git log -S.
    c["Resonance399Freq"] = 307.6932e6         # 2026-09-15 fitted (scan 20260915195725); 2026-09-16 chase REVERTED, see above

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 370e6, "FreqOld": 370e6},
        "Electrodes": {"Vx": 0.0162, "Vy": 0.0008, "Vz": 0.0110},  # 2026-08-27 FULL DC-Stark E-field re-null AT 60 G (user-directed; every prior null was done at 30 G). Sequential Vx->Vy->Vz, each fit with the prior axes already nulled; all three vertices INTERIOR, all R2 >= 0.9986, and the three min-Stark centers agree to 222 kHz on a ~15 MHz-wide revival (230.345 / 230.393 / 230.567 MHz), which also matches the day's independently measured 60 G revival 230.3631 -- so the 556 park was right and the three axes are mutually consistent. Vx -0.0175 -> 0.0162 (20260827135024, R2=0.9986, a=1.800 MHz/V^2); Vy 0.0006 -> 0.0008 (20260827141130, R2=0.9991, a=1225); Vz 0.0092 -> 0.0110 (20260827143208, R2=0.9991, a=871.7). Curvatures reproduce the 07-16 30 G values (1.816 / 1215 / 877) to within a few percent even though the field doubled, as expected for a DC-Stark polarizability that does not care about B. Vy moved only 0.2 mV and Vz 1.8 mV (both were already near-nulled); Vx moved 33.7 mV, the only substantive change. Scramble was 0 for all three (the Stark/Revival scans set it; REQUIRED -- random EOM jumps unlock 616). All three ran 861/861 clean. EOM616 ramp slope was 6.0 (2x slower, changed this session) for these runs; cost only ~0.72 shots/s since the EOM grid is 1 MHz. NOTE StarkVxRevival616Scan does NOT apply the 50-80 G -60 MHz high-field AOM offset its siblings do, so --green-freq-mhz was passed ALREADY SHIFTED (119.0016); the low-field number parks the 556 60 MHz off -> flat ~80% survival, no revival (see bug-starkvx-scan-missing-hf-aom-offset). [prior 2026-07-16 (Vy 0.0006 -> 0.0008, scan 20260827141130, 861 shots, R2=0.9991, a=1225 MHz/V^2 = 680x steeper than Vx so swept narrow +-0.1 V, vertex INTERIOR, min-Stark center 230.393 MHz -- agrees with the Vx axis 230.345 to 48 kHz, two independent estimates of the same 308 line). Vz still the 07-16 value. Vx re-null AT 60 G (first Stark null done at 60 G, not 30 G -- user-directed): Vx -0.0175 -> 0.0162 (scan 20260827135024, 861 shots, R2=0.9986, a=1.800 MHz/V^2, vertex INTERIOR, min-Stark center 230.345 MHz which agrees with the day's measured 60 G revival 230.3631 -> confirms the 556 park was right). Vy/Vz NOT yet re-nulled at 60 G (still the 07-16 30 G values). NOTE StarkVxRevival616Scan does NOT apply the 50-80 G -60 MHz high-field AOM offset that Revival616Scan/RydbergSpectrum556Scan do, so --green-freq-mhz must be passed ALREADY SHIFTED (119.0016, not 179.0016); passing the low-field number parks the 556 60 MHz off and gives a flat ~80% survival with no revival (see bug-starkvx-scan-missing-hf-aom-offset). Also: the EOM616 ramp slope in RydbergPushoutSurvivalSeq was made 2x slower this session (3.0 -> 6.0) per user, which costs little here (~0.72 shots/s) since the EOM steps are 1 MHz apart. [prior 2026-07-16 full DC-Stark E-field re-null (new 616/308 line ~234 MHz; revival relocated from the old 282). Sequential Vx->Vy->Vz, each fit after the prior nulled; all parabola vertices INTERIOR, min-Stark centers agree 234.26/234.23/234.31 MHz. Vx -0.042->-0.0175 (20260716162632, R2=0.9999, a=1.816 MHz/V^2); Vy 0.0066->0.0006 (20260716171645, R2=0.9997, a=1215); Vz 0.0059->0.0092 (20260716173232, R2=0.9992, a=877). Scramble MUST be OFF for the 616-EOM sweep (random EOM jumps unlock 616). [prior 06-28: Vx -0.042/Vy 0.0066/Vz 0.0059; old min-Stark ~282.08 MHz]
        "VSLMServo": 3.7,                        # 112 sites at 6A at 30dB
        "VIonizationSet5to8": 0 # DC Ionization voltage for electrodes 5-8 (V) must < 5V
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
        # X 0.0360 / Y 0.2450 set 2026-09-15 by R952 (scan 20260915110155): X is razor-sharp (dead by
        # 0.038), Y broad -> 0.245 = plateau centre. Z 0.175 = plateau centre from R903/R904 (08-31),
        # not re-swept since. History: git log -S.
        "BiasCoilCurrent": {"Ryd": 0, "X": 0.0360, "Y": 0.2450, "Z": 0.175},  # 2026-09-15 R952 (was X 0.0355 / Y 0.2400 / Z 0.175 @ 09-14) -- prior note: 2026-09-14 (was X 0.0354 / Y 0.2390 / Z 0.17 @ 08-31) -- prior note: 2026-08-31 R4 (scan 20260831123108, 108 shots): interior peak X 0.0355 / Y 0.2375 = 0.590; parabola vertices X 0.03545 / Y 0.2394, corr_x null 0.0353, corr_y null 0.2404 -- rate peak and both gradient nulls agree. ** THE OPTIMUM WALKS: X measured 0.0342 at 11:09 (R2) and 0.0353 at 12:26-12:33 (R3/R4), +1.1 mA in 1.3 h; R3 was edge-pinned because of it. A 60 G RydbergSpectrum556Scan (job 1619) ran in between -- Ryd-coil heating / residual field is the prime suspect. Re-check bias X after any high-field scan; a 1 mA error costs ~15% loading, 2 mA costs half. ** 2026-08-31 MOT-position re-map (scans 20260831110657 coarse + 20260831110858 zoom, 33x33_feedback11, LoadingTime 0.6 s, 174 shots, 3-4/pt): rate max X 0.0340 / Y 0.235 = 0.435 +/- 0.011, and corr(load,x) nulls at X 0.0342, corr(load,y) at Y 0.238 -- rate peak and BOTH gradient nulls coincide, so one point serves both axes. Prior X 0.0353 sat on the upper X cliff (0.0355 -> 0.16, 0.0365 -> 0.03) and interpolates to ~0.27 => this is ~+60% rate with the -0.67 x gradient removed. X is razor-sharp (FW ~2 mA, dead by 0.0375); Y broad. was X 0.0353 / Y 0.244 (08-27, measured at a saturated 0.8 s); X 0.0344 / Y 0.2533 (06-05); Y 0.262 (08-27 1-D) superseded
        # fast-loading opt 2026-06-05: HandoverTime was 30e-3
        "PowerBroaden": {"HandoverTime": 15e-3, "FreqDetuning": 0.7e6, "Amp": 0.8},
        # Per-beam X/h cooldown split, set 2026-09-14 by interleaved coordinate ascent R908-R912
        # (verify R912: load 0.5250 +- 0.0186, CV 0.195, per-site d' median 8.72). History: git log -S.
        "CoolDown": {"RampdownTime": 50e-3, "HoldTime": 200e-3,
                     "X": {"FreqDetuning": 0.20e6, "Amp": 0.25},   # 2026-09-14 (det was 0.35e6; amp unchanged)
                     "h": {"FreqDetuning": 0.30e6, "Amp": 0.20}}   # 2026-09-14 (was 0.35e6 / 0.15 as seeded)
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
        # them across ROUNDS (archive/tools/strobe_imaging_round.py --pulse-time). BeamPulseTime 1 us ~ AOM rise
        # -> partial pulse; verify on scope (trigger on the first PD pulse -- no scope-sync TTL wired).
        # WARN: small RecoolTime -> many cycles -> big sequence; validate at short Orca.ExposureTime first.
        #
        # Cool556 here is a SEPARATE recool set from the during-imaging Imag399.Cool556: with the 399 OFF
        # during recool there is no light shift -> the optimum detuning/amp differs. Seeded from
        # Imag399.Cool556; retune via archive/tools/strobe_imaging_round.py cool mode.
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

    # RearrangeCool556 -- the post-rearrangement recool that precedes STIRAPPushoutStep
    # (RearrangeCool556hXStep). Same channels/shape as Cool556hXStep but its OWN block, so the
    # STIRAP recool can be optimized independently of the RNR release-recapture Cool556 above.
    # 2026-08-05/06: the STIRAP science block used to recool via Cool556Step, which drives BOTH beams
    # from the top-level Cool556.FreqDetuning/Amp (0.14 MHz / 0.08) instead of the per-beam X/h
    # sub-blocks the 07-20 RNR campaign optimized. Switching to the per-beam values lifted mid->final
    # survival 0.341 -> 0.514 (20260805_231444 vs 20260806_000351; every gap point +6..+11 sigma),
    # identifying the recool as the dominant limiter. Seeded from the base Cool556 X/h so behaviour is
    # unchanged where no overlay exists; the STIRAP atoms arrive hotter than the RNR ones (SLM
    # transport with RearrCoolAmp = 0, plus two 399 exposures before the pushout), so expect this to
    # want a longer/stronger recool than Cool556 -- that is what the scan knobs are for.
    c["RearrangeCool556"] = {
        "Time": 5e-3,
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
        # Ryd bias coil current (A) during push-out. RydbergPushoutSurvivalSeq reads this to pick
        # the low-field (< 31 A) vs high-field (60-130 A) push-out step, and resolves the fallback
        # EAGERLY -- so it must exist here even though the scan always overrides it. 0 = low field,
        # matching the literal `g.BiasCoilCurrent.Ryd(0)` in both push-out steps.
        "BiasCoilCurrent": {"Ryd": 0},
        "STIRAP": {"delay": 1e-6, "reverse_delay": 1e-6, "gap": 10e-6},
        "MRabi": {"Freq": 4000, "Gain": 0},
        "Ramsey": {"Phase": 0},
    }

    # 616 AOM diverted/idle amplitude (RydbergPushoutStep restores this after pushout)
    c["AOM616Divert"] = {"Amp": 0.14}

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
    # stirap_gap / f_delay / r_delay for double_half_gaussian_*, pad_time_us for fall_quintic +
    # flat (a hold prepended before the main window; total = pad + pw), and optional trig_delay_us (per-
    # channel burst DLAY, s->us; 0 = fire on the edge, switch does the timing), plus
    # chirp_freq_MHz / chirp_profile for the chirped_* shapes (swept carrier; inert
    # otherwise -- chirp_freq_MHz = 0 is byte-identical to the un-prefixed shape).
    # shape gallery: pyctrl/tmp/pulse_10_examples.png ; pulse math: devices/sigilent_awg/pulse_waveform.py
    _AWG_CH_DEFAULTS_556 = {
        "carrier_freq_MHz": 143.4, "pulse_width_us": 1.437, "steepness": 3.5,
        "amplitude_scale": 1.0, "smooth_width_us": 0.0, "max_amplitude_vpp": 15,
        "trig_delay_us": 1.5,  # per-channel burst DLAY (us); >= the box's ~1.435us floor so it is
                               # HONORED (not clamped) -> deterministic edge->output latency. The
                               # seq must add this to its post-edge waits (edge + DLAY + 3*pw).
        # Swept carrier, read ONLY by shape = "chirped_rise_quintic" / "chirped_fall_quintic" /
        # "chirped_flat" (the last sweeps across its whole constant-amplitude burst):
        # chirp_freq_MHz = SIGNED TOTAL SPAN (final - initial, MHz) across the amplitude ramp;
        # chirp_profile = "linear" (constant rate) | "quintic" (rate zero at both ends). 0 = off.
        "chirp_freq_MHz": 0.0, "chirp_profile": "linear",
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
        # Swept carrier, read ONLY by shape = "chirped_rise_quintic" / "chirped_fall_quintic" /
        # "chirped_flat" (the last sweeps across its whole constant-amplitude burst):
        # chirp_freq_MHz = SIGNED TOTAL SPAN (final - initial, MHz) across the amplitude ramp;
        # chirp_profile = "linear" (constant rate) | "quintic" (rate zero at both ends). 0 = off.
        "chirp_freq_MHz": 0.0, "chirp_profile": "linear",
    }
    c["AWG308"] = {
        "resource_address": "USB0::62700::4353::SDG6XFCD801391::0::INSTR",
        "num_points": 10000,
        "sample_rate_MHz": 2500,
        "Ch1": dict(_AWG_CH_DEFAULTS_308, channel="C1", shape="rise_gaussian"),  # forward
        "Ch2": dict(_AWG_CH_DEFAULTS_308, channel="C2", shape="fall_gaussian"),  # reverse
    }

    # QICK FPGA_AWG (RFSoC4x2) microwave AWG defaults -- OUT-OF-BAND (not in the byte blob), TTL-
    # triggered on TTLQickTrig = FPGA1/TTL14. A scan opts in with g().runp().QICK = True and declares
    # the microwave sequence via g().QICK.* (mirrors the Siglent g().AWG.<name>.* convention). The run
    # loop (YbExptCtrl/awg_runtime.py -> devices/qick_awg) builds one program per unique swept point,
    # batch-uploads them ALL once at scan start, then ARMS the active one before every shot (the board
    # is one-shot: each TTL fires the armed program once, so it must be re-armed per shot). Params:
    #   template   -- which pulse sequence to build: "Sine" | "Rabi" | "Ramsey" | "Echo" (see
    #                 devices/qick_awg/templates.py). Fixed per scan; the swept scalar mints the programs.
    #   freq       -- carrier / DDS frequency, MHz (0..6000 usable; >3000 aliases past Nyquist).
    #   gain       -- DAC gain, -2**15..2**15-1. DEFAULT 0 = silent (safe; set nonzero to emit).
    #   rabi_freq  -- microwave Rabi frequency, Hz. Derives the pulse lengths: t_pi2 = 1/(4*rabi_freq),
    #                 t_pi = 2*t_pi2 (author sets the physics, lengths follow).
    #   phase      -- final-pulse (Pi2_Phase) phase, DEGREES (server-native; NO rad->deg conversion).
    #   wait_time  -- Ramsey/Echo free-evolution time, s (sweepable). Split into loop(N,[Wait]) chunks
    #                 so each Wait pulse fits the 16-bit HW length register (see templates._split_duration).
    #   drive_time -- Rabi drive time, s (sweepable). Chunked the same way when it exceeds the cap.
    # Per-scan override any field via g().QICK.<field>; a swept field (e.g. wait_time.scan(1)) becomes the
    # per-shot program-selection key. Example (Ramsey wait scan):
    #     g().QICK.template = "Ramsey"; g().QICK.gain = 3000; g().QICK.rabi_freq = 7.187e6
    #     g().QICK.wait_time.scan(1, np.linspace(...)); g().runp().QICK = True
    c["QICK"] = {
        "host": "192.168.0.72", "port": 1234,
        "template":   "Ramsey",
        "freq":       10863.04,
        "gain":       0,
        "rabi_freq":  7.187e6,
        "phase":      0.0,
        "wait_time":  1e-6,
        "drive_time": 500e-9,
        "duration":   1e-6,          # Sine template: single-tone length, s (sweepable)
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
        "TTL556RydAWG": {"on_delay": 0e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG556 trig_delay_us
        "TTL308RydAWG": {"on_delay": 0e-6, "off_delay": 0.0, "skip_time": 0.0,
                         "min_time": 0.1e-6, "off_val": False},   # = AWG308 trig_delay_us
        'TTL556RydAWGSwitch': {"on_delay": 1.05e-6, "off_delay": 1.05e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG556 trig_delay_us
        'TTL308RydAWGSwitch': {"on_delay": 0.85e-6, "off_delay": 0.85e-6, "skip_time": 0.0,
                         "min_time": 0, "off_val": False},   # = AWG308 trig_delay_us
        'TTLQickTrig': {"on_delay": 0e-6, "off_delay": 0e-6, "skip_time": 0.0,
                    "min_time": 0, "off_val": False},   # = AWG308 trig_delay_us
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
    #   Raise:   True = wait for a rising edge, False = falling edge. This is the PHYSICAL edge you
    #            get -- but only because pyctrl compensates for a firmware inversion: molecube2
    #            maps the flag backwards relative to the gateware, so the run loop sends its
    #            COMPLEMENT (engine_run._MOLECUBE2_TRIG_EDGE_INVERTED -- read that note before
    #            touching either side; "fixing" molecube2 without clearing that flag double-inverts
    #            and silently restores the bug). Verified 2026-08-16 end to end on scope
    #            192.168.0.27: with the compensation in place, Raise=True fires on the line's
    #            rising edge, Raise=False on the falling edge, each within one 20 us sample.
    #   Timeout: seconds. FPGA clock is 100 MHz and the bytecode timeout field is 24-bit, so the
    #            max is ~0.168 s; ~0.02 s = one 60 Hz period + margin (catches the next edge, then
    #            proceeds if the signal is absent -- it does not hang the shot).
    c["LineTrigger"] = {
        "Enable": True,
        "Device": "FPGA1",
        "Channel": 0,                          # trigger-input MUX index (WaitTrigger `chn:8`). 
        # 0 = TTL in 0 = TTL channel 24 (where the 60Hz is wired), 
        # 1 = TTL 52 / bd0-24
        # 2 = SMA12 (FMC2 clock1p)
        # 3 = SMA00 (FMC1 clock0p)
        # clock SMA10 (FMC2 clock0p)
        # TTLout24 SMA13 (FMC2 clock1n)
        # TTLout52 SMA04 (FMC 1 la32p)
        "Raise": True,                         # True = rising edge, False = falling (the run loop
                                               # inverts it for the firmware quirk; see above)
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
            # Amp 0.10 / det 0.22 MHz = central plateau, set 2026-07-15 (scans _132218/_132755); broad flat
            # plateau, no knife-edge. Blue-detuned LAC (BlueTweezerLoadingSeq) reached 0.71 vs 0.611 on 07-16.
            # History: git log -S.
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3,   # red LAC unchanged
                    "BlueLAC": {"FreqDetuning": -3.0e6}},                  # enhanced-loading det
            # Img1/Img2PIDSet 0.5/0.5, adopted 2026-07-19 (scans id 2804-2807).
            # SHARED-SETPOINT POLICY: the 399 imaging PID engages ONCE in the ROOT BlueMOTStep at THIS
            # (loading) pattern's setpoints and is HELD for the whole shot, so these two values are the
            # imaging power of ALL THREE images (3013/2198/2078). The mid/final patterns' own PIDSet are
            # inert and kept EQUAL on purpose (gotcha-imaging-pid-held-multiround-rearrange). Per-image
            # brightness comes from Imag399.Amp1/Amp2, which do not drift within a shot. History: git log -S.
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},  # LoadingTime 1.0 s per user 2026-07-15
            "Imag399": {
                # 2026-07-18: ND filters ADDED to the imaging path -> PIDSet is the light knob again
                # at higher (regulating) setpoints, so amps reverted to 1 (PID-servo convention,
                # gotcha-stale-dds-amps-pid-imaging). Re-optimizing Img1/Img2PIDSet at 100 ms.
                # [prev 2026-07-16 workaround: amps 0.14/0.08 -- servo railed, DDS amps were the only
                #  light knob; superseded by the ND filters.]
                # 2026-07-27: in a MULTI-ROUND rearrange these amps are the ONLY per-image light knob
                # (the PID is held at the root/loading setpoint). AOM knee: 0.5-1.0 is optically FLAT,
                # only <= 0.5 actually attenuates. 1 = full held power for the img1 (loading) frame.
                "Amp1": 1, "Amp2": 1,
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # X(0.16,0.20) / h(0.16,0.18) re-confirmed drift-free 2026-07-16 (R220 _225554, R221 _230133);
                    # more h amp HURTS here, unlike kagome. History: git log -S.
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
            # 2026-07-16 trap-depth feedback (amp scaling, campaigns/feedback/kagome/): CV 8.84 -> 5.50 -> 3.40
            # -> 2.98% in 3 rounds (f0 107.9428 MHz, ~314 uK, spread 1.79x -> 1.35x). KEEPER = fb3 =
            # live phase/kagome_2078_camfb.pt (+_r3.pt record; prev in phase_history/). 6 chronic
            # bad-survival sites (521,855,1083,1111,1192,1913; surv 0.56-0.86) x1.5-boosted -> 5/6
            # revived + fit ~280 uK; 521 still unfit (accepted). scans _041344/_042150/_0429xx/_044x.
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 0.5/0.5 unified 2026-07-27. As the FINAL pattern of RearrangeCommSeq2 this PIDSet is INERT --
            # the PID locks once in the root BlueMOTStep at the loading pattern and HOLDS
            # (gotcha-imaging-pid-held-multiround-rearrange). It DOES apply when 2078 is the ROOT pattern of a
            # standalone scan; 0.4/0.4 was that scan's 07-18 optimum, on the same flat plateau. git log -S.
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},
            "Imag399": {
                # 2026-07-27 EXPLICIT (was inherited base 1/1): with the PID held at the loading
                # pattern's setpoint these DDS amps are the ONLY per-image light knob for the img3
                # (final) frame -- dial them, never the PIDSet above. AOM knee: 0.5-1.0 is optically
                # FLAT, only <= 0.5 actually attenuates (07-16 R212).
                "Amp1": 1, "Amp2": 1,
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
            # Per-pattern detection mask (scan_prep hook): the imaging PSF is DOUBLE-LOBED (2.2 px separation,
            # see open-imaging-psf-double-lobe); (13, 3.5) captures BOTH lobes, +7% d' vs the default (9, 2.0).
            # Exposure buys separation where power cannot (servo floor). History: git log -S.
            "boxSize": 13, "maskSigma": 3.5,
            "Init": {"VSLMServo": 3.5},
            "GreenMOT": {"BiasCoilCurrent": {"X": 0.039, "Y": 0.255}},   # seed from tri_3013_camfb
            "LAC": {"FreqDetuning": 0.22e6, "Amp": 0.10, "Time": 30e-3},  # seed from tri_3013_camfb
            # 0.5/0.5. As the MIDDLE pattern of RearrangeCommSeq2 this PIDSet is INERT (the PID locks once in
            # the root BlueMOTStep at the loading pattern and HOLDS); kept EQUAL to that shared setpoint on
            # purpose (gotcha-imaging-pid-held-multiround-rearrange). It still applies when 2198 is the ROOT
            # pattern of a standalone scan -- keep in sync with tri_3013_camfb's PIDSet, and change per-image
            # brightness with Imag399.Amp1/Amp2, never here.
            # LOADING DEFOCUS z4 = -3 for this array, NOT the inherited -2: pass loading_defocus -3 on EVERY
            # 2198 scan (2026-07-16 z4 sweep, scans _014826.._015616). History: git log -S.
            "BlueMOT": {"Img1PIDSet": 0.5, "Img2PIDSet": 0.5, "LoadingTime": 1.0},
            "Imag399": {
                # Amps = 1 (the PID convention). The 07-16 0.175/0.14 light-down was a servo-floor workaround
                # (open-img1pid-servo-floor) and was REVERTED on 07-18 when ND filters restored servo range.
                # In the 2-round rearrange (held PID) these are the ONLY per-image light knob for the img2 frame.
                # AOM KNEE: 0.5-1.0 is optically FLAT, only <= 0.5 actually attenuates (07-16 R212). git log -S.
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

        # 2026-07-10: 33x33_feedback11 = PRODUCTION successor to feedback10 (one more amp-scale round r4,
        # measured with a DENSE mj=1 sweep 104.8-105.7/0.05). Measured CV 1.54% (1065/1068 sites -- dense
        # sweep recovered the marginal edge sites feedback10's coarse sweep dropped); split-half TRUE CV
        # ~2.2% (residual is REAL corner under-illumination [optical, needs a beam realign] + speckle floor
        # -- amp-scale actuator exhausted, Dphi flat 0.007 rad, transfer +0.27). IS the r4 keeper phase
        # (identical .pt, sha df7dde588a54). Overlay = EXACT COPY of 33x33_feedback9. See CAMPAIGN_STATE_fb9b.md.
        "33x33_feedback11": {
            "Orca": {"ExposureTime": 0.050},  # 2026-07-29 30 ms TRIAL RUN AND REVERTED -- 50 ms KEPT. Tried 50->30 ms to cut the per-shot 399 heating budget (at 50 ms both power maps were FLAT: PIDSet r60 and DDS-amp r64 = power-saturated). 30 ms LOST decisively: the amp map stopped being flat and rose monotonically toward full power (= photon-STARVED, not saturated), and at the matched 1.0/1.0 cell / same cooling / same det -5: fid 0.9949 vs 0.9992, surv 0.9899 vs 0.9937, d' 4.74 vs ~6.4 (30 ms r66 data_20260729_175910 vs 50 ms r64 data_20260729_173122). The BEST 30 ms cell (fid 0.9954, d' 4.31) still lost to a typical 50 ms cell. Lesson: the 50 ms flatness was the GOOD regime -- saturated = all the photons we need; shortening left saturation and cost Gaussian separation without buying survival. (was 0.025 @ 07-14; 35 ms opt W 1.0/0.35, 25 ms opt W 1.3/0.25.)
            # 2026-08-11 3.3 -> 1.9 (user directive). 1.9 was the measured operating point all
            # of 08-11: the loading plane z4 -4.0 below was measured AT 1.9 (17-plane sweep
            # 20260811_135519 + head-to-head 20260811_140133), and the day's 556 mj=0/mj=1
            # scans ran there. Also note VSLMServo appears NOT to move trap depth over
            # ~1.9-3.9 on this array (mj=1 depth-differential line moved only ~15 kHz on a
            # 2.5 MHz splitting between 1.9 and 3.3; loading flat 0.54-0.61 across a 2.7-3.9
            # sweep, job 773) -- the 532 power servo looks saturated or its setpoint is not
            # reaching the power loop, so treat 1.9 as "the point we characterize at", not as
            # a known-lower depth. (was 3.3 @ 08-10, 3.5 before that.)
            "Init": {"VSLMServo": 1.9},
            # Defocus -4.0 set 2026-08-12 by three interleaved 120-shot runs ordered -2.5/-4/-2.5 so drift
            # cannot fake it (scans 20260812_115809, _120509, _120845): -4 wins every brightness metric, and
            # the left-right tilt present at -2.5 VANISHES at -4. Read by slm_runtime._pattern_defocus as the
            # DEFAULT plane for any scan that does not set runp().loading_defocus itself.
            # WARNING -- REARRANGEMENT IS NOT COVERED BY THIS. The global SLM->camera affine is calibrated at
            # ONE plane (-5), so a rearranged run at another plane would map against a stale affine. The
            # rearrangement scans all set rp.loading_defocus explicitly (matched to rearrange_kwargs.extras.z4),
            # which OVERRIDES this key -- they stay on their own plane until the affine is re-bootstrapped.
            # History: git log -S.
            "SLM": {"Loading": {"Defocus": -4.0}},  # was -2.5 (see above); global base -5
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            # Img1PIDSet 1.4 / Img2PIDSet 0.4 -- set 2026-09-15 (interleave r1022; verify r1023
            # data_20260915_214327: survival 0.9930 +- 0.0009, d' median 7.79, spatially flat).
            # Img2PIDSet is an INERT knob (beam-2 servo fault) -> yb_skills. History: git log -S.
            "BlueMOT": {"Img1PIDSet": 1.35,       # 2026-09-17 r1262
                        "Img2PIDSet": 0.4,        # 2026-09-15 r1022
                        "LoadingTime": 0.25,      # 2026-09-15; knee 0.11-0.12 s (r1203)
                        "Amp": 0.5,               # 2026-09-17 r1208
                        "FreqDetuning": -48.5e6},  # 2026-09-17 r1206; plateau -51..-46 MHz
            # *** THIS VALUE IS A DETUNING FROM THE ATOMIC LINE, via BlueMOTStep.py:62 ***
            #   Freq_BlueMOT = Resonance399Freq + FreqDetuning. WHEN Resonance399Freq IS UPDATED, DO NOT TOUCH
            #   THIS NUMBER: the drive follows the atom and the true detuning -- what the MOT responds to -- is
            #   preserved. The one-off re-expression -47 -> -46.0932 was because the REFERENCE was corrected
            #   (308.6 -> 307.6932) without the atom having moved; that is NOT the daily case.
            # Green-MOT alignment re-derived 2026-09-15 (r977): parked X 0.0358 / Y 0.240 -- the joint
            #   compromise, within ~1 SEM of peak rate on both axes but materially FLATTER across the array,
            #   which is what keeps the corners loading. Z 0.175 = measured peak (r960), hard cliff above.
            #   CANARY TOLERANCE: +-0.7 mA in X costs ~10%, +-1.5 mA costs half. History: git log -S.
            "GreenMOT": {
                "BiasCoilCurrent": {"X": 0.0358, "Y": 0.236, "Z": 0.170},  # 2026-09-17 r1233/r1234
                "CoolDown": {"RampdownTime": 0.03},
            },
            # Verified 150 shots 2026-09-15 (r980, data_20260915_195018): loading 0.5983 +- 0.0012, true CV
            # 2.7% (was 6.1%), d' median 7.15, 99.8% of sites > 3, no warmup and no drift.
            # CANARY BASELINE (r981), 3-point bias-X bracket 0.0345/0.0358/0.0370: R = 1.52, S = 0.599,
            # grad_x = +0.006. Interpretation table -> yb_skills. History: git log -S.
            "Imag399": {
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    # Imaging/cooling re-opt: current point set 2026-09-14 (jobs 1995-1998, R920/R922/R923/R925).
                    # History: git log -S.
                    "X": {"FreqDetuning": 0.14e6, "Amp": 0.26},  # 2026-09-17 r1258
                    "h": {"FreqDetuning": 0.15e6, "Amp": 0.17},  # 2026-09-14 amp 0.21 -> 0.17, det unchanged (was 0.15e6/0.18 @ 08-27)
                },
                # 399 imaging detuning: 7 large moves since 08-07; current point set 2026-09-15.
                # History: git log -S.
                "FreqDetuning": -10e6,  # imaging re-opt 2026-09-15 (was -5e6 @ 09-07; plateau -12.5..-7.5)
                "Amp1": 1.0,   # 2026-08-10 REVERTED from the 08-07 0.7 -- that value was the regression
                "Amp2": 1.0,   # 2026-08-10 REVERTED from the 08-07 0.85 (see r700/r701 above)
            },
            "Cool556": {
                "Time": 5e-3, "FreqDetuning": 0.14e6, "Amp": 0.08,
                # RNR Cool556 X (0.15, 0.17) / h (0.15, 0.13). h.Amp 0.17 -> 0.13 on 2026-09-15 by a 40-pass 2x2
                # interleave (r1002, data_20260915_214808); X re-mapped against the moved h pin returned (0.15,
                # 0.17) (r1003, data_20260915_215103) = pins match returns both ways, CONVERGED. git log -S.
                "X": {"FreqDetuning": 0.15e6, "Amp": 0.17},  # RNR 2026-09-15 re-confirmed at the new h pin (was 0.12e6/0.14 @ 07-20; 0.15e6/0.17 @ 08-19)
                "h": {"FreqDetuning": 0.15e6, "Amp": 0.13},  # RNR re-opt 2026-09-15: amp 0.17 -> 0.13 (det unchanged)
            },
            # Post-rearrangement recool before STIRAPPushoutStep (RearrangeCool556hXStep). Seeded to
            # the Cool556 X/h values ABOVE, which is exactly what the 2026-08-06 A/B ran
            # (20260806_000351: mid->final 0.514 vs 0.341 with the old top-level Cool556Step values),
            # so this overlay reproduces that measurement bit-for-bit. Now scannable on its own axis
            # via RearrangeSTIRAPScan.py -- the STIRAP atoms arrive hotter than the RNR ones (SLM
            # transport at RearrCoolAmp = 0 + two 399 exposures), so a longer/stronger recool than the
            # RNR-tuned numbers is the open hypothesis for the residual (0.783 at 13.5 us total
            # trap-off vs RNR's 0.943 at 13 us).
            "RearrangeCool556": {
                "Time": 5e-3,
                "X": {"FreqDetuning": 0.12e6, "Amp": 0.14},
                "h": {"FreqDetuning": 0.12e6, "Amp": 0.14},
            },
        },
        # 2026-07-10: 33x33_feedback10 = PRODUCTION successor to 33x33_feedback9 after the optical-path
        # move. It IS the 33x33_feedback9b_r3 keeper phase (identical .pt, sha e5995eaab4a7), renamed for
        # production. Amplitude-scaling depth re-flatten restored CV 7.05% (post-move) -> 2.22% (split-half
        # true ~1.5%) in 3 rounds. Overlay = EXACT COPY of 33x33_feedback9 (depth-only campaign; cooling/
        # imaging unchanged, already optimized). Phase phase/33x33_feedback10.pt. See CAMPAIGN_STATE_fb9b.md.

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
            "Orca": {"ExposureTime": 0.050},
            # VSLMServo 0.6 = the NORMAL loading/imaging depth (user directive): load and image here and ramp
            # the trap up only just before the ping-pong transport step, so the imaging W must match THIS
            # depth. A VSLMServo 3.5 re-optimization was done and REVERTED (r800-r809). History: git log -S.
            "Init": {"VSLMServo": 0.6},
            # Imaging optimized AT THE LOADING DEPTH (VSLMServo 0.6, ~400 uK) and COMMITTED here because EVERY
            # scan that is not imaging_round (Spectrum556Scan, the transport scans, ...) reads this block --
            # leaving the working point as g()-overrides made those run with starved pre-ND values.
            # Set 2026-08-10, verified 100 shots (r824, data_20260810_182553): load 0.569, fidelity median
            # 0.99829, d' median 5.28, survival 0.9865 +- 0.0015, spatially flat.
            # NOTE the loading plane moves with depth (-6.5 here vs -3.5 at VSLMServo 3.5). History: git log -S.
            "BlueMOT": {"LoadingTime": 300e-3, "Img1PIDSet": 0.8, "Img2PIDSet": 1.0},
            "GreenMOT": {"CoolDown": {"HoldTime": 150e-3}},
            "LAC": {"FreqDetuning": 0.11e6, "Amp": 0.2, "Time": 30e-3},
            "SLM": {"Loading": {"Defocus": -6.5}},
            "Imag399": {
                "FreqDetuning": -2e6,       # 2026-08-10 (was -5e6)
                "Amp1": 1.0, "Amp2": 1.0,   # 2026-08-10 (were 0.11/0.22 = pre-ND-recal power)
                "Cool556": {
                    "FreqDetuning": 0.18e6, "Amp": 0.2,
                    "X": {"FreqDetuning": 0.18e6, "Amp": 0.25},  # 2026-08-10 (was 0.16e6/0.26)
                    "h": {"FreqDetuning": 0.26e6, "Amp": 0.08},  # 2026-08-10 (was 0.16e6/0.14)
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
                # Re-optimized 2026-06-30 on the depth-BALANCED array (trap-depth feedback flattened CV 11% ->
                # 1.54%): Amp1 0.28, cooling X (0.16,0.17) / h (0.20,0.17), z4 -1.5. Verified 150 shots (job 1140):
                # fidelity median 0.9965 (72.7% of sites >= 0.995, GATE MET), d' 4.59.
                # The loading defocus DRIFTS with thermal lensing (-1.2 -> -1.5 in one session) -- re-scan it.
                # History: git log -S.
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

    # "2x11x11_5um_back2um" = the bifocal array (phase/2x11x11_5um_back2um.pt): two 11x11 layers at
    # z4 = -5 +- 2.7778, OFFSET in xy by ~4.6 knm px (NOT xy-coincident). Because they are xy-separated
    # both layers are read from ONE image at ONE loading defocus via a 242-site NO-DEDUP detection grid
    # -- dedup would merge the 4.6-px pairs. Equal-focus point is -6.5, NOT the geometric -5.
    # *** This pattern has its OWN DEEP COPY, not the shared-dict alias the _3d name uses: edits here
    #     must NOT leak to 2x11x11_5um. ***
    # Working point set 2026-07-14 at 50 ms / defocus -6.5 (150-shot confirm _234903). VSLMServo stays
    # 0.39 -- the 0.5 test on 07-15 showed no gain. Still photon-limited. History: git log -S.
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
    d["TTLMultimeterTrig"] = 1
    
    # DDS
    d["Freq556MOTX"] = 118e6
    d["Amp556MOTX"] = 0
    d["Freq556RydbergMOTh"] = 110e6
    d["Amp556RydbergMOTh"] = 0
    d["Freq556RydbergHF"] = 120e6
    d["Amp556RydbergHF"] = 0.9
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
