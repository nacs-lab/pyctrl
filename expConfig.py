"""expConfig.py -- executable, actively-maintained twin of ``matlab_new/expConfig.m``.

pyctrl's PRODUCTION config source: channel aliases, channel default values, physical constants,
and AWG/Orca defaults. This replaces the frozen ``tests/reference/config_reference.json`` SNAPSHOT
for the live runner (the snapshot is demoted to the MATLAB-ground-truth reference for the drift
oracle), curing the silent staleness of the captured config -- the config is now CODE.

Because pyctrl serializes byte-identically to MATLAB, this module MUST stay numerically identical
to ``matlab_new/expConfig.m``. While MATLAB is still production, ``expConfig.m`` remains the
human-edited source and ``tests/test_exp_config.py`` (the drift oracle) asserts this module
resolves to the SAME config as the committed MATLAB capture; recalibrate ``expConfig.m``,
re-capture (``tools/capture_config_reference.m``), and mirror the change here until the oracle
passes. At cutover (scenario 3, MATLAB retired) this becomes the single source.

:func:`build_config` returns the raw dict ``SeqConfig`` consumes (same shape
``capture_config_reference.m`` emits). The long-lived runner reloads this module once per job
(``SeqConfig.load_real(reload=True)``), so edits go live without a restart -- only ``lib/`` (the
framework) and the expConfig snapshot capture still need a restart / re-capture.

Naming/units follow ``expConfig.m``: alias prefixes are load-bearing (``TTL*`` FPGA, ``V*`` NI DAQ,
``Freq*``/``Amp*`` DDS); defaults are Hz / s. Values mirror the ``.m`` literal forms for readability
(they resolve to the same IEEE-754 doubles); ints are float-coerced downstream by ``SeqConfig``.
"""


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
    c["Resonance556mj0Freq"] = 107.7503e6  # fit 2026-06-08 (Spectrum556Scan mj=0, 0-field, Lorentzian dip R^2=0.97, FWHM 55 kHz, 246 shots); was 107.735e6 (06-05); 107.717e6
    c["Resonance399Freq"] = 310e6              # not magic; changes with trap depth

    # Init: 2D MOT & Zeeman, electric fields, SLM servo
    c["Init"] = {
        "TwoDMOT": {"FreqDetuning": -20e6, "Amp": 1},
        "Zeeman": {"FreqDetuning": -36.5e6, "Amp": 0.6},
        "EOM616": {"Freq": 252.07e6, "FreqOld": 252.07e6},
        "Electrodes": {"Vx": -0.0233, "Vy": 0.0027, "Vz": 0.004859},
        "VSLMServo": 6,                        # 112 sites at 6A at 30dB
    }

    # BlueMOT
    c["BlueMOT"] = {
        "BFieldGradient": 30,                  # G/cm maximum
        "BiasCoilCurrent": {"Ryd": 3, "X": 0.1, "Y": 0, "Z": 0},
        "FreqDetuning": -44e6,                 # fast-loading opt 2026-06-05; was -40e6
        "Amp": 0.6,
        "LoadingTime": 230e-3,                 # fast-loading opt 2026-06-05 (loading saturates ~0.21); was 500e-3
    }

    # GreenMOT
    c["GreenMOT"] = {
        "BFieldRampTime": 100e-6,              # blue->green B-field ramp
        "BFieldGradient": 3,
        # fast-loading opt 2026-06-05: X was 0.039, Y was 0.27
        "BiasCoilCurrent": {"Ryd": 0, "X": 0.040, "Y": 0.268, "Z": 0.18},
        # fast-loading opt 2026-06-05: HandoverTime was 30e-3
        "PowerBroaden": {"HandoverTime": 15e-3, "FreqDetuning": 0.7e6, "Amp": 0.8},
        # fast-loading opt 2026-06-05: HoldTime was 200e-3, Amp was 0.2
        "CoolDown": {"RampdownTime": 50e-3, "HoldTime": 120e-3,
                     "FreqDetuning": 0.35e6, "Amp": 0.25},
    }

    # Absorption imaging
    c["AbsImag"] = {"TOF": 0, "ExposureTime": 50e-6, "BetweenImagsTime": 50e-3,
                    "Freq": 315e6, "Amp": 0.5}

    # SLM
    c["SLM"] = {
        "AOM": {"Freq": 120e6, "Amp": 0.55},
        "VServo": None,                        # cross-ref -> Init.VSLMServo (set below)
        "Modulation": {"Time": 10e-3, "Freq": 100e3, "Amp": 0},
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
        "FreqDetuning": -5e6, "Amp": 0.18,
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
        "Blue": {"Freq": 320e6, "Amp": 0},
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

    # ---- cross-references (mirror expConfig.m's const-to-const assignments) ----
    c["SLM"]["VServo"] = c["Init"]["VSLMServo"]
    c["LAC"]["BlueLAC"]["Resonance556mj0Freq"] = c["Resonance556mj0Freq"]
    c["Imag399"]["ExposureTime"] = c["Orca"]["ExposureTime"]
    return c


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
