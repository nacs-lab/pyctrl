"""RearrangeMWScan.py -- rearrange + forward STIRAP -> QICK microwave SPIN ECHO -> reverse STIRAP,
scanning the microwave FREQUENCY.

Copied from RearrangeSTIRAPScan_mjm1_trapON_reverseflat.py (the latest mj=-1 trap-ON STIRAP scan) and
extended with a QICK microwave definition. The full STIRAP round-trip is KEPT and FIXED at the verified
mj=-1 optima; the two flat-reverse axes it used to sweep are pinned at their documented best. The NEW
scan axis is the QICK carrier frequency.

Flow per shot (STIRAPPushoutStep): forward STIRAP (Ch1, ground->Rydberg) -> [fwd->rev gap: QICK
microwave spin echo, fired on TTLQickTrig=FPGA1/TTL14] -> reverse STIRAP (Ch2, Rydberg->ground). The
microwave is declared out-of-band via g().QICK.* + g().runp().QICK (the run loop batch-uploads one
program per swept freq and arms it per shot); the step fires the trigger only because g().Pushout.IfMW=1.
STIRAPGap is sized here to span the echo program (qick_program_duration), so the whole spin echo plays
inside the gap before the reverse pulse.

Spin echo (QICK "Echo" template): [pi/2, wait(T/2), pi, wait(T/2), pi/2] -- the pi refocuses dephasing,
so sweeping the pi/2 carrier maps the microwave resonance (echo spectroscopy on the Rydberg transition).
Pulse lengths derive from rabi_freq (t_pi2 = 1/(4*rabi_freq)); T (evolution) is fixed; ONLY freq scans.

METRIC: as the reverseflat parent -- target-only mid-conditioned RETURN survival = P(final=1|mid=1),
grouped by cfg["Params"]. The echo transfers/dephases the Rydberg population, so survival vs freq traces
the microwave lineshape.

Run it (pyctrl backend live; SLM server reachable; QICK server up on 192.168.0.72):
    cd pyctrl
    python YbScans/RearrangeMWScan.py --reps 8
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# FIXED forward optimum -- mj=-1 VRydTrap=0.2 verified (data_20260722_164834, 94.3% exc; trap ON).
CH1_CARRIER_MHZ = 142.944
EOM616_MHZ = 233.967
PW556_FWD = 6.0
PW308_FWD = 5.7
DELAY_FWD = 1.0e-6
VRYD_TRAP = 0.2

# FIXED reverse (flat de-excitation) optimum, pinned from the reverseflat parent's best cells:
#   flat 556 carrier 143.086, amp_scale 0.44 (data_20260722_173700); flat width 5.0us, reverse delay
#   0.3us (data_20260722_172621). These were the parent's two swept axes; here they are constants.
CARRIER556_CH2_FIXED = 143.086
AMPSCALE556_CH2_FIXED = 0.44
FLAT_PW_FIXED = 5.0        # us, flat reverse width (556 Ch2 = 308 Ch2)
RDELAY_FIXED = 0.3e-6      # s, reverse delay

# ---- QICK microwave SPIN ECHO (the new scan) ----
# Carrier FREQUENCY scan (MHz). Center = the MRabi carrier the parent carried (10863.04); sweep a span
# around it to find the microwave resonance. freq does NOT change the program duration.
FREQ_CENTER_MHZ = 10863.04
FREQ_SPAN_MHZ = 4.0                                  # total width of the sweep (MHz)
FREQ_N = 21                                          # points
FREQ_PTS = [round(float(v), 4) for v in
            np.linspace(FREQ_CENTER_MHZ - FREQ_SPAN_MHZ / 2,
                        FREQ_CENTER_MHZ + FREQ_SPAN_MHZ / 2, FREQ_N)]
MW_GAIN = 3000                                        # DAC gain (nonzero to emit; 0 = silent)
MW_RABI_FREQ_HZ = 7.187e6                             # derives t_pi2 = 1/(4*f), t_pi = 2*t_pi2
MW_WAIT_TIME_S = 1.0e-6                               # echo free-evolution time T (fixed)
MW_PHASE_DEG = 0.0                                    # final pi/2 phase (deg)
GAP_MARGIN_S = 0.5e-6                                 # STIRAPGap = echo duration + this margin

DELTA_556_MHZ = CH1_CARRIER_MHZ - 143.244
DELTA_616_MHZ = EOM616_MHZ - 234.089


def _pattern_cfg(name):
    table = {
        "47x47_feedbackwarm4": ("phase/47x47_feedbackwarm4.pt", [0, 0, 0, 0, 0]),
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, -0.75]),
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        "33x33_feedback9": ("phase/33x33_feedback9.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),
        "17x17_20um":      ("phase/17x17_20um.pt",      [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(verify, init_cfg, target_cfg):
    items = [_pattern_item(INIT_PATTERN, init_cfg), _pattern_item(TARGET_PATTERN, target_cfg)]
    if verify:
        items.append(_pattern_item(TARGET_PATTERN, target_cfg))
    return json.dumps(items)


def _echo_gap_s():
    """STIRAPGap needed to fit the spin-echo program = qick_program_duration(echo) + margin.

    freq-independent, so computed once at the center freq. Falls back to a fixed 2 us if the QICK
    helper can't be imported (keeps the scan buildable off the engine venv)."""
    params = {"template": "Echo", "freq": FREQ_CENTER_MHZ, "gain": MW_GAIN,
              "rabi_freq": MW_RABI_FREQ_HZ, "phase": MW_PHASE_DEG, "wait_time": MW_WAIT_TIME_S}
    try:
        from devices.qick_awg import qick_program_duration
        return qick_program_duration(params) + GAP_MARGIN_S
    except Exception:  # noqa: BLE001
        return MW_WAIT_TIME_S + 1.0e-6      # T + generous pulse allowance


import scan_bootstrap
scan_bootstrap.bootstrap()

from RearrangeSTIRAPSeq import RearrangeSTIRAPSeq


def build():
    from scan_group import ScanGroup

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- FORWARD (Ch1) FIXED at the mj=-1 VRydTrap=0.2 optimum ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FWD
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # ---- REVERSE (Ch2) FIXED flat de-excitation (parent's optima; no longer swept) ----
    g().AWG.AWG556.Ch2.shape = "flat"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CARRIER556_CH2_FIXED
    g().AWG.AWG556.Ch2.pulse_width_us = FLAT_PW_FIXED
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = AMPSCALE556_CH2_FIXED
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_FWD
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "flat"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = FLAT_PW_FIXED
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave SPIN ECHO -- the NEW scan axis (carrier frequency) ----
    # Declared out-of-band; the run loop batch-uploads one program per swept freq + arms per shot.
    g().QICK.template = "Echo"
    g().QICK.freq.scan(1, FREQ_PTS)                   # axis 1: carrier frequency (MHz)
    g().QICK.gain = MW_GAIN
    g().QICK.rabi_freq = MW_RABI_FREQ_HZ
    g().QICK.wait_time = MW_WAIT_TIME_S               # echo free-evolution T (fixed)
    g().QICK.phase = MW_PHASE_DEG
    g.runp().QICK = True                              # opt in -> engine_run wires setup/arm/cleanup

    g().Init.EOM616.Freq = EOM616_MHZ * 1e6

    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_FWD
    g().Pushout.STIRAPReverseDelay = RDELAY_FIXED
    g().Pushout.STIRAPGap = _echo_gap_s()            # sized to hold the whole spin echo
    g().Pushout.IfReverse = 1                        # round-trip: excite -> MW -> de-excite
    g().Pushout.IfMW = 1                             # fire TTLQickTrig in the fwd->rev gap
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = round((234.444 + DELTA_616_MHZ) * 1e6, 3)
    g().Pushout.Pump556Freq = round((143.3 + DELTA_556_MHZ) * 1e6, 3)
    g().Pushout.Pump556Amp = 0.5

    g().Pushout.Time369 = 2e-6
    g().Pushout.Vy = 4

    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = init_cfg["baked_zernike"]
    rp.warmup_kwargs.extras.final_phase_zernike = target_cfg["baked_zernike"]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "quadruple_no_topright"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    rp.NumPerGroup = 2000
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeMWScan(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().QICK.freq.size(1)                        # carrier-frequency points
    gap_us = _echo_gap_s() * 1e6
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 rearrange + STIRAP round-trip with a QICK microwave SPIN ECHO in the fwd->rev gap; "
        "SCAN the microwave carrier FREQUENCY. Forward fixed at the mj=-1 verified optimum (Ch1 556 %.3f "
        "/ EOM616 %.3f, pw556 %.1f/pw308 %.1f, delay +%.1fus; data_20260722_164834), reverse fixed at "
        "the flat de-excitation optimum (Ch2 556 %.3f amp %.2f, flat pw %.1fus, RD %.2fus; "
        "data_20260722_172621/173700), VRydTrap=%.1f trap-ON, IfReverse=1, IfMW=1. Echo = pi/2-T/2-pi-"
        "T/2-pi/2 (rabi_freq %.3gHz -> t_pi2 %.1fns, T=%.2fus fixed, gain %d); STIRAPGap=%.2fus holds the "
        "%.2fus echo. Scan QICK.freq %.4f..%.4f MHz (%d pts, center %.3f span %.1f). Metric = target-only "
        "mid-conditioned RETURN survival (group by Params) -> microwave lineshape. quadruple_no_topright "
        "on 33x33_feedback11."
        % (CH1_CARRIER_MHZ, EOM616_MHZ, PW556_FWD, PW308_FWD, DELAY_FWD * 1e6,
           CARRIER556_CH2_FIXED, AMPSCALE556_CH2_FIXED, FLAT_PW_FIXED, RDELAY_FIXED * 1e6, VRYD_TRAP,
           MW_RABI_FREQ_HZ, 1e9 / (4 * MW_RABI_FREQ_HZ), MW_WAIT_TIME_S * 1e6, MW_GAIN,
           gap_us, gap_us - GAP_MARGIN_S * 1e6,
           FREQ_PTS[0], FREQ_PTS[-1], n1, FREQ_CENTER_MHZ, FREQ_SPAN_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeMW_echo_freq",
                      description=desc, **opts)
    print("submitted rearrange+MW spin-echo freq scan -> id %s (url=%s, reps=%s, %d freq pts %.4f..%.4f "
          "MHz, gap=%.2fus)"
          % (did, url or "default", reps, n1, FREQ_PTS[0], FREQ_PTS[-1], gap_us))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 rearrange+STIRAP QICK spin-echo freq scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep")
    args = ap.parse_args()
    RearrangeMWScan(url=args.url, reps=args.reps)
