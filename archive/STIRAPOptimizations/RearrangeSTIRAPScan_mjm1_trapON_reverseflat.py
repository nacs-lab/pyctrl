"""RearrangeSTIRAPScan_mjm1_trapON_reverseflat.py -- de-excite via a FLAT (square) Ch2 pulse.

NEW de-excitation STRATEGY: instead of a reverse STIRAP (adiabatic quintic ramp on Ch2), drive the
Rydberg -> ground return with a FLAT, constant-amplitude carrier burst on Ch2 (556 + 308 both
shape='flat' -- a hard on/off square pulse, more like a resonant pi-pulse / Rabi drive than an
adiabatic passage). FORWARD is FIXED at the mj=-1 VRydTrap=0.2 verified optimum (Ch1: 556 142.944 /
EOM616 233.967, pw556 6.0 / pw308 5.7, delay +1.0us; data_20260722_164834, 94.3% fwd exc); trap ON
at VRydTrap=0.2; gap=0.1us; IfReverse=1.

Scan the flat-pulse knobs:
  * axis 1 = AWG556.Ch2.pulse_width_us = AWG308.Ch2.pulse_width_us  (flat burst DURATION, matched)
  * axis 2 = Pushout.STIRAPReverseDelay  (556<->308 overlap/order; SIGNED, >0 = 556 first)
Ch2 carriers matched (Ch2 556 = Ch1 556 = 142.944; Ch2 308 = 200 like Ch1).

METRIC: MAXIMIZE return survival = P(final=1|mid=1) over target sites (target-only mid-conditioned) --
HIGH survival = successfully de-excited back to ground. Group by cfg["Params"] (Rule 2); compare
against the quintic reverse STIRAP (data_20260722_170827, round-trip ~0.88).

NOTE (examine before running): for a FLAT resonant drive the optimal burst duration is likely SHORT
(a pi-pulse ~ half a Rabi period, sub-us to ~1us), so the inherited 1.5-3.5us sweep may be too long
-- adjust PW_REV_PTS after inspection.

Run it (pyctrl backend live; SLM server reachable):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_trapON_reverseflat.py --reps 8
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

# REVERSE FINE 2D (coarse data_20260721_225215 showed a FLAT ~55-70% region, RD near 0 best, no sharp
# matched diagonal). Fine-scan the strongest lever ReverseDelay x reverse pw (matched Ch2), around peak.
# 2026-07-21 fine (data_20260721_230333): reverse prefers SHORT pulse (pw3 best, monotonic down) + RD ~+0.2.
# pw3 was edge-railed -> extend DOWN to un-rail. RD tightened around the peak.
# 2026-07-22: scan the flat de-excite's FREQ + POWER (a Rabi pi-pulse should peak sharply in both).
# Fix flat width + RD at the flat best (data_20260722_172621: pw 5.0, RD 0.3us). Sweep Ch2 556 carrier
# (axis1) x Ch2 556 amplitude_scale (axis2).
# 2026-07-22 (3rd flat step): at the flat freq/power optimum (data_20260722_173700: carrier 143.086,
# amp_scale 0.44 -> 0.84), FIX carrier+amp and scan flat WIDTH (axis1) x reverse DELAY (axis2).
CARRIER556_CH2_FIXED = 143.086   # flat 556 carrier optimum
AMPSCALE556_CH2_FIXED = 0.44     # flat 556 amplitude_scale optimum (lower = correct pi-area)
FLAT_PW_PTS = [round(float(v), 4) for v in np.linspace(0.5, 6.0, 8)]         # us, axis 1 -- flat width
RDELAY_PTS = [round(float(v), 10) for v in np.linspace(-0.2e-6, 0.6e-6, 5)]  # s, axis 2 -- reverse delay

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

    # ---- FORWARD (Ch1) FIXED at the VRydTrap=2.0 optimum ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FWD
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # ---- REVERSE (Ch2) SWEPT; carriers matched. NEW de-excitation strategy: FLAT (square,
    #      constant-amplitude carrier burst) instead of the quintic ramp -- a hard on/off pulse. ----
    g().AWG.AWG556.Ch2.shape = "flat"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CARRIER556_CH2_FIXED      # FIXED flat 556 carrier optimum
    g().AWG.AWG556.Ch2.pulse_width_us.scan(1, FLAT_PW_PTS)          # axis 1: flat width
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = AMPSCALE556_CH2_FIXED      # FIXED flat 556 amp optimum
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_FWD
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "flat"                              # flat (square) reverse 308, FIXED
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us.scan(1, FLAT_PW_PTS)         # axis 1: flat width (matches 556 Ch2)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    g().Init.EOM616.Freq = EOM616_MHZ * 1e6

    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_FWD
    g().Pushout.STIRAPReverseDelay.scan(2, RDELAY_PTS)             # axis 2: reverse delay
    g().Pushout.STIRAPGap = 0.1e-6                                 # fixed short gap per user
    g().Pushout.IfReverse = 1                                      # round-trip: optimize the return
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


def RearrangeSTIRAPScan_mjm1_reverse3d(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch2.pulse_width_us.size(1)     # flat width
    n2 = g().Pushout.STIRAPReverseDelay.size(2)        # reverse delay
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 FLAT-pulse de-excitation -- WIDTH x DELAY scan at the flat freq/power optimum "
        "(VRydTrap=0.2 trap-ON). Ch2 556+308 shape='flat'. FORWARD fixed at the mj=-1 verified optimum "
        "(Ch1 556 %.3f / EOM616 %.3f, pw556 6.0/pw308 5.7, delay +1.0us; data_20260722_164834), "
        "gap=0.1us, IfReverse=1. Ch2 carrier FIXED %.3f + amp_scale FIXED %.2f (flat freq/power optimum, "
        "data_20260722_173700 -> 0.84). Scan flat width pw556=pw308 %s us (axis1) x reverse delay %s us "
        "(axis2), %dx%d=%d cells. Metric = MAXIMIZE target-only mid-conditioned RETURN survival (group by "
        "Params). Compare vs quintic reverse STIRAP (~0.88), flat width-only (~0.79), flat freq/power (~0.84). "
        "quadruple_no_topright on 33x33_feedback11."
        % (CH1_CARRIER_MHZ, EOM616_MHZ, CARRIER556_CH2_FIXED, AMPSCALE556_CH2_FIXED,
           FLAT_PW_PTS, [round(v*1e6,3) for v in RDELAY_PTS], n1, n2, n1 * n2))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_reverseFLAT_pwdelay",
                      description=desc, **opts)
    print("submitted reverse FLAT pw/delay -> id %s (url=%s, reps=%s, %dx%d=%d cells, flat pw %s us x RD %s us)"
          % (did, url or "default", reps, n1, n2, n1 * n2, FLAT_PW_PTS, [round(v*1e6,3) for v in RDELAY_PTS]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 REVERSE STIRAP coarse 3D optimization.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjm1_reverse3d(url=args.url, reps=args.reps)
