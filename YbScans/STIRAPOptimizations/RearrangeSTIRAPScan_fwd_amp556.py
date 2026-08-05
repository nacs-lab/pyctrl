"""RearrangeSTIRAPScan_fwd_amp556.py -- 2026-08-01 IS THE 556 AWG POWER-LIMITING? (1D amp scan)

Sweeps ``AWG.AWG556.Ch1.amplitude_scale`` 0.5 -> 1.0 at the locked pulse/freqs, forward only.

The question is NOT "which amp is best" -- amplitude_scale is already pinned at its 1.0 ceiling
(with max_amplitude_vpp 15). It is the SHAPE of the curve as it approaches 1.0:

  * still climbing at 1.0  -> the 556 is POWER-LIMITED; more 556 light would buy excitation, and
    the next move is a vpp bump (15 -> 18) or more optical power upstream.
  * flat / saturated near 1.0 -> the 556 is NOT the limit; stop chasing power and look elsewhere
    (beam uniformity over the target footprint, imaging-fidelity floor, 308 power).

Prior evidence, both to be re-tested rather than trusted: the canonical scan's note says an earlier
0.4-1.0 sweep at vpp 15 was "monotonic to ceiling, best=1.0 (still power-limited)", while the
runbook warns the 556 SDG6022X (200 MHz BW) CLAMPS near its bandwidth edge -- at the 143 MHz
carrier, vpp 17 and vpp 19 delivered the SAME light. So a bump above 15 may buy nothing even if
this curve says power-limited; confirm any vpp change by a measured survival drop, not by the
commanded number.

max_amplitude_vpp is deliberately NOT swept here: it is an AWG programming parameter, and a per-shot
vpp change is a different (and slower) code path. Settle the slope first; if it says power-limited,
test 15 vs 18 as a separate 2-point run.

Pulse + freqs are today's verified lock: pw556 6.8 / pw308 6.2 / delay 1.6 us, carrier 143.5 /
EOM616 234.1 (100-shot verify data_20260801_121937 = 95.39 +/- 0.20 %; freq-2D data_20260801_124041
re-confirmed the line).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned excitation. Group by ``Params``.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_amp556.py --reps 40
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- FIXED pulse + freqs = today's verified lock ----
PW556_US = 6.8          # <= 7 us ceiling
PW308_US = 6.2          # <= 7 us ceiling
DELAY_US = 1.6          # 308 fires first; MUST stay < PadTime 2 us
CARRIER_MHZ = 143.5
EOM616_MHZ = 234.1

# ---- SWEPT: the 556 forward amplitude_scale (dim 1). 1.0 = the current lock / vpp-15 ceiling ----
AMP556_LIST = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_VPP_556 = 15        # NOT swept -- see the docstring
# ------------------------------------------------------------------------------------ #


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
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

    # ---- Siglent AWG: forward Ch1 (556 rise + 308 fall). Ch2 = reverse, UNUSED here. ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CARRIER_MHZ  # fixed at the lock
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_US       # fixed at the verify winner
    g().AWG.AWG556.Ch1.max_amplitude_vpp = MAX_VPP_556
    g().AWG.AWG556.Ch1.amplitude_scale.scan(1, [float(v) for v in AMP556_LIST])   # SWEPT dim 1

    # Ch2 left exactly as the canonical scan has it (reverse OFF -> plays no role).
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 143.5   # reverse OFF; parked at the current lock
    g().AWG.AWG556.Ch2.pulse_width_us = 1.5
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_US       # fixed at the verify winner
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 7
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (unused) ----
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params ----
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6    # fixed at the lock (NOT swept -> Scramble may be 1)
    g().Pushout.VRydTrap = 2
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_US * 1e-6      # fixed at the verify winner
    g().Pushout.STIRAPReverseDelay = -0.25e-6   # reverse OFF; kept at the canonical value
    g().Pushout.STIRAPPadTime = 2e-6            # must exceed max(DELAY_US) us
    g().Pushout.STIRAPGap = 1e-6
    g().Pushout.IfReverse = 0                   # FORWARD optimization (runbook: keep 0, don't touch Ch2)
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6
    g().Pushout.Pump556Freq = 143.556e6
    g().Pushout.Pump556Amp = 0.5

    g().Pushout.Time369 = 3e-6
    g().Pushout.Vy = 4

    # ---- warmup_kwargs ----
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

    # ---- rearrange_kwargs ----
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "double_spacing"   # 2026-08-01 per user (canonical file's
    # bz_yale is stale for this campaign; first submit 20260801_114843 was aborted for that reason)
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params ----
    rp.NumPerGroup = 2000
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1          # ON (decorrelates drift from the swept axis). Safe here: EOM616 is
                             # FIXED, so the 616-unlock gotcha does not apply.
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeSTIRAPScan(url=None, reps=40):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "2026-08-01 forward-STIRAP 556 AMPLITUDE sweep -- is the 556 AWG power-limiting? "
        "AWG556.Ch1.amplitude_scale %s at max_amplitude_vpp %s, %d points x %s reps. Everything "
        "else at today's verified lock: pw556 %.1f / pw308 %.1f / delay %.2f us, carrier %.3f / "
        "EOM616 %.3f (100-shot verify data_20260801_121937 = 95.39 +/- 0.20 %%). Reverse OFF, "
        "Scramble 1 (EOM616 fixed), pattern double_spacing on 33x33_feedback11, VRydTrap 2, "
        "3-image. READ THE SLOPE AT 1.0: still climbing = power-limited (then test vpp 15 vs 18 "
        "separately); flat = saturated, the 556 is not the limit. Metric: TARGET-ONLY "
        "verify-conditioned; group by Params."
        % (AMP556_LIST, MAX_VPP_556, len(AMP556_LIST), reps,
           PW556_US, PW308_US, DELAY_US, CARRIER_MHZ, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_amp556", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_amp556 -> descriptor id %s (url=%s, reps=%s, "
          "%d amp points, NumImages=%d)"
          % (did, url or "default", reps, len(AMP556_LIST), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP 556 amp sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
