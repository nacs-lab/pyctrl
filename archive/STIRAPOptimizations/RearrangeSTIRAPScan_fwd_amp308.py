"""RearrangeSTIRAPScan_fwd_amp308.py -- 2026-08-01 IS THE 308 AWG POWER-LIMITING? (1D amp scan)

Sweeps ``AWG.AWG308.Ch1.amplitude_scale`` 0.5 -> 1.0 at the locked pulse/freqs, FORWARD ONLY
(``IfReverse = 0``). Companion to the 556 Ch1 sweep (RearrangeSTIRAPScan_fwd_amp556.py).

Note the current lock sits at 0.9, NOT at the 1.0 ceiling -- so unlike the 556 sweep, this one has
headroom ABOVE the operating point and can answer both "is there more to gain" and "how much are we
leaving on the table".

Read the SHAPE near the top:
  * still climbing at 1.0 -> the 308 is POWER-LIMITED; raise amplitude_scale (free) before touching
    max_amplitude_vpp (8).
  * flat -> the 308 is NOT the limit; with the 556 also saturating (data_20260801_130137: 95.61 %
    at amp 1.0, only +0.50 % over 0.9), the remaining levers are pulse shape/overlap, beam
    uniformity over the target footprint, and the imaging-fidelity floor.

Prior note to re-test rather than trust: max_amplitude_vpp was raised 5.5 -> 7.5 on 2026-07-15 with
the comment "amp saturated by 7.5"; it now sits at 8. If this curve is flat, that story holds.

max_amplitude_vpp is NOT swept (an AWG programming parameter; a per-shot change is a different and
slower code path). Settle the slope first.

Pulse + freqs are today's verified lock: pw556 6.8 / pw308 6.2 / delay 1.6 us, carrier 143.5 /
EOM616 234.1, 556 Ch1 amp 1.0 (100-shot verify data_20260801_121937 = 95.39 +/- 0.20 %).

pulse_width_us must NOT exceed 7 us (hardware ceiling, per user 2026-07-31).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned excitation. Group by ``Params``.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_amp308.py --reps 40
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

# ---- SWEPT: the 308 forward amplitude_scale (dim 1). Current lock = 0.9, ceiling = 1.0 ----
AMP308_LIST = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_VPP_308 = 8         # NOT swept -- see the docstring
AMP556CH1 = 1.0         # 556 forward held at its lock
MAX_VPP_556 = 15
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
    g().AWG.AWG556.Ch1.amplitude_scale = AMP556CH1     # 556 held at its lock

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
    g().AWG.AWG308.Ch1.max_amplitude_vpp = MAX_VPP_308
    g().AWG.AWG308.Ch1.amplitude_scale.scan(1, [float(v) for v in AMP308_LIST])   # SWEPT dim 1
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
        "2026-08-01 forward-STIRAP 308 AMPLITUDE sweep -- is the 308 AWG power-limiting? "
        "AWG308.Ch1.amplitude_scale %s at max_amplitude_vpp %s, %d points x %s reps (the lock sits "
        "at 0.9, so this has headroom ABOVE the operating point). 556 Ch1 amp held at %.1f. "
        "Everything else at today's verified lock: pw556 %.1f / pw308 %.1f / delay %.2f us, carrier "
        "%.3f / EOM616 %.3f (100-shot verify data_20260801_121937 = 95.39 +/- 0.20 %%). Companion "
        "to the 556 sweep data_20260801_130137 (95.61 %% at amp 1.0, only +0.50 %% over 0.9 -> "
        "asymptoting). Reverse OFF, Scramble 1 (EOM616 fixed), pattern double_spacing on "
        "33x33_feedback11, VRydTrap 2, 3-image. Metric: TARGET-ONLY verify-conditioned excitation; "
        "group by Params."
        % (AMP308_LIST, MAX_VPP_308, len(AMP308_LIST), reps, AMP556CH1,
           PW556_US, PW308_US, DELAY_US, CARRIER_MHZ, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_amp308", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_amp308 -> descriptor id %s (url=%s, reps=%s, "
          "%d amp points, NumImages=%d)"
          % (did, url or "default", reps, len(AMP308_LIST), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP 308 amp sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
