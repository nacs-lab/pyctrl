"""RearrangeSTIRAPScan_fwd_cart3d.py -- 2026-08-01 forward-STIRAP coarse Cartesian 3D.

STEP 2(a) of the stirap-optimization runbook: the coarse (pw556, pw308, delay) box used ONLY to
fit the efficient plane. Forward only (``IfReverse = 0`` per the runbook -- Ch2 left untouched).

Why now: the forward pulse widths were last optimized 2026-07-21 on the mj=0 QUADRUPLE pattern
(pw556 6.0 / pw308 5.7, data 20260721_182536). Since then the science array moved to ``bz_yale``
and the delay was re-located to +0.6 us (2026-07-30 pre-scan, data_20260730_162309 -- transfer only
for POSITIVE delay, broad 0.4-1.6 us). The 07-31 forward verify read 91.9 % excitation. The ridge
has NOT been re-walked at the new pattern/delay, so that is where the gain should be.

Frequencies are NOT swept: the 07-31 freq-2D redo (data_20260731_102020 + the +1:+1 diagonal
extension _102751) re-confirmed the pair 143.5 / 234.1 (line did not move).

Grid contains the current lock (6.0, 5.7, 0.6) as an ANCHOR cell -- the runbook's drift-floor check:
re-read it every round; if the round-over-round "gain" is inside the anchor's scan-to-scan spread,
stop extending and go to the fixed-point verify.

pulse_width_us must NOT exceed 7 us (hardware ceiling, per user 2026-07-31).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned survival -- LOW survival = GOOD
excitation. Group shots by the logged ``Params`` (Rule 2), NOT run_analysis' collapsed sweep.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_cart3d.py --reps 4
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- the coarse box (4 x 4 x 4 = 64 combos; anchor (6.0, 5.7, 0.6) is a grid point) ----
PW556_US = [4.0, 5.0, 6.0, 7.0]        # scan dim 1  (<= 7 us ceiling)
PW308_US = [3.7, 4.7, 5.7, 6.7]        # scan dim 2  (<= 7 us ceiling)
DELAY_US = [0.2, 0.6, 1.0, 1.4]        # scan dim 3  (positive = 308 fires first; < PadTime 2 us)

CARRIER_MHZ = 143.5     # 07-31 re-confirmed lock
EOM616_MHZ = 234.1      # PAIR w/ carrier (degenerate line runs +1:+1, EOM = carrier + 90.6)
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
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, [float(v) for v in PW556_US])   # SWEPT dim 1
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # Ch2 left exactly as the canonical scan has it (reverse OFF -> plays no role).
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CARRIER_MHZ
    g().AWG.AWG556.Ch2.pulse_width_us = 1.5
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us.scan(2, [float(v) for v in PW308_US])   # SWEPT dim 2
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
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6
    g().Pushout.VRydTrap = 2
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay.scan(3, [float(v) * 1e-6 for v in DELAY_US])      # SWEPT dim 3
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
    rp.Scramble = 1          # ON (decorrelates drift); EOM616 is NOT swept here, so no unlock risk
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeSTIRAPScan(url=None, reps=4):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "2026-08-01 forward-STIRAP COARSE CARTESIAN 3D (runbook step 2a): pw556 %s x pw308 %s x "
        "delay %s us = %d combos, reverse OFF, freqs locked at carrier %.3f / EOM616 %.3f. Anchor "
        "cell (6.0, 5.7, 0.6) = the 07-31 lock that verified 91.9%% excitation. Pattern double_spacing on "
        "33x33_feedback11, VRydTrap 2, 3-image (verify). Metric: TARGET-ONLY verify-conditioned "
        "survival (LOW = best excitation); group by Params, not run_analysis."
        % (PW556_US, PW308_US, DELAY_US,
           len(PW556_US) * len(PW308_US) * len(DELAY_US), CARRIER_MHZ, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_cart3d", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_cart3d -> descriptor id %s (url=%s, reps=%s, "
          "%d combos, NumImages=%d)"
          % (did, url or "default", reps,
             len(PW556_US) * len(PW308_US) * len(DELAY_US), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP coarse 3D scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
