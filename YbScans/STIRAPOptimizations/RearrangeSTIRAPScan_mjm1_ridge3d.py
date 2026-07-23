"""RearrangeSTIRAPScan_mjm1_ridge3d.py -- mj=-1 (pi-pol 308) STIRAP coarse 3D pulse scan.

Stage 1 of the mj=-1 campaign: with the two-photon pair LOCKED (from the freq-2D locate,
RearrangeSTIRAPScan_mjm1.py), coarse-scan the pulse over (pw556, pw308, STIRAPDelay) to map the
efficient ridge. Cheap (~3-4 shots/pt) -- used ONLY to fit the optimum plane pw556(delay)/pw308(delay),
then a fine 2D scan spans that plane + a perpendicular thickness (Stage 2), then the center of the
optimum is verified at 100 shots (Stage 3). All per the stirap-optimization runbook.

Metric = TARGET-ONLY, mid(verify)-conditioned survival (Rule 1); group shots by cfg["Params"] (Rule 2).

Run it (pyctrl backend live; SLM server reachable):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_ridge3d.py --reps 3
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- LOCKED mj=-1 two-photon pair (from freq-2D locate data_20260721_192444, VRydTrap=0.2) ----- #
# Best cell 99.2% exc / 0.78% surv (128 mid ev) on the diagonal carrier<->EOM616 resonance line;
# whole anti-diagonal >93%. Interior pick on the ridge.
CH1_CARRIER_MHZ = 143.544        # located 556 carrier
EOM616_MHZ = 234.55              # located EOM616 (308)
VRYD_TRAP = 2.0                  # 2026-07-21 re-optimize at VRydTrap=2.0 per user

# ---- coarse 3D pulse axes (map the ridge) ----------------------------------------------- #
# Centers seeded from the mj=0 plateau (pw556 5.5 / pw308 5.2 / delay 1.25us); coarse + wide so the
# fit can find the plane. Refine after the first ridge fit.
PW556_PTS = [float(v) for v in np.linspace(2.0, 8.0, 5)]      # us, axis 1 (fastest) -- broad+sparse (user 2026-07-21)
PW308_PTS = [float(v) for v in np.linspace(2.0, 8.0, 5)]      # us, axis 2 -- broad+sparse
DELAY_PTS = [float(v) for v in np.linspace(0.6e-6, 1.9e-6, 5)]  # s, axis 3

# secondary/pump freq deltas relative to the mj=-1 reference (kept faithful; pumps OFF):
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

    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, PW556_PTS)       # axis 1
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = round(143.4 + DELTA_556_MHZ, 6)
    g().AWG.AWG556.Ch2.pulse_width_us = 4.531
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us.scan(2, PW308_PTS)       # axis 2
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 4.094
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
    g().Pushout.STIRAPDelay.scan(3, DELAY_PTS)                 # axis 3
    g().Pushout.STIRAPReverseDelay = 0.132e-6
    g().Pushout.STIRAPGap = 0.1e-6
    g().Pushout.IfReverse = 0
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
    g().rearrange_kwargs.extras.pattern = "quadruple_no_topright"   # 2026-07-21 per user
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


def RearrangeSTIRAPScan_mjm1_ridge3d(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch1.pulse_width_us.size(1)
    n2 = g().AWG.AWG308.Ch1.pulse_width_us.size(2)
    n3 = g().Pushout.STIRAPDelay.size(3)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 (pi-pol 308) STIRAP COARSE 3D pulse scan (Stage 1). Pair LOCKED 556 %.3f / EOM616 %.3f "
        "MHz (freq-2D locate), 30 G, VRydTrap %.1f. Sweep pw556 x pw308 x STIRAPDelay to fit the "
        "efficient ridge -> fine 2D plane+perp (Stage 2) -> 100-shot verify (Stage 3). Metric "
        "target-only mid-conditioned (select_subset_stirap, group by Params)."
        % (CH1_CARRIER_MHZ, EOM616_MHZ, VRYD_TRAP))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_ridge3d",
                      description=desc, **opts)
    print("submitted RearrangeSTIRAPScan_mjm1_ridge3d -> id %s (url=%s, reps=%s, %dx%dx%d=%d cells, "
          "556 %.3f/EOM616 %.3f MHz, VRydTrap %.1f)"
          % (did, url or "default", reps, n1, n2, n3, n1 * n2 * n3, CH1_CARRIER_MHZ, EOM616_MHZ, VRYD_TRAP))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 (pi) STIRAP coarse 3D pulse scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjm1_ridge3d(url=args.url, reps=args.reps)
