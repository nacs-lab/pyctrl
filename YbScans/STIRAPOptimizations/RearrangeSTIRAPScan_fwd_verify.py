"""RearrangeSTIRAPScan_fwd_verify.py -- 2026-08-01 forward-STIRAP TOP-N VERIFY (runbook step 4).

Co-vary path: pw556 / pw308 / delay all on scan dim 1, so each listed CANDIDATE is one scan point
and gets ``--reps`` shots. Rule 2 applies -- re-pair the analysis by the logged ``Params``, never by
run_analysis' collapsed sweep.

Why we stopped scanning (runbook drift-floor rule): round 2 (fine, data_20260801_120540) best was
6.8/6.2/1.6 = 97.03 +/- 0.47 %, only +0.6 % over round 1's best -- but the REPEATED anchor cell
(7.0, 6.7, 0.6) re-read 94.37 +/- 0.93 % vs 96.44 +/- 0.84 % one round earlier, a 2.1 % scan-to-scan
swing. The round-over-round "gain" sits INSIDE the drift floor => broad plateau, argmax is riding
noise. Settle it at 100 shots/candidate instead of extending the grid.

Candidates = the round-2 top 5 (all within ~0.8 %, overlapping SEMs) + the incumbent 07-31 lock
(6.0, 5.7, 0.6) as a same-conditions reference, so the comparison that picks the lock is decisive.

pulse_width_us must NOT exceed 7 us (hardware ceiling). Delay must stay < STIRAPPadTime (2 us).

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_verify.py --reps 100

--- round-2 context (kept for provenance) ---
RearrangeSTIRAPScan_fwd_fine3d.py -- 2026-08-01 forward-STIRAP FINE 3D (round 2).

STEP 2(b) of the stirap-optimization runbook: sample the efficient plane densely near where round 1
put it, plus thickness across it. Forward only (``IfReverse = 0``), pattern ``double_spacing``.

Round 1 (coarse 4x4x4, data_20260801_115310, 256 shots, target-only verify-conditioned):
  * best cell pw556 7.0 / pw308 6.7 / delay 0.6 -> 96.44 +/- 0.84 %
  * ANCHOR (6.0, 5.7, 0.6 = the 07-31 lock) re-read 95.59 +/- 0.45 %
  * pw556 marginal MONOTONIC 34 / 61 / 81 / 88 % -> railed at the 7 us HARDWARE CEILING, so the
    optimum cannot be extended along that axis; sample finely just below it instead.
  * delay marginal still climbing (55 / 66 / 69 / 74 % at 0.2 / 0.6 / 1.0 / 1.4) -> extend to 1.6.
    Delay MUST stay < STIRAPPadTime (2 us) -- the step waits (PadTime - Delay).
  * ridge/plane fit: pw556(delay) = -0.478*delay + 6.412 ; pw308(delay) = 0.438*delay + 4.530
  * the top ~10 cells span 95.5-96.4 % with overlapping SEMs -> a PLATEAU; this round is to place
    it, and the top-N 100-shot verify (runbook step 4) is what picks the lock.

Anchor cell (6.0, 5.7, 0.6) is NOT in this grid -- round 1 already re-read it; the drift check for
this round is the repeated (7.0, 6.7, 0.6) round-1 best, which IS a grid point here.

pulse_width_us must NOT exceed 7 us (hardware ceiling, per user 2026-07-31).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned. Group by ``Params`` (Rule 2).

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_fine3d.py --reps 5
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- CANDIDATES: (pw556, pw308, delay) triples, CO-VARIED on scan dim 1 ----
# rows 1-5 = round-2 top 5 (data_20260801_120540); row 6 = the incumbent 07-31 lock.
CANDIDATES = [
    (6.8, 6.2, 1.6),   # round-2 best      97.03 +/- 0.47 %
    (6.2, 5.2, 1.2),   #                   96.45 +/- 0.41 %
    (7.0, 5.7, 0.6),   #                   96.24 +/- 0.23 %
    (6.5, 6.7, 1.6),   #                   96.23 +/- 0.55 %
    (7.0, 6.7, 1.6),   #                   96.21 +/- 0.43 %
    (6.0, 5.7, 0.6),   # incumbent 07-31 lock (round-1 anchor 95.59 %, round-2 not in grid)
]
PW556_US = [c[0] for c in CANDIDATES]
PW308_US = [c[1] for c in CANDIDATES]
DELAY_US = [c[2] for c in CANDIDATES]

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
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, [float(v) for v in PW556_US])   # CO-VARY dim 1
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
    g().AWG.AWG308.Ch1.pulse_width_us.scan(1, [float(v) for v in PW308_US])   # CO-VARY dim 1
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
    g().Pushout.STIRAPDelay.scan(1, [float(v) * 1e-6 for v in DELAY_US])      # CO-VARY dim 1
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


def RearrangeSTIRAPScan(url=None, reps=100):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "2026-08-01 forward-STIRAP TOP-N VERIFY (runbook step 4): %d candidates CO-VARIED on scan "
        "dim 1 at %s reps each -- %s as (pw556, pw308, delay_us). Rows 1-5 = the round-2 top 5 "
        "(data_20260801_120540, 97.03-96.21%%); row 6 = the incumbent 07-31 lock 6.0/5.7/0.6. "
        "Stopped scanning per the drift-floor rule: the round-2 gain (+0.6%%) sits inside the "
        "anchor's 2.1%% scan-to-scan swing (cell 7.0/6.7/0.6 read 96.44%% then 94.37%%). Reverse "
        "OFF, carrier %.3f / EOM616 %.3f, pattern double_spacing on 33x33_feedback11, VRydTrap 2, "
        "3-image. Metric: TARGET-ONLY verify-conditioned; RE-PAIR BY Params (co-vary -- Rule 2)."
        % (len(CANDIDATES), reps, CANDIDATES, CARRIER_MHZ, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_verify", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_verify -> descriptor id %s (url=%s, reps=%s, "
          "%d candidates, NumImages=%d)"
          % (did, url or "default", reps, len(CANDIDATES), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP top-N verify.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
