"""RearrangeSTIRAPScan_fwd_freq2d.py -- 2026-08-01 forward-STIRAP FREQUENCY 2D (round 3).

STEP 1 of the stirap-optimization runbook, run AFTER the pulse ridge + the 100-shot top-N verify:
re-pin the two-photon resonance (556 carrier x EOM616/308) at the freshly verified pulse.

The runbook's loop alternates freq-2D and ridge-3D. Today's order was ridge (coarse
data_20260801_115310 -> fine data_20260801_120540) -> 100-shot verify (data_20260801_121937) ->
this. The pulse below is the VERIFY WINNER, so the frequency map is measured at the pulse we
actually intend to lock, not at a stale seed.

The resonance is a DEGENERATE DIAGONAL line: carrier and EOM616 trade off +1:+1 (EOM = carrier +
90.6 at the current lock 143.5 / 234.1), so the useful readout is WHERE the diagonal sits, and any
pair on it is equivalent. The 07-31 freq-2D (data_20260731_102020 + diagonal extension _102751)
found the line had not moved; this round re-checks it a day later at the new pulse.

!! Scramble = 0 -- MANDATORY for any 616-EOM frequency sweep. Scrambling makes consecutive shots
   jump the EOM by large random amounts, which kicks the 616 laser OUT OF LOCK and aborts the scan
   (memory: gotcha-scramble-unlocks-616-eom-sweep). A monotonic sweep steps it gently.

pulse_width_us must NOT exceed 7 us (hardware ceiling, per user 2026-07-31).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned excitation. Group by ``Params``.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_freq2d.py --reps 4
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- FIXED pulse = the 100-shot verify winner (data_20260801_121937) ----
PW556_US = 6.8          # <= 7 us ceiling
PW308_US = 6.2          # <= 7 us ceiling
DELAY_US = 1.6          # 308 fires first; MUST stay < PadTime 2 us

# ---- SWEPT frequency box (10 x 10 = 100 combos) around the 07-31 lock 143.5 / 234.1 ----
CARRIER_LIST_MHZ = [round(float(v), 4) for v in np.linspace(142.9, 144.1, 10)]   # scan dim 1
EOM616_LIST_MHZ = [round(float(v), 4) for v in np.linspace(233.5, 234.7, 10)]    # scan dim 2
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
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(1, [float(v) for v in CARRIER_LIST_MHZ])  # SWEPT dim 1
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_US       # fixed at the verify winner
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

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
    g().Init.EOM616.Freq.scan(2, [float(v) * 1e6 for v in EOM616_LIST_MHZ])   # SWEPT dim 2
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
    rp.Scramble = 0          # MUST be 0: this sweeps EOM616, and scrambled EOM jumps unlock the
                             # 616 laser -> scan aborts (gotcha-scramble-unlocks-616-eom-sweep)
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
        "2026-08-01 forward-STIRAP FREQUENCY 2D round 3 (runbook step 1), run AFTER the ridge + the "
        "100-shot top-N verify: carrier %.3f-%.3f x EOM616 %.3f-%.3f MHz = %d combos at the VERIFIED "
        "pulse (pw556 %.1f / pw308 %.1f / delay %.2f us). Reverse OFF. SCRAMBLE 0 (mandatory for a "
        "616-EOM sweep -- scrambled EOM jumps unlock the 616). Pattern double_spacing on "
        "33x33_feedback11, VRydTrap 2, 3-image. The resonance is a DEGENERATE +1:+1 diagonal "
        "(EOM = carrier + 90.6 at the 07-31 lock 143.5/234.1); read off WHERE the diagonal sits. "
        "Metric: TARGET-ONLY verify-conditioned; group by Params."
        % (CARRIER_LIST_MHZ[0], CARRIER_LIST_MHZ[-1], EOM616_LIST_MHZ[0], EOM616_LIST_MHZ[-1],
           len(CARRIER_LIST_MHZ) * len(EOM616_LIST_MHZ), PW556_US, PW308_US, DELAY_US))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_freq2d", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_freq2d -> descriptor id %s (url=%s, reps=%s, "
          "%d combos, NumImages=%d)"
          % (did, url or "default", reps,
             len(CARRIER_LIST_MHZ) * len(EOM616_LIST_MHZ), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP freq-2D scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
