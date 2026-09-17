"""RearrangeSTIRAPScan_fwd_delay.py -- 2026-08-01 forward STIRAP DELAY sweep at 556 amp 1.0.

Sweeps ``Pushout.STIRAPDelay`` (the 556<->308 pulse overlap/order) at the post-redistribution 556
power, with ``AWG556.Ch1.amplitude_scale = 1``. Forward only (``IfReverse = 0``).

Why re-scan the delay: the 556 optical power on the Rydberg path went 18 -> 36 mW, which raises the
556 Rabi frequency ~1.4x. The adiabatic overlap that a STIRAP pulse pair wants depends on the two
Rabi frequencies, so the optimal delay does not stay put when one of them changes -- the earlier
delay optimum (+1.6 us, from the ridge scans data_20260801_115310 / _120540 at the OLD power) was
located before the redistribution. The amp rerun (data_20260801_141453) already showed the knee
moving left; this asks where the timing optimum went.

Sign convention: POSITIVE delay = 308 fires FIRST (counter-intuitive but correct -- see
STIRAPPushoutStep, the ``if Forward_Delay > 0`` branch gates 308 then waits before the 556 gate).
Only positive delays are scanned: the 2026-07-30 step-0 pre-scan (data_20260730_162309) found
transfer ONLY for positive delay, with <= 0 dead (~0.97 survival).

!! HARD BOUND: delay must stay BELOW ``STIRAPPadTime`` (2 us). The step waits
   ``(PadTime - Delay)`` before dropping AmpSLM, so delay >= PadTime would be a negative wait.
   The grid therefore stops at 1.8 us. If the optimum rails at 1.8, RAISE PadTime and re-scan --
   do not read a rail at the bound as an optimum.

Everything else at today's verified lock: pw556 6.8 / pw308 6.2 us, carrier 143.5 / EOM616 234.1,
308 Ch1 amp 0.9 (100-shot verify data_20260801_121937 = 95.39 +/- 0.20 %, at the OLD 556 power).

Metric (runbook Rule 1): TARGET-ONLY, verify(mid)-conditioned excitation. Group by ``Params``.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_fwd_delay.py --reps 30
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- FIXED pulse + freqs + amps = today's verified lock, 556 at full amp ----
PW556_US = 6.8          # <= 7 us ceiling
PW308_US = 6.2          # <= 7 us ceiling
CARRIER_MHZ = 143.5
EOM616_MHZ = 234.1
AMP556CH1 = 1.0         # per user: scan the delay AT amp 1.0
AMP308CH1 = 0.9         # 308 lock (saturated above ~0.7, data_20260801_132243)
MAX_VPP_556 = 15
MAX_VPP_308 = 8

# ---- SWEPT: Pushout.STIRAPDelay (dim 1). Positive = 308 first. MUST stay < PadTime 2 us ----
DELAY_LIST_US = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
PAD_TIME_US = 2.0       # hard bound on the delay -- see the docstring
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
    g().AWG.AWG556.Ch1.amplitude_scale = AMP556CH1     # full amp (per user)

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
    g().AWG.AWG308.Ch1.amplitude_scale = AMP308CH1
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
    g().Pushout.STIRAPDelay.scan(1, [float(v) * 1e-6 for v in DELAY_LIST_US])   # SWEPT dim 1
    g().Pushout.STIRAPReverseDelay = -0.25e-6   # reverse OFF; kept at the canonical value
    g().Pushout.STIRAPPadTime = PAD_TIME_US * 1e-6   # HARD BOUND: every delay must stay below it
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


def RearrangeSTIRAPScan(url=None, reps=30):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "2026-08-01 forward-STIRAP DELAY sweep at 556 amp 1.0, AFTER the 556 optical "
        "redistribution (Rydberg path 18 -> 36 mW). Pushout.STIRAPDelay %s us (positive = 308 "
        "fires first), %d points x %s reps. Re-locating the overlap optimum: the +1.6 us lock came "
        "from the ridge scans data_20260801_115310/_120540 at the OLD 556 power, and ~1.4x more 556 "
        "Rabi moves the adiabatic overlap. HARD BOUND: delay < STIRAPPadTime (%.1f us) -- the step "
        "waits (PadTime - Delay); if the optimum rails at %.1f us, RAISE PadTime and re-scan rather "
        "than reading the bound as an optimum. Fixed: pw556 %.1f / pw308 %.1f us, 556 amp %.1f / "
        "308 amp %.1f, carrier %.3f / EOM616 %.3f. Reverse OFF, Scramble 1, pattern double_spacing "
        "on 33x33_feedback11, VRydTrap 2, 3-image. Metric: TARGET-ONLY verify-conditioned "
        "excitation; group by Params."
        % (DELAY_LIST_US, len(DELAY_LIST_US), reps, PAD_TIME_US, DELAY_LIST_US[-1],
           PW556_US, PW308_US, AMP556CH1, AMP308CH1, CARRIER_MHZ, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPfwd_delay", **opts)
    print("submitted RearrangeSTIRAPScan_fwd_delay -> descriptor id %s (url=%s, reps=%s, "
          "%d delay points, NumImages=%d)"
          % (did, url or "default", reps, len(DELAY_LIST_US), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 forward-STIRAP delay sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
