"""RearrangeSTIRAPScan_mjm1_trapON_freq2d.py -- mj=-1 forward-STIRAP freq-2D, TRAP-ON VRydTrap=0.2.

STEP 1 (stirap-optimization runbook) of the mj=-1 trap-ON forward-STIRAP campaign. Tweezers LEFT
ON during the STIRAP pulse (STIRAPPushoutStep.py trap-off block COMMENTED OUT), so VRydTrap=0.2 is
the in-pulse trap depth. Relocate the mj=-1 two-photon resonance (556 carrier x EOM616/308).

mj=-1 anchor: 556 single-photon 3P1 mj=-1 ~143.515, production STIRAP carrier ~143.5 (30 G line);
308 (mj-1 pi) revival ~234.089 model. TODAY (post 308-polarization fix) the mj-1 308 line is
~234.27 MHz -- confirmed by the AT doublet (data_20260722_155653, centered 143.515, 308 on 234.27).
So center EOM616 on 234.27. Wide window (post-fix + trap-on light shift both uncertain).

Sweep ONLY carrier + EOM616. Pulse seeds pw 6.0/5.7 delay +1.0us (mj=0 quad seeds; re-optimized in
the ridge step), forward only, pattern quadruple_no_topright, VRydTrap=0.2.

Run it:
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_trapON_freq2d.py --reps 4
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# Freq-2D window -- mj=-1 trap-ON VRydTrap=0.2. Center carrier ~143.5 (30 G line, AT-confirmed) x
# EOM616 ~234.27 (post-pol-fix mj-1 308 line). Wide window (pol-fix + trap-on light shift uncertain).
CARRIER_MHZ = [round(float(v), 4) for v in np.linspace(142.5, 144.5, 10)]     # axis 1
EOM616_MHZ  = [round(float(v), 4) for v in np.linspace(233.3, 235.3, 10)]     # axis 2 (MHz)

# Non-swept pulse seeds (mj=0 quad seeds; re-optimized in the ridge step).
PW556_US = 6.0
PW308_US = 5.7
DELAY_US = 1.0
VRYD_TRAP = 0.2   # in-pulse trap depth (trap-off block COMMENTED -> trap ON during pulse)
# ------------------------------------------------------------------------------------ #


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

    # ---- Siglent AWG config -- forward Ch1 (556 rise + 308 fall); Ch2 unused (reverse OFF) --
    # COPIED from the user's canonical RearrangeSTIRAPScan.py (2026-07-21 mj=0 quad optimum).
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 142.944             # FIXED at the 97.5% freq-2D cell
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_US               # 6.0 (freq-2D seed pulses that gave 97.5%)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 143.567   # reverse OFF, unused
    g().AWG.AWG556.Ch2.pulse_width_us = 2
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_US               # 5.7 (mj=0 quad ridge-3D optimum)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (unused) ----
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (COPIED from canonical; freqs swept) ----
    g().Init.EOM616.Freq = 233.967e6   # FIXED at the 97.5% freq-2D cell
    g().Pushout.VRydTrap = VRYD_TRAP   # in-pulse trap depth (trap-off block DISABLED in the step)
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_US * 1e-6
    g().Pushout.STIRAPReverseDelay = 0.132e-6
    g().Pushout.STIRAPGap = 0.1e-6
    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6
    g().Pushout.Pump556Freq = 143.556e6
    g().Pushout.Pump556Amp = 0.5

    g().Pushout.Time369 = 2e-6
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

    # ---- rearrange_kwargs (COPIED from canonical) ----
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "quadruple_no_topright"   # canonical mj=0 33x33 pattern
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
    rp.Scramble = 1
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
        "mj=-1 forward-STIRAP FIXED-POINT VERIFY, TRAP-ON VRydTrap=0.2 (2026-07-22). Single-point "
        "config at the freq-2D 97.5%% optimum cell: carrier 142.944 / EOM616 233.967, pw556 6.0/pw308 "
        "5.7, delay +1.0us, IfReverse 0, pattern quadruple_no_topright on 33x33_feedback11. 100-shot "
        "verify to confirm the 97.5%% (which was an n=4 freq-2D cell -- checking it against drift). "
        "Metric: verify-conditioned survival (LOW=best excitation).")
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_trapON_verify", **opts)
    print("submitted RearrangeSTIRAPScan_mjm1_trapON_freq2d -> descriptor id %s (url=%s, reps=%s, "
          "verify=%s, NumImages=%d)" % (did, url or "default", reps, VERIFY_IMAGE,
                                        3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the trap-ON mj=0 freq-2D STIRAP scan (top-left).")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
