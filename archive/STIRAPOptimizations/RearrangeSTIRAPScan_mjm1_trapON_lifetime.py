"""RearrangeSTIRAPScan_mj0_trapON_lifetime.py -- Rydberg lifetime, TRAP-ON VRydTrap=0.2.

Sweeps ``Pushout.STIRAPGap`` (the forward-STIRAP -> readout hold; the atom sits in the Rydberg
state during the gap) at the LOCKED VRydTrap=0.2 trap-ON forward-STIRAP optimum. Survival vs gap
time -> Rydberg lifetime. Forward only (IfReverse 0), field-ionize at readout. Tweezers stay ON at
VRydTrap=0.2 during the pulse (STIRAPPushoutStep trap-off block COMMENTED OUT, 2026-07-22).

Locked config (2026-07-22 VRydTrap=0.2 campaign): freqs 556 143.289 / EOM616 281.967 MHz
(freq-2D data_20260722_141003); pulse pw556=pw308 7.0us / delay +0.8us (0.2 ridge interior
plateau cell, data_20260722_142447; the pw8/0.4 argmin sat on the grid edge so use the solid
interior cell).

Gap sweep: NON-uniform (same scheme as RearrangeSTIRAPScan_mj0.py) -- dense 0.1-100us (resolves the
fast decay) + sparse 120-400us tail (anchors the slow channel + floor), 28 pts.

Run it:
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mj0_trapON_lifetime.py --reps 6
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# LOCKED mj=-1 VRydTrap=0.2 forward-STIRAP verified optimum (data_20260722_164834, 94.3% exc).
CH1_CARRIER_MHZ = 142.944
EOM616_MHZ = 233.967
VRYD_TRAP = 0.2
PW556_US = 6.0
PW308_US = 5.7   # freq-2D/verify seed (NOT matched)
DELAY_US = 1.0

# Rydberg-hold (STIRAPGap) sweep in SECONDS: dense fast channel + sparse slow tail, 28 pts.
GAP_PTS = ([float(v) for v in np.linspace(0.1e-6, 100e-6, 20)]
           + [float(v) for v in np.linspace(120e-6, 400e-6, 8)])
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

    # ---- AWG: forward Ch1, freqs + pulse LOCKED at the 0.2 optimum ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_US
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # REVERSE leg (Ch2): carriers MATCHED to Ch1 (same two-photon resonance -- see reverse3d note),
    # pulse widths symmetric with the forward leg. Used because IfReverse=1 for the lifetime.
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CH1_CARRIER_MHZ   # matched two-photon resonance
    g().AWG.AWG556.Ch2.pulse_width_us = PW556_US            # reverse 556 (symmetric w/ forward)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_US
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = PW308_US            # reverse 308 (symmetric w/ forward)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out: LOCKED freqs/pulse/delay; SWEEP STIRAPGap (lifetime) ----
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6
    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_US * 1e-6
    g().Pushout.STIRAPReverseDelay = 0.132e-6
    g().Pushout.STIRAPGap.scan(1, GAP_PTS)      # LIFETIME sweep: Rydberg hold time (s)
    g().Pushout.IfReverse = 1   # reverse STIRAP de-excites Rydberg -> ground after the gap; decay
                                # during the gap = fails to return = loss -> THE lifetime signal
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6
    g().Pushout.Pump556Freq = 143.556e6
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


def RearrangeSTIRAPScan(url=None, reps=6):
    from yb_start_scan import ybStartScan

    g = build()
    ng = g().Pushout.STIRAPGap.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "mj=-1 Rydberg LIFETIME (STIRAPGap sweep), TRAP-ON VRydTrap=0.2 (2026-07-22). "
        "STIRAPPushoutStep trap-off block COMMENTED OUT -> tweezers ON at VRydTrap=0.2 during the "
        "pulse. Locked at the mj=-1 verified optimum (data_20260722_164834, 94.3%% exc): 556 142.944 "
        "/ EOM616 233.967 MHz, pw556 6.0/pw308 5.7 / delay +1.0us, IfReverse 1 (reverse STIRAP "
        "de-excites Rydberg->ground after the gap; Ch2 carriers matched, reverse pw 6.0/5.7, "
        "ReverseDelay 0.132us), pattern quadruple_no_topright on 33x33_feedback11. Sweep "
        "Pushout.STIRAPGap %d pts (dense 0.1-100us + sparse 120-400us tail); survival vs gap -> "
        "Rydberg lifetime (decay during gap = fails to return = loss). Metric: verify-conditioned "
        "survival." % ng)
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_trapON_lifetime", **opts)
    print("submitted RearrangeSTIRAPScan_mjm1_trapON_lifetime -> descriptor id %s (url=%s, reps=%s, "
          "verify=%s, NumImages=%d, gap pts=%d)" % (did, url or "default", reps, VERIFY_IMAGE,
                                                    3 if VERIFY_IMAGE else 2, ng))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the trap-ON VRydTrap=0.2 Rydberg-lifetime scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
