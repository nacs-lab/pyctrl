"""RearrangeSTIRAPScan_mj0.py -- n=71 3S1 mj=0 variant of RearrangeSTIRAPScan (STIRAP PULSE-OPT 3D).

NOW (2026-07-21, post freq-2D): freqs LOCKED on the mj=0 STIRAP pair (556 143.35 / EOM616 281.75);
this scan does the coarse Cartesian 3D pulse optimization over (pw556, pw308, STIRAPDelay), VRydTrap=0.2.
(Earlier revision was the 2D freq locate; see git history / the mj=0 config block below.)


TEMPORARY n=71 3S1 mj=0 fork of RearrangeSTIRAPScan.py (2026-07-21, 308 SIGMA-pol). IDENTICAL
rearrangement + STIRAP pulse params (widths / delays / amps / patterns / model) to the production
mj=-1 scan; ONLY the 556 + 616 STIRAP FREQUENCIES change + become a 2D locate scan.

n=71 3S1 mj=0 with 308 SIGMA polarization (was pi), measured 2026-07-21 (30 G):
  * 556 stays on the mj=-1 line (single-photon dip 143.5151 MHz; production STIRAP carrier ~143.244)
  * 616 sigma revival / 308 = 281.9389 MHz  (vs mj+1 pi 258.53, mj-1 pi 234.08)

2D locate sweeps AWG556.Ch1.carrier_freq_MHz x Init.EOM616.Freq:
  * 556 Ch1 carrier center = 143.5 MHz ; +/-0.6 @ 0.15 -> 9 pts
  * EOM616 center          = 282.0 MHz (near the fitted 281.94 sigma revival) ; +/-1.5 @ 0.5 -> 7 pts
  63-cell 2D freq locate; deepest-dip (lowest-survival) cell = the mj=0 STIRAP two-photon pair.

Secondary STIRAP freqs (Ch2 fall carrier, Pump556/Pump616 -- pumps OFF, IfPump=0) shifted by the same
mj=-1->mj=0 deltas (556 +0.256 MHz, 616 +47.911 MHz) to keep the pulse structure faithful.

Run it (pyctrl backend live; SLM server reachable -- scan-long slm lock mandatory):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mj0.py --reps 3
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- n=71 mj=0 STIRAP config ------------------------------------------------------------- #
import numpy as _np
# 2026-07-21 DOUBLE_SPACING LIFETIME: freq-2D (data 20260721_144208) confirmed the STIRAP pair
# is UNCHANGED at 143.50 / 282.00 MHz (atomic resonance, spacing-independent). Now the Rydberg
# lifetime: sweep STIRAPGap at the locked pulse + freqs, double_spacing, VRydTrap 0.2.
CH1_CARRIER_MHZ = 143.50          # locked mj=0 556 STIRAP carrier (double_spacing)
EOM616_MHZ = 282.00               # locked mj=0 616-EOM (308) resonance
PW556_FIXED = 5.5                  # locked plateau-center pulse
PW308_FIXED = 5.2
DELAY_FIXED = 1.25e-6
VRYD_TRAP = 1.0                    # 2026-07-21 fast-channel resolve: run VRydTrap 1.0 then 2.0
# 2026-07-21 FAST-CHANNEL RESOLVE: the >=300 us coarse-step scans left tau_fast unresolved.
# Custom NON-uniform gap: DENSE 0.1-100 us (~5 us step, 20 pts, resolves the fast decay) +
# SPARSE 120-400 us (~40 us, 8 pts, anchors the slow tail + floor). 28 pts total. Fits both channels.
GAP_PTS = ([float(v) for v in _np.linspace(0.1e-6, 100e-6, 20)]
           + [float(v) for v in _np.linspace(120e-6, 400e-6, 8)])   # SECONDS (no round)

# mj=-1 -> mj=0 (sigma) frequency deltas for the secondary/pump freqs:
DELTA_556_MHZ = CH1_CARRIER_MHZ - 143.244   # secondary/pump freq shift
DELTA_616_MHZ = EOM616_MHZ - 234.089
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
    from scan_export import matlab_colon

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ      # LOCKED 556 carrier
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FIXED             # FIXED plateau-center pw556
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = round(143.4 + DELTA_556_MHZ, 6)   # 143.656 (mj-1 143.4 shifted)
    g().AWG.AWG556.Ch2.pulse_width_us = 4.531
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_FIXED            # FIXED plateau-center pw308
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

    g().Init.EOM616.Freq = EOM616_MHZ * 1e6      # LOCKED 616-EOM (308) resonance

    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_FIXED        # FIXED at coarse-3D interior optimum 1.25 us
    g().Pushout.STIRAPReverseDelay = 0.132e-6
    g().Pushout.STIRAPGap.scan(1, GAP_PTS)   # LIFETIME sweep: Rydberg hold time (s)
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
    g().rearrange_kwargs.extras.pattern = "double_spacing"   # 2026-07-21 switched every-other -> double_spacing
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


def RearrangeSTIRAPScan_mj0(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan_mj0", **opts)
    ng = g().Pushout.STIRAPGap.size(1)
    print("submitted RearrangeSTIRAPScan_mj0 (DOUBLE_SPACING LIFETIME) -> descriptor id %s (url=%s, reps=%s, "
          "verify=%s, NumImages=%d, pattern double_spacing, freqs 556 %.3f/EOM616 %.3f MHz, "
          "pulse pw556 %.2f/pw308 %.2f/delay %.3f us, VRydTrap 0.2; STIRAPGap sweep %d pts 0.1-100 us)"
          % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2,
             CH1_CARRIER_MHZ, EOM616_MHZ, PW556_FIXED, PW308_FIXED, DELAY_FIXED*1e6, ng))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the n=71 3S1 mj=0 RearrangeSTIRAPScan 2D freq locate.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mj0(url=args.url, reps=args.reps)
