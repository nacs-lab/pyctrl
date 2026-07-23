"""RearrangeSTIRAPScan_mjm1.py -- n=71 3S1 mj=-1 (pi-pol 308) STIRAP optimization fork.

The NATIVE two-photon Rydberg line: 556 (1st leg, 3P1 mj=-1) + 308 (2nd leg, pi / Delta mj=0)
-> 6s.71s 3S1 mj=-1. Identical rearrangement + pulse structure to the mj=0/mj+1 forks; ONLY the
556 + 616 STIRAP frequencies change (to the mj=-1 line) and the scan runs the freq-2D locate first
(per the stirap-optimization runbook: locate the resonant pair BEFORE the pulse ridge-3D).

mj=-1 reference (30 G, pi-pol 308 -- the historical native line, per the mj0/mjp1 sibling docstrings):
  * 556 single-photon 3P1 mj=-1 dip  = 143.5151 MHz ; production STIRAP carrier ~143.244 MHz
  * 616 revival / 308 (pi)           = 234.089 MHz   (vs mj+1 pi 258.53, mj0 sigma 281.94)

STAGE (2026-07-21): FREQ-2D LOCATE. Sweep AWG556.Ch1.carrier_freq_MHz x Init.EOM616.Freq centered
on the mj=-1 pair, at a known-transferring pulse (pw556 5.5 / pw308 5.2 / delay 1.25 us, carried from
the mj=0 fork's plateau-center pulse -- exact widths don't matter for a FREQUENCY dip, only that
transfer happens). Deepest-dip (lowest target-conditioned survival) cell = the mj=-1 two-photon pair.
Next stages: lock the pair -> ridge-3D pulse opt (pw556 x pw308 x delay) -> top-N verify (>=100 shots).

Metric = TARGET-ONLY, mid(verify)-conditioned survival (stirap-optimization runbook Rule 1). Analyze
with select_subset_stirap + group by cfg["Params"] (Rule 2), NOT run_analysis's collapsed sweep.

Run it (pyctrl backend live; SLM server reachable -- scan-long slm lock mandatory):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1.py --reps 3
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- mj=-1 (pi-pol 308) 2D freq-locate centers + windows -------------------------------- #
# Centers = the historical mj=-1 production pair. The 556 carrier locate is centered on the
# production STIRAP carrier (143.244), NOT the single-photon dip (143.5151); the two-photon dip
# sits at the carrier that makes 556+308 resonant.
# 2026-07-21 Stage 2.5 RECHECK: re-locate the pair at the OPTIMUM pulse to confirm no freq shift.
CH1_CARRIER_CENTER_MHZ = 143.544            # located mj=-1 carrier (Stage 0 data 20260721_192444)
CH1_CARRIER_HALF_MHZ, CH1_CARRIER_STEP_MHZ = 0.45, 0.15   # +/-0.45 @ 0.15 -> 7 pts (tighter recheck)
EOM616_CENTER_MHZ = 234.55                   # located mj=-1 EOM616/308
EOM616_HALF_MHZ, EOM616_STEP_MHZ = 0.6, 0.2             # +/-0.6 @ 0.2 -> 7 pts (233.95-235.15)

# OPTIMUM pulse from the fine matched-ridge (Stage 2 data 20260721_195612): pw556=pw308 4.0 / delay 1.267us.
PW556_FIXED = 4.0
PW308_FIXED = 4.0
DELAY_FIXED = 1.267e-6
VRYD_TRAP = 0.2   # 2026-07-21 per user

# mj=-1 -> mj=-1 (this line) frequency deltas for the secondary/pump freqs (0 at center):
DELTA_556_MHZ = CH1_CARRIER_CENTER_MHZ - 143.244   # 0.0 (kept for parity with the sibling forks)
DELTA_616_MHZ = EOM616_CENTER_MHZ - 234.089        # 0.0
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

    # 2D freq-locate axes (MATLAB-exact colon sweeps).
    c0, ch, cs = CH1_CARRIER_CENTER_MHZ, CH1_CARRIER_HALF_MHZ, CH1_CARRIER_STEP_MHZ
    e0, eh, es = EOM616_CENTER_MHZ, EOM616_HALF_MHZ, EOM616_STEP_MHZ
    carrier_pts = matlab_colon(c0 - ch, cs, c0 + ch)              # MHz, axis 1
    eom_pts = [v * 1e6 for v in matlab_colon(e0 - eh, es, e0 + eh)]  # Hz, axis 2

    g = ScanGroup()

    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(1, carrier_pts)   # 556 carrier locate axis
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FIXED
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = round(143.4 + DELTA_556_MHZ, 6)   # mj-1 143.4 (reverse OFF, unused)
    g().AWG.AWG556.Ch2.pulse_width_us = 4.531
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_FIXED
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

    g().Init.EOM616.Freq.scan(2, eom_pts)        # 616-EOM (308) locate axis

    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_FIXED
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


def RearrangeSTIRAPScan_mjm1(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch1.carrier_freq_MHz.size(1)
    n2 = g().Init.EOM616.Freq.size(2)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 (pi-pol 308) STIRAP freq-2D LOCATE. Native two-photon Rydberg line 3P1 mj=-1 + 308 pi "
        "-> 6s.71s 3S1 mj=-1. Sweep 556 Ch1 carrier x EOM616 (308) about the historical mj=-1 pair "
        "(556 143.244 / EOM616 234.089), transferring pulse pw556 5.5/pw308 5.2/delay 1.25us, 30 G. "
        "Deepest target-conditioned survival cell = the mj=-1 two-photon pair; feeds the ridge-3D "
        "pulse opt next. Metric: target-only mid-conditioned (select_subset_stirap, group by Params)."
    )
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_freq2D",
                      description=desc, **opts)
    print("submitted RearrangeSTIRAPScan_mjm1 FREQ-2D LOCATE -> descriptor id %s (url=%s, reps=%s, "
          "verify=%s, NumImages=%d, %dx%d=%d cells, 556 %.3f+/-%.2f/EOM616 %.3f+/-%.2f MHz)"
          % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2, n1, n2, n1 * n2,
             CH1_CARRIER_CENTER_MHZ, CH1_CARRIER_HALF_MHZ, EOM616_CENTER_MHZ, EOM616_HALF_MHZ))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the n=71 3S1 mj=-1 (pi) RearrangeSTIRAPScan freq-2D locate.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjm1(url=args.url, reps=args.reps)
