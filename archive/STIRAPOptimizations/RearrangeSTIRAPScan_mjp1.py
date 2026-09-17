"""RearrangeSTIRAPScan_mjp1.py -- mj=+1 variant of RearrangeSTIRAPScan (2D freq LOCATE scan).

TEMPORARY mj=+1 fork of RearrangeSTIRAPScan.py (2026-07-21). IDENTICAL rearrangement + STIRAP
pulse params (widths / delays / amps / patterns / model) to the production mj=-1 scan; ONLY the
556 + 616 STIRAP FREQUENCIES are moved to the mj=+1 Zeeman state and turned into a 2D locate scan.

mj=+1 frequencies measured 2026-07-21 (30 G):
  * mj=+1 556 Rydberg dip   = 72.4228 MHz   (mj=-1 was 143.5151; symmetric about mj=0 @ 107.9501)
  * mj=+1 616 revival / 308 = 258.5323 MHz  (mj=-1 was 234.078; +24.5 MHz mj-dependent 308 shift)

The 2D locate sweeps the mj=-1 "pair" axes -- AWG556.Ch1.carrier_freq_MHz x Init.EOM616.Freq --
centred on the mj=+1 frequencies, carrying over the mj=-1 dip->AWG-carrier offset (mj=-1 carrier
143.244 sat ~0.271 MHz BELOW its 143.5151 dip; EOM616 234.089 ~= its 234.078 revival).
  * 556 Ch1 carrier center = 72.4228 - 0.271 = 72.152 MHz ; +/-0.6 @ 0.15 MHz -> 9 pts
  * EOM616 center          = 258.5323 MHz            ; +/-1.5 @ 0.5 MHz -> 7 pts
  63-cell 2D freq locate; deepest-dip (lowest-survival) cell = the mj=+1 STIRAP two-photon pair.

The secondary STIRAP freqs (Ch2 fall carrier, Pump556/Pump616 -- pumps are OFF here, IfPump=0)
are shifted by the same mj=-1->mj=+1 deltas (556 -71.092 MHz, 616 +24.443 MHz) to keep the pulse
structure faithful.

Run it (pyctrl backend live; SLM server reachable -- scan-long slm lock mandatory):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjp1.py --reps 3
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
# img2 verify frame on/off (see RearrangeSTIRAPScan docstring). Per-scan constant.
VERIFY_IMAGE = True

# LOADING (dense load) / TARGET (science array) SLM patterns. (SAME as mj=-1.)
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- mj=+1 2D freq-locate centers + windows (the ONLY physics difference vs mj=-1) --- #
# 2026-07-21 ZOOM: first locate (data 20260721_105214) pinned EOM616 ~258.53 but the 556 carrier
# edge-railed HIGH (dip deepened monotonically to 0.20 @ 72.602, the top of the window). Extend 556
# UP to bracket the true optimum: 556 72.4-73.6 @ 0.15 (9 pts) x EOM616 258.0-259.0 @ 0.25 (5 pts).
# (was: 556 center 72.152 +/-0.6 @0.15 ; EOM616 258.5323 +/-1.5 @0.5)
CH1_CARRIER_CENTER_MHZ = 73.0     # 72.4-73.6 brackets the edge-pinned high-556 optimum
CH1_CARRIER_HALF_MHZ, CH1_CARRIER_STEP_MHZ = 0.6, 0.15    # +/-0.6 @ 0.15 -> 9 pts
EOM616_CENTER_MHZ = 258.5         # first locate pinned this row
EOM616_HALF_MHZ, EOM616_STEP_MHZ = 0.5, 0.25            # +/-0.5 @ 0.25 -> 5 pts

# mj=-1 -> mj=+1 frequency deltas (for the secondary/pump freqs, kept faithful):
DELTA_556_MHZ = CH1_CARRIER_CENTER_MHZ - 143.244   # -71.092 MHz
DELTA_616_MHZ = EOM616_CENTER_MHZ - 234.089        # +24.443 MHz
# ------------------------------------------------------------------------------------ #


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        # CONFIRMED
        "47x47_feedbackwarm4": ("phase/47x47_feedbackwarm4.pt", [0, 0, 0, 0, 0]),
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, -0.75]),
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        # NAME-IMPLIED (confirm the baked Zernike before trusting)
        "33x33_feedback9": ("phase/33x33_feedback9.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),  # 2026-07-10 fb9 depth-reflattened (post optics move); production successor
        "17x17_20um":      ("phase/17x17_20um.pt",      [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    """One imagePatternsJson entry (per camera frame)."""
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(verify, init_cfg, target_cfg):
    """Per-frame detection declaration."""
    items = [_pattern_item(INIT_PATTERN, init_cfg), _pattern_item(TARGET_PATTERN, target_cfg)]
    if verify:
        items.append(_pattern_item(TARGET_PATTERN, target_cfg))
    return json.dumps(items)


import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RearrangeSTIRAPSeq import RearrangeSTIRAPSeq


def build():
    """Build (do NOT submit) the ScanGroup -- mj=+1 2D freq locate."""
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    # ---- frame layout ----
    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- Siglent AWG config -- SAME pulse params as mj=-1; 556 freqs shifted to mj=+1 ----
    ch1_carriers = [round(v, 6) for v in matlab_colon(
        CH1_CARRIER_CENTER_MHZ - CH1_CARRIER_HALF_MHZ, CH1_CARRIER_STEP_MHZ,
        CH1_CARRIER_CENTER_MHZ + CH1_CARRIER_HALF_MHZ)]

    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(1, ch1_carriers)   # mj=+1 2D axis-1 (~72.15 MHz)
    g().AWG.AWG556.Ch1.pulse_width_us = 4.5
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = round(143.4 + DELTA_556_MHZ, 6)   # 72.308 MHz (mj=-1 143.4 shifted)
    g().AWG.AWG556.Ch2.pulse_width_us = 4.531
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3.9
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 4.094
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (unused) ----
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params -- SAME as mj=-1; EOM616 = mj=+1 2D axis-2 ----
    eom616_freqs = [round(v * 1e6, 3) for v in matlab_colon(
        EOM616_CENTER_MHZ - EOM616_HALF_MHZ, EOM616_STEP_MHZ,
        EOM616_CENTER_MHZ + EOM616_HALF_MHZ)]
    g().Init.EOM616.Freq.scan(2, eom616_freqs)   # mj=+1 2D axis-2 (~258.5 MHz)

    g().Pushout.VRydTrap = 0.2
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = 1.25e-6
    g().Pushout.STIRAPReverseDelay = 0.132e-6
    g().Pushout.STIRAPGap = 100e-6
    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = round((234.444 + DELTA_616_MHZ) * 1e6, 3)   # pump OFF; kept faithful
    g().Pushout.Pump556Freq = round((143.3 + DELTA_556_MHZ) * 1e6, 3)     # pump OFF; kept faithful
    g().Pushout.Pump556Amp = 0.5

    g().Pushout.Time369 = 2e-6
    g().Pushout.Vy = 4

    # ---- warmup_kwargs (SAME as mj=-1) ----
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

    # ---- rearrange_kwargs (SAME as mj=-1) ----
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "every-other"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params (runp) ----
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


def RearrangeSTIRAPScan_mjp1(url=None, reps=3):
    """Build + SUBMIT the mj=+1 2D freq-locate scan. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan_mjp1", **opts)
    n1 = g().AWG.AWG556.Ch1.carrier_freq_MHz.size(1)
    n2 = g().Init.EOM616.Freq.size(2)
    print("submitted RearrangeSTIRAPScan_mjp1 -> descriptor id %s (url=%s, reps=%s, verify=%s, "
          "NumImages=%d, 2D %d x %d = %d cells: 556 Ch1 %.3f+-%.2f MHz x EOM616 %.3f+-%.2f MHz)"
          % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2,
             n1, n2, n1 * n2, CH1_CARRIER_CENTER_MHZ, CH1_CARRIER_HALF_MHZ,
             EOM616_CENTER_MHZ, EOM616_HALF_MHZ))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=+1 RearrangeSTIRAPScan 2D freq locate.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjp1(url=args.url, reps=args.reps)
