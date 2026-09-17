"""RearrangeSTIRAPScan_mjm1_reverse3d.py -- optimize the REVERSE STIRAP leg (Ch2) at VRydTrap=2.0.

The round-trip return ceiling was low (~0.58) => the reverse STIRAP (Rydberg -> ground de-excitation) is
lossy. FORWARD is fixed at its VRydTrap=2.0 optimum (Ch1: pw556=pw308=6.0, delay 1.45us; freqs
143.544/234.45); gap=0.1us; IfReverse=1. This scan optimizes the REVERSE knobs (Ch2 + STIRAPReverseDelay)
exactly like the forward coarse 3D:
  * axis 1 = AWG556.Ch2.pulse_width_us  (reverse 556, fastest)
  * axis 2 = AWG308.Ch2.pulse_width_us  (reverse 308)
  * axis 3 = Pushout.STIRAPReverseDelay (reverse 556<->308 overlap/order; SIGNED, >0 = 556 first)
  -> 5x5x5 broad+sparse. Reverse carriers set matched (Ch2 556 = Ch1 556 = 143.544, shares the two-photon
     resonance; Ch2 308 = 200 like Ch1).

METRIC: MAXIMIZE return survival = P(final=1|mid=1) over target sites (target-only mid-conditioned) --
HIGH survival = successfully de-excited back to ground. (Opposite sign to forward excitation, but the same
subset_stats/Params machinery; group by cfg["Params"], Rule 2.)

Run it (pyctrl backend live; SLM server reachable):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_reverse3d.py --reps 3
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# FIXED forward optimum -- mj=-1 VRydTrap=0.2 verified (data_20260722_164834, 94.3% exc; trap ON).
CH1_CARRIER_MHZ = 142.944
EOM616_MHZ = 233.967
PW556_FWD = 6.0
PW308_FWD = 5.7
DELAY_FWD = 1.0e-6
VRYD_TRAP = 0.2

# REVERSE FINE 2D (coarse data_20260721_225215 showed a FLAT ~55-70% region, RD near 0 best, no sharp
# matched diagonal). Fine-scan the strongest lever ReverseDelay x reverse pw (matched Ch2), around peak.
# 2026-07-21 fine (data_20260721_230333): reverse prefers SHORT pulse (pw3 best, monotonic down) + RD ~+0.2.
# pw3 was edge-railed -> extend DOWN to un-rail. RD tightened around the peak.
RDELAY_PTS = [float(v) for v in np.linspace(-0.1e-6, 0.5e-6, 4)]  # s, axis 2 -- RD near the +0.2 peak
PW_REV_PTS = [float(v) for v in np.linspace(1.5, 3.5, 5)]         # us, axis 1 -- reverse pw, EXTENDED DOWN
PW556_REV_PTS = PW_REV_PTS
PW308_REV_PTS = PW_REV_PTS

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

    # ---- FORWARD (Ch1) FIXED at the VRydTrap=2.0 optimum ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FWD
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # ---- REVERSE (Ch2) SWEPT; carriers matched to the two-photon resonance ----
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CH1_CARRIER_MHZ           # matched (was 143.4+delta -- WRONG)
    g().AWG.AWG556.Ch2.pulse_width_us.scan(1, PW_REV_PTS)           # axis 1 (reverse pw, matched)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_FWD
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us.scan(1, PW_REV_PTS)           # axis 1 -- pw308_Ch2 MIRRORS pw556_Ch2 (matched)
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
    g().Pushout.STIRAPDelay = DELAY_FWD
    g().Pushout.STIRAPReverseDelay.scan(2, RDELAY_PTS)             # axis 2 (fine RD)
    g().Pushout.STIRAPGap = 0.1e-6                                 # fixed short gap per user
    g().Pushout.IfReverse = 1                                      # round-trip: optimize the return
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


def RearrangeSTIRAPScan_mjm1_reverse3d(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch2.pulse_width_us.size(1)   # reverse pw (matched)
    n2 = g().Pushout.STIRAPReverseDelay.size(2)      # fine RD
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 REVERSE STIRAP FINE 2D (VRydTrap=2.0). Coarse (data_20260721_225215) showed a FLAT ~55-70%% "
        "reverse region, RD near 0 best, no sharp matched diagonal. FORWARD fixed (Ch1 pw6/delay1.45; 556 "
        "%.3f / EOM616 %.3f), gap=0.1us, IfReverse=1, Ch2 carrier matched 143.544. Fine-scan reverse "
        "pw556_Ch2=pw308_Ch2 %s us (axis1) x STIRAPReverseDelay %s us (axis2), %dx%d=%d cells. Metric = "
        "MAXIMIZE target-only mid-conditioned RETURN survival (group by Params). quadruple_no_topright."
        % (CH1_CARRIER_MHZ, EOM616_MHZ, PW_REV_PTS, [round(v*1e6,3) for v in RDELAY_PTS], n1, n2, n1 * n2))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_reverse_fine2d",
                      description=desc, **opts)
    print("submitted reverse FINE 2D -> id %s (url=%s, reps=%s, %dx%d=%d cells, reverse pw %s x RD %s us)"
          % (did, url or "default", reps, n1, n2, n1 * n2, PW_REV_PTS, [round(v*1e6,3) for v in RDELAY_PTS]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 REVERSE STIRAP coarse 3D optimization.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjm1_reverse3d(url=args.url, reps=args.reps)
