"""RearrangeSTIRAPScan_mjm1_fine2d.py -- mj=-1 STIRAP fine 2D plane+perp co-vary scan (Stage 2).

With the two-photon pair LOCKED (556 143.544 / EOM616 234.55) and the ridge plane fitted from the
coarse 3D (data_20260721_193126):
    pw556(delay) = -0.318*delay + 6.069  us
    pw308(delay) = +0.429*delay + 4.435  us
this scan samples the fitted plane FINELY along delay AND steps PERPENDICULAR to the plane (the
direction the coarse fit does not span) to nail the plane's position + thickness -- per the
stirap-optimization runbook. Implemented as a CO-VARYING PATH: pw556, pw308, STIRAPDelay all on scan
dim 1 (Rule 2 -- re-pair the analysis by cfg["Params"], NOT run_analysis's collapsed sweep).

Path = DELAY_PTS (9, 0.6..2.2us -- extends past the coarse 1.9us rail) x PERP (3 offsets) = 27 pts.
PERP moves both pw556,pw308 along the in-(pw556,pw308) plane-normal (0.429, 0.318)/|.| by {-d,0,+d}.

Metric = TARGET-ONLY, mid(verify)-conditioned survival (Rule 1); group by cfg["Params"] (Rule 2).

Run it (pyctrl backend live; SLM server reachable):
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_fine2d.py --reps 4
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# LOCKED mj=-1 two-photon pair (freq-2D locate data_20260721_192444)
CH1_CARRIER_MHZ = 143.544
EOM616_MHZ = 234.55
VRYD_TRAP = 2.0   # 2026-07-21 re-optimize at VRydTrap=2.0 per user

# The broad 2-8us coarse 3D (data_20260721_194442) showed the efficient region is a DIAGONAL RIDGE
# where pw556 ~= pw308 (matched pulse areas), ~90-95% all along it 3.5..8 us -- a broad plateau with
# NO sharp optimum, the short end (3.5/3.5 ~93%) tied within noise with the long end. Per user
# (prefer ~4 us / shorter pulses = less Rydberg decoherence), fine-scan the MATCHED ridge pw556=pw308
# from 3..7 us x delay as a clean 2D grid, and pick the interior best (short-side-preferred on a tie).
PW_US = [round(float(v), 4) for v in np.linspace(3.0, 7.0, 5)]        # pw556 = pw308, us -- axis 1
DELAY_US = [round(float(v), 4) for v in np.linspace(0.6, 1.6, 4)]     # STIRAPDelay, us -- axis 2
PW556_PTS = PW_US                                                     # matched: pw308 mirrors pw556
DELAY_PTS = [round(d * 1e-6, 10) for d in DELAY_US]

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
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, PW556_PTS)      # pw556, axis 1
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
    g().AWG.AWG308.Ch1.pulse_width_us.scan(1, PW556_PTS)      # pw308 MIRRORS pw556 (matched ridge), axis 1
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
    g().Pushout.STIRAPDelay.scan(2, DELAY_PTS)               # delay, axis 2 (clean 2D grid)
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


def RearrangeSTIRAPScan_mjm1_fine2d(url=None, reps=4):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch1.pulse_width_us.size(1)   # pw
    n2 = g().Pushout.STIRAPDelay.size(2)             # delay
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 (pi-pol 308) STIRAP FINE 2D matched-ridge scan (Stage 2). Pair 556 %.3f / EOM616 %.3f, "
        "30 G, VRydTrap %.1f, pattern quadruple_no_topright. Broad 2-8us coarse showed a DIAGONAL ridge "
        "pw556~=pw308 (matched areas), broad ~90-95%% plateau. Fine-scan the MATCHED ridge pw556=pw308 "
        "%s us (axis1) x STIRAPDelay %s us (axis2), %dx%d=%d cells. Pick interior best (short-pref on tie). "
        "Metric target-only mid-conditioned (Rule 1), group by Params (Rule 2)."
        % (CH1_CARRIER_MHZ, EOM616_MHZ, VRYD_TRAP, PW_US, DELAY_US, n1, n2, n1 * n2))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_fine2d",
                      description=desc, **opts)
    print("submitted RearrangeSTIRAPScan_mjm1_fine2d -> id %s (url=%s, reps=%s, %dx%d=%d cells, "
          "pw556=pw308 %s us x delay %s us, 556 %.3f/EOM616 %.3f, VRydTrap %.1f)"
          % (did, url or "default", reps, n1, n2, n1 * n2, PW_US, DELAY_US,
             CH1_CARRIER_MHZ, EOM616_MHZ, VRYD_TRAP))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 (pi) STIRAP fine 2D plane+perp co-vary scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=4, help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_mjm1_fine2d(url=args.url, reps=args.reps)
