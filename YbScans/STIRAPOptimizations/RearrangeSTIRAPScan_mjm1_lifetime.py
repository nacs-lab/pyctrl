"""RearrangeSTIRAPScan_mjm1_lifetime.py -- mj=-1 Rydberg LIFETIME vs VRydTrap {1,2,3}.

At the mj=-1 forward-STIRAP optimum (556 143.544 / EOM616 234.45, pw556=pw308 4.0us, delay 1.267us,
30 G, pattern quadruple_no_topright), sweep the Rydberg hold time STIRAPGap (survival vs hold = the
Rydberg lifetime) at THREE trap depths VRydTrap = 1, 2, 3.

  * axis 1 = Pushout.STIRAPGap  (hold time, s) -- SAME range as the recent lifetime scans
    (RearrangeSTIRAPScan_mj0.py): 0.1-100us dense (20 pts) + 120-400us sparse (8 pts) = 28 pts.
  * axis 2 = Pushout.VRydTrap in {1, 2, 3}.
  -> 28 x 3 = 84 cells. Forward STIRAP (IfReverse=0); the gap = Rydberg hold before push/image.

Metric = target-only mid(verify)-conditioned survival (Rule 1); group by cfg["Params"] (Rule 2).

Run it:
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mjm1_lifetime.py --reps 3
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# mj=-1 forward-STIRAP optimum (campaign 2026-07-21)
CH1_CARRIER_MHZ = 143.544
EOM616_MHZ = 234.45
# 2026-07-21 use the VRydTrap=2.0-re-optimized pulse (data_20260721_221857/_222516): pw6/delay1.45us,
# 95.67% target-only excitation (was pw4/delay1.267 at the VRyd=0.2 optimum).
PW556_FIXED = 6.0
PW308_FIXED = 6.0
DELAY_FIXED = 1.45e-6
# 2026-07-21 OPTIMIZED REVERSE leg (data_20260721_230955): reverse prefers SHORT pulse pw2 + RD +0.1us
# = 72.6% return (was a blind pw6 copy of forward = 58% ceiling). Reverse Ch2 uses these, NOT the forward pw.
PW_REV_FIXED = 2.0
RDELAY_FIXED = 0.1e-6

# lifetime sweep: STIRAPGap (hold time) 1D, at a SINGLE fixed VRydTrap (one time-scan per trap depth).
GAP_PTS = ([float(v) for v in np.linspace(0.1e-6, 100e-6, 20)]
           + [float(v) for v in np.linspace(120e-6, 400e-6, 8)])   # SECONDS -- same as mj0 recent lifetime
VRYD_ONE = 0.2   # fixed trap depth for THIS run (override via --vryd); run one per {0.2,1,2,3}

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
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_FIXED
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # REVERSE leg (Ch2): round-trip STIRAP de-excites Rydberg -> ground. Carrier MATCHES Ch1 (shares
    # the two-photon resonance w/ EOM616); pulses mirror the forward matched optimum (4.0 us).
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CH1_CARRIER_MHZ
    g().AWG.AWG556.Ch2.pulse_width_us = PW_REV_FIXED   # optimized reverse pw (short)
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
    g().AWG.AWG308.Ch2.pulse_width_us = PW_REV_FIXED   # optimized reverse pw (matched, short)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    g().Init.EOM616.Freq = EOM616_MHZ * 1e6

    g().Pushout.VRydTrap = VRYD_ONE                 # FIXED trap depth (1D gap scan)
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = DELAY_FIXED
    g().Pushout.STIRAPReverseDelay = RDELAY_FIXED   # 2026-07-21 optimized reverse delay (+0.1us)
    g().Pushout.STIRAPGap.scan(1, GAP_PTS)          # axis 1: Rydberg HOLD time (round-trip lifetime)
    g().Pushout.IfReverse = 1   # ROUND-TRIP: excite -> hold in Rydberg -> de-excite to ground. HIGH survival = returned.
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


def RearrangeSTIRAPScan_mjm1_lifetime(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().Pushout.STIRAPGap.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 (pi-pol 308) ROUND-TRIP STIRAP Rydberg LIFETIME, VRydTrap=%.1f (IfReverse=1). At the "
        "optimum pulse (556 %.3f / EOM616 %.3f MHz, pw556=pw308 %.1f us fwd+rev, delay %.3f us, reverse "
        "delay 0.125us, 30 G, quadruple_no_topright): excite -> HOLD in Rydberg (STIRAPGap) -> de-excite "
        "to ground -> image. HIGH survival = returned; survival decays with hold = lifetime. 1D STIRAPGap "
        "sweep (0.1-100us dense + 120-400us sparse, %d pts) at this single trap depth. Fit biexp "
        "(fast+slow+floor). Metric target-only mid-conditioned (group by Params)."
        % (VRYD_ONE, CH1_CARRIER_MHZ, EOM616_MHZ, PW556_FIXED, DELAY_FIXED * 1e6, n1))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_lifetime_v%s" % VRYD_ONE,
                      description=desc, **opts)
    print("submitted mjm1 lifetime VRydTrap=%.1f -> id %s (url=%s, reps=%s, %d gap pts, 556 %.3f/EOM616 %.3f)"
          % (VRYD_ONE, did, url or "default", reps, n1, CH1_CARRIER_MHZ, EOM616_MHZ))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit ONE mj=-1 (pi) Rydberg lifetime time-scan at a fixed VRydTrap.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=6, help="passes over the gap sweep")
    ap.add_argument("--vryd", type=float, default=None, help="fixed VRydTrap for this run (0.2/1/2/3)")
    args = ap.parse_args()
    if args.vryd is not None:
        VRYD_ONE = args.vryd
    RearrangeSTIRAPScan_mjm1_lifetime(url=args.url, reps=args.reps)
