"""RearrangeSTIRAPScan_mj0_trapON_ridge3d.py -- forward-STIRAP ridge-3D, TRAP-ON (VRydTrap=2).

STEP 2 (stirap-optimization runbook) of the trap-ON forward-STIRAP campaign. Freqs LOCKED at the
trap-ON two-photon pair pinned by the freq-2D zoom (data_20260722_121224): carrier 141.767 /
EOM616 284.333 MHz. Now re-optimize the pulse shape at trap-on -- the pw556 6.0 / pw308 5.7 /
delay +1.0us seeds are trap-OFF-tuned and gave 77.5% excitation at the pinned freqs.

The efficient region is a DIAGONAL RIDGE where pw556 ~= pw308 (matched pulse areas), so scan the
MATCHED ridge (pw556 = pw308, axis 1) x STIRAPDelay (axis 2) as a clean 2D grid (runbook step 2b).
Re-pair analysis by ``Params`` (Rule 2); check the in-plane best is INTERIOR (Rule / Lessons).
Tweezers stay ON at VRydTrap=2 (STIRAPPushoutStep trap-off block disabled). Forward only.

Run it:
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan_mj0_trapON_ridge3d.py --reps 4
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# LOCKED mj=-1 two-photon pair, TRAP-ON VRydTrap=0.2 (freq-2D data_20260722_161150; 97.5% exc).
# Trap ON during pulse (STIRAPPushoutStep trap-off block COMMENTED OUT).
CH1_CARRIER_MHZ = 142.944
EOM616_MHZ = 233.967
VRYD_TRAP = 0.2

# REAL Cartesian 3D: pw556 (axis1) x pw308 (axis2, INDEPENDENT) x STIRAPDelay (axis3). The matched
# ridge (pw556=pw308) may miss the true optimum if it sits off the diagonal; scan them independently.
# Coarse 4x4x3 = 48 cells; find the optimum, then fit/zoom.
PW556_US = [round(float(v), 4) for v in np.linspace(4.0, 8.0, 4)]     # pw556, us -- axis 1
PW308_US = [round(float(v), 4) for v in np.linspace(4.0, 8.0, 4)]     # pw308, us -- axis 2 (independent)
DELAY_US = [round(float(v), 4) for v in np.linspace(0.4, 1.6, 3)]     # STIRAPDelay, us -- axis 3
PW556_PTS = PW556_US
PW308_PTS = PW308_US
DELAY_PTS = [round(d * 1e-6, 10) for d in DELAY_US]
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

    # ---- AWG: forward Ch1 (556 rise + 308 fall), pw matched-ridge scanned on axis 1 ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CH1_CARRIER_MHZ     # LOCKED
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, PW556_PTS)      # pw556, axis 1
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = CH1_CARRIER_MHZ     # reverse OFF, unused
    g().AWG.AWG556.Ch2.pulse_width_us = 2
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us.scan(2, PW308_PTS)      # pw308 INDEPENDENT, axis 2
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out: freqs LOCKED, delay scanned on axis 2 ----
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6                   # LOCKED
    g().Pushout.VRydTrap = VRYD_TRAP
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay.scan(3, DELAY_PTS)               # delay, axis 3
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


def RearrangeSTIRAPScan(url=None, reps=4):
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch1.pulse_width_us.size(1)
    n2 = g().AWG.AWG308.Ch1.pulse_width_us.size(2)
    n3 = g().Pushout.STIRAPDelay.size(3)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "mj=-1 forward-STIRAP REAL Cartesian 3D (INDEPENDENT pw556 x pw308 x delay), TRAP-ON "
        "VRydTrap=0.2 (2026-07-22). The matched-ridge 2D (pw556=pw308) capped at ~96%% and may miss "
        "an off-diagonal optimum, so scan pw556/pw308 independently. Freqs LOCKED at the mj=-1 pair "
        "(freq-2D data_20260722_161150): carrier 142.944 / EOM616 233.967 (97.5%% exc). pw556 %s "
        "(axis1) x pw308 %s (axis2) x STIRAPDelay %s us (axis3), %dx%dx%d=%d cells, pattern "
        "quadruple_no_topright on 33x33_feedback11, IfReverse 0. Metric: verify-conditioned survival "
        "(LOW=best); re-pair by Params (runbook Rule 2), check interior."
        % (PW556_US, PW308_US, DELAY_US, n1, n2, n3, n1*n2*n3))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmjm1_trapON_cart3d", **opts)
    print("submitted RearrangeSTIRAPScan_mjm1_trapON_cart3d -> descriptor id %s (url=%s, reps=%s, "
          "verify=%s, NumImages=%d, grid=%dx%dx%d=%d)" % (did, url or "default", reps, VERIFY_IMAGE,
                                                          3 if VERIFY_IMAGE else 2, n1, n2, n3, n1*n2*n3))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the trap-ON mj=0 ridge-3D STIRAP scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
