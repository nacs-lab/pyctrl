"""RearrangeSTIRAPScan_rev_556freq.py -- 2026-08-01 REVERSE 556 carrier sweep (vdW hypothesis).

Hypothesis under test (per user): the reverse STIRAP under-performs because the atom comes back
down through a Rydberg level that is SHIFTED by van-der-Waals interaction with its excited
neighbours. If so, the reverse two-photon resonance does NOT sit at the forward pair, and locking
the reverse to the forward frequencies -- which is what the config does today -- puts the reverse
leg off resonance by the interaction energy.

So: scan the reverse 556 frequency INDEPENDENTLY of the forward pair. The 308 leg is left alone --
the 616 EOM cannot be retuned between the two STIRAPs on the current hardware (see below), and per
user 2026-08-01 that route is off the table for now, so this scan moves the ONE reverse frequency
that is freely addressable.

!! HOW THE REVERSE FREQUENCIES ARE ADDRESSABLE (this differs from the forward freq-2D):
   * ``FreqEOM616`` (DDS7) is set ONCE per shot from ``Init.EOM616.Freq``
     (STIRAPPushoutStep.py:78; only the IfPump branch re-ramps it). Forward and reverse SHARE it,
     so the 616 EOM cannot carry a separate reverse value. Do not try to sweep it here.
   * What IS per-leg is the AWG CARRIER on Ch2: ``AWG556.Ch2.carrier_freq_MHz`` (the 556 AOM drive
     for the reverse pulse) and ``AWG308.Ch2.carrier_freq_MHz`` (the 308 AOM drive). Both are
     independent of the Ch1 forward values, so the reverse two-photon detuning is scanned through
     them -- the 308 leg via its AOM, not via the 616 EOM.
   Ch2 556 has been MATCHED to Ch1 (143.5) since 2026-07-31 on the argument that it is the same
   transition; the vdW hypothesis is precisely the reason that argument can fail.

!! METRIC FLIPS: reverse ON means the atom is driven back DOWN and kept, so HIGH survival = good
   RETURN. 07-31 best return 0.876 +/- 0.009; the 08-01 re-measure at the new forward pulse read
   only ~0.60 (data_20260801_131012), which is the deficit this scan is chasing.

Caveats to keep in mind when reading the map:
   * Sweeping an AOM carrier changes diffraction efficiency AND beam pointing, not just frequency.
     The forward freq-2D swept +/-0.6 MHz cleanly, so keep the range comparable; a broad, shallow
     optimum may be an AOM-efficiency envelope rather than a resonance.
   * A vdW shift is a PAIR shift: it depends on how many neighbours are excited, so it varies
     shot-to-shot and site-to-site. Expect the reverse line to be BROADENED and asymmetric, not
     just displaced. A clean narrow shift would argue for something else (e.g. a Stark shift).
   * The decisive follow-up, if a shift shows up, is to repeat at a DIFFERENT array spacing: vdW
     scales as 1/R^6, so the shift must shrink strongly with spacing. Nothing else in this scan
     distinguishes vdW from a constant offset.

Forward leg + Ch2 pulse config are held at today's values; only the reverse 556 carrier moves.

What the outcomes mean:
   * a peak DISPLACED from the forward 143.5 -> the reverse resonance really is shifted; read the
     offset as the interaction/AC-Stark energy and re-lock Ch2 there (it is a Ch2-only change, the
     forward is untouched).
   * a peak AT 143.5 -> the reverse is already on resonance and the ~0.60 return deficit
     (data_20260801_131012 vs 0.876 on 07-31) is NOT a detuning problem; go back to the Ch2 pulse
     timing (widths + STIRAPReverseDelay), which is still tuned to the OLD forward pulse.
   * no structure at all -> the reverse is not frequency-sensitive at this Rabi frequency; widen
     the range before concluding.

Run it:
    cd pyctrl
    python YbScans/STIRAPOptimizations/RearrangeSTIRAPScan_rev_556freq.py --reps 20
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- FORWARD leg: today's verified lock (NOT swept) ----
PW556_US = 6.8          # <= 7 us ceiling
PW308_US = 6.2          # <= 7 us ceiling
CARRIER_MHZ = 143.5     # forward 556 (Ch1)
EOM616_MHZ = 234.1      # SHARED by both legs -- cannot differ for the reverse
AMP556CH1 = 1.0
AMP308CH1 = 0.9
DELAY_US = 1.6          # forward overlap lock
PAD_TIME_US = 2.0
MAX_VPP_556 = 15
MAX_VPP_308 = 8

# ---- REVERSE pulse config: the 07-31 round-3 best (NOT swept here) ----
PW556CH2_US = 1.5
PW308CH2_US = 7.0       # at the 7 us hardware ceiling
AMP556CH2 = 1.0
AMP308CH2 = 0.95
REVERSE_DELAY_US = -0.25
STIRAP_GAP_US = 1.0

# ---- SWEPT: the REVERSE 556 AOM carrier (dim 1), +/- 1.0 MHz around the forward lock 143.5 ----
REV556_LIST_MHZ = [142.5, 142.7, 142.9, 143.1, 143.3, 143.5, 143.7, 143.9, 144.1, 144.3, 144.5]
REV308_MHZ = 200.0      # reverse 308 AOM carrier -- FIXED (the 616 EOM route is unavailable)
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
    # SWEPT dim 1 -- the reverse 556, independent of the forward Ch1 carrier
    g().AWG.AWG556.Ch2.carrier_freq_MHz.scan(1, [float(v) for v in REV556_LIST_MHZ])
    g().AWG.AWG556.Ch2.pulse_width_us = PW556CH2_US
    g().AWG.AWG556.Ch2.max_amplitude_vpp = MAX_VPP_556
    g().AWG.AWG556.Ch2.amplitude_scale = AMP556CH2
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_US       # fixed at the verify winner
    g().AWG.AWG308.Ch1.max_amplitude_vpp = MAX_VPP_308
    g().AWG.AWG308.Ch1.amplitude_scale = AMP308CH1
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = REV308_MHZ   # fixed; 616-EOM retune unavailable
    g().AWG.AWG308.Ch2.pulse_width_us = PW308CH2_US    # 7 us ceiling (07-31 railed here)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = MAX_VPP_308
    g().AWG.AWG308.Ch2.amplitude_scale = AMP308CH2

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
    g().Pushout.STIRAPDelay = DELAY_US * 1e-6            # forward overlap lock (not swept)
    g().Pushout.STIRAPReverseDelay = REVERSE_DELAY_US * 1e-6   # 07-31 round-3 best
    g().Pushout.STIRAPPadTime = PAD_TIME_US * 1e-6   # HARD BOUND: every delay must stay below it
    g().Pushout.STIRAPGap = STIRAP_GAP_US * 1e-6
    g().Pushout.IfReverse = 1                   # REVERSE ON -- required for Ch2 to play at all
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


def RearrangeSTIRAPScan(url=None, reps=20):
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    opts["description"] = (
        "2026-08-01 REVERSE 556 carrier sweep -- testing whether the reverse two-photon resonance "
        "is SHIFTED off the forward pair (user's vdW hypothesis: the return leg sees a Rydberg "
        "level shifted by interaction with excited neighbours). AWG556.Ch2.carrier_freq_MHz %s, "
        "%d points x %s reps, independent of the forward Ch1 carrier %.3f. The 616 EOM CANNOT be "
        "retuned between the two STIRAPs (FreqEOM616 is set once per shot and shared by both legs), "
        "so the 308 reverse leg is left at its %.1f MHz AOM carrier and only the 556 moves. "
        "IfReverse = 1. METRIC FLIPS: HIGH survival = good RETURN (07-31 best 0.876 +/- 0.009; the "
        "08-01 re-measure read only ~0.60, data_20260801_131012 -- that deficit is what this "
        "chases). Reverse pulse at the 07-31 best: Ch2 pw556 %.1f / pw308 %.1f us, ReverseDelay "
        "%.2f us, Gap %.1f us. Forward at today's lock: pw556 %.1f / pw308 %.1f / delay %.2f us, "
        "556 amp %.1f / 308 amp %.1f, EOM616 %.3f. Caveat: an AOM-carrier sweep moves diffraction "
        "efficiency and beam pointing as well as frequency, so a broad shallow optimum may be an "
        "efficiency envelope. Scramble 1, pattern double_spacing on 33x33_feedback11, VRydTrap 2, "
        "3-image. Metric: TARGET-ONLY verify-conditioned SURVIVAL; group by Params."
        % (REV556_LIST_MHZ, len(REV556_LIST_MHZ), reps, CARRIER_MHZ, REV308_MHZ,
           PW556CH2_US, PW308CH2_US, REVERSE_DELAY_US, STIRAP_GAP_US,
           PW556_US, PW308_US, DELAY_US, AMP556CH1, AMP308CH1, EOM616_MHZ))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPrev_556freq", **opts)
    print("submitted RearrangeSTIRAPScan_rev_556freq -> descriptor id %s (url=%s, reps=%s, "
          "%d freq points, NumImages=%d, IfReverse=1)"
          % (did, url or "default", reps, len(REV556_LIST_MHZ), 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2026-08-01 reverse 556 carrier sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
