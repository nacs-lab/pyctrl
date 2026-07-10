"""HalfPulseSTIRAPScan.py -- half-pulse (rise/fall) STIRAP push-out on a REARRANGED array.

RearrangeSTIRAPScan with the full-gaussian AWG pulses replaced by the HALF-PULSE scheme
(2026-07-09, after the 2D freq map 20260709182920 locked the working point):

  * AWG556 = ``rise_gaussian``  -- envelope grows to its peak at the END of the main window;
    the 556 gate (opened ``STIRAP.delay`` after the 308 gate, width = pulse_width) cuts the
    burst AT the peak -> a true half pulse.
  * AWG308 = ``fall_gaussian``  -- peak AT the trigger (gate rise, t=0), decaying; the gate
    (width = delay + pulse_width/2) truncates the tail.

Counterintuitive (reversed) ordering is the step's existing gate timing: 308 (at max) first,
556 rising up later -- STIRAPPushoutStep needs no change.

Frequencies are FIXED at the 20260709182920 dip: EOM616 = 234.0 MHz, 556 carrier = 143.1 MHz.

Two modes (submit both back-to-back with --mode both):
  * ``amp`` -- 1D sweep of the 556 pulse amplitude 3 -> 15 Vpp. The AWG's max_amplitude_vpp is
    a scan-CONSTANT (AWGManager sets it once from the first point), so the sweep is
    ``amplitude_scale`` = vpp/15 with max_amplitude_vpp = 15 (identical output voltage).
  * ``sd``  -- 2D steepness x delay: steepness swept on BOTH AWGs together (same axis ->
    paired), ``Pushout.STIRAP.delay`` 0.5 -> 2.0 us on axis 2.

Run it:
    cd pyctrl
    python YbScans/HalfPulseSTIRAPScan.py --mode both     # queue amp then sd
    python YbScans/HalfPulseSTIRAPScan.py --mode amp --reps 15
    python YbScans/HalfPulseSTIRAPScan.py --mode sd --reps 10
"""

import argparse
import json
import os
import sys
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# Optimized working point (scan 20260709182920 dip).
EOM616_FREQ_HZ = 234.0e6
CARRIER_556_MHZ = 143.1

# amp mode: 556 amplitude sweep 3 -> 15 Vpp as amplitude_scale @ max 15 Vpp.
AMP_VPP_MAX = 15.0
AMP_VPP_VALUES = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]          # 7 pts
# sd mode: steepness (BOTH AWGs, paired) x STIRAP delay.
# Round 1 (20260709191536, 11 Vpp): st 2-8 x delay 0.5-2.0 -> transfer only at
# st=2, best 0.598 @ (2, 0.5), edge-pinned. Round 2 (20260709194604, corner
# extension st 1-2.5 x 0-0.8): interior best 0.260 @ (1.5, 0.6). Round 3
# (user-corrected ranges): st 2-8 x delay 0.5-2.5 at the SATURATED 556
# amplitude 13 Vpp (amp scan 20260709195736: 0.271 @ 13 Vpp, saturated >= 13).
# Round 4: transfer confined to st<2.5 in rounds 1-3 -> fine map st 1-3.
STEEPNESS_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]  # 9 pts
DELAY_VALUES_US = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]                 # 6 pts
SD_VPP_556 = 13.0

# Converged half-pulse working point (rounds 2+4: best 0.260-0.294 there).
BEST_STEEPNESS = 1.5
BEST_DELAY_S = 0.9e-6
BEST_VPP_556 = 13.0

# amp308 mode: 308 amplitude sweep (scale @ max 5.5 Vpp -- the known-safe max).
AMP308_VPP_MAX = 5.5
AMP308_VPP_VALUES = [1.0, 1.75, 2.5, 3.25, 4.0, 4.75, 5.5]       # 7 pts

# width mode: pulse width swept PAIRED on both AWG windows AND the step's gate
# param (guassian_pulse_width) -- the gated burst REPLAYS the waveform if the
# window is shorter than the gate and truncates it if longer, so they must move
# together.
WIDTH_VALUES_US = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]           # 7 pts
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
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),  # 2026-07-10 fb9 depth-reflattened (post optics move); production successor
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


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    if root not in sys.path:
        sys.path.insert(0, root)


def build(mode):
    """Build (do NOT submit) the ScanGroup for ``mode`` in ('amp', 'sd')."""
    _bootstrap()
    from scan_group import ScanGroup

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    # ---- frame layout ----
    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- Siglent AWG config: HALF PULSES (556 rise, 308 fall), sharp edges ----
    # Width / steepness / amplitudes / delay / gate width are MODE-dependent
    # (set in the branches below); only the shape family + carriers are common.
    g().AWG.AWG556.shape = "rise_gaussian"
    g().AWG.AWG556.carrier_freq_MHz = CARRIER_556_MHZ
    g().AWG.AWG556.smooth_width_us = 0
    g().AWG.AWG308.shape = "fall_gaussian"
    g().AWG.AWG308.carrier_freq_MHz = 200
    g().AWG.AWG308.smooth_width_us = 0

    if mode == "amp":
        # 556 amplitude 3 -> 15 Vpp via amplitude_scale @ fixed max 15 Vpp
        # (max_amplitude_vpp is scan-constant in AWGManager), at the converged
        # working point.
        g().AWG.AWG556.pulse_width_us = 5
        g().AWG.AWG308.pulse_width_us = 5
        g().AWG.AWG556.max_amplitude_vpp = AMP_VPP_MAX
        g().AWG.AWG556.amplitude_scale.scan(
            1, [v / AMP_VPP_MAX for v in AMP_VPP_VALUES])
        g().AWG.AWG308.max_amplitude_vpp = 5.5
        g().AWG.AWG308.amplitude_scale = 1
        g().AWG.AWG556.steepness = BEST_STEEPNESS
        g().AWG.AWG308.steepness = BEST_STEEPNESS
        g().Pushout.STIRAP.delay = BEST_DELAY_S
        g().Pushout.STIRAP.guassian_pulse_width = 5e-6
    elif mode == "amp308":
        # 308 amplitude 1 -> 5.5 Vpp via amplitude_scale @ fixed max 5.5 Vpp,
        # 556 held at its saturated optimum.
        g().AWG.AWG556.pulse_width_us = 5
        g().AWG.AWG308.pulse_width_us = 5
        g().AWG.AWG556.max_amplitude_vpp = BEST_VPP_556
        g().AWG.AWG556.amplitude_scale = 1
        g().AWG.AWG308.max_amplitude_vpp = AMP308_VPP_MAX
        g().AWG.AWG308.amplitude_scale.scan(
            1, [v / AMP308_VPP_MAX for v in AMP308_VPP_VALUES])
        g().AWG.AWG556.steepness = BEST_STEEPNESS
        g().AWG.AWG308.steepness = BEST_STEEPNESS
        g().Pushout.STIRAP.delay = BEST_DELAY_S
        g().Pushout.STIRAP.guassian_pulse_width = 5e-6
    elif mode == "width":
        # Pulse width swept PAIRED: both AWG windows + the step's gate width
        # (guassian_pulse_width) on ONE axis -- the gated burst replays a
        # too-short window and truncates a too-long one, so they move together.
        g().AWG.AWG556.pulse_width_us.scan(1, WIDTH_VALUES_US)
        g().AWG.AWG308.pulse_width_us.scan(1, WIDTH_VALUES_US)
        g().Pushout.STIRAP.guassian_pulse_width.scan(
            1, [v * 1e-6 for v in WIDTH_VALUES_US])
        g().AWG.AWG556.max_amplitude_vpp = BEST_VPP_556
        g().AWG.AWG556.amplitude_scale = 1
        g().AWG.AWG308.max_amplitude_vpp = 5.5
        g().AWG.AWG308.amplitude_scale = 1
        g().AWG.AWG556.steepness = BEST_STEEPNESS
        g().AWG.AWG308.steepness = BEST_STEEPNESS
        g().Pushout.STIRAP.delay = BEST_DELAY_S
    elif mode == "sd":
        # steepness on BOTH AWGs paired on axis 1; delay on axis 2. 556 at the
        # saturated amplitude from the amp scan (SD_VPP_556).
        g().AWG.AWG556.pulse_width_us = 5
        g().AWG.AWG308.pulse_width_us = 5
        g().AWG.AWG556.max_amplitude_vpp = SD_VPP_556
        g().AWG.AWG556.amplitude_scale = 1
        g().AWG.AWG308.max_amplitude_vpp = 5.5
        g().AWG.AWG308.amplitude_scale = 1
        g().AWG.AWG556.steepness.scan(1, STEEPNESS_VALUES)
        g().AWG.AWG308.steepness.scan(1, STEEPNESS_VALUES)
        g().Pushout.STIRAP.delay.scan(2, [v * 1e-6 for v in DELAY_VALUES_US])
        g().Pushout.STIRAP.guassian_pulse_width = 5e-6
    else:
        raise ValueError("mode must be one of amp/amp308/width/sd (got %r)" % mode)

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; unused) ----
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (working point from 20260709182920) ----
    # (guassian_pulse_width + delay are mode-dependent, set in the branches above)
    g().Init.EOM616.Freq = EOM616_FREQ_HZ
    g().Pushout.VRydTrap = 0.03
    g().Pushout.STIRAP.ifReverse = False
    g().Pushout.STIRAP.reverse_delay = 1.5e-6
    # 1 us between the 556 gate close and the 369 auto-ionization pulse. The
    # half pulse CUTS the 556 at its PEAK, so the AOM fall tail overlaps the
    # 369 without this (seen on the Rydberg scope, first half-pulse shots
    # 2026-07-09; full gaussians end at ~zero so waitTime=0 never showed it).
    g().Pushout.STIRAP.waitTime = 1e-6
    g().Pushout.STIRAP.gap = 1e-6
    g().Pushout.Amp369 = 1
    g().Pushout.Time369 = 1e-6
    g().Pushout.BiasCoilCurrent.Ryd = 30

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
    g().rearrange_kwargs.nsteps = 60
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

    # ---- run params ----
    n_points = {"amp": len(AMP_VPP_VALUES),
                "amp308": len(AMP308_VPP_VALUES),
                "width": len(WIDTH_VALUES_US),
                "sd": len(STEEPNESS_VALUES) * len(DELAY_VALUES_US)}[mode]
    rp.NumPerGroup = n_points * (10 if mode == "sd" else 15)    # matches default reps
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


_DESCRIPTIONS = {
    "amp": ("Half-pulse STIRAP (556 rise_gaussian, 308 fall_gaussian, counterintuitive "
            "308-first ordering) 556-amplitude sweep 3-15 Vpp (amplitude_scale 0.2-1.0 @ "
            "max 15 Vpp) AT the round-2 sd-map optimum (20260709194604): steepness 1.5 "
            "both AWGs, delay 0.6 us (survival|verify 0.260 there @ 11 Vpp). Frequencies "
            "fixed at the 20260709182920 full-gaussian dip (EOM616 234.0 MHz, carrier "
            "143.1 MHz). First amp run 20260709191207 (steepness 4, delay 1.3 us) was flat "
            "~0.91 -- no overlap. 2026-07-09 half-pulse campaign; full-gaussian baseline "
            "dip 0.117."),
    "sd":  ("Half-pulse STIRAP (556 rise_gaussian, 308 fall_gaussian) 2D adiabaticity map "
            "ROUND 4: FINE steepness 1-3 (0.25 steps, BOTH AWGs paired) x delay 0.5-2.5 us, "
            "556 @ 13 Vpp -- rounds 1-3 (20260709191536 / 194604 / 210217) confined all "
            "transfer to steepness < 2.5 (round-3 @13Vpp: only st=2 column live, best 0.545 "
            "@ 2/0.5us; round-2 interior best 0.260 @ 1.5/0.6us @11Vpp). Frequencies fixed "
            "at the 20260709182920 dip (EOM616 234.0 MHz, carrier 143.1 MHz); 2026-07-09 "
            "half-pulse STIRAP campaign."),
    "amp308": ("Half-pulse STIRAP 308-amplitude sweep 1-5.5 Vpp (amplitude_scale @ max 5.5 "
               "Vpp, the known-safe 308 max) at the converged working point: steepness 1.5 "
               "both AWGs, delay 0.9 us, 556 @ 13 Vpp (saturated; amp scan 20260709195736), "
               "widths 5 us. Frequencies fixed at the 20260709182920 dip (EOM616 234.0 MHz, "
               "carrier 143.1 MHz). Best so far 0.26-0.29 (sd rounds 20260709194604 / "
               "212734); 2026-07-09 half-pulse STIRAP campaign."),
    "width": ("Half-pulse STIRAP pulse-width sweep 2-10 us, PAIRED on both AWG windows AND "
              "the step gate param guassian_pulse_width (gated burst replays a too-short "
              "window / truncates a too-long one). Working point: steepness 1.5 both, delay "
              "0.9 us, 556 @ 13 Vpp, 308 @ 5.5 Vpp, freqs at the 20260709182920 dip. Best "
              "so far 0.26-0.29; 2026-07-09 half-pulse STIRAP campaign."),
}


def HalfPulseSTIRAPScan(mode, url=None, reps=None):
    """Build + SUBMIT one mode. Returns the descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    if reps is None:
        reps = 10 if mode == "sd" else 15
    g = build(mode)
    label = "HalfPulseSTIRAP_%s" % mode
    did = ybStartScan("RearrangeSTIRAPSeq", g, url=url, label=label, rep=reps,
                      description=_DESCRIPTIONS[mode])
    print("submitted %s -> descriptor id %s (reps=%s)" % (label, did, reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit half-pulse STIRAP scans.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--mode", choices=("amp", "amp308", "width", "sd", "both"),
                    default="both",
                    help="'both' = amp then sd (legacy pairing); amp308/width "
                         "are the converged-working-point follow-ups")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the sweep (default: sd 10, others 15)")
    args = ap.parse_args()
    if args.mode == "both":
        HalfPulseSTIRAPScan("amp", url=args.url, reps=args.reps)
        HalfPulseSTIRAPScan("sd", url=args.url, reps=args.reps)
    else:
        HalfPulseSTIRAPScan(args.mode, url=args.url, reps=args.reps)
