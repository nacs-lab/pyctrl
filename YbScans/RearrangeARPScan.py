"""RearrangeARPScan.py -- two-photon adiabatic rapid passage on a REARRANGED array.

Copy of RearrangeRabiScan (same seq, same steps, same rearrangement prologue and ionization
readout); the ONLY change is the science pulse. Instead of a flat square burst at fixed frequency,
the forward 556+308 burst uses ``chirped_flat``: constant amplitude, but the carrier sweeps across
the two-photon resonance during the pulse. Population is dragged adiabatically from |g> to the
Rydberg state, so survival should fall to a PLATEAU and stay there as the sweep is made longer or
wider -- unlike the Rabi scan, where it oscillates.

WHICH LEG IS CHIRPED -- the 556. The 556 leg sets the INTERMEDIATE (3P1) detuning and the 308 leg
sets the two-photon detuning at fixed intermediate detuning, so either one can carry the sweep (but
NOT both together: chirping them in lockstep sweeps nothing, which is exactly the cancellation that
made axis 2 of scan 20260831144355 a Rabi-amplitude knob instead of a detuning). Here:

  * AWG556.Ch1 is ``chirped_flat`` and carries the chirp.
  * AWG308.Ch1 is plain ``flat`` at a fixed 200 MHz. NOTE a ``flat`` shape IGNORES chirp_freq_MHz
    entirely, so do not leave a chirp scan on it expecting an effect.

The price of sweeping the 556 rather than the 308 is that the INTERMEDIATE detuning sweeps too, so
the two-photon Rabi frequency is not constant during the pulse: Omega_eff peaks at ~1.95 MHz on the
3P1 line and falls off with a half-width of ~2 MHz optical (crossover fit of 20260831144355), so a
wide sweep spends most of its time where the drive is weak. The passage still works while it stays
adiabatic, but the result depends on span partly through that amplitude variation rather than
through the passage alone. Chirp the 308 instead if you want the intermediate detuning held still.

Axes: ONE axis. Axis 1 pairs the 556 chirp SPAN (signed final-minus-initial, MHz) with the 556
START frequency, so the MIDPOINT of every sweep sits on the same frequency -- each point is a
symmetric passage about that frequency rather than a sweep that wanders off as the span grows.
A span of 0, if the list includes one, is byte-identical to a plain flat burst and so is a free
fixed-frequency control point. Burst duration is FIXED at 3 us here (the parent's duration axis is
commented out, not deleted).

UNITS: everything on the 556 is AOM drive frequency, and beam A is DOUBLE-PASSED, so the optical
excursion at the atoms is TWICE these numbers -- ``detuning_intermediate_MHz = 1`` is 2 MHz of real
intermediate detuning, and a span of 20 sweeps 40 MHz optically.

Pulse timing is unchanged from the parent: the 308 gate fires first and the 556 gate
``stirap_delay`` later, and the 308 ``pad_time_us`` hold covers that offset at full amplitude, so
the two-photon overlap is exactly the 556 width. The 556 itself has ``pad_time_us = 0``, so its
sweep starts with the pulse.

    cd pyctrl
    python YbScans/RearrangeARPScan.py --reps 10
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
VERIFY_IMAGE = True

INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
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
    """Build (do NOT submit) the ScanGroup."""
    from scan_group import ScanGroup

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- FORWARD (Ch1) = chirped_flat two-photon burst: constant amplitude, swept carrier ----
    # 308 is lengthened by stirap_delay so the two-photon OVERLAP is exactly the swept width.
    pw556_us = [round(float(v), 4) for v in np.linspace(1, 10, 10)]  # >= ~0.2 (AWG arb density)
    stirap_delay = 1.8e-6          # for aligning the 308 and 556 pulses, not much physical meaning
    pw308_us = [round(float(v), 4) for v in pw556_us]

    # Axis 1 = the 556 chirp SPAN (signed, final - initial, MHz) across the main window.
    # A span of 0, if present, is byte-identical to a plain flat burst -> a free control point.
    chirp_span_MHz = [-2, -1, 0, 1, 2] #[round(float(v), 4) for v in np.linspace(0.0, 4.0, 9)]

    # Centre every sweep on the same frequency: start half a span low, so the MIDPOINT of the chirp
    # lands on f556_centre whatever the span. Derived elementwise from chirp_span_MHz, so the two
    # axis-1 lists stay the same length and stay paired if the span list is re-spaced, re-lengthened
    # or made non-uniform (all of which silently broke the earlier endpoint-linspace version).
    # AOM units: beam A is double-passed, so the optical excursion is 2x these numbers.
    F556_RES_MHz = 119.24 #119.2                 # two-photon resonance for THIS geometry (AOM drive)
    detuning_intermediate_MHz = -5         # offset from the 3P1 line (AOM -> 2 MHz optical)
    f556_centre = F556_RES_MHz + detuning_intermediate_MHz
    freq_556_MHz = [round(f556_centre - s / 2, 4) for s in chirp_span_MHz]

    # 556: the SWEPT leg. carrier_freq_MHz and chirp_freq_MHz are paired on axis 1 so each point is
    # a symmetric passage about f556_centre. "linear" holds the sweep rate constant across the
    # window; "quintic" would instead ramp the rate on and off (zero at both ends), which is the
    # gentler adiabatic choice if the edges of the sweep turn out to matter.
    g().AWG.AWG556.Ch1.shape = "chirped_flat"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 118.5 #.scan(1, np.linspace(117.5, 119.5, 10)) #freq_556_MHz) 
    g().AWG.AWG556.Ch1.chirp_profile = "linear"
    g().AWG.AWG556.Ch1.chirp_freq_MHz = -1 #.scan(1, chirp_span_MHz)
    g().AWG.AWG556.Ch1.pulse_width_us = 5 #.scan(2, pw556_us)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87
    g().AWG.AWG556.Ch1.pad_time_us = 0.0

    # 308: NOT swept -- a plain flat burst at a fixed 200 MHz, present only to complete the two
    # photons. Its pad_time_us hold covers the stirap_delay gate offset at full amplitude. A "flat"
    # shape ignores chirp_freq_MHz, so adding a chirp scan here would silently do nothing.
    g().AWG.AWG308.Ch1.shape = "flat"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 5 #.scan(2, pw308_us)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale.scan(1, [0, 1]) # = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2.0

    # ---- REVERSE (Ch2) UNUSED (IfReverse = 0): armed but never gated --------------------
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 118.2700
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1

    g.runp().AWGs = ["AWG556", "AWG308"]

    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- push-out params. BiasCoilCurrent.Ryd = 60 -> RearrangeSTIRAPSeq picks the HIGH-field
    #      branch, STIRAPHighFieldPushoutStep (< 31 G would pick STIRAPPushoutStep). ----------
    g().Init.EOM616.Freq = 228.778e6 #.scan(2, np.linspace(227e6, 231e6, 10))   #229.88e6 + detuning_intermediate_MHz  # parked; the 308 AWG carries the sweep instead

    g().Pushout.VRydTrap = 2.0
    g().Pushout.BiasCoilCurrent.Ryd = 60

    g().Pushout.STIRAPDelay = stirap_delay
    g().Pushout.STIRAPReverseDelay = 0.0
    g().Pushout.STIRAPPadTime = 2e-6   # trap off at the 1.5 us AWG trigger->output latency
    g().Pushout.STIRAPGap = 1e-6       # unused (IfReverse = IfPump = 0)

    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.IfRecoveryIonization = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6
    g().Pushout.Pump556Freq = 143.556e6
    g().Pushout.Pump556Amp = 0.5
    g().Pushout.SLMAOMAmpGap = 0.55
    g().Pushout.IfGatePulses = 1

    # Ionization right after the burst -- identical to the STIRAP scan.
    g().Pushout.IonizationViaDAC = 0
    g().Pushout.TimeIonization = 0.1e-6
    g().Init.VIonizationSet5to8 = 4
    g().Pushout.TIonizationAlign = 0.5e-6

    # ---- warmup_kwargs -------------------------------------------------------------------
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

    g().rearrange_kwargs.extras.wgs_warm = True
    g().rearrange_kwargs.extras.wgs_pad = 2048
    g().rearrange_kwargs.extras.wgs_iters = 3

    # ---- rearrange_kwargs -----------------------------------------------------------------
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "quad_10x10"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN
    g().rearrange_kwargs.extras.scienceStep = "stirap"

    # ---- run params -----------------------------------------------------------------------
    rp.NumPerGroup = 30
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 0   # EOM616 parked here, but keep 0: Scramble ON has unlocked 616 before
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeARPScan(url=None, reps=10):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    n1 = g().AWG.AWG556.Ch1.pulse_width_us.size(1)
    n2 = g().AWG.AWG308.Ch1.chirp_freq_MHz.size(2)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "Two-photon adiabatic rapid passage on a rearranged array: chirped_flat forward-only "
        "556+308 burst (constant amplitude, swept carrier). Duration on axis 1, 308 chirp span on "
        "axis 2 with span = 0 as the flat-Rabi control; 556 parked on resonance so only the "
        "two-photon detuning sweeps. Quintic sweep profile. Ionization right after the burst -> "
        "img3 = ground-state readout."
    )
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeARPScan",
                      description=desc, **opts)
    print("submitted RearrangeARPScan -> descriptor id %s (url=%s, reps=%s, %dx%d "
          "durations x chirp spans, verify=%s, NumImages=%d)"
          % (did, url or "default", reps, n1, n2, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeARPScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeARPScan(url=args.url, reps=args.reps)
