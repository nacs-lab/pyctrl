"""RearrangeRabiScan.py -- two-photon Rabi oscillation on a REARRANGED array.

Stems from RearrangeSTIRAPScan (same seq, same steps -- nothing new): SLM-rearrangement prologue
(load -> img1 -> rearrange -> img2 verify) then the STIRAP push-out science block, but the science
pulse is a FORWARD-ONLY, FLAT (square) two-photon 556+308 burst whose DURATION is swept. Reverse
STIRAP is off; electrode ionization fires right after the burst exactly as in STIRAP, so img3 is a
GROUND-STATE readout: population left in |g> survives, Rydberg population is ionized away. Survival
vs pulse duration = the Rabi oscillation.

Pulse timing: the 308 gate fires first and the 556 gate ``stirap_delay`` later (the step requires
delay > 0), so 308 is made ``stirap_delay`` longer than 556 -- the two-photon OVERLAP window is
exactly the swept 556 width. As in STIRAP the trap comes back ~1.5 us (AWG trigger->output latency)
before the burst tail ends; unchanged from the parent scan.

    cd pyctrl
    python YbScans/RearrangeRabiScan.py --reps 10
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

    # ---- FORWARD (Ch1) = flat two-photon burst, duration swept on axis 1 -----------------
    # 308 is lengthened by stirap_delay so the two-photon OVERLAP is exactly the swept width.
    pw556_us = [round(float(v), 4) for v in np.linspace(0.1, 2, 30)]  # >= ~0.2 (AWG arb density)
    stirap_delay = 1.8e-6          # for aligning the 308 and 556 pulses, not much physical meaning
    pw308_us = [round(float(v), 4) for v in pw556_us]
    detuning_MHz = np.array([0])  # two-photon detuning sweep
    f556_MHz = (119.2 + detuning_MHz).tolist() #(119.0273 + detuning_MHz).tolist()
    f616 = (229.88 + detuning_MHz) * 1e6 #(230.5625 + detuning_MHz) * 1e6
    f616 = f616.tolist()  

    g().AWG.AWG556.Ch1.shape = "flat"
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(2, f556_MHz)  # two-photon resonance for THIS geometry
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, pw556_us)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87
    g().AWG.AWG556.Ch1.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "flat"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us.scan(1, pw308_us)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 1
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
    g().Init.EOM616.Freq.scan(2, f616)  # parked; never ramped during the burst

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
    g().rearrange_kwargs.extras.pattern = "quadruple_spacing"
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
    rp.Scramble = 1   # EOM616 not swept
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeGRRabiScan(url=None, reps=10):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    #pw556_us, _ = g.get_scanaxis(1, 1, "AWG.AWG556.Ch1.pulse_width_us")
    n1 = g().AWG.AWG556.Ch1.pulse_width_us.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = ()
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeGRRabiScan",
                      description=desc, **opts)
    print("submitted RearrangeGRRabiScan -> descriptor id %s (url=%s, reps=%s, %d widths, "
          "verify=%s, NumImages=%d)"
          % (did, url or "default", reps, n1, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeGRRabiScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeGRRabiScan(url=args.url, reps=args.reps)
