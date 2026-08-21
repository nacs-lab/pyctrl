"""RearrangeMWScan.py -- rearrange + forward STIRAP -> QICK microwave -> reverse STIRAP, scanning the
microwave carrier FREQUENCY.

Copied from RearrangeSTIRAPScan_mjm1_trapON_reverseflat.py (the latest mj=-1 trap-ON STIRAP scan) and
extended with a QICK microwave definition. The STIRAP round-trip is FIXED at the mj=-1 verified forward
optimum + reference quintic-reverse seeds; the NEW scan axis is the QICK carrier frequency.
2026-08-19: ported to the HIGH-FIELD (60 G) operating point from RearrangeSTIRAPScan (forward lock
118.8856 / EOM616 230.4316, delay 1.333 us, reverse delay -0.2 us). CAUTION: QICK.freq is still the
20 G MW resonance -- re-locate it at 60 G before any fixed-freq run.

Flow per shot (STIRAPPushoutStep): forward STIRAP (Ch1, ground->Rydberg) -> [fwd->rev gap: QICK
microwave, fired on TTLQickTrig=FPGA1/TTL14] -> reverse STIRAP (Ch2, Rydberg->ground). The microwave is
declared out-of-band via g().QICK.* + g().runp().QICK (the run loop batch-uploads one program per swept
freq and arms it per shot); the step fires the trigger only because g().Pushout.IfMW=1. STIRAPGap must
span the microwave program so it plays entirely inside the gap before the reverse pulse.

QICK template is configurable (g().QICK.template): "Sine" = a single tone of g().QICK.duration; "Echo"
= pi/2-T/2-pi-T/2-pi/2 echo spectroscopy (pulse lengths from rabi_freq, T = wait_time). Sweeping the
carrier maps the microwave resonance. The description + printout READ the QICK/STIRAP params back from
the built ScanGroup, so they always reflect whatever build() actually sets (no hardcoded params).

METRIC: target-only mid-conditioned RETURN survival = P(final=1|mid=1), grouped by cfg["Params"]. The
microwave transfers/dephases the Rydberg population, so survival vs freq traces the lineshape.

Run it (pyctrl backend live; SLM server reachable; QICK server up on 192.168.0.72):
    cd pyctrl
    python YbScans/RearrangeMWScan.py --reps 8
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"


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


def _dig(tree, *keys, default=None):
    """Safely walk the g.get_fixed() nested dict; return default if any key is absent."""
    for k in keys:
        if not isinstance(tree, dict) or k not in tree:
            return default
        tree = tree[k]
    return tree


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

    # ---- FORWARD (Ch1) at the 2026-08-19 HIGH-FIELD (60 G) lock ----------------------------
    # 2026-08-19: RETUNED 20 G -> 60 G high field (ported from RearrangeSTIRAPScan). 20 G / 30 G
    # values preserved in the trailing comments below. 60 G forward lock (data_20260819_092526):
    # carrier/EOM616 = 118.8856/230.4316 at the along-ridge floor (survival 0.0267 +- 0.0042);
    # ridge line for re-derivation: carrier = 118.982 + 0.814*(EOM616 - 230.55).
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 118.8856   # 60 G lock; was 131.78 (20 G), 142.944 (30 G mj=-1)
    g().AWG.AWG556.Ch1.pulse_width_us = 3   # unchanged 20 G -> 60 G; was 6.0 (30 G mj=-1)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87   # 60 G; was 0.5 (20 G), 1 (30 G)

    # ---- REVERSE (Ch2) quintic de-excitation (60 G: delay confirmed data_20260819_095938; widths = incumbent 2/2, 08-19 width-2D pending) ----
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 118.8856   # 60 G, matched to forward carrier; was 131.78 (20 G), 142.944 (30 G)
    g().AWG.AWG556.Ch2.pulse_width_us = 2   # incumbent (08-19 reverse width-2D pending); job 926: flat along MATCHED-width diagonal
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9   # 60 G (RearrangeSTIRAPScan value); was 0.51 (20 G)
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3   # unchanged 20 G -> 60 G; was 5.7 (30 G mj=-1)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 1   # unchanged 20 G -> 60 G; was 0.9 (30 G)
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2   # MUST stay matched to 556 Ch2 (job 926: mismatch 4.0/1.0 collapses survival to 0.377)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1   # 20 G; was 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave -- the NEW scan axis (carrier frequency); template set below ----
    # Declared out-of-band; the run loop batch-uploads one program per swept freq + arms per shot.
    g().QICK.template = "Sine"   # "Sine" = single tone of g().QICK.duration; "Echo" = pi/2-T/2-pi-T/2-pi/2 echo spectroscopy
    # axis 1: QICK carrier-frequency sweep (MHz).
    
    Freq_PTS = [round(float(v), 12) for v in np.linspace(11320, 11350, 30)]   # 2026-07-23 ZOOM on dip 3 (71 3S1 mj=-1 -> 71 3P2 mj=-2, sigma-): coarse scan
   
    # 2026-08-12: 20 G MW resonance, f0 = 11334.93 +- 0.02 MHz, from the LOW-POWER limit of the
    # 616-revival-destruction gain ladder (gains 8000/4000/2000/1000/500 -> FWHM 15.7/3.8/1.07/0.34/0.21
    # MHz, FWHM ~ gain^1.59). f0 SHIFTS WITH POWER: the gain-8000 centre reads 11334.33, i.e. 0.6 MHz
    # LOW -- do not take a centre from a saturated scan. Was 11318.7562 at 30 G.
    # !! 2026-08-19 60 G port: this is still the 20 G value -- the MW resonance moves with field.
    # RE-LOCATE at 60 G (freq sweep, Freq_PTS above) before trusting any fixed-freq scan here.
    g().QICK.freq = 11275.3252  #11333.48 20 G value, STALE at 60 G (see note above)
    g().QICK.gain = 10000                             # DAC gain (nonzero to emit; 0 = silent).
    # NOTE the "20MHz Rabi, T_pi 25ns" claim previously on this line is NOT supported: a 2026-08-12
    # duration scan at gain 2000 (job 943) gives T_pi ~250 ns; rabi_freq below (4.825e6) is closer.

    # seconds -- DO NOT round(,4): sub-us second-scale values (1e-7..5e-6) all round to 0.0,
    # which zeros QICK.duration and trips the HW-min-pulse guard (see job #90). round in us if needed.
    # DECAY scan: gain8000 Rabi ~20MHz (period 50ns). Sample 3 periods (150ns, 24pts ~8/period) in
    # short windows starting at 0.05, 1, 2, 5, 10 us -> resolve fast Rabi locally, track amplitude
    # decay across the 0-10us baseline. 5 windows x 24 = 120 pts.
    
    #_WSTARTS = [0.05e-6, 1.0e-6, 2.0e-6, 5.0e-6, 10.0e-6]
    #_WSPAN = 0.15e-6   # 3 periods at 20MHz
    #_WPTS = 24
    #MW_TIME_PTS = [round(float(v), 12) for s in _WSTARTS for v in np.linspace(s, s + _WSPAN, _WPTS)]
    
    #MW_TIME_PTS = [round(float(v), 12) for v in np.linspace(0.05e-6, 1.0e-6, 40)]
    MW_TIME_PTS = np.concatenate([np.linspace(0.01e-6, 0.15e-6, 24), np.linspace(1.01e-6, 1.1e-6, 16), np.linspace(2.01e-6, 2.1e-6, 16)])  # 25 pts
    
    g().QICK.duration.scan(1, MW_TIME_PTS)                          # sine template: single-tone length (s)
    g().QICK.rabi_freq = 4.825e6                      # derives t_pi2 = 1/(4*f), t_pi = 2*t_pi2
    g.runp().QICK = True                              # opt in -> engine_run wires setup/arm/cleanup

    g().Init.EOM616.Freq = 230.4316e6   # 60 G lock (pairs with carrier 118.8856); was 236.5e6 (20 G), 233.967e6 (30 G mj=-1)

    g().Pushout.VRydTrap = 2.0   # unchanged 20 G -> 60 G; was 1 at 30 G
    g().Pushout.BiasCoilCurrent.Ryd = 60   # 60 G high field; was 20, 30
    g().Pushout.STIRAPDelay = 1.333e-6   # 60 G mid-plateau (data_20260818_174505: peak 1.222, plateau to 2.0); was 1.556e-6 (20 G)
    g().Pushout.STIRAPReverseDelay = -0.2e-6   # 60 G confirmed (data_20260819_095938: peak of -0.2..+0.3 plateau); was -0.0556e-6 (20 G)
    g().Pushout.STIRAPGap.scan(1, MW_TIME_PTS)   #= 1.5e-6    #_echo_gap_s()            # sized to hold the whole spin echo
    g().Pushout.IfReverse = 1                        # round-trip: excite -> MW -> de-excite
    g().Pushout.IfMW = 1                             # fire TTLQickTrig in the fwd->rev gap
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6   # mj=0 pump616 (RearrangeSTIRAPScan; pumps OFF, IfPump=0)
    g().Pushout.Pump556Freq = 143.556e6   # mj=0 pump556 (RearrangeSTIRAPScan; pumps OFF, IfPump=0)
    g().Pushout.Pump556Amp = 0.5

    # ---- trap timing during the pulse / gap (ported from RearrangeSTIRAPScan 2026-08-12) ----
    # NOTE: both knobs change the trap the atoms see INSIDE the fwd->rev gap, which is exactly where
    # the microwave acts -- the light shift there moves, so re-verify the MW resonance after this.
    g().Pushout.STIRAPPadTime = 2e-6   # moves the AmpSLM=0 trap-off point INSIDE the forward pulse
    g().Pushout.SLMAOMAmpGap = 0.55    # trap depth held through the fwd->rev gap (was full depth)
    g().Pushout.IfGatePulses = 1       # 1 = fire AWG gate pulses; 0 = the no-pulse A/B control

    # ---- ionization (TTL path; renamed from Pushout.Time369 in 721b20c -- the old name is DEAD) ----
    g().Pushout.IonizationViaDAC = 0    # 0 = TTL switch, 1 = legacy DAC electrode ramp
    g().Pushout.TimeIonization = 0.1e-6
    g().Pushout.TIonizationAlign = 0.5e-6
    g().Init.VIonizationSet5to8 = 4     # DC level held by Dev1/2, asserted in InitStep; must be < 5 V

    #g().Pushout.Vy = 4

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

    # Warm-started phase-locked WGS transit frames (server-side producer) instead of pure SLMnet.
    g().rearrange_kwargs.extras.wgs_warm = True
    g().rearrange_kwargs.extras.wgs_pad = 2048
    g().rearrange_kwargs.extras.wgs_iters = 3   # >= 3 (2 is the contract-quality cliff)

    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2_eviction"
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


def RearrangeMWScan(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    # Everything below is READ BACK from the built ScanGroup (single source of truth) so the
    # description tracks whatever build() actually set -- no hardcoded params to drift out of sync.
    fx = g.get_fixed(1)                                       # fixed-param tree
    freqs, _ = g.get_scanaxis(1, 1, 1)                       # actual swept carrier list (MHz); field 1 = QICK.freq (sole axis)
    n1 = len(freqs)
    nan = float("nan")

    fwd = _dig(fx, "AWG", "AWG556", "Ch1", default={})
    rev = _dig(fx, "AWG", "AWG556", "Ch2", default={})
    pp = _dig(fx, "Pushout", default={})
    q = _dig(fx, "QICK", default={})
    eom616_mhz = _dig(fx, "Init", "EOM616", "Freq", default=nan) / 1e6
    pw308 = _dig(fx, "AWG", "AWG308", "Ch1", "pulse_width_us", default=nan)

    # QICK knobs vary by template (Sine: duration; Echo: rabi_freq/wait_time) -- report whatever is set.
    qbits = ["template=%s" % q.get("template", "?"), "gain=%s" % q.get("gain", "?")]
    for key, lbl, scale, unit in (("duration", "dur", 1e6, "us"), ("rabi_freq", "rabi", 1e-6, "MHz"),
                                  ("wait_time", "T", 1e6, "us"), ("phase", "phase", 1.0, "deg")):
        if q.get(key) is not None:
            qbits.append("%s=%.4g%s" % (lbl, q[key] * scale, unit))

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "60 G high-field rearrange + STIRAP round-trip with a QICK microwave pulse in the fwd->rev gap; SCAN the "
        "microwave carrier FREQUENCY. Forward: Ch1 556 %s %.3fMHz / EOM616 %.3fMHz, pw556 %.4g/pw308 "
        "%.4gus, delay %+.4gus, VRydTrap %.4g (trap-ON). Reverse: Ch2 556 %s %.3fMHz amp %.4g, pw %.4gus, "
        "RD %+.4gus. QICK[%s]. STIRAPGap %.4gus, IfReverse=%s, IfMW=%s. Scan QICK.freq %.4f..%.4f MHz "
        "(%d pts). Metric = target-only mid-conditioned RETURN survival (group by Params) -> lineshape. "
        "octuple_spacing on 33x33_feedback11."
        % (fwd.get("shape", "?"), fwd.get("carrier_freq_MHz", nan), eom616_mhz,
           fwd.get("pulse_width_us", nan), pw308,
           pp.get("STIRAPDelay", nan) * 1e6, pp.get("VRydTrap", nan),
           rev.get("shape", "?"), rev.get("carrier_freq_MHz", nan), rev.get("amplitude_scale", nan),
           rev.get("pulse_width_us", nan), pp.get("STIRAPReverseDelay", nan) * 1e6,
           ", ".join(qbits), pp.get("STIRAPGap", nan) * 1e6,
           pp.get("IfReverse", "?"), pp.get("IfMW", "?"),
           freqs[0], freqs[-1], n1))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeMW_freq",
                      description=desc, **opts)
    print("submitted RearrangeMW freq scan -> id %s (url=%s, reps=%s, %d freq pts %.4f..%.4f MHz)"
          % (did, url or "default", reps, n1, freqs[0], freqs[-1]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 rearrange+STIRAP QICK spin-echo freq scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10, help="passes over the sweep")
    args = ap.parse_args()
    RearrangeMWScan(url=args.url, reps=args.reps)
