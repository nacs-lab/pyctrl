"""RearrangeMWRabiSpectrumScan.py -- rearrange + forward STIRAP -> QICK MW PI PULSE -> reverse
STIRAP, SWEEPING the microwave carrier FREQUENCY at a FIXED pulse length (Rabi spectrum).

Step 2 of the coherent MW-resonance fit (step 1 = the time-domain Rabi in RearrangeMWScan, which
calibrates Omega / T_pi at the operating gain). Here the pulse length is FIXED at T_pi and the
carrier is swept: return survival vs detuning traces the Rabi lineshape
    P(delta) = Omega^2/(Omega^2+delta^2) * sin^2(sqrt(Omega^2+delta^2)*t/2),
a symmetric sinc-like DIP centred on the MW resonance -> fit the centre for f0. FWHM ~ Omega ~
1/(2*T_pi), so the frequency step should be <= Omega/5 when zooming.

2026-08-19: created for the 60 G high-field port. STIRAP round-trip = the 60 G lock from
RearrangeSTIRAPScan (carrier 118.8856 / EOM616 230.4316, delay 1.333 us, reverse delay -0.2 us).
STARTING POINTS ARE THE 20 G VALUES (user directive): centre f0 = 11334.93 MHz (20 G low-power
revival-ladder value) and T_pi = 250 ns at gain 2000 (job 943). The 60 G resonance may sit well
outside the default window -- if the scan comes back flat, widen with --mw-lo/--mw-hi (or
coarse-locate first with the revival-destruction method, Revival616 path, which needs no f0 prior).

Flow per shot (STIRAPPushoutStep): forward STIRAP (Ch1, ground->Rydberg) -> [fwd->rev gap: QICK MW
pi pulse, fired on TTLQickTrig=FPGA1/TTL14] -> reverse STIRAP (Ch2, Rydberg->ground). STIRAPGap ==
QICK.duration EXACTLY (the RearrangeMWScan convention; any margin only adds MW-free Rydberg hold).

METRIC: target-only mid-conditioned RETURN survival = P(final=1|mid=1), grouped by cfg["Params"].
On-resonance pi pulse transfers S->P; P does not reverse-STIRAP back -> survival DIP at f0.

Run it (pyctrl backend live; SLM server reachable; QICK server up on 192.168.0.72):
    cd pyctrl
    python YbScans/RearrangeMWRabiSpectrumScan.py --reps 8
    python YbScans/RearrangeMWRabiSpectrumScan.py --mw-lo 11250 --mw-hi 11340 --mw-pts 91
    python YbScans/RearrangeMWRabiSpectrumScan.py --t-pi-ns 500 --gain 1000   # narrower line
"""
import argparse
import json
import numpy as np


VERIFY_IMAGE = True
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# ---- MW Rabi-spectrum knobs (CLI-overridable in __main__) --------------------------------
# 20 G STARTING POINTS (2026-08-12 campaign): f0 = 11334.93 MHz (low-power limit of the
# revival-destruction gain ladder), T_pi ~250 ns at gain 2000 (job 943 duration scan).
# 2026-08-19: window recentred on 11270 MHz per user (the located 60 G line; was the blind
# 11250-11300 bracket below the 20 G f0 11334.93). Span tightened to +-10 MHz around it.
MW_CENTER_MHZ = 11275.0       # 60 G line locate (user directive 2026-08-19)
MW_SPAN_MHZ = 20.0            # full window width (11260-11280 MHz)
MW_PTS = 41                   # 0.5 MHz step (FWHM ~2 MHz -> ~4 pts/FWHM)
MW_LO = None                  # explicit lo/hi override the centre+span pair when set
MW_HI = None
MW_GAIN = 2000                # DAC gain; T_PI below is only calibrated AT this gain
# 2026-08-19 user correction: Omega ~4 MHz at gain 2000 -> period 250 ns, so 250 ns is a 2pi
# pulse (returns population, NO dip). T_pi = half period = 125 ns.
T_PI_NS = 125.0               # fixed pi pulse (ns); re-calibrate with a duration scan at f0


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
    # 60 G forward lock (data_20260819_092526): carrier/EOM616 = 118.8856/230.4316 at the
    # along-ridge floor (survival 0.0267 +- 0.0042); ridge line for re-derivation:
    # carrier = 118.982 + 0.814*(EOM616 - 230.55). Same operating point as RearrangeMWScan.
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 118.8856   # 60 G lock
    g().AWG.AWG556.Ch1.pulse_width_us = 3
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87

    # ---- REVERSE (Ch2) quintic de-excitation (60 G: delay confirmed data_20260819_095938; widths = incumbent 2/2, 08-19 width-2D pending) ----
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 118.8856   # matched to forward carrier
    g().AWG.AWG556.Ch2.pulse_width_us = 2   # incumbent (08-19 reverse width-2D pending)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2   # MUST stay matched to 556 Ch2
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave -- RABI SPECTRUM: fixed pi pulse, SWEEP the carrier ---------------
    if MW_LO is not None and MW_HI is not None:
        lo, hi = float(MW_LO), float(MW_HI)
    else:
        lo = MW_CENTER_MHZ - MW_SPAN_MHZ / 2.0
        hi = MW_CENTER_MHZ + MW_SPAN_MHZ / 2.0
    FREQ_PTS = [round(float(v), 6) for v in np.linspace(lo, hi, MW_PTS)]

    t_pi_s = round(T_PI_NS * 1e-9, 12)   # round(,12) NOT (,4): sub-us values zero out at 4 dp (job #90)

    g().QICK.template = "Sine"
    g().QICK.freq.scan(1, FREQ_PTS)                   # SWEPT dim 1: MW carrier (MHz)
    g().QICK.gain = MW_GAIN                           # T_PI is calibrated AT this gain only
    g().QICK.duration = t_pi_s                        # FIXED pi pulse (20 G seed: 250 ns @ gain 2000)
    g().QICK.rabi_freq = 4.825e6                      # unused by Sine template (Echo t_pi derivation only)
    g().QICK.wait_time = 1e-6                         # unused by Sine template
    g().QICK.phase = 0.0
    g.runp().QICK = True                              # opt in -> engine_run wires setup/arm/cleanup

    g().Init.EOM616.Freq = 230.4316e6   # 60 G lock (pairs with carrier 118.8856)

    g().Pushout.VRydTrap = 2.0
    g().Pushout.BiasCoilCurrent.Ryd = 60   # 60 G high field
    g().Pushout.STIRAPDelay = 1.333e-6   # 60 G mid-plateau (data_20260818_174505)
    g().Pushout.STIRAPReverseDelay = -0.2e-6   # 60 G confirmed (data_20260819_095938)
    # 2026-08-19 FIX: gap == duration (125 ns) BROKE the round trip -- with reverse delay -0.2 us
    # the reverse pulse started before the gap ended (baseline mid->final survival collapsed to
    # ~0.23, flat vs MW freq, job 1226). Fixed 1 us gap: the pi pulse sits inside it, matches the
    # >= 1 us gaps of every working round-trip scan today (return ~0.93); 1 us extra Rydberg hold
    # is negligible vs the ~100 us lifetime.
    g().Pushout.STIRAPGap = 1e-6
    g().Pushout.IfReverse = 1                        # round-trip: excite -> MW -> de-excite
    g().Pushout.IfMW = 1                             # fire TTLQickTrig in the fwd->rev gap
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6   # mj=0 pump616 (pumps OFF, IfPump=0)
    g().Pushout.Pump556Freq = 143.556e6   # mj=0 pump556 (pumps OFF, IfPump=0)
    g().Pushout.Pump556Amp = 0.5

    # ---- trap timing during the pulse / gap ------------------------------------------------
    # NOTE: the light shift inside the gap moves the MW line -- fit f0 under the SAME
    # SLMAOMAmpGap/VRydTrap the consumer scans (RearrangeMWScan/RearrangeEchoScan) use.
    g().Pushout.STIRAPPadTime = 2e-6   # moves the AmpSLM=0 trap-off point INSIDE the forward pulse
    g().Pushout.SLMAOMAmpGap = 0.55    # trap depth held through the fwd->rev gap
    g().Pushout.IfGatePulses = 1       # 1 = fire AWG gate pulses; 0 = the no-pulse A/B control

    # ---- ionization (TTL path) --------------------------------------------------------------
    g().Pushout.IonizationViaDAC = 0    # 0 = TTL switch, 1 = legacy DAC electrode ramp
    g().Pushout.TimeIonization = 0.1e-6
    g().Pushout.TIonizationAlign = 0.5e-6
    g().Init.VIonizationSet5to8 = 4     # DC level held by Dev1/2, asserted in InitStep; must be < 5 V

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
    g().rearrange_kwargs.protocol = "rearrange2_eviction"   # matches RearrangeSTIRAPScan's dimer_20um combo
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "quadruple_spacing"   # 2026-08-19 user directive (was quadruple_spacing)
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    rp.NumPerGroup = 2000
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1   # EOM616 fixed -> scramble fine (decorrelates drift from the swept MW axis)
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeMWRabiSpectrumScan(url=None, reps=10):
    from yb_start_scan import ybStartScan

    g = build()
    # Read back from the built ScanGroup (single source of truth) so the description tracks
    # whatever build() actually set -- no hardcoded params to drift out of sync.
    fx = g.get_fixed(1)
    freqs, _ = g.get_scanaxis(1, 1, 1)                       # axis 1 = QICK.freq (MHz)
    n1 = len(freqs)
    nan = float("nan")

    fwd = _dig(fx, "AWG", "AWG556", "Ch1", default={})
    rev = _dig(fx, "AWG", "AWG556", "Ch2", default={})
    pp = _dig(fx, "Pushout", default={})
    q = _dig(fx, "QICK", default={})
    eom616_mhz = _dig(fx, "Init", "EOM616", "Freq", default=nan) / 1e6

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "60 G high-field MW RABI SPECTRUM: rearrange + STIRAP round-trip, FIXED QICK pi pulse "
        "(%.4g ns, gain %s) in the fwd->rev gap; SCAN the MW carrier %.4f..%.4f MHz (%d pts). "
        "Return survival vs freq = sinc-like Rabi DIP at f0 (fit the centre). 20 G seeds: f0 "
        "11334.93, T_pi 250 ns @ gain 2000 -- if flat, widen the window or revival-locate first. "
        "Forward: Ch1 556 %s %.4fMHz / EOM616 %.4fMHz, delay %+.4gus, VRydTrap %.4g. Reverse: Ch2 "
        "%.4fMHz amp %.4g, RD %+.4gus. STIRAPGap == duration. Metric = target-only mid-conditioned "
        "RETURN survival (group by Params). dimer_20um on 33x33_feedback11."
        % (q.get("duration", nan) * 1e9, q.get("gain", "?"), freqs[0], freqs[-1], n1,
           fwd.get("shape", "?"), fwd.get("carrier_freq_MHz", nan), eom616_mhz,
           pp.get("STIRAPDelay", nan) * 1e6, pp.get("VRydTrap", nan),
           rev.get("carrier_freq_MHz", nan), rev.get("amplitude_scale", nan),
           pp.get("STIRAPReverseDelay", nan) * 1e6))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeMWRabiSpectrum_60G",
                      description=desc, **opts)
    print("submitted RearrangeMWRabiSpectrum_60G -> id %s (url=%s, reps=%s; %d freq pts "
          "%.4f..%.4f MHz, T_pi %.4g ns @ gain %s)"
          % (did, url or "default", reps, n1, freqs[0], freqs[-1],
             q.get("duration", nan) * 1e9, q.get("gain", "?")))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 60 G MW Rabi-spectrum scan (fixed pi pulse, sweep QICK.freq).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10, help="passes over the sweep (0 = forever)")
    ap.add_argument("--mw-center", type=float, default=MW_CENTER_MHZ,
                    help="window centre MHz (default %.2f = the 20 G f0 seed)" % MW_CENTER_MHZ)
    ap.add_argument("--mw-span", type=float, default=MW_SPAN_MHZ,
                    help="full window width MHz (default %.1f)" % MW_SPAN_MHZ)
    ap.add_argument("--mw-lo", type=float, default=None, help="explicit window lo MHz (with --mw-hi, overrides centre/span)")
    ap.add_argument("--mw-hi", type=float, default=None, help="explicit window hi MHz")
    ap.add_argument("--mw-pts", type=int, default=MW_PTS, help="number of frequency points (default %d)" % MW_PTS)
    ap.add_argument("--gain", type=int, default=MW_GAIN,
                    help="QICK DAC gain (default %d; T_pi is only calibrated at this gain)" % MW_GAIN)
    ap.add_argument("--t-pi-ns", type=float, default=T_PI_NS,
                    help="fixed pulse length ns (default %.0f = the 20 G gain-2000 T_pi)" % T_PI_NS)
    args = ap.parse_args()
    MW_CENTER_MHZ = args.mw_center; MW_SPAN_MHZ = args.mw_span
    MW_LO = args.mw_lo; MW_HI = args.mw_hi; MW_PTS = args.mw_pts
    MW_GAIN = args.gain; T_PI_NS = args.t_pi_ns
    if (MW_LO is None) != (MW_HI is None):
        ap.error("--mw-lo and --mw-hi must be given together")
    print("MW window %s | %d pts | T_pi %.4g ns @ gain %d"
          % ("%.4f-%.4f MHz" % (MW_LO, MW_HI) if MW_LO is not None
             else "%.4f +- %.4f MHz" % (MW_CENTER_MHZ, MW_SPAN_MHZ / 2), MW_PTS, T_PI_NS, MW_GAIN))
    RearrangeMWRabiSpectrumScan(url=args.url, reps=args.reps)
