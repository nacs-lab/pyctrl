"""RearrangeEchoScan.py -- rearrange + forward STIRAP -> QICK microwave ECHO -> reverse STIRAP.
2D scan of the closing-pi/2 PHASE (0..2pi) x the echo free-evolution TIME T (0..10 us).

Copied from RearrangeMWScan and switched to the Echo template with a 2D phase x T sweep. The STIRAP
round-trip is FIXED at the mj=-1 verified forward optimum + reference quintic-reverse seeds; the MW
carrier is FIXED on the located sigma- resonance (~11318.7 MHz). Echo = pi/2 - T/2 - pi - T/2 - pi/2:
sweeping the closing-pi/2 phase at each T gives a phase-scan echo (fitted phase -> signed detuning,
no fringe/f0 degeneracy); sweeping T gives the Hahn-echo coherence decay (T2echo).

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

    # ---- FORWARD (Ch1) FIXED at the mj=-1 VRydTrap=0.2 optimum ----
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 142.944   # mj=-1 forward optimum, verified (data_20260722_164834, 94.3% exc; trap ON)
    g().AWG.AWG556.Ch1.pulse_width_us = 6.0   # mj=-1 verified optimum
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 1

    # ---- REVERSE (Ch2) quintic de-excitation (reference quintic seeds; pw/RD re-tune pending) ----
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 142.944   # matched to forward carrier (quintic-reverse ref convention)
    g().AWG.AWG556.Ch2.pulse_width_us = 2   # reverse width us (556 Ch2 = 308 Ch2; seed, unverified)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 1.0   # quintic-reverse ref seed (was FLAT-era 0.44)
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 5.7   # mj=-1 verified optimum
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2   # reverse width us (matches 556 Ch2)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95

    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave -- ECHO 2D scan: axis1 = final-pi/2 PHASE (0..2pi), axis2 = free-evolution T ----
    # Echo template = pi/2 - T/2 - pi - T/2 - pi/2 (pulse lengths from rabi_freq; T = wait_time).
    # 2D: sweep the closing-pi/2 PHASE (a phase-scan Ramsey/echo -> the fitted phase gives signed
    # detuning directly, no fringe/f0 degeneracy) AND the free-evolution time T (-> echo decay / T2).
    # Freq FIXED on the located resonance. Most precise + up-to-date: 2026-07-24 overnight eightfold
    # Ramsey T=1us global fit f0 = 11318.7562 MHz (R2 0.999), same config as this scan (sigma- line,
    # 71 3S1 mj=-1 -> 71 3P2 mj=-2). rabi_freq from the on-res MW Rabi Omega/2pi = 4.825 MHz (T_pi 104 ns).
    g().QICK.template = "Echo"
    g().QICK.freq = 11318.7562                       # MHz, FIXED on resonance (2026-07-24 Ramsey global)

    # DIPOLAR EXCHANGE: read the SS<->PP populations directly, so NO phase scan -- fix closing-pi/2
    # phase = 0 and sweep ONLY the free-evolution time T (1D). (The phase scan was for single-atom
    # coherence readout; the dipolar oscillation shows directly in the 2-atom populations vs T.)
    # axis 1: echo free-evolution T = wait_time (s). Predicted dipolar freq ~483 kHz -> period
    # 2.070 us; 10 pts/period -> step 0.207 us, over 0.05..10 us (~4.8 periods, 49 pts).
    # STIRAPGap auto-sizes for max T=10 us. round(,12) not (,4): sub-us values round to 0.0 at 4 dp
    # and zero the pulse (job #90 HW-min guard).
    _F_DIP = 483e3                                    # predicted dipolar-exchange freq (Hz)
    _STEP = (1.0 / _F_DIP) / 10.0                     # 10 samples per oscillation period
    TIME_PTS = [round(float(v), 12) for v in np.arange(0.05e-6, 15e-6 + _STEP / 2, _STEP)]  # s (0.05..15us)

    g().QICK.phase = 0.0                              # FIXED closing-pi/2 phase (deg) -- no phase scan
    g().QICK.wait_time.scan(1, TIME_PTS)             # axis 1 (sole): echo free-evolution T (s)
    g().QICK.gain = 2000                             # DAC gain (nonzero to emit; 0 = silent)
    g().QICK.rabi_freq = 4.825e6                     # derives t_pi2 = 1/(4*f), t_pi = 2*t_pi2 (from MW Rabi)
    g.runp().QICK = True                             # opt in -> engine_run wires setup/arm/cleanup

    g().Init.EOM616.Freq = 233.967e6   # mj=-1 two-photon (308) resonance

    g().Pushout.VRydTrap = 0.2   # in-pulse trap depth (trap ON)
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay = 1.0e-6   # mj=-1 verified optimum (308-first)
    g().Pushout.STIRAPReverseDelay = -0.5e-6   # quintic reverse delay s (data_20260722_172621)
    # STIRAPGap must span the whole QICK program so the MW plays entirely inside the gap. It CO-SCANS
    # with wait_time (same axis 1): per T point, gap = qick_program_duration(that T's echo) + margin.
    # (Co-scanning instead of one fixed max-T gap keeps the Rydberg hold as SHORT as possible at each
    # T -> less lifetime loss during the gap.) qick_program_duration is freq-independent (durations
    # come from rabi_freq + wait_time), so the actual QICK.freq doesn't matter here.
    _tmpl = g().QICK.template()
    _GAP_MARGIN = 0e-6 #0.5e-6                               # margin on top of each point's program length
    def _prog_dur(Tval):
        return qick_program_duration({"template": _tmpl, "freq": 11318.7562, "gain": 2000,
                                      "rabi_freq": 4.825e6, "phase": 0.0, "wait_time": Tval})
    try:
        from devices.qick_awg import qick_program_duration
        GAP_PTS = [round(_prog_dur(T) + _GAP_MARGIN, 12) for T in TIME_PTS]
    except Exception:  # noqa: BLE001 - off the engine venv: T + pulses (Ramsey 2*t_pi2, Echo 4*t_pi2) + margin
        _tpi2 = 1.0 / (4 * 4.825e6)
        _npulse = 4 if str(_tmpl).lower() == "echo" else 2   # Echo: 2*pi/2 + pi = 4*t_pi2; Ramsey: 2*pi/2
        GAP_PTS = [round(T + _npulse * _tpi2 + _GAP_MARGIN, 12) for T in TIME_PTS]
    g().Pushout.STIRAPGap.scan(1, GAP_PTS)            # co-scans axis 1 with QICK.wait_time
    g().Pushout.IfReverse = 1                        # round-trip: excite -> MW -> de-excite
    g().Pushout.IfMW = 1                             # fire TTLQickTrig in the fwd->rev gap
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = round((234.444 + (233.967 - 234.089)) * 1e6, 3)   # + fwd EOM616 offset from model
    g().Pushout.Pump556Freq = round((143.3 + (142.944 - 143.244)) * 1e6, 3)     # + fwd 556 carrier offset from model
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
    g().rearrange_kwargs.protocol = "rearrange2_eviction"
    g().rearrange_kwargs.extras.eviction_shift_px = 11.5
    g().rearrange_kwargs.extras.eviction_nsteps = 17
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "eight_pairs"
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


def RearrangeEchoScan(url=None, reps=3):
    from yb_start_scan import ybStartScan

    g = build()
    # Sole swept axis = QICK.wait_time (T); phase is fixed at 0.
    tt, _ = g.get_scanaxis(1, 1, 1)                          # axis 1 = QICK.wait_time (s)
    n1 = len(tt)
    nan = float("nan")

    fx = g.get_fixed(1)
    fwd = _dig(fx, "AWG", "AWG556", "Ch1", default={})
    pp = _dig(fx, "Pushout", default={})
    q = _dig(fx, "QICK", default={})
    eom616_mhz = _dig(fx, "Init", "EOM616", "Freq", default=nan) / 1e6

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    desc = (
        "mj=-1 rearrange + STIRAP round-trip; QICK ECHO (pi/2-T/2-pi-T/2-pi/2) in the fwd->rev gap. "
        "1D scan of free-evolution T %.3f..%.3f us (%d pts, ~10/period for a %.0f kHz dipolar osc); "
        "closing-pi/2 phase FIXED 0. QICK freq FIXED %.4f MHz (sigma- line), rabi_freq %.4g MHz, gain %s. "
        "STIRAPGap %.4gus (sized for max T). Forward: Ch1 556 %s %.3fMHz / EOM616 %.3fMHz, VRydTrap %.4g "
        "(trap-ON), IfReverse=%s IfMW=%s. Metric = per-PAIR joint 2-atom state (SS/SP/PS/PP) -> "
        "dipolar-exchange SS<->PP oscillation + SP/PS leakage (echo refocuses on-site V_SS/V_PP). "
        "eight_pairs (8 pairs, 20um intra-pair) on 33x33_feedback11."
        % (tt[0] * 1e6, tt[-1] * 1e6, n1, 483.0,
           q.get("freq", nan), q.get("rabi_freq", nan) * 1e-6, q.get("gain", "?"),
           pp.get("STIRAPGap", nan) * 1e6,
           fwd.get("shape", "?"), fwd.get("carrier_freq_MHz", nan), eom616_mhz,
           pp.get("VRydTrap", nan), pp.get("IfReverse", "?"), pp.get("IfMW", "?")))
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeDipolar_T",
                      description=desc, **opts)
    print("submitted RearrangeDipolar 1D T scan -> id %s (url=%s, reps=%s; T %.3f..%.3f us x%d pts, "
          "phase=0 fixed, freq %.4f MHz)"
          % (did, url or "default", reps, tt[0] * 1e6, tt[-1] * 1e6, n1, q.get("freq", nan)))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the mj=-1 rearrange+STIRAP QICK spin-echo 2D (phase x T) scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep")
    args = ap.parse_args()
    RearrangeEchoScan(url=args.url, reps=args.reps)
