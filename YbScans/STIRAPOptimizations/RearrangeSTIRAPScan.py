"""RearrangeSTIRAPScan.py -- THE parameterized forward-STIRAP scan.

One file, driven entirely from the command line. Every stage of the STIRAP runbook
(experiment-running/references/stirap-optimization.md) is a set of flags on THIS script -- you
should never need to copy it to run a different sweep.

Why it is parameterized (2026-09-16): STIRAPOptimizations/ had accumulated 72 near-identical
copies, one per sweep, because every axis was hardcoded in build() and the only CLI flags were
--url/--reps. Each run already captures its own source via the pyctrl code-snapshot store
(log/code_snapshots/_runs/<scan_id>/, alive and firing on every scan), so a per-sweep file
preserves nothing that provenance does not already hold. The old copies are in
pyctrl/archive/STIRAPOptimizations/.

--------------------------------------------------------------------------------------------
AXIS SPECS -- every physics flag takes one of:
    3.0                 a scalar          -> PINNED
    3.0,3.5,4.0         a comma list      -> SWEPT over exactly those values
    3.0:7.0:5           start:stop:npts   -> SWEPT over linspace(3.0, 7.0, 5)

HOW AXES BECOME SCAN DIMS
    0 swept axes  -> a single fixed point (the fixed-point verify).
    1 swept axis  -> 1-D sweep on dim 1.
    2 swept axes  -> a 2-D GRID (first on dim 1, second on dim 2), n1*n2 combos.
    --covary      -> ALL swept axes go on dim 1 as a PAIRED PATH (equal lengths required),
                     n combos. This is the runbook's ridge-tube / top-N-verify shape.
    More than 2 swept axes without --covary is refused (the engine grids 2 dims).

WHAT IS ENFORCED FOR YOU (each of these has cost a real run)
    * Scramble is set AUTOMATICALLY: 0 whenever --eom616 is swept, else 1. A scrambled EOM
      sweep kicks the 616 out of lock (gotcha-scramble-unlocks-616-eom-sweep) -- the scan dies
      around shot 5-7 or reads flat because 308 never shelves.
    * --eom616 is given in MHz and converted to Hz internally. Init.EOM616.Freq is a Hz field;
      a bare "393.533" once drove the EOM to ~394 Hz and silently wasted eight runs (09-15).
    * Delay must be STRICTLY POSITIVE. STIRAPPushoutStep fires the 308 gate, waits
      Forward_Delay, then fires 556, so delay <= 0 is not representable and raises at
      sequence-BUILD time, per shot, killing the whole job (09-15, job 2073).
    * Pulse widths must be <= 7 us (hardware ceiling, per user 2026-07-31).
    * amplitude_scale must be in [0, 1].
    * NumPerGroup = reps * n_combos, computed here, so the planned and submitted shot counts
      cannot disagree.

EXAMPLES -- the runbook's stages, verbatim
    # frequency 2-D (carrier x EOM616). Scramble auto-0.
    python RearrangeSTIRAPScan.py --carrier 96.10:97.30:13 --eom616 391.85:393.05:9 \
        --reps 4 --note "freq-2D round 1"

    # pulse width x delay 2-D -- ALWAYS co-scan a width with the delay; the delay is the
    # width's partner (it sets the 556<->308 overlap), not an independent knob.
    python RearrangeSTIRAPScan.py --pw308 3:7:5 --delay 0.5:4.5:5 --reps 12 --note "pw308 x delay"

    # 1-D amplitude
    python RearrangeSTIRAPScan.py --amp556 0.3:1.0:8 --reps 30 --note "556 amp 1-D"

    # top-N verify: a PAIRED path, 100 shots each
    python RearrangeSTIRAPScan.py --covary --reps 100 \
        --pw556 4.0,3.5,3.0 --pw308 4.0,3.0,3.0 --delay 1.35,1.00,1.35 --note "stage-C top-N"

    # fixed-point verify (no swept axis)
    python RearrangeSTIRAPScan.py --reps 100 --note "fixed-point verify at the 09-16 lock"

    # see what would be submitted, without submitting
    python RearrangeSTIRAPScan.py --pw308 3:7:5 --delay 0.5:4.5:5 --note x --dry-run

ANALYSIS (runbook Rules 1 + 2) -- unchanged by this refactor:
  Rule 1: metric is the TARGET-MASKED, verify(mid)-conditioned survival.
  Rule 2: group shots by the logged 1-indexed ``Params``; run_analysis collapses and re-orders
          a co-vary path and will mis-attribute it. 2-D decode is COLUMN-MAJOR, dim-0 fastest.
"""

import argparse
import json
import numpy as np

import scan_bootstrap
scan_bootstrap.bootstrap()

from RearrangeSTIRAPSeq import RearrangeSTIRAPSeq


# ------------------------------- layout + patterns ---------------------------------------- #
VERIFY_IMAGE = True          # 3 frames: img1 load / mid verify / img2 science. Not sweepable.
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

PULSE_WIDTH_CEILING_US = 7.0     # hardware ceiling, per user 2026-07-31

# ------------------------------- the current operating point ------------------------------- #
# THE DEFAULTS BLOCK IS THE LOCK. Update a value here when a scan establishes a better one, and
# say which scan did it -- that one line is the whole provenance trail a reader needs, because
# the run itself already carries its exact source in the code-snapshot store.
#
# Operating point 2026-09-16: 3P1 mj=+1 / 60 G / 66 3S1, every_other on 33x33_feedback11.
DEFAULTS = {
    # two-photon resonance (degenerate diagonal LINE -- lock a PAIR, not each independently)
    "carrier": 96.7036,   # MHz. 09-16 freq top-N verify (20260916182720) KEPT this incumbent:
                          # best alternative only 1.8 sigma and the 4-shot/cell grid ranking
                          # reshuffled under 9x the statistics.
    "eom616": 392.4517,   # MHz (converted to Hz below). Partner of the carrier above.
                          # 09-16 freq-2D ridge: carrier = 97.269 + 1.02*(eom616 - 392.948).
    # pulse area
    "pw556": 3.0,         # us. 09-16 stage-C top-N verify (20260916173225), 100 shots/pt.
    "pw308": 3.0,         # us. 09-16 pw308 x delay (20260916185017): 3.0 best ON-RIDGE by
                          # 4.1 sigma; lengthening 308 does NOT help at pw556 3.0. Matched-width
                          # (BOTH legs long, as in the 08-01 mj=0 95% lock) is still untested.
    "delay": 1.35,        # us, STRICTLY > 0. 1.5 read marginally better in the 09-16 grid
                          # (0.3501 +- 0.0077) but on a 1 us grid; 1.35 is the 100-shot-verified
                          # value, so it stays until a finer delay scan says otherwise.
    # drive amplitudes
    "amp556": 1.0,        # 09-16 1-D amp scan (20260916190146, 30 shots/pt): MONOTONIC to the
                          # ceiling, 0.3730 +- 0.0054 at 1.0, beating 0.90 by 2.4 sigma. Raised
                          # from the long-standing 0.87. NOTE the curve is still climbing ~5 pp
                          # per 0.1 step at 1.0 -- 556 power is the BINDING CONSTRAINT and the
                          # optimum is outside reach; more needs vpp556, not amplitude_scale.
    "amp308": 1.0,        # already at ceiling; the 308 leg is power-limited.
    "vpp556": 14.0,       # max_amplitude_vpp. Raising this is a HARDWARE DRIVE CHANGE -- the
                          # AWG clamps near its bandwidth edge (vpp 17 vs 19 delivered identical
                          # light at the old 143 MHz carrier), so confirm a real survival drop
                          # before believing extra power arrived.
    "vpp308": 8.0,        # amp saturated by 7.5 (07-15).
    "vryd": 2.0,          # Pushout.VRydTrap
    "field": 60.0,        # Pushout.BiasCoilCurrent.Ryd, gauss
}

# Axis name -> (ScanGroup path, CLI-unit -> engine-unit scale)
AXES = {
    "carrier": ("AWG.AWG556.Ch1.carrier_freq_MHz", 1.0),
    "eom616":  ("Init.EOM616.Freq",                1e6),   # MHz in, Hz out
    "pw556":   ("AWG.AWG556.Ch1.pulse_width_us",   1.0),
    "pw308":   ("AWG.AWG308.Ch1.pulse_width_us",   1.0),
    "delay":   ("Pushout.STIRAPDelay",             1e-6),  # us in, seconds out
    "amp556":  ("AWG.AWG556.Ch1.amplitude_scale",  1.0),
    "amp308":  ("AWG.AWG308.Ch1.amplitude_scale",  1.0),
    "vryd":    ("Pushout.VRydTrap",                1.0),
}


def parse_spec(name, text):
    """'3.0' -> [3.0] (pinned) | '3,4,5' -> list | '3:7:5' -> linspace. Returns (values, swept)."""
    t = str(text).strip()
    if ":" in t:
        parts = t.split(":")
        if len(parts) != 3:
            raise SystemExit("--%s: range spec must be start:stop:npts, got %r" % (name, text))
        a, b, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 2:
            raise SystemExit("--%s: npts must be >= 2 in a range spec, got %d" % (name, n))
        return [round(float(v), 6) for v in np.linspace(a, b, n)], True
    if "," in t:
        vals = [round(float(v), 6) for v in t.split(",") if v.strip() != ""]
        if len(vals) < 2:
            raise SystemExit("--%s: comma list needs >= 2 values, got %r" % (name, text))
        return vals, True
    return [float(t)], False


def validate(values):
    """Physics/hardware limits that have each killed a real run. Hard errors, not warnings."""
    for w in ("pw556", "pw308"):
        for v in values[w]:
            if v > PULSE_WIDTH_CEILING_US:
                raise SystemExit("--%s = %.3f us exceeds the %.1f us hardware ceiling"
                                 % (w, v, PULSE_WIDTH_CEILING_US))
            if v <= 0:
                raise SystemExit("--%s must be > 0, got %.3f" % (w, v))
    for v in values["delay"]:
        if v <= 0:
            raise SystemExit(
                "--delay = %.3f us: the delay must be STRICTLY POSITIVE. STIRAPPushoutStep fires "
                "308, waits Forward_Delay, then fires 556, so delay <= 0 is not representable "
                "and raises at sequence-build time, killing the job (09-15, job 2073)." % v)
    for a in ("amp556", "amp308"):
        for v in values[a]:
            if not (0.0 <= v <= 1.0):
                raise SystemExit("--%s = %.3f is outside [0, 1]" % (a, v))


def plan(values, covary):
    """Decide which axes are swept and on which dim. Returns (assignments, n_combos, swept)."""
    swept = [k for k in AXES if len(values[k]) > 1]
    if not swept:
        return {}, 1, []
    if covary:
        lens = {k: len(values[k]) for k in swept}
        if len(set(lens.values())) != 1:
            raise SystemExit("--covary needs every swept axis to have the SAME length; got %s"
                             % lens)
        n = list(lens.values())[0]
        return {k: 1 for k in swept}, n, swept
    if len(swept) > 2:
        raise SystemExit(
            "%d swept axes (%s) but the engine grids only 2 dims. Either sweep at most 2, or "
            "pass --covary to put them all on dim 1 as a paired path." % (len(swept), ", ".join(swept)))
    order = [k for k in AXES if k in swept]          # stable, AXES order
    assign = {k: i + 1 for i, k in enumerate(order)}
    n = 1
    for k in order:
        n *= len(values[k])
    return assign, n, order


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        "47x47_feedbackwarm4": ("phase/47x47_feedbackwarm4.pt", [0, 0, 0, 0, 0]),
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, -0.75]),
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4": ("phase/3270_z4eq4.pt", [0, 0, 0, 0, -4]),
        "33x33_feedback9": ("phase/33x33_feedback9.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),
        "17x17_20um": ("phase/17x17_20um.pt", [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise SystemExit("Unknown pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(verify, init_cfg, target_cfg, init_name, target_name):
    items = [_pattern_item(init_name, init_cfg), _pattern_item(target_name, target_cfg)]
    if verify:
        items.append(_pattern_item(target_name, target_cfg))
    return json.dumps(items)


def build(values, covary=False, reps=10, init_pattern=INIT_PATTERN,
          target_pattern=TARGET_PATTERN):
    """Build (do NOT submit) the ScanGroup. Exercisable offline, no backend needed."""
    from scan_group import ScanGroup

    validate(values)
    assign, n_combos, swept = plan(values, covary)

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(init_pattern)
    target_cfg = _pattern_cfg(target_pattern)
    g = ScanGroup()

    # ---- frame layout ---------------------------------------------------------------- #
    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- the swept / pinned physics axes ---------------------------------------------- #
    for name, (path, scale) in AXES.items():
        vals = [v * scale for v in values[name]]
        node = g()
        parts = path.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        leaf = parts[-1]
        if name in assign:
            getattr(node, leaf).scan(assign[name], [float(v) for v in vals])
        else:
            setattr(node, leaf, float(vals[0]))

    # ---- AWG shapes + drive levels (not swept) ---------------------------------------- #
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    g().AWG.AWG556.Ch1.max_amplitude_vpp = float(values["vpp556"][0])
    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.max_amplitude_vpp = float(values["vpp308"][0])
    g().AWG.AWG308.Ch1.pad_time_us = 2

    # Reverse channels: forward-only scans keep IfReverse=0 and do NOT touch Ch2.
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 119.0363
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 14
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0
    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1
    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave (deferred port; unused) --------------------------------------- #
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out ---------------------------------------------------------------- #
    g().Pushout.BiasCoilCurrent.Ryd = float(values["field"][0])
    g().Pushout.STIRAPReverseDelay = 0.0
    g().Pushout.STIRAPPadTime = 2e-6
    g().Pushout.STIRAPGap = 1e-6
    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.IfRecoveryIonization = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6
    g().Pushout.Pump556Freq = 143.556e6
    g().Pushout.Pump556Amp = 0.5
    g().Pushout.SLMAOMAmpGap = 0.55
    g().Pushout.IfGatePulses = 1
    g().Pushout.IonizationViaDAC = 0
    g().Pushout.TimeIonization = 0.1e-6
    g().Init.VIonizationSet5to8 = 4
    g().Pushout.TIonizationAlign = 0.5e-6

    # ---- warmup + rearrangement --------------------------------------------------------- #
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
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "every_other"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.initial_pattern = init_pattern
    g().rearrange_kwargs.extras.final_pattern = target_pattern
    g().rearrange_kwargs.extras.scienceStep = "stirap"

    # ---- run params --------------------------------------------------------------------- #
    rp.NumPerGroup = int(reps) * int(n_combos)
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    # AUTOMATIC, never a flag: a scrambled EOM sweep unlocks the 616.
    rp.Scramble = 0 if "eom616" in assign else 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg,
                                                init_pattern, target_pattern)
    return g, assign, n_combos, swept


def describe(values, assign, n_combos, swept, reps, note):
    """Compose the mandatory description= from what is ACTUALLY swept, plus the operator's note."""
    if swept:
        shape = "co-vary PATH on dim 1" if all(d == 1 for d in assign.values()) and len(swept) > 1 \
                else ("1-D" if len(swept) == 1 else "2-D grid")
        axes = "; ".join(
            "%s (dim %d, %d pts %s..%s)" % (k, assign[k], len(values[k]), values[k][0], values[k][-1])
            for k in swept)
    else:
        shape, axes = "FIXED POINT (no swept axis)", "none"
    pinned = ", ".join("%s=%s" % (k, values[k][0]) for k in
                       ("carrier", "eom616", "pw556", "pw308", "delay", "amp556", "amp308")
                       if k not in swept)
    scramble = 0 if "eom616" in assign else 1
    return (
        "STIRAP forward, 3P1 mj=+1 / 60 G / 66 3S1, every_other on %s. %s: %s. "
        "%d combos x %d passes = %d shots. PINNED: %s (eom616 in MHz here; submitted in Hz). "
        "NOTE: %s. Analysis: MINIMIZE survival; Rule 1 target-masked verify(mid)-conditioned "
        "metric; Rule 2 group by the logged 1-indexed Params -- run_analysis collapses and "
        "re-orders a co-vary path; 2-D decode COLUMN-MAJOR, dim-0 fastest. IfReverse=0. "
        "Scramble=%d (%s)."
        % (TARGET_PATTERN, shape, axes, n_combos, reps, n_combos * reps, pinned, note,
           scramble, "EOM616 swept -- MUST be 0" if scramble == 0 else "no EOM sweep")
    )


def main():
    ap = argparse.ArgumentParser(
        description="Parameterized forward-STIRAP scan. Every axis takes a scalar (pin), "
                    "'a,b,c' (list) or 'start:stop:npts' (linspace).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in AXES:
        ap.add_argument("--" + name.replace("_", "-"), default=str(DEFAULTS[name]),
                        help="default %s" % DEFAULTS[name])
    ap.add_argument("--vpp556", default=str(DEFAULTS["vpp556"]),
                    help="AWG556 max_amplitude_vpp (HARDWARE DRIVE LEVEL)")
    ap.add_argument("--vpp308", default=str(DEFAULTS["vpp308"]))
    ap.add_argument("--field", default=str(DEFAULTS["field"]), help="Ryd bias coil, gauss")
    ap.add_argument("--covary", action="store_true",
                    help="put ALL swept axes on dim 1 as a paired path (equal lengths)")
    ap.add_argument("--reps", type=int, default=10, help="passes over the sweep")
    ap.add_argument("--note", required=True,
                    help="REQUIRED: why this run exists. Goes into the saved description.")
    ap.add_argument("--init-pattern", default=INIT_PATTERN)
    ap.add_argument("--target-pattern", default=TARGET_PATTERN)
    ap.add_argument("--url", default=None, help="ExptServer URL (default tcp://127.0.0.1:1408)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + print the plan, do NOT submit")
    args = ap.parse_args()

    values = {}
    for name in list(AXES) + ["vpp556", "vpp308", "field"]:
        vals, _ = parse_spec(name, getattr(args, name.replace("-", "_")))
        values[name] = vals
    for name in ("vpp556", "vpp308", "field"):
        if len(values[name]) > 1:
            raise SystemExit("--%s is not sweepable here (it is an AWG programming / field "
                             "parameter); pass a single value." % name)

    g, assign, n_combos, swept = build(values, covary=args.covary, reps=args.reps,
                                       init_pattern=args.init_pattern,
                                       target_pattern=args.target_pattern)
    desc = describe(values, assign, n_combos, swept, args.reps, args.note)

    print("swept  : %s" % (", ".join("%s -> dim %d (%d pts)" % (k, assign[k], len(values[k]))
                                     for k in swept) or "none (fixed point)"))
    print("combos : %d   reps: %d   TOTAL SHOTS: %d" % (n_combos, args.reps, n_combos * args.reps))
    print("Scramble: %d %s" % (0 if "eom616" in assign else 1,
                               "(EOM616 swept -- forced 0)" if "eom616" in assign else ""))
    if args.dry_run:
        print("\n--dry-run: not submitting.\ndescription=%s" % desc)
        return

    from yb_start_scan import ybStartScan
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=args.url, label="RearrangeSTIRAPScan",
                      description=desc, rep=args.reps)
    print("submitted RearrangeSTIRAPScan -> descriptor id %s (NumImages=%d)"
          % (did, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    main()
