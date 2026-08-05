"""WarmWGSTricksStepSweep.py -- do the rearrange2 "tricks" (adaptive OVERDRIVE, arrive-and-hold
STEP SCHEDULING, MOVER BOOST) beat the plain warm-WGS transit, and by how much?

Every mode is a 2-D scan: dim 1 = ``nsteps`` (10..60), dim 2 = the trick setting, with the
trick's OWN off-cell as dim-2 point 0.  That is the point of the design: the control is
INTERLEAVED shot-to-shot with the treatment inside one job (per-pass Scramble on), so loading
drift / imaging drift / GPU load cancel between the arms instead of separating two jobs by
15 minutes.  Each mode therefore also re-measures the no-tricks survival-vs-nsteps curve.

Why nsteps 10..60: with linear scheduling every atom moves ``L_i / nsteps`` knm-1024 px per
frame, so the nsteps axis IS the per-frame step-size axis.  For the every_other checkerboard
target on 33x33_feedback11 the Hungarian moves are ~1 lattice pitch (~23 knm px), so nsteps 10
is ~2.3 px/frame (far over the ~1.25 px trap-overlap cliff -> abysmal) and nsteps 60 is
~0.38 px/frame (well under the ~0.9 px heating onset -> should be at the imaging cap).  Any
trick that wins has to show up as a LEFT-SHIFT of that curve: same survival at fewer frames.

Modes (dim-2 axis; everything else identical, all trick knobs written EXPLICITLY in every mode
because the server extras are STICKY):

  od     ``target_clamp`` = [0, 0.2, 1.0] with ``overdrive=True``.  target_clamp is the largest
         admissible clamped-pixel fraction and alpha is the largest grid value that respects it,
         so 0 -> alpha 0 = OD exactly OFF (the control), 0.2 = clamp-limited adaptive OD,
         1.0 = full alpha=1.0 zero-lag landing.  OD pre-distorts the written retardance so the
         LC lands on the next frame sooner; its win only exists where survival is SETTLE-limited,
         which at 0.696 ms is the low-nsteps end.
  sched  parallel dim-2 pair (``dynamic``, ``max_step_size``) =
         [(0, 0), (1, 0.7), (1, 1.0)] = linear / arrive-and-hold capped at 0.7 px per frame /
         capped at 1.0 px (the deployed default).  Arrive-and-hold moves every atom at
         ``max(cap, L_i/F)``, so short movers land early and sit settled-bright.  Prediction to
         test: at LOW nsteps L_i/F > cap and the cap is inert (== linear); at HIGH nsteps the cap
         FORCES atoms up to the cap, which is a step-size increase -- 1.0 px is at the heating
         onset (0.9 px, 08-01 campaign) so it should HURT, while 0.7 px is thermally free and
         buys early arrival.  0.7 vs 1.0 is therefore the real question, not dynamic vs linear.
  boost  ``mover_boost`` = [1.0, 1.41, 2.0] (1.0 = amps all 1.0 = OFF).  Multiplies the MOVING
         spots' target amplitude for the whole transit; statics stay at 1.0.  Warm-WGS realizes
         per-spot amplitude FAITHFULLY (an amplitude-blind model silently ignores it), which is
         why this is a warm-WGS test.  Campaign evidence: x1.41 -> ~+45% mover
         survival-brightness.  Watch the cost: fixed total power means boosted movers dim the
         statics, so 2.0 can lose static atoms even as movers improve.
  combo  parallel dim-2 pair of full configs: [all off] vs [OD 1.0 + dynamic cap 0.7 + boost
         1.41] -- the direct "how much better than no-tricks can we get" answer.
  cells  PHASE 2 (data-driven): arbitrary dim-2 cells from ``--cells``, for the pairwise and
         three-way COMBINATIONS of whichever single tricks actually won in phase 1:
             --cells "tc=0,dyn=0,cap=0,boost=1; tc=1,boost=1.41; dyn=1,cap=0.7,boost=1.41"
         Each ``;``-separated cell is ``k=v`` pairs over tc / dyn / cap / boost (omitted keys take
         the OFF value), the FIRST cell is the control, and ``overdrive`` stays True in every cell
         (OD is switched off by ``tc=0``) so the whole scan keeps ONE stream-cache signature --
         no torch.compile / cudagraph rebuild as the cells interleave.

PRECOMPUTE.  The extras ``precompute`` / ``precompute_host`` are written True, but be clear that
they are INERT for rearrange2: the streaming bridge never passes ``precompute`` into
``run_positions``, and ``_run_host_stream`` gates only on ``watermark``.  The real "precompute
the whole transit" knob for rearrange2 is ``watermark >= n_frames`` -- so this scan sets
``watermark = 512`` (internally clamped to n_frames), which makes the producer solve and stage
EVERY frame before the first SLM write.  That is what protects a warm-WGS run from GPU
contention: a slow solve then costs dead time before the move instead of a LATE frame during it.
NOTE it is sticky -- a later production rearrangement scan inherits watermark 512 unless it sets
its own.

Single-round (RearrangeCommSeq, 2 images): 33x33_feedback11 -> EVERY_OTHER checkerboard subset.
Frames from warm-started phase-locked WGS (pad 2048 x 3 iters, ~0.55-0.9 ms/frame).

The label MUST contain 'rearrang' (the Analysis tab's slm_diag sync is name-gated).

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/WarmWGSTricksStepSweep.py --mode od    --dry-run
    python YbScans/RearrangeDiagnostics/WarmWGSTricksStepSweep.py --mode od    --reps 8 --force
"""
import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

NSTEPS = [10, 15, 20, 25, 30, 35, 40, 50, 60]   # dim 1 = the per-frame step-size axis
WGS_PAD = 2048                                  # warm-WGS FFT pad (full teacher efficiency)
WGS_ITERS = 3                                   # >= 3 (2 is the contract-quality cliff)
DEFOCUS = -4                                    # matched loading_defocus / rearrange z4
PERIOD_MS = 0.696                               # the panel bank-ready floor
WATERMARK = 512                                 # >= n_frames => stage the WHOLE transit first
TAU_RISE_MS = 1.7                               # rig-measured LC settle (OD)
TAU_FALL_MS = 3.7

# dim-2 axes, per mode
OD_CLAMPS = [0.0, 0.2, 1.0]
SCHED_DYNAMIC = [0, 1, 1]
SCHED_MAXSTEP = [0.0, 0.7, 1.0]
BOOSTS = [1.0, 1.41, 2.0]
COMBO_CLAMP = [0.0, 1.0]
COMBO_DYNAMIC = [0, 1]
COMBO_MAXSTEP = [0.0, 0.7]
COMBO_BOOST = [1.0, 1.41]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames image the SAME load array (the target is a subset of it).  Name = the phase
    file BASENAME -- the detection registry + expConfig ByPattern are keyed by it."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _common(g, nsteps_list):
    """Everything identical in every mode.  Returns the rearrange_kwargs handle."""
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.extras.final_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    g().rearrange_kwargs.extras.n_rounds = 1
    rk = g().rearrange_kwargs
    rk.protocol = "rearrange2"
    rk.nsteps.scan(1, [int(n) for n in nsteps_list])   # dim 1: the step-size axis
    rk.step_period_ms = float(PERIOD_MS)
    rk.min_frame_ms = 0.695

    # ---- warm-started phase-locked WGS transit frames ----
    rk.extras.wgs_warm = True
    rk.extras.wgs_pad = int(WGS_PAD)
    rk.extras.wgs_iters = int(WGS_ITERS)
    rk.extras.wgs_beam = "gaussian"

    # ---- target + assignment ----
    rk.extras.pattern = "every_other"          # checkerboard subset ((row+col) parity 0)
    rk.extras.prob_hungarian = True

    # ---- staging: watermark IS rearrange2's precompute (see module docstring) ----
    rk.extras.watermark = int(WATERMARK)
    rk.extras.precompute = True                # inert for rearrange2; written for the sticky log
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False

    # ---- OD physics constants (the strength itself is per-mode) ----
    rk.extras.tau_rise_ms = float(TAU_RISE_MS)
    rk.extras.tau_fall_ms = float(TAU_FALL_MS)
    rk.extras.settle_lut = "auto"              # measured destination-keyed tau LUT when present
    rk.extras.full_lut_loaded = False          # stroke = 532 LUT [0, 2pi]

    # ---- phase / bookend ----
    rk.extras.ifEnhanced = True                # BlueLAC loading
    rk.extras.z4 = DEFOCUS                     # transit plane == loading plane
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN
    rk.extras.skip_final_phase = False         # WGS bookend written (settled, non-overdriven)
    rk.extras.flip_phase = False
    rk.extras.phase_alpha_piston = 0.0

    rp.NumPerGroup = 100000                    # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1                            # per-pass shuffle -> the arms interleave
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return rk


def _set_od(rk, clamps, overdrive=True, dim=None):
    if dim is None:
        rk.extras.overdrive = bool(overdrive)
        rk.extras.target_clamp = float(clamps)
    else:
        rk.extras.overdrive = True             # tc=0 -> alpha 0 -> OD off (the control cell)
        rk.extras.target_clamp.scan(dim, [float(c) for c in clamps])


def _set_sched(rk, dynamic, max_step, dim=None):
    if dim is None:
        rk.extras.dynamic = bool(dynamic)
        rk.extras.max_step_size = float(max_step)
    else:
        rk.extras.dynamic.scan(dim, [int(d) for d in dynamic])
        rk.extras.max_step_size.scan(dim, [float(s) for s in max_step])


def _set_ramp(rk, ramp, dim=None):
    """extras.ramp_frames: hold the first N transit frames for N+1..2 panel refreshes (server-side
    period ramp, added 2026-08-02, default 0 = OFF). SCALAR so it can be a swept axis. STICKY --
    written explicitly (0) in every mode so a later scan cannot inherit a ramp."""
    if dim is None:
        rk.extras.ramp_frames = int(ramp)
    else:
        rk.extras.ramp_frames.scan(dim, [int(r) for r in ramp])


def _set_boost(rk, boost, dim=None):
    if dim is None:
        rk.extras.mover_boost = float(boost)
    else:
        rk.extras.mover_boost.scan(dim, [float(b) for b in boost])


CELL_OFF = {"tc": 0.0, "dyn": 0, "cap": 0.0, "boost": 1.0, "ramp": 0}


def parse_cells(spec):
    """``"tc=0,dyn=0; tc=1,boost=1.41"`` -> [{tc,dyn,cap,boost}, ...] (first = the control).

    Keys omitted in a cell take their OFF value, so a cell only names the knobs it turns on."""
    cells = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        cell = dict(CELL_OFF)
        for kv in chunk.split(","):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            k = k.strip().lower()
            if k not in CELL_OFF:
                raise ValueError("unknown cell key %r (want tc/dyn/cap/boost)" % k)
            cell[k] = int(float(v)) if k in ("dyn", "ramp") else float(v)
        cells.append(cell)
    if len(cells) < 2:
        raise ValueError("--cells needs at least a control cell and one treatment cell")
    return cells


def _cell_name(c):
    if (c["tc"] == 0 and not c["dyn"] and c["boost"] in (0.0, 1.0)
            and not c.get("ramp")):
        return "no tricks"
    bits = []
    if c["tc"]:
        bits.append("OD tc=%g" % c["tc"])
    if c["dyn"]:
        bits.append("cap %g" % c["cap"])
    if c["boost"] not in (0.0, 1.0):
        bits.append("boost x%g" % c["boost"])
    if c.get("ramp"):
        bits.append("ramp %d" % c["ramp"])
    return " + ".join(bits)


def build(mode, nsteps_list=None, cells=None):
    """Build (do NOT submit) the ScanGroup.  Returns (seq_name, g, description)."""
    _bootstrap()
    from scan_group import ScanGroup

    ns = list(nsteps_list or NSTEPS)
    g = ScanGroup()
    rk = _common(g, ns)
    head = ("warm-WGS (pad %d x %d iters, gaussian beam) rearrange2 transit, 33x33_feedback11 -> "
            "EVERY_OTHER checkerboard, single round, step_period_ms=%g, watermark=%d (whole "
            "transit staged before the first write), z4=%g. nsteps %s on dim 1 = the per-frame "
            "step-size axis. "
            % (WGS_PAD, WGS_ITERS, PERIOD_MS, WATERMARK, DEFOCUS,
               ",".join(str(n) for n in ns)))

    if mode == "od":
        _set_od(rk, OD_CLAMPS, dim=2)
        _set_sched(rk, 0, 0.0)
        _set_boost(rk, 0.0)
        _set_ramp(rk, 0)
        desc = (head + "TRICK = adaptive OVERDRIVE: dim 2 sweeps target_clamp %s with "
                "overdrive=True, where 0 selects alpha=0 and is therefore the OD-OFF control "
                "INTERLEAVED with the treatment (per-pass Scramble), 0.2 is clamp-limited "
                "adaptive OD and 1.0 is full zero-lag landing. tau_rise/fall %g/%g ms, "
                "settle_lut auto, stroke = 532 LUT [0,2pi]. Question: does pre-distorting the "
                "written retardance so the LC lands on the next frame sooner buy survival at "
                "SMALL nsteps (where the transit is settle-limited), i.e. does the "
                "survival-vs-nsteps curve shift LEFT? Scheduling is linear and mover_boost off "
                "in every cell, so overdrive is the only variable."
                % (OD_CLAMPS, TAU_RISE_MS, TAU_FALL_MS))
    elif mode == "sched":
        _set_od(rk, 0.0, overdrive=False)
        _set_sched(rk, SCHED_DYNAMIC, SCHED_MAXSTEP, dim=2)
        _set_boost(rk, 0.0)
        _set_ramp(rk, 0)
        desc = (head + "TRICK = STEP SCHEDULING: dim 2 is the parallel pair (dynamic, "
                "max_step_size) = [(linear, -), (arrive-and-hold, cap 0.7 px/frame), "
                "(arrive-and-hold, cap 1.0 px/frame = the deployed default)], so linear is the "
                "control interleaved with both caps. Arrive-and-hold moves every atom at "
                "max(cap, L_i/nsteps) and holds it settled-bright once it lands. At low nsteps "
                "L_i/nsteps exceeds the cap and the cap is inert; at high nsteps the cap RAISES "
                "the step of atoms that could have gone slower -- 1.0 px sits at the measured "
                "0.9 px heating onset while 0.7 px is thermally free, so the test is whether a "
                "short fast-but-safe move beats a long gentle one. Overdrive off and mover_boost "
                "off in every cell.")
    elif mode == "boost":
        _set_od(rk, 0.0, overdrive=False)
        _set_sched(rk, 0, 0.0)
        _set_boost(rk, BOOSTS, dim=2)
        _set_ramp(rk, 0)
        desc = (head + "TRICK = MOVER BOOST: dim 2 sweeps mover_boost %s (1.0 = all amps 1.0 = "
                "OFF, the interleaved control). The MOVING spots' target amplitude is multiplied "
                "for the whole transit while statics stay at 1.0; warm-WGS realizes per-spot "
                "amplitude faithfully (an amplitude-blind model ignores the channel), so this is "
                "a warm-WGS-only knob. Prior evidence: x1.41 gave ~+45%% mover "
                "survival-brightness. Read BOTH signs of the effect -- total power is fixed, so "
                "boosted movers necessarily dim the statics and 2.0 may lose static atoms while "
                "it saves movers; per-site survival split by mover/static is the readout. "
                "Overdrive off, linear scheduling in every cell." % BOOSTS)
    elif mode == "combo":
        _set_od(rk, COMBO_CLAMP, dim=2)
        _set_sched(rk, COMBO_DYNAMIC, COMBO_MAXSTEP, dim=2)
        _set_boost(rk, COMBO_BOOST, dim=2)
        _set_ramp(rk, 0)
        desc = (head + "TRICK = ALL THREE STACKED: dim 2 has two cells only -- [no tricks: OD "
                "off, linear, boost off] vs [full OD target_clamp 1.0 + arrive-and-hold cap 0.7 "
                "px + mover_boost 1.41] -- interleaved shot-to-shot. This is the direct answer "
                "to 'can the tricks beat the plain warm-WGS transit, and by how much', measured "
                "as the nsteps shift at matched survival. Only meaningful alongside the three "
                "single-trick modes, which say WHICH term carries any gain.")
    elif mode == "cells":
        cl = list(cells or [])
        if len(cl) < 2:
            raise ValueError("mode 'cells' needs --cells with >= 2 cells")
        _set_od(rk, [c["tc"] for c in cl], dim=2)          # overdrive True, tc=0 => OD off
        _set_sched(rk, [c["dyn"] for c in cl], [c["cap"] for c in cl], dim=2)
        _set_boost(rk, [c["boost"] for c in cl], dim=2)
        _set_ramp(rk, [c["ramp"] for c in cl], dim=2)
        desc = (head + "TRICK = COMBINATIONS (phase 2, chosen from the phase-1 single-trick "
                "results): dim 2 interleaves %d configs -- %s -- with the first as the no-tricks "
                "control. overdrive=True in every cell with OD switched off by target_clamp=0, so "
                "the whole scan keeps ONE server stream-cache signature and no cell change can "
                "trigger a recompile. Question: do the single-trick gains ADD, saturate, or fight "
                "each other, and does any combination raise the false-positive rate."
                % (len(cl), " | ".join(_cell_name(c) for c in cl)))
    else:
        raise ValueError("unknown mode %r" % mode)

    return "RearrangeCommSeq", g, desc


def main():
    ap = argparse.ArgumentParser(
        description="warm-WGS rearrange2: overdrive / step-scheduling / mover-boost vs nsteps.")
    ap.add_argument("--mode", required=True,
                    choices=("od", "sched", "boost", "combo", "cells"))
    ap.add_argument("--cells", default=None,
                    help="mode 'cells': ';'-separated dim-2 cells of k=v over tc/dyn/cap/boost, "
                         "first = control, e.g. "
                         "\"tc=0; tc=1; ramp=3; tc=0.2,ramp=3\"  (keys tc/dyn/cap/boost/ramp)")
    ap.add_argument("--nsteps", default=None,
                    help="comma-separated nsteps override (default %s)"
                         % ",".join(str(n) for n in NSTEPS))
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (shots = reps * npts). Without it the scan runs "
                         "to NumPerGroup = 100000, i.e. until aborted.")
    ap.add_argument("--label", default=None, help="must contain 'rearrang'")
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    ns = [int(x) for x in args.nsteps.split(",") if x.strip()] if args.nsteps else None
    cells = None
    if args.mode == "cells":
        if not args.cells:
            ap.error("--mode cells needs --cells")
        cells = parse_cells(args.cells)
    seq_name, g, desc = build(args.mode, nsteps_list=ns, cells=cells)
    label = args.label or ("WarmWGSRearrangeTricks_%s" % args.mode)
    if "rearrang" not in label.lower():
        ap.error("label must contain 'rearrang' (Analysis-tab slm_diag name gate)")

    if args.dry_run:
        print("seq=%s  mode=%s  nseq=%d  label=%s" % (seq_name, args.mode, g.nseq(), label))
        print("nsteps (dim 1) = %s" % (ns or NSTEPS))
        print("shots = %d * reps" % g.nseq())
        print("desc: %s" % desc)
        return

    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label=label, description=desc, **opts)
    print("submitted %s -> descriptor id %s (%d pts, reps=%s)"
          % (label, did, g.nseq(), args.reps))
    return did


if __name__ == "__main__":
    main()
