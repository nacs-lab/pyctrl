"""RearrangeYbMethodsNstepsSweep.py -- the comparemethods measurement, redone on the "Yb" target.

ONE interleaved job: which transit producer reaches the imaging maximum at the fewest transit
frames, when the target is the ``yb`` letterform instead of ``every_other``?  (user, 2026-09-02)

This is a direct descendant of ``RearrangeMethodsNstepsSweep.py`` (run 858 ``20260812_052648``,
which is the whole data content of ``tmp/slmnet_visuals/comparemethods.png`` via
``campaigns/rearr/methods/final_imax.json``).  Two deliberate changes:

1. TARGET = ``yb`` (``slm/patterns/yb.txt``, 27x27 bitmap, 267 'X' cells, auto-centred on the
   33x33 -> lattice rows 3-29 / cols 3-29, DC site (16,16) is NOT a target) instead of the
   generated ``every_other`` (536 targets).  267 targets against ~600-690 loaded atoms is a
   LARGE surplus -- unlike 08-12, where 536 targets against ~575 server-counted atoms left
   ~21 unfilled per shot and capped raw target-aware survival at ~0.96.  So raw TP and paired
   survival should agree here by construction, not by cut.

2. THE IMAGING MAXIMUM IS AN IN-JOB ARM.  08-12 only carried ``no_move`` (dst := src), and the
   CORRECTION of 2026-08-13 had to reconstruct the actual imaging maximum from a SEPARATE run
   (r909, 0.9941 vs the no_move control's 0.9915) -- a cross-run number on the denominator of
   every crossing.  Here the pure-imaging control rides IN the job.

ARMS (every knob a PARALLEL axis on ONE flat cell axis -- see below)

  model      SLMnet direct_flat transit frames          (wgs_warm 0, tc 0)
  warm       warm-started phase-locked WGS              (wgs_warm 1, tc 0, pad 2048 x 3 iters)
  warm+od    warm-WGS + adaptive overdrive              (wgs_warm 1, tc 0.2)
  no_move    warm-WGS with server ``no_move`` = 1: dst := src.  SAME frames, pacing, prefill,
             bookend and two images, ZERO displacement.  = imaging + in-trap dwell + SLM
             refreshes.  Sparse (4 nsteps).
  imaging    ``no_transit`` = 1 (LAB-side, rearrange_callbacks): NO rearrange() call at all, so
             zero frames, zero writes, zero prefill.  Literally img1 -> img2 on the untouched
             loading array.  = THE IMAGING MAXIMUM, and the denominator everything normalises
             to (user, 2026-09-02).  ~1/10 of the shots.

  The ``imaging`` cells sit at the SAME nsteps values as the ``no_move`` cells, which makes the
  pair a 1:1 subtraction: (no_move - imaging) at each nsteps IS the dwell + refresh cost that
  08-13 could only bound indirectly (it estimated -0.0026 from a dwell slope).  ``nsteps`` is
  inert on the imaging arm by construction, so those 4 cells double as a null check: if they
  are not flat in nsteps, something in the arm is not actually idle.

WHY A FLAT 1-D CELL LIST instead of a rectangular nsteps x arm grid: reps are uniform per cell
(``rep`` = passes over the whole list), so the ONLY way to sample an arm less often is to give
it fewer cells.  Flattening also removes the 2-D column-major reshape trap (memory
gotcha-2d-scan-reshape-column-major).  Main: 3 arms x 12 nsteps = 36, + 4 no_move + 4 imaging
= 44 cells; the imaging control is 4/44 = 9.1 % of the shots and ``Scramble = 1`` spreads it
over the whole run so it tracks drift.

NO RECOMPILE ACROSS CELLS.  The rearrange2 stream cache is keyed on (model_fn, len(grid),
infer_mode, overdrive, tau_rise, tau_fall, step_period_ms, max_frames, tools_reload_gen,
settle_lut).  ``wgs_warm`` / ``no_move`` are NOT in it, ``no_transit`` never reaches the server
at all (rearrange_callbacks pops it before setup_rearrangement), and the warm producer and the
graphed model producer live in separate slots on the same stream.  ``overdrive`` IS in the
signature, which is why EVERY cell sets ``overdrive = True`` and switches OD off with
``target_clamp = 0`` (alpha 0 -> the written frame equals the non-OD frame): one signature for
the whole job, one build.  Do not edit SLM-server code while this runs -- the tools reload
marker is in the signature.

TIMING IS MEASURED, NOT ASSUMED.  Every cell writes ``watermark = 512`` (>= n_frames, so the
whole transit is staged before the first write and playback rides the panel floor instead of the
producer) and the analysis reads the realized period per cell from the ledger
(``paced_total_ms / (n_frames - 1)``), requiring the transport arms to agree to <= 1 % before
any nsteps-axis claim.

PLANE: z4 = -5 (user, 2026-09-02) -- today's rig operating plane, which every scan since 08-12
runs at, NOT the 08-12 campaign's -4.  Depth spread is 1.7x tighter at -5 (CV 5.68 vs 9.51 %)
and imaging is tied (r904/r905), but it does mean the absolute frames-to-99% here are not
stackable on the ``every_other`` figure without noting the plane.

Ledger labels every shot: ``produce_path`` (wgs_warm2048x3 vs graphed), ``target_clamp``,
``no_move``, ``step_fracs``.  The ``imaging`` arm produces NO ledger row at all (it never calls
the server) -- that absence, plus the scan's own ``no_transit`` axis in the descriptor, is how
the analysis identifies it.

The label MUST contain 'rearrang' (the Analysis tab's slm_diag sync is name-gated).

Run:
    cd pyctrl
    ./.venv-engine-py312/Scripts/python.exe YbScans/RearrangeDiagnostics/RearrangeYbMethodsNstepsSweep.py --dry-run
    ...                                     --mode cal            # 14 cells x 6 = 84 shots, ~4 min
    ...                                     --mode main           # 44 cells x 45 = 1980 shots, ~86 min
"""
import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

TARGET_PATTERN = "yb"      # slm/patterns/yb.txt -- 27x27, 267 target sites, centred on the 33x33

PERIOD_MS = 0.696          # the SLM write floor
MIN_FRAME_MS = 0.695
WGS_PAD, WGS_ITERS = 2048, 3
WATERMARK = 512            # >= n_frames => stage the WHOLE transit, then play at the floor
DEFOCUS = -5               # today's operating plane (user, 2026-09-02; 08-12 campaign was -4)
TAU_RISE_MS, TAU_FALL_MS = 1.7, 3.7
OD_CLAMP = 0.2             # the arm that pays (flat 0.2-0.7; 1.0 penalises low nsteps)

# nsteps grids.  NS_MAIN is CUT FROM CALIBRATION run 20260902_174124 (14 cells x 6 reps, taken
# after the loading fix; the earlier cal 20260902_172756 ran at loading 0.38 and agrees on the
# shape but not the level).  The naive expectation was wrong in the informative direction: 264
# targets is HALF every_other's 536, but the curve did NOT move down-scale.  every_other's
# Hungarian move is ~one lattice pitch so almost nothing travels; the yb letterform is COMPACT,
# so 250 of the 264 targets (median) are filled by an atom that has to cross the array.
# Target-aware survival on the cal:
#
#     ns          25      40      55      75      (imaging maximum = 0.9896 +- 0.0012)
#     model     0.353   0.808   0.873   0.926
#     warm      0.391   0.763   0.954   0.954
#     warm+od   0.488   0.731   0.767   0.942
#     no_move                   0.988
#
# Two things that set this grid.  (1) The arms SEPARATE over 25-75 and the rise is steep, so the
# crossing statistic needs its density there, not above it.  (2) warm reads the same 0.954 at 55
# and 75 while the imaging maximum is 0.9896 -- 99 % of which is 0.9797.  If that plateau is
# REAL, no arm ever reaches the crossing and the campaign has no answer; if it is the 6-rep noise
# it will lift with statistics.  Either way it has to be settled, hence the anchors out to 200
# frames (139 ms of transit) even though nothing in the cal suggests the curve is still climbing.
NS_MAIN = [30, 40, 50, 60, 70, 80, 90, 105, 120, 140, 165, 200]    # 12 pts, 3 arms
NS_CAL = [25, 40, 55, 75]                                          # what the cals above ran

# The no_move (dwell + writes) arm and the pure-imaging arm share their nsteps values so each
# pair subtracts 1:1.  nsteps is inert on the imaging arm -- those cells are replicates plus a
# flatness null check.
NS_PAIRED_MAIN = [30, 70, 120, 200]
NS_PAIRED_CAL = [55]

# arm -> (wgs_warm, target_clamp, no_move, no_transit)
ARMS = {
    "model":   (0, 0.0, 0, 0),
    "warm":    (1, 0.0, 0, 0),
    "warm+od": (1, OD_CLAMP, 0, 0),
}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def cell_list(mode="main", ns=None, paired_ns=None, arms=None):
    """The flat cell list: [(arm, nsteps, wgs_warm, target_clamp, no_move, no_transit), ...]."""
    nsteps = list(ns) if ns else (NS_MAIN if mode == "main" else NS_CAL)
    pns = list(paired_ns) if paired_ns else (NS_PAIRED_MAIN if mode == "main" else NS_PAIRED_CAL)
    names = list(arms) if arms else list(ARMS)
    cells = []
    for name in names:
        w, tc, nm, nt = ARMS[name]
        for n in nsteps:
            cells.append((name, int(n), w, tc, nm, nt))
    for n in pns:                       # dwell + writes control (warm producer, zero displacement)
        cells.append(("no_move", int(n), 1, 0.0, 1, 0))
    for n in pns:                       # THE IMAGING MAXIMUM: two images, nothing between them
        cells.append(("imaging", int(n), 1, 0.0, 0, 1))
    return cells


def build(cells, vservo=None):
    """Build (do NOT submit) the ScanGroup.  Every knob is written explicitly in every cell --
    server extras are STICKY, so an unset one inherits the previous scan's value."""
    _bootstrap()
    from scan_group import ScanGroup

    g = ScanGroup()
    # OPTIONAL trap-depth override.  Default = ByPattern[33x33_feedback11]'s VSLMServo 1.9.
    # Set BOTH keys, exactly as the daily scans do -- Init.VSLMServo arms the servo and
    # SLM.VServo is what the SLM step asserts.
    if vservo is not None:
        g().Init.VSLMServo = float(vservo)
        g().SLM.VServo = float(vservo)

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

    rk = g().rearrange_kwargs
    rk.protocol = "rearrange2"
    rk.step_period_ms = float(PERIOD_MS)
    rk.min_frame_ms = float(MIN_FRAME_MS)
    rk.extras.n_rounds = 1

    # ---- the flat cell axis: every arm knob is a PARALLEL axis on dim 1 ----
    rk.nsteps.scan(1, [int(c[1]) for c in cells])
    rk.extras.wgs_warm.scan(1, [int(c[2]) for c in cells])
    rk.extras.target_clamp.scan(1, [float(c[3]) for c in cells])
    rk.extras.no_move.scan(1, [int(c[4]) for c in cells])
    # LAB-SIDE ONLY: popped by rearrange_callbacks.pre_run before setup_rearrangement, so it
    # never reaches the server and never touches the rearrange2 stream signature.
    rk.extras.no_transit.scan(1, [int(c[5]) for c in cells])

    # ---- constant across every cell ----
    rk.extras.overdrive = True          # IN the stream signature; OD is switched off by tc=0
    rk.extras.tau_rise_ms = float(TAU_RISE_MS)
    rk.extras.tau_fall_ms = float(TAU_FALL_MS)
    rk.extras.settle_lut = "auto"
    rk.extras.full_lut_loaded = False
    rk.extras.wgs_pad = int(WGS_PAD)
    rk.extras.wgs_iters = int(WGS_ITERS)
    rk.extras.wgs_beam = "gaussian"
    rk.extras.pattern = TARGET_PATTERN
    rk.extras.prob_hungarian = True
    rk.extras.dynamic = False           # linear schedule
    rk.extras.half_first = 0            # 08-12's NULL trick; sticky, so pinned off
    rk.extras.max_step_size = 0.0
    rk.extras.mover_boost = 0.0
    rk.extras.ramp_frames = 0           # sticky, so pinned off
    rk.extras.frame_repeats = 0         # ditto (a stale list would silently re-add dwell)
    rk.extras.watermark = int(WATERMARK)
    rk.extras.precompute = True         # inert for rearrange2; written for the sticky log
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.skip_final_phase = False  # WGS bookend written, settled, non-overdriven
    rk.extras.flip_phase = False
    rk.extras.phase_alpha_piston = 0.0
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN
    rk.extras.description = (
        "ONE interleaved job, %d cells: transit producer vs nsteps on the '%s' target (267 "
        "sites, 27x27 bitmap centred on %s) at %.3f ms, z4 %d. Arms = SLMnet model / warm-WGS "
        "%dx%d / warm+overdrive tc=%.2f, plus TWO controls at matched nsteps: no_move (dst := "
        "src -- same frames, pacing, prefill and bookend, zero displacement = imaging + dwell + "
        "refresh) and no_transit (NO rearrange call at all -- zero frames, zero writes = THE "
        "IMAGING MAXIMUM, the denominator everything normalises to). Their difference is the "
        "dwell+refresh term 08-13 could only bound indirectly. overdrive=True in every cell "
        "with tc=0 as the OD-off control => ONE stream signature, no recompile between cells. "
        "watermark %d stages the whole transit so every arm plays at the panel floor; realized "
        "period is read back per cell from the ledger."
        % (len(cells), TARGET_PATTERN, PATTERN, PERIOD_MS, DEFOCUS,
           WGS_PAD, WGS_ITERS, OD_CLAMP, WATERMARK))

    rp.NumPerGroup = 100000             # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1                     # per-pass shuffle -> the arms interleave
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    rp.imagePatternsJson = json.dumps([it, dict(it)])
    return "RearrangeCommSeq", g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--mode", default="main", choices=("main", "cal"))
    ap.add_argument("--reps", type=int, default=None, help="passes over the cell list")
    ap.add_argument("--nsteps", default=None, help="comma list override for the 3 transport arms")
    ap.add_argument("--paired-nsteps", default=None,
                    help="comma list override for the no_move + imaging control cells")
    ap.add_argument("--arms", default=None, help="comma list subset of %s" % list(ARMS))
    ap.add_argument("--vservo", type=float, default=None,
                    help="override VSLMServo (Init + SLM) for the whole scan; default = the "
                         "pattern's ByPattern value (1.9)")
    ap.add_argument("--label", default=None, help="must still contain 'rearrang'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.nsteps.split(",")] if args.nsteps else None
    pns = [int(x) for x in args.paired_nsteps.split(",")] if args.paired_nsteps else None
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    cells = cell_list(args.mode, ns, pns, arms)
    reps = args.reps if args.reps is not None else (6 if args.mode == "cal" else 45)

    n_ctrl = sum(1 for c in cells if c[5])
    if args.vservo is not None:
        print("VSLMServo override: %.2f" % args.vservo)
    print("%d cells x %d reps = %d shots (~%.0f min at 2.6 s/shot); imaging control %d/%d "
          "cells = %.1f %% of shots"
          % (len(cells), reps, len(cells) * reps, len(cells) * reps * 2.6 / 60,
             n_ctrl, len(cells), 100.0 * n_ctrl / len(cells)))
    for i, c in enumerate(cells, 1):
        print("  %2d  %-9s n=%-3d wgs_warm=%d tc=%.2f no_move=%d no_transit=%d"
              % (i, c[0], c[1], c[2], c[3], c[4], c[5]))
    seq_name, g = build(cells, vservo=args.vservo)
    print("scansize(1) =", g.scansize(1))
    assert g.scansize(1) == len(cells), "cell axis length mismatch"
    if args.dry_run:
        return

    from yb_start_scan import ybStartScan
    label = args.label or ("RearrangeYbMethodsNsteps_%s" % args.mode)
    assert "earrang" in label, "label must contain 'rearrang' (slm_diag sync is name-gated)"
    g.runp().NumPerGroup = reps * len(cells)
    did = ybStartScan(seq_name, g, url=args.url, label=label, rep=reps)
    print("submitted %s -> id %s (%d cells, reps %d)" % (label, did, len(cells), reps))
    return did


if __name__ == "__main__":
    main()
