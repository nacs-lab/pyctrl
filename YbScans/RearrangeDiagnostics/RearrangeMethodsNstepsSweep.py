"""RearrangeMethodsNstepsSweep.py -- ONE interleaved job: which transit producer/trick reaches the
imaging-limited ceiling at the fewest transit frames?  (paper Fig 4 redo, 2026-08-12)

The existing Fig 4 stitches three different days together (07-30 SLMnet/warm-WGS pair, 08-02
overdrive run, 08-02 period-ramp run), so its vertical offsets are partly baseline drift, and the
08-02 re-analysis showed the period ramp's apparent gain is bought TIME, not physics
(0.899 vs 0.705 ms/frame; at matched move time its own control wins).  This scan removes both
problems: every arm is a cell of ONE job with ``Scramble = 1``, so loading / imaging / GPU
conditions and the day itself are shared, and every arm plays the SAME number of frames at the
SAME commanded period -- the trick under test (``half_first``) costs no time by construction.

ARMS (dim-2-style settings, carried as PARALLEL axes on ONE flat axis -- see below)

  model      SLMnet direct_flat transit frames      (wgs_warm 0)
  warm       warm-started phase-locked WGS         (wgs_warm 1, pad 2048 x 3 iters)
  warm+od    warm-WGS + adaptive overdrive          (target_clamp 0.2)
  warm+hf4   warm-WGS + HALF-SIZE FIRST 4 STEPS     (half_first 4)
  ceiling    warm-WGS with ``no_move`` = 1: dst := src, i.e. the SAME frames, pacing, prefill,
             bookend and two images with ZERO displacement.  This is the ceiling every other arm
             is chasing (two images + in-trap dwell + SLM refreshes), measured shot-interleaved
             instead of assumed from a plateau.
  ceilingM   ONE extra cell: the same control on the MODEL producer, which bounds the only
             arm-dependent timing left (prefill: warm-WGS solves ~0.73 ms/frame vs the model's
             ~0.59, i.e. ~7 ms more dead time at nsteps 50).

WHY A FLAT 1-D CELL LIST instead of a rectangular nsteps x arm grid: reps are uniform per cell
(``rep`` = passes over the whole list), so the ONLY way to sample the ceiling arm less often than
the real arms is to give it fewer cells.  Flattening also removes the 2-D column-major reshape
trap (memory gotcha-2d-scan-reshape-column-major).  Here the ceiling gets 4 nsteps (+1 model
cell) against 4 arms x 11 nsteps, i.e. ~10 % of the shots, and ``Scramble`` spreads them over the
whole run so they track drift.

NO RECOMPILE ACROSS CELLS (audited live 2026-08-12).  The rearrange2 stream cache is keyed on
(model_fn, len(grid), infer_mode, overdrive, tau_rise, tau_fall, step_period_ms, max_frames,
tools_reload_gen, settle_lut).  ``wgs_warm`` is NOT in it, and the warm producer (cached per
pad/iters/beam/capacity) and the graphed model producer live in separate slots on the same stream
-- so alternating producers shot to shot is a cache hit both ways.  ``overdrive`` IS in the
signature, which is why EVERY cell sets ``overdrive = True`` and switches OD off with
``target_clamp = 0`` (alpha 0 -> the written frame equals the non-OD frame): one signature for
the whole job, one build, paid by the setup prewarm hook.  Do not edit SLM-server code while this
runs -- the tools reload marker is in the signature.

TIMING IS MEASURED, NOT ASSUMED.  Every cell writes ``watermark = 512`` (>= n_frames, so the
whole transit is staged before the first write and playback rides the panel floor instead of the
producer) and the analysis reads the realized period per cell from the ledger
(``paced_total_ms / (n_frames - 1)``), requiring the arms to agree to <= 1 % before any
nsteps-axis claim.  The 08-02 OD run played 10-13 % slow against a commanded 0.696 for reasons
still unexplained, so this check is not optional.

Ledger labels every shot: ``produce_path`` (wgs_warm2048x3 vs graphed), ``target_clamp``,
``half_first``, ``no_move``, ``step_fracs`` -- so an arm is never inferred from the scan grid.

The label MUST contain 'rearrang' (the Analysis tab's slm_diag sync is name-gated).

Run:
    cd pyctrl
    ./.venv-engine-py312/Scripts/python.exe YbScans/RearrangeDiagnostics/RearrangeMethodsNstepsSweep.py --dry-run
    ...                                     --mode cal  --reps 6     # 5 arms x 4 nsteps
    ...                                     --mode main --reps 50
"""
import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

PERIOD_MS = 0.696          # the SLM write floor
MIN_FRAME_MS = 0.695
WGS_PAD, WGS_ITERS = 2048, 3
WATERMARK = 512            # >= n_frames => stage the WHOLE transit, then play at the floor
DEFOCUS = -4               # today's operating point (measured 08-11 at VSLMServo 1.9)
TAU_RISE_MS, TAU_FALL_MS = 1.7, 3.7
OD_CLAMP = 0.2             # the arm that pays (flat 0.2-0.7; 1.0 penalises low nsteps)
HALF_FIRST = 4             # first 4 steps at half size (user spec, 2026-08-12)

# nsteps grids.  main: sparse below the elbow, DENSE 44-66 where the arms close on the ceiling,
# plus a high anchor that measures the ceiling itself.  cal: the 4 points that locate where each
# arm saturates (and prove the plumbing) in ~2 min of apparatus time.
NS_MAIN = [20, 30, 38, 44, 48, 52, 56, 60, 66, 74, 88]
NS_CAL = [35, 50, 65, 85]
NS_CEIL_MAIN = [20, 44, 60, 88]     # the ceiling arm's own (sparser) nsteps
NS_CEIL_CAL = [35, 85]
NS_CEIL_MODEL = 60                  # the single model-producer ceiling cell (prefill bound)

# arm -> (wgs_warm, target_clamp, half_first, no_move)
ARMS = {
    "model":    (0, 0.0, 0, 0),
    "warm":     (1, 0.0, 0, 0),
    "warm+od":  (1, OD_CLAMP, 0, 0),
    "warm+hf4": (1, 0.0, HALF_FIRST, 0),
}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def cell_list(mode="main", ns=None, ceil_ns=None, arms=None):
    """The flat cell list: [(arm, nsteps, wgs_warm, target_clamp, half_first, no_move), ...]."""
    nsteps = list(ns) if ns else (NS_MAIN if mode == "main" else NS_CAL)
    cns = list(ceil_ns) if ceil_ns else (NS_CEIL_MAIN if mode == "main" else NS_CEIL_CAL)
    names = list(arms) if arms else list(ARMS)
    cells = []
    for name in names:
        w, tc, hf, nm = ARMS[name]
        for n in nsteps:
            cells.append((name, int(n), w, tc, hf, nm))
    for n in cns:                                   # warm-WGS ceiling (no transport)
        cells.append(("ceiling", int(n), 1, 0.0, 0, 1))
    cells.append(("ceilingM", int(NS_CEIL_MODEL), 0, 0.0, 0, 1))   # prefill bound
    return cells


def build(cells, vservo=None):
    """Build (do NOT submit) the ScanGroup.  Every knob is written explicitly in every cell --
    server extras are STICKY, so an unset one inherits the previous scan's value."""
    _bootstrap()
    from scan_group import ScanGroup

    g = ScanGroup()
    # OPTIONAL trap-depth override. ByPattern[33x33_feedback11] carries VSLMServo 1.9 (the 08-11
    # operating point); the Fig-4-era runs were at 3.5. Set BOTH keys, exactly as the daily scans
    # do -- Init.VSLMServo arms the servo and SLM.VServo is what the SLM step asserts.
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
    rk.extras.half_first.scan(1, [int(c[4]) for c in cells])
    rk.extras.no_move.scan(1, [int(c[5]) for c in cells])

    # ---- constant across every cell ----
    rk.extras.overdrive = True          # IN the stream signature; OD is switched off by tc=0
    rk.extras.tau_rise_ms = float(TAU_RISE_MS)
    rk.extras.tau_fall_ms = float(TAU_FALL_MS)
    rk.extras.settle_lut = "auto"
    rk.extras.full_lut_loaded = False
    rk.extras.wgs_pad = int(WGS_PAD)
    rk.extras.wgs_iters = int(WGS_ITERS)
    rk.extras.wgs_beam = "gaussian"
    rk.extras.pattern = "every_other"
    rk.extras.prob_hungarian = True
    rk.extras.dynamic = False           # linear schedule (half_first only applies to linear)
    rk.extras.max_step_size = 0.0
    rk.extras.mover_boost = 0.0
    rk.extras.ramp_frames = 0           # NOT the trick under test; sticky, so pinned off
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
        "ONE interleaved job, %d cells: transit producer/trick vs nsteps on the every_other "
        "target of %s at %.3f ms. Arms = SLMnet model / warm-WGS %dx%d / warm+overdrive tc=%.2f "
        "/ warm+half_first=%d (first 4 steps half size, same frame count and duration) / "
        "no_move CEILING (dst := src, warm producer) + one model-producer ceiling cell. "
        "overdrive=True in every cell with tc=0 as the OD-off control => ONE stream signature, "
        "no recompile between cells. watermark %d stages the whole transit so every arm plays at "
        "the panel floor; realized period is read back per cell from the ledger."
        % (len(cells), PATTERN, PERIOD_MS, WGS_PAD, WGS_ITERS, OD_CLAMP, HALF_FIRST, WATERMARK))

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
    ap.add_argument("--nsteps", default=None, help="comma list override for the 4 real arms")
    ap.add_argument("--ceil-nsteps", default=None, help="comma list override for the ceiling arm")
    ap.add_argument("--arms", default=None, help="comma list subset of %s" % list(ARMS))
    ap.add_argument("--vservo", type=float, default=None,
                    help="override VSLMServo (Init + SLM) for the whole scan; default = the "
                         "pattern's ByPattern value (1.9)")
    ap.add_argument("--label", default=None, help="must still contain 'rearrang'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.nsteps.split(",")] if args.nsteps else None
    cns = [int(x) for x in args.ceil_nsteps.split(",")] if args.ceil_nsteps else None
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    cells = cell_list(args.mode, ns, cns, arms)
    reps = args.reps if args.reps is not None else (6 if args.mode == "cal" else 50)

    if args.vservo is not None:
        print("VSLMServo override: %.2f" % args.vservo)
    print("%d cells x %d reps = %d shots (~%.0f min at 2.6 s/shot)"
          % (len(cells), reps, len(cells) * reps, len(cells) * reps * 2.6 / 60))
    for i, c in enumerate(cells, 1):
        print("  %2d  %-9s n=%-3d wgs_warm=%d tc=%.2f half_first=%d no_move=%d"
              % (i, c[0], c[1], c[2], c[3], c[4], c[5]))
    seq_name, g = build(cells, vservo=args.vservo)
    print("scansize(1) =", g.scansize(1))
    assert g.scansize(1) == len(cells), "cell axis length mismatch"
    if args.dry_run:
        return

    from yb_start_scan import ybStartScan
    label = args.label or ("RearrangeMethodsNsteps_%s" % args.mode)
    assert "earrang" in label, "label must contain 'rearrang' (slm_diag sync is name-gated)"
    g.runp().NumPerGroup = reps * len(cells)
    did = ybStartScan(seq_name, g, url=args.url, label=label, rep=reps)
    print("submitted %s -> id %s (%d cells, reps %d)" % (label, did, len(cells), reps))
    return did


if __name__ == "__main__":
    main()
