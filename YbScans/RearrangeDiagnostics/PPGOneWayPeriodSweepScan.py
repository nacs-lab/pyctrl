"""PPGOneWayPeriodSweepScan.py -- one-way +x transport: step_size x PERIOD at fixed nsteps=50.

*** OFFLINE RE-DETECTION REQUIRED. LIVE img2 SURVIVAL IS INVALID except at step_size=0. ***

THE QUESTION
  Slower pacing lets the atom tolerate a bigger per-frame stroke (the cliff moves 1.30 -> 1.70
  knm px between 0.696 and 3.0 ms), but each frame costs more wall-clock. So which pacing moves
  atoms FASTEST?  Sweep the period in INTEGER MULTIPLES of the 0.696 ms SLM write floor and read
  off two curves:

      d99(T)      -- the 99%-per-step stroke vs period          (how far per step)
      d99(T) / T  -- the 99%-per-step SPEED vs period           (how far per unit time)

  ``d99`` is expected to rise and saturate (the 07-31 round-trip campaign saw the cliff saturate
  at ~1.67 px by 1.5 ms), so ``d99/T`` should PEAK somewhere and then fall as 1/T once the stroke
  stops improving. That peak is the fastest safe transport pacing, and it is the deliverable.

  Periods are integer multiples of 0.696 because that is the measured SLM ``Write_image`` floor:
  a commanded period below it is silently clamped (REARRANGEMENT_TESTING records 0.25/0.5/1.0/2.0
  ms all realising ~2.11-2.13 ms in an earlier configuration), so only multiples are honest.

STRUCTURE. Plain ``RearrangeCommSeq`` survival (no release-recapture) -- this scan is about the
SURVIVAL cliff vs pacing; the thermometry lives in ``PPGOneWayHeatScan``.
``return_trip=False`` => nsteps=50 means 50 steps and the array RESTS displaced by
``dx = step_size * 50``.

WHY THE LIVE SURVIVAL IS WRONG. The array ends translated, and ``imagePatternsJson`` is resolved
ONCE per scan, so the declared frame-1 grid (the unshifted array) is correct only at step 0.
Recover survival per cell offline on the grid shifted by that cell's own ``dx``:

    python pyctrl/tools/ppg_transport_analyze.py --scan-id <id>

The step axis includes 2.25 and 2.5 knm px because at the slow end of the sweep the cliff is
near 1.7 px and the base grid would not reach its far side -- without those points the erfc fit
at 5-8x has nothing to anchor the tail on.

ROI. At step 2.5 x 50 steps the array moves 125 knm px = 5.1 lattice pitches and the leading
columns leave the production ROI (66 of 1068 sites at the extreme). Those cells are KEPT: the
analysis intersects the img1 and shifted-img2 grids per cell and scores only on-sensor sites,
recording ``n_offsensor_sites``. Filter at analysis time, not here.

Run::

    python YbScans/RearrangeDiagnostics/PPGOneWayPeriodSweepScan.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGOneWayPeriodSweepScan.py --reps 10 --force
"""

import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0

BASE_PERIOD_MS = 0.696
PERIOD_MULTS = [1, 2, 3, 4, 5, 6, 7, 8]
NSTEPS = 50

# Base grid + the two far-side points, used at EVERY period so the axis is identical across the
# sweep (the whole point is to compare cliffs, which needs a common step axis).
STEP_SIZES = [0.0, 0.25, 0.50, 0.75, 1.00, 1.15, 1.25, 1.325, 1.40, 1.475,
              1.55, 1.65, 1.80, 2.00, 2.25, 2.50]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(steps=None, mults=None, nsteps=NSTEPS):
    """2-D [step_size x step_period_ms] at fixed one-way nsteps. -> (seq, g, steps, periods)."""
    _bootstrap()
    from scan_group import ScanGroup

    steps = [float(s) for s in (steps if steps else STEP_SIZES)]
    mults = [int(m) for m in (mults if mults else PERIOD_MULTS)]
    periods = [round(m * BASE_PERIOD_MS, 6) for m in mults]

    g = ScanGroup()
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.extras.final_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    g().rearrange_kwargs.extras.n_rounds = 1
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.nsteps = int(nsteps)

    # dim 1: step magnitude (knm px), folded to the xyz 3-vector [s,0,0] per shot.
    rk.extras.step_x.scan(1, list(steps))
    rk.extras.step_y = 0.0
    rk.extras.step_z = 0.0
    # dim 2: the per-frame period, integer multiples of the SLM write floor.
    rk.step_period_ms.scan(2, list(periods))

    rk.extras.return_trip = False               # ONE-WAY
    rk.extras.hold_ms = 0.0                     # no turnaround one-way (explicit; extras sticky)
    rk.extras.depth = False                     # LATERAL, explicit (a prior axial run is sticky)
    rk.extras.true_defocus = False
    rk.extras.depth_piston_corr = 0.0
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    rp.NumPerGroup = 100000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return "RearrangeCommSeq", g, steps, periods


def _desc(steps, periods, nsteps):
    return (
        "*** OFFLINE RE-DETECTION REQUIRED -- LIVE img2 SURVIVAL INVALID except at step=0 *** "
        "ONE-WAY +x grating transport, PERIOD SWEEP (pingponggrating return_trip=False): 2-D "
        "extras.step_x %s knm px x step_period_ms %s ms (= %s x the 0.696 ms SLM Write_image "
        "floor; only integer multiples are honest because a commanded period below the floor is "
        "silently clamped) at FIXED nsteps=%d. One-way => %d steps per shot and the array RESTS "
        "displaced by dx = step*%d, so img2 sees a TRANSLATED array; the declared frame-1 grid is "
        "the UNSHIFTED one and is right only at step=0. Recompute survival offline per cell on "
        "the grid shifted by that cell's own dx (pyctrl/tools/ppg_transport_analyze.py). "
        "DELIVERABLES: the 99%%-per-step stroke d99 vs period T, and the 99%%-per-step SPEED "
        "d99/T vs T. d99 is expected to rise then SATURATE (the 07-31 round-trip campaign saw the "
        "cliff saturate at ~1.67 px by 1.5 ms), so d99/T should PEAK and then fall as 1/T -- that "
        "peak is the fastest safe transport pacing. Step axis carries 2.25/2.5 knm px so the erfc "
        "tail is anchored at the slow end where the cliff sits near 1.7 px. Cells past ~75 knm px "
        "travel leave the production ROI and are KEPT: the analysis scores them on the sites still "
        "on-sensor and records n_offsensor_sites (max 66 of 1068 at step 2.5 x 50). "
        "hold_ms=0, precompute+precompute_host, array 33x33_feedback11, z4 = loading_defocus = -4, "
        "ifEnhanced=True."
        % (",".join("%g" % s for s in steps), ",".join("%g" % p for p in periods),
           ",".join("%g" % (p / BASE_PERIOD_MS) for p in periods), nsteps, nsteps, nsteps)
    )


def main():
    ap = argparse.ArgumentParser(description="One-way transport: step x period sweep.")
    ap.add_argument("--steps", default=None)
    ap.add_argument("--mults", default=None, help="comma-separated integer period multiples")
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    kw = {"nsteps": args.nsteps}
    if args.steps:
        kw["steps"] = [float(x) for x in args.steps.split(",") if x.strip()]
    if args.mults:
        kw["mults"] = [int(x) for x in args.mults.split(",") if x.strip()]

    seq, g, steps, periods = build(**kw)
    if args.dry_run:
        print("seq=%s  nseq=%d  nsteps=%d ONE-WAY" % (seq, g.nseq(), args.nsteps))
        print("steps   (%d): %s" % (len(steps), ", ".join("%g" % s for s in steps)))
        print("periods (%d): %s ms" % (len(periods), ", ".join("%g" % p for p in periods)))
        print("shots at %d reps: %d (~%.2f h at 2.5 s/shot)"
              % (args.reps, g.nseq() * args.reps, g.nseq() * args.reps * 2.5 / 3600))
        print("motion time/shot: %s ms"
              % ", ".join("%.0f" % ((args.nsteps + 1) * p) for p in periods))
        print("max dx = %g knm px (%.2f pitch)"
              % (max(steps) * args.nsteps, max(steps) * args.nsteps / 24.48))
        return
    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    from yb_start_scan import ybStartScan
    did = ybStartScan(seq, g, url=args.url, label="PPGOneWayPeriodSweep_n%d" % args.nsteps,
                      description=_desc(steps, periods, args.nsteps),
                      **({"rep": args.reps} if args.reps else {}))
    print("submitted PPGOneWayPeriodSweep -> id %s (%d pts, %d reps = %d shots)"
          % (did, g.nseq(), args.reps, g.nseq() * args.reps))


if __name__ == "__main__":
    main()
