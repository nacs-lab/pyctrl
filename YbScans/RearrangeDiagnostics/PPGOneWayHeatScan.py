"""PPGOneWayHeatScan.py -- temperature vs STEP SIZE for ONE-WAY +x grating transport.

*** OFFLINE RE-DETECTION REQUIRED FOR SURVIVAL. See the note at the bottom. ***

THE QUESTION
  The 08-05 one-way transport scans measured SURVIVAL vs (step_size, nsteps) and found a per-step
  cliff at ~1.32 knm-px @ 0.696 ms.  Survival alone cannot say whether a stroke BELOW that cliff
  is thermally free: the 07-31 round-trip campaign found radial heating switches on at ~0.9 px,
  i.e. ~0.70x the survival cliff, so a stroke chosen from survival data alone can silently leave
  the array hot.  This scan measures the TEMPERATURE directly, by release-and-recapture, across
  the WHOLE step-size grid at fixed ``nsteps = 50``, one-way.

  It is the one-way counterpart of ``PPGHeatOpPointScan --mode heat``, which swept nsteps at a
  few fixed operating points.  Here nsteps is pinned and the STEP AXIS is swept, because the
  one-way survival curves are parametrized by step size and the deliverable is "how hot is the
  array at the stroke I would actually pick".

STRUCTURE (seq ``RearrangeRnRHeatCommSeq``, identical to the 07-31 campaign)
      img1 -> Cool556 5 ms (prepared cold state) -> pingponggrating ONE-WAY motion (50 steps)
           -> PostRearrCool with amps 0 (NO post-motion cooling, 0.5 ms dark hold)
           -> release t -> recapture -> img2
  Post-motion cooling is OFF so the release probes the post-motion temperature; production's
  5 ms cool would re-thermalize to ~8 uK and hide the effect.

ONE-WAY (``return_trip=False``) is the whole point and it changes two things vs the 07-31 runs:
  1. ``nsteps = 50`` means **50 steps**, not 100 -- there is no return leg.  Per-step survival is
     ``S**(1/50)``, not ``S**(1/100)``.
  2. The array ENDS displaced by ``dx = step_size * 50`` knm-px, so img2 sees a TRANSLATED array
     and the live detection grid is wrong for every step size except 0.  Survival must be
     re-derived offline (see below).  **The TEMPERATURE fit is unaffected**: the release curve is
     normalized by its own t=0 point at the same cell, so any constant per-cell detection
     efficiency divides out -- but only if the SAME (wrong-or-right) grid is used for all release
     times within a cell, which it is.  The temperatures are therefore usable straight from the
     standard release-recapture fit; only the absolute S(t0) needs the offline pass.

THE t=0 CONTROL SUBTLETY (07-31 campaign caveat #4, and why the release grid starts 0, 0.5)
  ``RearrangeRnRStep`` RETURNS EARLY at exactly ``Time == 0`` -- no ``AmpSLM`` toggle, no
  ``TTLSampleAndHold`` re-assert, no 3 us AOM settle -- so the t=0 shot never pays the trap
  restore that every t>0 shot pays.  Measured cost: S(6 us) exceeds S(t=0) by 0.010-0.012 at
  large nsteps.  ``t = 0.5 us`` takes the FULL branch (the release is 0.5 us, far shorter than
  the ~6 us where a cold atom starts to leave), so it is the restore-only reference.  Keeping
  BOTH brackets the effect: t=0 is the byte-clean held-in-traps baseline, t=0.5 is the
  same-bytes-as-the-data baseline.

STATISTICS.  12 reps (the 07-31 campaign used 8) -- 1.5x the data per point, so the survivor
distribution can be examined for bias rather than assumed thermal.

STEP GRID.  Base 14 values, dense through the measured 0.696 ms elbow.  At 4x (2.784 ms) the
cliff moves right (1.30 px @ 0.696 -> ~1.70 px @ 2.8), so ``--period4`` adds 2.25 and 2.5.

Run::

    python YbScans/RearrangeDiagnostics/PPGOneWayHeatScan.py --period 0.696 --dry-run
    python YbScans/RearrangeDiagnostics/PPGOneWayHeatScan.py --period 0.696 --reps 12 --force
    python YbScans/RearrangeDiagnostics/PPGOneWayHeatScan.py --period 2.784 --reps 12 --force
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

BASE_PERIOD_MS = 0.696                 # the SLM write floor; all periods are integer multiples
NSTEPS = 50                            # ONE-WAY -> 50 steps per shot (not 100)
HOLD_MS = 0.0                          # no turnaround exists one-way; explicit anyway (sticky)
POST_COOL_HOLD_MS = 0.5

# Base step axis (knm-px/step): dense 0.075-0.10 through the measured 0.696 ms elbow.
STEP_SIZES = [0.0, 0.25, 0.50, 0.75, 1.00, 1.15, 1.25, 1.325, 1.40, 1.475, 1.55, 1.65, 1.80, 2.00]
# Added at 4x pacing, where the cliff sits near 1.70 px and the base grid stops too early.
STEP_EXTRA_4X = [2.25, 2.50]

# 07-31 campaign's release grid, kept so temperatures are directly comparable.
# 0 = byte-clean held-in-traps; 0.5 = the SAME-bytes restore-only reference (see docstring).
RELEASE_TIMES_US = [0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Frame 0 = the loading array; frame 1 DECLARES the same (unshifted) array.

    One-way motion leaves the array displaced by a per-cell ``dx``, and imagePatternsJson is
    resolved once per scan, so no single declared grid can be right for every cell. Declaring
    the unshifted array is the honest choice: it makes frame 1 correct exactly at step_size=0
    and obviously-wrong elsewhere, rather than pretending. Survival is recovered offline.
    """
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def steps_for_period(period_ms, extra=True):
    """Base grid, plus the two far-side points when running at >= 4x the base period."""
    s = list(STEP_SIZES)
    if extra and period_ms >= 3.9 * BASE_PERIOD_MS:
        s += list(STEP_EXTRA_4X)
    return sorted(set(s))


def build(period_ms, steps=None, times_us=None, nsteps=NSTEPS, cool=False):
    """2-D [step_size x ReleaseRecapture.Time] at fixed nsteps, one-way. -> (seq_name, g)."""
    _bootstrap()
    from scan_group import ScanGroup

    steps = [float(s) for s in (steps if steps else steps_for_period(period_ms))]
    times = [float(t) * 1e-6 for t in (times_us or RELEASE_TIMES_US)]

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
    rk.step_period_ms = float(period_ms)
    rk.nsteps = int(nsteps)

    # ONE-WAY: rest on the fully-shifted frame. This is what makes it a transport measurement.
    rk.extras.return_trip = False
    rk.extras.hold_ms = HOLD_MS                 # no turnaround one-way; explicit (extras sticky)

    # dim 1: the RADIAL step magnitude, knm px. step_x is folded to the xyz 3-vector [s,0,0] by
    # rearrange_callbacks._fold_step_xyz -- a list-valued swept axis breaks the lab-side grid.
    rk.extras.step_x.scan(1, list(steps))
    rk.extras.step_y = 0.0
    rk.extras.step_z = 0.0

    # LATERAL mode, set EXPLICITLY: the server extras are sticky across scans and a prior axial
    # run leaves depth=True / true_defocus=True behind.
    rk.extras.depth = False
    rk.extras.true_defocus = False
    rk.extras.depth_piston_corr = 0.0
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (production)
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # dim 2: the release time. See the docstring for why 0 AND 0.5 us are both present.
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0

    # POST-MOTION cooling OFF so the release probes the post-motion temperature.
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(POST_COOL_HOLD_MS) * 1e-3

    rp.NumPerGroup = 100000                     # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return "RearrangeRnRHeatCommSeq", g, steps, times


def _desc(period_ms, steps, times, nsteps):
    mult = period_ms / BASE_PERIOD_MS
    return (
        "*** SURVIVAL NEEDS OFFLINE RE-DETECTION (temperatures do NOT) *** "
        "ONE-WAY +x grating transport THERMOMETRY (pingponggrating return_trip=False, seq "
        "RearrangeRnRHeatCommSeq): 2-D extras.step_x %s knm px x ReleaseRecapture.Time %s us at "
        "FIXED nsteps=%d, step_period_ms=%.3f (= %.0fx the 0.696 ms SLM write floor). "
        "One-way => %d steps per shot (NOT 2n) and the array RESTS displaced by dx = step*%d, so "
        "img2 sees a TRANSLATED array: the declared frame-1 grid is the UNSHIFTED one and is "
        "correct only at step=0. Survival must be recomputed offline per cell on the grid shifted "
        "by that cell's own dx (pyctrl/tools/ppg_transport_analyze.py). TEMPERATURES ARE "
        "UNAFFECTED -- each release curve is normalized by its own t=0 point in the same cell, so "
        "a constant per-cell detection efficiency divides out. "
        "PURPOSE: the 08-05 one-way scans put the 99%%-per-step SURVIVAL cliff at ~1.32 knm px "
        "@ 0.696 ms, but the 07-31 round-trip campaign showed radial heating switches on at "
        "~0.9 px = 0.70x that cliff -- so a stroke chosen from survival alone can leave the array "
        "hot. This measures T(step) directly across the whole grid at the stroke one would "
        "actually pick. "
        "RELEASE GRID starts 0 AND 0.5 us on purpose: RearrangeRnRStep returns early at exactly "
        "t=0 (no AmpSLM toggle, no TTLSampleAndHold re-assert, no 3 us AOM settle), so the t=0 "
        "shot does NOT pay the trap restore every t>0 shot pays (~1%% survival, 07-31 caveat #4); "
        "t=0.5 us takes the full branch and is the restore-only reference. "
        "hold_ms=0 (no turnaround exists one-way). Post-motion cooling OFF (PostRearrCool amps 0 "
        "+ %.1f ms dark hold) so the release probes the POST-MOTION temperature. 12 reps = 1.5x "
        "the 07-31 campaign's 8, so the survivor distribution can be checked for bias rather than "
        "assumed thermal. Array 33x33_feedback11, z4 = loading_defocus = -4, ifEnhanced=True."
        % (",".join("%g" % s for s in steps),
           ",".join("%g" % (t * 1e6) for t in times),
           nsteps, period_ms, mult, nsteps, nsteps, POST_COOL_HOLD_MS)
    )


def submit(period_ms, reps=12, url=None, **kw):
    seq_name, g, steps, times = build(period_ms, **kw)
    from yb_start_scan import ybStartScan
    label = "PPGOneWayHeat_p%gms" % period_ms
    did = ybStartScan(seq_name, g, url=url, label=label,
                      description=_desc(period_ms, steps, times, kw.get("nsteps", NSTEPS)),
                      **({"rep": reps} if reps else {}))
    print("submitted %s -> id %s (%d pts = %d steps x %d times, %d reps = %d shots)"
          % (label, did, g.nseq(), len(steps), len(times), reps, g.nseq() * (reps or 1)))
    return did


def main():
    ap = argparse.ArgumentParser(description="One-way +x transport thermometry (R&R vs step).")
    ap.add_argument("--period", type=float, default=BASE_PERIOD_MS)
    ap.add_argument("--steps", default=None, help="comma-separated knm px/step")
    ap.add_argument("--times", default=None, help="comma-separated release times, us")
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--cool", action="store_true", help="restore production post-motion cooling")
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    kw = dict(nsteps=args.nsteps, cool=args.cool)
    if args.steps:
        kw["steps"] = [float(x) for x in args.steps.split(",") if x.strip()]
    if args.times:
        kw["times_us"] = [float(x) for x in args.times.split(",") if x.strip()]

    if args.dry_run:
        seq, g, steps, times = build(args.period, **kw)
        print("seq=%s  nseq=%d  period=%g ms (%.1fx base)  nsteps=%d  ONE-WAY"
              % (seq, g.nseq(), args.period, args.period / BASE_PERIOD_MS, args.nsteps))
        print("steps (%d): %s" % (len(steps), ", ".join("%g" % s for s in steps)))
        print("times (%d): %s us" % (len(times), ", ".join("%g" % (t * 1e6) for t in times)))
        print("shots at %d reps: %d  (~%.2f h at 2.5 s/shot)"
              % (args.reps, g.nseq() * args.reps, g.nseq() * args.reps * 2.5 / 3600))
        print("max dx = %g knm px (= %.1f cam px, %.2f pitch)"
              % (max(steps) * args.nsteps, max(steps) * args.nsteps * 2.268,
                 max(steps) * args.nsteps / 24.48))
        return
    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    submit(args.period, reps=args.reps, url=args.url, **kw)


if __name__ == "__main__":
    main()
