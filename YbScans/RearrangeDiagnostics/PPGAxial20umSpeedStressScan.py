"""PPGAxial20umSpeedStressScan.py -- how FAST can a 20 um axial crossing be made?

THE QUESTION
  The 2-layer array (``phase/2x11x11_5um_z20um.pt``) puts its two planes 20 um apart, and every
  3-D protocol on it has to cross that gap.  The nominal focus walk spends 50 steps x 0.696 ms
  = 35.5 ms one way.  How much of that is actually needed?  This scan holds the DISTANCE fixed at
  the full 20 um and sweeps only how many frames it is cut into, i.e. the per-step stroke and
  therefore the wall-clock.

WHY IT RUNS ON THE PRODUCTION 33x33 ARRAY, NOT THE 2-LAYER ONE
  The measurement is "does an atom survive being carried 20 um out and back", which needs (a) an
  in-focus image before, (b) the transport, (c) an in-focus image after.  A global
  ``pingponggrating depth`` ROUND TRIP gives exactly that on ANY array: ``return_trip=True`` walks
  out ``nsteps`` frames and back, resting on the WGS initial frame, so img1 and img2 are the same
  sites at the same focus and their ratio is the pure cost of the crossing.  Doing it on
  ``33x33_feedback11`` buys a calibrated affine, tuned per-site thresholds, ~1068 sites of
  statistics per shot, and no dependence on the (still unoptimized) 2-layer detection.  The 2-layer
  array adds nothing here -- its layer structure is irrelevant to a global translation.

THE AXIS
  Total one-way stroke pinned at %(tot).4f rad of PV ANSI Z4 = 20.0 um at the 2026-07-27 recal
  (0.798 um/rad, the same constant the 20 um array was generated with).  ``step_size`` = total /
  nsteps, so every cell travels the SAME distance and only the pacing changes::

      nsteps   rad/step    um/step    one-way (n+1)*0.696 ms   round trip (2n+1)*0.696 ms

  The known per-frame axial cliff is ~2.25 rad of PV Z4 (2026-06-08/09, 3106 shots), so the sweep
  deliberately straddles it: the fast end is expected to fail and the interesting number is WHERE.
  Tonight's warm-WGS result (job 432) says a properly phase-corrected per-step axial kick costs
  nothing out to 4.8 rad/step, which -- if it transfers to the grating -- would put the 20 um
  crossing at ~6 steps = 4.9 ms one way.  Job 428 says the grating is 10-50x worse per step than
  the warm producer at 4.5 rad/step, so the honest prior is somewhere in 8-20 steps.

dim 2 = ``depth_piston_corr`` {0.441, 0.60}: NOT decoration.  Job 428 measured the grating's piston
  null to be DIRECTION-DEPENDENT (c+ ~ 0.463 for +z, c- ~ 0.611 for -z) and worth 0.955 vs 0.891 at
  nsteps=30.  A round trip runs BOTH directions, so no single constant can null both legs and the
  right value is an empirical compromise -- measure it rather than assume it.  (If a genuinely
  per-leg null is wanted, the server's ``leg_defocus`` path exposes ``depth_piston_corr_alt``; that
  is a different scan -- see PPGAxialLegDefocusCalibScan.)

hold_ms = 0 on purpose.  Job 424 showed the n-independent loss channel is ENTIRELY the turnaround
  dwell (f -> 0 at hold 0, saturating by ~3 ms), and a dwell would also swamp the very wall-clock
  this scan is trying to minimize.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxial20umSpeedStressScan.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxial20umSpeedStressScan.py --force
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))

# ---- array / operating point (same as every other RearrangeDiagnostics grating scan) ----
PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0
PERIOD_MS = 0.696

# ---- the fixed distance ----
UM_PER_RAD = 0.798                 # 2026-07-27 recal; the constant the 20 um array was built with
TOTAL_UM = 20.0
TOTAL_RAD = TOTAL_UM / UM_PER_RAD  # 25.0627 rad of PV ANSI Z4, one way

# ---- dim 1: how many frames that distance is cut into ----
NSTEPS_LIST = [50, 30, 18, 12, 8, 5, 3, 2, 1]
# ...plus a STATIC control appended to the same paired axis: the SAME 50-frame pacing with
# step_size = 0, i.e. 101 SLM writes that move nothing.  Without it "n=50 survival" conflates the
# cost of the motion with the cost of writing 101 frames on top of a held array; with it the
# ratio is the motion alone, and it also re-measures the array's non-rearrange survival cap on the
# same shots (the ~95-97% floor) instead of importing it from another run.
STATIC_CONTROL_NSTEPS = 50

# ---- dim 2: the grating's piston null, whose optimum is direction-dependent ----
CORR_LIST = [0.441, 0.60]

HOLD_MS = 0.0
NUM_PER_GROUP = 540                # 9 x 2 = 18 cells -> 30 shots/cell x ~650 loaded sites


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _axis(nsteps_list, static=True):
    """The paired dim-1 axis: (nsteps, step_size) with the distance held constant, plus the
    zero-motion control at the slowest pacing."""
    ns = [int(n) for n in nsteps_list]
    steps = [TOTAL_RAD / float(n) for n in ns]
    if static:
        ns = ns + [int(STATIC_CONTROL_NSTEPS)]
        steps = steps + [0.0]
    return ns, steps


def _table(ns, steps):
    return [(n, s, s * UM_PER_RAD, (n + 1) * PERIOD_MS, (2 * n + 1) * PERIOD_MS)
            for n, s in zip(ns, steps)]


def build(nsteps_list=None, corr_list=None, period_ms=PERIOD_MS, hold_ms=HOLD_MS, static=True):
    """Build (do NOT submit) the paired [nsteps x step_size] x [corr] ScanGroup."""
    _bootstrap()
    from scan_group import ScanGroup

    corrs = [float(c) for c in (corr_list if corr_list else CORR_LIST)]
    ns, steps = _axis(nsteps_list if nsteps_list else NSTEPS_LIST, static=static)

    seq_name = "RearrangeCommSeq"
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

    rk = g().rearrange_kwargs
    rk.extras.n_rounds = 1
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)
    rk.extras.depth = True
    rk.extras.true_defocus = False     # step_size in RADIANS of PV ANSI Z4
    # dim 1, PAIRED: nsteps and its matching step_size, so the DISTANCE is constant across the axis
    # and the only thing that changes is the pacing.  Pairing (rather than a 2-D grid) is what makes
    # this a speed scan instead of a step-size scan.
    rk.nsteps.scan(1, list(ns))
    rk.extras.step_size.scan(1, list(steps))
    # dim 2: the grating piston null (direction-dependent -- see the module docstring)
    rk.extras.depth_piston_corr.scan(2, list(corrs))
    rk.extras.no_depth_piston = True
    rk.extras.depth_fill_frac = None   # sticky legacy override -- must be cleared explicitly
    rk.extras.return_trip = True       # OUT AND BACK: img1/img2 are the same sites at the same focus
    rk.extras.hold_ms = float(hold_ms)
    rk.extras.piston = 0.0
    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return seq_name, g, ns, steps, corrs


def _desc(ns, steps, corrs, period_ms, hold_ms):
    rows = _table(ns, steps)
    tab = "; ".join("n=%d: %.3f rad/step (%.2f um), one-way %.1f ms, round trip %.1f ms"
                    % r for r in rows)
    return (
        "AXIAL SPEED STRESS TEST -- how fast can 20 um be crossed?  pingponggrating depth ROUND "
        "TRIP on %s at z4 = loading_defocus = %+.1f, ifEnhanced, precompute + precompute_host, "
        "step_period_ms=%.3f, hold_ms=%g, true_defocus=False so step_size is RADIANS of PV ANSI Z4. "
        "The one-way stroke is PINNED at %.4f rad = %.1f um (0.798 um/rad, the 2026-07-27 recal the "
        "20 um two-layer array was generated with) and dim 1 PAIRS nsteps with step_size = "
        "total/nsteps, so every cell travels the SAME distance and only the pacing changes: %s.  "
        "return_trip=True -> 2*nsteps+1 frames resting on the WGS initial frame, so img1 and img2 "
        "are the same sites at the same focus and their ratio is the pure cost of the crossing "
        "(this is why the scan does not need the two-layer array -- a global translation does not "
        "care about layer structure, and 33x33_feedback11 brings a calibrated affine, tuned "
        "thresholds and ~1068 sites/shot).  The last dim-1 cell is a STATIC CONTROL -- the same "
        "50-frame pacing at step_size=0, i.e. 101 SLM writes that move nothing -- so the motion's "
        "cost can be separated from the cost of writing the frames, and the array's non-rearrange "
        "survival cap is re-measured on the same shots instead of imported.  dim 2 = "
        "depth_piston_corr %s: job 428 measured the "
        "grating's piston null to be DIRECTION-DEPENDENT (c+ ~ 0.463, c- ~ 0.611, worth 0.955 vs "
        "0.891 at nsteps=30) and a round trip runs both legs, so the best single constant is an "
        "empirical compromise.  hold_ms=0 because job 424 showed the n-independent loss channel is "
        "entirely the turnaround dwell, and a dwell would swamp the wall-clock being minimized.  "
        "PRIOR: the per-frame axial cliff is ~2.25 rad PV (2026-06-08/09, 3106 shots) -> failure "
        "expected below ~12 steps; but tonight's warm-WGS job 432 found no per-step cost out to "
        "4.8 rad/step, and job 428 put the grating 10-50x worse per step than the warm producer at "
        "4.5 rad, so the honest bracket is 8-20 steps = 6-15 ms one way.  The answer is the "
        "smallest nsteps whose survival still matches the n=50 baseline."
        % (PATTERN, DEFOCUS, period_ms, hold_ms, TOTAL_RAD, TOTAL_UM, tab, corrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--nsteps", default=None, help="comma-separated nsteps list")
    ap.add_argument("--corr", default=None, help="comma-separated depth_piston_corr list")
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold", type=float, default=HOLD_MS)
    ap.add_argument("--no-static", action="store_true",
                    help="drop the step_size=0 zero-motion control cell")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    nl = [int(v) for v in args.nsteps.split(",")] if args.nsteps else None
    cl = [float(v) for v in args.corr.split(",")] if args.corr else None
    seq_name, g, ns, steps, corrs = build(nsteps_list=nl, corr_list=cl,
                                          period_ms=args.period, hold_ms=args.hold,
                                          static=not args.no_static)
    g.runp().NumPerGroup = int(args.num_per_group)

    print("seq=%s nseq=%d  total stroke %.4f rad (%.1f um) one way, corr=%s, period=%g, hold=%g"
          % (seq_name, g.nseq(), TOTAL_RAD, TOTAL_UM, corrs, args.period, args.hold))
    for n, s, um, t1, t2 in _table(ns, steps):
        print("   nsteps %3d  %7.3f rad/step  %6.2f um/step   one-way %6.2f ms   round trip %6.2f ms"
              % (n, s, um, t1 * args.period / PERIOD_MS, t2 * args.period / PERIOD_MS))
    if args.dry_run:
        seen = set()
        for i in range(g.nseq()):
            e = g.getseq(i)["rearrange_kwargs"]
            seen.add((e.get("nsteps"), round(e["extras"].get("step_size"), 6),
                      e["extras"].get("depth_piston_corr")))
        print("  %d distinct (nsteps, step, corr) cells" % len(seen))
        s0 = g.getseq(0)["rearrange_kwargs"]
        print("  cell0: %s" % {k: s0["extras"].get(k) for k in
                               ("step_size", "depth", "true_defocus", "depth_piston_corr",
                                "no_depth_piston", "return_trip", "hold_ms", "precompute")})
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label="PPGAxial20umSpeedStressRearrangeScan",
                      description=_desc(ns, steps, corrs, args.period, args.hold), **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
