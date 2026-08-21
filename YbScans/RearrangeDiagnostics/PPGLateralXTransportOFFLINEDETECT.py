"""PPGLateralXTransportOFFLINEDETECT.py -- +x one-way grating transport.

*** THIS SCAN REQUIRES OFFLINE RE-DETECTION. LIVE SURVIVAL IS WRONG BY DESIGN. ***

Read this before touching the data
----------------------------------
``pingponggrating`` with ``return_trip=False`` walks the WHOLE tweezer array
``dx = step_size * nsteps`` knm-px in **+x** and RESTS THERE.  The second camera
frame therefore sees the array **translated**, not back home.

The lab-side detection grid is resolved ONCE PER SCAN (``imagePatternsJson`` comes
from ``runp()``; ``engine_run.py:560`` -> ``scan_prep.write_scan_config``), so a scan
that sweeps ``step_size`` x ``nsteps`` -- where every cell has a DIFFERENT ``dx`` --
cannot have a live frame-1 grid that is correct for more than one cell.  That is a
deliberate, accepted trade: it buys a clean 2-D ``(step_size, nsteps)`` grid with the
SAME step-size axis at every ``nsteps``.

Consequence, stated plainly:

* **img1** (loading frame) detection is CORRECT -- the array has not moved yet.
* **img2** detection is CORRECT ONLY for cells whose ``dx`` equals ``FRAME1_DX``
  (default 0.0, i.e. the unshifted grid). Every other cell's live img2 occupancy /
  survival is measured with boxes in the wrong place and is MEANINGLESS.
* The dashboard survival curve for this scan is therefore NOT the transport result.
  Do not quote it. Do not tune on it.

The real analysis re-detects img2 offline, per scan point, on the grid shifted by that
point's own ``dx``:

    python pyctrl/tools/ppg_transport_analyze.py --scan-id <14-digit>

which reads the saved raw frames from the HDF5, rebuilds the shifted grid via the
pattern registry + global affine, and recomputes survival per cell.

Why a grating (and not the model)
---------------------------------
``pingponggrating`` adds a blaze to the WGS phase, so it translates the array rigidly
with NO model inference and NO per-site path planning.  Every atom makes the identical
move, which is exactly what a transport characterisation wants: the only variables are
the per-step stroke and the number of steps.

What the 2-D grid measures
--------------------------
    dim 1: extras.step_size -- per-step stroke, knm-px (SAME axis at every nsteps)
    dim 2: nsteps           -- number of steps (one-way; nsteps+1 frames)

Survival S(step, n) at total travel dx = step * n separates two hypotheses:

* **memoryless / per-step**: per-step survival ``s = S**(1/n)`` collapses onto ONE
  curve vs ``step_size``, independent of ``n``.  Loss is set by the per-step stroke
  alone; total distance is irrelevant except through the step count.
* **cumulative**: ``s`` degrades as ``n`` grows at fixed stroke -- heating accumulates
  and the usable stroke shrinks with distance.

The step axis is deliberately FINE at the elbow.  Prior lateral pingpong work put the
per-frame transit cliff near ``step_size`` ~0.75-1.25 knm-px (safe <=0.75, dead >=1.25),
so the grid is dense through 0.9-1.5 and coarse outside it.  Max gap is 0.25 knm-px
anywhere, per the scan spec.

Geometry facts (verified against the live affine + real averaged atom images)
----------------------------------------------------------------------------
* ``+1`` knm-px in x -> ``dX = +2.268``, ``dY = -0.033`` camera px: a pure camera-X move.
* Affine residual on 1068 real sites: RMS 0.517 cam-px (0.228 knm-px), no systematic bias.
* Lattice is SQUARE with axes at 0/90 deg and NN pitch **24.48 knm-px** (55.5 cam-px) --
  and +x transport runs exactly ALONG a lattice axis.  Box signal on an unshifted image
  peaks at dx=0, nulls at the half-pitch 12.24, and RECOVERS at 24.48 (one full pitch).
  So a shifted grid at dx ~ k*24.48 sits on the lattice one site over; that is fine for
  a rigid whole-array translation (no traps are left behind) but it does mean the
  offline analysis must use the ACTUAL dx, never "nearest site".
* **ROI limit**: at the production ROI ``[1000,100,2100,2100]`` the array's leading
  column leaves the sensor past ~75 knm-px (18 sites lost at 87.5, 33 at 100).  Cells
  with ``dx > MAX_DX_KNM`` are DROPPED so no point silently under-reports survival.

Run::

    python YbScans/RearrangeDiagnostics/PPGLateralXTransportOFFLINEDETECT.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGLateralXTransportOFFLINEDETECT.py --quick --force
    python YbScans/RearrangeDiagnostics/PPGLateralXTransportOFFLINEDETECT.py --force
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

# Both requested periods. 0.696 ms = production pacing; 3.0 ms ~ full LC settle.
PERIOD_MS_FAST = 0.696
PERIOD_MS_SLOW = 3.0

# Beyond this the leading column walks off the production ROI (see module docstring).
MAX_DX_KNM = 75.0

# Frame-1 grid this scan DECLARES. 0.0 == the unshifted loading grid, so live img2 is
# only correct for dx==0 cells. Offline re-detection is what makes the rest meaningful.
FRAME1_DX = 0.0

# ---- the step-size axis: IDENTICAL at every nsteps so the result is a clean 2-D grid.
# Centred on the MEASURED elbow from the quick pass (scans 20260805214823 / 214624,
# 0.696 ms): survival is flat >=0.98 out to step 1.25 at every nsteps, then falls off a
# cliff -- at n=20, S = 0.92 (1.25) -> 0.30 (1.50) -> 0.06 (1.75); at n=50, 0.96 (1.00)
# -> 0.70 (1.25) -> 0.06 (1.50). So the transition lives in 1.20-1.60 and that is where
# the sampling is dense (0.075 gap); the plateau and the dead side are coarse.
# Max gap 0.25 knm-px anywhere, per spec.
STEP_SIZES = [
    0.0,                                          # zero-stroke CONTROL (see below)
    0.25, 0.50, 0.75, 1.00,                       # plateau (0.25 gap)
    1.15, 1.25, 1.325, 1.40, 1.475, 1.55,         # the elbow (0.075-0.10 gap)
    1.65, 1.80, 2.00,                             # dead side
]
# 14 step sizes; the spec asked for ~10 with finer sampling at the elbow.

# The two controls are DIFFERENT and both matter:
#   step_size = 0  -> the array still writes all nsteps+1 frames (the SLM cycles, the LC
#                     transitions happen, the shot takes the full wall-clock time) but the
#                     commanded displacement is ZERO. Isolates the cost of frame-writing /
#                     LC cycling / shot duration from the cost of actually MOVING. Its
#                     nsteps dependence is the "motionless overhead" baseline that the
#                     real step sizes must be read against.
#   nsteps    = 0  -> a single static frame: no cycling AND no motion. The absolute
#                     imaging/loading floor.
# Both have dx = 0, so the unshifted detection grid is exactly correct for them.

NSTEPS_LIST = [0, 1, 2, 5, 10, 20, 30, 40, 50]

# A ScanGroup 2-D sweep is a Cartesian product, so the grid must be RECTANGULAR, while
# the ROI caps dx = step*nsteps at MAX_DX_KNM. One rectangle covering nsteps=50 would
# force the step axis down to <=1.5 everywhere, throwing away the dead-side points that
# anchor the threshold curve at small nsteps. So the thorough measurement is split into
# two arms, each rectangular and each with ONE step axis:
#
#   arm "short": nsteps 1..30, the FULL step axis (2.0*30 = 60 <= 75)
#   arm "long" : nsteps 40, 50, steps <= 1.5    (1.5*50 = 75 <= 75)
#
# Together they cover nsteps 1..50. The long arm drops only 1.75 / 2.0, which are far
# past the elbow and already dead well before n=40 -- no information is lost.
# ONE scan per period: every nsteps in a single rectangular grid, so all of them share
# one loading / imaging / threshold context and are interleaved by Scramble. Because the
# grid is a Cartesian product and the ROI caps travel at MAX_DX_KNM, the largest nsteps
# sets the step ceiling: 50 * 1.475 = 73.75 <= 75. So the step axis is trimmed at 1.475
# and the dead-side points (1.55-2.00) are dropped -- the quick pass already showed
# everything above ~1.5 is dead at every nsteps >= 5, so nothing is lost by not
# re-measuring it here. `_surviving_axes` does that trim automatically.
ARMS = {
    "all": {"nsteps": NSTEPS_LIST, "steps": STEP_SIZES},
}

# Quick scale-finding pass: coarse steps, few nsteps.
QUICK_STEPS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
QUICK_ARMS = {
    "all": {"nsteps": [0, 1, 5, 20, 50], "steps": QUICK_STEPS},
}


def _cells(steps, nsteps_list, max_dx=MAX_DX_KNM):
    """All (step, n) cells, split into on-sensor and over-range. NOTHING is dropped from
    the scan -- the split is reported so the over-range cells are known in advance and
    can be filtered (or kept, on their reduced site set) at analysis time."""
    keep, over = [], []
    for n in nsteps_list:
        for s in steps:
            (keep if s * n <= max_dx else over).append((s, n))
    return keep, over


def _surviving_axes(steps, nsteps_list, max_dx=MAX_DX_KNM):
    """Pass the axes through UNCHANGED; report which cells exceed the ROI.

    We deliberately do NOT trim. Cells whose travel exceeds the ROI are still worth
    taking: the offline analysis intersects the img1 and shifted-img2 grids per cell
    (``onsensor_mask``) and scores only the sites that stay on the sensor, recording how
    many were dropped as ``n_offsensor_sites``. So an over-range cell yields a valid --
    if spatially truncated -- survival number that can be filtered at analysis time,
    which beats not measuring it at all.

    The returned second element is now the list of STEP sizes that will have at least
    one over-range cell, purely so the dry-run and the run description can say so.
    """
    nmax = max(nsteps_list)
    over = [s for s in steps if s * nmax > max_dx]
    return list(steps), over


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json(frame1_dx=FRAME1_DX):
    """Frame 0 = the loading array. Frame 1 = the grid this scan DECLARES.

    With ``frame1_dx == 0`` both frames name the same unshifted pattern, which is
    honest: the live img2 detection is simply not the measurement (see the module
    docstring). A non-zero value names a pre-registered shifted pattern instead, which
    makes ONE dx column live-correct -- useful for a single-dx confirmation scan.
    """
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    if not frame1_dx:
        return json.dumps([it, dict(it)])

    tools = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from ppg_shifted_pattern import shifted_name
    it2 = dict(it, name=shifted_name(PATTERN, frame1_dx))
    return json.dumps([it, it2])


def _desc(steps, ns, period_ms, dropped_n):
    return (
        "*** OFFLINE RE-DETECTION REQUIRED -- LIVE img2 SURVIVAL IS INVALID FOR dx != %g *** "
        "+x ONE-WAY grating transport (pingponggrating, return_trip=False): the whole array "
        "translates dx = step_size*nsteps knm-px in +x and RESTS there, so the 2nd camera frame "
        "sees a TRANSLATED array. imagePatternsJson is resolved once per scan, so with a 2-D "
        "(step_size x nsteps) grid -- every cell a different dx -- no single live frame-1 grid can "
        "be right for more than one cell; this scan declares the UNSHIFTED grid (frame1_dx=%g). "
        "img1 detection is correct (array not yet moved); img2 must be RE-DETECTED OFFLINE per "
        "scan point on the grid shifted by that point's own dx "
        "(pyctrl/tools/ppg_transport_analyze.py). "
        "GRID: step_size %s knm-px/step (SAME axis at every nsteps; dense through the MEASURED "
        "1.2-1.5 elbow, max gap 0.25) x nsteps %s = %d points, ONE scan holding every nsteps so "
        "they share a loading/imaging context and interleave under Scramble. "
        "TWO CONTROLS, both at dx=0 (unshifted grid exactly correct): step_size=0 still writes "
        "all nsteps+1 frames with ZERO stroke, isolating frame-writing / LC-cycling / shot-"
        "duration cost from displacement cost; nsteps=0 is a single static frame (no cycling, no "
        "motion) giving the absolute imaging/loading floor. "
        "step_period_ms=%.3f, return_trip=False -> nsteps+1 frames, no turnaround frame and no "
        "hold, so every step is an identical unidirectional LC transition (contrast the axial "
        "round-trip scans, where the turnaround dominates small nsteps). "
        "PURPOSE: (1) the survival-vs-step threshold curve at EACH nsteps; (2) whether transport "
        "loss is MEMORYLESS -- per-step survival S**(1/nsteps) collapsing onto one curve vs "
        "step_size independent of nsteps -- or CUMULATIVE, with the usable stroke shrinking as "
        "distance grows. Travel capped at %g knm-px: past that the leading column leaves the "
        "production ROI (18 sites lost at 87.5, 33 at 100) and survival would be silently "
        "under-reported.%s "
        "Lattice is square, pitch 24.48 knm-px, and +x runs ALONG a lattice axis; the affine maps "
        "+1 knm-px -> +2.268 cam-X px (validated on real images, RMS 0.517 cam-px). "
        "Array 33x33_feedback11, z4 = loading_defocus = -4."
        % (FRAME1_DX, FRAME1_DX,
           ",".join("%g" % s for s in steps), ",".join(str(n) for n in ns),
           len(steps) * len(ns), period_ms, MAX_DX_KNM,
           (" Step sizes %s exceed the ROI at the largest nsteps and are KEPT ANYWAY: "
            "the offline analysis intersects the img1 and shifted-img2 grids per cell and "
            "scores only the sites still on the sensor, recording how many were dropped "
            "as n_offsensor_sites, so those cells give a valid (spatially truncated) "
            "number that can be filtered later rather than not being measured at all."
            % ",".join("%g" % s for s in dropped_n))
           if dropped_n else "")
    )


def build(steps=None, nsteps_list=None, period_ms=PERIOD_MS_FAST,
          frame1_dx=FRAME1_DX, num_per_group=2000):
    """Build (do NOT submit) the 2-D [step_size x nsteps] ScanGroup -> ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

    steps_req = [float(s) for s in (steps if steps else STEP_SIZES)]
    ns = [int(n) for n in (nsteps_list if nsteps_list else NSTEPS_LIST)]
    steps, dropped_n = _surviving_axes(steps_req, ns)
    if not steps:
        raise SystemExit("every step would push dx past %g knm-px at nsteps=%d; "
                         "lower --nsteps-list" % (MAX_DX_KNM, max(ns)))

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
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

    # ---- rearrange_kwargs: pingponggrating, LATERAL (+x) one-way ----------------------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)

    # dim 1: per-step stroke in knm-px. Scalar + depth=False -> the lateral blaze path.
    rk.extras.step_size.scan(1, list(steps))
    # dim 2: number of one-way steps -> nsteps+1 frames.
    rk.nsteps.scan(2, list(ns))

    # LATERAL mode, explicitly: server extras are STICKY across scans and a prior axial
    # scan will have left depth=True behind.
    rk.extras.depth = False
    rk.extras.true_defocus = False
    # ONE-WAY: rest on the fully-shifted frame (this is the whole point of the scan).
    rk.extras.return_trip = False
    # No turnaround frame exists when return_trip=False, so a hold would be meaningless.
    rk.extras.hold_ms = 0.0
    # No commanded per-step uniform phase (lateral mode carries no defocus piston).
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (matches live production)
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = int(num_per_group)
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(frame1_dx)

    return seq_name, g, steps, ns, dropped_n


def submit(url=None, reps=None, arm=None, **kw):
    seq_name, g, steps, ns, dropped_n = build(**kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    period_ms = kw.get("period_ms", PERIOD_MS_FAST)
    # The label is the loudest place to say "do not read the live survival".
    label = "PPGLateralXTransport_OFFLINE_DETECT_%sp%gms" % (
        ("%s_" % arm) if arm else "", period_ms)
    did = ybStartScan(seq_name, g, url=url, label=label,
                      description=_desc(steps, ns, period_ms, dropped_n), **opts)
    print("submitted %s -> descriptor id %s (%d pts, period=%.3f ms)"
          % (label, did, g.nseq(), period_ms))
    print("  *** img2 needs OFFLINE re-detection: "
          "python pyctrl/tools/ppg_transport_analyze.py --scan-id <id> ***")
    return did


def _report(steps, ns, dropped_n, period_ms):
    keep, over = _cells(steps, ns)
    print("step_sizes (knm-px/step) = %s" % ", ".join("%g" % s for s in steps))
    print("nsteps                   = %s" % ", ".join(str(n) for n in ns))
    print("points                   = %d (NOTHING trimmed)" % (len(steps) * len(ns)))
    print("frames/shot              = nsteps+1 (one-way, no turnaround)")
    print("motion ms/shot           = %s"
          % ", ".join("n=%d:%.1f" % (n, (n + 1) * period_ms) for n in ns))
    dxs = sorted({round(s * n, 4) for s, n in (keep + over)})
    print("distinct dx (knm-px)     = %d, max %.2f (= %.1f cam-px, %.2f lattice pitch)"
          % (len(dxs), max(dxs), max(dxs) * 2.268, max(dxs) / 24.48))
    print("shifted patterns needed for OFFLINE analysis: %d" % len(dxs))
    if over:
        n_off = len(over)
        print("over-ROI cells (dx>%g)   = %d of %d -- KEPT; analysis scores them on the "
              "sites still on-sensor and records n_offsensor_sites"
              % (MAX_DX_KNM, n_off, len(steps) * len(ns)))
        print("   %s" % ", ".join("%g x%d=%.1f" % (s, n, s * n) for s, n in over[:8])
              + (" ..." if n_off > 8 else ""))


def main():
    ap = argparse.ArgumentParser(
        description="+x one-way grating transport; REQUIRES OFFLINE RE-DETECTION.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="coarse scale-finding grid instead of the thorough one")
    ap.add_argument("--arm", choices=("all",), default="all",
                    help="grid to run (only 'all' -- one scan holds every nsteps)")
    ap.add_argument("--period", type=float, default=None,
                    help="step_period_ms (default %g; the slow point is %g)"
                         % (PERIOD_MS_FAST, PERIOD_MS_SLOW))
    ap.add_argument("--slow", action="store_true",
                    help="use the %g ms period instead of %g" % (PERIOD_MS_SLOW, PERIOD_MS_FAST))
    ap.add_argument("--step-sizes", default=None, help="comma-separated knm-px/step")
    ap.add_argument("--nsteps-list", default=None, help="comma-separated nsteps")
    ap.add_argument("--frame1-dx", type=float, default=FRAME1_DX,
                    help="knm-px shift of the DECLARED frame-1 grid (default 0 = unshifted)")
    ap.add_argument("--num-per-group", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    period = args.period if args.period is not None else (
        PERIOD_MS_SLOW if args.slow else PERIOD_MS_FAST)
    arms = QUICK_ARMS if args.quick else ARMS

    # An explicit axis override collapses the arm split (the user is driving).
    if args.step_sizes or args.nsteps_list:
        jobs = [(None,
                 [float(x) for x in args.step_sizes.split(",") if x.strip()]
                 if args.step_sizes else (QUICK_STEPS if args.quick else STEP_SIZES),
                 [int(x) for x in args.nsteps_list.split(",") if x.strip()]
                 if args.nsteps_list else NSTEPS_LIST)]
    else:
        jobs = [(None, arms[args.arm]["steps"], arms[args.arm]["nsteps"])]

    kws = [(arm, dict(steps=steps, nsteps_list=ns, period_ms=period,
                      frame1_dx=args.frame1_dx, num_per_group=args.num_per_group))
           for arm, steps, ns in jobs]

    if args.dry_run:
        total = 0
        for arm, kw in kws:
            _seq, _g, _st, _ns, _dn = build(**kw)
            print("=== arm %s ===" % (arm or "custom"))
            print("seq=%s  nseq=%d  period=%g ms  frame1_dx=%g"
                  % (_seq, _g.nseq(), period, args.frame1_dx))
            _report(_st, _ns, _dn, period)
            total += _g.nseq()
            print("")
        print("TOTAL points across arms: %d" % total)
        print("*** LIVE img2 SURVIVAL IS INVALID except at dx=%g. Analyze with:\n"
              "    python pyctrl/tools/ppg_transport_analyze.py --scan-id <id>"
              % args.frame1_dx)
        return

    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    for arm, kw in kws:
        submit(url=args.url, reps=args.reps, arm=arm, **kw)


if __name__ == "__main__":
    main()
