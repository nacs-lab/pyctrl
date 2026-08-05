"""PPGSphericalNullTrueDefocusSweep.py -- the Z12 spherical null, redone under TRUE DEFOCUS.

The 2026-07-28 parabolic-Z4 spherical sweep (job 252, sid 20260728185338) found the two axial
directions peaking at OPPOSITE-SIGN corrective spherical: +4 rad at C12 = +2.79, -4 rad at
C12 = -3.49 -- a 6.3 rad SPLIT, with the asymmetry A = S(+) - S(-) never nulling at either optimum
(A = +0.44, 38 sigma, at the symmetric optimum C12 = +0.06). A single spherical value could not
satisfy both directions.

WHY: the paraxial ANSI Z4 map is WRONG by a rho^4 (primary-spherical) term whose coefficient is
PROPORTIONAL TO THE COMMANDED STEP, sign included -- measured on the live server as
**-0.398 rad per um of commanded step** (beam-weighted rho^4 fit of ``Z4_map*kappa - exact_map``).
Because it flips sign with the step direction, it ADDS to the objective's own STATIC spherical ``A``
on one leg and SUBTRACTS on the other:

    +z:  A - 0.398*s        -z:  A + 0.398*s

With ``A`` negative (as job 252's positive fitted C12 on the +z leg implies) the +z leg got an
ACCIDENTAL CANCELLATION -- which is exactly why +z looked better than -z under parabolic Z4, and why
the two legs demanded opposite-sign corrections.

``true_defocus=True`` removes that direction-odd term entirely. Job 321 (step x nsteps, true
defocus) then shows the asymmetry REVERSED -- -z now better than +z, by +0.31 in survival at
|s| = 2.4 um / nsteps 90 -- i.e. the parabolic map had been MASKING the rig's own preference.

THIS SCAN tests the resulting prediction directly:

    dim 1: extras.step_size     = {-2.4, 0, +2.4} um axial (true_defocus) -- the two directions plus
                                  the no-motion control. 2.4 um == 3.0 rad of Z4-PV, matching job
                                  252's |s| = 3.2 um / 4 rad point closely enough to compare while
                                  staying inside the alive band found in job 321.
    dim 2: extras.distortion_z12 = C12 from -5 to +5 in 0.5 steps, 21 values (PV rad, ANSI index 12,
                                  folded to distortion_zernike[12] per shot by
                                  rearrange_callbacks._fold_distortion_zernike)

    => 3 x 21 = 63 points.

PREDICTIONS (stated up front so the result can falsify them):
  * the 6.3 rad branch SPLIT collapses toward zero -- both legs should now peak at a COMMON C12,
    because the direction-odd parabolic residual that forced opposite-sign optima is gone;
  * that common optimum measures the objective's OWN static spherical ``A`` (the quantity job 252
    could not isolate), so a SINGLE C12 should lift BOTH directions instead of trading them;
  * the residual asymmetry at the common optimum is whatever is NOT spherical (beam decentre /
    coma / the fact that the atoms sit at loading_defocus = -4, so +-s trace different regions of
    the objective's aberration field).
  If instead the split SURVIVES at ~6 rad, the residual is not the parabolic approximation and the
  ``true_defocus`` story above is wrong.

Config matches the settled true-defocus operating point (job 321/323): 0.696 ms pacing with a 5 ms
turnaround hold, ``depth_piston_corr = 0.617`` rad/um (the measured true-defocus null, job 320 --
under ``true_defocus`` the server takes this DIRECTLY in rad of uniform phase per MICRON, no kappa
rescaling), ``no_depth_piston=True``, ``piston=0``, ``depth_fill_frac`` CLEARED (an explicit positive
value would override the corr via the deprecated legacy path).

``nsteps=40`` at 2.4 um = 192 um of total path -- inside the memoryless regime (job 321 put the
cumulative onset for 2.4 um at nsteps ~ 60-70, ~290-340 um), so this measures PER-STEP spherical
physics and not accumulated damage. ``reverse_zernike=False``: a corrective spherical is STATIC and
must keep its sign on the return leg (which retraces the same displacements).

NOTE ``distortion_zernike`` bypasses the uint8 precompute cache and forces the deduped float build
(nsteps+1 = 41 unique frames, ~4 MB each), so precompute/precompute_host stay on but the hot loop is
float-built; this raises the period floor slightly. Watch for pacing overrun in the diag if 0.696 ms
is not being realised.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGSphericalNullTrueDefocusSweep.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGSphericalNullTrueDefocusSweep.py --force
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

NSTEPS = 40                   # 2*40 = 80 steps; at 2.4 um -> 192 um path (memoryless regime)
PERIOD_MS = 0.696
HOLD_MS = 5.0                 # fixed dwell at the turnaround frame
PISTON_CORR = 0.617           # rad of uniform phase per UM (true_defocus numeraire; job 320)

# dim 1: the two axial directions + the no-motion control, MICRONS (true_defocus).
STEP_UM = 2.4
STEP_SIZES = [-STEP_UM, 0.0, STEP_UM]
# dim 2: corrective primary spherical C12 (PV rad), -5..+5 every 0.5 -> 21 values. Same RANGE as
# job 252 (which needed +2.79 / -3.49) but half the density, since the question is where the two
# branch optima sit relative to each other, not their shape to 0.25 rad.
C12_VALUES = [round(-5.0 + 0.5 * i, 2) for i in range(21)]

RUN_DESC = (
    "Z12 spherical null REDONE UNDER TRUE DEFOCUS: step_size {-2.4, 0, +2.4} um axial "
    "(true_defocus=True) x distortion_z12 -5..+5 every 0.5 (21 values) = 63 points. "
    "Job 252 (parabolic Z4) found the two legs peaking at OPPOSITE-SIGN spherical (+4 rad at "
    "C12=+2.79, -4 rad at C12=-3.49; 6.3 rad split) with the asymmetry never nulling (A=+0.44, "
    "38 sigma, at the symmetric optimum). CAUSE: the paraxial Z4 map carries a rho^4 error whose "
    "coefficient is proportional to the commanded step INCLUDING SIGN (measured -0.398 rad per um "
    "of step on the live server), so it adds to the objective's static spherical A on one leg and "
    "subtracts on the other -- an accidental cancellation that made +z look better and forced "
    "opposite-sign corrections. true_defocus removes that direction-odd term, and job 321 duly "
    "shows the asymmetry REVERSED (-z better by +0.31 survival at 2.4 um / nsteps 90). "
    "PREDICTION: the 6.3 rad branch split collapses, both legs peak at a COMMON C12 that measures "
    "the objective's OWN static spherical, and one C12 lifts both directions; the residual "
    "asymmetry there is the NON-spherical part (decentre/coma, plus the atoms sitting at "
    "loading_defocus=-4 so +-s trace different aberration field). If the split SURVIVES at ~6 rad, "
    "the true_defocus explanation is wrong. Config = the settled true-defocus point: 0.696 ms + "
    "hold_ms=5, depth_piston_corr=0.617 rad/um (job 320 null), no_depth_piston=True, piston=0, "
    "depth_fill_frac cleared, nsteps=40 (192 um path at 2.4 um -- inside the memoryless regime, "
    "onset was ~290-340 um), reverse_zernike=False (static correction keeps its sign on the return "
    "leg). distortion_zernike bypasses the uint8 cache -> deduped float build. "
    "Array 33x33_feedback11, z4 = loading_defocus = -4."
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (the ping-pong returns to the source sites). Name =
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(step_um=STEP_UM, nsteps=NSTEPS, period_ms=PERIOD_MS, hold_ms=HOLD_MS,
          piston_corr=PISTON_CORR, c12_values=None, true_defocus=True):
    """Build (do NOT submit) the 2-D [step_size x C12] ScanGroup. Returns ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

    steps = [-float(step_um), 0.0, float(step_um)]
    c12 = [float(c) for c in (c12_values if c12_values else C12_VALUES)]

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

    # ---- rearrange_kwargs: pingponggrating, TRUE-DEFOCUS depth mode --------------------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(period_ms)

    # dim 1: signed axial step in UM (scalar + depth=True -> the pure axial ping-pong).
    rk.extras.step_size.scan(1, list(steps))
    rk.extras.depth = True
    # dim 2: corrective primary spherical, scalar -> distortion_zernike[12] per shot (a
    # list-valued swept axis would break the lab-side N-D scan grid).
    rk.extras.distortion_z12.scan(2, list(c12))
    # A corrective spherical is STATIC: same sign on the outward and return legs (the return leg
    # retraces the same displacements, it does not go to -z). Also halves the unique-frame count.
    rk.extras.reverse_zernike = False

    # TRUE DEFOCUS -- the whole point. EXPLICIT (defaults False, and server extras are sticky).
    rk.extras.true_defocus = True if true_defocus else False

    # Piston nulled at the MEASURED true-defocus value so the spherical is the only residual left.
    # Under true_defocus the server reads this directly as rad of uniform phase per UM.
    rk.extras.depth_piston_corr = float(piston_corr)
    # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (legacy
    # beam-model path) and is sticky across scans -- job 252 SET it to 0.65, so it must be cleared.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0
    # depth_fill_center: left UNSET -> the server's measured beam centroid (-0.0684, +0.0117).

    # Fixed dwell at the turnaround frame so every step is guaranteed settled.
    rk.extras.hold_ms = float(hold_ms)

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1300                       # -> ceil(1300/63) = 21 passes = 1323 shots
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def submit(url=None, reps=None, **kw):
    seq_name, g = build(**kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="PPGSphericalNullTrueDefocusSweep",
                      description=RUN_DESC, **opts)
    print("submitted PPGSphericalNullTrueDefocusSweep -> descriptor id %s (%d pts; step=+-%.1f um, "
          "nsteps=%d, period=%.3f ms, hold=%.1f, corr=%.3f rad/um, true_defocus=True; url=%s)"
          % (did, g.nseq(), kw.get("step_um", STEP_UM), kw.get("nsteps", NSTEPS),
             kw.get("period_ms", PERIOD_MS), kw.get("hold_ms", HOLD_MS),
             kw.get("piston_corr", PISTON_CORR), url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Z12 spherical null under true_defocus (pingponggrating).")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--step-um", type=float, default=STEP_UM,
                    help="axial step magnitude in um (default %.1f)" % STEP_UM)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR,
                    help="depth_piston_corr in rad/um (default %.3f)" % PISTON_CORR)
    ap.add_argument("--c12", default=None,
                    help="comma-separated C12 values (default -5..+5 every 0.5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    _c = ([float(x) for x in args.c12.split(",") if x.strip()] if args.c12 else None)
    kw = dict(step_um=args.step_um, nsteps=args.nsteps, period_ms=args.period,
              hold_ms=args.hold_ms, piston_corr=args.corr, c12_values=_c)
    if args.dry_run:
        _seq, _g = build(**kw)
        print("seq=%s  nseq=%d  nsteps=%d (%d frames)  period=%g ms  hold=%g  corr=%g rad/um"
              % (_seq, _g.nseq(), args.nsteps, 2 * args.nsteps + 1, args.period,
                 args.hold_ms, args.corr))
        print("step_sizes [um] = %s" % [-args.step_um, 0.0, args.step_um])
        print("  total path     = %.0f um (2*nsteps*|step|)" % (2 * args.nsteps * args.step_um))
        print("C12 = %s" % (_c if _c else C12_VALUES))
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, **kw)
