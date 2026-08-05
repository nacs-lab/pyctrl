"""PPGAxialAsymmetryProbe.py -- cheap probes of the +z / -z axial per-step LOSS ASYMMETRY.

THE OBSERVATION (job 323, true_defocus, piston nulled at 0.617 rad/um): the per-step loss is
**2-4x larger moving +z than -z**, at every step size from 1.6 to 3.0 um. Under the OLD parabolic
Z4 map the asymmetry ran the OTHER way (job 252: +z better), so switching to the exact axial phase
did not remove it -- it unmasked the rig's own preference.

TWO FACTS THAT NARROW THE SUSPECT LIST (both from job 323):
  * loss is PER-STEP, not excursion-driven. At matched one-way excursion the per-step loss differs
    by up to 18x depending on how it was reached (excursion 96 um: 0.00199/step at s=2.4 um vs
    0.00011/step at s=1.6 um). So a static aberration field sampled at large |z| is NOT the driver
    -- what matters is the SIZE OF EACH JUMP.
  * loss rises very steeply with step size (~0.0002/step at 1.6 um -> ~0.011/step at 3.0 um, ~50x
    for 2x in step), i.e. a threshold-like overlap/recapture process, not a smooth power law.

Given loss is per-step, the surviving mechanisms for the DIRECTION asymmetry are:
  (A) an ASYMMETRIC AXIAL TRAP PROFILE from residual (objective) spherical -- the axial PSF is not
      symmetric fore/aft of focus, so a jump of +s and a jump of -s leave the atom with different
      overlap with the shifted trap. Tested by ``PPGSphericalNullTrueDefocusSweep`` (job 324).
  (B) LC SETTLE RISE/FALL ASYMMETRY -- the SLM's liquid crystal relaxes at different rates for
      increasing vs decreasing phase (established 2026-07-22: negative piston worse, "LC fall
      slower"). A +z ramp and a -z ramp drive the LC in opposite senses, so within a finite step
      period the two directions realise DIFFERENT actual trajectories. **This file's ``period``
      mode tests exactly this.**
  (C) gravity / the weak axial axis -- z is the soft direction (16.5 kHz vs 85.7 kHz radial), so
      moving with vs against gravity changes the escape barrier. Tested by ``depth`` mode (trap
      power), not yet wired here.

MODE ``period`` (the decisive, ~15-point test for B):

    dim 1: extras.step_size      = {-2.4, 0, +2.4} um  -- both directions + no-motion control
    dim 2: rearrange_kwargs.step_period_ms = {0.696, 1.4, 3.0, 6.0, 10.0}

    => 3 x 5 = 15 points.

Operating point chosen so the asymmetry is WELL MEASURED rather than saturated: at s = 2.4 um and
nsteps = 40, job 323 gives S(+) = 0.853 and S(-) = 0.951 -- both comfortably off 0 and 1, so a
change in either direction is visible. nsteps = 40 also keeps the total path at 192 um, INSIDE the
memoryless regime (job 323 put the cumulative-excess onset near 300 um), so this measures per-step
physics uncontaminated by the accumulating term.

READING IT:
  * if the loss ratio (loss+/loss-) FALLS TOWARD 1 as the period grows, the asymmetry is DYNAMIC --
    LC settle, mechanism (B). The fix is pacing (or a settle LUT), not optics.
  * if the ratio is FLAT vs period, the asymmetry is STATIC/optical -- consistent with (A) or (C),
    and the spherical sweep + a trap-depth probe become the next discriminators.
  * the absolute losses should also fall with period either way (more settle time is strictly
    better); it is the RATIO that discriminates.

NOTE the 5 ms turnaround hold is kept (``hold_ms``) so the turnaround itself is always settled and
cannot masquerade as a period effect: only the PER-STEP settle time is being varied.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialAsymmetryProbe.py --mode period --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialAsymmetryProbe.py --mode period --force
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

STEP_UM = 2.4                 # job 323: S(+)=0.853 / S(-)=0.951 at nsteps 40 -- ideal sensitivity
NSTEPS = 40                   # 192 um total path: inside the memoryless regime (onset ~300 um)
HOLD_MS = 5.0
PISTON_CORR = 0.617           # rad/um, the measured true-defocus null (job 320)
PERIODS = [0.696, 1.4, 3.0, 6.0, 10.0]

RUN_DESC_PERIOD = (
    "AXIAL +z/-z ASYMMETRY PROBE, period mode: step_size {-2.4, 0, +2.4} um (true_defocus) x "
    "step_period_ms {0.696, 1.4, 3.0, 6.0, 10.0} = 15 points, nsteps=40, hold_ms=5, "
    "depth_piston_corr=0.617 rad/um. "
    "WHY: job 323 finds per-step loss 2-4x WORSE moving +z than -z at every step size 1.6-3.0 um "
    "(and job 252 under the old parabolic Z4 had the asymmetry the OTHER way, so true_defocus "
    "unmasked the rig's own preference rather than causing it). Job 323 also shows the loss is "
    "PER-STEP not excursion-driven (at matched one-way excursion 96 um the per-step loss differs "
    "18x: 0.00199 at s=2.4 vs 0.00011 at s=1.6), which rules out a static aberration field sampled "
    "at large |z| and points at the size of each individual jump. "
    "THIS TEST discriminates a DYNAMIC cause (LC settle rise/fall asymmetry -- the liquid crystal "
    "relaxes at different rates for increasing vs decreasing phase, established 2026-07-22 'LC fall "
    "slower') from a STATIC/optical one (asymmetric axial trap profile from residual spherical, or "
    "gravity on the soft axial axis). If loss+/loss- falls toward 1 as the period grows -> DYNAMIC, "
    "fix with pacing/settle LUT. If flat vs period -> STATIC, and the spherical sweep (job 324) plus "
    "a trap-depth probe are the next discriminators. Absolute loss should fall with period either "
    "way; the RATIO is the discriminator. The 5 ms turnaround hold is retained so only the PER-STEP "
    "settle time varies. Operating point s=2.4 um / nsteps=40 chosen because both directions sit "
    "well off 0 and 1 (0.853 / 0.951). Array 33x33_feedback11, z4 = loading_defocus = -4."
)


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


def build(mode="period", step_um=STEP_UM, nsteps=NSTEPS, hold_ms=HOLD_MS,
          piston_corr=PISTON_CORR, periods=None):
    """Build (do NOT submit) the probe ScanGroup. Returns ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

    if mode != "period":
        raise ValueError("only mode='period' is implemented (got %r)" % mode)
    pers = [float(p) for p in (periods if periods else PERIODS)]
    # step_um may be a scalar (the original 3-row probe) or a LIST of magnitudes, in which case
    # dim 1 becomes the full signed axis [-max..-min, 0, +min..+max] -- used to re-measure the
    # TRUE axial reach at a settled pacing once the 0.696 ms LC artefact is out of the way.
    if isinstance(step_um, (list, tuple)):
        mags = sorted({float(a) for a in step_um if float(a) > 0})
        steps = [-a for a in reversed(mags)] + [0.0] + list(mags)
    else:
        steps = [-float(step_um), 0.0, float(step_um)]

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

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
    rk.protocol = "pingponggrating"
    rk.nsteps = int(nsteps)

    # dim 1: the two directions + the no-motion control (um, true_defocus).
    rk.extras.step_size.scan(1, list(steps))
    rk.extras.depth = True
    # dim 2: the PER-STEP settle time.
    rk.step_period_ms.scan(2, list(pers))

    rk.extras.true_defocus = True
    rk.extras.depth_piston_corr = float(piston_corr)
    # MANDATORY: an explicit positive depth_fill_frac overrides depth_piston_corr and is sticky.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0
    # Turnaround stays settled in EVERY cell, so only the per-step period is the variable.
    rk.extras.hold_ms = float(hold_ms)

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    rp.NumPerGroup = 450                        # -> 30 passes over 15 pts
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
    did = ybStartScan(seq_name, g, url=url, label="PPGAxialAsymmetryProbe_period",
                      description=RUN_DESC_PERIOD, **opts)
    print("submitted PPGAxialAsymmetryProbe (period) -> descriptor id %s (%d pts; step=+-%.1f um, "
          "nsteps=%d, hold=%.1f, corr=%.3f rad/um; url=%s)"
          % (did, g.nseq(), (max(kw["step_um"]) if isinstance(kw.get("step_um"), list)
                              else kw.get("step_um", STEP_UM)), kw.get("nsteps", NSTEPS),
             kw.get("hold_ms", HOLD_MS), kw.get("piston_corr", PISTON_CORR), url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cheap +z/-z axial asymmetry probes.")
    ap.add_argument("--mode", default="period", choices=("period",))
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--step-um", default=None,
                    help="axial step magnitude in um; a COMMA-SEPARATED list makes dim 1 the full "
                         "signed axis (default %.1f)" % STEP_UM)
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR)
    ap.add_argument("--periods", default=None,
                    help="comma-separated step_period_ms values (default %s)"
                         % ",".join("%g" % p for p in PERIODS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    _p = ([float(x) for x in args.periods.split(",") if x.strip()] if args.periods else None)
    _su = STEP_UM
    if args.step_um:
        _vals = [float(x) for x in str(args.step_um).split(",") if x.strip()]
        _su = _vals if len(_vals) > 1 else _vals[0]
    kw = dict(mode=args.mode, step_um=_su, nsteps=args.nsteps,
              hold_ms=args.hold_ms, piston_corr=args.corr, periods=_p)
    if args.dry_run:
        _seq, _g = build(**kw)
        pers = _p if _p else PERIODS
        print("seq=%s  nseq=%d  nsteps=%d (%d frames)  hold=%g  corr=%g rad/um"
              % (_seq, _g.nseq(), args.nsteps, 2 * args.nsteps + 1, args.hold_ms, args.corr))
        _st = ([-a for a in reversed(sorted(_su))] + [0.0] + sorted(_su)
               if isinstance(_su, list) else [-_su, 0.0, _su])
        print("steps [um] = %s" % _st)
        print("periods    = %s" % pers)
        print("motion ms  = %s"
              % ["%.0f" % ((2 * args.nsteps + 1) * p + args.hold_ms) for p in pers])
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, **kw)
