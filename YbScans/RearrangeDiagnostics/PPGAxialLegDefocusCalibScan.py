"""PPGAxialLegDefocusCalibScan.py -- ARE THE AXIAL STEPS REAL?  A null measurement of how many
microns a radian of ANSI Z4 actually delivers, using the atoms themselves as the detector.

THE PROBLEM
  Every axial number in this campaign is quoted in RADIANS of PV ANSI Z4, and every conversion to
  microns leans on one constant (0.798 um/rad, 2026-07-27; train3d still carries 0.905).  That
  constant has never been checked against the atoms -- it comes from optics plus the lateral
  calibration.  If Z4 under- or over-delivers, every "we moved the atom N um" statement in the
  campaign is wrong by the same factor, including the 20 um layer gap the two-layer array was
  generated with.

THE MEASUREMENT (server-side ``leg_defocus``, added 2026-07-30 -- confirmed live via /eval)
  ``pingponggrating`` can run the OUT and BACK legs of a round trip on DIFFERENT axial maps:
    * outward leg: parabolic ANSI Z4 ``2*rho^2 - c``, amplitude in radians (what we always use),
    * return leg: the EXACT spherical map ``k*(1 - sqrt(1 - NA^2 rho^2))``, amplitude in MICRONS,
      seeded at the same nominal displacement through ``kappa = k*NA^2/4`` rad-of-Z4-PV per um.
  The exact map's micron amplitude is fixed by PHYSICS (lambda = 0.532 um, NA = %(na).2f) rather
  than by a calibration, so it is a ruler.  If a radian of Z4 really moves the atoms by 1/kappa um,
  the return leg exactly retraces the outward leg and the atom lands back in its trap.  Any
  mismatch leaves a residual defocus at the end of the trip and the atom is lost.

  ``return_leg_scale`` turns that into a NULL: it multiplies the return leg's micron amplitude, so
  survival peaks at the value R where the two legs' PHYSICAL travel matches.  Then

        realized microns per radian of Z4  =  R / kappa  =  R * %(inv).4f um/rad

  R = 1 confirms today's constant.  R != 1 rescales every micron in the campaign -- including the
  20 um layer gap -- by exactly that factor.  This is the cleanest available answer to "are we
  actually realizing the steps we claim", and unlike the layer-arrival measurement it needs no
  second layer, no model, and no new detection: it runs on the production 33x33 array with its
  calibrated affine, tuned thresholds and ~1068 sites per shot.

WHAT IT DOES *NOT* MEASURE
  The ruler is only as good as its NA.  ``true_defocus_na`` defaults to %(na).2f (the server's
  measured value, 0.5%% from the 0.6517 pitch estimate); a wrong NA moves the exact map's microns
  as NA^-2 and R absorbs it.  So R pins the Z4 map against the exact map at the ASSUMED NA -- a
  self-consistency test of the axial model, not an NA-free absolute.  The absolute anchor is the
  20 um layer arrival (SLM3DLayerMoveVerifyScan); these two are complementary and should agree.

THE AXES
  dim 1  return_leg_scale %(scales)s -- the null.  Range is deliberately wide: the rho^4 spherical
         residual that Z4 drops is a ~21%% fraction of the commanded defocus, so a null as far out
         as ~0.8 or ~1.2 is physically plausible and a narrow bracket could miss it entirely.
  dim 2  PAIRED (step_size, leg_defocus) -- amplitude, plus the control.  The two maps AGREE in the
         paraxial limit, so the signal only exists at large stroke and the two amplitudes test that
         the null does not drift with amplitude (it must not: a genuine um/rad is amplitude-
         independent, whereas an uncorrected aberration would grow).  The third cell is
         ``leg_defocus="off"`` at the smaller amplitude: a same-map round trip, i.e. the survival
         baseline every scaled cell must be compared against.  ``return_leg_scale`` is ignored
         there, so its 8 repeats are just a high-statistics baseline.

PISTON: each leg is held at ITS OWN measured null, not the auto-transfer.  ``depth_piston_corr``
  = %(c)s rad per rad-of-Z4-PV is the parabolic branch's measured null (07-15, reconfirmed 07-28);
  ``depth_piston_corr_alt`` = %(calt)s rad/um is the exact branch's own measured null (job 320,
  07-29).  The server would otherwise auto-transfer 0.441*kappa = 0.550 rad/um so that both legs
  apply the same uniform phase per step; that is the right default when the alt null is unknown,
  but here it is known, and putting each leg on the PEAK of its own piston ridge is what keeps the
  residual survival variation attributable to amplitude mismatch instead of piston.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialLegDefocusCalibScan.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialLegDefocusCalibScan.py --force
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0
PERIOD_MS = 0.696

# The server's exact-map constants (rearrange_actual._TRUE_DEFOCUS_*): kappa = k*NA^2/4.
LAMBDA_UM = 0.532
TRUE_DEFOCUS_NA = 0.65
KAPPA = (2.0 * math.pi / LAMBDA_UM) * TRUE_DEFOCUS_NA ** 2 / 4.0   # rad of Z4-PV per um
UM_PER_RAD_EXACT = 1.0 / KAPPA                                      # 0.8016

# dim 1 -- the null axis.
SCALES = [0.70, 0.80, 0.88, 0.94, 1.00, 1.06, 1.14, 1.25]

# dim 2 -- PAIRED (step_size rad/step, leg_defocus).  nsteps is fixed so the pacing is constant and
# only the amplitude changes; both per-step values sit well inside the ~2.25 rad per-frame cliff.
NSTEPS = 30
STEP_PAIRED = [25.0627 / NSTEPS, 43.86 / NSTEPS, 25.0627 / NSTEPS]   # 0.835, 1.462, 0.835 rad/step
LEG_PAIRED = ["z4_out", "z4_out", "off"]                             # third = same-map control

CORR_Z4 = 0.441        # parabolic branch, measured null (07-15, reconfirmed 07-28)
CORR_ALT = 0.617       # exact branch, measured null in rad/um (job 320, 07-29)
HOLD_MS = 0.0
NUM_PER_GROUP = 600    # 8 x 3 = 24 cells -> 25 shots/cell x ~650 loaded sites


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(scales=None, nsteps=NSTEPS, period_ms=PERIOD_MS, hold_ms=HOLD_MS):
    _bootstrap()
    from scan_group import ScanGroup

    sc = [float(v) for v in (scales if scales else SCALES)]

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
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(period_ms)
    rk.extras.depth = True
    rk.extras.true_defocus = False          # PRIMARY branch = parabolic Z4; step_size in radians
    rk.extras.true_defocus_na = TRUE_DEFOCUS_NA
    rk.extras.return_trip = True            # leg_defocus requires a return leg
    rk.extras.return_leg_scale.scan(1, list(sc))
    rk.extras.step_size.scan(2, list(STEP_PAIRED))
    rk.extras.leg_defocus.scan(2, list(LEG_PAIRED))
    rk.extras.no_depth_piston = True
    rk.extras.depth_piston_corr = float(CORR_Z4)
    rk.extras.depth_piston_corr_alt = float(CORR_ALT)
    rk.extras.depth_fill_frac = None        # sticky legacy override -- must be cleared explicitly
    rk.extras.hold_ms = float(hold_ms)
    rk.extras.piston = 0.0
    # leg_defocus needs one frame per (displacement, LEG), so the setup-time uint8 cache is refused
    # and the frames are built per shot; precompute still keeps the hot loop write-only.
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
    return seq_name, g, sc


def _desc(sc, nsteps, period_ms, hold_ms):
    totals = ["%.2f rad (%.1f um nominal)" % (s * nsteps, s * nsteps * UM_PER_RAD_EXACT)
              for s in STEP_PAIRED]
    return (
        "AXIAL CALIBRATION NULL -- how many microns does a radian of ANSI Z4 actually deliver?  "
        "pingponggrating depth ROUND TRIP on %s at z4 = loading_defocus = %+.1f, ifEnhanced, "
        "precompute + precompute_host, nsteps=%d, step_period_ms=%.3f, hold_ms=%g.  Uses the "
        "server's leg_defocus mode (added 2026-07-30): the OUTWARD leg runs the parabolic ANSI Z4 "
        "map with step_size in RADIANS (true_defocus=False, the primary branch), and the RETURN leg "
        "runs the EXACT spherical map k*(1-sqrt(1-NA^2 rho^2)) with its amplitude in MICRONS, "
        "seeded at the same nominal displacement through kappa = k*NA^2/4 = %.5f rad-of-Z4-PV per "
        "um (lambda %.3f um, true_defocus_na %.2f).  The exact map's micron amplitude is fixed by "
        "physics rather than by a calibration, so it is a RULER: if a radian of Z4 really moves the "
        "atoms by %.4f um the return leg retraces the outward leg exactly and the atom lands back "
        "in its trap; any mismatch leaves a residual defocus and the atom is lost.  dim 1 = "
        "return_leg_scale %s, which multiplies the return leg's micron amplitude -- survival PEAKS "
        "at the R where the two legs' physical travel matches, and the realized calibration is then "
        "R/kappa = R*%.4f um/rad.  R=1 confirms today's 0.798-0.80 um/rad; R!=1 rescales every "
        "micron in this campaign, including the 20 um two-layer gap, by that factor.  Range is "
        "deliberately wide because the rho^4 residual Z4 drops is a ~21%% fraction of the commanded "
        "defocus.  dim 2 PAIRS step_size with leg_defocus: %s at leg_defocus=z4_out (the two maps "
        "agree in the paraxial limit, so the signal only exists at large stroke, and two amplitudes "
        "test that the null does NOT drift -- a genuine um/rad is amplitude-independent whereas an "
        "uncorrected aberration would grow), plus a leg_defocus='off' same-map control at the "
        "smaller amplitude that provides the survival baseline (return_leg_scale is ignored there, "
        "so its repeats are one high-statistics point).  PISTON: each leg is held at its OWN "
        "measured null rather than the server's auto-transfer -- depth_piston_corr=%.3f rad per "
        "rad-of-Z4-PV (parabolic, 07-15 + 07-28) and depth_piston_corr_alt=%.3f rad/um (exact, job "
        "320 07-29); the auto-transfer would instead put the exact leg at 0.441*kappa=0.550 rad/um. "
        "CAVEAT to quote with any result: the ruler is only as good as the NA, so R pins Z4 against "
        "the exact map at the ASSUMED NA %.2f -- a self-consistency test of the axial model, not an "
        "NA-free absolute.  The NA-free anchor is the 20 um layer-arrival scan; the two must agree."
        % (PATTERN, DEFOCUS, nsteps, period_ms, hold_ms, KAPPA, LAMBDA_UM, TRUE_DEFOCUS_NA,
           UM_PER_RAD_EXACT, sc, UM_PER_RAD_EXACT, totals, CORR_Z4, CORR_ALT, TRUE_DEFOCUS_NA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--scales", default=None, help="comma-separated return_leg_scale list")
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold", type=float, default=HOLD_MS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    sc = [float(v) for v in args.scales.split(",")] if args.scales else None
    seq_name, g, sc = build(scales=sc, nsteps=args.nsteps, period_ms=args.period,
                            hold_ms=args.hold)
    g.runp().NumPerGroup = int(args.num_per_group)
    print("seq=%s nseq=%d  kappa=%.5f rad/um -> %.4f um/rad (NA %.2f, lambda %.3f)"
          % (seq_name, g.nseq(), KAPPA, UM_PER_RAD_EXACT, TRUE_DEFOCUS_NA, LAMBDA_UM))
    print("  dim1 return_leg_scale = %s" % sc)
    for s, leg in zip(STEP_PAIRED, LEG_PAIRED):
        print("  dim2 step %.4f rad/step x %d = %.2f rad (%.1f um nominal)   leg_defocus=%s"
              % (s, args.nsteps, s * args.nsteps, s * args.nsteps * UM_PER_RAD_EXACT, leg))
    if args.dry_run:
        seen = set()
        for i in range(g.nseq()):
            e = g.getseq(i)["rearrange_kwargs"]["extras"]
            seen.add((round(e.get("step_size"), 6), e.get("leg_defocus"),
                      e.get("return_leg_scale")))
        print("  %d distinct (step, leg, scale) cells" % len(seen))
        s0 = g.getseq(0)["rearrange_kwargs"]["extras"]
        print("  cell0: %s" % {k: s0.get(k) for k in
                               ("step_size", "leg_defocus", "return_leg_scale", "true_defocus",
                                "true_defocus_na", "depth_piston_corr", "depth_piston_corr_alt",
                                "no_depth_piston", "return_trip", "precompute")})
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label="PPGAxialLegDefocusCalibRearrangeScan",
                      description=_desc(sc, args.nsteps, args.period, args.hold), **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
