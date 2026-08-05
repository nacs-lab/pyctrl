"""PPGAxialPistonCorrSweep.py -- AXIAL capability x depth-piston correction (pingponggrating).

Step 1 of the axial re-measure: find the piston-correction constant that maximises axial reach,
BEFORE the fine axial-step/period map (step 3) is run with the winner.

    dim 1: extras.step_size        = signed AXIAL step (rad PV ANSI Z4 per frame), 9 values
                                     -7, -6, -5, -4, 0, +4, +5, +6, +7 -- both directions
    dim 2: extras.depth_piston_corr = the piston-correction constant, 7 values 0.2 .. 0.8
                                     every 0.1

    => 9 x 7 = 63 points. Survival img1 -> img2 at the SOURCE sites (pingponggrating moves the
    LOADED tweezers out-and-back with no target assignment, so this is pure transit survival).

ZOOMED RE-SCOPE (2026-07-29, superseding the first 357-point attempt -- job 307, aborted at 260
shots): the reconnaissance ranges were cut to the region that actually determines the answer, for
~5.7x the data rate. ``depth_piston_corr`` 0.2..0.8 brackets the 0.441 server default (the
2026-07-15 survival null) with margin instead of scanning -1..+1; |step| is capped at 7 and the
small steps 1/2/3 are dropped because they suffer little loss, so the ridge is shallow and poorly
localized there, whereas |step| >= 4 pins it sharply. The +-{4..7} pairs still give the
two-direction structure needed to separate a genuine piston null from a direction-dependent
(decentre) asymmetry, and step_size=0 remains the no-motion control.

WHAT ``depth_piston_corr`` IS (the 2026-07-28/29 server reparametrization, and the reason this scan
exists): the depth map written to the panel is ``2*rho^2 - c``, and ``c`` is the constant that sets
how much UNIFORM (piston) phase each axial step carries. ``depth_piston_corr`` commands ``c``
DIRECTLY, in rad of uniform phase per rad of Z4-PV step. It REPLACES the old ``depth_fill_frac``
beam-model parametrization (an intensity-weighted Gaussian mean; f = 0.675 gave the same c = 0.441),
because physically the constant is the coherence-apodized FIELD-weighted map mean and the fill-frac
label was a sqrt(2)-off reparametrization of it -- so the measured number is now commanded directly.
The server default is ``c = 0.441`` (the survival-null from the 2026-07-15 fill sweep at |s| = 3;
~0.55 rad per um of axial step at 0.80 um/rad). This scan sweeps c through and PAST that default in
both signs to re-find the survival ridge, and -- because the ridge is where the per-step uniform
transient vanishes -- the c that maximises axial reach.

WHY THE SWEEP MUST BE IN ``depth_piston_corr``, NOT ``depth_fill_frac``: the requested -1..+1 range
is NOT reachable through fill_frac. fill_frac maps to c as an intensity-weighted mean of
``2*rho^2 >= 0``, so it can only ever produce c > 0 (and non-linearly). Negative c -- half the range
asked for -- is representable ONLY via the direct parametrization.

``depth_fill_frac`` IS EXPLICITLY CLEARED (set to None) HERE. The server extras are STICKY across
scans within one backend session, and an explicit positive ``depth_fill_frac`` TAKES PRECEDENCE over
``depth_piston_corr`` (it activates the deprecated legacy beam-model path and ``depth_piston_corr``
is then ignored entirely -- ``_fill_params`` returns corr=None). Since the historical default was
0.65 and many earlier scans set it, leaving it sticky would silently pin every one of the 63 points
to the SAME c and return a perfectly flat, meaningless map. Clearing it is mandatory, not defensive.

``depth_fill_center`` is left at the server's measured beam centroid (-0.0684, +0.0117): it centres
the Z4 map (and the blaze) on the beam rather than the panel, which is separate physics from the
subtraction constant and is not what this scan sweeps.

PERIOD 3 ms as requested -- the pacing at which the 2026-07-22 campaign showed LC-settle artefacts
are gone, so the loss measured here is real transit/piston loss. ``nsteps=20`` with the default
return trip => 2*20+1 = 41 frames = 123 ms of motion per shot (kept at the 07-28 depth-map value so
this map is directly comparable to jobs 244/248/250).

Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus = -4.

*** PREREQUISITE -- READ BEFORE RUNNING ***
``depth_piston_corr`` must be LIVE on the SLM server. As of 2026-07-29 17:56 the reparametrization
is present in the server's rearrange_actual.py ON DISK but the running process still had the OLD
code in memory (``_fill_params`` returning a 3-tuple with fill_frac=0.65, and
``_depth_piston_subtract`` without the ``piston_corr`` argument). The key is read via
``extras.get`` and is therefore silently IGNORED by the old code -- no error, just a flat map. Verify
it is live BEFORE trusting a run, e.g. via /eval:

    import slmnet.experimental.tools.rearrange_actual as ra
    ra._fill_params({})            # must return a 4-TUPLE (ndp, corr, frac, center) with corr=0.441
    hasattr(ra, "_DEPTH_PISTON_CORR_DEFAULT")   # must be True

``--force`` is required to submit precisely so this check is not skipped by reflex.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialPistonCorrSweep.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialPistonCorrSweep.py --force
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

NSTEPS = 20           # return trip -> 2*20+1 = 41 frames = 123 ms at 3 ms
PERIOD_MS = 3.0

# dim 1: signed axial step (rad PV ANSI Z4 per frame). 0 appears ONCE, both directions otherwise.
# ZOOMED (2026-07-29 re-scope): |step| capped at 7 and the small steps 1/2/3 dropped -- the small
# ones carry little diagnostic weight for locating the piston ridge (the loss they suffer is tiny,
# so the ridge is shallow and poorly determined there), while |step| >= 4 is where the ridge is
# sharp. 9 values instead of 17.
_STEP_ABS = [4.0, 5.0, 6.0, 7.0]
STEP_SIZES = [-a for a in reversed(_STEP_ABS)] + [0.0] + list(_STEP_ABS)      # 9 values, ascending
# dim 2: the piston-correction constant c. ZOOMED to 0.2 .. 0.8 every 0.1 (7 values) -- brackets the
# 0.441 server default (the 2026-07-15 survival null) with margin on both sides, instead of the
# full -1..+1 reconnaissance range. 9 x 7 = 63 points, ~5.7x faster than the original 357.
PISTON_CORRS = [round(0.2 + 0.1 * i, 4) for i in range(7)]

RUN_DESC = (
    "AXIAL capability x depth-piston correction (pingponggrating, WGS grating path), ZOOMED: signed "
    "step_size (rad PV ANSI Z4/frame) -7,-6,-5,-4, 0, +4,+5,+6,+7 (9 values, both directions) x "
    "depth_piston_corr 0.2..0.8 every 0.1 (7 values) = 63 points. Re-scope of the aborted 357-point "
    "job 307 (stopped at 260 shots): corr zoomed to bracket the 0.441 default (2026-07-15 survival "
    "null) with margin, |step| capped at 7, and small steps 1/2/3 dropped -- they suffer little loss "
    "so the ridge is shallow/poorly localized there, while |step|>=4 pins it sharply. ~5.7x faster. "
    "The +-{4..7} pairs retain the two-direction structure needed to tell a genuine piston null from "
    "a direction-dependent decentre asymmetry; step_size=0 is the no-motion control. "
    "depth_piston_corr is the "
    "2026-07-28/29 reparametrization: it commands the constant c in the written depth map "
    "2*rho^2 - c DIRECTLY, in rad of uniform phase per rad of Z4-PV step, replacing the old "
    "depth_fill_frac beam model (f=0.675 <-> c=0.441, the server default and the 2026-07-15 "
    "survival null). Swept in corr and NOT in fill_frac because fill_frac is an intensity-weighted "
    "mean of 2*rho^2 >= 0 and so cannot represent the negative half of the requested range. "
    "depth_fill_frac is EXPLICITLY CLEARED (None): it is sticky across scans and TAKES PRECEDENCE "
    "over depth_piston_corr, so a leftover 0.65 would pin all 63 points to one c and return a flat "
    "map. depth_fill_center left at the measured beam centroid (-0.0684, +0.0117). nsteps=20, "
    "step_period_ms=3.0, return trip (41 frames, 123 ms motion) -- 3 ms is past the LC-settle "
    "artefacts so the loss is real transit/piston loss; matches jobs 244/248/250 for comparability. "
    "precompute + precompute_host on. Step 1 of the axial re-measure: picks the c for the "
    "fine axial step x period map. Array 33x33_feedback11, z4 = loading_defocus = -4."
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (pingponggrating returns to the source sites). Name =
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _step_sizes(step_abs=None):
    """Signed step axis from a list of MAGNITUDES: ``[-max..-min, 0, +min..+max]``, 0 once.

    ``step_abs=None`` -> the module default :data:`_STEP_ABS`. Overridable per run (``--step-abs``)
    so the |step| window can be moved without editing the module while another job is running off
    this same file."""
    mags = sorted({float(a) for a in (step_abs if step_abs else _STEP_ABS) if float(a) > 0})
    return [-a for a in reversed(mags)] + [0.0] + list(mags)


def build(nsteps=NSTEPS, period_ms=PERIOD_MS, step_abs=None, piston_corrs=None,
          true_defocus=False, true_defocus_na=None):
    """Build (do NOT submit) the 2-D [axial step x piston_corr] ScanGroup. Returns
    ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

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

    # ---- rearrange_kwargs: pingponggrating, DEPTH mode ---------------------------------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(period_ms)

    # Scalar step_size + depth=True -> pure Z4 defocus ping-pong (NOT the xyz 3-vector path;
    # a list-valued swept axis would break the lab-side N-D scan grid).
    _steps = _step_sizes(step_abs)
    _corrs = [float(c) for c in (piston_corrs if piston_corrs else PISTON_CORRS)]
    rk.extras.step_size.scan(1, list(_steps))
    rk.extras.depth = True

    # dim 2: the piston-correction constant, commanded directly (see the module docstring).
    rk.extras.depth_piston_corr.scan(2, list(_corrs))

    # TRUE DEFOCUS (server extra, 2026-07-29): replace the PARAXIAL ANSI Z4 map (2*rho^2 - c, only
    # the rho^2 term of the real axial-shift wavefront) with the EXACT spherical form
    # S(rho) = k*(1 - sqrt(1 - NA^2 rho^2)). Z4 omits the rho^4 term = primary spherical whose size
    # is a FIXED FRACTION (~0.21) of the commanded defocus, so it does NOT wash out at large
    # amplitude -- it GROWS with it. The parabolic map is diffraction-limited only to ~20-30 rad of
    # Z4-PV (~16-24 um), while these sweeps ramp to nsteps*step = 50-200 rad (40-160 um), where the
    # residual is 0.2-0.7 waves and the turnaround frame's Strehl has collapsed -- i.e. the peak
    # frames were not producing a diffraction-limited trap at all.
    #
    # *** UNITS CHANGE: with true_defocus=True, step_size is MICRONS of axial displacement, NOT
    # radians of Z4-PV *** (kappa = k*NA^2/4 = 1.2475 rad/um at NA 0.65, i.e. 0.8016 um/rad -- the
    # established axial scale). depth_piston_corr KEEPS its rad-of-Z4-PV numeraire and is rescaled
    # by kappa internally, so the 0.2..0.8-style range and the earlier 0.49/0.51 nulls remain the
    # right reference. But the map SHAPE changed and the null is the field-weighted mean of the map,
    # so the server docstring explicitly says to RE-NULL depth_piston_corr under true_defocus before
    # trusting large-amplitude piston scans -- which is exactly what this scan does.
    #
    # Set EXPLICITLY in both states (server extras are sticky across scans in one backend session,
    # and this defaults FALSE), so a true_defocus run cannot inherit False and silently be a
    # parabolic re-run, nor can a later parabolic run inherit True.
    rk.extras.true_defocus = bool(true_defocus)
    if true_defocus and true_defocus_na is not None:
        # NA is not a free parameter (rho=1 is the knm-1024 Nyquist edge -> NA_disk = 0.6517,
        # default 0.65); override only deliberately. Must stay <~0.68 or the panel corners
        # (rho ~ 1.47) clamp flat and dump light to DC.
        rk.extras.true_defocus_na = float(true_defocus_na)
    # MANDATORY: clear the sticky legacy override. An explicit positive depth_fill_frac wins over
    # depth_piston_corr (legacy beam-model path -> corr ignored), which would flatten the whole map.
    rk.extras.depth_fill_frac = None
    # ON so the correction constant is actually applied (c = depth_piston_corr). EXPLICIT because
    # the server extras are sticky and the depth x piston maps (jobs 244/248/250) set this False.
    rk.extras.no_depth_piston = True
    # depth_fill_center: left UNSET -> the server's measured beam centroid (-0.0684, +0.0117).

    # No commanded per-step uniform phase ON TOP of the map: the map's own constant is the variable.
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (matches live production)
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1000                       # -> ceil(1000/63) = 16 passes = 1008 shots
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def PPGAxialPistonCorrSweep(url=None, reps=None, nsteps=NSTEPS, period_ms=PERIOD_MS,
                            step_abs=None, piston_corrs=None,
                            true_defocus=False, true_defocus_na=None):
    seq_name, g = build(nsteps=nsteps, period_ms=period_ms,
                        step_abs=step_abs, piston_corrs=piston_corrs,
                        true_defocus=true_defocus, true_defocus_na=true_defocus_na)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    _steps = _step_sizes(step_abs)
    _corrs = [float(c) for c in (piston_corrs if piston_corrs else PISTON_CORRS)]
    label = "PPGAxialPistonCorrSweep"
    # Non-default axes go in the LABEL so the queue/dashboard distinguishes the runs, and in the
    # DESCRIPTION so the provenance of each run says what it actually swept (the module RUN_DESC
    # describes the defaults).
    desc = RUN_DESC
    if step_abs or piston_corrs or float(period_ms) != PERIOD_MS or true_defocus:
        label += "_p%g" % float(period_ms)
        if true_defocus:
            label += "_TD"
        if step_abs:
            label += "_s" + "-".join("%g" % a for a in sorted({float(x) for x in step_abs}))
        _unit = "um axial (true_defocus)" if true_defocus else "rad Z4-PV"
        desc += (" THIS RUN OVERRIDES THE DEFAULTS ABOVE: step_size = %s [%s]; "
                 "depth_piston_corr = %s [rad of uniform phase per rad of Z4-PV, rescaled by "
                 "kappa internally]; step_period_ms = %g; nsteps = %d (excursion = nsteps*step)."
                 % (_steps, _unit, _corrs, float(period_ms), int(nsteps)))
        if true_defocus:
            desc += (" TRUE_DEFOCUS ON: the exact spherical axial phase "
                     "k*(1 - sqrt(1 - NA^2 rho^2)) replaces the PARAXIAL ANSI Z4 parabola, whose "
                     "omitted rho^4 term is primary spherical at a fixed ~0.21 fraction of the "
                     "commanded defocus and therefore GROWS with amplitude (Z4 is "
                     "diffraction-limited only to ~20-30 rad Z4-PV / ~16-24 um, while these sweeps "
                     "ramp far past that). step_size is now MICRONS; NA=%s. Re-nulls "
                     "depth_piston_corr under the NEW map shape, as the server docstring requires."
                     % (true_defocus_na if true_defocus_na is not None else "0.65 (default)"))
    did = ybStartScan(seq_name, g, url=url, label=label, description=desc, **opts)
    print("submitted %s -> descriptor id %s (%d axial steps x %d piston_corr = %d pts; nsteps=%d "
          "(%d frames), period=%.3f ms, no_depth_piston=True, depth_fill_frac cleared; url=%s)"
          % (label, did, len(_steps), len(_corrs), g.nseq(), int(nsteps),
             2 * int(nsteps) + 1, float(period_ms), url or "default"))
    print("   steps = %s" % _steps)
    print("   corrs = %s" % _corrs)
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Axial step x depth_piston_corr 2-D sweep (pingponggrating).")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--nsteps", type=int, default=NSTEPS,
                    help="steps each way (default %d -> %d frames)" % (NSTEPS, 2 * NSTEPS + 1))
    ap.add_argument("--period", type=float, default=PERIOD_MS,
                    help="step_period_ms (default %.3f)" % PERIOD_MS)
    ap.add_argument("--step-abs", default=None,
                    help="comma-separated step MAGNITUDES; the signed axis is built as "
                         "[-max..-min, 0, +min..+max] (default %s)"
                         % ",".join("%g" % a for a in _STEP_ABS))
    ap.add_argument("--corrs", default=None,
                    help="comma-separated depth_piston_corr values (default %s)"
                         % ",".join("%g" % c for c in PISTON_CORRS))
    ap.add_argument("--true-defocus", action="store_true",
                    help="use the EXACT spherical axial phase instead of the paraxial ANSI Z4 "
                         "parabola. NOTE this reinterprets step_size from rad of Z4-PV to MICRONS "
                         "(0.8016 um/rad); depth_piston_corr keeps its rad numeraire.")
    ap.add_argument("--true-defocus-na", type=float, default=None,
                    help="override the pupil NA (default 0.65 = the knm-1024 Nyquist edge); "
                         "must stay <~0.68 or the panel corners clamp flat")
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    ap.add_argument("--force", action="store_true",
                    help="required to submit: confirms depth_piston_corr is LIVE on the SLM "
                         "server (see the PREREQUISITE block in the module docstring -- the old "
                         "in-memory code ignores the key silently and returns a flat map)")
    args = ap.parse_args()
    _sa = ([float(x) for x in args.step_abs.split(",") if x.strip()]
           if args.step_abs else None)
    _pc = ([float(x) for x in args.corrs.split(",") if x.strip()]
           if args.corrs else None)
    if args.dry_run:
        _seq, _g = build(nsteps=args.nsteps, period_ms=args.period,
                         step_abs=_sa, piston_corrs=_pc,
                         true_defocus=args.true_defocus,
                         true_defocus_na=args.true_defocus_na)
        _st = _step_sizes(_sa)
        _u = "um axial" if args.true_defocus else "rad Z4-PV"
        print("seq=%s  nseq=%d  nsteps=%d (%d frames)  period=%g ms  true_defocus=%s"
              % (_seq, _g.nseq(), args.nsteps, 2 * args.nsteps + 1, args.period,
                 args.true_defocus))
        print("step_sizes [%s] = %s" % (_u, _st))
        print("  -> turnaround excursion = nsteps*step = %s"
              % ["%g" % (args.nsteps * s) for s in _st])
        print("piston_corrs = %s" % (_pc if _pc else PISTON_CORRS))
    elif not args.force:
        ap.error("refusing to submit without --force: verify depth_piston_corr is LIVE on the SLM "
                 "server first (ra._fill_params({}) must return a 4-tuple with corr=0.441). The "
                 "old in-memory code IGNORES the key silently -> flat, worthless map. "
                 "Use --dry-run to inspect the build.")
    else:
        PPGAxialPistonCorrSweep(url=args.url, reps=args.reps,
                                nsteps=args.nsteps, period_ms=args.period,
                                step_abs=_sa, piston_corrs=_pc,
                                true_defocus=args.true_defocus,
                                true_defocus_na=args.true_defocus_na)
