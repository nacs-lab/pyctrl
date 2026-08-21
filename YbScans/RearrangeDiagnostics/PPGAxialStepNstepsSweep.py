"""PPGAxialStepNstepsSweep.py -- axial step size x nsteps, at the nulled true-defocus piston.

Separates PER-STEP loss from CUMULATIVE loss in the true-defocus axial ping-pong:

    dim 1: extras.step_size = signed AXIAL step, MICRONS of axial displacement (true_defocus=True),
                              both directions
    dim 2: extras.nsteps    = {1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90}

    => survival img1 -> img2 at the SOURCE sites (pingponggrating returns the loaded tweezers to
    where they started, no target assignment, so this is pure transit survival).

A shot with ``nsteps = n`` and ``return=True`` makes **2n steps** (n out + n back) over 2n+1 frames.
So the natural normalization is the PER-STEP survival

    s_step = S ** (1 / (2n))

If loss is purely per-step and memoryless, ``s_step`` is INDEPENDENT of n at fixed step size, and
the whole family collapses onto one curve vs step size. Departures are the physics of interest:
``s_step`` FALLING with n means cumulative damage (heating accumulating faster than a fixed
per-step probability -- each step is worse than the last), while ``s_step`` RISING with n would mean
the fixed per-shot overheads dominate at small n.

*** TWO KNOWN BIASES AT SMALL nsteps -- do not read them as physics ***

1. THE TURNAROUND HOLD. ``hold_ms`` (default 5 ms here) is a FIXED per-shot dwell at the turnaround
   frame k = nsteps, inserted so every step is guaranteed to have settled. Its cost per step is
   ``hold_ms / (2n)`` -- 2.5 ms/step at n=1 but only 0.028 ms/step at n=90. So the SMALL-n points
   get a disproportionate settle benefit, precisely where the onset of any cumulative curve would
   show up. The hold is recorded in the scan description so the analysis can model it.

2. THE TURNAROUND ITSELF. At the turnaround the array reverses direction, so that one frame's LC
   transition differs from every other (it is preceded and followed by motion in opposite senses).
   That single anomalous transition is 1 of 2n steps -- 50% of the steps at n=1, 1.1% at n=90. The
   n=1 shot is ALL turnaround (out, then immediately back), n=2 is half, n=5 a fifth. So the first
   few nsteps values are qualitatively different objects, not just shorter versions of the long ones.

Consequence for interpretation: a per-step curve that looks flat from n=10 upward but deviates at
n = 1, 2, 5 is EXPECTED from the two effects above and is NOT evidence of a cumulative mechanism.
Read the trend over the LARGE-n points, and treat n <= 5 as a separate (settle-advantaged,
turnaround-dominated) regime. The honest test for cumulative loss is whether ``s_step`` drifts
across n = 10..90, where both biases are small and slowly varying.

PISTON: ``depth_piston_corr = 0.617``, the measured true-defocus null (job 320, 2026-07-29, fitted
over |s| <= 2.4 um; the server now interprets this DIRECTLY in rad of uniform phase per MICRON of
axial step when ``true_defocus=True`` -- no kappa rescaling, unlike the older rad-of-Z4-PV numeraire
that produced the equivalent 0.494).

STEP AXIS: deliberately COARSE (5 magnitudes) -- enough to locate the cliff and watch it move with
nsteps, not to resolve its shape, since nsteps is the expensive axis and the point of this scan. Kept
WIDE because the cliff MOVES INWARD as nsteps grows (more steps = more accumulated exposure at the
same per-step size, so the usable step shrinks with n). Anchored on job 320 at n=25, where S was 0.99
at 1.6 um, 0.89-0.98 at 2.4 um and 0.38-0.56 at 3.2 um: so 0.8/1.6 should stay alive far out in n,
2.4 is the n=25 knee and should die somewhere mid-range, 3.2 should die early, and 4.8 is the far
control.

PACING: 0.696 ms (production, chosen for speed) + the 5 ms turnaround hold. Frames per shot = 2n+1,
so wall-clock motion = (2n+1)*0.696 + 5 ms: 7.1 ms at n=1, 131 ms at n=90.

Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus = -4.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialStepNstepsSweep.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialStepNstepsSweep.py --force
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0          # historical default; override with --defocus (the 08-07 campaign uses -5)

BLUE_DETUNING_MHZ = -48.0     # 2026-08-07 job 309: plateau -46..-50; -40/-42 read 0.002
BLUE_LOADING_TIME_S = 0.467   # job 390 vs 386 A/B: 0.25 s -> fill 0.153, 0.467 s -> fill 0.378

PERIOD_MS = 0.696
HOLD_MS = 5.0                 # fixed dwell at the turnaround frame (k = nsteps)
PISTON_CORR = 0.617           # rad of uniform phase per UM of axial step (true_defocus numeraire)

# dim 1: signed axial step, MICRONS. Deliberately COARSE -- just enough to locate the cliff and see
# how it moves with nsteps, not to resolve its shape (the nsteps axis is the expensive one and the
# point of this scan). Kept WIDE because the cliff marches inward as nsteps grows: 0.8 and 1.6 sit
# in the safe/marginal band at n=25, 2.4 was the n=25 knee, 3.2 was already mostly dead there and
# should die early, and 4.8 is the far control. 0 appears ONCE (no-motion control).
_STEP_ABS = [0.8, 1.6, 2.4, 3.2, 4.8]
STEP_SIZES = [-a for a in reversed(_STEP_ABS)] + [0.0] + list(_STEP_ABS)     # 11 values
# dim 2: steps each way. 2*nsteps steps per shot. Out to 90 -> 180 steps, 181 frames, ~131 ms motion.
NSTEPS_LIST = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]

RUN_DESC = (
    "AXIAL step x nsteps at the nulled true-defocus piston (pingponggrating, WGS grating path): "
    "signed step_size in MICRONS -4.8,-3.2,-2.4,-1.6,-0.8, 0, +0.8..+4.8 (11 values, both "
    "directions; deliberately COARSE -- enough to locate the cliff and watch it move, not to resolve "
    "its shape, since the nsteps axis is the expensive one) x nsteps "
    "{1,2,5,10,15,20,30,40,50,60,70,80,90} (13 values) = 143 points. "
    "true_defocus=True (exact spherical axial phase, step in um) with depth_piston_corr=0.617 -- the "
    "measured true-defocus null from job 320, now interpreted DIRECTLY as rad of uniform phase per "
    "um (server semantics changed 2026-07-29; the older rad-of-Z4-PV numeraire gave the equivalent "
    "0.494). step_period_ms=0.696 (production pacing) + hold_ms=5.0, a fixed dwell at the turnaround "
    "frame so every step is guaranteed settled. return=True => 2n+1 frames and 2*nsteps steps per "
    "shot, so the analysis normalizes to per-step survival S**(1/(2n)). "
    "PURPOSE: test whether per-step loss is n-INDEPENDENT (purely per-step, memoryless -> all "
    "nsteps collapse onto one curve vs step size) or DRIFTS with n (cumulative damage), and at what "
    "total movement it becomes significant. "
    "TWO KNOWN SMALL-n BIASES, not physics: (1) the 5 ms hold is a fixed per-shot cost worth "
    "hold_ms/(2n) per step -- 2.5 ms/step at n=1 vs 0.028 at n=90, so small n is settle-advantaged; "
    "(2) the turnaround frame is the one anomalous LC transition (direction reversal) and is 1 of 2n "
    "steps -- 50% of the shot at n=1, 1.1% at n=90, so n<=5 is turnaround-dominated. Read the "
    "cumulative trend over n=10..90. depth_fill_frac cleared (it would override the corr). "
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


def build(step_abs=None, nsteps_list=None, period_ms=PERIOD_MS, hold_ms=HOLD_MS,
          piston_corr=PISTON_CORR, true_defocus=True, defocus=DEFOCUS,
          img_pid=(0.80, 1.00)):
    """Build (do NOT submit) the 2-D [step_size x nsteps] ScanGroup. Returns ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

    mags = sorted({float(a) for a in (step_abs if step_abs else _STEP_ABS) if float(a) > 0})
    steps = [-a for a in reversed(mags)] + [0.0] + list(mags)
    ns = [int(n) for n in (nsteps_list if nsteps_list else NSTEPS_LIST)]

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

    # LOADING RECOVERY (jobs 309/310): 399 blue capture had drifted off plateau -- the old
    # -40/-42 MHz work-point read 0.002 fill. -48 MHz + 0.25 s restores 0.467. Applied as a
    # per-scan g() override; expConfig.py is NOT modified.
    g().BlueMOT.FreqDetuning = BLUE_DETUNING_MHZ * 1e6

    # GREENMOT BIAS-X (jobs 375 + 376, 2026-08-07 05:50). expConfig holds 0.040 A (the 06/05
    # optimum); the resonance has DRIFTED and 0.040 is now DEAD -- job 375 measured 0.002
    # loading there against 0.309 at 0.035, and the fine scan 376 put the peak at 0.0345
    # (0.442 loading, monotonic shoulders both sides, full width ~3 mA).
    #
    # THIS was the slow loading decline (0.61 -> 0.23 over 90 min). Ruled out first, by
    # measurement rather than assumption: the 399 blue detuning (job 367 re-verified -48 as
    # optimal, curve identical to job 309 four hours earlier), the oven (372.06 vs 372.08 C)
    # and imaging (which was IMPROVING, d-prime 11.3). The MOT coils cooled 19.2 -> 17.3 C
    # over the same window, which drifts the field and hence the cloud position relative to
    # the array -- the runbook flags bias-X as a near-vertical resonance where 0.01 A is the
    # difference between full loading and zero, and this is that failure in the wild.
    g().GreenMOT.BiasCoilCurrent.X = 0.0343   # plateau centre (376 full stats: peak 0.0340=0.419, 0.0345=0.407, CV 0.42 both)
    g().BlueMOT.LoadingTime = BLUE_LOADING_TIME_S

    # IMAGING (jobs 331 + 340/341/342, 2026-08-07). Pinned EXPLICITLY at the documented
    # ByPattern 33x33_feedback11 values so every descriptor records them; this is NOT a
    # deviation from config.
    #
    # A 3-point A/B inside THIS sequence (static no-motion cells, 20 shots each) showed
    # PIDSet is NOT the lever here: 0.8/1.0 -> sep 4.04 ADU / d' 6.8, 0.9/0.64 -> 3.80/6.4,
    # 0.63/0.43 -> 3.81/6.5. Identical within noise across a 2.3x span of Img2PIDSet.
    #
    # THE REAL GAP IS THE SEQUENCE, NOT THE POWER. The imaging sequence
    # (ImagingPushoutSurvivalSeq, job 331) reaches sep 7.58 ADU / d' 12.6 at the same nominal
    # setpoint, i.e. ~2x this sequence. Prime suspects, both documented and both OUTSIDE the
    # amps-only scope authorised for this campaign: (a) the imaging tool pins a different 556
    # imaging cooling (h at 0.22 MHz / 0.20 vs the pattern's 0.16 / 0.13), and (b) memory
    # gotcha-imaging-pid-held-multiround-rearrange -- the imaging PID locks once at the root
    # BlueMOT and per-pattern PIDSet never reaches the rearrange images.
    #
    # CAMPAIGN IMAGING CONDITION TO QUOTE WITH EVERY RESULT: separation ~3.9 ADU, d' ~6.5,
    # static two-image survival ~0.97, loading ~0.66. Absolute survivals sit on that floor;
    # per-scan control normalization removes it, bare numbers do NOT. If the sequence gap is
    # closed later, every absolute number here shifts up and the campaign should be re-quoted.
    g().BlueMOT.Img1PIDSet = float(img_pid[0])
    g().BlueMOT.Img2PIDSet = float(img_pid[1])

    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)

    # dim 1: signed axial step (um under true_defocus). Scalar + depth=True -> the pure axial
    # ping-pong (NOT the xyz 3-vector path; a list-valued swept axis breaks the lab-side grid).
    rk.extras.step_size.scan(1, list(steps))
    rk.extras.depth = True
    # dim 2: steps each way -> 2*nsteps steps and 2*nsteps+1 frames per shot.
    rk.nsteps.scan(2, list(ns))

    # TRUE DEFOCUS: exact spherical axial phase k*(1 - sqrt(1 - NA^2 rho^2)) instead of the
    # PARAXIAL ANSI Z4 parabola (whose omitted rho^4 term is primary spherical at a fixed ~0.21
    # fraction of the commanded defocus, so it GROWS with amplitude). step_size is MICRONS here.
    # EXPLICIT because the server extras are sticky across scans and this defaults False.
    rk.extras.true_defocus = True if true_defocus else False

    # The measured true-defocus piston null. Under true_defocus the server takes this DIRECTLY as
    # rad of uniform phase per UM of axial step (semantics changed 2026-07-29; verified live:
    # meta c_sub_unit == 'rad_per_um' and c_sub_applied == the raw value, no kappa rescaling).
    rk.extras.depth_piston_corr = float(piston_corr)
    # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (legacy
    # beam-model path -> corr ignored) and is sticky across scans, which would silently pin the
    # piston to a stale value for every point.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    # depth_fill_center: left UNSET -> the server's measured beam centroid (-0.0684, +0.0117).

    # Fixed dwell at the TURNAROUND frame (k = nsteps) so every step is guaranteed settled.
    # NOTE this is a per-SHOT cost, hence worth hold_ms/(2*nsteps) per step -- a known small-n
    # bias (see the module docstring), not physics.
    rk.extras.hold_ms = float(hold_ms)

    # No commanded per-step uniform phase on top of the map (the map's own constant is set above).
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (matches live production)
    rk.extras.z4 = float(defocus)               # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    npts = len(steps) * len(ns)
    rp.NumPerGroup = 2000                       # ~11 passes over 187 pts
    rp.loading_defocus = float(defocus)
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
    did = ybStartScan(seq_name, g, url=url, label="PPGAxialStepNstepsSweep",
                      description=RUN_DESC, **opts)
    print("submitted PPGAxialStepNstepsSweep -> descriptor id %s (%d pts; period=%.3f ms, "
          "hold_ms=%.1f, true_defocus=True, depth_piston_corr=%.3f rad/um; url=%s)"
          % (did, g.nseq(), kw.get("period_ms", PERIOD_MS), kw.get("hold_ms", HOLD_MS),
             kw.get("piston_corr", PISTON_CORR), url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Axial step x nsteps sweep at the nulled true-defocus piston.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--step-abs", default=None,
                    help="comma-separated step MAGNITUDES in um (default %s)"
                         % ",".join("%g" % a for a in _STEP_ABS))
    ap.add_argument("--nsteps-list", default=None,
                    help="comma-separated nsteps values (default %s)"
                         % ",".join(str(n) for n in NSTEPS_LIST))
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS,
                    help="dwell at the turnaround frame (default %.1f)" % HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR,
                    help="depth_piston_corr in rad/um under true_defocus (default %.3f)"
                         % PISTON_CORR)
    ap.add_argument("--defocus", type=float, default=DEFOCUS,
                    help="ANSI z4 loading defocus / rearrange focal plane (default %g). "
                         "The 2026-08-07 axial campaign uses -5: every 08-06 run at -5 loaded "
                         "0.50-0.60 while -4 loaded 0.11-0.26 on the same MOT and array."
                         % DEFOCUS)
    ap.add_argument("--img-pid", type=float, nargs=2, metavar=("IMG1","IMG2"),
                    default=(0.80, 1.00),
                    help="BlueMOT.Img1/Img2PIDSet (V); default = documented ByPattern")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    _sa = ([float(x) for x in args.step_abs.split(",") if x.strip()]
           if args.step_abs else None)
    _nl = ([int(x) for x in args.nsteps_list.split(",") if x.strip()]
           if args.nsteps_list else None)
    kw = dict(step_abs=_sa, nsteps_list=_nl, period_ms=args.period, defocus=args.defocus,
              img_pid=tuple(args.img_pid),
              hold_ms=args.hold_ms, piston_corr=args.corr)
    if args.dry_run:
        _seq, _g = build(**kw)
        mags = sorted({float(a) for a in (_sa if _sa else _STEP_ABS) if float(a) > 0})
        steps = [-a for a in reversed(mags)] + [0.0] + list(mags)
        ns = _nl if _nl else NSTEPS_LIST
        print("seq=%s  nseq=%d  period=%g ms  hold_ms=%g  corr=%g rad/um  true_defocus=True"
              % (_seq, _g.nseq(), args.period, args.hold_ms, args.corr))
        print("step_sizes [um] = %s" % steps)
        print("nsteps          = %s" % list(ns))
        print("frames/shot     = %s" % [2 * n + 1 for n in ns])
        print("steps/shot      = %s" % [2 * n for n in ns])
        print("motion ms/shot  = %s"
              % ["%.1f" % ((2 * n + 1) * args.period + args.hold_ms) for n in ns])
        print("hold cost/step  = %s ms"
              % ["%.3f" % (args.hold_ms / (2.0 * n)) for n in ns])
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, **kw)
