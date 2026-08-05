"""PPG2DDistanceDirectionSweep.py -- FINAL 2-D lateral movement-capability map (pingponggrating).

The definitive re-measure of how far the array can be moved LATERALLY per frame, and whether that
reach depends on the direction of travel:

    dim 1: extras.step_r         = per-frame move DISTANCE (px), 15 values
                                   0, 0.25, 0.5, 0.75, then 1.0 .. 2.0 every 0.1
    dim 2: extras.step_theta_deg = direction around the unit circle, 8 values 0..315 every 45 deg
                                   (0 = +x, 90 = +y, measured from the +x axis)

    => 15 x 8 = 120 points. Survival img1 -> img2 at the SOURCE sites (pingponggrating moves the
    LOADED tweezers out-and-back with no target assignment, so this is pure transit survival).

POLAR AXES, NOT x/y: the two swept axes are the physical knobs of interest (distance, direction),
so the analysis grid is a native [15 x 8] map -- no reshaping, no derived columns. Per shot,
``rearrange_callbacks._fold_step_polar`` turns (step_r, step_theta_deg) into scalar
``step_x = r*cos(theta)`` / ``step_y = r*sin(theta)``, which ``_fold_step_xyz`` then folds into the
xyz ``step_size`` 3-vector the dispatcher reads. Sweeping step_x/step_y directly CANNOT express a
direction (it needs two co-varying scalars), and a list-valued ``step_size`` axis breaks the
lab-side N-D scan grid ("concatenation axis doesn't match along axis 0" at dequeue).

100 STEPS OF MOTION: ``nsteps=50`` with the default return trip (``return=True``) gives the
2*50+1 = 101 frames = 50 steps out + 50 steps back that the campaign asks for.

NO PHASE CHANGE ACROSS STEPS: ``piston=0`` AND ``no_depth_piston=True``, so the net commanded
uniform phase change per frame is exactly zero. This is a purely LATERAL sweep (step_z = 0), so
the Z4 defocus map -- and hence the depth piston-correction constant -- never enters; but
``no_depth_piston=True`` is still set EXPLICITLY because the server extras are STICKY across scans
in one backend session and a previous depth scan may have left it False.

BEAM CENTRE IS THE POINT OF ``depth_fill_center``: it is left at the server's measured default
(-35.0/512, +6.0/512) = (-0.0684, +0.0117) normalized, i.e. the calibrated beam centroid (2026-07-15
sids 20260715034955 [x] + 20260715041404 [y]) -- NOT overridden here. It recentres the BLAZE (both
x and y) on the beam instead of the panel. That matters for exactly this measurement: the blaze is
an ODD map about its centre, so a symmetric beam of any width integrates to zero over it and
underfill alone adds NO grating piston -- the only way a moving grating picks up a per-step
effective piston is a beam DECENTRE, which grows with step distance and FLIPS SIGN with the
direction of travel. Left uncentred, that decentre piston would masquerade as a real
direction-dependent movement limit (and is the leading suspect for the +-y cliff asymmetry in
``open-ppg-ycliff-asymmetry``). Centring it nulls the artefact so this map measures GEOMETRY.
The default only applies while ``no_depth_piston`` is True -- another reason it is set explicitly.

PACING: ``step_period_ms=3.0``. The 2026-07-22 campaign showed the 0.696 ms cliff (~1.28 px) and
its heating are LC-SETTLE artefacts, not geometric: >= 1.4 ms removes both, and at 3 ms the true
geometric cliff sits at ~1.65 px ~= 1.16 w0/step (dead by 2.0). 3 ms therefore measures the real
optical/geometric reach, which is what "movement capability" means here, and the 0..2.0 px range
brackets that cliff on both sides. 101 frames x 3 ms = 303 ms of motion per shot.
``precompute`` + ``precompute_host`` ON so the paced loop is write-only and 3 ms is really realised.

Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus = -4.

NOTE the ``step_r = 0`` row is the in-scan NO-MOTION control (frames are still written, so it
controls for the write process itself). It is physically ONE point but appears in all 8 direction
columns; those 8 cells are independent repeats of the same condition, so they double as a
consistency check (they must agree within statistics) and as 8x statistics on the control.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPG2DDistanceDirectionSweep.py
    python YbScans/RearrangeDiagnostics/PPG2DDistanceDirectionSweep.py --dry-run
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

NSTEPS = 50           # return trip -> 2*50+1 = 101 frames = 50 steps out + 50 back
PERIOD_MS = 3.0

# dim 1: per-frame move distance (px). Coarse below 1 px, then 0.1 through the geometric cliff.
DISTANCES = [0.0, 0.25, 0.5, 0.75,
             1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# dim 2: direction of travel (deg from +x), every 45 deg around the unit circle.
ANGLES_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

RUN_DESC = (
    "FINAL 2-D lateral movement-capability map (pingponggrating): step_r = per-frame distance "
    "0, 0.25, 0.5, 0.75, 1.0..2.0 every 0.1 (15 values) x step_theta_deg = direction 0..315 every "
    "45 deg (8 values) = 120 points. Polar axes folded per shot to step_x/step_y -> xyz step_size "
    "(_fold_step_polar + _fold_step_xyz), so the analysis grid is a native [15 x 8] distance x "
    "direction map. nsteps=50 with the default return trip = 101 frames = 50 steps out + 50 back "
    "(the requested 100 steps). NO phase change across steps: piston=0 and no_depth_piston=True "
    "(explicit -- server extras are sticky). depth_fill_center left at the measured beam centroid "
    "(-0.0684, +0.0117) so the BLAZE is centred on the beam: a beam decentre is the only way a "
    "moving grating picks up a per-step piston, it grows with distance and flips sign with "
    "direction, and uncentred it would masquerade as a direction-dependent movement limit (leading "
    "suspect for the +-y cliff asymmetry). step_period_ms=3.0 -- >=1.4 ms removes the LC-settle "
    "artefacts of the 0.696 ms cliff (~1.28 px), so this measures the TRUE geometric cliff "
    "(~1.65 px ~= 1.16 w0/step at 3 ms), which 0..2.0 px brackets. precompute + precompute_host on. "
    "step_r=0 row = no-motion control (frames still written), repeated in all 8 direction columns "
    "as a consistency check. Array 33x33_feedback11, z4 = loading_defocus = -4."
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (pingponggrating returns to the source sites), so img1
    and img2 are both detected against 33x33_feedback11's registry grid + thresholds. The name is
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(nsteps=NSTEPS, period_ms=PERIOD_MS):
    """Build (do NOT submit) the 2-D [distance x direction] ScanGroup. Returns ``(seq_name, g)``."""
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

    # ---- rearrange_kwargs: pingponggrating, LATERAL xyz mode ---------------------------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(period_ms)

    # POLAR axes -> per-shot step_x/step_y -> xyz step_size (see the module docstring).
    rk.extras.step_r.scan(1, list(DISTANCES))
    rk.extras.step_theta_deg.scan(2, list(ANGLES_DEG))
    # Purely lateral: no axial component. Set explicitly so a sticky step_z from a previous
    # depth scan in the same backend session cannot leak in.
    rk.extras.step_z = 0.0
    # NOT depth mode -- the xyz 3-vector path (scalar step_size + depth=True is the axial path).
    rk.extras.depth = False

    # NO phase change across steps: zero commanded per-step uniform phase, and the depth map
    # made pistonless. EXPLICIT because the server extras are sticky across scans.
    rk.extras.piston = 0.0
    rk.extras.no_depth_piston = True
    # depth_fill_center: LEFT UNSET on purpose -> the server's measured beam centroid
    # (-35.0/512, +6.0/512), which recentres the BLAZE on the beam and nulls the
    # decentre piston that would otherwise fake a direction-dependent limit. Only applied
    # while no_depth_piston is True (set above).
    # depth_fill_frac / depth_piston_corr: irrelevant here (no defocus term in a lateral move).

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (matches live production)
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1000                       # -> ceil(1000/120) = 9 passes = 1080 shots
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def PPG2DDistanceDirectionSweep(url=None, reps=None, nsteps=NSTEPS, period_ms=PERIOD_MS):
    seq_name, g = build(nsteps=nsteps, period_ms=period_ms)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "PPG2DDistanceDirectionSweep"
    did = ybStartScan(seq_name, g, url=url, label=label, description=RUN_DESC, **opts)
    print("submitted %s -> descriptor id %s (%d distances x %d directions = %d pts; nsteps=%d "
          "(%d frames), period=%.3f ms, piston=0, no_depth_piston=True; url=%s)"
          % (label, did, len(DISTANCES), len(ANGLES_DEG), g.nseq(), int(nsteps),
             2 * int(nsteps) + 1, float(period_ms), url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="FINAL 2-D lateral movement map: distance x direction (pingponggrating).")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--nsteps", type=int, default=NSTEPS,
                    help="steps each way (default %d -> %d frames = %d steps)"
                         % (NSTEPS, 2 * NSTEPS + 1, 2 * NSTEPS))
    ap.add_argument("--period", type=float, default=PERIOD_MS,
                    help="step_period_ms (default %.3f)" % PERIOD_MS)
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    if args.dry_run:
        _seq, _g = build(nsteps=args.nsteps, period_ms=args.period)
        print("seq=%s  nseq=%d  nsteps=%d (%d frames = %d steps)\ndistances=%s\nangles_deg=%s"
              % (_seq, _g.nseq(), args.nsteps, 2 * args.nsteps + 1, 2 * args.nsteps,
                 DISTANCES, ANGLES_DEG))
    else:
        PPG2DDistanceDirectionSweep(url=args.url, reps=args.reps,
                                    nsteps=args.nsteps, period_ms=args.period)
