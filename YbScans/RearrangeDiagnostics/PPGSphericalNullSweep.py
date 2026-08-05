"""PPGSphericalNullSweep.py -- pingponggrating primary-spherical (ANSI Z12) NULL sweep.

Measures the corrective primary-spherical coefficient model-free via a depth-mode z-asymmetry
null. The objective's off-plane spherical makes +z survive better than -z; adding a corrective
Z12 on the TRANSIT frames cancels it, so the asymmetry

    A(C12) = surv(step_size = +4) - surv(step_size = -4)

crosses ZERO at the correction -- and there both branches should also sit HIGHER (deeper z reach)
than anywhere else. The ``step_size = 0`` row is the in-scan no-motion control: it isolates any
effect of the Z12 aberration by itself, with no depth transit.

    dim 1: extras.step_size    = {-4, 0, +4} rad PV ANSI Z4 per frame (the two z-directions
                                 plus the no-motion control)
    dim 2: extras.distortion_z12 = C12 from -5 to +5 in 0.25 steps, 41 values (PV rad, ANSI
                                 index 12 = primary spherical = the 13th element of the
                                 coefficient list; folded to distortion_zernike[12] per shot by
                                 rearrange_callbacks._fold_distortion_zernike -- a list-valued
                                 swept axis would break the lab-side live curve)

    => 3 x 41 = 123 points.

PISTON IS NULLED, NOT SWEPT (unlike the depth x piston maps, jobs 244/248/250). For the spherical
to be the ONLY residual, the depth map is made genuinely pistonless: ``no_depth_piston=True`` with
the measured beam (``depth_fill_frac=0.65``, ``depth_fill_center=[-0.0689, 0.0118]``) and
``piston=0``. Left at the maps' ``no_depth_piston=False`` the canonical Z4 would inject
0.336 * 4 = 1.34 rad/step of uniform phase -- past the ~1.0-1.2 rad/step loss onset measured in the
2026-07-22 campaign -- which would floor survival at both +-4 and swamp the asymmetry being
measured. Override if a piston-loaded null is wanted instead.

``reverse_zernike=False`` is EXPLICIT: a corrective spherical is a STATIC aberration and must keep
the same sign on both legs of the triangle (the return leg retraces the same +z displacements, it
does not go to -z). It also halves the unique-frame count. NOTE ``distortion_zernike`` bypasses the
uint8 precompute cache and forces the deduped float build -- 51 unique frames at nsteps=50, ~4 MB
each (~200 MB); ``reverse_zernike=True`` would roughly double that.

nsteps=50, step_period_ms=3.0 => 2*50+1 = 101 frames = 303 ms of motion per shot.
Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus = -4.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGSphericalNullSweep.py
    python YbScans/RearrangeDiagnostics/PPGSphericalNullSweep.py --dry-run
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

NSTEPS = 50
PERIOD_MS = 3.0

# dim 1: the two z-directions + the no-motion control (rad PV ANSI Z4 per frame).
STEP_SIZES = [-4.0, 0.0, 4.0]
# dim 2: corrective primary spherical C12 (PV rad), -5 .. +5 every 0.25 -> 41 values.
C12_VALUES = [round(-5.0 + 0.25 * i, 2) for i in range(41)]

# Measured beam (pinned) -- makes the depth map pistonless AS THE ATOMS SEE IT, so the only
# residual left for the sweep to null is the spherical.
FILL_FRAC = 0.65
FILL_CENTER = [-0.0689, 0.0118]      # normalized [cx, cy]

RUN_DESC = (
    "pingponggrating primary-spherical (ANSI Z12) NULL sweep: step_size {-4, 0, +4} rad PV Z4 x "
    "distortion_z12 -5..+5 every 0.25 (41 pts) = 123 points. Finds the corrective C12 where the "
    "z-asymmetry A(C12) = surv(+4) - surv(-4) crosses zero (and both branches peak). step_size=0 "
    "row = no-motion control isolating the aberration alone. Piston is NULLED not swept: "
    "no_depth_piston=True with the measured beam (fill 0.65, center [-0.0689, 0.0118]) and "
    "piston=0, so the spherical is the only residual -- at no_depth_piston=False the canonical Z4 "
    "would inject 0.336*4 = 1.34 rad/step, past the ~1.0-1.2 rad/step loss onset, and floor both "
    "branches. reverse_zernike=False (a static corrective aberration keeps its sign on the return "
    "leg). nsteps=50, step_period_ms=3.0 (101 frames, 303 ms motion), precompute + precompute_host "
    "on (distortion forces the deduped float build, cache bypassed). Array 33x33_feedback11, "
    "z4 = loading_defocus = -4."
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (the triangle returns to the source sites). Name =
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build():
    """Build (do NOT submit) the 2-D [step_size x C12] ScanGroup. Returns ``(seq_name, g)``."""
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
    rk.nsteps = NSTEPS
    rk.step_period_ms = PERIOD_MS

    # dim 1: signed depth step (scalar + depth=True -> pure Z4 defocus ping-pong).
    rk.extras.step_size.scan(1, list(STEP_SIZES))
    rk.extras.depth = True
    # dim 2: corrective primary spherical, scalar -> distortion_zernike[12] per shot.
    rk.extras.distortion_z12.scan(2, list(C12_VALUES))
    # Static correction: SAME sign on the outward and return legs (the return leg retraces the
    # same +z displacements). Also keeps the unique-frame count at nsteps+1 rather than ~2x.
    rk.extras.reverse_zernike = False

    # Piston NULLED so the spherical is the only residual. All three EXPLICIT -- server extras are
    # sticky across scans in a backend session and the intervening depth x piston maps ran with
    # no_depth_piston=False and a swept piston.
    rk.extras.no_depth_piston = True
    rk.extras.depth_fill_frac = FILL_FRAC
    rk.extras.depth_fill_center = list(FILL_CENTER)
    rk.extras.piston = 0.0

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1500                       # -> ceil(1500/123) = 13 passes = 1599 shots
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def PPGSphericalNullSweep(url=None, reps=None):
    seq_name, g = build()
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="PPGSphericalNullSweep",
                      description=RUN_DESC, **opts)
    print("submitted PPGSphericalNullSweep -> descriptor id %s (%d step_sizes x %d C12 = %d pts; "
          "nsteps=%d, period=%.3f ms, no_depth_piston=True fill %.2f; url=%s)"
          % (did, len(STEP_SIZES), len(C12_VALUES), g.nseq(), NSTEPS, PERIOD_MS, FILL_FRAC,
             url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="pingponggrating spherical (Z12) null sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    if args.dry_run:
        _seq, _g = build()
        print("seq=%s  nseq=%d  nsteps=%d (%d frames)\nsteps=%s\nC12=%s"
              % (_seq, _g.nseq(), NSTEPS, 2 * NSTEPS + 1, STEP_SIZES, C12_VALUES))
    else:
        PPGSphericalNullSweep(url=args.url, reps=args.reps)
