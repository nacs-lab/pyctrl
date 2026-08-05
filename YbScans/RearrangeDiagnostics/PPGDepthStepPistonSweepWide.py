"""PPGDepthStepPistonSweepWide.py -- WIDE-range re-run of the pingponggrating depth x piston map.

Same protocol/config as :mod:`PPGDepthStepPistonSweep` (job 244, sid 20260728155015) -- back on the
WGS **grating** path, NOT the 3-D model -- but the depth axis is extended to +-10 rad instead of
+-5, trading fineness at small |step| for reach:

    dim 1: extras.step_size = signed DEPTH step (radians of PV ANSI Z4 per frame), 19 values
           -10, -8..-1 (in ones), 0, +1..+8 (in ones), +10
    dim 2: extras.piston = commanded per-step uniform SLM phase, 11 pts evenly on [0, 2*pi]
           (0 and 2*pi both included -- they wrap to the same command, a built-in cross-check)

    => 19 x 11 = 209 points, survival img1 -> img2 at the source sites.

The narrow map (job 244, |step| <= 5 at 0.5 spacing below 1) already resolves the small-|step|
behaviour; this one asks where the depth reach actually ends and whether the piston ridge
(``piston = -0.336 * step_size (mod 2*pi)``, from the canonical Z4 map's +0.336 panel mean with
``no_depth_piston`` OFF) keeps tracking out to +-10 rad -- i.e. whether it wraps 2*pi more than
once across the range, which it must if the intrinsic piston really is linear in step_size
(0.336 * 10 = 3.36 rad, so the ridge should cross the piston axis roughly twice per direction).

nsteps=20, step_period_ms=3.0, return trip => 2*20+1 = 41 frames = 123 ms of motion.
``precompute`` + ``precompute_host`` ON so the paced loop is write-only and 3 ms is realised.
Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus = -4.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGDepthStepPistonSweepWide.py
    python YbScans/RearrangeDiagnostics/PPGDepthStepPistonSweepWide.py --dry-run
"""

import argparse
import json
import math
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0

NSTEPS = 20
PERIOD_MS = 3.0
N_PISTON = 11

# dim 1: signed depth step (PV ANSI Z4 rad/frame). Coarse (unit) spacing 1..8, then 10; 0 once.
_STEP_ABS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
STEP_SIZES = [-a for a in reversed(_STEP_ABS)] + [0.0] + list(_STEP_ABS)   # 19 values, ascending
# dim 2: commanded per-step piston, evenly on [0, 2*pi] inclusive.
PISTONS = [round(2.0 * math.pi * i / (N_PISTON - 1), 6) for i in range(N_PISTON)]

RUN_DESC = (
    "WIDE pingponggrating DEPTH 2-D map (WGS grating path, not the 3-D model): signed step_size "
    "(PV ANSI Z4 rad/frame) -10, -8..-1, 0, +1..+8, +10 -- 19 values, unit spacing -- x commanded "
    "piston 11 pts on [0,2pi]. Extends job 244 (sid 20260728155015, |step|<=5) to +-10 rad to find "
    "where the depth reach ends and to test whether the piston ridge keeps tracking "
    "piston = -0.336*step_size (mod 2pi) far enough to wrap 2pi more than once (0.336*10 = 3.36 rad). "
    "nsteps=20, step_period_ms=3.0, return trip (41 frames), precompute + precompute_host on, "
    "no_depth_piston OFF (canonical 2*rho^2-1). Array 33x33_feedback11, z4 = loading_defocus = -4."
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


def build(nsteps=NSTEPS):
    """Build (do NOT submit) the 2-D [step_size x piston] ScanGroup. Returns ``(seq_name, g)``.

    ``nsteps`` is the only knob: the same wide depth x piston grid can be run at a different
    number of triangle steps (2*nsteps+1 frames) to separate per-step loss from cumulative
    loss -- the depth reach should be set by the per-step |step_size|, the total heating by
    nsteps."""
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
    rk.step_period_ms = PERIOD_MS

    # Scalar step_size + depth=True -> pure Z4 defocus ping-pong (NOT the xyz 3-vector path).
    rk.extras.step_size.scan(1, list(STEP_SIZES))
    rk.extras.depth = True
    rk.extras.piston.scan(2, list(PISTONS))

    # OFF: keep the canonical ANSI Z4 (2*rho^2 - 1) with its +0.336 panel mean, so the intrinsic
    # per-step piston is part of what is mapped. EXPLICIT -- server extras are sticky across scans
    # in one backend session and this defaults to True since 2026-07-15. Also re-asserts it after
    # the intervening 3-D `pingpong` scans, which do not pass it at all.
    rk.extras.no_depth_piston = False

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1500                       # -> ceil(1500/209) = 8 passes = 1672 shots
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def PPGDepthStepPistonSweepWide(url=None, reps=None, nsteps=NSTEPS):
    seq_name, g = build(nsteps=nsteps)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    # Label carries a non-default nsteps so the queue / dashboard distinguishes the runs.
    label = "PPGDepthStepPistonSweepWide"
    if int(nsteps) != NSTEPS:
        label += "_n%d" % int(nsteps)
    desc = RUN_DESC
    if int(nsteps) != NSTEPS:
        desc += (" THIS RUN: nsteps=%d (not the %d of the companion run) -- %d frames, %.0f ms of "
                 "motion; pairs with the nsteps=%d run to separate per-step loss from cumulative "
                 "loss at the same per-step |step_size|."
                 % (int(nsteps), NSTEPS, 2 * int(nsteps) + 1,
                    (2 * int(nsteps) + 1) * PERIOD_MS, NSTEPS))
    did = ybStartScan(seq_name, g, url=url, label=label, description=desc, **opts)
    print("submitted %s -> descriptor id %s (%d steps x %d pistons = %d "
          "pts; nsteps=%d, period=%.3f ms, no_depth_piston=False; url=%s)"
          % (label, did, len(STEP_SIZES), len(PISTONS), g.nseq(), int(nsteps), PERIOD_MS,
             url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WIDE pingponggrating depth step x piston 2-D sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--nsteps", type=int, default=NSTEPS,
                    help="triangle steps each way (default %d -> %d frames)"
                         % (NSTEPS, 2 * NSTEPS + 1))
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    if args.dry_run:
        _seq, _g = build(nsteps=args.nsteps)
        print("seq=%s  nseq=%d  nsteps=%d (%d frames)\nsteps=%s\npistons=%s"
              % (_seq, _g.nseq(), args.nsteps, 2 * args.nsteps + 1, STEP_SIZES, PISTONS))
    else:
        PPGDepthStepPistonSweepWide(url=args.url, reps=args.reps, nsteps=args.nsteps)
