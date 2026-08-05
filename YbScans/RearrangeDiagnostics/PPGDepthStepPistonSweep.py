"""PPGDepthStepPistonSweep.py -- pingponggrating DEPTH 2-D sweep: step size x piston phase.

    dim 1: extras.step_size  = signed DEPTH step (radians of PV ANSI Z4 per frame), 19 values
           -5 .. -1 (0.5 spacing), 0, +1 .. +5  -- scalar => depth=True => pure Z4 defocus ping-pong
    dim 2: extras.piston     = commanded per-step uniform SLM phase, 11 pts evenly on [0, 2*pi]
           (0 and 2*pi both included -- they wrap to the same command, so the two end columns are a
           built-in consistency check)

    => 19 x 11 = 209 points, survival img1 -> img2 at the SOURCE sites (pingponggrating moves the
    LOADED tweezers out-and-back, no target assignment, so this is pure transit survival).

WHY: ``no_depth_piston`` is deliberately OFF here, so the depth map is the canonical ANSI Z4
``2*rho^2 - 1``.  Over the FULL SQUARE panel that map has spatial mean ~ +0.336, so it already
injects ``+0.336 * step_size`` rad/step of uniform (piston) phase; the commanded ``piston`` adds on
top, giving a net uniform change per step of ``piston + 0.336 * step_size``.  The 07-22 heating
campaign showed the pure-piston transient is what kills atoms (benign <= 0.8 rad/step, dead by
3*pi/4).  So this map should show a survival RIDGE along ``piston = -0.336 * step_size (mod 2*pi)``
-- i.e. the piston column that cancels the map's own mean at each step size.  The ridge slope
directly calibrates the effective (beam-weighted) panel mean ``c`` that ``no_depth_piston`` /
``depth_fill_frac`` are supposed to subtract: ``weighted_mean = c_used - k`` for ridge slope ``k``.

nsteps=20, step_period_ms=3.0 (the 07-22 "geometric" pacing -- >= 1.4 ms removes the LC-settle
artefacts, so the loss seen here is real transit/piston loss, not settle).  return=True (default)
=> 2*20+1 = 41 frames per shot, 41 x 3 ms = 123 ms of motion.
``precompute`` + ``precompute_host`` ON so the paced loop is write-only and 3 ms is really realised.

Array / focal plane MATCH the live 2026-07-28 production config: 33x33_feedback11, z4 =
loading_defocus = -4, enhanced (BlueLAC) loading.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGDepthStepPistonSweep.py
    python YbScans/RearrangeDiagnostics/PPGDepthStepPistonSweep.py --reps 7
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
DEFOCUS = -4.0                      # rearrange z4 == rp.loading_defocus (same focal plane)

NSTEPS = 20
PERIOD_MS = 3.0
N_PISTON = 11

# dim 1: signed depth step (PV ANSI Z4 rad/frame). 0 appears ONCE.
_STEP_ABS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
STEP_SIZES = [-a for a in reversed(_STEP_ABS)] + [0.0] + list(_STEP_ABS)   # 19 values, ascending
# dim 2: commanded per-step piston, evenly on [0, 2*pi] inclusive.
PISTONS = [round(2.0 * math.pi * i / (N_PISTON - 1), 6) for i in range(N_PISTON)]

RUN_DESC = (
    "pingponggrating DEPTH 2-D map: signed step_size (PV ANSI Z4 rad/frame, -5..+5, 19 values) x "
    "commanded piston (11 pts on [0,2pi]). nsteps=20, step_period_ms=3.0, return trip (41 frames), "
    "precompute + precompute_host on, no_depth_piston OFF (canonical 2*rho^2-1, panel mean ~+0.336 "
    "=> intrinsic +0.336*step_size rad/step of piston). Looking for the survival ridge at "
    "piston = -0.336*step_size (mod 2pi): its slope calibrates the beam-weighted panel mean that "
    "no_depth_piston/depth_fill_frac subtract. Array 33x33_feedback11, z4 = loading_defocus = -4."
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (pingponggrating returns to the source sites), so
    img1 and img2 are both detected against 33x33_feedback11's registry grid + thresholds. The
    name is the phase-file BASENAME -- what the registry / ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build():
    """Build (do NOT submit) the 2-D [step_size x piston] ScanGroup. Returns ``(seq_name, g)``."""
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

    # Scalar step_size + depth=True -> pure Z4 defocus ping-pong (NOT the xyz 3-vector path;
    # a list-valued swept axis would break the lab-side N-D scan grid).
    rk.extras.step_size.scan(1, list(STEP_SIZES))
    rk.extras.depth = True
    rk.extras.piston.scan(2, list(PISTONS))

    # OFF per request: keep the canonical ANSI Z4 (2*rho^2 - 1) with its +0.336 panel mean, so the
    # intrinsic per-step piston is part of what we are mapping. Set EXPLICITLY -- the server extras
    # are sticky across scans in one backend session and default to True since 2026-07-15.
    rk.extras.no_depth_piston = False
    # depth_fill_frac / depth_fill_center are only read when no_depth_piston is True; left unset.

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (matches live production today)
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1500                       # ~1500 shots -> ceil(1500/209) = 8 passes
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def PPGDepthStepPistonSweep(url=None, reps=None):
    seq_name, g = build()
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="PPGDepthStepPistonSweep",
                      description=RUN_DESC, **opts)
    print("submitted PPGDepthStepPistonSweep -> descriptor id %s (%d steps x %d pistons = %d pts; "
          "nsteps=%d, period=%.3f ms, no_depth_piston=False; url=%s)"
          % (did, len(STEP_SIZES), len(PISTONS), g.nseq(), NSTEPS, PERIOD_MS, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="pingponggrating depth step x piston 2-D sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    if args.dry_run:
        _seq, _g = build()
        print("seq=%s  nseq=%d  steps=%s  pistons=%s"
              % (_seq, _g.nseq(), STEP_SIZES, PISTONS))
    else:
        PPGDepthStepPistonSweep(url=args.url, reps=args.reps)
