"""Pingpong3DDepthStepPistonSweep.py -- MODEL-DRIVEN 3-D re-run of the pingponggrating depth map.

Re-runs the 2026-07-28 15:50 grating scan (``sid=20260728155015``, job 244,
:mod:`PPGDepthStepPistonSweep`) on the SAME 2-D sweep grid, but the axial motion is produced by
the 3-D SLMnet model (``pingpong`` protocol, ``depth3d``) instead of a Z4 blaze/defocus grating:

    dim 1: extras.step_size_z = signed DEPTH step, 19 values -5 .. -1 (0.5 spacing), 0, +1 .. +5
           SAME units as the grating scan's step_size: radians of ANSI Z4 PV (2*rho^2 - 1),
           ~0.8 um/rad, so the two scans' axes transfer 1:1.
    dim 2: extras.piston = per-triangle-step uniform phase, 11 pts evenly on [0, 2*pi] inclusive
           (applied to every tweezer, zero at the endpoints/bookends).

    => 19 x 11 = 209 points, survival img1 -> img2 at the source sites.

The full 19 x 11 grid is deliberately kept identical to job 244's (the request enumerated a
17 x 10 subset -- the values that survived its analysis -- but the point is a like-for-like
comparison against the grating map, so nothing is dropped here).

Server-side contract (SLM server, 2026-07-28 update to the ``pingpong`` protocol):
  * ``step_size_z`` / ``depth3d`` / ``piston`` are NEW kwargs; the edit hot-reloads on the first
    ``setup_rearrangement`` call, so ``GET /slm/protocols/pingpong`` only lists them after one
    setup has run. Verify they are listed once this scan is running.
  * ``depth3d=True`` is CONSTANT across shots so the ``step_size_z = 0`` point stays on the 3-D
    model path (a proper control); without it that point silently falls back to the 2-D model
    and is not comparable.
  * ``full_n=True`` -- all sites move, matching the grating's rigid full-far-field shift (avoids
    the sparse-N photon-heating confound).
  * ``model_filename`` is the v1 3-D (direct3d) checkpoint. NOT ``experiment_3d_v2`` -- its
    training encoding does not match the server's runtime builder yet. Passing model_filename
    bumps the run-id and RESETS the sticky param cache, which is why every parameter below is
    passed explicitly.
  * ``gpu_target_graph`` is deliberately NOT passed: its default True is required (the dispatcher
    errors rather than silently falling back when a 3-D protocol has no GPU builder).
  * ``depth`` / ``no_depth_piston`` are grating-only and are NOT passed.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/Pingpong3DDepthStepPistonSweep.py
    python YbScans/RearrangeDiagnostics/Pingpong3DDepthStepPistonSweep.py --dry-run
"""

import argparse
import json
import math
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
# v1 3-D (direct3d) checkpoint -- NOT experiment_3d_v2.
MODEL_FILENAME = "SLMnet/checkpoints/experiment_3d/models/base5x5/direct_best.pth"
DEFOCUS = -4.0          # -> zernike_coeffs [0,0,0,0,-4] (model) AND loading_zernike [0,0,0,0,-4]

NSTEPS = 10          # 2026-07-28 re-run: halved from the grating scan's 20 (-> 21 frames, 63 ms)
PERIOD_MS = 3.0
N_PISTON = 11

# dim 1: signed DEPTH step (PV ANSI Z4 rad/frame) -- identical axis to job 244's step_size.
_STEP_ABS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
STEP_SIZES_Z = [-a for a in reversed(_STEP_ABS)] + [0.0] + list(_STEP_ABS)   # 19, ascending
# dim 2: commanded per-step piston, evenly on [0, 2*pi] inclusive -- identical axis to job 244.
PISTONS = [round(2.0 * math.pi * i / (N_PISTON - 1), 6) for i in range(N_PISTON)]

RUN_DESC = (
    "3-D MODEL re-run of the pingponggrating depth map (sid 20260728155015 / job 244). protocol "
    "pingpong + depth3d=True + full_n=True with the v1 3-D direct3d checkpoint "
    "(experiment_3d/base5x5/direct_best.pth); step_size=0 (no lateral motion), axial motion from "
    "the model. Same 2-D grid as the grating scan: step_size_z -5..+5 (19 values, rad ANSI Z4 PV, "
    "~0.8 um/rad) x piston 11 pts on [0,2pi]. nsteps=10 (halved vs the nsteps=20 grating map and "
    "vs the first 3-D run, job 245 -- 21 frames / 63 ms of motion), step_period_ms=3.0, precompute + "
    "precompute_host on. Array 33x33_feedback11, loading_zernike = zernike_coeffs = [0,0,0,0,-4]. "
    "Point of comparison: does model-generated axial motion beat the Z4-grating ping-pong, and "
    "does the piston ridge survive when the depth is model-encoded rather than blazed?"
)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames are the SAME array (the triangle returns to the source sites). Name = the
    phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build():
    """Build (do NOT submit) the 2-D [step_size_z x piston] ScanGroup. Returns ``(seq_name, g)``."""
    _bootstrap()
    from scan_group import ScanGroup

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME     # bumps run-id -> resets sticky params
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

    # ---- rearrange_kwargs: model-driven 3-D pingpong -----------------------------------
    # EVERY parameter explicit -- the model_filename change resets the server's sticky cache.
    rk = g().rearrange_kwargs
    rk.protocol = "pingpong"
    rk.nsteps = NSTEPS
    rk.step_period_ms = PERIOD_MS
    rk.extras.n_rounds = 1

    rk.extras.step_size = 0.0                    # no lateral motion; pure depth
    rk.extras.step_size_z.scan(1, list(STEP_SIZES_Z))
    rk.extras.depth3d = True                     # CONSTANT: keeps step_size_z=0 on the 3-D path
    rk.extras.full_n = True                      # all sites move (matches the grating's rigid shift)
    rk.extras.piston.scan(2, list(PISTONS))

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                  # BlueLAC loading (as in job 244)
    rk.extras.z4 = DEFOCUS                       # -> zernike_coeffs [0,0,0,0,-4] (model frames)
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN
    rk.extras.RearrCoolAmp = 0                   # seq-side; = the RearrangeCommSeq default
    rk.extras.RearrCoolDet = 130000              # seq-side; = the RearrangeCommSeq default
    # NOT passed on purpose: depth / no_depth_piston (grating-only), gpu_target_graph (its
    # default True is required by the 3-D dispatcher).

    # ---- run params (runp) --------------------------------------------------------------
    rp.NumPerGroup = 1500                        # -> ceil(1500/209) = 8 passes = 1672 shots
    rp.loading_defocus = DEFOCUS                 # -> loading_zernike [0,0,0,0,-4]
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def Pingpong3DDepthStepPistonSweep(url=None, reps=None):
    seq_name, g = build()
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="Pingpong3DDepthStepPistonSweep",
                      description=RUN_DESC, **opts)
    print("submitted Pingpong3DDepthStepPistonSweep -> descriptor id %s (%d step_size_z x %d "
          "pistons = %d pts; nsteps=%d, period=%.3f ms, depth3d=True, full_n=True; url=%s)"
          % (did, len(STEP_SIZES_Z), len(PISTONS), g.nseq(), NSTEPS, PERIOD_MS, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="model-driven 3-D pingpong depth x piston 2-D sweep.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    if args.dry_run:
        _seq, _g = build()
        print("seq=%s  nseq=%d\nstep_size_z=%s\npistons=%s"
              % (_seq, _g.nseq(), STEP_SIZES_Z, PISTONS))
    else:
        Pingpong3DDepthStepPistonSweep(url=args.url, reps=args.reps)
