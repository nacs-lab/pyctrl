"""SnakeRearrangementScan.py -- rearrange the 47x47_uniform array into the 'snake' target.

A snake-specific copy of SLMRearrangementScan.py. Same single-round RearrangeCommSeq flow
(at dequeue: grab the scan-long ``slm`` lock + write the 47x47 loading phase; per shot: grab
``compute``, push setup_rearrangement, detect img1 bits, rearrange into the target, store img1;
detect img2 bits, update_rearrange, store img2; release compute; keepalive slm).

Differences from SLMRearrangementScan:
  * INIT/TARGET loading pattern = 47x47_uniform (the currently-loaded array).
  * Rearrangement TARGET SHAPE = the 'snake' bitmap (slm/patterns/snake.txt on the server,
    47x47, 335 sites) via ``rearrange_kwargs.extras.pattern = "snake"``.
  * nsteps is SWEPT (axis 1); step_period_ms = 1 ms and precompute = True are pinned.
  * Model = sinc_3x3_experiment/direct (newest 2D uniform-array model, 2026-06-09).

Run it (pyctrl backend live; SLM server reachable -- the scan-long slm lock is mandatory):
    cd pyctrl
    python YbScans/SnakeRearrangementScan.py --reps 20
    python YbScans/SnakeRearrangementScan.py --url tcp://127.0.0.1:1408 --reps 20
"""

import argparse
import os
import sys


# --------------------------- PATTERN SELECTION ---------------------------- #
# Loading (trap) array -- both init and final are the 47x47 uniform array.
INIT_PATTERN = "47x47_uniform"
TARGET_PATTERN = "47x47_uniform"
# Rearrangement target SHAPE (slm/patterns/<name>.txt on the server).
REARRANGE_PATTERN = "snake"
# nsteps sweep (rearrangement movement steps); step_period pinned at 1 ms.
NSTEPS_SWEEP = [100, 140, 180]
# -------------------------------------------------------------------------- #

# Best/newest 2D uniform-array model (2026-06-09). NOT experiment_3d (3D/two-layer) and NOT the
# older experiment_sinc_ampmap_v3 (2026-04-14).
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_two_layer_dd10.0": ("phase/33x33_two_layer_dd10.0.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4": ("phase/3270_z4eq4.pt", [0, 0, 0, 0, -4]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SnakeRearrangementScan(url=None, reps=None):
    """Build + submit the single-round snake rearrangement scan. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (forwarded ONCE at dequeue with reset_params) -------------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = init_cfg["baked_zernike"]
    rp.warmup_kwargs.extras.final_phase_zernike = target_cfg["baked_zernike"]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs (per-shot setup) ---------------------------------------------
    g().rearrange_kwargs.nsteps.scan(1, [int(n) for n in NSTEPS_SWEEP])   # SWEEP nsteps
    g().rearrange_kwargs.step_period_ms = 1.0                              # pinned 1 ms
    g().rearrange_kwargs.protocol = "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.pattern = REARRANGE_PATTERN               # the snake bitmap
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True                         # pinned True (no sweep)
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params --------------------------------------------------------------------
    rp.NumPerGroup = 100000
    rp.loading_defocus = -5                        # ANSI z4; matched to extras.z4
    rp.NumImages = n_rounds + 1                    # img1 (load) + one frame per round
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1                       # hold slm lock for the whole scan

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SnakeRearrangementScan", **opts)
    print("submitted SnakeRearrangementScan -> descriptor id %s (url=%s, reps=%s, nseq=%d)"
          % (did, url or "default", reps, g.nseq()))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit SnakeRearrangementScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the nsteps sweep (finite -> writes Params; omit -> derived)")
    args = ap.parse_args()
    SnakeRearrangementScan(url=args.url, reps=args.reps)
