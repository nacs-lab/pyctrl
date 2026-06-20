"""SLMRearrangement3DBetaZweightSweep.py -- 3-D (2x15x15) prob_hungarian beta x z_weight sweep.

Based on ``SLMRearrangement3DScan.py`` (single-round 3-D SLM rearrangement, seq = ``RearrangeCommSeq``,
2-layer 15x15 diamond) with the imaging-weighted Hungarian turned ON and a 2-D parameter sweep:

  * prob_hungarian = True (FIXED on)            -- the imaging-weighted assignment is active.
  * nsteps = 66, step_period_ms = 1             -- (per request) shorter transit than the 2-D scan.
  * prob_hungarian_beta  : SWEEP [0, 1, 10, 100, 1000, 10000, 100000, 1e6, 1e7]   (dim 1)
        the weight on the -log(p) confidence penalty. beta=0 reproduces the plain (distance-only)
        3-D Hungarian -> it is the in-sweep CONTROL. Larger beta trades transit distance to avoid
        low-confidence atoms (only matters under surplus, n_loaded > n_active_targets).
  * z_weight_px_per_rad  : SWEEP [0, 0.5, 1, 1.5, 2, 3, 5]              (dim 2)
        the ratio between z (defocus radian) units and xy (knm-1024 px) units in the 3-D Hungarian
        cost. 0 = ignore z in the assignment (pure xy); larger = z separations dominate. The 3-D
        assignment cost is dy^2 + dx^2 + (z_weight * dz_rad)^2. Server default = 1.0.

  9 betas x 7 z_weights = 63 points; 50 PASSES/point -> 3150 TOTAL shots (reps != shots -- one
  pass = all 63 points; see the SHOT BUDGET block below). Scramble=1 interleaves all 63 points
  each pass for a clean grid (no time-correlated bias).

  beta=0 column AND z_weight=0 row are the two controls; the goal is to find the best (beta,
  z_weight) for 3-D rearrangement survival.

This is the standard ``SLMRearrangement3DScan`` machinery otherwise -- per-plane derivation
(planes_z_rad), per-plane detection (imagePatternsJson), the front_layer target. The 3-D path uses
the GPU chirp-splat target builder (``gpu_target_graph=True``, set explicitly here -- the dispatcher
REFUSES to run a 3-D run through the 2-D fallback). prob_hungarian's per-atom -beta*log(p) cost is
folded into the 3-D assignment exactly as in 2-D (assign_pairs_3d loaded_cost_offset, server dee4fdb).

Run it:
    cd pyctrl
    python YbScans/SLMRearrangement3DBetaZweightSweep.py            # 3150 shots (50 passes x 63 pts)
    python YbScans/SLMRearrangement3DBetaZweightSweep.py --reps 1   # smoke test: 1 pass = 63 shots
    python YbScans/SLMRearrangement3DBetaZweightSweep.py --url tcp://127.0.0.1:1408

Prereq: pyctrl backend running at --url; SLM server reachable AND running the dee4fdb build (verify
``prob_hungarian`` in GET /slm/protocols/rearrange) AND the 3-D build (planes_z_rad derivation +
experiment_3d model). NOTE: as of 2026-06-13 a live on-rig 3-D rearrange shot was still UNTESTED --
smoke-test with --reps 1 and eyeball the per-plane detection + first survival before the full sweep.
"""

import argparse
import json
import os
import sys


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# 2-layer 15x15 diamond (two planes at +-2.5 um). Equal init/target -> consolidate to the front layer.
INIT_PATTERN = "2x15x15_xyoffset_5um"
TARGET_PATTERN = "2x15x15_xyoffset_5um"
# ------------------------------------------------------------------------------------ #

# 3-D rearrangement checkpoint (server-side path). Verified present on the live server 2026-06-13.
MODEL_FILENAME = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"

# ANSI z4 (rad) for the WHOLE scan: drops the stack MIDPLANE onto the science camera. Layers sit at
# loading_defocus +- planes_z_rad. -5 is the established 2-D/midplane value.
LOADING_DEFOCUS = -5

# Cross-plane xy-dedup radius (knm-1024 px) for the 3-D grid derivation (validated 2-15 -> 445 sites).
DEDUP_XY_KNM = 6

# -------- THE SWEEP --------
# prob_hungarian_beta (dim 1) -- 9 values incl. the very-large end (1e6, 1e7).
BETA_VALUES = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
# z_weight_px_per_rad (dim 2).
Z_WEIGHT_VALUES = [0, 0.5, 1, 1.5, 2, 3, 5]

# ---- SHOT BUDGET -- READ CAREFULLY: reps != shots ---------------------------------------
#   * One PASS    = one sweep through ALL grid points = N_POINTS sequences.
#   * REPS_PER_POINT = number of PASSES = the ``rep`` opt = how many times EACH grid point runs.
#   * TOTAL SHOTS = REPS_PER_POINT * N_POINTS   (NOT REPS_PER_POINT alone, NOT NumPerGroup).
# The scan is submitted with rep=REPS_PER_POINT (the explicit pass-count override, which BYPASSES
# the NumPerGroup/nseqs formula -- see yb_start_scan.py / sequence_runner._build_scan_order), so it
# runs EXACTLY TOTAL_SHOTS then STOPS. NumPerGroup is ALSO set = TOTAL_SHOTS so the two agree and
# the dashboard's "shots scheduled" is honest. NEVER set NumPerGroup to a giant run-forever
# sentinel here: it only governs the count when ``rep`` is absent, but it pollutes the displayed
# total -- that trap is what made a 49-point x 50-pass sweep read as "1,000,000 shots".
REPS_PER_POINT = 50
N_POINTS = len(BETA_VALUES) * len(Z_WEIGHT_VALUES)   # 9 x 7 = 63 grid points (= nseqs per pass)
TOTAL_SHOTS = REPS_PER_POINT * N_POINTS              # 63 x 50 = 3150 shots


def _pattern_cfg(name):
    """Pattern name -> {phase_path, baked_zernike, legacy, planes_z_rad} (3-D entries of
    ybLoadingPatternCfg.m). planes_z_rad is [] for a flat 2-D pattern."""
    table = {
        # 2-layer 15x15 diamond; zernike-free base; layers +-2.5 um = +-0.768 rad, 5 um xy offset.
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, 0], [-0.768, 0.768]),
        # Flat 2-D patterns (planes_z_rad empty).
        "47x47_feedbackwarm3": ("phase/47x47_feedbackwarm3.pt", [0, 0, 0, 0, 0], []),
        "33x33_uniform":       ("phase/33x33_uniform.pt",       [0, 0, 0, 0, 0], []),
        "3270_z4eq4":          ("phase/3270_z4eq4.pt",          [0, 0, 0, 0, -4], []),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked, planes = table[name]
    return {"phase_path": path,
            "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked),
            "planes_z_rad": [float(z) for z in planes]}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json(init_cfg, target_cfg):
    """Per-frame detection declaration (img1 = INIT, img2 = TARGET), each carrying planes_z_rad so
    lab-side detection extracts per-plane and bit ordering matches the server's 3-D grid."""
    def _item(cfg, name):
        it = {"name": name,
              "base_phase_path": cfg["phase_path"],
              "order": "col_up",
              "legacy_zerniked": bool(cfg["legacy"])}
        if cfg["legacy"]:
            it["baked_zernike"] = cfg["baked_zernike"]
        if cfg["planes_z_rad"]:
            it["planes_z_rad"] = cfg["planes_z_rad"]
        return it
    return json.dumps([_item(init_cfg, INIT_PATTERN), _item(target_cfg, TARGET_PATTERN)])


def SLMRearrangement3DBetaZweightSweep(url=None, reps=None):
    """Build + submit the 3-D beta x z_weight rearrangement sweep. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1
    # Effective passes (per grid point). Default = REPS_PER_POINT (full run); --reps overrides
    # (e.g. --reps 1 for a 1-pass = N_POINTS-shot smoke test).
    eff_reps = REPS_PER_POINT if reps is None else int(reps)

    if not init_cfg["planes_z_rad"] and not target_cfg["planes_z_rad"]:
        raise ValueError("Neither INIT nor TARGET declares planes_z_rad -- this is a flat 2-D run.")

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = init_cfg["baked_zernike"]
    rp.warmup_kwargs.extras.final_phase_zernike = target_cfg["baked_zernike"]

    # 3-D opt-in: axial layer depths -> per-plane grid derivation. Shared planes (init == target).
    if init_cfg["planes_z_rad"] == target_cfg["planes_z_rad"]:
        rp.warmup_kwargs.extras.planes_z_rad = init_cfg["planes_z_rad"]
    else:
        rp.warmup_kwargs.extras.init_grid_planes_z_rad = init_cfg["planes_z_rad"]
        rp.warmup_kwargs.extras.target_grid_planes_z_rad = target_cfg["planes_z_rad"]
    rp.warmup_kwargs.extras.dedup_xy_knm = DEDUP_XY_KNM

    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs (g(); per-shot setup) ---------------------------------------
    g().rearrange_kwargs.nsteps = 66
    g().rearrange_kwargs.step_period_ms = 1
    g().rearrange_kwargs.protocol = "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.pattern = "front_layer"   # fill the front plane (3-D target)
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.gpu_target_graph = True   # 3-D REQUIRES the GPU chirp-splat builder
    g().rearrange_kwargs.extras.z4 = LOADING_DEFOCUS      # model frames at the load midplane
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- imaging-weighted Hungarian: ON, with the beta x z_weight grid -----------------
    g().rearrange_kwargs.extras.prob_hungarian = True     # FIXED on (not swept)
    # SWEPT params must be left UNSET to a scalar so ScanGroup accepts the .scan().
    g().rearrange_kwargs.extras.prob_hungarian_beta.scan(1, [float(b) for b in BETA_VALUES])
    g().rearrange_kwargs.extras.z_weight_px_per_rad.scan(2, [float(w) for w in Z_WEIGHT_VALUES])

    # ---- run params (runp) ------------------------------------------------------------
    rp.isRearrange = 1                           # force rearrange path (model IS set, but explicit)
    # = TOTAL shots for this run (eff_reps passes x N_POINTS). Set = the real total (NOT a
    # run-forever sentinel) so StackNum = ceil(NumPerGroup/nseqs) = eff_reps even if ``rep`` were
    # dropped, and the dashboard shows the honest shot count.
    rp.NumPerGroup = eff_reps * N_POINTS
    rp.loading_defocus = LOADING_DEFOCUS
    rp.NumImages = n_rounds + 1                  # img1 (load) + img2 (target)
    rp.Scramble = 1                              # interleave all 49 points each pass
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    rp.imagePatternsJson = _image_patterns_json(init_cfg, target_cfg)

    # rep = explicit PASS count (passes through the whole grid). Overrides the NumPerGroup formula,
    # so the run does EXACTLY eff_reps * N_POINTS shots then STOPS. ALWAYS set it.
    opts = {"rep": eff_reps}

    did = ybStartScan("RearrangeCommSeq", g, url=url,
                      label="SLMRearrangement3DBetaZweightSweep", **opts)
    print("submitted SLMRearrangement3DBetaZweightSweep -> descriptor id %s (url=%s, model=%s)"
          % (did, url or "default", MODEL_FILENAME))
    print("  sweep: %d betas x %d z_weights = %d points; %d passes/point -> %d TOTAL shots"
          % (len(BETA_VALUES), len(Z_WEIGHT_VALUES), N_POINTS, eff_reps, eff_reps * N_POINTS))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 3-D beta x z_weight rearrangement sweep to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="PASSES per grid point (default 50 -> 3150 shots = 50x63); "
                         "1 -> 63-shot smoke test. NOTE: reps are PASSES, not total shots.")
    args = ap.parse_args()
    SLMRearrangement3DBetaZweightSweep(url=args.url, reps=args.reps)
