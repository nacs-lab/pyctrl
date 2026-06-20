"""SLMRearrangement3DScan.py -- 3-D (multi-layer) SLM-rearrangement scan (pyctrl).

3-D sibling of ``SLMRearrangementScan.py``. Builds the single-round SLM-rearrangement ScanGroup
(seq = ``RearrangeCommSeq``) and submits it to the RUNNING pyctrl backend over ZMQ. The ONLY
differences from the 2-D scan are the bits that make the SLM server derive + rearrange the array
PER AXIAL PLANE instead of as one flat 2-D grid:

  1. ``warmup_kwargs.extras.planes_z_rad`` -- the axial layer depths (radians of the ANSI
     ``2*rho^2-1`` defocus map; 1 rad ~ 3.26 um). The server flattens this into the dequeue
     ``setup_rearrangement`` body, where ``_resolve_grid`` reads it as a sticky extra and runs the
     3-D grid derivation (``_derive_grid_3d_impl``: refocus to each plane, extract per layer) for
     BOTH the init and target grids. Absent -> byte-identical 2-D derivation. (A shared
     ``planes_z_rad`` covers both grids; if the init and target arrays have DIFFERENT planes the
     server also accepts per-grid ``init_grid_planes_z_rad`` / ``target_grid_planes_z_rad`` -- we
     emit those automatically when INIT and TARGET planes differ.)
  2. ``imagePatternsJson`` declares ``planes_z_rad`` per camera frame, so the LAB-side atom
     detection (the bitstring fed to ``rearrange``) is per-plane too and its site count + ordering
     match the server's 3-D ``init_grid``. (Same per-plane-detection path as
     ``TwoLayer2x15ImagingScan.py``.)
  3. ``runp().isRearrange = 1`` -- forces the dequeue setup + per-shot rearrange path even though
     the model is left BLANK (the run-loop normally infers "this is a rearrangement scan" from a
     non-empty ``warmup_kwargs.model_filename``; see ``runner._is_rearrange_scan``).

Everything else (the scan-long ``slm`` lock, per-shot ``compute`` lock, setup/reload/rearrange,
img1->bits->rearrange->img2->update flow) is the unchanged ``RearrangeCommSeq`` machinery -- the
3-D nature lives entirely in the grid the server derives and the bits the lab detects.

  >> MODEL IS LEFT BLANK ON PURPOSE <<
  Set MODEL_FILENAME below to your 3-D checkpoint when ready (e.g. an experiment_3d model). While
  it is "", no model_filename is sent: the dequeue setup still derives the 3-D grids (no model
  needed for derivation), the scan-long lock + loading phase still write, and detection still
  runs -- but the per-shot ``rearrange`` call will error ("no model") until you fill this in. So
  you can dry-run the loading/detection/grid-derivation half today and flip on rearrangement by
  editing one line.

Run it:
    cd pyctrl
    python YbScans/SLMRearrangement3DScan.py
    python YbScans/SLMRearrangement3DScan.py --reps 1
    python YbScans/SLMRearrangement3DScan.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url, the SLM server must be reachable (the
scan-long slm lock is mandatory), AND the server must be running a 3-D-capable build (the
``planes_z_rad`` grid-derivation path). On a pre-3-D server, planes_z_rad is ignored and you get a
flat 2-D derivation instead -- check the server log for a "3D[Np]" tag on the derived grid.

VALIDATE BEFORE TRUSTING (3-D-specific): confirm the lab-side detection site count + ordering
match the server's 3-D init_grid. The server sorts 3-D sites plane-by-plane (see
``_derive_grid_3d_impl``); the lab detector orders sites from the per-pattern registry (same server
extraction, same ``order``), so they SHOULD agree -- but a mismatch silently scrambles the
``bits[i]`` -> site mapping, so eyeball it on the first run (``--isinit``-style raw-image check via
TwoLayer2x15ImagingScan, then a low-rep run here).
"""

import argparse
import json
import os
import sys


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# Initial (loading) and final (target) SLM patterns, resolved to a server-side phase + baked
# Zernike + axial planes by _pattern_cfg below (port of ybLoadingPatternCfg.m's 3-D entry). For a
# plain "fill the same multi-layer array" rearrangement leave them equal. Default: the 2-layer
# 15x15 diamond (two planes at +-2.5 um).
INIT_PATTERN = "2x15x15_xyoffset_5um"
TARGET_PATTERN = "2x15x15_xyoffset_5um"
# ------------------------------------------------------------------------------------ #

# 3-D rearrangement checkpoint (server-side path, relative to the SLM PC's slm/ project root).
# Verified present on the live server 2026-06-13 via /eval.
MODEL_FILENAME = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"

# ANSI z4 (rad) written for the WHOLE scan: drops the stack MIDPLANE onto the science camera (the
# plane the global SLM->camera affine is calibrated against). The two layers then sit at
# loading_defocus +- planes_z_rad. -5 is the established 2-D/midplane value (see
# TwoLayer2x15ImagingScan.py). The per-layer depths are declared via planes_z_rad, RELATIVE to
# this midplane -- NOT added here.
LOADING_DEFOCUS = -5

# Cross-plane xy-dedup radius (knm-1024 px) for the 3-D grid derivation. The two layers are only
# +-2.5 um apart (within DOF) but xy-offset, so each spot clears the global amplitude threshold in
# BOTH refocus planes and is double-counted (876 sites = 2x ~445). dedup_xy_knm keeps the brightest
# detection per xy-cluster -> the true ~445 sites with correct per-plane labels. Validated
# 2026-06-13 via /eval: radius 2-15 all give 445 = [224, 221] (insensitive in that range).
# WIRED 2026-06-13: _derive_grid_3d_impl resolves the dedup radius as explicit-arg ->
# setup extra 'dedup_xy_knm' (this) -> module default DEDUP_XY_KNM_OVERRIDE (6.0). The same dedup
# is applied to the lab registry derive, so the detection grid + server init_grid share the 445
# per-plane order. (For a future same-(x,y) bifocal array set this to 0 to disable.)
DEDUP_XY_KNM = 6


def _pattern_cfg(name):
    """Pattern name -> {phase_path, baked_zernike, legacy, planes_z_rad} (port of the 3-D entries
    of ybLoadingPatternCfg.m). ``planes_z_rad`` is [] for a flat 2-D pattern, non-empty (axial
    layer depths in ANSI 2*rho^2-1 radians) for a multi-layer array."""
    table = {
        # 3-D (multi-layer) arrays: planes_z_rad declares the axial layer depths.
        # 2-layer 15x15 diamond; zernike-free base; layers +-2.5 um = +-0.768 rad, 5 um xy offset.
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, 0], [-0.768, 0.768]),

        # Flat 2-D patterns (planes_z_rad empty) -- usable as a degenerate single-plane target, or
        # to mix a 2-D load with a 3-D target. Mirror SLMRearrangementScan.py's table.
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json(init_cfg, target_cfg):
    """Build the explicit per-frame detection declaration (img1 = loading/INIT, img2 = post-
    rearrangement/TARGET), each carrying its ``planes_z_rad`` so the lab-side detection extracts
    per-plane and its bit ordering matches the server's 3-D grid. Explicit imagePatternsJson wins
    over the runner's auto-synthesis (which does NOT know about planes_z_rad)."""
    def _item(cfg, name):
        # order="col_up" == the server's setup_rearrangement sweep_order default (the camera-matched
        # frame). Keeping the registry record in the SAME order as the server init_grid makes the
        # record/display/thresholds consistent with the server; the detector ALSO consumes the
        # server init_grid directly (single source), so this is belt-and-suspenders.
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


def SLMRearrangement3DScan(url=None, reps=None):
    """Build + submit the single-round 3-D SLM rearrangement scan. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1

    if not init_cfg["planes_z_rad"] and not target_cfg["planes_z_rad"]:
        raise ValueError(
            "SLMRearrangement3DScan: neither INIT_PATTERN (%r) nor TARGET_PATTERN (%r) declares "
            "planes_z_rad -- this would be a flat 2-D run. Use SLMRearrangementScan.py for 2-D, or "
            "pick a multi-layer pattern here." % (INIT_PATTERN, TARGET_PATTERN))

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    # Model left BLANK on purpose (see module docstring): only send model_filename when set, so the
    # server keeps whatever it has cached (likely none) and grid derivation still runs without one.
    if MODEL_FILENAME:
        rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = init_cfg["baked_zernike"]
    rp.warmup_kwargs.extras.final_phase_zernike = target_cfg["baked_zernike"]

    # ---- 3-D opt-in: axial layer depths -> per-plane grid derivation on the server ----------
    # Flattened into the dequeue setup_rearrangement body; _resolve_grid reads it (sticky) and
    # runs _derive_grid_3d_impl for the derived grids. A SHARED planes_z_rad covers both init and
    # target; if they differ we send per-grid keys instead (the server prefers
    # init_grid_planes_z_rad / target_grid_planes_z_rad over the shared one).
    if init_cfg["planes_z_rad"] == target_cfg["planes_z_rad"]:
        rp.warmup_kwargs.extras.planes_z_rad = init_cfg["planes_z_rad"]
    else:
        # Per-grid planes (3-D load -> differently-stacked 3-D target, or a 2-D <-> 3-D mix). An
        # empty list = that grid is plain 2-D.
        rp.warmup_kwargs.extras.init_grid_planes_z_rad = init_cfg["planes_z_rad"]
        rp.warmup_kwargs.extras.target_grid_planes_z_rad = target_cfg["planes_z_rad"]

    # Cross-plane xy-dedup so the within-DOF ghost (one spot detected in both refocus planes) is
    # collapsed -> the true 445 sites in per-plane order (instead of 876). The server applies the
    # same dedup to the lab registry derive, so detection grid order == server init_grid order.
    rp.warmup_kwargs.extras.dedup_xy_knm = DEDUP_XY_KNM

    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) -----------------------------
    g().rearrange_kwargs.nsteps = 200
    g().rearrange_kwargs.step_period_ms = 1
    g().rearrange_kwargs.protocol = "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    # 3-D target pattern: fill the FRONT layer (every site in the front plane; empties the back).
    # Requires a 3-D target grid -- provided by the shared planes_z_rad above. front_layer_sign
    # picks which z-sign is "front" (server default = +1; set extras.front_layer_sign = -1 to flip).
    g().rearrange_kwargs.extras.pattern = "front_layer"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.hw_sequence = False
    # Model-frame carrier defocus: MATCH LOADING_DEFOCUS so the model frames sit at the SAME
    # midplane as the loaded stack (no transit-defocus mismatch between the load and the
    # rearrangement frames). The per-LAYER depth is added on top of this carrier via planes_z_rad;
    # this just pins the stack centre to the load plane (== rp.loading_defocus). Mirrors the 2-D
    # scan's "extras.z4 = loading_defocus (same focal plane)".
    g().rearrange_kwargs.extras.z4 = LOADING_DEFOCUS
    # Per-bseq cooling/imaging overlay (expConfig ByPattern): bseq1 (initial dense-load image) is
    # tagged with initial_pattern, bseq2 (post-rearrangement image) with final_pattern. No effect
    # until ByPattern is populated for these names.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params (runp) ------------------------------------------------------------
    # Force the rearrangement path even with a BLANK model (the run loop otherwise infers it from a
    # non-empty model_filename; see runner._is_rearrange_scan).
    rp.isRearrange = 1
    rp.NumPerGroup = 100000
    rp.loading_defocus = LOADING_DEFOCUS
    rp.NumImages = n_rounds + 1                  # img1 (load) + one frame per round (img2 = target)
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    # ---- per-frame detection declaration (per-plane; matches the server's 3-D grid) ----
    rp.imagePatternsJson = _image_patterns_json(init_cfg, target_cfg)

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMRearrangement3DScan", **opts)
    print("submitted SLMRearrangement3DScan -> descriptor id %s (url=%s, model=%s, planes=%s)"
          % (did, url or "default", MODEL_FILENAME or "<BLANK>",
             init_cfg["planes_z_rad"] or target_cfg["planes_z_rad"]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit SLMRearrangement3DScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    args = ap.parse_args()
    SLMRearrangement3DScan(url=args.url, reps=args.reps)
