"""PipelineDemoScan.py -- run the server-side ``pipeline_demo`` protocol at its DEFAULTS.

``pipeline_demo`` (SLM server, ``tools/rearrange_protocols._protocol_pipeline_demo``; NOT in the
local SLMnet checkout -- inspected live via ``/eval`` 2026-07-30) is the full-pipeline
demonstrator: it tiles the array into 2x2 blocks and walks each of the four corner atoms of every
block along its OWN ``(dx, dy, dz)``, then STAYS there for the second image. There is no return
leg and no WGS bookend -- the camera sees the displaced state.

Key properties (from the protocol docstring):
  * Blocks come from the LATTICE geometry (init_info rows/cols), NOT from the per-shot loading, so
    the frame stack is identical every shot; the analysis post-selects fully-loaded blocks.
  * On the production 33x33 array (``phase/33x33_feedback11.pt``, 1068 spots, 21-site central
    hole) the tiling gives 248 blocks = 992 moved atoms + 76 held leftover sites.
  * ALWAYS a 3-D run (it emits an ``inter_z`` track even when every dz is 0), and it flips two
    dispatcher defaults ON for itself: ``wgs3d_warm=True`` (warm matched-filter 3-D WGS producer
    -- no 3-D checkpoint needed) and ``skip_final_phase=True`` (no bookend on top of frame N).

2026-07-30 STATUS -- the first run at FULL server defaults (job 390, data 20260730_163944) lost
EVERY atom: img1 627.3 atoms/shot -> img2 4.1 (0.38%, the noise floor), with no lattice anywhere
in the 2100x2100 frame. Crucially the 76 HELD non-block sites died too, and those never move
(hold_non_block=True pins them at init position/phase/depth in every frame) -- so the loss is
GLOBAL to the transit, not a consequence of how far the corners travelled. The run was otherwise
clean (99 shots, no runner/server errors, detector 1068/1068). This file is now the FOLLOW-UP:
displacement scaled down 5x + an nsteps sweep whose nsteps=0 point is a no-motion control, which
separates "the atoms could not follow the move" from "the 3-D warm-WGS / skip_final_phase frame
path does not hold atoms at all".

WHAT nsteps DOES: it SPLITS the fixed total displacement -- frame k sits at fraction k/nsteps of
``block_displacement``, frame nsteps is fully displaced. It does NOT repeat the displacement. So
sweeping it varies per-frame step size AND transit duration (nsteps * step_period_ms) together.

PARAMETERS (block_displacement + nsteps set explicitly below; everything else = server default):
    block_displacement = the server default scaled DOWN 5x
                       = [[1.6, 1.6, 0], [0, 0, -1], [2.2, 0, 0], [1, 1, 1]]
                         rows = corners in the FIXED order (TR, TL, BL, BR),
                         cols = (dx, dy, dz); dx/dy in knm-1024 px, dz in rad of PV quad-defocus
                         (~0.798 um/rad).  Server default = [[8,8,0],[0,0,-5],[11,0,0],[5,5,5]]
    nsteps             = SWEPT over [0, 1, 3, 5, 10, 20, 50] (axis 1)
    block_rot90        = 0        (corner-ROLE rotation; vectors themselves are never rotated)
    block_row_origin   = 0        (2x2 tiling starts at lattice row 0)
    block_col_origin   = 0        (   "        "         "     column 0)
    hold_non_block     = True     (the 76 leftover sites are pinned at init in EVERY frame, so
                                   frame 0 reproduces the full init_grid population)
    hold_ms            = 0.0      (no extra dwell on the final displaced frame)
    wgs3d_warm         = True     (protocol-specific default)
    skip_final_phase   = True     (protocol-specific default)
    precompute         = True     (2026-07-30: turned ON. Dispatcher default is False, but the
                                   protocol docstring recommends True and it is free here -- the
                                   frame stack is shot-independent)

SCAN-LEVEL knobs (not protocol defaults -- the protocol has no say in these): 0.696 ms/frame,
matched defocus -5 (the current 33x33_feedback11 operating point, same as RearrangeSTIRAPScan),
33x33_feedback11 for BOTH frames.

CAVEAT on img2 detection: the atoms are DISPLACED by up to 11 knm px (~45% of the 24.5 px pitch)
when img2 is taken, but img2 is detected against the undisplaced 33x33_feedback11 registry grid --
so the img2 bitstring / dashboard survival is NOT meaningful here. Look at the raw img2 FRAME (the
2x2 blocks blown apart) -- that is what this demo shows.

Run it:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PipelineDemoScan.py
    python YbScans/RearrangeDiagnostics/PipelineDemoScan.py --reps 100

Submission is a plain foreground ENQUEUE (ybStartScan appends to the backend queue) -- it never
aborts or preempts whatever is already running.
"""

import argparse
import json
import os
import sys


# --------------------------- EDIT ME: pattern + transit ----------------------------- #
# Same array for load and "target": pipeline_demo ignores target_grid entirely (blocks are read
# off the INIT lattice), but the warmup handshake still wants a final_phase.
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

# Per-corner displacement, rows = (TR, TL, BL, BR), cols = (dx, dy, dz). 2026-07-30: the SERVER
# DEFAULT ([[8,8,0],[0,0,-5],[11,0,0],[5,5,5]]) lost EVERY atom -- including the 76 HELD non-block
# sites, which never move (run 20260730_163944: img1 627 atoms/shot -> img2 4.1 = noise floor), so
# the loss was global to the transit, not a per-corner displacement effect. This is the default
# scaled DOWN 5x.
# 2026-07-30 (job 393): dz DOUBLED so the axial move is visible as defocus blur -- +-2 rad =
# +-1.6 um (0.798 um/rad), comfortably more than the depth of focus. NOTE 2 x 0 = 0, so TR and BL
# stay in-plane by construction; that is useful, not a bug -- they are the in-plane control pair
# in the same image as the two axially-moved corners. Give TR/BL a nonzero dz here if you want all
# four moving axially.
# 2026-07-30 (job 395): xy QUADRUPLED (= 4/5 of the original server default) and dz set to a
# round +-3.00 um per corner.
#
# UNITS -- use 0.9057 um/rad, NOT the 0.798 in the server's pipeline_demo docstring. The live
# constant is train3d.UM_PER_RAD = 8*s^2/(pi*lambda) with s = KNM1024_TO_UM = 0.435 um/knm px,
# verified on the server 2026-07-30 (memory: open-rearr-zernike-depth-um-per-rad, RESOLVED
# 2026-07-01; it superseded an older 3.26). So +-3.00 um = +-3.312 rad.
# Lateral scale: 1 knm-1024 px = 0.435 um, so 8.8 knm px = 3.83 um = ~19.9 camera px -- now well
# above the ~2.2 px PSF, i.e. the block split should be visible in SINGLE shots, not just
# block-averaged. Site pitch is 24.5 knm px (10.7 um), so the largest move is 36% of the spacing.
# 2026-07-30 (job 396): dz TRIPLED again to +-9.00 um. Kernel-table check before submitting:
# wgs3d_z_max = |dz| + 0.5 = 10.44 rad at the default wgs3d_dz = 0.05 gives Z = 417 slices and a
# patch radius R = 34 (S = 69) -> ~4 MB of kernel (vs ~1 MB at +-3 um), against 26 GB free on the
# GPU. No memory concern; the bigger patch just costs a little more per-frame solve time, which
# precompute + 3 ms/frame absorb.
UM_PER_RAD = 0.9057
DZ_UM = 9.00
_DZ = round(DZ_UM / UM_PER_RAD, 4)          # 3.312 rad
DISPLACEMENT = [[6.4, 6.4, 0.0],
                [0.0, 0.0, -_DZ],
                [8.8, 0.0, 0.0],
                [4.0, 4.0, _DZ]]

# Transit length. SWEPT (axis 1): 0 = the protocol's own no-motion control (nsteps <= 0 degenerates
# to a single undisplaced frame -- still 3-D, still warm-WGS, still no WGS bookend), so an empty
# img2 at nsteps 0 indicts the 3-D WGS / skip_final_phase path rather than the motion itself.
NSTEPS_LIST = [0, 1, 3, 5, 10, 20, 50]

# 2026-07-30 (job 391 -> job 392): step period raised 0.696 -> 3 ms and precompute turned ON.
# 0.696 ms is the SLM's ImageWriteComplete boundary, i.e. the host_stream floor -- computing a
# frame during flipping can miss it. precompute=True builds the whole stack before the transit
# starts (free here: the frame stack is shot-independent), and 3 ms/frame gives the liquid crystal
# ~4.3x the settle budget per step. Transit time is now nsteps * 3 ms = 0/3/9/15/30/60/150 ms.
# 2026-07-30: back to the HEALTHY operating point, 3 ms + precompute.
# Job 394 settled the mechanism question: at 0.696 ms WITH precompute there is no nsteps collapse
# (0.633 at nsteps 50, vs 0.006 for job 391's 0.696 ms WITHOUT precompute), so job 391's collapse
# was COMPUTE STARVATION -- warm WGS solving ~0.55-0.9 ms/frame inside a 0.696 ms host_stream
# window and missing write deadlines. A residual per-flip cost at 0.696 ms remains (held sites,
# which never move but are re-rendered every frame, ran 0.852 at 3 ms vs 0.721 at 0.696 ms over
# 20 frames), so 3 ms is the operating point -- and the xy move is now 4x larger, which wants MORE
# settle per step, not less.
STEP_PERIOD_MS = 3.0
PRECOMPUTE = True

# Matched defocus: rearrange model z4 == rp.loading_defocus so the transit frames sit at the same
# focal plane as the loaded atoms. -5 = the current 33x33_feedback11 operating point.
DEFOCUS = -5

# FRAME PRODUCER (2026-07-30, job 397): the protocol forces wgs3d_warm=True on itself, i.e. every
# transit frame is solved by the warm matched-filter 3-D WGS producer. Set USE_3D_MODEL=True to
# pass wgs3d_warm=False explicitly and run the 3-D SLMNet CNN instead, which then REQUIRES a 3-D
# checkpoint + gpu_target_graph=True (chirp-splat target builder).
#
# Z-RANGE NOTE: the shipped direct3d/base5x5 checkpoint was trained under the OLD 3.26 um/rad
# guess, so it learned rad = um/3.26 and its "15-35 um" training layers are |z| ~ 4.6-10.7 RAD
# (memory: open-rearr-zernike-depth-um-per-rad). The model is rad-native, so the current
# dz = +-9.94 rad sits INSIDE that trained band -- this is the right dz at which to compare it
# against warm WGS. Physically that band is ~4.2-9.7 um at the corrected 0.9057 um/rad.
USE_3D_MODEL = True
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
MODEL_3D = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
MODEL_FILENAME = MODEL_3D if USE_3D_MODEL else MODEL_2D
# ------------------------------------------------------------------------------------ #


def _bootstrap():
    """Scans in YbScans subdirs get the SUBdir as sys.path[0], so add the pyctrl roots by hand."""
    here = os.path.dirname(os.path.abspath(__file__))          # .../YbScans/RearrangeDiagnostics
    root = os.path.dirname(os.path.dirname(here))              # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    if root not in sys.path:
        sys.path.insert(0, root)


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback9": ("phase/33x33_feedback9.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "47x47_feedbackwarm4": ("phase/47x47_feedbackwarm4.pt", [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    """One imagePatternsJson entry (per camera frame). ``order='col'`` matches the runner's
    synthesized default + the server sweep_order the detection grid is derived in."""
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(init_cfg, target_cfg):
    """[LOADING, POST-DISPLACEMENT]. Both frames use the SAME (undisplaced) registry grid -- see
    the module docstring's caveat: img2's bits are not a meaningful metric for this demo."""
    return json.dumps([_pattern_item(INIT_PATTERN, init_cfg),
                       _pattern_item(TARGET_PATTERN, target_cfg)])


_bootstrap()

from RearrangeCommSeq import RearrangeCommSeq          # noqa: E402


def build():
    """Build (do NOT submit) the ScanGroup. Separate so it can be exercised offline."""
    from scan_group import ScanGroup

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    g().rearrange_kwargs.extras.n_rounds = 1        # img1 -> displace -> img2

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) --------------
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

    # ---- rearrange_kwargs (g(); per-shot setup) -----------------------------------------
    # PROTOCOL DEFAULTS: block_displacement / block_rot90 / block_row_origin / block_col_origin /
    # hold_non_block / hold_ms are all DELIBERATELY NOT SET, so the server uses its own defaults
    # (see the module docstring). wgs3d_warm + skip_final_phase are likewise left alone -- the
    # dispatcher turns both ON for this protocol.
    g().rearrange_kwargs.protocol = "pipeline_demo"
    g().rearrange_kwargs.extras.block_displacement = DISPLACEMENT
    # nsteps SPLITS the fixed total displacement (frame k sits at fraction k/nsteps); it does NOT
    # repeat it. So this axis sweeps per-frame step size AND transit duration together
    # (nsteps * 0.696 ms = 0 / 0.7 / 2.1 / 3.5 / 7.0 / 13.9 / 34.8 ms).
    g().rearrange_kwargs.nsteps.scan(1, NSTEPS_LIST)
    g().rearrange_kwargs.step_period_ms = STEP_PERIOD_MS
    # Precompute the whole frame stack before the transit starts. Free for this protocol -- the
    # stack is shot-independent (blocks come from lattice geometry, not from the loading), so it
    # is built once and reused every shot.
    g().rearrange_kwargs.extras.precompute = PRECOMPUTE

    # Frame producer: the protocol's own default is wgs3d_warm=True (warm matched-filter 3-D WGS).
    # Passing False explicitly routes every frame through the 3-D SLMNet CNN + the chirp-splat GPU
    # target builder, which the dispatcher only enables when gpu_target_graph is on.
    if USE_3D_MODEL:
        g().rearrange_kwargs.extras.wgs3d_warm = False
        g().rearrange_kwargs.extras.gpu_target_graph = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.z4 = DEFOCUS         # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = 2000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(init_cfg, target_cfg)

    return g


def PipelineDemoScan(url=None, reps=20):
    """Build + ENQUEUE the pipeline_demo scan on the running pyctrl backend (foreground queue --
    does not abort/preempt anything already running). Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    description = (
        ("3-D SLMNet MODEL frame producer (wgs3d_warm=False + gpu_target_graph=True, checkpoint "
         "experiment_3d/base5x5_fp16) instead of the protocol's default warm 3-D WGS -- direct A/B "
         "against job 396, which is identical in every other respect. The checkpoint was trained "
         "under the old 3.26 um/rad guess so it is rad-native over |z| ~ 4.6-10.7 rad; dz = "
         "+-9.94 rad sits inside that band. "
         if USE_3D_MODEL else "") +
        "BIG-MOVE RUN: xy QUADRUPLED (= 4/5 of the original server default; largest move 8.8 knm "
        "px = 3.83 um = ~19.9 camera px, well above the ~2.2 px PSF, so the 2x2 split should be "
        "visible in SINGLE shots) and dz set to a round +-3.00 um = +-3.312 rad using the LIVE "
        "constant 0.9057 um/rad (train3d.UM_PER_RAD -- NOT the stale 0.798 in the pipeline_demo "
        "docstring). Back to the healthy 3 ms + precompute operating point: job 394 showed the "
        "nsteps collapse is COMPUTE STARVATION (0.696 ms + precompute = 0.633 at nsteps 50 vs "
        "0.006 without), with a smaller residual per-flip cost at 0.696 ms (held-site survival "
        "0.852 at 3 ms vs 0.721 at 0.696 ms). pipeline_demo on %s, displacement %s "
        "(TR/TL/BL/BR, dx,dy knm px, dz rad), SWEEPING nsteps over %s @ %g ms/frame (= %s ms "
        "transit; nsteps SPLITS the fixed total displacement, so this varies step size AND "
        "duration together; nsteps=0 is the protocol's own no-motion control -- one undisplaced "
        "frame, still 3-D / warm-WGS / no bookend). Everything else at protocol defaults "
        "(rot90=0, origins (0,0), hold_non_block=True, hold_ms=0, wgs3d_warm=True, "
        "skip_final_phase=True), matched defocus %g. FOLLOW-UP to job 391 (20260730_165552, same "
        "sweep at 0.696 ms/frame, precompute off), where survival fell MONOTONICALLY with nsteps "
        "-- 0.912 / 0.673 / 0.659 / 0.557 / 0.122 / 0.015 / 0.006 for nsteps 0/1/3/5/10/20/50 -- "
        "i.e. loss tracked TRANSIT TIME even though the per-frame step was shrinking, while the "
        "static single-frame point (nsteps=0) held 0.91. This run tests whether the streaming "
        "path was compute-starved at the 0.696 ms host_stream floor: precompute builds the whole "
        "stack up front and 3 ms/frame gives ~4.3x the settle budget. img2 bits are detected on "
        "the UNDISPLACED grid and are not a fill metric."
        % (INIT_PATTERN, DISPLACEMENT, NSTEPS_LIST, STEP_PERIOD_MS,
           [round(n * STEP_PERIOD_MS, 1) for n in NSTEPS_LIST], DEFOCUS))
    did = ybStartScan(RearrangeCommSeq, g, url=url, label="PipelineDemoScan",
                      description=description, **opts)
    print("submitted PipelineDemoScan -> descriptor id %s (url=%s, reps=%s, pattern=%s, "
          "nsteps sweep=%s, disp=%s, defocus=%g)"
          % (did, url or "default", reps, INIT_PATTERN, NSTEPS_LIST, DISPLACEMENT, DEFOCUS))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit PipelineDemoScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=20,
                    help="passes over the 7-point nsteps sweep (20 -> 140 shots); 0 = forever")
    args = ap.parse_args()
    PipelineDemoScan(url=args.url, reps=args.reps)
