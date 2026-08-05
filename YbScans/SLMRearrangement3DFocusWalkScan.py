"""SLMRearrangement3DFocusWalkScan.py -- 2-layer (20 um) 3-D rearrangement with a mid-shot AXIAL
FOCUS WALK: load both layers, image the BACK layer, walk the array axially so the FRONT layer is in
focus, image it, then rearrange BOTH layers' atoms into the FRONT layer and image the result.

Seq: ``Rearrange3DFocusWalkCommSeq`` (new; 3 camera frames, 2 mid-shot SLM-server handoffs). Read
that module's docstring for the per-shot mechanics -- this file is the numbers + the wiring.

WHY THIS SCAN EXISTS
  ``SLMRearrangement3DScan`` works on a bifocal array whose layers are ~5 um apart, i.e. inside the
  imaging depth of field: both layers show up in one image (they are also xy-offset there), so one
  frame reads the whole 3-D grid. THIS array (``phase/2x11x11_5um_z20um.pt``, generated 2026-07-29)
  is the opposite regime and the point of the measurement:
    * 2 x 11x11 sites, 5 um lateral pitch, layers **xy-ALIGNED** (stacked along the optical axis),
    * axial gap **20 um** = z4 +-12.5313 rad (the 2026-07-27 recal: 0.798 um/rad, 2.45 knm-px/um --
      NOT train3d's 0.905/0.435), sim CV 0.0015, front/back amplitude balance 1.0002.
  20 um is ~9 Rayleigh ranges, so exactly ONE layer is ever resolvable: the other contributes a
  diffuse halo. The occupancy of both layers therefore has to be read in TWO images, and the only
  way to bring the second layer onto the camera focal plane is to MOVE THE WHOLE ARRAY axially --
  the ``pingponggrating depth`` walk, phase-only (no model), one Z4 ramp on top of the loading WGS.

FLOW (one shot)
  load 2 layers @ carrier %(load)+.4f          -> img1: BACK layer (z4 %(back)+.4f) in focus
  walk %(nsteps)d x %(step)+.4f rad = %(walk)+.4f rad  -> img2: FRONT layer (z4 %(front)+.4f) in focus,
                                                  SAME camera pixels (layers are xy-aligned)
  3-D rearrange(both layers -> FRONT layer)    -> img3: the assembled front-layer array
  Fidelity = img3 occupancy of the 121 front-layer target sites, given the (img1 + img2) loaded set.

THE THREE CARRIERS (all ANSI z4, PV, radians -- one place, derived from ONE geometry constant)
  Z_CAM_DEFOCUS %(zcam)+.1f  = the carrier at which a FLAT array is in focus on the science camera
                        (= slm_runtime.DEFAULT_LOADING_DEFOCUS = the affine's calibration plane
                        = the dz block's z_ref). If the camera focus moves, this is the ONE number
                        to re-measure (sweep --load-defocus and watch img1 sharpness).
  LOADING_DEFOCUS %(load)+.4f = Z_CAM - z_back : written for the whole scan by SlmScanSession, and
                        re-written by reload_rearrange at every shot start -> the BACK layer sits at
                        the camera plane while the FRONT layer is 20 um out.
  POST-WALK %(post)+.4f       = LOADING_DEFOCUS + walk : the carrier the array RESTS at after the walk
                        (``return=False``). It is BOTH the 3-D model frames' carrier
                        (``rearrange_kwargs2.extras.z4``) AND the carrier the WGS bookend must be
                        written at (``rearrange_kwargs2.extras.loading_zernike``) -- see below.

THE BOOKEND RE-WRITE (the one subtle bit; no server-side code change needed)
  The server applies ONE ``loading_zernike`` to BOTH cached WGS write phases, but this scan needs
  the INITIAL write (the load) at LOADING_DEFOCUS and the FINAL write (the rearrange bookend, which
  is what img3 is taken on) at the POST-WALK carrier -- they differ by the whole 20 um walk. So the
  seq's second per-shot setup (``rearrange_kwargs2``) re-sends ``final_phase`` together with
  ``skip_grid_derive=True`` and the post-walk ``loading_zernike``: the server then rebuilds ONLY
  ``_rearrange_final_phase`` (= base + post-walk z4) and leaves the initial write phase and BOTH
  derived grids untouched (no re-derivation FFT mid-shot). Result: the walk's last frame and the
  bookend carry the SAME quadratic term -- identical holograms up to an optically inert global
  piston (the walk's depth map is the pistonless ``2*rho^2 - <2*rho^2>_beam``, the bookend's is ANSI
  ``2*rho^2 - 1``) -- so the bookend write is a no-jump continuation instead of a 25-rad (20 um)
  snap back that would drop every atom and defocus img3.

DETECTION -- ONE 121-site single-plane pattern for all three frames
  Verified on the live server 2026-07-29 (/eval): with ``planes_z_rad=[%(back).4f, %(front).4f]`` +
  ``dedup_xy_knm=0`` the derive returns 242 sites = [121 back, 121 front] whose halves are the same
  xy to <0.2 knm px, and the SINGLE-plane derive (``planes_z_rad=[%(back).4f]``) is byte-identical to
  the matching half (max|d| = 0.0). So this scan declares ONE detection pattern -- the same phase
  with a single declared plane -> a 121-site record whose order matches each server half exactly --
  for ALL THREE frames. Consequences, all of them wanted:
    * one registry record + ONE per-site threshold set, shared by the three frames (they image the
      same 121 camera boxes at the same optical configuration),
    * the affine's linear-defocus (dz) term is identical and CORRECT for every frame at any loading
      carrier (the walk equals the layer separation, so the in-focus layer's total defocus deviation
      is the same in img1 and img2/img3),
    * the seq composes the server's 242-site vector as [img1 121 | img2 121] and posts
      "0"*121 + img3 bits to ``/slm/results`` (the back layer is emptied by the rearrangement).

FIRST RUN / BOOTSTRAP -- use ``--walk-only``
  This pattern has no registry record and no per-site thresholds yet, and detection cannot
  bootstrap itself inside a rearrangement scan (a shot with no thresholds is never published, so
  the monitor never accumulates). ``--walk-only`` runs load -> img1 -> walk -> img2 -> img3 with NO
  rearrangement and NO detection requirement: the frames publish, the monitor fits the thresholds,
  and the run doubles as the walk's own survival/heating measurement (img3 vs img2 with nothing but
  the walk between them). THEN run the full scan.

KNOWN RISK -- the 3-D model is OUT OF DISTRIBUTION at this depth
  The shipped ``direct3d`` checkpoint was trained on layer gaps of 15-35 um under the OLD 3.26
  um/rad scale, i.e. **4.6-10.7 rad** of Z4 (train3d.py's own note). This array's sites sit at
  +-12.5313 rad about the (post-walk) carrier -- the model input z's are ~2.3x beyond anything it
  saw. The chirp-patch encoding itself is analytic and in range (the builder's ladder covers +-40
  "um" = +-44 rad, and a 12.5-rad patch is ~32 px inside the 65-px patch), but the CNN's hologram
  quality at that depth is unverified. If the 3-D transit loses atoms, that is the first suspect --
  not the walk (which is phase-only and model-free, and is exactly what --walk-only isolates).
  Mitigation already in place: the carrier is centred BETWEEN the layers, which MINIMISES max|z|
  (+-12.53 rather than 0/-25.06).

Run it (pyctrl backend live at --url; the SLM server must be reachable -- the scan-long slm lock is
mandatory):
    cd pyctrl
    python YbScans/SLMRearrangement3DFocusWalkScan.py --walk-only --reps 30   # FIRST: bootstrap
    python YbScans/SLMRearrangement3DFocusWalkScan.py                         # full flow
    python YbScans/SLMRearrangement3DFocusWalkScan.py --load-defocus 8.5      # re-focus img1
    python YbScans/SLMRearrangement3DFocusWalkScan.py --walk-steps 80         # finer walk step
"""

import argparse
import json
import os
import sys


# ======================= ARRAY GEOMETRY (from the generator) ======================== #
# phase/2x11x11_5um_z20um.pt, analysis/wgs3d_2x11x11_z20um_2026-07-29/summary.json:
# 2 x 11x11, 5 um pitch, xy-ALIGNED layers, measured layer z4 = +-12.5326 rad (nominal +-12.5313 =
# +-10 um at 0.798 um/rad), zernike-free base (stack centre at z4 = 0).
PHASE_PATH = "phase/2x11x11_5um_z20um.pt"
Z_HALF_RAD = 12.531328320802004         # half the axial gap, rad of PV ANSI Z4 (2*rho^2-1)
N_PER_PLANE = 121                       # 11x11 sites per layer (242 total, layer-major)

# Which layer is imaged FIRST ("back") and which is the rearrangement TARGET ("front"). The walk
# always moves the FRONT layer onto the BACK layer's plane, so these two also fix the walk's sign;
# the physical z-sign of the rig is NOT involved (this is a relative move inside one hologram).
# Swap the two if the rig turns out to prefer loading/imaging the other layer first -- everything
# below (loading carrier, walk, front_layer_sign) follows automatically.
BACK_LAYER_Z = -Z_HALF_RAD              # in focus during img1 (the dense load readout)
FRONT_LAYER_Z = +Z_HALF_RAD             # in focus during img2 + img3 (the rearrangement target)
PLANES_Z_RAD = [BACK_LAYER_Z, FRONT_LAYER_Z]     # declared order == server layer-major site order

# ======================= FOCUS / CARRIERS ======================== #
# The carrier at which a FLAT array is in focus on the science camera (= slm_runtime's
# DEFAULT_LOADING_DEFOCUS, the global affine's calibration plane, and the dz block's z_ref).
Z_CAM_DEFOCUS = -5.0
LOADING_DEFOCUS = Z_CAM_DEFOCUS - BACK_LAYER_Z    # +7.5313 -> BACK layer at the camera plane
WALK_TOTAL_RAD = BACK_LAYER_Z - FRONT_LAYER_Z     # -25.0627 -> FRONT layer onto that same plane
POST_WALK_DEFOCUS = LOADING_DEFOCUS + WALK_TOTAL_RAD

# ======================= THE WALK (pingponggrating, depth mode) ======================== #
# nsteps sets the per-step axial stroke: 50 steps over 25.0627 rad = 0.5013 rad/step = 0.40 um/step,
# i.e. ~0.18 of the Rayleigh range and ~4x inside the measured axial per-frame cliff (~2.25 rad PV).
# 51 frames at 0.696 ms = 35.5 ms of walk. return=False so the panel RESTS on the walked frame.
WALK_STEPS = 50
WALK_PERIOD_MS = 0.696
# Beam model for the pistonless depth map (user 2026-07-29): subtract the BEAM-WEIGHTED mean of
# 2*rho^2 for a Gaussian of 1/e^2 radius 0.675 (rho units) so piston=0 really is the
# zero-commanded-uniform-phase point for a 25-rad stroke. depth_fill_center is left at the server's
# measured default.
DEPTH_FILL_FRAC = 0.675

# ======================= THE 3-D REARRANGEMENT ======================== #
# 3-D checkpoint (server-side path). See "KNOWN RISK" in the module docstring: it is out of its
# trained depth range at +-12.53 rad.
MODEL_FILENAME = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
NSTEPS = 40                 # transit frames (as SLMRearrangement3DScan)
STEP_PERIOD_MS = 0.696      # -> 27.8 ms transit; 0.63 rad/frame axial for a cross-layer atom

# Detection / ByPattern name for all three frames: the SINGLE-plane (121-site) record of this array.
DETECT_PATTERN = "2x11x11_5um_z20um_layer"
# Peak-extraction cut, used for BOTH the server's 242-site rearrange grid (warmup) and the lab's
# 121-site detection record (imagePatternsJson) so the two come from one extraction.
DERIVE_THRESHOLD = 0.35

# Walk-only default (no 3-D rearrangement). CLI --walk-only / env YB_WALK_ONLY=1.
WALK_ONLY = bool(int(os.environ.get("YB_WALK_ONLY", "0")))


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Per-frame detection declaration: the SAME single-plane pattern for all three frames.

    ``planes_z_rad=[BACK_LAYER_Z]`` makes the lab-side registry derive refocus to ONE layer and
    return its 121 sites in the same order as the matching half of the server's 242-site 3-D grid
    (verified byte-identical). Declaring both planes here instead would give a 242-site record whose
    two halves share the same xy -- the server-grid position match cannot bijectively resolve that,
    and the dz term would then be applied with the pre-walk carrier to the front half (~6 px of
    bogus offset on img2/img3). ``order='col_up'`` == the server's sweep_order default, and
    ``threshold`` is pinned to the SAME cut the server's warmup derive uses (0.35) so both grids
    come out of the same extraction (121 sites at 0.30/0.35/0.40 either way -- verified 2026-07-29 --
    but pinning it keeps the lab and server grids from drifting apart if that changes)."""
    item = {"name": DETECT_PATTERN, "base_phase_path": PHASE_PATH, "order": "col_up",
            "legacy_zerniked": False, "planes_z_rad": [BACK_LAYER_Z],
            "threshold": DERIVE_THRESHOLD}
    return json.dumps([item, item, item])


def build(walk_only=None, load_defocus=None, walk_steps=None, nsteps=None):
    """Build (do NOT submit) the ScanGroup. Returns ``(seq_name, g)``. Kept separate from the
    submit wrapper so it can be exercised offline without touching the live backend."""
    _bootstrap()
    from scan_group import ScanGroup

    walk_only = WALK_ONLY if walk_only is None else bool(walk_only)
    load_z4 = LOADING_DEFOCUS if load_defocus is None else float(load_defocus)
    n_walk = WALK_STEPS if walk_steps is None else int(walk_steps)
    n_transit = NSTEPS if nsteps is None else int(nsteps)
    if n_walk < 1:
        raise ValueError("walk_steps must be >= 1 (it divides the %.4f rad walk)" % WALK_TOTAL_RAD)
    walk_step = WALK_TOTAL_RAD / float(n_walk)
    post_walk_z4 = load_z4 + WALK_TOTAL_RAD
    # +1 -> the max-z layer is the target; -1 -> the min-z layer (server's `front_layer` pattern).
    front_sign = 1 if FRONT_LAYER_Z >= BACK_LAYER_Z else -1

    g = ScanGroup()
    rp = g.runp()

    # ---- seq-local declaration (NOT forwarded to the SLM server) -----------------------
    # Sites per axial layer: only used to size an all-zero vector when detection produced nothing
    # (walk-only / un-bootstrapped thresholds); a successful detection supersedes it.
    g().focus_walk.n_per_plane = N_PER_PLANE

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params=True) ---------
    # Both grids are derived from the SAME 3-D phase with BOTH planes declared -> init_grid and
    # target_grid are the same 242 sites with the same z labels, which is what keeps the transit's
    # per-site z interpolation physical (a front-layer atom stays put, a back-layer atom climbs the
    # full 25 rad). loading_zernike is added by the runner from rp.loading_defocus.
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [0, 0, 0, 0, 0]   # zernike-free base
    rp.warmup_kwargs.extras.final_phase_zernike = [0, 0, 0, 0, 0]
    rp.warmup_kwargs.extras.planes_z_rad = PLANES_Z_RAD
    # Cross-plane xy-dedup MUST be 0: the two layers are xy-COINCIDENT, so ANY dedup radius merges
    # them (242 -> 121). The 20 um gap already makes each plane's extraction clean (axial contrast
    # ~27 in the generator's own check), so no dedup is needed to suppress double detections.
    rp.warmup_kwargs.extras.dedup_xy_knm = 0
    rp.warmup_kwargs.derive_threshold = DERIVE_THRESHOLD
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True

    # ---- rearrange_kwargs = per-shot setup #1: THE AXIAL FOCUS WALK --------------------
    # Pushed by the seq's pre_run (sticky, no reset_params). protocol pingponggrating is a
    # dispatcher short-circuit: each frame is WGS_initial + k*step_size*Z4 written straight to the
    # panel -- no model, no bookend. return=False leaves the panel on the fully-walked frame, which
    # is where img2/img3 are taken (and, after the setup #2 re-write below, exactly the bookend).
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = n_walk
    g().rearrange_kwargs.step_period_ms = WALK_PERIOD_MS
    g().rearrange_kwargs.extras.depth = True             # step_size in rad of PV ANSI Z4
    g().rearrange_kwargs.extras.step_size = walk_step
    g().rearrange_kwargs.extras.return_trip = False      # one-way: rest on the walked frame
    g().rearrange_kwargs.extras.no_depth_piston = True   # pistonless depth map (server default)
    g().rearrange_kwargs.extras.depth_fill_frac = DEPTH_FILL_FRAC
    g().rearrange_kwargs.extras.piston = 0.0
    # Pre-convert the (nsteps+1) unique frames to uint8 at setup time -> the walk's hot loop is
    # write-only (period floor at the SLM write latency instead of compute+write). Rebuilt on every
    # per-shot setup, i.e. once per shot, BEFORE the sequence starts (not on held atoms).
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.n_rounds = 1             # ONE rearrangement round (3 images)
    g().rearrange_kwargs.extras.ifEnhanced = False
    if walk_only:
        g().rearrange_kwargs.extras.walk_only = True     # read by the seq; skips setup #2 + rearrange
    # Per-bseq expConfig ByPattern keys (same array + same in-focus layer for all three frames).
    # No ByPattern entry exists for this name yet -> base config + the explicit g() overrides below.
    g().rearrange_kwargs.extras.initial_pattern = DETECT_PATTERN
    g().rearrange_kwargs.extras.middle_pattern = DETECT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = DETECT_PATTERN

    # ---- rearrange_kwargs2 = per-shot setup #2: THE 3-D REARRANGEMENT ------------------
    # Pushed mid-shot by the seq (between img2 and the rearrange call), sticky on top of setup #1.
    if not walk_only:
        g().rearrange_kwargs2.protocol = "rearrange"
        g().rearrange_kwargs2.nsteps = n_transit
        g().rearrange_kwargs2.step_period_ms = STEP_PERIOD_MS
        # Re-send the final phase so the server rebuilds the WGS BOOKEND at the POST-WALK carrier
        # (the whole point -- see the module docstring). skip_grid_derive keeps the cached
        # target_grid: no mid-shot 4096^2 FFT, and the z labels stay in the un-walked frame the
        # transit interpolation needs.
        g().rearrange_kwargs2.final_phase = PHASE_PATH
        g().rearrange_kwargs2.skip_grid_derive = True
        g().rearrange_kwargs2.extras.loading_zernike = [0, 0, 0, 0, post_walk_z4]
        # Model-frame carrier: the SAME post-walk plane, so the transit frames and the walked array
        # share a focal plane. Per-site depth (+-12.53) rides on top of this via the 3-D grids.
        g().rearrange_kwargs2.extras.z4 = post_walk_z4
        # Target = every site of the front layer (the one now in focus). front_layer_sign picks
        # which z-sign counts as "front".
        g().rearrange_kwargs2.extras.pattern = "front_layer"
        g().rearrange_kwargs2.extras.front_layer_sign = front_sign
        # 3-D REQUIRES the GPU chirp-splat target builder (the dispatcher refuses to feed a 2-D
        # point-paint input to the 3-D model).
        g().rearrange_kwargs2.extras.gpu_target_graph = True
        g().rearrange_kwargs2.extras.precompute = True
        g().rearrange_kwargs2.extras.precompute_host = True
        g().rearrange_kwargs2.extras.hw_sequence = False
        g().rearrange_kwargs2.extras.block_max_size = 256
        # Imaging-weighted assignment: this scan runs in atom SURPLUS (up to 242 loaded vs 121
        # targets), the only regime where the -beta*log(p) term can change the pairing.
        g().rearrange_kwargs2.extras.prob_hungarian = True
        g().rearrange_kwargs2.extras.prob_hungarian_beta = 300

    # ---- imaging / loading params for this array ---------------------------------------
    # There is no expConfig ByPattern entry for DETECT_PATTERN, so the array-specific params are set
    # HERE (a scan-level g() override wins in every bseq, which is what we want -- all three frames
    # image the same 121 boxes). Values = ByPattern["2x11x11_5um_back2um"], the tuned entry for the
    # other 242-site 11x11 two-layer array (07-14 campaign, 50 ms): the closest calibrated starting
    # point. Re-optimize per-array (cooling_img_round / imaging_round) and then consider promoting
    # them into an expConfig ByPattern entry.
    g().Init.VSLMServo = 0.39                    # 242 traps (base 3.7 is for 1000+ site arrays)
    g().BlueMOT.Img1PIDSet = float(os.environ.get("YB_IMG1PID", "0.625"))
    g().BlueMOT.Img2PIDSet = float(os.environ.get("YB_IMG2PID", "0.35"))
    g().LAC.FreqDetuning = 0.11e6
    g().LAC.Amp = 0.2
    g().LAC.Time = 30e-3
    g().Imag399.Cool556.X.FreqDetuning = 0.14e6
    g().Imag399.Cool556.X.Amp = 0.26
    g().Imag399.Cool556.h.FreqDetuning = 0.13e6
    g().Imag399.Cool556.h.Amp = 0.20
    g().Cool556.X.FreqDetuning = 0.16e6
    g().Cool556.X.Amp = 0.14
    g().Cool556.h.FreqDetuning = 0.16e6
    g().Cool556.h.Amp = 0.12

    # ---- run params -------------------------------------------------------------------
    rp.NumPerGroup = 100000
    rp.loading_defocus = load_z4                 # BACK layer at the camera plane (img1)
    rp.NumImages = 3                             # img1 back, img2 front (post-walk), img3 final
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    # Force the rearrangement path (locks + dequeue setup + the seq's server context) even in
    # walk-only mode, where the model is never used but the walk still goes through /slm/rearrange.
    rp.isRearrange = 1
    rp.imagePatternsJson = _image_patterns_json()

    return "Rearrange3DFocusWalkCommSeq", g


def SLMRearrangement3DFocusWalkScan(url=None, reps=None, walk_only=None, load_defocus=None,
                                    walk_steps=None, nsteps=None, description=None):
    """Build + SUBMIT the 3-D focus-walk rearrangement scan. Returns the descriptor id."""
    seq_name, g = build(walk_only=walk_only, load_defocus=load_defocus,
                        walk_steps=walk_steps, nsteps=nsteps)
    from yb_start_scan import ybStartScan

    walk_only = WALK_ONLY if walk_only is None else bool(walk_only)
    load_z4 = LOADING_DEFOCUS if load_defocus is None else float(load_defocus)
    n_walk = WALK_STEPS if walk_steps is None else int(walk_steps)
    step = WALK_TOTAL_RAD / float(n_walk)

    if description is None:
        description = (
            "2-layer 20 um 3-D rearrangement with a mid-shot AXIAL FOCUS WALK on %s (2x11x11, 5 um "
            "pitch, xy-ALIGNED layers at z4 %+.4f / %+.4f rad = 20 um apart). Shot: load at carrier "
            "%+.4f (BACK layer in focus) -> img1 -> pingponggrating depth walk %d x %+.4f rad "
            "(= %+.4f, return=False, no_depth_piston fill %.3f) -> img2 (FRONT layer in focus, same "
            "camera pixels) -> %s -> img3. Detection: ONE single-plane 121-site pattern %r for all "
            "three frames; the server's 242-site vector is composed [img1 back | img2 front] and the "
            "result posted as 0*121 + img3. %s"
            % (PHASE_PATH, BACK_LAYER_Z, FRONT_LAYER_Z, load_z4, n_walk, step, WALK_TOTAL_RAD,
               DEPTH_FILL_FRAC,
               "NO rearrangement (WALK-ONLY: bootstraps the pattern thresholds + measures the "
               "walk's own survival)" if walk_only else
               "3-D rearrange both layers -> front layer (pattern front_layer, nsteps %d @ %.3f ms, "
               "model %s, bookend re-written at the post-walk carrier %+.4f)"
               % (nsteps or NSTEPS, STEP_PERIOD_MS, MODEL_FILENAME, load_z4 + WALK_TOTAL_RAD),
               DETECT_PATTERN,
               "NOTE the 3-D model is out of its trained depth range at +-12.53 rad (trained "
               "4.6-10.7 rad of layer gap) -- if the transit loses atoms, suspect that first."))

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "SLMRearrange3DFocusWalk%s" % ("_walkonly" if walk_only else "")
    did = ybStartScan(seq_name, g, url=url, label=label, description=description, **opts)
    print("submitted %s -> descriptor id %s (url=%s)\n"
          "  load carrier %+.4f (back layer z4 %+.4f in focus) -> walk %d x %+.4f rad = %+.4f "
          "-> post-walk carrier %+.4f (front layer z4 %+.4f in focus)\n"
          "  mode: %s   detection pattern: %s (121 sites/layer, 242 server sites)"
          % (label, did, url or "default", load_z4, BACK_LAYER_Z, n_walk, step, WALK_TOTAL_RAD,
             load_z4 + WALK_TOTAL_RAD, FRONT_LAYER_Z,
             "WALK ONLY (no rearrangement)" if walk_only else
             "full flow (3-D rearrange -> front layer)", DETECT_PATTERN))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 2-layer 20 um 3-D focus-walk rearrangement scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    ap.add_argument("--walk-only", action="store_true",
                    help="load -> img1 -> walk -> img2 -> img3 with NO rearrangement. RUN THIS "
                         "FIRST on a new array: it bootstraps the pattern registry record + "
                         "per-site thresholds the full flow needs, and measures the walk's own "
                         "survival/heating.")
    ap.add_argument("--load-defocus", type=float, default=None,
                    help="ANSI z4 carrier written for the whole scan (default %+.4f = camera plane "
                         "%.1f minus the back layer's %+.4f). Sweep this to re-focus img1; the walk "
                         "and the post-walk bookend/model carriers follow automatically."
                         % (LOADING_DEFOCUS, Z_CAM_DEFOCUS, BACK_LAYER_Z))
    ap.add_argument("--walk-steps", type=int, default=None,
                    help="frames in the axial walk (default %d -> %.4f rad = %.2f um per step)"
                         % (WALK_STEPS, abs(WALK_TOTAL_RAD) / WALK_STEPS,
                            abs(WALK_TOTAL_RAD) / WALK_STEPS * 0.798))
    ap.add_argument("--nsteps", type=int, default=None,
                    help="3-D rearrangement transit frames (default %d)" % NSTEPS)
    ap.add_argument("--description", default=None, help="override the run description")
    args = ap.parse_args()
    SLMRearrangement3DFocusWalkScan(url=args.url, reps=args.reps,
                                    walk_only=args.walk_only or None,
                                    load_defocus=args.load_defocus, walk_steps=args.walk_steps,
                                    nsteps=args.nsteps, description=args.description)
