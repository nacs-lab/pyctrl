"""SLMRearrangeSphere3DFocusStackScan.py -- rearrange 33x33_feedback11 into a 50-atom SPHERE of
radius 20 um, then image it one axial slice at a time.

Seq: ``RearrangeSphereFocusStackCommSeq`` (2 camera frames, 2 mid-shot SLM-server handoffs).  Read
that module's docstring for the per-shot mechanics and ``campaigns/geometry/sphere3d/sphere_geometry.py`` for the
sphere itself; this file is the numbers + the wiring.

WHAT ONE SHOT DOES
  load the production 33x33 array  -> img1 (which sites loaded)
  warm 3-D WGS transit: the nearest loaded atoms -> the 50 sphere sites (bookend = the sphere WGS)
  pingponggrating depth walk of the WHOLE sphere by %(walk)s rad (phase-only, one-way)
  ramp the trap servo %(v0).2f -> %(v1).2f V (50 traps do not want the 1068-trap power)
  -> img2: the sphere at that carrier, with ONE latitude level in focus

THE SCAN AXIS IS THE FOCUS
  The sphere is 40 um deep and the imaging depth of field is ~4 um, so a frame can only ever show
  one level.  The scan sweeps the walk amplitude over the sphere's own level depths -- one cell
  per level, %(nlvl)d cells -- so each cell is a different slice of the SAME rigid object.  Average
  each cell's frames and stack them and you have the sphere; that stack is the deliverable.
  The mapping walk -> level is ``walk = -z_level`` (a site is in focus when the global carrier
  cancels its own depth), which also makes this run a direct check of the rig's z SIGN: if the
  in-focus ring appears at the opposite end of the axis, the sign is flipped and nothing else
  about the scan changes.

WHY A WALK AND NOT A PER-SHOT TARGET OFFSET
  Commanding the sphere at carrier+delta would work optically but every cell would then assemble a
  DIFFERENT object over a different transit length, and per-level brightness would inherit that.
  Assembling one sphere and translating it afterwards keeps the object, the transit, and the
  survival identical across cells -- the only thing that changes is a global quadratic phase.

THE TWO PER-SHOT SETUPS (the "initial phase" is handed back and forth once per shot)
  ``pingponggrating`` ramps Z4 on top of the server's cached WGS **initial_phase**.  So:
    * setup #1 (``rearrange_kwargs``, pushed by ``pre_run``) re-sends ``initial_phase`` = the
      LOADING hologram with ``skip_grid_derive=True``, so this shot's ``reload_rearrange`` writes
      the 33x33 array -- undoing the previous shot's walk setup;
    * setup #2 (``rearrange_kwargs2``, pushed mid-shot) re-sends ``initial_phase`` = the SPHERE
      with ``skip_grid_derive=True``, so the walk starts from the hologram physically on the panel
      (the bookend the assembly just wrote) instead of snapping back to the loading array.
  ``skip_grid_derive`` keeps BOTH derived grids (no mid-shot FFT, 3-D z labels intact).  Neither
  setup passes ``model_filename`` or ``reset_params``, so the run id does not bump.

WARM 3-D WGS, NOT THE MODEL (``wgs3d_warm=True``)
  The sphere's sites sit at up to +-25.06 rad of Z4.  The direct3d checkpoint quantizes per-spot z
  onto a 1 um = 1.105 rad ladder and was trained on layer gaps of 4.6-10.7 rad, i.e. it is both
  coarse and out of distribution here; the warm matched-filter 3-D WGS producer solves every frame
  exactly at its own depth with no model and no target image (and beat the model above ~2 rad/step
  in the 2026-07-31 A/B).  ``wgs3d_z_max`` is PINNED (not auto) so the kernel table is built once
  for the whole scan instead of being rebuilt whenever stochastic loading changes max|z|.

TRAP DEPTH -- the one thing that is genuinely different about a 50-trap array
  The 532 servo sets TOTAL power, so 50 traps at the 1068-trap setpoint are ~21x deeper than
  production: enough light shift to put the 399 imaging light badly off resonance.  The seq
  therefore ramps ``VSLMservo`` down over %(rampms).0f ms AFTER the walk and BEFORE img2 (adiabatic
  at 85 kHz radial), so transport happens deep and imaging happens at roughly the production
  per-trap depth.  %(v1).2f V is ``1068-trap setpoint x 50/1068 x 2`` -- the 2x is margin for the
  3-D hologram's lower on-grid efficiency (~0.30 of total power lands on the sites, measured on
  the 2-layer array).  It is a STARTING value: check it (trap frequency or a survival/imaging
  round on the sphere pattern) before trusting the fidelity numbers.

FIRST RUN / BOOTSTRAP -- use ``--assemble-only``
  The sphere pattern has no registry record and no per-site thresholds yet.  ``--assemble-only``
  runs load -> img1 -> assemble -> img2 with NO walk: it answers "did the 3-D assembly work?"
  before the walk adds a second unknown, and its frames are what
  ``campaigns/imaging/imaging_det/bootstrap_pattern_thresholds.py`` fits the sphere's thresholds from.  Nothing in
  the shot path needs those thresholds (img2 is never fed back to the server -- the server scores
  bits against the 1068-site init_grid and this array is a different 50-site grid), but the
  dashboard and any per-site analysis do.

Run it (pyctrl backend live at --url; the SLM server must be reachable -- the scan-long slm lock
is mandatory).  The sphere hologram must exist on the server first:
``campaigns/geometry/sphere3d/gen_sphere50_r20um.py``.
    cd pyctrl
    python YbScans/SLMRearrangeSphere3DFocusStackScan.py --dry-run
    python YbScans/SLMRearrangeSphere3DFocusStackScan.py --assemble-only --force   # FIRST
    python YbScans/SLMRearrangeSphere3DFocusStackScan.py --force                   # the stack
    python YbScans/SLMRearrangeSphere3DFocusStackScan.py --fine --force            # 15-slice stack
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(HERE)
REPO = os.path.dirname(PYCTRL)
sys.path.insert(0, os.path.join(REPO, "_sphere3d"))

import sphere_geometry as SPHERE       # noqa: E402  (the ONE definition of the target geometry)

# ======================= THE TWO ARRAYS ======================== #
LOAD_PATTERN = "33x33_feedback11"                  # production loading array (calibrated)
LOAD_PHASE = "phase/33x33_feedback11.pt"
LOAD_BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]

SPHERE_PATTERN = "sphere50_r20um"                  # made by campaigns/geometry/sphere3d/gen_sphere50_r20um.py
SPHERE_PHASE = "phase/sphere50_r20um.pt"
PLANES_Z_RAD = [float(z) for z in SPHERE.PLANES_Z_RAD]
N_PER_PLANE = [int(n) for n in SPHERE.N_PER_PLANE]

# The 2-D production checkpoint. The WARM path never runs a model, but setup_rearrangement still
# wants a checkpoint (and the run record names it); same choice as SelectiveAxialFocusScan.
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# Peak-extraction cut for BOTH derived grids. 0.3 is the cut the live 33x33_feedback11 registry
# record was built at -- the server's init_grid and the lab's detection grid MUST agree in length
# (1068) or the bits vector is rejected. Verify the sphere derives 50 sites at this cut when the
# hologram is generated (the generator prints its own count; /slm/initialize_loading_pattern
# reports the lab-side one).
DERIVE_THRESHOLD = 0.3

# ======================= FOCUS / CARRIER ======================== #
LOADING_DEFOCUS = -4.0        # production carrier for 33x33_feedback11 (= the camera focal plane)

# ======================= THE TRANSIT ======================== #
# Lateral: the 50 targets fill a 98 knm-px disk and the Hungarian pairs the NEAREST loaded atoms,
# so under enhanced loading the sources come from a ~230 px box -> worst transit ~120 px. 160
# frames keeps that under the 0.75 knm-px/frame transit-step limit (the cliff is at >= 1.25).
# Axial: 25.06 rad / 160 = 0.157 rad/frame, ~14x inside the ~2.25 rad per-frame axial cliff.
NSTEPS = 160
STEP_PERIOD_MS = 0.696        # -> 111 ms transit
DEPTH_PISTON_CORR_3D = -0.5   # the measured ridge for the PER-SITE 3-D path (jobs 416-432)
WGS3D_Z_MAX = 25.6            # kernel table half-range: max|z| 25.06 + margin. PINNED (see above)
WGS3D_RADIUS_FRAC = 0.7
BLOCK_MAX_SIZE = 256
PROB_HUNGARIAN_BETA = 300     # atom SURPLUS (hundreds loaded, 50 targets) -> confidence-weighted

# ======================= THE WALK ======================== #
WALK_STEPS = 50               # 50 frames -> <= 0.50 rad/frame at the largest walk; 35 ms
WALK_PERIOD_MS = 0.696
DEPTH_PISTON_CORR_GRATING = 0.441   # the grating's OWN measured null (NOT the -0.5 above; the two
                                    # paths share one sticky extras dict, so both are set)

# ======================= TRAP DEPTH ======================== #
LOAD_SERVO = 3.7              # the production 1068-trap setpoint (expConfig Init.VSLMServo)
SPHERE_SERVO = 0.35           # 50 traps, 2x production per-trap depth (see the docstring)
SERVO_RAMP_MS = 10.0


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def walk_values(fine=False):
    """The scan axis: total walk (rad of PV ANSI Z4) per cell.

    Default = one cell per sphere level, ``walk = -z_level``, so each cell puts exactly one
    latitude ring on the camera focal plane.  ``fine`` interleaves the midpoints (2x the slices,
    i.e. a stack sampled at ~3 um instead of ~6 um) for a smoother reconstruction at 2x the shots.
    """
    walks = [float(w) for w in SPHERE.WALK_RAD_FOR_PLANE]
    walks.sort()
    if not fine:
        return walks
    out = []
    for i, w in enumerate(walks):
        out.append(w)
        if i + 1 < len(walks):
            out.append(0.5 * (w + walks[i + 1]))
    return out


def _image_patterns_json():
    """Per-frame detection declaration.

    frame 0 -- the production 33x33 loading array: the pattern whose grid + per-site thresholds
    the rearrange call's probability vector comes from.  ``order='col'`` and threshold 0.3 are the
    live record's own settings (registry ``33x33_feedback11``, 1068 sites); they must not drift or
    the bits length stops matching the server's init_grid.

    frame 1 -- the sphere: a 3-D record, all %d levels declared, so the lab-side derive refocuses
    to each level in turn and returns all %d sites with their z labels.  Its thresholds are
    bootstrapped offline (see the module docstring); nothing in the shot path blocks on them.
    """ % (len(PLANES_Z_RAD), SPHERE.N_ATOMS)
    load = {"name": LOAD_PATTERN, "base_phase_path": LOAD_PHASE, "order": "col",
            "legacy_zerniked": False, "threshold": DERIVE_THRESHOLD}
    sph = {"name": SPHERE_PATTERN, "base_phase_path": SPHERE_PHASE, "order": "col",
           "legacy_zerniked": False, "planes_z_rad": PLANES_Z_RAD,
           "threshold": DERIVE_THRESHOLD}
    return json.dumps([load, sph])


def build(fine=False, assemble_only=False, nsteps=NSTEPS, walk_steps=WALK_STEPS,
          sphere_servo=SPHERE_SERVO, servo_ramp_ms=SERVO_RAMP_MS, walks=None,
          load_defocus=LOADING_DEFOCUS):
    """Build (do NOT submit) the ScanGroup.  Returns ``(seq_name, g, walks)``.  Kept separate from
    the submit wrapper so it can be exercised offline without touching the live backend."""
    _bootstrap()
    from scan_group import ScanGroup

    walks = [float(w) for w in (walks if walks else walk_values(fine))]
    if int(walk_steps) < 1:
        raise ValueError("walk_steps must be >= 1 (it divides each cell's walk)")

    g = ScanGroup()
    rp = g.runp()

    # ---- seq-local declarations (NOT forwarded to the SLM server) ----------------------
    # The trap-servo ramp the seq runs after the walk (50 traps at the 1068-trap setpoint would be
    # ~21x too deep; see the module docstring).
    g().sphere.servo = float(sphere_servo)
    g().sphere.servo_ramp = float(servo_ramp_ms) * 1e-3
    g().sphere.n_sites = int(SPHERE.N_ATOMS)

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params=True) ---------
    # init_grid comes from the LOADING phase declared as a single 3-D plane at z = 0, target_grid
    # from the SPHERE phase with all 8 levels declared. Both grids must carry z or the assignment
    # silently falls back to 2-D (``_assign_pairs_auto`` dispatches to 3-D only when BOTH do) and
    # the transit would ignore depth entirely.
    rp.warmup_kwargs.model_filename = MODEL_2D
    rp.warmup_kwargs.initial_phase = LOAD_PHASE
    rp.warmup_kwargs.final_phase = SPHERE_PHASE
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = list(LOAD_BAKED_ZERNIKE)
    rp.warmup_kwargs.extras.final_phase_zernike = [0, 0, 0, 0, 0]   # sphere is zernike-free
    rp.warmup_kwargs.extras.init_grid_planes_z_rad = [0.0]
    rp.warmup_kwargs.extras.target_grid_planes_z_rad = list(PLANES_Z_RAD)
    # Cross-plane xy-dedup MUST be 0: the sphere's two POLES are on the optical axis and therefore
    # xy-COINCIDENT, so any dedup radius merges them (50 -> 49 sites) and the target grid stops
    # matching the declared geometry.
    rp.warmup_kwargs.extras.dedup_xy_knm = 0
    rp.warmup_kwargs.derive_threshold = DERIVE_THRESHOLD
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True

    # ---- rearrange_kwargs = per-shot setup #1: THE 3-D ASSEMBLY ------------------------
    rk = g().rearrange_kwargs
    rk.protocol = "rearrange"
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(STEP_PERIOD_MS)
    # Hand the "initial phase" role BACK to the loading hologram (the previous shot's walk setup
    # left the sphere there) so this shot's reload_rearrange writes the 33x33 array.
    # skip_grid_derive keeps both cached grids -- no per-shot 4096^2 FFT.
    rk.initial_phase = LOAD_PHASE
    rk.skip_grid_derive = True
    rk.extras.wgs3d_warm = True          # warm matched-filter 3-D WGS, NOT the direct3d model
    rk.extras.wgs3d_z_max = float(WGS3D_Z_MAX)
    rk.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)
    rk.extras.depth_piston_corr = float(DEPTH_PISTON_CORR_3D)
    rk.extras.precompute = True          # solve the whole frame stack before the paced loop
    rk.extras.hw_sequence = False
    rk.extras.block_max_size = int(BLOCK_MAX_SIZE)
    rk.extras.ghost_fraction = 0.0       # sparse-N: only the moving atoms exist in the frames
    rk.extras.prob_hungarian = True      # atom surplus -> pair the atoms most likely to be there
    rk.extras.prob_hungarian_beta = int(PROB_HUNGARIAN_BETA)
    rk.extras.z4 = float(load_defocus)   # transit-frame carrier = the loading/camera plane
    rk.extras.n_rounds = 1
    rk.extras.ifEnhanced = True
    rk.extras.initial_pattern = LOAD_PATTERN
    rk.extras.final_pattern = SPHERE_PATTERN
    if assemble_only:
        rk.extras.assemble_only = True   # read by the seq: skip setup #2 + the walk entirely

    # ---- rearrange_kwargs2 = per-shot setup #2: THE AXIAL WALK -------------------------
    # Pushed mid-shot, between the assembly and img2. protocol pingponggrating is a dispatcher
    # short-circuit: each frame is initial_phase + k*step_size*Z4 written straight to the panel --
    # no model, no bookend. return=False rests on the walked frame, which is what img2 sees.
    if not assemble_only:
        rk2 = g().rearrange_kwargs2
        rk2.protocol = "pingponggrating"
        rk2.nsteps = int(walk_steps)
        rk2.step_period_ms = float(WALK_PERIOD_MS)
        # THE load-bearing line: make the SPHERE the phase the grating ramps on top of.
        rk2.initial_phase = SPHERE_PHASE
        rk2.skip_grid_derive = True
        rk2.extras.depth = True                  # step_size in rad of PV ANSI Z4
        rk2.extras.return_trip = False           # one-way: rest on the walked frame
        rk2.extras.no_depth_piston = True
        # The grating's piston null is its OWN constant; the extras dict is shared with setup #1,
        # so this MUST be re-stated or the 3-D path's -0.5 would leak into the walk.
        rk2.extras.depth_piston_corr = float(DEPTH_PISTON_CORR_GRATING)
        rk2.extras.piston = 0.0
        rk2.extras.precompute = True             # uint8 frames built at setup -> write-only loop
        # THE SCAN AXIS: per-step amplitude, i.e. this cell's total walk / walk_steps.
        rk2.extras.step_size.scan(1, [w / float(walk_steps) for w in walks])

    # ---- run params -------------------------------------------------------------------
    rp.NumPerGroup = 100000
    rp.loading_defocus = float(load_defocus)
    rp.NumImages = 2                             # img1 loading array, img2 the walked sphere
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.isRearrange = 1
    rp.imagePatternsJson = _image_patterns_json()

    return "RearrangeSphereFocusStackCommSeq", g, walks


def _desc(walks, nsteps, walk_steps, sphere_servo, servo_ramp_ms, assemble_only):
    lvl = ", ".join("%+.1f um -> walk %+.2f rad" % (z, -z / SPHERE.UM_PER_RAD)
                    for z in SPHERE.PLANES_Z_UM)
    return (
        "3-D SPHERE + FOCUS STACK.  Rearrange %s (1068 sites, carrier %+.1f) into a %d-atom "
        "SPHERICAL SHELL of radius %.0f um (%s, %d levels at z = %s um, populated %s; centre %.0f "
        "knm px right of the zeroth order so the physical DC block cannot eat it; lateral NN 3.8 "
        "um, 3-D NN 7.1 um).  Per shot: img1 on %s -> warm 3-D WGS transit (wgs3d_warm=True, "
        "z_max %.1f, radius_frac %.2f, depth_piston_corr %+.2f, nsteps %d @ %.3f ms = %.0f ms, "
        "<= 0.75 knm-px and 0.157 rad per frame, ghost_fraction 0, prob_hungarian beta %d in the "
        "atom-surplus regime) with the sphere WGS as the bookend -> %s -> VSLMservo ramp %.2f -> "
        "%.2f V over %.0f ms (50 traps at the 1068-trap setpoint are ~21x too deep; transport "
        "stays deep, imaging lands near production per-trap depth) -> img2.  %s  The 3-D model is "
        "deliberately NOT used: direct3d quantizes per-spot z on a 1.105 rad ladder and was "
        "trained at 4.6-10.7 rad of layer gap, while this object spans +-25.06 rad.  Both grids "
        "are declared 3-D (init: one plane at z=0; target: the 8 sphere levels) because the "
        "assignment only goes 3-D when BOTH carry z.  dedup_xy_knm=0 is mandatory: the two POLES "
        "are xy-coincident by construction and any dedup would merge them.  Level -> walk map: %s."
        % (LOAD_PATTERN, LOADING_DEFOCUS, SPHERE.N_ATOMS, SPHERE.RADIUS_UM, SPHERE_PATTERN,
           len(PLANES_Z_RAD), ", ".join("%+g" % z for z in SPHERE.PLANES_Z_UM),
           list(N_PER_PLANE), abs(SPHERE.CENTER_YX_KNM1024[1] - 512.0), LOAD_PATTERN,
           WGS3D_Z_MAX, WGS3D_RADIUS_FRAC, DEPTH_PISTON_CORR_3D, nsteps, STEP_PERIOD_MS,
           nsteps * STEP_PERIOD_MS, PROB_HUNGARIAN_BETA,
           ("NO walk (ASSEMBLE-ONLY: bootstraps the sphere pattern's thresholds and isolates the "
            "assembly)" if assemble_only else
            "pingponggrating depth walk of the whole sphere, %d frames, one-way, "
            "depth_piston_corr %.3f, scanned over %d amplitudes %s rad"
            % (walk_steps, DEPTH_PISTON_CORR_GRATING, len(walks),
               ["%+.2f" % w for w in walks])),
           LOAD_SERVO, sphere_servo, servo_ramp_ms,
           ("The walk starts from the SPHERE hologram (setup #2 re-sends it as initial_phase with "
            "skip_grid_derive), so frame 0 of the walk continues the bookend instead of snapping "
            "back to the loading array; setup #1 hands the role back each shot."
            if not assemble_only else
            "img2 images the sphere at the loading carrier, i.e. its equatorial levels."),
           lvl))


def main():
    ap = argparse.ArgumentParser(
        description="Submit the 50-atom 20 um sphere rearrangement + axial focus-stack scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=None,
                    help="shots per pass (default 150 x n_cells)")
    ap.add_argument("--fine", action="store_true",
                    help="interleave the level midpoints -> 2x slices (~3 um sampling), 2x shots")
    ap.add_argument("--walks", default=None,
                    help="explicit comma-separated total walks in rad (overrides --fine)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="no walk: load -> img1 -> assemble -> img2. RUN THIS FIRST on a new "
                         "sphere -- it isolates the 3-D assembly and gives "
                         "campaigns/imaging/imaging_det/bootstrap_pattern_thresholds.py the frames it needs.")
    ap.add_argument("--nsteps", type=int, default=NSTEPS, help="transit frames (default %d)" % NSTEPS)
    ap.add_argument("--walk-steps", type=int, default=WALK_STEPS,
                    help="frames in the axial walk (default %d)" % WALK_STEPS)
    ap.add_argument("--servo", type=float, default=SPHERE_SERVO,
                    help="VSLMservo the seq ramps to before img2 (default %.2f V)" % SPHERE_SERVO)
    ap.add_argument("--servo-ramp-ms", type=float, default=SERVO_RAMP_MS)
    ap.add_argument("--load-defocus", type=float, default=LOADING_DEFOCUS)
    ap.add_argument("--description", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    walks = ([float(v) for v in args.walks.split(",")] if args.walks else None)
    seq_name, g, walks = build(
        fine=args.fine, assemble_only=args.assemble_only, nsteps=args.nsteps,
        walk_steps=args.walk_steps, sphere_servo=args.servo, servo_ramp_ms=args.servo_ramp_ms,
        walks=walks, load_defocus=args.load_defocus)
    n_cells = g.nseq()
    g.runp().NumPerGroup = int(args.num_per_group if args.num_per_group else 150 * n_cells)

    print(SPHERE.summary_text())
    print("\nseq=%s  cells=%d  shots/pass=%d" % (seq_name, n_cells, g.runp().NumPerGroup(0)))
    print("  transit: nsteps=%d @ %.3f ms = %.0f ms (warm 3-D WGS, z_max %.1f)"
          % (args.nsteps, STEP_PERIOD_MS, args.nsteps * STEP_PERIOD_MS, WGS3D_Z_MAX))
    if args.assemble_only:
        print("  ASSEMBLE-ONLY: no walk, img2 images the sphere at the loading carrier")
    else:
        print("  walk: %d frames @ %.3f ms = %.0f ms, one-way; amplitudes (rad) %s"
              % (args.walk_steps, WALK_PERIOD_MS, args.walk_steps * WALK_PERIOD_MS,
                 ["%+.2f" % w for w in walks]))
        print("        max per-frame axial step %.3f rad (cliff ~2.25)"
              % (max(abs(w) for w in walks) / args.walk_steps))
    print("  trap servo: ramp %.2f -> %.2f V over %.0f ms before img2"
          % (LOAD_SERVO, args.servo, args.servo_ramp_ms))

    if args.dry_run:
        s0 = g.getseq(0)
        print("\n  cell0 rearrange_kwargs: %s"
              % {k: s0["rearrange_kwargs"].get(k) for k in ("protocol", "nsteps", "initial_phase",
                                                            "skip_grid_derive")})
        if not args.assemble_only:
            steps = sorted({round(g.getseq(i)["rearrange_kwargs2"]["extras"]["step_size"], 6)
                            for i in range(n_cells)})
            print("  distinct grating step_size: %s" % steps)
            print("  -> total walks: %s"
                  % ["%+.2f" % (v * args.walk_steps) for v in steps])
        return

    if not args.force:
        raise SystemExit("refusing to submit without --force")

    from yb_start_scan import ybStartScan
    desc = args.description or _desc(walks, args.nsteps, args.walk_steps, args.servo,
                                     args.servo_ramp_ms, args.assemble_only)
    opts = {"rep": args.reps} if args.reps is not None else {}
    label = "SLMRearrangeSphere3DFocusStack%s" % ("_assembleonly" if args.assemble_only else "")
    did = ybStartScan(seq_name, g, url=args.url, label=label, description=desc, **opts)
    print("\nsubmitted %s -> descriptor id %s (url=%s)" % (label, did, args.url or "default"))
    return did


if __name__ == "__main__":
    main()
