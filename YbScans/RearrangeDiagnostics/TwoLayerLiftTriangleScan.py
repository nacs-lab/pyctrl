"""TwoLayerLiftTriangleScan.py -- build a TWO-LAYER array out of one loaded plane and read BOTH
layers out in the same shot, with the full trap array present in every frame.

THE PROTOCOL (4 camera frames, 3 SLM rounds; seq ``TwoLayerLiftCommSeq``)

  img1    the loading array: 1068 sites of ``33x33_feedback11``, ~0.5 fill.
  ROUND 0 2-D warm-WGS compaction, loading -> the EVERY-OTHER checkerboard ``(row+col) even``
          (536 targets).  Supply-saturated (~640 atoms at 0.6 fill), so it comes out near-full.
          Bookend = the FULL loading WGS -- the dead traps are put back, not dropped.
  img2    the compacted array.  Same 1068 traps as img1, ~536 of them occupied.
  ROUND 1 the 3-D LIFT.  Target = the two-layer WGS: 816 sites at the loading plane (532 dead
          non-every-other + 284 NEAR) and 252 FAR sites carried +%(um).0f um and staggered a
          quarter sublattice pitch into a triangular (centred-rectangular) lattice.
          ``target_bits`` makes only the 284 near + 252 far sites active, so the 536 atoms map
          1:1: the near half is assigned to the sites it is already sitting on (zero cost) and
          the odd/odd half goes up.  Bookend = the two-layer WGS -- again all 1068 traps.
  img3    at the loading plane.  %(um).0f um is ~9 Rayleigh ranges, so only the NEAR layer
          resolves: this is "the atoms left behind", ~284 sites' worth, and they should all still
          be alive.  Same trap count, same depth as img1/img2.
  ROUND 2 ``pingponggrating``, true-defocus, ONE-WAY: the whole hologram walks -%(um).0f um so
          the FAR layer arrives at the camera plane (and the near layer leaves it).
  img4    the far layer, in focus, as a staggered lattice.  Detected on its own registry grid.

  img3 and img4 are disjoint sets of atoms, so together they are the assembly's fidelity.

TRAP DEPTH IS CONSTANT ACROSS ALL FOUR FRAMES.  Every bookend is a FULL 1068-site WGS -- the
compaction's bookend restores the loading array rather than writing an every-other one, and the
two-layer bookend keeps the dead traps too.  Same trap count every frame -> same power per trap ->
the four frames are directly comparable with no depth correction, and img2/img3 reuse the existing
``33x33_feedback11`` grid and per-site thresholds.  (The TRANSIT frames are still sparse: an
assignment protocol builds its frames from the paired atoms only.  That is between images and
touches no imaging condition; every atom keeps its own trap throughout.)

THREE ROUNDS, THREE SETUPS.  ``protocol`` / ``pattern`` / ``target_bits`` / ``nsteps`` / the phases
are all server-STICKY and the rounds disagree about every one, so each round pushes its own
``setup_rearrangement`` (``rearrange_kwargs`` / ``_kwargs2`` / ``_kwargs3``).  Round 0's setup
re-asserts the LOADING phases every shot -- load-bearing, because round 2 leaves the server's
cached ``initial_phase`` pointing at the two-layer hologram and ``reload_rearrange`` writes
whatever that is at the next shot's start.  Cost: the grid derives, measured at 40-75 ms each on
the server (2026-08-07), i.e. ~0.25 s of extra in-trap time per shot against a ~300 s lifetime.

THE nsteps NUMBERS, AND THE ATOM DATA BEHIND THEM (all at the 0.696 ms SLM write floor)
  * ROUND 0 = %(n0)d frames.  Warm-WGS ``every_other`` survival vs nsteps on THIS array
    (2026-08-02 tricks campaign, run #645, loading-post-selected): 0.963 / 0.971 / 0.975 / 0.983
    at nsteps 35 / 40 / 45 / 50, against a MEASURED detection floor of 0.9885 +- 0.002 at n=80.
    50 is the first point at the floor.  Per-frame stroke 24.5/50 = 0.49 knm px, just under the
    campaign's 99.99 %%/step stroke (0.50 px) and well under the 0.9 px heating onset.
  * ROUND 1 = %(n1)d frames.  Set by the LATERAL leg, not the axial one: %(lat).2f knm px of
    stagger over %(n1)d frames = %(lat_step).2f px/frame, under the 99.95 %%/step warm-WGS stroke
    (0.70 px).  The axial leg comes along for free at %(ax_step).2f rad/step = %(ax_um).2f um/step,
    ~4x inside the warm 3-D producer's measured free regime (job 432: per-step loss
    0.000 +- 0.0005 out to 4.8 rad/step with the per-atom phase correction applied).
  * ROUND 2 = %(n2)d frames = %(walk_step).2f um/step.  The axial grating cliff (99 %%/step) is
    3.4-3.5 um/step at 0.696 ms (2026-08-07 axial campaign, jobs 369/386, timing-cut) and the
    heating onset sits at ~0.75x the cliff (~2.5 um), so %(walk_step).2f um is 0.24x the cliff --
    per-step loss <1e-4, thermally free, whole-move loss <~0.3%%.  This leg is deliberately the
    SLOW one: the far-layer atoms have already paid for the lift and there is no second chance.

  Transit budget: %(n0)d + %(n1)d + %(n2)d frames + 3 bookends ~ %(ms).0f ms of SLM playback.

CONSTANTS ARE THE SERVER'S, NOT A DOCSTRING'S.  ``train3d.UM_PER_RAD`` on the live server is
**%(umrad).6f** um/rad (the 2026-07-27 recal -- the constant the 20 um two-layer array was built
with) and ``KNM1024_TO_UM`` is **%(knmum).6f**.  The 0.9057 / 0.435 pair quoted in some scan
docstrings is the older derivation and is NOT what this rig is running; the two-layer hologram was
solved against the live values, so the scan uses them too.

THE +z / -z SIGN AND THE PISTON NULL -- the two things to check on the rig
  * ``--walk-sign`` (default -1) is the sign of the grating amplitude.  The dispatcher's
    displacement index is non-negative, so direction lives entirely in the amplitude sign, and
    "negative amplitude -> the array moves -z" is a rig convention, not a theorem.  A dark img4
    with a healthy img3 means flip it -- that is the whole diagnosis.
  * ``--walk-corr`` (default %(corr).3f rad/um) is the true-defocus piston null for the walk.
    0.617 rad/um is the measured GLOBAL null (job 320); the null is DIRECTION-DEPENDENT (job 428
    found c+ ~ 0.463 vs c- ~ 0.611 in the parabolic numeraire) and the dedicated -z null scan
    (jobs 302/303) was cancelled before it ran, so the -z optimum is provisional and worth one
    ``--walk-corr`` sweep before trusting a low-loss number from this leg.
  The per-atom correction the WARM 3-D producer uses in round 1 is a DIFFERENT constant in a
  DIFFERENT numeraire: ``depth_piston_corr = -0.5`` rad per rad of PV quad-defocus, the measured
  ridge (jobs 416/418, bracketed as an interior maximum).

SERVER-SIDE STATE (already built + verified, 2026-08-07 -- see ``campaigns/geometry/twolayer_lift/build_phase.py``)
    phase/fb11_2layer_full_z20.pt   1068-site 3-D WGS, warm-started from the camera-feedbacked
                                    33x33_feedback11 phase.  Re-derived with
                                    ``planes_z_rad=[0, %(zrad).4f]``: 1068 sites, layer split
                                    816/252, every site on its intended target to <=0.114 knm px,
                                    z exact.
    phase/fb11_2layer_far.pt        the 252 far sites, 2-D at their post-lift xy; registered
                                    lab-side as pattern ``fb11_2layer_far`` (252 sites).
    campaigns/geometry/twolayer_lift/target_bits.json round 1's 1068-long target mask (536 ones) IN THE SERVER'S OWN
                                    DERIVED ORDER -- taken from the server, never re-derived here.
  Still owed before quoting numbers: per-site thresholds for ``fb11_2layer_far`` (they build up
  over the first shots), and a first-shot check of the walk sign.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/TwoLayerLiftTriangleScan.py --dry-run
    python YbScans/RearrangeDiagnostics/TwoLayerLiftTriangleScan.py --reps 30 --force
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(PYCTRL)

# ---- the loaded array ------------------------------------------------------------------
PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# The 3-D direct3d checkpoint (fp16 deployable copy) -- the frame producer for the LIFT.
# CAVEAT worth carrying: it was trained under the OLD 3.26 um/rad guess, so it is rad-native over
# |z| ~ 4.6-10.7 rad.  This lift ramps 0 -> 25.08 rad, so the later frames are OUTSIDE that band
# and the model is extrapolating there.  Its chirp-splat z ladder is also quantized at ~1 um
# (~1.25 rad), which is why raising the lift nsteps far above ~20 stops refining the AXIAL motion
# and only helps the lateral leg.
MODEL_3D = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
# warm-WGS 3D is the DEFAULT producer: measured +0.09 mover survival over direct3d at both 10
# and 20 um (jobs 540 vs 545, 80 reps each).  The 2-D checkpoint is loaded because the warm
# path never runs the model; --slmnet3d swaps in MODEL_3D.
MODEL_FILENAME = MODEL_2D
DEFOCUS = -5.0                 # loading plane == rearrange plane (the 2026-08-07 production value)

# ---- the two-layer array (built + verified on the server 2026-08-07) --------------------
FULL_PHASE = "phase/fb11_2layer_full_z10um.pt"   # 10 um: the working distance (0.96 vs 0.67 at 20)
FAR_PATTERN = "fb11_2layer_far"
FAR_PHASE = "phase/fb11_2layer_far.pt"
TARGET_BITS_JSON = os.path.join(ROOT, "_twolayer_lift", "target_bits_z10um_th040.json")
EO_BITS_JSON = os.path.join(ROOT, "_twolayer_lift", "eo_bits_th040.json")

# ---- scales: the LIVE server constants (slmnet.train3d), not the older derivation -------
UM_PER_RAD = 0.7974368729415139        # train3d.UM_PER_RAD, read from the server 2026-08-07
KNM_UM = 0.4081632653061224            # train3d.KNM1024_TO_UM
LIFT_UM = 10.0
LIFT_RAD = LIFT_UM / UM_PER_RAD        # 25.0804 rad of PV quad-defocus
LAT_STAGGER_PX = 12.249931338528302    # a quarter of the every-other sublattice pitch (2 x 24.4999)

# ---- transit ---------------------------------------------------------------------------
PERIOD_MS = 0.696              # round 0 stays at the SLM write floor
POST_PERIOD_MS = 2.0           # legs 2 and 3 -- the pacing the working axial results used
COMPACT_NSTEPS = 50            # round 0 -- at the measured detection floor
LIFT_NSTEPS = 40               # round 1 -- 0.31 px/frame lateral, 0.63 rad/step axial.
#                                NOTE on the MODEL producer: the chirp-splat z ladder is
#                                quantized at ~1 um (~1.25 rad), so below that a finer nsteps
#                                stops refining the AXIAL move (consecutive frames land on the
#                                same z slice) and only halves the LATERAL step.
WALK_NSTEPS = 50               # round 2 -- 0.50 rad/step (0.40 um), 0.12x the axial cliff.
#                                The grating map is analytic (no z ladder), so unlike the lift
#                                this IS a genuine refinement of the motion.
# The grating leg runs on the PARABOLIC ANSI Z4 map (true_defocus=False), NOT the exact-spherical
# one, so that both axial legs speak the SAME unit -- radians of PV Z4.  That is what makes the
# two legs' DISTANCES exactly equal and their piston corrections directly comparable instead of
# related by a um/rad constant.  In this numeraire the measured grating null is 0.441 rad per rad
# of Z4-PV (the 0.617 figure is the rad/um null of the exact-spherical map and does NOT apply).
# Trade-off, stated: the parabolic map carries a rho^4 residual the exact map does not.
WALK_CORR = 0.441              # rad per rad of Z4-PV -- the measured grating null
WALK_SIGN = -1                 # amplitude sign that walks the array -z (CONFIRM on the rig)
WATERMARK = 512                # >= n_frames => stage the WHOLE transit before the first SLM write
WGS3D_RADIUS_FRAC = 0.7        # warm 3-D kernel patch radius scale (0.7 buys ~2x solve time)
DEPTH_PISTON_CORR_3D = -0.5    # rad per rad of PV quad-defocus -- the warm producer's own ridge
# The lift's LINEAR phase correction is the only per-step phase term that exists: frame k
# subtracts corr*dz[k,i] from atom i's commanded phase, i.e. it scales with THAT atom's
# accumulated axial displacement, so it is a per-step correction proportional to step size.
# (There is no lateral counterpart -- lateral_piston_corr does not exist on the server.)
# -0.5 is the ridge MEASURED for a pure-axial random-z pingpong (jobs 416/418, bracketed as an
# interior maximum). This lift is a different move -- assignment + a 12.25 px lateral leg on top
# of the 25.08 rad axial one -- so that ridge is inherited, not measured here. --lift-corr with
# more than one value sweeps it on dim 1.
LIFT_CORR_SWEEP = [-0.9, -0.7, -0.5, -0.3, 0.0, 0.5]

# ---- loading / imaging work point (2026-08-07 recovery; per-scan g() overrides only) ----
BLUE_DETUNING_MHZ = -48.0      # jobs 309/310: the -40/-42 work point read 0.002 fill
BLUE_LOADING_TIME_S = 0.467    # job 390 vs 386: 0.153 -> 0.378 fill for +0.217 s/shot
GREEN_BIAS_X_A = 0.0343        # jobs 375/376: 0.040 is now dead; plateau centre 0.0343
# Imaging setpoints are NOT pinned by default -- the scan inherits whatever ByPattern/base says,
# so an imaging fix made in config actually takes effect (see the note at the g() overrides).
# --img-pid P1 P2 forces them when you deliberately want a specific power.
IMG_PID = None

NUM_PER_GROUP = 3000


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _bits(path, kind):
    """A site mask emitted by ``campaigns/geometry/twolayer_lift/build_phase.py --verify``, IN THE SERVER'S OWN
    DERIVED INDEX ORDER.

    Both masks are produced by asking the server to derive the grid and matching each derived site
    to its intended target.  Recomputing that order here would mean re-implementing
    ``_sort_grid_rowmajor`` (plus, for the two-layer grid, the layer-major concatenation) and
    hoping the two agree -- the server's own answer cannot drift."""
    with open(path, "r") as fh:
        rec = json.load(fh)
    bits = [int(v) for v in rec["bits"]]
    if sum(bits) != rec.get("n_ones", sum(bits)):
        raise ValueError("%s is inconsistent (n_ones mismatch)" % os.path.basename(path))
    return bits, rec


def target_bits(path=TARGET_BITS_JSON):
    """Round 1's active-target mask over the TWO-LAYER grid: the 284 near + 252 far sites."""
    return _bits(path, "target")


def eo_bits(path=EO_BITS_JSON):
    """The every-other mask over the LOADING grid -- round 1's declared occupancy vector when the
    scan runs without the post-compaction image (round 0 is supply-saturated, so "every target is
    filled" is the right declaration)."""
    return _bits(path, "every_other")


def _image_patterns_json(mid_image=True, do_walk=True, lift_image=True):
    """One entry per camera frame.  Every frame except the far-layer one images the SAME 1068-site
    loading grid -- the compaction and the lift move atoms WITHIN it or out of its focal plane,
    they never change it -- so they share one registry record and one threshold set.  Only the
    far-layer frame needs its own grid, and it only exists when the walk runs."""
    def it(name, path):
        return {"name": name, "base_phase_path": path, "order": "col",
                "legacy_zerniked": False}
    load = it(PATTERN, PHASE_PATH)
    frames = [load]                                    # the loading frame, always
    if mid_image:
        frames.append(dict(load))                      # post-compaction, same grid
    if lift_image or not do_walk:
        frames.append(dict(load))                      # NEAR layer, same grid (the lift is the
        #                                                last round when the walk is off, so it is
        #                                                always read out -- mirrors seq _layout)
    if do_walk:
        frames.append(it(FAR_PATTERN, FAR_PHASE))      # FAR layer, its own grid
    return json.dumps(frames)


def build(compact_nsteps=COMPACT_NSTEPS, lift_nsteps=LIFT_NSTEPS, walk_nsteps=WALK_NSTEPS,
          walk_corr=WALK_CORR, walk_sign=WALK_SIGN, period_ms=PERIOD_MS, defocus=DEFOCUS,
          lift_period_ms=None, bits=None, mid_image=True, mid_bits=None, do_walk=True,
          lift_image=True, img_pid=None, lift_corr=None, warm_wgs3d=True,
          hold_unpaired=True):
    """Build (do NOT submit) the ScanGroup.  Returns ``(seq_name, g, meta)``."""
    _bootstrap()
    from scan_group import ScanGroup

    bits = bits if bits is not None else target_bits()[0]
    eo = eo_bits()[0]
    # The declared vector is needed by ANY round with no camera frame in front of it -- the lift
    # when mid_image is off, and equally the lift when lift_image is off.
    if mid_bits is None and not (mid_image and lift_image):
        mid_bits = list(eo)
    seq_name = "TwoLayerLiftCommSeq"
    g = ScanGroup()

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) ------------------
    # reset_params CLEARS the server's sticky extras, so nothing this scan does not set can leak
    # in from an earlier scan -- which matters here because `pattern` and `target_bits` are
    # mutually exclusive and each round turns one of them off.
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_2D if warm_wgs3d else MODEL_3D
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
    # 0.40: the only window where BOTH derives give 1068 under this run's declarations.
    # At 0.45 the single-plane init derive (planes=[0.0], needed so the assignment takes the
    # 3-D branch) drops sites -> 'bits length 1068 != expected 1060' on every shot (job 563).
    # At 0.30 and below the target picks up a sidelobe. Masks regenerated at 0.40 to match.
    rp.warmup_kwargs.derive_threshold = 0.40

    # ---- loading + imaging work point (2026-08-07 recovery; g() overrides, expConfig untouched)
    g().BlueMOT.FreqDetuning = BLUE_DETUNING_MHZ * 1e6
    g().BlueMOT.LoadingTime = BLUE_LOADING_TIME_S
    g().GreenMOT.BiasCoilCurrent.X = GREEN_BIAS_X_A
    # IMAGING IS DELIBERATELY NOT PINNED HERE (2026-08-07).  An earlier version set
    # BlueMOT.Img1/Img2PIDSet via g(), which BEATS ByPattern in every bseq -- so it would silently
    # override any imaging fix made in expConfig/ByPattern and the scan would keep running the old
    # power.  The frames all image the same pattern, so they share whatever the live config says;
    # that is the property this scan needs (identical imaging in every frame), and it does not
    # require pinning a value.  Pass --img-pid only to deliberately force a setpoint.
    if img_pid is not None:
        g().BlueMOT.Img1PIDSet = float(img_pid[0])
        g().BlueMOT.Img2PIDSet = float(img_pid[1])

    # =====================================================================================
    # ROUND 0 -- 2-D compaction to every_other, bookend = the FULL loading array
    # =====================================================================================
    rk = g().rearrange_kwargs
    rk.protocol = "rearrange"
    rk.nsteps = int(compact_nsteps)
    rk.step_period_ms = float(period_ms)
    # The phases are re-asserted EVERY shot: round 2 leaves the server's cached initial_phase
    # pointing at the two-layer hologram, and reload_rearrange writes whatever that is at the next
    # shot's start.  final_phase = the SAME loading array, which is what makes the post-compaction
    # bookend a full 1068-trap array instead of an every-other one.
    rk.initial_phase = PHASE_PATH
    rk.final_phase = PHASE_PATH
    rk.extras.n_rounds = 1
    # TARGETS ARE ALWAYS `target_bits`, NEVER `pattern` -- in EITHER round.
    #
    # The server rejects a call carrying both ("pass either target_bits OR pattern, not both") and
    # its extras dict is MERGE-ONLY: passing `pattern=None` does NOT clear a sticky `every_other`
    # from an earlier setup, and passing `target_bits=None` does not clear a sticky mask.  So a
    # scan that uses `pattern` in one round and `target_bits` in another poisons itself on the
    # FIRST shot and every shot after (2026-08-07, job 511: round 1 died on shot 1, round 0 on
    # every shot after that).  Using the same always-non-None key in both rounds means each setup
    # simply overwrites the previous one and nothing ever has to be cleared.
    #
    # This mask is the every-other checkerboard over the LOADING grid, in the server's own derived
    # index order -- the same set `pattern="every_other"` would have selected, taken from the
    # server rather than trusted (see campaigns/geometry/twolayer_lift/build_phase.py).
    rk.extras.target_bits = list(eo)           # 536 checkerboard targets
    # MUST be an EMPTY LIST, not None.  The extras dict is merge-only and None is dropped in
    # transit, so `None` does NOT clear round 1's [0, z] declaration: round 0 then derives the
    # FLAT loading phase refocused at the far plane and returns a different site count
    # (1067/1060 instead of 1068), so target_bits no longer matches and every shot after the
    # first fails with 'target_bits length 1068 != len(target_grid)'.  [] is falsy server-side
    # (same 2-D path as None) but is actually written into the sticky dict.
    rk.extras.init_grid_planes_z_rad = []      # cleared for round 0 (2-D), see below
    rk.extras.target_grid_planes_z_rad = []
    # hold_unpaired is STICKY and belongs to the LIFT ONLY.  Left inherited, round 0's
    # compaction runs with 532 stationary ghost traps in every transit frame and moving
    # atoms are intercepted by them en route -- the 'leave the ghosts in place' failure the
    # rearrange skill warns about.  Job 554 measured it: dead-trap occupancy 0.3175 (169
    # atoms at sites that should be empty, vs 0.011 normally) and the compaction degraded
    # 0.987 -> 0.854.  Round 0 must say False explicitly.
    rk.extras.hold_unpaired = False
    rk.extras.prob_hungarian = True            # img1 probabilities weight the assignment
    rk.extras.wgs_warm = True                  # warm-started phase-locked WGS transit frames
    rk.extras.wgs_pad = 2048
    rk.extras.wgs_iters = 3                    # >= 3 (2 is the contract-quality cliff)
    rk.extras.wgs_beam = "gaussian"
    rk.extras.wgs3d_warm = False               # 2-D round: no depth track
    # PRECOMPUTE: solve the whole frame stack on the GPU BEFORE the paced loop, so the hot loop is
    # write-only and a slow warm-WGS solve costs dead time before the move instead of a LATE frame
    # during it.  `precompute` is the real knob on the model-paced `rearrange` path; `watermark` is
    # the equivalent for the rearrange2 streaming producer (>= n_frames => stage every frame first,
    # internally clamped).  BOTH are set so the behaviour does not depend on which path the server
    # routes this protocol through, and so the frame count is never the limit.
    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.watermark = int(WATERMARK)
    rk.extras.hw_sequence = False
    rk.extras.skip_final_phase = False         # the FULL-array bookend is the point
    rk.extras.flip_phase = False
    rk.extras.random_z = False                 # EXPLICIT: sticky from the random-z campaigns
    rk.extras.random_z_max = 0.0
    rk.extras.hold_ms = 0.0
    rk.extras.overdrive = True                 # +0.04..+0.11 survival over nsteps 15-30, ~+0.006
    rk.extras.target_clamp = 0.2               # at the tail (2026-08-02, 8/8 cells positive)
    rk.extras.tau_rise_ms = 1.7
    rk.extras.tau_fall_ms = 3.7
    rk.extras.settle_lut = "auto"
    rk.extras.mover_boost = 1.0                # OFF: boosting movers would dim the statics and
    #                                            break the constant-depth premise of this scan
    rk.extras.ifEnhanced = True                # BlueLAC loading
    rk.extras.z4 = float(defocus)              # transit plane == loading plane
    rk.extras.initial_pattern = PATTERN        # loading frame: ByPattern + detection grid
    rk.extras.middle_pattern = PATTERN         # post-compaction frame (mid_image only)
    rk.extras.lift_pattern = PATTERN           # NEAR-layer frame (same 1068-site grid)
    rk.extras.final_pattern = FAR_PATTERN      # FAR-layer frame (its own grid)
    # mid_image is read at BUILD time by the seq: 1 -> a camera frame between the compaction and
    # the lift (debug); 0 -> the two rounds run back to back on one frame (the demo shape).  Every
    # extra image costs ~1-3% on every atom downstream of it, so it is off for real numbers.
    rk.extras.mid_image = 1 if mid_image else 0
    # do_walk = 0 drops ROUND 2 and its frame: the far layer is still assembled and left 20 um
    # away, just never imaged.  What is left is the cheapest useful question -- what does the lift
    # cost the atoms that DON'T move? -- and it keeps the shot to three frames.
    rk.extras.do_walk = 1 if do_walk else 0
    # lift_image = 0 drops the NEAR-layer frame, so the lift and the walk run back to back and the
    # shot's last image is the FAR layer.  That is the configuration that measures the AXIALLY
    # MOVED atoms (compacted frame -> far-layer frame); with it on you see the ones left behind.
    rk.extras.lift_image = 1 if lift_image else 0

    # =====================================================================================
    # ROUND 1 -- the 3-D lift, bookend = the two-layer array (still 1068 traps)
    # =====================================================================================
    rk2 = g().rearrange_kwargs2
    rk2.protocol = "rearrange"
    rk2.nsteps = int(lift_nsteps)
    rk2.step_period_ms = float(lift_period_ms or POST_PERIOD_MS)
    rk2.initial_phase = PHASE_PATH             # atoms are still on the loading lattice
    rk2.final_phase = FULL_PHASE               # target + bookend: the two-layer array
    rk2.extras.n_rounds = 1
    # The 3-D opt-in: the TARGET grid is extracted layer-major at these two declared depths, which
    # is where inter_z (and therefore the lift itself) comes from.
    # BOTH grids must carry z_rad or the assignment SILENTLY falls back to 2-D:
    # _assign_pairs_auto dispatches to assign_pairs_3d only when both do, and the 2-D branch
    # never emits inter_z -- so the transit carries NO depth and the lift becomes a purely
    # lateral move, with no error anywhere.  Measured on job 560: after the lift all 252
    # atoms sat at the commanded staggered xy (241 ADU, 99.8% of sites -- identical to known
    # atoms) but still in the LOADING plane, so the grating then carried them out of focus.
    # The init grid is flat, hence a single plane at zero.
    rk2.extras.init_grid_planes_z_rad = [0.0]
    rk2.extras.target_grid_planes_z_rad = [0.0, float(LIFT_RAD)]
    # Same key as round 0 (see the note there), so this simply overwrites it -- 536 active targets
    # = 284 near + 252 far, over the TWO-LAYER grid this time.
    rk2.extras.target_bits = list(bits)
    rk2.extras.prob_hungarian = True
    # FRAME PRODUCER for the lift: the 3-D SLMnet model (direct3d) rather than the warm
    # matched-filter 3-D WGS.  gpu_target_graph is REQUIRED -- the dispatcher raises rather than
    # silently feed the 3-D model a 2-D point-paint input.  --warm-wgs3d switches back.
    rk2.extras.wgs3d_warm = bool(warm_wgs3d)
    if not warm_wgs3d:
        rk2.extras.gpu_target_graph = True
    rk2.extras.wgs3d_z_max = float(LIFT_RAD) + 0.5   # pinned: one kernel-table build for the scan
    rk2.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)
    rk2.extras.wgs_warm = True
    rk2.extras.wgs_pad = 2048
    rk2.extras.wgs_iters = 3
    rk2.extras.wgs_beam = "gaussian"
    # Per-atom defocus-phase correction: frame k subtracts corr*dz[k,i] from atom i's commanded
    # phase.  MEASURED ridge -0.5 (jobs 416/418); at 0 the atoms carry a phase transient that
    # kills them well before 1 rad/step.
    # true_defocus is a STICKY, GRATING-ONLY flag that selects how depth_piston_corr is read:
    # rad per um of step when True, rad per rad of PV quad-defocus when False.  Round 2 sets
    # it True, so without this explicit False the lift would inherit True from the PREVIOUS
    # shot's grating setup -- making shot 1 differ from every later shot, silently, in the
    # units of the very constant this scan sweeps.  Same class of trap as the pattern /
    # target_bits collision that killed job 511.
    rk2.extras.true_defocus = False
    # LINEAR phase correction for the lift (rad of spot phase per rad of PV quad-defocus).
    _corr = [float(v) for v in (lift_corr if lift_corr else [DEPTH_PISTON_CORR_3D])]
    if len(_corr) == 1:
        rk2.extras.depth_piston_corr = _corr[0]
    else:
        rk2.extras.depth_piston_corr.scan(1, _corr)   # dim 1, interleaved by Scramble
    rk2.extras.piston = 0.0
    rk2.extras.precompute = True
    rk2.extras.precompute_host = True
    rk2.extras.watermark = int(WATERMARK)      # same reasoning as round 0
    rk2.extras.hw_sequence = False
    # NO BOOKEND: the panel keeps the transit's own final frame.  Writing a WGS over the atoms
    # at the end of the lift is a different hologram for the same atoms -- the thing that was
    # suspected of killing the far layer.  The grating now rides on the last written frame.
    rk2.extras.skip_final_phase = True
    # hold_unpaired keeps the 532 unloaded/untargeted sites emitted STATIC in every transit
    # frame (server patch 2026-08-07, gated, default off), so the full 1068-trap array stays on
    # the panel and per-trap depth does not jump mid-shot.  UNTESTED ON HARDWARE.
    # hold_unpaired is the NEWEST, least-tested link in this chain (server patch written
    # today, never validated on hardware).  --no-hold-unpaired turns it off so it can be
    # falsified: with it off the transit drops the 532 dead traps (depth jumps mid-shot,
    # which is a known cost) but is otherwise the stock assignment path.
    rk2.extras.hold_unpaired = bool(hold_unpaired)
    rk2.extras.random_z = False
    rk2.extras.random_z_max = 0.0
    rk2.extras.hold_ms = 0.0
    rk2.extras.mover_boost = 1.0
    rk2.extras.overdrive = False               # 3-D producer: leave the LC pre-distortion out
    rk2.extras.z4 = float(defocus)
    rk2.extras.ifEnhanced = True

    # =====================================================================================
    # ROUND 2 -- the one-way axial grating walk, riding on the two-layer hologram
    # =====================================================================================
    rk3 = g().rearrange_kwargs3
    rk3.protocol = "pingponggrating"
    rk3.nsteps = int(walk_nsteps)
    rk3.step_period_ms = float(POST_PERIOD_MS)
    # initial_phase is SWAPPED because pingponggrating builds every frame on the cached INITIAL
    # phase, not on whatever is on the panel; skip_grid_derive keeps that swap free (same lattice,
    # no FFT).  Frame 0 is then the round-1 bookend itself -- continuous.
    rk3.initial_phase = FULL_PHASE
    rk3.skip_grid_derive = True
    rk3.extras.initial_phase_zernike = list(BAKED_ZERNIKE)
    rk3.extras.loading_zernike = [0.0, 0.0, 0.0, 0.0, float(defocus)]
    rk3.extras.z4 = float(defocus)
    rk3.extras.depth = True
    rk3.extras.true_defocus = False            # PARABOLIC ANSI Z4 -- step_size in RADIANS of
    #                                            PV Z4, the SAME unit the lift's z track uses
    # WALK SIGN is a swept axis when more than one is given.  Direction lives entirely in the
    # amplitude sign (the dispatcher's displacement index is non-negative), so sweeping the sign
    # IS sweeping the signed step.  A 2-D [lift corr x walk sign] map answers both questions at
    # once: which direction brings the far layer to the camera, and -- inside the column that
    # works -- what the lift's phase correction should be.
    _signs = [int(v) for v in (walk_sign if isinstance(walk_sign, (list, tuple))
                               else [walk_sign])]
    _step = LIFT_RAD / float(walk_nsteps)
    if len(_signs) == 1:
        rk3.extras.step_size = _signs[0] * _step
    else:
        rk3.extras.step_size.scan(2, [sg * _step for sg in _signs])
    rk3.extras.return_trip = False             # ONE WAY: rest at the walked plane for img4
    # The grating's OWN linear correction (rad of uniform phase per um of true-defocus step).
    # Independent of the lift's: different leg, different numeraire (per-um vs per-rad-of-Z4),
    # and it is GLOBAL (one uniform phase per frame) where the lift's is PER ATOM.  Several
    # values sweep it on dim 2, so a lift x grating map is one scan.
    _wcorr = [float(v) for v in (walk_corr if isinstance(walk_corr, (list, tuple))
                                 else [walk_corr])]
    if len(_wcorr) == 1:
        rk3.extras.depth_piston_corr = _wcorr[0]
    else:
        rk3.extras.depth_piston_corr.scan(2, _wcorr)
    rk3.extras.no_depth_piston = True
    rk3.extras.depth_fill_frac = None          # sticky legacy override -- clear it explicitly
    rk3.extras.piston = 0.0
    rk3.extras.hold_ms = 0.0
    # PRECOMPUTE, round 2.  On the pingponggrating path `precompute` is the STRONGEST form: it
    # fires the setup-time uint8 blaze-frame cache, so all nsteps+1 written frames exist before the
    # paced loop and the hot loop is pure SLM write (the effective period floor becomes the
    # Write_image latency alone).  `watermark` is inert for a dispatcher-special -- set anyway so
    # all three rounds carry the same staging declaration and none can silently be the odd one out.
    # NOTE the uint8 cache is dropped whenever the sticky protocol changes, which this scan does
    # every shot (round 0's setup re-asserts `rearrange`), so it is rebuilt once per shot inside
    # setup #3 -- before the walk, with the atoms held, not during the move.
    # PRECOMPUTE OFF -- with it on the grating builds its frames at SETUP time from the cached
    # WGS, i.e. before the lift has run, and replays a flat-array hologram over the 3-D state.
    # That was the root cause of the empty far layer (jobs 520-528).
    rk3.extras.precompute = False
    rk3.extras.precompute_host = False
    # Build the grating on the phase ACTUALLY LAST WRITTEN (the lift's final frame).
    rk3.extras.grating_base_last_phase = True
    rk3.extras.hw_sequence = False
    rk3.extras.skip_final_phase = True         # the grating writes no bookend anyway

    # ---- seq-local declarations (never forwarded to the server) ------------------------------
    # Without the mid image there is no frame in front of round 1, so its occupancy vector is
    # DECLARED: every every-other site is taken to be filled (round 0 is supply-saturated).  That
    # makes the lift a fixed geometric permutation, identical every shot.
    if mid_bits is not None:
        g().two_layer.mid_bits = [float(v) for v in mid_bits]

    # ---- run params (runp) -------------------------------------------------------------------
    rp.NumPerGroup = NUM_PER_GROUP
    rp.loading_defocus = float(defocus)
    n_images = 1 + (1 if mid_image else 0) + (1 if lift_image else 0) + (1 if do_walk else 0)
    if not do_walk and not lift_image:
        n_images += 1          # the lift is then the LAST round and is always read out
    rp.NumImages = n_images
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(mid_image=mid_image, do_walk=do_walk,
                                               lift_image=lift_image)

    frames = ((compact_nsteps + 1) + (lift_nsteps + 1)
              + ((walk_nsteps + 1) if do_walk else 0))
    meta = {
        "compact_nsteps": int(compact_nsteps), "lift_nsteps": int(lift_nsteps),
        "walk_nsteps": int(walk_nsteps),
        "compact_step_px": 24.499862677056605 / float(compact_nsteps),
        "lift_lat_step_px": LAT_STAGGER_PX / float(lift_nsteps),
        "lift_ax_step_rad": LIFT_RAD / float(lift_nsteps),
        "walk_step_rad": abs(LIFT_RAD / float(walk_nsteps)),
        "walk_step_um": abs(LIFT_UM / float(walk_nsteps)),
        "walk_corr": _wcorr, "walk_sign": _signs,
        "frames": frames, "ms": frames * float(period_ms), "n_targets": sum(bits),
        "mid_image": bool(mid_image), "do_walk": bool(do_walk),
        "lift_image": bool(lift_image), "n_images": n_images, "lift_corr": _corr,
    }
    return seq_name, g, meta


def _desc(meta):
    mid = (
        "DEBUG SHAPE, 4 frames: an extra image sits between the compaction and the lift so the "
        "compacted array can be seen on its own, and round 1's occupancy vector is MEASURED from "
        "it.  That image is NOT part of the measurement -- it costs the usual ~1-3%% per extra "
        "image on every atom downstream, so the demo runs --no-mid-image (3 frames), where the "
        "two rounds go back to back and round 1's vector is DECLARED as 'every every-other site "
        "is filled' (round 0 is supply-saturated, and the declaration makes the lift a fixed "
        "geometric permutation, identical every shot).  "
        if meta["mid_image"] else
        "DEMO SHAPE, 3 frames: no image between the compaction and the lift -- the two rounds run "
        "back to back and round 1's occupancy vector is DECLARED as 'every every-other site is "
        "filled' (round 0 is supply-saturated at ~640 atoms against 536 targets, and the "
        "declaration makes the lift a fixed geometric permutation, identical every shot).  Drops "
        "the ~1-3%% an extra image costs every atom downstream of it.  ")
    if not meta["do_walk"]:
        mid += (
            "ROUND 2 (the axial walk) AND ITS FRAME ARE OFF for this run: the far layer is still "
            "assembled and left 20 um away, just never imaged.  So this run answers only the "
            "cheapest useful question -- what does the assembly cost the atoms that DO NOT move, "
            "i.e. the stationary near layer -- and nothing in it measures the far layer's fill, "
            "the walk, or the grating's piston null.  Turn do_walk back on for those.  ")
    return mid + (
        "TWO-LAYER ASSEMBLY + BOTH-LAYER READOUT on %s, %d camera frames, three SLM rounds, and "
        "the FULL 1068-trap array present in every frame so per-trap depth is constant across all "
        "of them.  FRAME 1 = the loading array.  ROUND 0: 2-D warm-WGS compaction to the "
        "every-other checkerboard (536 targets, supply-saturated at ~640 loaded), bookend = the "
        "FULL loading WGS -- the dead traps are put back rather than dropped, which is what keeps "
        "every later frame at the loading frame's depth.  "
        "ROUND 1: the 3-D lift.  Target = a 1068-site two-layer WGS (816 sites at the loading "
        "plane = 532 dead + 284 near; 252 far sites at +%.1f um = %+.4f rad PV on the LIVE "
        "train3d.UM_PER_RAD = %.6f, staggered %.2f knm px = %.2f um into a triangular "
        "centred-rectangular lattice), with target_bits activating only the 284 near + 252 far "
        "sites so the 536 atoms map 1:1 -- the near half onto the sites it already occupies (zero "
        "cost), the odd/odd half up.  Mostly axial with a slight diagonal: %.2f um lateral against "
        "%.1f um axial.  Bookend = the two-layer WGS, again all 1068 traps.  The NEAR-LAYER FRAME "
        "is taken at the loading plane, where %.1f um is ~9 Rayleigh ranges, so only the near "
        "layer resolves: the atoms left behind, ~284 sites' worth, on the loading grid and "
        "thresholds.  ROUND 2: pingponggrating, "
        "true_defocus, ONE-WAY, %d x %.2f um = %.1f um, so the far layer arrives at the camera "
        "plane; %.2f um/step is 0.24x the measured axial cliff (3.4-3.5 um/step at 0.696 ms, "
        "2026-08-07 jobs 369/386) and well under the ~0.75x-cliff heating onset -- deliberately "
        "the slow leg.  The FAR-LAYER FRAME is detected on its own registry grid (fb11_2layer_far, "
        "252 sites).  nsteps: round 0 = %d (0.49 knm px/frame; measured 0.983 survival against a "
        "0.9885 detection floor, 2026-08-02 run #645), round 1 = %d (set by the LATERAL stagger at "
        "%.2f px/frame, under the 0.70 px 99.95%%/step stroke; the axial leg rides along at %.2f "
        "rad/step, ~4x inside the warm 3-D producer's free regime -- job 432, 0.000 +- 0.0005 "
        "loss/step out to 4.8 rad/step at depth_piston_corr = -0.5), round 2 = %d.  Transit %d "
        "frames x 0.696 ms = %.0f ms.  Each round pushes its OWN setup_rearrangement (protocol / "
        "pattern / target_bits / nsteps / phases are all sticky and the rounds disagree about all "
        "of them); round 0 re-asserts the loading phases every shot because round 2 leaves the "
        "cached initial_phase on the two-layer hologram and reload_rearrange would otherwise load "
        "into it.  Grid derives cost 40-75 ms each (measured).  Grating piston null %s rad/rad "
        "and amplitude sign %s are BOTH provisional: the measured Z4-PV null is 0.441 (job "
        "320), the null is direction-dependent, and the -z scan was never run -- a dark far-layer "
        "frame with a healthy near-layer one means flip the sign.  z4 = loading_defocus = %+.1f, "
        "ifEnhanced, prob_hungarian, overdrive target_clamp 0.2 on round 0 only, mover_boost off "
        "everywhere (boosting movers would dim the statics and break the constant-depth premise). "
        " READOUT: the near-layer frame gives the near layer's fill and the far-layer frame the "
        "far layer's, on disjoint atoms."
        % (PATTERN, meta["n_images"],
           LIFT_UM, LIFT_RAD, UM_PER_RAD, LAT_STAGGER_PX, LAT_STAGGER_PX * KNM_UM,
           LAT_STAGGER_PX * KNM_UM, LIFT_UM, LIFT_UM,
           meta["walk_nsteps"], meta["walk_step_um"], LIFT_UM, meta["walk_step_um"],
           meta["compact_nsteps"], meta["lift_nsteps"], meta["lift_lat_step_px"],
           meta["lift_ax_step_rad"], meta["walk_nsteps"], meta["frames"], meta["ms"],
           ", ".join("%.3f" % v for v in meta["walk_corr"]),
           "/".join("%+d" % v for v in meta["walk_sign"]), DEFOCUS))


def main():
    ap = argparse.ArgumentParser(
        description="Two-layer assembly (compaction -> 3-D lift -> axial walk), 4 frames.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--compact-nsteps", type=int, default=COMPACT_NSTEPS)
    ap.add_argument("--lift-nsteps", type=int, default=LIFT_NSTEPS)
    ap.add_argument("--walk-nsteps", type=int, default=WALK_NSTEPS)
    ap.add_argument("--walk-corr", type=float, nargs="+", default=[WALK_CORR], metavar="C",
                    help="the GRATING leg's linear correction, rad/um of true-defocus step "
                         "(default %g = provisional; the measured GLOBAL null is 0.617 and it is "
                         "direction-dependent). SEVERAL values sweep it on dim 2, giving a lift x "
                         "grating map -- note that costs n_lift * n_walk cells." % WALK_CORR)
    ap.add_argument("--walk-sign", type=int, nargs="+", default=[WALK_SIGN], choices=(-1, 1),
                    help="grating amplitude sign. ONE pins it; BOTH (-1 1) sweeps it on dim 2, "
                         "which is the direct test of which direction brings the far layer to "
                         "the camera plane -- the sign is a rig convention, not a theorem.")
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--lift-period", type=float, default=None,
                    help="pacing for the 3-D lift only (default: --period)")
    ap.add_argument("--defocus", type=float, default=DEFOCUS)
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--img-pid", type=float, nargs=2, metavar=("PID1", "PID2"), default=None,
                    help="force BlueMOT.Img1/Img2PIDSet via g(). Default: DON'T -- inherit the "
                         "live ByPattern/base value so a config-side imaging fix takes effect.")
    ap.add_argument("--no-hold-unpaired", dest="hold_unpaired", action="store_false",
                    help="turn OFF the hold_unpaired server patch on the lift.")
    ap.add_argument("--slmnet3d", dest="warm_wgs3d", action="store_false",
                    help="use the direct3d SLMnet model for the lift instead of warm-WGS 3D "
                         "(the default). Warm measured +0.09 better at both 10 and 20 um.")
    ap.add_argument("--lift-corr", type=float, nargs="+", default=None, metavar="C",
                    help="the lift's linear phase correction, rad per rad of PV quad-defocus "
                         "(frame k subtracts corr*dz[k,i] from atom i's phase). One value pins "
                         "it (default %g); SEVERAL sweep it on dim 1, interleaved. Suggested "
                         "sweep: %s -- brackets the inherited -0.5 ridge, includes 0 (the "
                         "uncorrected control) and one WRONG-SIGN point, which is the decisive "
                         "check that the term does anything at all."
                         % (DEPTH_PISTON_CORR_3D, LIFT_CORR_SWEEP))
    ap.add_argument("--no-lift-image", dest="lift_image", action="store_false",
                    help="drop the NEAR-layer frame, so the lift and the walk run back to back "
                         "and the last image is the FAR layer. This is the AXIAL-MOVE "
                         "measurement: survival follows the atoms that went up 20 um and were "
                         "walked back to the camera plane, not the ones left behind.")
    ap.add_argument("--no-walk", dest="do_walk", action="store_false",
                    help="drop ROUND 2 and its frame. The far layer is still assembled and left "
                         "20 um away, just never imaged -- what is left is the survival of the "
                         "STATIONARY near layer through the assembly, in 3 frames.")
    ap.add_argument("--no-mid-image", dest="mid_image", action="store_false",
                    help="THE DEMO SHAPE: drop the image between the compaction and the lift "
                         "(3 frames instead of 4). Round 1's occupancy vector is then declared "
                         "from the every-other mask instead of measured. Worth ~1-3%% survival on "
                         "every atom downstream.")
    ap.add_argument("--label", default="TwoLayerLiftTriangleRearrangeScan",
                    help="must contain 'rearrang' (Analysis-tab slm_diag name gate)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    bits, rec = target_bits()
    seq_name, g, meta = build(
        compact_nsteps=args.compact_nsteps, lift_nsteps=args.lift_nsteps,
        walk_nsteps=args.walk_nsteps, walk_corr=args.walk_corr, walk_sign=args.walk_sign,
        period_ms=args.period, defocus=args.defocus, lift_period_ms=args.lift_period, bits=bits,
        mid_image=args.mid_image, do_walk=args.do_walk, lift_image=args.lift_image,
        img_pid=args.img_pid, lift_corr=args.lift_corr, warm_wgs3d=args.warm_wgs3d,
        hold_unpaired=args.hold_unpaired)
    g.runp().NumPerGroup = int(args.num_per_group)

    names = (["load"] + (["compacted"] if meta["mid_image"] else [])
             + (["near"] if (meta["lift_image"] or not meta["do_walk"]) else [])
             + (["far"] if meta["do_walk"] else []))
    print("seq=%s  nseq=%d  (single point; --reps sets the shot count)" % (seq_name, g.nseq()))
    print("  images          : %d (%s); round-1 vector %s"
          % (meta["n_images"], " / ".join(names),
             "MEASURED" if meta["mid_image"] else "DECLARED"))
    if not meta["do_walk"]:
        print("  ROUND 2 OFF     : far layer assembled but never imaged -- this run measures the "
              "STATIONARY near layer only")
    print("  two-layer phase : %s (target_bits %d of %d active)"
          % (rec["phase"], meta["n_targets"], len(bits)))
    print("  lift            : %+.1f um = %+.4f rad PV (live %.6f um/rad); lateral stagger "
          "%.2f knm px = %.2f um" % (LIFT_UM, LIFT_RAD, UM_PER_RAD,
                                     LAT_STAGGER_PX, LAT_STAGGER_PX * KNM_UM))
    print("  round0 compact  : n=%3d -> %.2f knm px/frame"
          % (meta["compact_nsteps"], meta["compact_step_px"]))
    print("  round1 lift     : n=%3d -> %.2f px/frame lateral, %.2f rad/step (%.2f um/step) axial"
          % (meta["lift_nsteps"], meta["lift_lat_step_px"], meta["lift_ax_step_rad"],
             meta["lift_ax_step_rad"] * UM_PER_RAD))
    print("  round2 walk     : %s"
          % (("n=%3d -> %.4f rad/step of PV Z4 (= %.3f um), sign %s, corr %s rad/rad"
              % (meta["walk_nsteps"], meta["walk_step_rad"], meta["walk_step_um"],
                 "/".join("%+d" % v for v in meta["walk_sign"]),
                 ", ".join("%.3f" % v for v in meta["walk_corr"])))
             if meta["do_walk"] else "OFF (not run this scan)"))
    print("  transit         : %d frames x %.3f ms = %.0f ms"
          % (meta["frames"], args.period, meta["ms"]))
    if args.dry_run:
        s0 = g.getseq(0)
        for name in ("rearrange_kwargs", "rearrange_kwargs2", "rearrange_kwargs3"):
            k = s0[name]
            e = k.get("extras", {})
            print("  %-18s protocol=%-16s nsteps=%-3s init=%s final=%s"
                  % (name, k.get("protocol"), k.get("nsteps"),
                     k.get("initial_phase"), k.get("final_phase")))
            print("  %-18s pattern=%r target_bits=%s planes=%s wgs3d=%s corr=%s"
                  % ("", e.get("pattern"),
                     ("<%d ones/%d>" % (sum(e["target_bits"]), len(e["target_bits"])))
                     if e.get("target_bits") else e.get("target_bits"),
                     e.get("target_grid_planes_z_rad"), e.get("wgs3d_warm"),
                     e.get("depth_piston_corr")))
        print("  imagePatterns   : %s"
              % [p["name"] for p in json.loads(g.runp().imagePatternsJson())])
        return

    if "rearrang" not in args.label.lower():
        ap.error("label must contain 'rearrang' (Analysis-tab slm_diag name gate)")
    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label=args.label,
                      description=_desc(meta), **opts)
    print("submitted -> descriptor id %s" % did)
    return did


if __name__ == "__main__":
    main()
