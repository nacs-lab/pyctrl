"""TwoLayerAxialProbeScan.py -- STAGE 1 of the two-layer campaign: does SITE-SELECTIVE AXIAL
movement work at all, and in which direction?

Deliberately the simplest thing that can answer that.  No two-layer hologram, no assignment round
for the lift, no new detection pattern -- because every one of those was a way for the measurement
to fail for reasons that are not the physics:

  * the lift is ``pingpong`` + ``move_idx`` + ``step_size_z`` + ``oneway``, i.e. a UNIFORM axial
    kick applied to a chosen SUBSET.  The distance is then a scalar (``step_size_z * nsteps``) and
    sweeps cleanly, instead of being baked into a hologram that needs solving and verifying per
    distance;
  * ``ghost_fraction = 1.0`` keeps every non-mover emitted STATIONARY in each transit frame, so all
    1068 traps stay on the panel throughout and per-trap depth is constant across every image;
  * ``oneway = True`` sets ``skip_final_phase``, so NOTHING is re-displayed after the move -- the
    panel keeps the transit's own final frame.  Writing a WGS bookend there is a different
    hologram for the same atoms, i.e. a discontinuous jump imposed exactly when they are most
    fragile, and it is the leading suspect for the far layer coming back empty in jobs 520-522;
  * the move is PURELY AXIAL, so the lifted sites keep their xy and every frame detects on the
    existing ``33x33_feedback11`` grid and thresholds.

THE SHOT (3 frames, 3 SLM rounds)

  img1     the loading array, 1068 sites.
  ROUND 0  2-D warm-WGS compaction -> the every-other checkerboard (536 targets), bookend = the
           FULL loading WGS.  Unchanged from the assembly scan and already measured healthy
           (fill 0.987, 2026-08-07 job 520).
  img2     the compacted array, same grid + thresholds.
  ROUND 1  the LIFT: the odd/odd half (252 sites) is kicked +D axially over ``lift_nsteps``
           frames while everything else is held still.  No bookend.
  ROUND 2  the WALK: ``pingponggrating`` translates the WHOLE hologram by -D (or +D -- see the
           relative-sign axis), riding on the phase ACTUALLY LAST WRITTEN rather than a cached
           WGS (server extra ``grating_base_last_phase``, added 2026-08-07 and gated OFF by
           default, so nothing else changes behaviour).
  img3     if the lift and the walk cancel, the lifted atoms are back at the camera plane and the
           held ones are D away; the odd/odd sites light up and the even/even ones do not.

THE TWO AXES (one paired dim-1 axis of ``n_dist * n_sign`` cells, interleaved by Scramble)

  DISTANCE moves BOTH legs together, so they always cancel: lift = +D, walk = -relsign * D.
  RELATIVE SIGN is the only thing that changes their relation:
      +1  the walk OPPOSES the lift  -> lifted atoms return to the camera plane (expect signal)
      -1  the walk FOLLOWS the lift  -> lifted atoms end at 2D, held ones at -D (expect nothing)
  The -1 column is the control that makes the +1 column mean something: if BOTH are dark the
  atoms never survived the lift, and if BOTH are bright the readout is not layer-selective.

WHAT COMES NEXT (do not conflate the stages)
  This measures the axial move ONLY.  Once it works, stage 2 adds ``ghost_fraction``/``full_n`` to
  the 3-D ASSIGNMENT protocol (``_protocol_rearrange`` has neither today -- verified live) so the
  full diagonal-to-triangular-lattice rearrangement can keep its dead traps in place too, and the
  campaign graduates to that.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/TwoLayerAxialProbeScan.py --dry-run
    python YbScans/RearrangeDiagnostics/TwoLayerAxialProbeScan.py --reps 18 --force
    python YbScans/RearrangeDiagnostics/TwoLayerAxialProbeScan.py --model --reps 18 --force
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(PYCTRL)

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
MODEL_3D = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
DEFOCUS = -5.0

UM_PER_RAD = 0.7974368729415139      # live train3d.UM_PER_RAD
GEOM_NPZ = os.path.join(ROOT, "_twolayer_lift", "two_layer_full.npz")
EO_BITS_JSON = os.path.join(ROOT, "_twolayer_lift", "eo_bits.json")

# ---- the axes ---------------------------------------------------------------------------
DIST_UM = [1.0, 5.0, 10.0, 15.0, 20.0]
REL_SIGN = [1, -1]                   # +1 walk opposes the lift (expect signal); -1 follows it

# ---- pacing / steps ---------------------------------------------------------------------
PERIOD_MS = 0.696                    # round 0 stays at the SLM write floor
POST_PERIOD_MS = 2.0                 # legs 2 and 3
COMPACT_NSTEPS = 50
LIFT_NSTEPS = 40
WALK_NSTEPS = 50
LIFT_CORR = -0.5                     # per-atom, rad per rad of PV quad-defocus (measured ridge)
WALK_CORR = 0.441                    # global, rad per rad of Z4-PV (measured grating null)
WGS3D_Z_MAX = 25.6                   # pinned >= max |z| so the kernel table is built ONCE
WGS3D_RADIUS_FRAC = 0.7

BLUE_DETUNING_MHZ = -48.0
BLUE_LOADING_TIME_S = 0.467
GREEN_BIAS_X_A = 0.0343
# Deliberately SMALL: this is a fast-iteration diagnostic scan and --reps sets the real shot
# count.  A large NumPerGroup only matters when --reps is forgotten, and then it runs away
# (4000 shots ~ 5.5 h at 5 s/shot).  Raise it explicitly when a statistics run is wanted.
NUM_PER_GROUP = 300


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def geometry():
    """``(move_idx, eo_bits)``: the far (odd/odd) site indices to lift, and the every-other mask
    round 0 targets.  Both are in the SERVER'S init_grid index order -- the loading grid derives
    to the registry coords 1:1 (max residual 0.031 knm px, verified 2026-08-07), and the mask was
    emitted through that same mapping by ``campaigns/geometry/twolayer_lift/build_phase.py``."""
    g = np.load(GEOM_NPZ)
    move_idx = np.where(g["far"])[0].astype(int).tolist()
    with open(EO_BITS_JSON) as fh:
        eo = [int(v) for v in json.load(fh)["bits"]]
    return move_idx, eo


def cells(dist_um=None, rel_sign=None):
    """The paired dim-1 axis: every (distance, relative sign) combination, flattened.

    Flattened rather than a 2-D product because BOTH legs' step sizes are derived from the SAME
    pair -- the lift's ``step_size_z`` and the grating's ``step_size`` have to move together, and a
    2-D grid cannot express one value derived from two axes.  Flat also sidesteps the
    column-major reshape trap when this is analysed."""
    ds = [float(d) for d in (dist_um or DIST_UM)]
    ss = [int(s) for s in (rel_sign or REL_SIGN)]
    return [(d, s) for s in ss for d in ds]


def build(dist_um=None, rel_sign=None, lift_nsteps=LIFT_NSTEPS, walk_nsteps=WALK_NSTEPS,
          compact_nsteps=COMPACT_NSTEPS, period_ms=PERIOD_MS, post_period_ms=POST_PERIOD_MS,
          defocus=DEFOCUS, warm=True, lift_corr=LIFT_CORR, walk_corr=WALK_CORR,
          do_walk=True, walk_um=None, lift_oneway=True, walk_precompute=False,
          walk_nsteps_sweep=None, walk_true_defocus=False, walk_return_trip=False,
          mid_image=False, lift_image=True):
    """Build (do NOT submit) the ScanGroup.  Returns ``(seq_name, g, meta)``."""
    _bootstrap()
    from scan_group import ScanGroup

    move_idx, eo = geometry()
    cl = cells(dist_um, rel_sign)
    # Both legs derive from the same (D, sign) cell, as PARALLEL dim-1 lists.
    lift_step = [(d / UM_PER_RAD) / float(lift_nsteps) for d, _s in cl]
    # walk_um decouples the WALK distance from the lift distance -- a FOCUS SCAN: hold the lift
    # at D and ask which walk brings the moved atoms back into the camera plane.  If they are
    # alive at +D the return peaks at walk == D; if nothing peaks anywhere they were already
    # lost during the lift.  Default (None) keeps the two locked together.
    _wu = [float(v) for v in (walk_um if walk_um else [])]
    _wn = [int(v) for v in (walk_nsteps_sweep if walk_nsteps_sweep else [])]
    walk_step = [-s * (d / UM_PER_RAD) / float(walk_nsteps) for d, s in cl]

    seq_name = "TwoLayerLiftCommSeq"
    g = ScanGroup()

    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_2D if warm else MODEL_3D
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
    rp.warmup_kwargs.derive_threshold = 0.45   # 0.35 admits a sidelobe (see build_phase.py)

    g().BlueMOT.FreqDetuning = BLUE_DETUNING_MHZ * 1e6
    g().BlueMOT.LoadingTime = BLUE_LOADING_TIME_S
    g().GreenMOT.BiasCoilCurrent.X = GREEN_BIAS_X_A
    # Imaging is NOT pinned -- inherit ByPattern so a config-side fix takes effect.

    # ================= ROUND 0 -- 2-D compaction to every_other ==========================
    rk = g().rearrange_kwargs
    rk.protocol = "rearrange"
    rk.nsteps = int(compact_nsteps)
    rk.step_period_ms = float(period_ms)
    rk.initial_phase = PHASE_PATH
    rk.final_phase = PHASE_PATH            # bookend = the FULL loading array
    rk.extras.n_rounds = 1
    rk.extras.target_bits = list(eo)       # targets via the mask, NEVER `pattern` (they are
    #                                        mutually exclusive and neither clears the other)
    rk.extras.target_grid_planes_z_rad = None
    rk.extras.prob_hungarian = True
    rk.extras.wgs_warm = True
    rk.extras.wgs_pad = 2048
    rk.extras.wgs_iters = 3
    rk.extras.wgs_beam = "gaussian"
    rk.extras.wgs3d_warm = False
    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.watermark = 512
    rk.extras.hw_sequence = False
    rk.extras.skip_final_phase = False
    rk.extras.overdrive = True
    rk.extras.target_clamp = 0.2
    rk.extras.tau_rise_ms = 1.7
    rk.extras.tau_fall_ms = 3.7
    rk.extras.settle_lut = "auto"
    rk.extras.mover_boost = 1.0
    rk.extras.random_z = False
    rk.extras.random_z_max = 0.0
    rk.extras.hold_ms = 0.0
    rk.extras.ifEnhanced = True
    rk.extras.z4 = float(defocus)
    rk.extras.initial_pattern = PATTERN
    rk.extras.middle_pattern = PATTERN
    rk.extras.lift_pattern = PATTERN
    rk.extras.final_pattern = PATTERN      # every frame is the SAME grid (purely axial move)
    # FRAME SET (2026-08-07, post-verification).  The compaction is now trusted (fill 0.987
    # measured repeatedly), so its verify frame is dropped: every image costs ~1-3% on every atom
    # downstream of it, and it is not part of the measurement any more.  What is left images BOTH
    # layers -- the near layer after the lift ("what stayed") and the far layer after the global
    # transport ("what moved") -- which is the demo readout.
    #   mid_image  = 0  no post-compaction frame; round 0 and the lift share one handoff
    #   lift_image = 1  the NEAR layer, at the loading plane
    #   do_walk    = 1  the FAR layer, after the grating brings it to the camera plane
    # CONSEQUENCE for analysis: with no verify frame there is no per-shot MEASURED occupancy
    # between the compaction and the lift, so mover survival loses its natural denominator.  The
    # metric becomes absolute FILL of each layer (n_atoms / 252 far, / 284 near) rather than a
    # conditional survival.  Round 1's bit vector is then the DECLARED every-other mask.
    rk.extras.mid_image = 1 if mid_image else 0
    rk.extras.lift_image = 1 if lift_image else 0
    rk.extras.do_walk = 1 if do_walk else 0

    # ================= ROUND 1 -- the site-selective AXIAL lift ==========================
    rk2 = g().rearrange_kwargs2
    rk2.protocol = "pingpong"
    rk2.nsteps = int(lift_nsteps)
    rk2.step_period_ms = float(post_period_ms)
    rk2.extras.move_idx = list(move_idx)   # ONLY these sites move
    rk2.extras.ghost_fraction = 1.0        # keep every non-mover emitted, stationary -> the full
    #                                        1068-trap array stays on the panel, constant depth
    rk2.extras.full_n = False
    # oneway=False makes the LIFT itself an out-and-back triangle: the movers return to the
    # loading plane by the SAME per-site mechanism that took them out, with no grating involved.
    # That is the clean isolation -- if they come back alive the lift works and the problem is
    # the handoff to the grating; if they do not, the lift is what kills them.
    rk2.extras.oneway = bool(lift_oneway)
    rk2.extras.skip_final_phase = True     # EXPLICIT: nothing is re-displayed after the move
    rk2.extras.step_size = 0.0             # no lateral component in stage 1
    rk2.extras.direction = 0.0
    rk2.extras.step_size_z.scan(1, [float(v) for v in lift_step])
    rk2.extras.depth3d = True
    rk2.extras.random_z = False
    rk2.extras.random_z_max = 0.0
    rk2.extras.piston = 0.0
    # The lift's per-step phase correction.  A LIST sweeps it on dim 2 (dim 1 is the (D, sign)
    # axis), so a single-cell dim 1 + a corr list is a clean 1-D correction scan.
    _lc = [float(v) for v in (lift_corr if isinstance(lift_corr, (list, tuple)) else [lift_corr])]
    if len(_lc) == 1:
        rk2.extras.depth_piston_corr = _lc[0]
    else:
        rk2.extras.depth_piston_corr.scan(2, _lc)
    rk2.extras.lateral_piston_corr = 0.0   # inert at step_size 0; pinned so it cannot inherit
    rk2.extras.true_defocus = False        # sticky grating-only flag: keep the lift in rad-of-Z4
    rk2.extras.wgs3d_warm = bool(warm)
    if not warm:
        rk2.extras.gpu_target_graph = True
    rk2.extras.wgs3d_z_max = float(WGS3D_Z_MAX)
    rk2.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)
    rk2.extras.precompute = True
    rk2.extras.precompute_host = True
    rk2.extras.hold_ms = 0.0
    rk2.extras.z4 = float(defocus)

    # ================= ROUND 2 -- the global axial grating walk ==========================
    rk3 = g().rearrange_kwargs3
    rk3.protocol = "pingponggrating"
    if not _wn:
        rk3.nsteps = int(walk_nsteps)   # fixed unless the nsteps sweep owns this axis
    rk3.step_period_ms = float(post_period_ms)
    rk3.initial_phase = PHASE_PATH
    rk3.skip_grid_derive = True
    # THE POINT: build the grating on the phase actually last written -- the lift's final transit
    # frame -- instead of a cached WGS.  Gated server-side (2026-08-07); absent -> old behaviour.
    rk3.extras.grating_base_last_phase = True
    rk3.extras.depth = True
    # MAP CHOICE.  False = parabolic ANSI Z4 (step in RAD, same unit as the lift, so the legs
    # cancel exactly but the map carries a rho^4 residual that grows with excursion).  True =
    # EXACT spherical (step in MICRONS, no rho^4 error -- what the 2026-07 axial campaign moved to).
    # With true_defocus the correction numeraire changes too: rad/um, null 0.617, not 0.441.
    rk3.extras.true_defocus = bool(walk_true_defocus)
    if walk_true_defocus:
        # exact-spherical: step_size is in MICRONS, so rebuild both legs' walk steps in um
        walk_step = [-s * d / float(walk_nsteps) for d, s in cl]
    if _wn:
        # WALK-NSTEPS SWEEP: nsteps PAIRED with step_size on dim 2 so the DISTANCE is constant and
        # only the pacing changes.  More steps = smaller per-step BUT longer spent off-plane, so
        # this separates per-step transport loss from time-off-plane loss.
        _d0, _s0 = cl[0]
        rk3.nsteps.scan(2, [int(k) for k in _wn])
        _tot = _d0 if walk_true_defocus else (_d0 / UM_PER_RAD)
        rk3.extras.step_size.scan(2, [-_s0 * _tot / float(k) for k in _wn])
    elif _wu:
        # FOCUS SCAN: walk magnitude on dim 2, independent of the lift's dim-1 distance.
        _s0 = cl[0][1]
        rk3.extras.step_size.scan(2, [-_s0 * (u / UM_PER_RAD) / float(walk_nsteps)
                                      for u in _wu])
    else:
        rk3.extras.step_size.scan(1, [float(v) for v in walk_step])
    # return_trip=True walks OUT and BACK, resting at displacement 0.  With the lift set to
    # zero amplitude that is the grating's cost on a FLAT, in-plane array through the SAME
    # base-phase handoff -- that is, it separates 'the grating is broken here' from 'the
    # grating cannot cleanly translate spots that are already off-plane'.
    rk3.extras.return_trip = bool(walk_return_trip)
    _wc = [float(v) for v in (walk_corr if isinstance(walk_corr, (list, tuple)) else [walk_corr])]
    if len(_wc) == 1:
        rk3.extras.depth_piston_corr = _wc[0]
    else:
        rk3.extras.depth_piston_corr.scan(2, _wc)
    rk3.extras.no_depth_piston = True
    rk3.extras.depth_fill_frac = None
    rk3.extras.piston = 0.0
    rk3.extras.hold_ms = 0.0
    # PRECOMPUTE MUST BE OFF ON THE GRATING LEG.  With it on, pingponggrating builds its uint8
    # frame cache at SETUP TIME from the cached initial_phase -- i.e. BEFORE the lift has run --
    # so the frames replay the 33x33 WGS over the 3-D configuration and the gated
    # grating_base_last_phase (a RUNTIME path) is bypassed entirely.  Symptom: even a
    # ZERO-amplitude walk costs half the movers (0.123 vs 0.228 with no grating at all, job 528),
    # i.e. the damage is the phase being written, not the motion.
    rk3.extras.precompute = bool(walk_precompute)
    rk3.extras.precompute_host = bool(walk_precompute)
    rk3.extras.hw_sequence = False
    rk3.extras.skip_final_phase = True
    rk3.extras.z4 = float(defocus)
    rk3.extras.loading_zernike = [0.0, 0.0, 0.0, 0.0, float(defocus)]

    # The lift and the walk share ONE handoff (no near-layer frame), so the TRAILING round
    # has no camera frame in front of it and the seq requires a declared vector for it.
    # pingponggrating ignores its bits entirely -- this only has to be the right LENGTH
    # (len(init_grid) = 1068); without it every shot cancels.
    if not (mid_image and lift_image):
        g().two_layer.mid_bits = [float(v) for v in eo]

    rp.NumPerGroup = NUM_PER_GROUP
    rp.loading_defocus = float(defocus)
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    _frames_pat = [it]                                   # load, always
    if mid_image:
        _frames_pat.append(dict(it))                     # post-compaction verify
    if lift_image or not do_walk:
        _frames_pat.append(dict(it))                     # NEAR layer
    if do_walk:
        _frames_pat.append(dict(it))                     # FAR layer (same grid: axial only)
    rp.imagePatternsJson = json.dumps(_frames_pat)
    rp.NumImages = len(_frames_pat)

    frames = (compact_nsteps + 1) + (lift_nsteps + 1) + (walk_nsteps + 1)
    meta = {"cells": cl, "lift_step": lift_step, "walk_step": walk_step, "warm": bool(warm),
            "lift_corr": _lc, "walk_corr": _wc, "walk_um": _wu,
            "n_move": len(move_idx), "n_eo": sum(eo), "frames": frames,
            "lift_nsteps": lift_nsteps, "walk_nsteps": walk_nsteps,
            "ms": (compact_nsteps + 1) * period_ms
                  + ((lift_nsteps + 1) + (walk_nsteps + 1)) * post_period_ms}
    return seq_name, g, meta


def _desc(meta):
    return (
        "STAGE 1 of the two-layer campaign -- does SITE-SELECTIVE AXIAL movement work, and in "
        "which direction?  Three frames on ONE grid (%s): img1 load, ROUND 0 warm-WGS compaction "
        "to the every-other checkerboard (%d targets, bookend = the FULL loading WGS), img2 "
        "compacted, ROUND 1 the LIFT, ROUND 2 the WALK, img3.  The lift is pingpong + move_idx "
        "(%d odd/odd sites) + step_size_z + oneway: a UNIFORM axial kick on a chosen SUBSET, so "
        "the distance is a scalar and no two-layer hologram, target grid or extra detection "
        "pattern is involved -- each of which was a way for this to fail for non-physics reasons. "
        "ghost_fraction=1.0 keeps every non-mover emitted stationary, so all 1068 traps stay on "
        "the panel and per-trap depth is constant across all three images.  oneway sets "
        "skip_final_phase, so NOTHING is re-displayed after the move: the panel keeps the "
        "transit's own final frame.  That is the fix under test -- writing a WGS bookend there is "
        "a different hologram for the same atoms, and is the leading suspect for the far layer "
        "returning empty in jobs 520-522 (near layer survived 0.894 while the far layer read "
        "0.027, i.e. exactly the atoms whose 3-D phases a fresh hologram has to reproduce).  The "
        "grating then rides on the phase ACTUALLY LAST WRITTEN via the server extra "
        "grating_base_last_phase (added 2026-08-07, gated OFF by default so no other run "
        "changes).  Purely axial, so the lifted sites keep their xy and every frame detects on "
        "the existing %s grid + thresholds.  ONE paired dim-1 axis of %d cells: distance %s um "
        "moves BOTH legs together (lift +D, walk -relsign*D, so they cancel exactly -- both in "
        "radians of PV Z4, no um/rad conversion between them) and RELATIVE SIGN %s is the only "
        "thing that changes their relation.  +1 = the walk opposes the lift, lifted atoms return "
        "to the camera plane (expect the odd/odd sites to light up and the even/even ones not to); "
        "-1 = the walk follows it, lifted atoms end at 2D and held ones at -D (expect nothing).  "
        "The -1 column is what makes the +1 column mean something: both dark = the atoms never "
        "survived the lift; both bright = the readout is not layer-selective.  Producer: %s.  "
        "lift n=%d, walk n=%d, legs 2-3 at 2.0 ms (round 0 at the 0.696 ms write floor), "
        "depth_piston_corr -0.5 per-atom on the lift and 0.441 global on the grating, precompute "
        "everywhere, z4 = loading_defocus = %+.1f, ifEnhanced.  Transit %d frames ~ %.0f ms."
        % (PATTERN, meta["n_eo"], meta["n_move"], PATTERN, len(meta["cells"]),
           sorted({d for d, _ in meta["cells"]}), sorted({s for _, s in meta["cells"]}),
           "warm matched-filter 3-D WGS" if meta["warm"] else "direct3d SLMnet model",
           meta["lift_nsteps"], meta["walk_nsteps"], DEFOCUS, meta["frames"], meta["ms"]))


def main():
    ap = argparse.ArgumentParser(description="Stage 1: site-selective axial move + grating return.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--dist-um", type=float, nargs="+", default=None)
    ap.add_argument("--rel-sign", type=int, nargs="+", default=None, choices=(-1, 1))
    ap.add_argument("--lift-nsteps", type=int, default=LIFT_NSTEPS)
    ap.add_argument("--walk-nsteps", type=int, default=WALK_NSTEPS)
    ap.add_argument("--post-period", type=float, default=POST_PERIOD_MS)
    ap.add_argument("--lift-corr", type=float, nargs="+", default=[LIFT_CORR], metavar="C",
                    help="lift per-step phase correction, rad per rad of PV quad-defocus. "
                         "SEVERAL values sweep it on dim 2.")
    ap.add_argument("--walk-corr", type=float, nargs="+", default=[WALK_CORR], metavar="C",
                    help="grating correction, rad per rad of Z4-PV. SEVERAL sweep it on dim 2.")
    ap.add_argument("--no-walk", dest="do_walk", action="store_false",
                    help="LIFT ONLY: drop the grating leg and image at the loading plane. "
                         "The non-movers never move, so any loss they take is hologram "
                         "damage from the lifted spots -- not transport.")
    ap.add_argument("--verify-frame", dest="mid_image", action="store_true",
                    help="re-enable the post-compaction verify frame (default OFF).")
    ap.add_argument("--no-near-frame", dest="lift_image", action="store_false",
                    help="drop the NEAR-layer frame after the lift.")
    ap.add_argument("--walk-return-trip", action="store_true",
                    help="grating walks OUT AND BACK (rests at 0). With --dist-um 0 this "
                         "measures the grating cost on a FLAT array through the same "
                         "handoff.")
    ap.add_argument("--walk-true-defocus", action="store_true",
                    help="grating on the EXACT spherical map (step in um, corr in rad/um, "
                         "null 0.617) instead of the parabolic ANSI Z4.")
    ap.add_argument("--walk-nsteps-sweep", type=int, nargs="+", default=None, metavar="N",
                    help="sweep the WALK nsteps on dim 2 at constant distance -- more "
                         "steps = smaller per-step but longer off-plane.")
    ap.add_argument("--walk-precompute", action="store_true",
                    help="re-enable the grating setup-time frame cache (DEFAULT OFF: it "
                         "builds frames from the cached WGS before the lift runs).")
    ap.add_argument("--lift-roundtrip", dest="lift_oneway", action="store_false",
                    help="make the LIFT out-and-back (oneway=False) so the movers return by the "
                         "same mechanism -- isolates the lift from the grating handoff.")
    ap.add_argument("--walk-um", type=float, nargs="+", default=None, metavar="U",
                    help="FOCUS SCAN: walk distance(s) in um, decoupled from the lift distance "
                         "and swept on dim 2. Finds where the moved atoms actually are.")
    ap.add_argument("--model", action="store_true",
                    help="use the direct3d SLMnet model for the lift instead of warm 3-D WGS")
    ap.add_argument("--label", default=None, help="must contain 'rearrang'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seq_name, g, meta = build(dist_um=args.dist_um, rel_sign=args.rel_sign,
                              lift_nsteps=args.lift_nsteps, walk_nsteps=args.walk_nsteps,
                              post_period_ms=args.post_period, warm=not args.model,
                              do_walk=args.do_walk, walk_um=args.walk_um,
                              lift_oneway=args.lift_oneway,
                              walk_precompute=args.walk_precompute,
                              walk_nsteps_sweep=args.walk_nsteps_sweep,
                              walk_true_defocus=args.walk_true_defocus,
                              walk_return_trip=args.walk_return_trip,
                              mid_image=args.mid_image, lift_image=args.lift_image,
                              lift_corr=args.lift_corr, walk_corr=args.walk_corr)
    print("seq=%s  nseq=%d  producer=%s" % (seq_name, g.nseq(),
                                            "warm 3-D WGS" if meta["warm"] else "direct3d model"))
    print("  movers %d of 1068 (odd/odd); round-0 targets %d" % (meta["n_move"], meta["n_eo"]))
    print("  lift n=%d, walk n=%d, legs 2-3 at %.2f ms; transit %d frames ~ %.0f ms"
          % (meta["lift_nsteps"], meta["walk_nsteps"], args.post_period,
             meta["frames"], meta["ms"]))
    print("   D(um)  sign   lift step_size_z    walk step_size   (rad of PV Z4)")
    for (d, s), a, b in zip(meta["cells"], meta["lift_step"], meta["walk_step"]):
        print("   %5.1f  %+d      %+8.4f          %+8.4f" % (d, s, a, b))
    if args.dry_run:
        s0 = g.getseq(0)
        for nm in ("rearrange_kwargs", "rearrange_kwargs2", "rearrange_kwargs3"):
            k = s0[nm]; e = k.get("extras", {})
            print("  %-18s protocol=%-16s nsteps=%s" % (nm, k.get("protocol"), k.get("nsteps")))
        e2 = s0["rearrange_kwargs2"]["extras"]
        print("  lift: move_idx=%d sites ghost_fraction=%s oneway=%s skip_final=%s wgs3d_warm=%s"
              % (len(e2["move_idx"]), e2.get("ghost_fraction"), e2.get("oneway"),
                 e2.get("skip_final_phase"), e2.get("wgs3d_warm")))
        e3 = s0["rearrange_kwargs3"]["extras"]
        print("  walk: grating_base_last_phase=%s true_defocus=%s corr=%s"
              % (e3.get("grating_base_last_phase"), e3.get("true_defocus"),
                 e3.get("depth_piston_corr")))
        return
    label = args.label or ("TwoLayerAxialProbe%sRearrangeScan"
                           % ("Warm" if meta["warm"] else "Model"))
    if "rearrang" not in label.lower():
        ap.error("label must contain 'rearrang'")
    if not args.force:
        ap.error("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label=label, description=_desc(meta), **opts)
    print("submitted %s -> descriptor id %s" % (label, did))
    return did


if __name__ == "__main__":
    main()
