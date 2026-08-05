"""RandomZAxialRearrangeStepSweep.py -- SITE-SPECIFIC random axial (depth) pingpong.

Every MOVING tweezer gets its OWN signed per-step depth kick, drawn uniformly from
``[-random_z_max, +random_z_max]`` radians of PV quad-defocus (``2*rho^2 - 1``, the same
unit as ``pingponggrating``'s depth ``step_size``; ~0.9 um/rad, but NOTHING here is
converted to um -- the whole scan lives in radians).  The draw is HELD CONSTANT across the
out-and-back triangle, so site i walks ``disp_idx[k] * s_i`` and every atom returns to its
own start depth.  Because the draw is keyed by INIT-GRID SITE at a fixed
``random_z_seed``, a given site keeps the same ``s_i`` shot after shot: ONE shot samples
the whole survival-vs-step-size curve at once (~500 movers spread over the full range),
and the whole scan accumulates per-site statistics in stable Delta-z bins.

    ONE axis, PAIRED (nsteps and the draw half-width move together):
        rearrange_kwargs.nsteps  =  2,  4,  8,   15,   25,  40   (-> 2*nsteps steps out+back)
        extras.random_z_max      = 12, 12, 12,  6.4, 3.84, 2.4   (rad/step)
        extras.depth_piston_corr = -0.5 everywhere (the measured ridge -- see below)

    The draw half-width is capped per cell at ``Z_PEAK_CAP / nsteps`` because the PEAK
    excursion is ``nsteps * s_i``: 12 rad/step at nsteps=40 would be 480 rad (~430 um) of
    axial travel, and the warm producer's kernel-table patch radius grows linearly with the
    table's z_max (solve cost ~ S^2), so the cap is what makes large nsteps affordable at
    all.  96 rad is set by that cost, with ``wgs3d_radius_frac`` 0.7 buying back ~2x.

    => 6 points, survival img1 -> img2 at the SOURCE sites (two-way pingpong returns every
       atom to where it started; no target assignment).  ~86 shots/cell x ~650 movers =
       ~56k atom-events per nsteps, so |s| can be binned finely out to the cap.

THE PHASE CORRECTION (``depth_piston_corr``, now FIXED at the measured ridge).  Encoding defocus advances a trap's
far-field phase: the panel map ``2*rho^2 - c`` leaves a per-step uniform advance that
``pingponggrating`` nulls with ``depth_piston_corr`` (measured 0.441 rad per rad of Z4-PV
there; the 0.617 number is the *rad/um* null of the exact-spherical map and does NOT apply
to a rad-of-Z4 axis).  The model / warm-WGS path commands per-spot phase, so its version is
PER ATOM: frame k subtracts ``corr * dz[k, i]`` using THAT atom's own accumulated axial
displacement -- which is the only form that can work here, since under ``random_z`` every
atom defocuses by a different amount and one global ``piston`` cannot track them.  Left at
the server default 0.0 the run is the UNCOMPENSATED case: a per-site RANDOM phase transient
of ~0.5*s_i rad/step riding on the axial motion, well past the ~0.8 rad/step where pure
piston transients start killing atoms (2026-07-22 campaign).  That baseline is MEASURED and
COMPLETE: job 414, scan data_20260731_014411, 504 shots / 264k atom-events at corr = 0.

MEASURED (jobs 416 + 418, warm arm; axes {-0.5, 0, +0.5} then {-0.9,-0.7,-0.5,-0.3}): the ridge is
NEGATIVE and large.  Pooled survival over nsteps>=1 was, at |s| 1-2 / 2-3 / 3-4 rad/step,
0.989/0.983/0.941 at corr=-0.5 versus 0.952/0.820/0.392 uncompensated and
0.747/0.303/0.237 at +0.5 -- i.e. the sign that "looks right" makes it WORSE than doing
nothing, and the atom's phase must ADVANCE ~+0.5 rad per rad of z (the kwarg SUBTRACTS
corr*dz).  Magnitude agrees with the grating's 0.441 null; sign agrees with the server's
~-0.60 note for the 3-D model encoding.  The single sharpest number: nsteps=8 at 3.5
rad/step went 0.163 -> 0.938.  The nsteps=0 cells were flat across corr (0.9895 / 0.9892 /
0.9926), confirming the axis is inert without motion.  Job 418 then bracketed it as an
INTERIOR maximum -- 0.802 / 0.928 / 0.978 / 0.971 at -0.9 / -0.7 / -0.5 / -0.3 for |s| 2-3 --
so corr is now FIXED at -0.5 and the freed axis goes into statistics at large step size.

The primary graph is NOT the 8-cell grid -- it is the SITE-RESOLVED curve: the server
persists, per shot, the two parallel lists ``random_z_step_rad`` (each moving slot's
effective per-step z) and ``random_z_site_idx`` (its init_grid site index) into the
rearrange diag, which the lab pulls into ``slm_diag.h5``.  Joining those against the
per-site img1/img2 logicals gives survival vs |Delta z per step| in 1-rad bins, one curve
per nsteps.  The draw is SIGNED and zero-mean -- a positive ``random_z_max`` gives BOTH
directions (each site draws from ``U(-max, +max)``, so random magnitude AND random sign),
which is also the +z/-z asymmetry measurement.

TWO ARMS (identical in every other respect; run both, compare):
  * ``--warm``  (default): ``wgs3d_warm=True`` -- every transit frame solved by the warm
    matched-filter 3-D WGS producer (``tools/rearrange2/warm_wgs3d``).  Exact per-spot
    depth on a ``wgs3d_dz`` = 0.05 rad ladder, no model, no target image.
  * ``--model``: ``wgs3d_warm=False`` + ``gpu_target_graph=True`` -- the 3-D direct3d
    SLMnet checkpoint + the chirp-splat target builder.

Producer caveats that DO differ between the arms (they matter only in the tails):
  * The model's chirp-splat ladder is quantized at 1 um (~1.10 rad at
    ``train3d.UM_PER_RAD`` = 0.9057) and CLAMPS at +-40 um (~+-44 rad).  So per-step draws
    well under ~1.1 rad collapse toward 0 there, and with Z_PEAK_CAP = 96 rad the model arm
    DOES clamp on the outer part of every cell's biggest draws (the warm arm does not).
    Physically that is a distinction without a difference: the axial wing depth at 44 rad
    (~40 um ~ 17 z_R) is already ~1 uK against an 8 uK atom, so 44 and 96 rad are equally
    "gone" -- but it is a real command difference and should be quoted, not hidden.
  * The warm producer RAISES if a spot leaves its kernel table, and the table is rebuilt
    (+ CUDA graph recaptured) whenever ``wgs3d_z_max`` changes -- and the auto default is
    ``max|inter_z| + 0.5``, which moves EVERY shot with stochastic loading.  So this scan
    pins ``wgs3d_z_max`` to a constant that covers the worst cell (``Z_PEAK_CAP + 0.5``):
    one build for the whole scan, identical per-frame cost in every cell.

PACING: 0.696 ms/frame with ``precompute=True`` -- the whole frame stack is solved before
the paced loop starts, so the hot loop is write-only and the per-frame WGS/inference cost
cannot starve the pacing (job 394's failure mode).  Frames per shot = 2*nsteps + 1, i.e.
5..81 frames = 3..56 ms of motion, plus a ``hold_ms`` = 10 ms dwell at the TURNAROUND frame
(the peak excursion).  The dwell is NOT a bystander: comparing job 412 (hold 0) against job
414 (hold 10) at matched |s| and nsteps gives 0.993 vs 0.615 at nsteps=1, |s| 5-6.  The atom
does not follow a far axial jump -- it sits in the Lorentzian WING of its own defocused trap
(U = U0/(1+(z/z_R)^2), still ~46 uK at 8 rad against an 8 uK atom), so it survives a fast
out-and-back and leaks out during a dwell.  Loss is dwell/heating-rate limited rather than
capture-limited, which is why the far tail plateaus instead of going to zero.  10 ms is the
chosen operating point; the dwell is held FIXED here so nsteps and step size are the only
things varying.

Dead sites are DROPPED (``ghost_fraction=0.0``): only the loaded atoms exist in the transit
frames, the standard sparse-N isolation used in rearrangement.  The WGS bookend restores
the full array for img2, so detection is unchanged.

Array / focal plane match the live production config: 33x33_feedback11, z4 = loading_defocus
= -4, ifEnhanced loading.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/RandomZAxialRearrangeStepSweep.py --dry-run
    python YbScans/RearrangeDiagnostics/RandomZAxialRearrangeStepSweep.py --warm
    python YbScans/RearrangeDiagnostics/RandomZAxialRearrangeStepSweep.py --model
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
# 2-D production checkpoint -- used by the WARM arm, where the model never runs.
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# 3-D direct3d checkpoint (fp16 deployable copy) -- required by the MODEL arm.
MODEL_3D = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
DEFOCUS = -4.0

# dim 1, PAIRED: triangle half-length (2*nsteps steps per shot) and the half-width of the
# per-site per-step axial draw (RADIANS of PV quad-defocus).  Same length, same scan dim.
# nsteps=0 and 1 DROPPED: the static control is validated (flat at 0.99 across every draw and
# every corr, jobs 414/416/418), and a single out-and-back step is uninformative -- job 412
# showed no cliff at all out to 8 rad/step at nsteps=1.  The physics is in MANY steps at LARGE
# per-step size, so every remaining shot goes there.
NSTEPS = [2, 4, 8, 15, 25, 40]
S_MAX_RAD = 12.0              # 0..12 rad/step draw range where the peak cap allows it
Z_PEAK_CAP = 96.0             # cap on the PEAK excursion nsteps*s_i (rad) -- see docstring
RANDOM_Z_MAX = [S_MAX_RAD if n <= 0 else min(S_MAX_RAD, Z_PEAK_CAP / n) for n in NSTEPS]
RANDOM_Z_MAX = [round(v, 2) for v in RANDOM_Z_MAX]   # 12,12,12,8,4,2.13,1.28,0.8

# dim 2: per-atom defocus-phase correction, rad of spot phase per rad of PV quad-defocus.
# MEASURED (job 416, warm arm, 100 shots): the ridge is NEGATIVE.  Pooled survival over
# nsteps>=1 at |s| 3-4 was 0.941 / 0.392 / 0.237 for corr -0.5 / 0.0 / +0.5, and -0.5 won
# every |s| band -- so the atom's phase must ADVANCE ~+0.5 rad per rad of z (the code
# SUBTRACTS corr*dz), magnitude matching the grating's 0.441 null and sign matching the
# server's ~-0.60 note for the 3-D model encoding.  -0.5 was the EDGE of that axis, so this
# axis recentres on negatives to bracket the true optimum; all four values are near-optimal,
# so every cell also contributes usable step-size curves.
DEPTH_PISTON_CORR = -0.5      # SINGLE value now: the ridge is bracketed (interior maximum in
#                               job 418 -- 0.802/0.928/0.978/0.971 at -0.9/-0.7/-0.5/-0.3 for
#                               |s| 2-3), so further shots on that axis buy nothing.  Freeing
#                               it gives 4x the statistics per (nsteps, step-size) cell.
# Warm-producer patch radius scale.  R grows linearly with the table's z_max and the per-frame
# solve cost goes as S^2 = (2R+1)^2, so the 3x bigger Z_PEAK_CAP would be ~8x slower at
# radius_frac 1.0.  The module's own note: 0.85x is free and 0.70x "costs almost nothing"
# (below ~0.55x the spot CV rises).  0.7 buys back ~2x.
WGS3D_RADIUS_FRAC = 0.7

HOLD_MS = 10.0                # dwell at the TURNAROUND frame (the requested delay_ms)

# ---- --holdsep mode: separate the two n-INDEPENDENT loss channels --------------------
# The (1-f)*p^(2n) decomposition (jobs 416/418/420) shows p = 1.0000 +- 0.0002 below
# ~4 rad/step: survive the first jump and 50 more steps cost nothing.  But the fit only
# identifies f as n-INDEPENDENT, and the 10 ms turnaround dwell is ALSO once per shot, so f
# conflates "failed the first jump" with "escaped the trap wing during the dwell".  Sweeping
# the dwell separates them: escape-during-dwell must grow with hold_ms (-> 0 as hold -> 0),
# a genuine capture failure must not.  Same corr, same cap, same table as job 420.
HOLD_MS_LIST = [0.0, 1.0, 3.0, 10.0, 30.0]     # dim 2
NSTEPS_HOLD = [2, 8, 15]                       # dim 1 -- enough n to still fit (f, p)

# ---- --fixedm mode: nsteps at FIXED array context ------------------------------------
# The step sweep pairs nsteps with random_z_max (12,12,12,6.4,3.84,2.4), so at fixed own-|s|
# the large-nsteps cells also have a QUIETER array around each atom.  That is a confound, and
# it is not small: warm survival at |s| 3.4-4.2 reads 0.932 / 0.770 / 0.734 / 0.829 / 0.917
# for n = 2/4/8/15/25 -- a dip and a recovery, i.e. the "recovery" tracks m falling, not n.
# Here random_z_max is CONSTANT across nsteps so n is the only variable, and hold rides dim 2
# so the memoryless test can be done with and without the dwell in one scan.
# Cap 120 rad (not 96) so the fixed draw can reach 4.8 rad/step -- job 412 puts the
# memoryless-to-accelerating transition near 4 rad, and it must be inside the range.
# ---- --moveidx mode: separate OWN motion from ARRAY POLLUTION ------------------------
# Job 432 (all sites move, m capped at 4.8) gives survival 0.896 at |s| ~4, nsteps 8.
# Job 420 (same corr, same own |s|, but the array also contains sites commanded to 12
# rad/step) gives 0.734.  Same atom, same step -- the difference is what the REST of the
# array was doing, i.e. hologram quality degraded by extreme-z spots sharing the frame.
# move_idx moves only a SUBSET against a stationary background (ghost_fraction defaults to
# 1.0 there), which buys two things at once:
#   * the movers can be pushed past 4.8 rad/step without making the whole array noisy
#   * the STATICS become innocent-bystander probes -- they are commanded to hold still, so
#     any loss they suffer when the movers go deep is hologram damage, full stop.
MOVE_EVERY = 4                                 # move every 4th init_grid site (~25%)
N_INIT_SITES = 1068                            # 33x33_feedback11
MOVE_IDX = list(range(0, N_INIT_SITES, MOVE_EVERY))
NSTEPS_MOVEIDX = [2, 4, 8]                     # dim 1; peak = 96 rad at n=8, m=12
M_MOVEIDX = [4.8, 12.0]                        # dim 2: the pollution level itself

NSTEPS_FIXEDM = [2, 4, 8, 15, 25]
M_FIXED = 4.8                                  # rad/step, every cell; peak = 120 at n=25
Z_PEAK_CAP_FIXEDM = 120.0
HOLD_FIXEDM = [0.0, 10.0]                      # dim 2
PERIOD_MS = 0.696             # production pacing; precompute keeps the hot loop write-only
RANDOM_Z_SEED = 20260731      # site-keyed draw; FIXED so a site keeps its s_i all scan
NUM_PER_GROUP = 516           # -> ceil(516/6) = 86 passes = 516 shots (86 per cell)

# Warm-producer kernel table: pinned so it is built ONCE (see module docstring).
WGS3D_Z_MAX = Z_PEAK_CAP + 0.5                            # 32.5 rad


def _bootstrap():
    """Scans in YbScans subdirs get the SUBdir as sys.path[0]; add the pyctrl roots."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames are the SAME array (the triangle returns to the source sites).  Name =
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern key on."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _run_desc(warm, holdsep=False, fixedm=False, moveidx=False):
    if moveidx:
        return (
            "MOVER vs BYSTANDER run.  pingpong random-depth as before (step_size=0, "
            "random_z ref='step', corr=%g, precompute, step_period_ms=%.3f, hold_ms=0, "
            "array %s, z4=%g) but with extras.move_idx = every %dth init_grid site (%d of "
            "%d), so ONLY that subset executes the triangle and every other site is emitted "
            "STATIONARY at its own init position (ghost_fraction defaults to 1.0).  Grid: "
            "nsteps %s x random_z_max %s rad/step.  WHY: at the SAME own step size and the "
            "SAME correction, job 432 (array capped at 4.8 rad/step) gives 0.896 at |s|~4 / "
            "nsteps 8 while job 420 (array containing sites at 12 rad/step) gives 0.734 -- "
            "so most of the apparent axial cliff may be hologram degradation from extreme-z "
            "spots sharing the frame rather than the atom's own transport.  This run "
            "separates them: the movers can be driven past 4.8 rad/step without making the "
            "whole array noisy, and the STATICS are commanded to hold still, so any loss "
            "they take when the movers go deep is hologram damage with no transport "
            "component at all.  The random_z_max axis is the pollution level."
            % (DEPTH_PISTON_CORR, PERIOD_MS, PATTERN, DEFOCUS, MOVE_EVERY, len(MOVE_IDX),
               N_INIT_SITES, NSTEPS_MOVEIDX, M_MOVEIDX))
    if fixedm:
        return (
            "FIXED-ARRAY-CONTEXT nsteps test.  Same site-specific random axial pingpong "
            "(pingpong, step_size=0, random_z ref='step', ghost_fraction=0, precompute, "
            "step_period_ms=%.3f, corr=%g, seed=%d, array %s, z4=%g), but random_z_max is "
            "CONSTANT at %g rad/step for every nsteps %s, and hold_ms rides dim 2 as %s.  "
            "WHY: the step sweep PAIRS nsteps with random_z_max (12,12,12,6.4,3.84,2.4), so "
            "its large-nsteps cells also have a quieter array around each atom -- at |s| "
            "3.4-4.2 warm survival read 0.932/0.770/0.734/0.829/0.917 for n=2/4/8/15/25, a "
            "dip AND a recovery that tracks m falling rather than n rising.  With m fixed, n "
            "is the only variable and the memoryless test p = S^(1/2n) is clean.  Job 412 "
            "(hold 0, corr 0, m unconfounded) found p CONSTANT for n>=4 below ~4 rad/step "
            "(0.978/0.982 at |s| 2-3, 0.916/0.913 at 3-4 for n=4/8) but FALLING above it "
            "(0.879/0.836 at 4-5) -- i.e. memoryless below the knee, accelerating above.  "
            "This run repeats that at the CORRECTED operating point and out to n=25, with "
            "and without the dwell.  Cap %g rad so the fixed draw reaches 4.8 rad/step and "
            "brackets the transition."
            % (PERIOD_MS, DEPTH_PISTON_CORR, RANDOM_Z_SEED, PATTERN, DEFOCUS, M_FIXED,
               NSTEPS_FIXEDM, HOLD_FIXEDM, Z_PEAK_CAP_FIXEDM))
    if holdsep:
        return (
            "DWELL SEPARATION run.  Same site-specific random axial pingpong as the step "
            "sweep (protocol pingpong, step_size=0, random_z with random_z_ref='step', "
            "ghost_fraction=0, precompute, step_period_ms=%.3f, corr=%g, seed=%d, array %s, "
            "z4=%g), but the sweep is nsteps %s (paired with random_z_max %s rad/step, peak "
            "cap %g rad) x extras.hold_ms %s ms.  WHY: the (1-f)*p^(2n) decomposition of "
            "jobs 416/418/420 finds p = 1.0000 +- 0.0002 below ~4 rad/step -- an atom that "
            "survives the first jump pays NOTHING for 50 more steps -- so the loss is a "
            "single n-INDEPENDENT channel f.  But the turnaround dwell is ALSO once per "
            "shot, so f conflates 'failed to be captured by the first jump' with 'escaped "
            "the Lorentzian trap wing during the dwell'.  Sweeping the dwell separates "
            "them: escape-during-dwell must vanish as hold_ms -> 0, a capture failure must "
            "not.  Prior evidence that the dwell matters a lot: job 412 (hold 0) vs job 414 "
            "(hold 10) gave 0.993 vs 0.615 at nsteps=1, |s| 5-6 rad/step."
            % (PERIOD_MS, DEPTH_PISTON_CORR, RANDOM_Z_SEED, PATTERN, DEFOCUS, NSTEPS_HOLD,
               [round(min(S_MAX_RAD, Z_PEAK_CAP / n), 2) for n in NSTEPS_HOLD], Z_PEAK_CAP,
               HOLD_MS_LIST))
    producer = (
        ("WARM matched-filter 3-D WGS frame producer (wgs3d_warm=True, pad 2048, "
         "dz 0.05 rad, wgs3d_z_max pinned at %.1f rad so the kernel table is built once)"
         % WGS3D_Z_MAX)
        if warm else
        ("3-D direct3d SLMnet MODEL frame producer (wgs3d_warm=False + "
         "gpu_target_graph=True, checkpoint experiment_3d/base5x5_fp16); chirp-splat "
         "z ladder quantized at 1 um (~1.10 rad) and clamped at +-44 rad"))
    return (
        "SITE-SPECIFIC RANDOM AXIAL pingpong: protocol=pingpong, step_size=0 (no lateral "
        "motion), random_z=True with random_z_ref='step' -- every MOVING tweezer draws its "
        "own signed PER-STEP depth kick from U(-random_z_max, +random_z_max) rad of PV "
        "quad-defocus, held constant across the out-and-back triangle so each atom returns "
        "to its own start depth.  NO um conversion anywhere; the axis is radians.  Draw is "
        "keyed by init_grid SITE at random_z_seed=%d, so a site keeps its s_i for the whole "
        "scan and the diag lists random_z_step_rad / random_z_site_idx give a site-resolved "
        "survival-vs-step-size curve from every shot.  1-D PAIRED sweep (%d cells): nsteps "
        "%s with random_z_max %s rad/step -- the draw half-width is capped at "
        "Z_PEAK_CAP/nsteps (%g rad) so the peak excursion nsteps*s_i never exceeds %g rad, "
        "which keeps BOTH producers unclamped and the warm kernel table small; nsteps=0 is "
        "the static no-motion control.  ghost_fraction=0 -- dead sites dropped, only the "
        "loaded atoms move (sparse-N isolation, as in rearrangement); WGS bookend restores "
        "the full array for img2.  %s.  dim 2 = extras.depth_piston_corr %s -- the PER-ATOM "
        "defocus-phase correction (frame k subtracts corr*dz[k,i] from atom i's commanded "
        "phase, rad per rad of PV quad-defocus; same unit as pingponggrating's kwarg, whose "
        "grating null is 0.441).  0.0 is the UNCOMPENSATED control: without it every atom "
        "carries a per-site RANDOM ~0.5*s_i rad/step phase transient, past the ~0.8 rad/step "
        "where pure piston kills (07-22 campaign).  Both signs are swept because the ridge "
        "is producer-dependent (the 3-D model's encoding is quoted ~-0.60 rad/rad; the warm "
        "producer commands phase directly and has its own).  piston=0 (the uniform channel "
        "cannot track a per-site dz).  precompute=True, step_period_ms=%.3f, hold_ms=%g "
        "(dwell at the turnaround frame).  Array %s, z4 = loading_defocus = %g, ifEnhanced.  "
        "Questions: where does the per-atom phase correction null (sign + magnitude, per "
        "producer), where is the per-step AXIAL cliff once it IS nulled, does the cliff shift "
        "with nsteps (cumulative vs per-step damage), is +z worse than -z, and does the warm "
        "3-D WGS producer beat the 3-D model?"
        % (RANDOM_Z_SEED, len(NSTEPS), NSTEPS, RANDOM_Z_MAX, Z_PEAK_CAP, Z_PEAK_CAP,
           producer, DEPTH_PISTON_CORR, PERIOD_MS, HOLD_MS, PATTERN, DEFOCUS))


def build(warm=True, num_per_group=NUM_PER_GROUP, holdsep=False, fixedm=False,
          moveidx=False):
    """Build (do NOT submit) the 2-D [random_z_max x nsteps] ScanGroup.

    ``warm=True``  -> warm matched-filter 3-D WGS transit frames.
    ``warm=False`` -> 3-D direct3d SLMnet model transit frames (A/B control).
    Returns ``(seq_name, g)``.
    """
    _bootstrap()
    from scan_group import ScanGroup

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
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
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs: pure-depth random pingpong ----------------------------------
    # EVERY parameter explicit -- model_filename resets the server's sticky param cache.
    rk = g().rearrange_kwargs
    rk.protocol = "pingpong"
    if moveidx:
        ns_list = list(NSTEPS_MOVEIDX)
        rz_list = None                            # random_z_max rides dim 2 instead
    elif fixedm:
        ns_list = list(NSTEPS_FIXEDM)
        rz_list = [M_FIXED] * len(ns_list)        # CONSTANT array context
    elif holdsep:
        ns_list = list(NSTEPS_HOLD)
        rz_list = [round(min(S_MAX_RAD, Z_PEAK_CAP / n), 2) for n in ns_list]
    else:
        ns_list = list(NSTEPS)
        rz_list = [S_MAX_RAD if n <= 0 else round(min(S_MAX_RAD, Z_PEAK_CAP / n), 2)
                   for n in ns_list]
    rk.nsteps.scan(1, ns_list)                    # dim 1, PAIRED with random_z_max
    rk.step_period_ms = PERIOD_MS
    rk.extras.n_rounds = 1

    rk.extras.direction = 0.0                     # irrelevant at step_size=0, must be numeric
    rk.extras.step_size = 0.0                     # NO lateral motion; pure depth
    rk.extras.step_size_z = 0.0                   # no common-mode offset: zero-mean spread
    rk.extras.depth3d = True                      # keeps random_z_max=0 on the 3-D path
    rk.extras.random_z = True
    # PAIRED with nsteps on dim 1: cell k runs (NSTEPS[k], RANDOM_Z_MAX[k]).
    if moveidx:
        rk.extras.random_z_max.scan(2, list(M_MOVEIDX))   # dim 2 = the pollution level
        rk.extras.move_idx = list(MOVE_IDX)               # movers; the rest stay put
    else:
        rk.extras.random_z_max.scan(1, rz_list)   # rad/step, signed draw +-max
    rk.extras.random_z_ref = "step"               # bound the PER-STEP increment
    rk.extras.random_z_seed = RANDOM_Z_SEED       # site-keyed, stable across the scan
    # move_idx implies ghost_fraction 1.0 (KEEP the non-movers -- they are the probe), but
    # pass it EXPLICITLY: the extras dict is merge-only and every earlier scan in this
    # session set 0.0, which would drop the static background and destroy the probe.
    # Everywhere else drop the dead sites so only loaded atoms exist in the frames.
    rk.extras.ghost_fraction = 1.0 if moveidx else 0.0
    rk.extras.full_n = False
    rk.extras.oneway = False                      # out-and-back; atoms return to start depth
    rk.extras.piston = 0.0                        # uniform channel OFF; the per-atom
    #                                               correction below is the one that can
    #                                               track a per-site random dz
    rk.extras.depth_piston_corr = float(DEPTH_PISTON_CORR)   # at the measured ridge
    rk.extras.lateral_piston_corr = 0.0           # no lateral motion -> inert, pinned anyway
    if moveidx:
        rk.extras.hold_ms = 0.0                   # no dwell: isolate transport
    elif fixedm:
        rk.extras.hold_ms.scan(2, list(HOLD_FIXEDM))
    elif holdsep:
        # dim 2: the dwell itself.  Escape-from-the-wing scales with it; a first-jump
        # capture failure does not.  hold_ms = 0 is the no-dwell end of the lever arm.
        rk.extras.hold_ms.scan(2, list(HOLD_MS_LIST))
    else:
        rk.extras.hold_ms = HOLD_MS               # dwell at the turnaround (peak) frame

    # ---- frame producer ----------------------------------------------------------------
    rk.extras.wgs3d_warm = bool(warm)
    if warm:
        # Pin the kernel table: the auto default (max|inter_z| + 0.5) moves every shot with
        # stochastic loading and would rebuild the table + recapture the graph each time.
        rk.extras.wgs3d_z_max = float(Z_PEAK_CAP_FIXEDM + 0.5 if fixedm else WGS3D_Z_MAX)
        rk.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)
    else:
        rk.extras.gpu_target_graph = True         # REQUIRED by the 3-D model path

    rk.extras.precompute = True                   # solve the whole stack before the paced loop
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS                        # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN
    rk.extras.description = _run_desc(warm)

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = int(num_per_group)
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def RandomZAxialRearrangeStepSweep(url=None, reps=None, warm=True,
                                   num_per_group=NUM_PER_GROUP, holdsep=False,
                                   fixedm=False, moveidx=False):
    seq_name, g = build(warm=warm, num_per_group=num_per_group, holdsep=holdsep,
                        fixedm=fixedm, moveidx=moveidx)
    from yb_start_scan import ybStartScan
    # Label keeps 'Rearrange' -- the Analysis tab's slm_diag sync is name-gated.
    label = (("WarmWGS" if warm else "Model")
             + ("RandomZAxialMoveIdxRearrangeScan" if moveidx
                else "RandomZAxialFixedMRearrangeScan" if fixedm
                else "RandomZAxialHoldSepRearrangeScan" if holdsep
                else "RandomZAxialRearrangeStepSweep"))
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label=label,
                      description=_run_desc(warm, holdsep, fixedm, moveidx), **opts)
    if fixedm:
        ns_l, rz_l = NSTEPS_FIXEDM, [M_FIXED] * len(NSTEPS_FIXEDM)
    elif holdsep:
        ns_l = NSTEPS_HOLD
        rz_l = [round(min(S_MAX_RAD, Z_PEAK_CAP / n), 2) for n in ns_l]
    else:
        ns_l = NSTEPS
        rz_l = [round(min(S_MAX_RAD, Z_PEAK_CAP / n), 2) if n > 0 else S_MAX_RAD
                for n in ns_l]
    print("submitted %s -> descriptor id %s (%d cells; nsteps=%s, random_z_max=%s rad/step, "
          "depth_piston_corr=%s, hold_ms=%s; period=%.3f ms, seed=%d, %s)"
          % (label, did, g.nseq(), ns_l, rz_l, DEPTH_PISTON_CORR,
             HOLD_FIXEDM if fixedm else HOLD_MS_LIST if holdsep else HOLD_MS,
             PERIOD_MS, RANDOM_Z_SEED,
             "warm 3-D WGS (z_max %.1f rad)" % WGS3D_Z_MAX if warm else "3-D model"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="site-specific random axial pingpong: random_z_max x nsteps.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="explicit pass count (overrides the NumPerGroup-derived StackNum)")
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--model", action="store_true",
                    help="A/B control: 3-D SLMnet model frames (wgs3d_warm=False)")
    ap.add_argument("--warm", action="store_true",
                    help="warm 3-D WGS frames (the default)")
    ap.add_argument("--holdsep", action="store_true",
                    help="dwell-separation mode: nsteps %s x hold_ms %s at corr %.1f"
                         % (NSTEPS_HOLD, HOLD_MS_LIST, DEPTH_PISTON_CORR))
    ap.add_argument("--fixedm", action="store_true",
                    help="fixed array context: nsteps %s at random_z_max %g x hold %s"
                         % (NSTEPS_FIXEDM, M_FIXED, HOLD_FIXEDM))
    ap.add_argument("--moveidx", action="store_true",
                    help="mover/bystander: every %dth site moves, rest static; nsteps %s x "
                         "random_z_max %s" % (MOVE_EVERY, NSTEPS_MOVEIDX, M_MOVEIDX))
    ap.add_argument("--dry-run", action="store_true", help="build only, do not submit")
    args = ap.parse_args()
    _warm = not args.model
    if args.dry_run:
        _seq, _g = build(warm=_warm, num_per_group=args.num_per_group,
                         holdsep=args.holdsep, fixedm=args.fixedm,
                         moveidx=args.moveidx)
        print("seq=%s  nseq=%d  warm=%s\nnsteps       =%s\nrandom_z_max =%s\n"
              "peak |z|     =%s\nwgs3d_z_max=%.1f rad  hold_ms=%g"
              % (_seq, _g.nseq(), _warm, NSTEPS, RANDOM_Z_MAX,
                 [round(n * m, 2) for n, m in zip(NSTEPS, RANDOM_Z_MAX)],
                 WGS3D_Z_MAX, HOLD_MS))
    else:
        RandomZAxialRearrangeStepSweep(url=args.url, reps=args.reps, warm=_warm,
                                       num_per_group=args.num_per_group,
                                       holdsep=args.holdsep, fixedm=args.fixedm,
                         moveidx=args.moveidx)
