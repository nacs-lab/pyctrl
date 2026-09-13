"""WarmWGSAxialPingpongHeatScan.py -- AXIAL ping-pong of the LIVE SPOTS ONLY (warm 3-D WGS
transit frames), with release-and-recapture thermometry.  The direct test of whether the
near-plane visit penalty needs the motion to be a GLOBAL GRATING.

THE QUESTION (2026-08-27, user)
  The 2026-08-19 ``turnmech`` campaign established, on ``pingponggrating`` + ``depth=True``:

      loss per visit to displacement 0   4.07(8) %
      ... at 3 um from the plane          1.83(4) %
      ... at 6 um                         0.55(2) %
      ... at >= 12 um                     0.00(1) %      (decay scale ~3 um ~ 1.3 z_R)

  Mid-flight reversals are FREE; the cost is per RE-VISIT near the original focal plane.  The
  working hypothesis is interference of the displaced array with residual undiffracted / ghost
  light that stays focused at the original plane.

  EVERY ONE OF THOSE SHOTS USED A GLOBAL GRATING.  ``pingponggrating`` adds one ANSI-Z4 defocus
  to the WHOLE hologram, so (a) every site moves, dead ones included, (b) the array leaves and
  returns to the *stored WGS phase byte-for-byte* at displacement 0, and (c) whatever
  undiffracted light exists is common to the entire field and never moves.  All three are
  candidate mechanisms and the grating cannot separate them.

  THIS SCAN BREAKS THE GLOBALITY.  ``protocol = "pingpong"`` + ``wgs3d_warm = True`` +
  ``step_size_z`` moves ONLY the loaded spots, by re-solving each transit frame with the warm
  matched-filter 3-D WGS producer:

    * the motion is a per-spot commanded depth, not a global phase -- there is no grating.
    * the unfilled sites are KEPT AND STATIONARY (``ghost_fraction = 1.0``), so they are
      innocent-bystander probes: any loss THEY take is hologram damage with no transport
      component at all.  (The rearrangement default ``ghost_fraction = 0.0`` DROPS them; that
      would confound "the array changed" with "the atoms moved".)
    * frame 0 and the last frame are still solved holograms, not the stored WGS phase, so a
      penalty tied to the pristine-WGS frame per se cannot appear.

  READING IT (the whole point):

  | outcome | conclusion |
  |---|---|
  | penalty SURVIVES at the grating's per-visit size | not about globality: the atoms genuinely dislike re-crossing their own start plane, and the mechanism must be local to each trap (its own defocused partner orders, or a per-trap depth modulation) |
  | penalty VANISHES | it was the global grating -- undiffracted / zeroth-order light that stays at the plane while the whole array walks off it, exactly the standing hypothesis, and live-spot transport is FREE |
  | penalty REDUCED but nonzero | both channels; the split is the measurement |
  | the STATIONARY bystanders also lose | hologram quality degrades when the movers go deep (the job-420-vs-432 effect); quote it before attributing anything to transport |

  The per-site analysis of the grating runs (``campaigns/ppg/turnmech/persite_visitcost.py``)
  adds a second, sharper prediction: there the per-visit cost is extremely NON-UNIFORM -- median
  site 2.0 %/visit but the top decile carries 50 % of the total, split-half r = 0.85, and with
  NO spatial organisation (Moran z +1.5, radial rho -0.14).  A single localized stray beam
  cannot produce a spatially unstructured but site-reproducible tail; a per-trap property
  (each trap's own ghost/partner order, or its own depth modulation) can.  So the per-site map
  of THIS run is a co-primary deliverable: if the expensive sites are the SAME sites, the
  mechanism is a fixed per-trap property and survives the change of producer.

THE THREE MODES

``--mode compare``  (item 1: the globality test)
  2-D [per-step axial stroke x release time] at fixed ``nsteps = 25`` (50 steps, one
  out-and-back).  The stroke axis runs 0 -> 3.00 um and therefore SPANS the grating campaign's
  whole story in one scan: 1.50 um is the arm where the g ladder found the penalty ABSENT,
  3.00 um the arm where it was DOMINANT.  ``step_size_z = 0`` is the frame-count-matched
  zero-motion control, which is what separates "SLM writes cost survival on their own" (a
  zero-motion 400-frame shot measured 0.9495 vs 0.9908 below 300 frames, job 315) from the
  transport itself.  ``z4 = loading_defocus = -4``, ``ifEnhanced``, post-motion cooling OFF,
  imaging inherited from ByPattern -- all matched to the grating campaign.

  WHAT IS NOT MATCHED, and it is a hard producer limit rather than a choice: the STEP COUNT.
  The kernel table caps the peak excursion at ~96 rad (see ``Z_PEAK_CAP_RAD``), and 3.00 um/step
  reaches that at 25 steps out, so the grating's 80-step arm is unavailable at this stroke
  (150 rad).  MATCHED STROKE was chosen over matched step count (user, 2026-08-27) because the
  penalty is stroke-ACTIVATED: at 1.5 um there is nothing to detect, so a matched-step run at
  1.5 um could return a null for a trivial reason and would not test the hypothesis.  Quote
  "50 steps, 3.00 um/step" with any cross-protocol number.
  For the matched-STEP-COUNT comparison instead: ``--nsteps 40 --step-um 1.5``.

``--mode heat``  (item 2: the loss + heating curves the user asked for)
  2-D [nsteps x release time] at fixed stroke, run once per period (``--period 0.696`` and
  ``2.784``).  Default ``nsteps 25`` at the 3.00 um stroke -- 25 is also the RADIAL set's nsteps,
  which is the consistency the user asked for, and it is the only value that fits the cap at this
  stroke.  The nsteps-40 arm is available at a smaller stroke: ``--nsteps 40,25 --step-um 1.5``.
  The release axis is the 07-31 / 08-11 campaigns' grid, so ``analyze_heat_nsteps`` /
  ``analyze_ppgroup`` read it unchanged.

``--mode ladder``  (THE MEASUREMENT -- the globality test proper)
  2-D [``pingpong_group`` x release time] at FIXED total steps and FIXED frame count, sweeping
  only the number of INTERIOR returns to displacement 0 (the near-plane visits).  This is the
  warm-WGS, live-spots-only replication of the grating g-ladder that produced 4.07(8) %/visit,
  and it is the arm that actually answers whether the penalty needs a global grating.

  Why it needs a server patch, and what the patch does.  The built-in trajectory here is ONE
  triangle ``0 -> n -> 0``, which touches the plane only at its two ENDPOINTS: **zero INTERIOR
  visits**.  (Verified against the fold: ``_fold_ppg_pingpong_group`` at ``g = N/2`` builds
  exactly this trajectory and yields interior-zero-visits = 0 -- which is precisely why the
  campaign used that arm as its zero-visit REFERENCE and fitted the slope on the multi-cycle
  arms.)  ``extras.disp_schedule`` -- the key the ladders are expressed through -- existed on
  the server for ``pingponggrating`` only.  It was extended to this protocol on 2026-08-29 by
  ``campaigns/ppg/turnmech/server_patch_pingpong_disp_schedule.py``, which overrides ``disp_idx``
  at its definition so the depth track AND the per-atom ``depth_piston_corr`` phase correction
  follow the schedule automatically (both are computed FROM ``disp_idx``).  A uniform schedule
  was verified on the live server to reproduce the built-in triangle element for element in xy,
  phase and ``inter_z``, so nothing else changes.

  Total steps is **40**, not the grating's 80: at the matched 3.00 um stroke the kernel-table cap
  (~96 rad) is busted by an 80-step ladder's reference arm (g=40 -> peak 150 rad).  Halving N
  keeps the MATCHED STROKE -- the axis the penalty switches on -- and still gives a longer
  visit-count lever arm (9/4/1/0) than the grating's own (7/3/1/0).

``--mode trough``  (the F1 control)
  Same machinery, the nonzero-trough oscillation: ``0 -> peak``, k oscillations
  ``peak <-> peak-L``, ``peak -> 0``.  Every reversal is MID-FLIGHT at nonzero displacement and
  there is exactly ONE departure from and ONE arrival at the plane in every arm, so a
  per-REVERSAL cost separates from a per-VISIT cost.  Run at the SAME 40 total steps as the
  ladder so the two share a frame count and a per-step background and can be fitted JOINTLY,
  exactly as the 08-19 joint fit did.

PACING AND WHY IT MATTERS HERE
  ``precompute = True`` -- the whole frame stack is solved before the paced loop starts, so the
  hot loop is write-only and the per-frame WGS solve cannot starve the pacing (job 394's failure
  mode).  This is not optional for a warm-WGS run at 0.696 ms/frame.

  ``wgs3d_z_max`` is PINNED.  The warm producer RAISES if a spot leaves its kernel table, and the
  table is rebuilt (+ CUDA graph recaptured) whenever ``wgs3d_z_max`` changes -- and the auto
  default is ``max|inter_z| + 0.5``, which moves EVERY shot with stochastic loading.  Pinning it
  to cover the worst cell buys one build for the whole scan and identical per-frame cost in every
  cell.  ``wgs3d_radius_frac = 0.7`` buys back ~2x solve cost at no measured spot-CV price.

``wgs_iters = 3``, NOT 4.  The user suggested 4; every production scan in the tree uses 3, and
the source comment is explicit -- ``>= 3 (2 is the contract-quality cliff)``
(``RearrangeSTIRAPScan``, ``SLMRearrangementScan``, ``TwoLayerLift*``, ``RearrangeMWScan``, ...).
3 is the validated operating point and 4 has never been characterised on the rig, so 3 is used
and ``--wgs-iters`` is offered for a deliberate override.

AXIAL UNITS.  ``step_size_z`` is RADIANS of PV quad-defocus (``2*rho^2 - 1``), the same unit as
``pingponggrating``'s depth ``step_size`` under ``true_defocus = False``.  The grating campaign
ran in MICRONS under ``true_defocus = True``, so the comparison needs the conversion
**0.798 um per rad** (2026-07-27 recal; ``8*s^2/(pi*lambda)`` with s = 0.408 um/knm-px).  Every
stroke here is therefore quoted BOTH ways and ``--step-um`` accepts microns directly.

THE PER-ATOM PHASE CORRECTION.  ``depth_piston_corr = -0.5`` rad per rad of Z4-PV, the ridge
MEASURED on this exact producer (jobs 416 / 418: 0.802 / 0.928 / 0.978 / 0.971 at
-0.9 / -0.7 / -0.5 / -0.3 for |s| 2-3, an interior maximum).  This is NOT the grating's +0.441:
the model / warm path commands per-spot phase, so the correction is PER ATOM and carries the
opposite sign convention (the kwarg SUBTRACTS ``corr * dz``).  Getting it wrong is not a detail --
uncompensated, nsteps=8 at 3.5 rad/step read 0.163 vs 0.938 corrected.

Run (pyctrl backend live)::

    cd pyctrl
    python YbScans/RearrangeDiagnostics/WarmWGSAxialPingpongHeatScan.py --mode compare --dry-run
    python YbScans/RearrangeDiagnostics/WarmWGSAxialPingpongHeatScan.py --mode compare --force
    python YbScans/RearrangeDiagnostics/WarmWGSAxialPingpongHeatScan.py --mode heat \
        --period 0.696 --nsteps 40 --force
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
# 2-D production checkpoint. The warm 3-D producer never runs the model, but model_filename is
# what resets the server's sticky param cache, so it is always set.
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

DEFOCUS = -4.0                 # z4 == loading_defocus; the plane every axial number was measured in
UM_PER_RAD = 0.798             # PV ANSI-Z4 -> axial um (2026-07-27 recal; 8*s^2/(pi*lambda))
POST_COOL_HOLD_MS = 0.5        # dark hold replacing the post-motion cool

# --- warm 3-D WGS producer ------------------------------------------------------------------
WGS_ITERS = 3                  # >= 3 (2 is the contract-quality cliff); NOT 4 -- see docstring
WGS3D_DZ = 0.05                # kernel-table depth quantum (rad)
# Patch-radius scale.  0.7 was taken from the module docstring's "0.70x costs almost nothing
# (measured)", but that does NOT hold at 1089 spots: an offline A/B on the live producer
# (2026-08-31, beam flat, iters 2, table z_max 75.7) measured spot-CV
#     rfrac 0.55 -> 0.282 | 0.70 -> 0.246 | 0.85 -> 0.221 | 1.00 -> 0.164
# i.e. monotone, and 0.7 costs ~50% more CV than the server default.  Use the default.
WGS3D_RADIUS_FRAC = 1.0
WGS3D_FIRST_ITERS = 5          # wgs3d_first_iters default; frame 0 uses max(first, iters)
WGS3D_BEAM = "flat"            # wgs3d_beam server default -- set explicitly, was inherited
ABORT_LIMIT_MS = 45000.0       # server default 5000 is 5x too small for warm 3-D (see build())
DEPTH_PISTON_CORR = -0.5       # rad of spot phase per rad of Z4-PV -- MEASURED ridge, this producer

# --- THE HARD CONSTRAINT: the warm 3-D producer's kernel table ------------------------------
# The producer RAISES if a spot leaves its kernel table, so wgs3d_z_max must cover the PEAK
# excursion nsteps*step_size_z.  The table is Z = z_max/wgs3d_dz slices with a patch radius R
# that grows LINEARLY in z_max, and the kernel is O(Z * S^2) with S = 2R+1, i.e. ~z_max^3:
#
#     measured (job 396 pre-flight):  z_max 10.44 rad -> Z = 417, R = 34, S = 69, ~4 MB
#     scaled:                         z_max   96 rad  -> R ~ 314  -> ~3 GB   (the randomz
#                                       campaign's own affordability ceiling, with radius_frac
#                                       0.7 buying back ~2x)
#     scaled:                         z_max  226 rad  -> R ~ 736  -> ~40 GB  INFEASIBLE
#
# So Z_PEAK_CAP = 96 rad (~77 um) is inherited from the randomz campaign as the largest peak
# excursion this producer can afford, and it is ENFORCED in build() rather than left to blow up
# on the rig.  CONSEQUENCE, and it is a real limitation to quote with any result: at the 80-step
# (nsteps 40) frame count that matches the grating campaign, the cap allows at most
# 96/40 = 2.4 rad/step = 1.92 um/step -- BELOW the grating's 3.00 um operating point.  The
# comparison at matched 3.00 um/step is therefore only available at FEWER steps (nsteps <= 25,
# peak 94 rad), which is exactly why the nsteps-25 arm the user asked for does double duty here.
# RAISED 96 -> 130 on 2026-09-01.  96 was inherited from the randomz campaign as a SOLVE-COST
# ceiling, not a memory or correctness limit, and at nsteps 25 it caps the stroke at 3.06 um --
# which reaches the 0.696 ms axial cliff (2.93 um -> 91.8 rad) but TRUNCATES the 2.784 ms one
# (3.73 um -> 116.9 rad), i.e. exactly where the slow-pacing falloff lives.  Measured cost:
#     step 2.93 um -> z 91.8 rad,  R 246, S 493, kernel 116 MB
#     step 3.73 um -> z 116.9 rad, R 312, S 625, kernel 188 MB
#     step 4.00 um -> z 125.3 rad, R 334, S 669, kernel 215 MB
# Memory is trivial against ~26-34 GB free; the real cost is the per-frame solve (~n*S^2), and
# S 625 vs 493 is only 1.6x, which `precompute=True` moves off the paced loop entirely.
Z_PEAK_CAP_RAD = 130.0

# Reuse the outbound solve for the mirrored return frame (server patch
# server_patch_wgs3d_symmetric_reuse.py, applied 2026-09-01).  Exactly 2x on any
# out-and-back; the server verifies symmetry numerically per shot before reusing.
SYMMETRIC_REUSE = True

# --- the axial stroke axis (mode compare) ---------------------------------------------------
# rad of PV quad-defocus.  The axis DELIBERATELY reaches the grating campaign's own 3.00 um/step
# (3.7594 rad), which is why compare runs at NSTEPS_COMPARE = 25 rather than 40: at 40 the
# kernel-table cap would stop the axis at 1.92 um, short of the stroke where the penalty is
# known to be dominant, and a null there would be uninformative (the g ladder found the penalty
# ABSENT at 1.5 um).  At nsteps 25 the peak is 94 rad, just inside the cap.
# 1.88 rad (1.50 um) is kept on the axis as the grating's OTHER measured stroke -- the arm where
# the penalty vanished -- so this one scan spans "no effect" to "full effect" on the grating.
# 0 is the frame-count-matched zero-motion control.
STEP_RAD_COMPARE = [0.0, 0.94, 1.88, 2.82, 3.7594]      # = 0, 0.75, 1.50, 2.25, 3.00 um
NSTEPS_COMPARE = 25            # -> 50 steps; peak 94 rad at the top of the axis (cap 96)
# DEFAULT STROKE = the grating campaign's own 3.00 um/step (user decision 2026-08-27: MATCHED
# STROKE beats matched step count).  The penalty is stroke-ACTIVATED -- the g ladder found it
# dominant at 3.0 um and ABSENT at 1.5 um -- so a run at 1.5 um could return a null for the
# trivial reason that there was nothing to see, and would not test the globality hypothesis at
# all.  Matching the stroke keeps the run in the regime where the effect exists.
#
# CONSEQUENCE: 3.7594 rad x 40 = 150 rad busts the ~96 rad kernel-table cap, so the nsteps-40
# arm is NOT available at this stroke and NSTEPS_HEAT is 25 alone (peak 94 rad, just inside).
# 25 is also the nsteps the RADIAL set uses, which is the consistency the user asked for.
# The step count therefore differs from the grating's 80-step arm (50 vs 80); quote it.
STEP_RAD_HEAT = 3.7594                                  # = 3.00 um, the campaign's stroke

# --- mode heat -----------------------------------------------------------------------------
# 25 ONLY at the default 3.00 um stroke -- see STEP_RAD_HEAT: nsteps 40 would peak at 150 rad,
# past the kernel-table cap.  25 is the radial set's nsteps, so this is also the consistency arm.
# The matched-STEP-COUNT (nsteps 40) comparison is available at a smaller stroke:
#     --nsteps 40,25 --step-um 1.5
NSTEPS_HEAT = [25]

# --- mode ladder / trough: THE MEASUREMENT (needs the 2026-08-29 server patch) --------------
# The g-ladder holds the TOTAL step count and the FRAME count fixed and sweeps only how many
# times the array returns to displacement 0 (interior near-plane visits = N/(2g) - 1).  N must
# be an even multiple of 2g at every g, so every schedule ends at displacement 0 -- which is
# what puts each atom back in its source trap and keeps LIVE img2 detection valid.
#
# N = 40, NOT the grating's 80.  At the matched 3.00 um/step stroke the kernel-table cap
# (Z_PEAK_CAP_RAD, ~96 rad) allows a peak of at most 25 steps out, and the grating ladder's
# zero-visit reference arm g = N/2 = 40 would peak at 150 rad.  Halving N keeps the MATCHED
# STROKE (the axis the penalty actually switches on) and keeps a true zero-visit reference:
#     g =  2 -> 9 visits, peak  6 um      g = 10 -> 1 visit,  peak 30 um
#     g =  4 -> 4 visits, peak 12 um      g = 20 -> 0 visits, peak 60 um  (the REFERENCE,
#                                              frame-for-frame the built-in single triangle)
# 9/4/1/0 is a longer lever arm in the visit count than the grating's own 7/3/1/0, so the
# per-visit slope is at least as well determined despite the halved step count.
LADDER_TOTAL_STEPS = 40
LADDER_GROUPS = [2, 4, 10, 20]
# The F1 trough control: 0 -> peak, k oscillations peak <-> peak-L, peak -> 0.  Every reversal
# is MID-FLIGHT at nonzero displacement and there is exactly ONE departure from and ONE arrival
# at the plane regardless of k, so a per-REVERSAL cost and a per-VISIT cost separate.
# Run at the SAME total step count as the ladder (40) so the two share a frame count and a
# per-step background and can be fitted JOINTLY, exactly as the 08-19 joint fit did.
# peak = 8 is what N = 40 admits for every leg (N minus the out-and-back to the peak must be a
# whole number of 2L oscillations, which peak 10 fails at L = 3 and 6).  The troughs then sit at
# (8 - L) steps out = 15 / 18 / 21 um at the 3.00 um stroke -- ALL comfortably past the ~12 um
# beyond which the grating found reversals free (0.00(1) %), which is the point of the control:
# every reversal happens where a reversal is known to cost nothing, so any loss that survives is
# not a reversal cost.  L = 6 is dropped: N = 40 cannot tile it at peak 8 (a loud fold error).
TROUGH_LEGS = [3, 2, 1]
TROUGH_PEAK = 8
PERIODS_MS = [0.696, 2.784]    # the SLM write floor and 4x it (NOT 3.0 -- see the campaign note)
# The 07-31 / 08-11 release grid, unchanged so the existing analyzers read this run as-is.
# 0 is the byte-clean held-in-traps baseline, 0.5 us the restore-matched reference
# (RearrangeRnRStep short-circuits on exactly Time == 0, so t=0 never pays the trap restore).
DEF_TIMES_US = [0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90]
TIMES_US_COMPARE = [0, 0.5]    # survival only; a 2-point release cannot fit a temperature

# IMAGING: inherit the ByPattern overlay by default (see PPGPingpongGroupHeatScan's note -- g()
# BEATS the overlay, and the 399 detuning walks ~6 MHz/day, so a hardcoded constant silently
# pins a stale imaging state). DDS Imag399.Amp1/Amp2 are left alone either way
# (gotcha-stale-dds-amps-pid-imaging).
IMAG_DETUNING_MHZ = None
IMG1_PIDSET = None
IMG2_PIDSET = None


def _bootstrap():
    """Scans in YbScans subdirs get the SUBdir as sys.path[0]; add the pyctrl roots."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames are the SAME array: the ping-pong triangle returns every atom to its source
    trap, so live img2 detection is valid and no offline re-detection is needed.  Name = the
    phase-file BASENAME, which is what the detection registry and expConfig ByPattern key on
    (an alias silently falls back to a day-folder grid)."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def peak_rad(step_rad, nsteps):
    """Peak axial excursion of the out-and-back triangle, in rad and um."""
    return float(step_rad) * int(nsteps)


def build(mode="compare", step_rad=None, nsteps=None, period_ms=0.696, times_us=None,
          defocus=DEFOCUS, cool=False, keep_ghosts=True, wgs_iters=WGS_ITERS,
          corr=DEPTH_PISTON_CORR, img_det=IMAG_DETUNING_MHZ,
          img_pid=(IMG1_PIDSET, IMG2_PIDSET), groups=None, troughs=None, peak=TROUGH_PEAK,
          total_steps=LADDER_TOTAL_STEPS, abort_limit_ms=ABORT_LIMIT_MS,
          z_max=None, symmetric_reuse=SYMMETRIC_REUSE):
    """Build (do NOT submit) the ScanGroup.  Returns ``(seq_name, g)``.

    ``mode='compare'``: 2-D [step_size_z x release time], nsteps fixed -- the stroke axis.
    ``mode='heat'``:    2-D [nsteps x release time], stroke fixed -- loss + heating curves.
    ``mode='ladder'``:  2-D [pingpong_group x release time] -- THE MEASUREMENT: fixed total
        steps and fixed frame count, sweeping only the number of interior returns to
        displacement 0 (the near-plane visits).  This is the warm-WGS replication of the
        grating g-ladder that produced 4.07 %/visit.
    ``mode='trough'``:  2-D [pingpong_trough_leg x release time] -- the F1 control: every
        reversal at NONZERO displacement, exactly one departure from and one arrival at the
        plane at every arm, so a per-reversal cost separates from a per-visit cost.
    """
    _bootstrap()
    from scan_group import ScanGroup

    if mode not in ("compare", "heat", "ladder", "trough"):
        raise ValueError("mode must be 'compare', 'heat', 'ladder' or 'trough'")

    steps = ([float(s) for s in (step_rad or STEP_RAD_COMPARE)] if mode == "compare"
             else None)
    ns = ([int(n) for n in (nsteps or NSTEPS_HEAT)] if mode == "heat" else None)
    ns_fixed = int((nsteps or [NSTEPS_COMPARE])[0]) if mode == "compare" else None
    step_fixed = (float(step_rad[0]) if (mode in ("heat", "ladder", "trough") and step_rad)
                  else STEP_RAD_HEAT)
    gl = [int(x) for x in (groups or LADDER_GROUPS)]
    tl = [int(x) for x in (troughs or TROUGH_LEGS)]
    n_tot = int(total_steps)
    times = [float(t) * 1e-6 for t in
             (times_us or (TIMES_US_COMPARE if mode == "compare" else DEF_TIMES_US))]

    g = ScanGroup()

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -----------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_2D
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

    # imaging: only override when explicitly asked (g() BEATS the ByPattern overlay).
    if img_det is not None:
        g().Imag399.FreqDetuning = float(img_det) * 1e6
    if img_pid and img_pid[0] is not None:
        g().BlueMOT.Img1PIDSet = float(img_pid[0])
    if img_pid and img_pid[1] is not None:
        g().BlueMOT.Img2PIDSet = float(img_pid[1])

    # ---- rearrange_kwargs: pure-depth UNIFORM pingpong of the live spots -------------------
    # Every parameter explicit: the server's setup extras are STICKY across scans in one
    # backend session, so anything a previous campaign set (random_z, move_idx, ghost_fraction,
    # oneway, hold_ms, piston, depth3d) must be pinned here or it is inherited silently.
    rk = g().rearrange_kwargs
    rk.protocol = "pingpong"
    rk.step_period_ms = float(period_ms)
    rk.extras.n_rounds = 1

    rk.extras.direction = 0.0                   # inert at step_size 0, must still be numeric
    rk.extras.step_size = 0.0                   # NO lateral motion; pure depth
    rk.extras.depth3d = True                    # 3-D depth path
    rk.extras.random_z = False                  # UNIFORM, not per-site: this is the whole point
    rk.extras.random_z_max = 0.0                # pinned -- sticky from the randomz campaign
    rk.extras.oneway = False                    # out-and-back; every atom returns to its start z
    rk.extras.full_n = False

    if mode == "compare":
        rk.nsteps = int(ns_fixed)               # 25 -> 50 steps, one out-and-back
        rk.extras.step_size_z.scan(1, steps)    # dim 1: the per-step axial stroke (rad Z4-PV)
        z_worst = max(abs(s) * ns_fixed for s in steps)
    elif mode == "heat":
        rk.nsteps.scan(1, ns)                   # dim 1: nsteps (-> 2*nsteps total steps)
        rk.extras.step_size_z = float(step_fixed)
        z_worst = max(abs(step_fixed) * n for n in ns)
    else:
        # ---- ladder / trough: the schedule modes (server patch 2026-08-29) --------------
        # `nsteps` here is only a PLACEHOLDER: the per-shot fold
        # (rearrange_callbacks._fold_ppg_pingpong_group) overwrites it with the cell's own
        # max(disp), which is what the server sizes its unique-phase array by.  Only the
        # dequeue-time warmup setup_rearrangement sees this value, so it must be the LARGEST
        # peak in the scan -- otherwise the warm producer's kernel table is built too small.
        # THE SWITCH that activates the fold is extras.pingpong_nsteps (the TOTAL step count);
        # the pingpong_* keys are consumed lab-side and only `disp_schedule` reaches the server.
        rk.extras.step_size_z = float(step_fixed)
        rk.extras.pingpong_nsteps = int(n_tot)
        if mode == "ladder":
            rk.nsteps = int(max(gl))
            rk.extras.pingpong_group.scan(1, gl)          # dim 1: the visit-count ladder
            peak_disp = max(gl)
        else:
            rk.nsteps = int(peak)
            rk.extras.pingpong_peak = int(peak)
            rk.extras.pingpong_trough_leg.scan(1, tl)     # dim 1: mid-flight reversal count
            peak_disp = int(peak)
        z_worst = abs(step_fixed) * peak_disp

    # KEEP the unfilled sites, stationary -- the user's explicit requirement, and the thing that
    # makes them innocent-bystander probes for hologram damage. The rearrangement default is
    # ghost_fraction = 0.0 (DROP dead sites, sparse-N isolation); every randomz-campaign scan in
    # this tree set 0.0, and the extras dict is merge-only, so this MUST be explicit.
    rk.extras.ghost_fraction = 1.0 if keep_ghosts else 0.0

    # per-ATOM defocus-phase correction (the uniform `piston` channel cannot do this job).
    rk.extras.piston = 0.0
    rk.extras.depth_piston_corr = float(corr)
    rk.extras.lateral_piston_corr = 0.0         # no lateral motion -> inert, pinned anyway
    rk.extras.hold_ms = 0.0                     # NO turnaround dwell: it is its own loss channel
                                                # (job 412 vs 414: 0.993 vs 0.615 at |s| 5-6)

    # ---- warm 3-D WGS frame producer ------------------------------------------------------
    # ENFORCE the kernel-table cap rather than letting the producer raise (or the GPU OOM) on
    # the rig. See Z_PEAK_CAP_RAD's note: the kernel is ~z_max^3, so this is not a soft limit.
    if z_worst > Z_PEAK_CAP_RAD:
        # The "steps out" that set the peak: nsteps for compare/heat, max(disp) for a schedule.
        if mode == "compare":
            n_peak = ns_fixed
        elif mode == "heat":
            n_peak = max(ns)
        else:
            n_peak = max(gl) if mode == "ladder" else int(peak)
        raise SystemExit(
            "peak axial excursion %.1f rad exceeds Z_PEAK_CAP_RAD = %.1f rad (~%.0f um).\n"
            "The warm 3-D producer's kernel table is O(z_max^3) -- at %.0f rad the patch "
            "radius is ~%d (vs 34 at the measured 10.44 rad) and the kernel is of order %.0f "
            "GB, against ~26-34 GB of GPU. The producer RAISES when a spot leaves the table.\n"
            "Fix: lower --step-rad/--step-um, or lower the peak displacement (--nsteps for "
            "compare/heat, --groups / --peak for ladder/trough). At a peak of %d steps the "
            "largest affordable stroke is %.2f rad = %.2f um."
            % (z_worst, Z_PEAK_CAP_RAD, Z_PEAK_CAP_RAD * UM_PER_RAD, z_worst,
               int(34 * z_worst / 10.44), 4e-3 * (z_worst / 10.44) ** 3,
               n_peak, Z_PEAK_CAP_RAD / n_peak, Z_PEAK_CAP_RAD / n_peak * UM_PER_RAD))

    rk.extras.wgs3d_warm = True
    # KEY NAMESPACE (verified on the live server 2026-08-31, rearrange_actual.py:4391-4397).
    # The 3-D warm path reads `wgs3d_*` ONLY.  `wgs_iters` / `wgs_pad` / `wgs_beam` are the
    # 2-D warm producer's keys (read at line 1721) and are SILENTLY IGNORED here -- the
    # 2026-08-31 ladder and trough runs set `wgs_iters=3` and therefore actually ran at the
    # `wgs3d_iters` DEFAULT OF 2, below the contract-quality cliff, and at the `wgs3d_beam`
    # default "flat" rather than anything chosen.  Both keys are now set explicitly.
    rk.extras.wgs3d_iters = int(wgs_iters)
    rk.extras.wgs3d_first_iters = int(WGS3D_FIRST_ITERS)
    rk.extras.wgs3d_beam = str(WGS3D_BEAM)
    rk.extras.wgs3d_dz = float(WGS3D_DZ)
    # PIN the kernel table (auto default = max|inter_z| + 0.5, which moves every shot with
    # stochastic loading -> a table rebuild + CUDA-graph recapture per shot).
    # `z_max` override exists so a stroke axis SPLIT across jobs can pin each job's table
    # to ITS OWN worst stroke instead of the whole grid's (see --z-max / --split-stroke).
    rk.extras.wgs3d_z_max = float(z_worst + 0.5 if z_max is None else z_max)
    # Halve precompute on the symmetric out-and-back: the RETURN frames command an
    # identical target (same xy, z, per-spot phase incl. depth_piston_corr), so the
    # server replays the outbound solve instead of re-solving.  Server-side check is
    # numeric per shot, so an asymmetric trajectory silently solves every frame.
    # MEASURED 2026-09-01: 41.2 -> 20.6 ms/frame at z_max 94.5 (1.98x, 51 -> 26 solves).
    rk.extras.wgs3d_symmetric_reuse = bool(symmetric_reuse)
    rk.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)

    # ABORT CEILING.  The server's hard wall-clock ceiling defaults to 5000 ms for the WHOLE
    # shot (precompute + paced loop, rearrange_actual.py ~3559).  MEASURED 2026-09-01 from
    # shot_time.csv: a grating shot spans 2.10 s median, but a warm-WGS 3-D shot spans
    # **24.25 s median / 25.5 s p95**, so EVERY warm shot
    # blew the 5 s ceiling and was cancelled mid-flight -- which leaves partial SLM state.
    # ROOT CAUSE (found 2026-09-01): that ~430 ms/frame was NOT S^2 scaling of the kernel --
    # it was GPU MEMORY EXHAUSTION.  A cached warm producer + kernel table were pinning the
    # whole card (measured 0.00 GB free of 34.2 GB).  Freeing them recovered 23.7 GB and the
    # per-frame cost collapsed.  Measured after the clear, n=1089, iters 3, beam flat:
    #     z_max  10.0  S  65    7.5 ms/frame   (8.7 GB free)
    #     z_max  30.6  S 173   10.1 ms/frame   (23.7 GB free)
    #     z_max  75.7  S 407   26.7 ms/frame   (13.9 GB free)
    #     z_max  94.5  S 505   41.2 ms/frame   (8.7 GB free)   <- THIS scan's config
    #     z_max 125.8  S 669  691.5 ms/frame   (0.00 GB free)  <- the cliff, memory-bound
    # Cost is smooth in S until free memory hits zero, then jumps 17x.  So the ceiling is
    # GPU HEADROOM, not stroke.  With symmetric reuse this scan runs ~20.6 ms/frame ->
    # ~1.05 s of solve per 51-frame shot, comfortably inside the 45 s limit below.
    # Retrospect: job 1677 (the ladder, z_max 75.7) had 77 % of shots over 5 s while its trough
    # control had only 8 %, and the abort fraction rises with peak depth (deeper -> bigger kernel
    # -> slower solve), which is very likely the source of that run's inexplicable depth ordering.
    # 45 s gives ~1.8x headroom over the measured p95 while still bounding a true hang.
    rk.extras.abort_limit_ms = float(abort_limit_ms)
    rk.extras.precompute = True                 # solve the stack BEFORE the paced loop
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = float(defocus)
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # POST-MOTION cooling OFF unless --cool: the release curve must probe the POST-MOTION
    # temperature; production's 5 ms cool re-thermalizes to ~8 uK and hides the whole effect.
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(POST_COOL_HOLD_MS) * 1e-3

    # dim 2: the release-recapture thermometer.
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0               # set for faithfulness; UNREAD by the step

    # ---- run params ------------------------------------------------------------------------
    rp.NumPerGroup = 100000                     # upper bound; --reps sets the pass count
    rp.loading_defocus = float(defocus)
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return "RearrangeRnRHeatCommSeq", g


def _img_note(img_det, img_pid):
    bits = []
    if img_det is not None:
        bits.append("Imag399.FreqDetuning pinned to %+.1f MHz by a g() override" % img_det)
    if img_pid and img_pid[0] is not None:
        bits.append("Img1PIDSet %.2f" % img_pid[0])
    if img_pid and img_pid[1] is not None:
        bits.append("Img2PIDSet %.2f" % img_pid[1])
    if not bits:
        return ("INHERITED from ByPattern[%s] -- no g() override, so this scan images in exactly "
                "the state the array was last calibrated to (a hardcoded working set would BEAT "
                "the overlay, and the 399 detuning walks ~6 MHz/day). DDS Imag399.Amp1/Amp2 left "
                "alone." % PATTERN)
    return "; ".join(bits) + " (everything else inherited from the ByPattern overlay)."


_WHY = (
    "WHY: the 2026-08-19 turnmech campaign found, on pingponggrating + depth=True, that the "
    "axial ping-pong penalty is NOT the direction reversal (mid-flight reversals are free) but "
    "the RE-VISIT to the original focal plane -- 4.07(8) % per visit at displacement 0, "
    "1.83(4) % at 3 um, 0.55(2) % at 6 um, 0.00(1) % beyond 12 um, decaying over ~3 um ~ 1.3 "
    "z_R. The standing (unproven) hypothesis is interference of the displaced array with "
    "residual undiffracted / ghost light that stays focused at the original plane. EVERY shot "
    "behind that result used a GLOBAL GRATING: one ANSI-Z4 defocus on the whole hologram, so "
    "every site moves (dead ones included), displacement 0 IS the stored WGS phase byte-for-"
    "byte, and the undiffracted light is common to the whole field and never moves. This run "
    "breaks the globality: protocol=pingpong + wgs3d_warm moves ONLY the loaded spots by "
    "re-solving each transit frame, so there is no grating and no pristine-WGS bookend "
    "mid-sequence, while the unfilled sites are KEPT AND STATIONARY (ghost_fraction=1.0) as "
    "innocent-bystander probes -- any loss THEY take is hologram damage with no transport "
    "component. Penalty survives at grating size -> the effect is local to each trap, not about "
    "globality; penalty vanishes -> it WAS the global grating / zeroth order, and live-spot "
    "axial transport is free. Second prediction, from the per-site analysis of the grating runs "
    "(campaigns/ppg/turnmech/persite_visitcost.py): there the per-visit cost is highly "
    "NON-UNIFORM -- median site 2.0 %/visit, top decile carrying 50 % of the total, split-half "
    "r = 0.85, and NO spatial structure (Moran z +1.5, radial rho -0.14). If the same sites are "
    "expensive here, the mechanism is a fixed per-trap property that survives the change of "
    "producer.")

_CONFIG = (
    "CONFIG: warm matched-filter 3-D WGS transit frames (wgs3d_warm=True, wgs_iters=%d -- the "
    "validated value, '>= 3, 2 is the contract-quality cliff'; NOT 4, which has never been "
    "characterised on the rig -- wgs3d_dz=%g rad, wgs3d_radius_frac=%g, wgs3d_z_max PINNED so "
    "the kernel table is built ONCE instead of rebuilt per shot by the auto default). "
    "step_size=0 (no lateral motion), depth3d=True, random_z=False (UNIFORM per-step depth: the "
    "point is a like-for-like comparison against the grating, not a per-site draw), oneway=False "
    "(out-and-back, so every atom returns to its own start depth and LIVE img2 detection is "
    "valid -- no offline re-detection). depth_piston_corr=%+.2f rad per rad of Z4-PV: the "
    "PER-ATOM defocus-phase correction, at the ridge MEASURED on this exact producer (jobs "
    "416/418, an interior maximum -- 0.802/0.928/0.978/0.971 at -0.9/-0.7/-0.5/-0.3 for |s| "
    "2-3). This is NOT the grating's +0.441: the warm path commands per-spot phase, so the "
    "correction is per atom and carries the opposite sign convention. Uncompensated, nsteps=8 at "
    "3.5 rad/step read 0.163 vs 0.938 corrected. piston=0 (the uniform channel cannot track a "
    "per-atom dz). hold_ms=0 -- the turnaround dwell is deliberately OFF, it is its own loss "
    "channel (job 412 vs 414: 0.993 vs 0.615 at nsteps=1, |s| 5-6). precompute=True + "
    "precompute_host: the whole frame stack is solved before the paced loop, so the per-frame WGS "
    "solve cannot starve the pacing (job 394's failure mode) -- not optional at 0.696 ms/frame. "
    "SEQ RearrangeRnRHeatCommSeq: img1 -> Cool556 5 ms -> motion -> %s -> release t -> recapture "
    "-> img2; t=0 is pure motion survival, t=0.5 us the restore-matched reference "
    "(RearrangeRnRStep short-circuits on exactly Time==0, so t=0 never pays the trap restore that "
    "every t>0 shot pays -- worth ~1 %% after many frames). AXIAL UNITS: step_size_z is RADIANS "
    "of PV quad-defocus (2*rho^2-1); the grating campaign ran in MICRONS under true_defocus, so "
    "the comparison uses 1 rad = %.3f um (2026-07-27 recal, 8*s^2/(pi*lambda) at s = 0.408 "
    "um/knm-px). IMAGING: %s Array %s, z4 = loading_defocus = %g, ifEnhanced.")


def _cfg(cool, corr, wgs_iters, img_det, img_pid, defocus):
    return _CONFIG % (
        wgs_iters, WGS3D_DZ, WGS3D_RADIUS_FRAC, corr,
        ("post-motion cooling ON (re-thermalizes to ~8 uK and HIDES the effect)" if cool else
         "NO post-motion cooling (PostRearrCool amps 0 + %g ms dark hold)" % POST_COOL_HOLD_MS),
        UM_PER_RAD, _img_note(img_det, img_pid), PATTERN, defocus)


def _desc_compare(steps, ns_fixed, period_ms, times_us, defocus, cool, corr, wgs_iters,
                  img_det, img_pid, keep_ghosts):
    return (
        "WARM-WGS AXIAL PING-PONG of the LIVE SPOTS ONLY -- is the near-plane visit penalty a "
        "GLOBAL-GRATING effect? dim1 = extras.step_size_z = %s rad of PV quad-defocus "
        "(= %s um at %.3f um/rad), including 0 as the frame-count-matched zero-motion control "
        "(SLM writes cost survival on their own: a zero-motion 400-frame shot measured 0.9495 vs "
        "0.9908 below 300 frames, job 315). dim2 = release time %s us (survival, not a "
        "temperature -- 2 points cannot fit one). nsteps=%d -> %d total steps, one out-and-back, "
        "peak excursion %s rad = %s um. step_period_ms=%g. Unfilled sites %s. %s %s"
        % (",".join("%g" % s for s in steps),
           ",".join("%.2f" % (s * UM_PER_RAD) for s in steps), UM_PER_RAD,
           ",".join("%g" % t for t in times_us), ns_fixed, 2 * ns_fixed,
           ",".join("%g" % (s * ns_fixed) for s in steps),
           ",".join("%.1f" % (s * ns_fixed * UM_PER_RAD) for s in steps), period_ms,
           ("KEPT AND STATIONARY (ghost_fraction=1.0) as bystander probes"
            if keep_ghosts else "DROPPED (ghost_fraction=0.0, sparse-N isolation)"),
           _WHY, _cfg(cool, corr, wgs_iters, img_det, img_pid, defocus)))


def _desc_heat(ns, step_fixed, period_ms, times_us, defocus, cool, corr, wgs_iters,
               img_det, img_pid, keep_ghosts):
    return (
        "WARM-WGS AXIAL PING-PONG of the LIVE SPOTS ONLY -- LOSS + HEATING curves. "
        "dim1 = nsteps %s (-> %s total steps, peak excursion %s rad = %s um); "
        "dim2 = release time %s us (the 07-31/08-11 grid, so analyze_heat_nsteps / "
        "analyze_ppgroup read this run unchanged). Fixed stroke step_size_z=%g rad = %.2f um "
        "and step_period_ms=%g. Unfilled sites %s. "
        "This is the live-spot counterpart of the pure-grating axial loss/heating curves. The two "
        "periods are the 0.696 ms SLM write floor and 4x it (2.784 ms); NOTE the older axial "
        "numbers were taken at 3.0 ms, which is NOT an integer multiple of the write floor, and "
        "2.784 is the 4x value all newer work uses. WHAT IS MATCHED AND WHAT IS NOT: the STROKE "
        "is matched to the grating campaign's operating point (3.00 um/step, the arm where the "
        "near-plane penalty was DOMINANT); the STEP COUNT is not (50 here vs the grating's 80). "
        "That is a hard producer limit -- the warm 3-D kernel table is O(z_max^3) and caps the "
        "peak excursion at ~96 rad (~77 um), which 3.00 um/step reaches at 25 steps out. Matched "
        "stroke was chosen over matched step count deliberately: the penalty is stroke-ACTIVATED "
        "(absent at 1.5 um, dominant at 3.0), so a matched-step run at the smaller stroke could "
        "return a null for a trivial reason and would not test the hypothesis. nsteps 25 is also "
        "the RADIAL set's value, so this arm doubles as the radial-consistency point. %s %s"
        % (",".join(str(n) for n in ns), ",".join(str(2 * n) for n in ns),
           ",".join("%g" % (step_fixed * n) for n in ns),
           ",".join("%.1f" % (step_fixed * n * UM_PER_RAD) for n in ns),
           ",".join("%g" % t for t in times_us), step_fixed, step_fixed * UM_PER_RAD,
           period_ms,
           ("KEPT AND STATIONARY (ghost_fraction=1.0) as bystander probes"
            if keep_ghosts else "DROPPED (ghost_fraction=0.0)"),
           _WHY, _cfg(cool, corr, wgs_iters, img_det, img_pid, defocus)))


def schedule(group=None, trough=None, peak=TROUGH_PEAK, n_total=LADDER_TOTAL_STEPS):
    """The displacement-index list the per-shot fold will build -- obtained by calling the FOLD
    ITSELF, so a dry run can never drift from what actually runs on the rig."""
    _bootstrap()
    import rearrange_callbacks as rc
    extras = {"step_size": 0.0, "step_size_z": 1.0, "pingpong_nsteps": int(n_total)}
    if trough is not None:
        extras["pingpong_trough_leg"] = int(trough)
        extras["pingpong_peak"] = int(peak)
    else:
        extras["pingpong_group"] = int(group)
    return rc._fold_ppg_pingpong_group({"extras": extras})["extras"]["disp_schedule"]


def visits(disp):
    """Interior returns to displacement 0 -- the near-plane visits the campaign charges for.
    The two ENDPOINTS are not counted: every schedule starts and ends at 0 by construction."""
    return sum(1 for i in range(1, len(disp) - 1) if disp[i] == 0)


def reversals(disp):
    """Interior local extrema -- the velocity reversals (free above ~12 um, per F1)."""
    return sum(1 for i in range(1, len(disp) - 1)
               if (disp[i] > disp[i - 1] and disp[i] > disp[i + 1])
               or (disp[i] < disp[i - 1] and disp[i] < disp[i + 1]))


def _desc_ladder(mode, gl, tl, peak, n_tot, step_fixed, period_ms, times_us, defocus, cool,
                 corr, wgs_iters, img_det, img_pid, keep_ghosts):
    if mode == "ladder":
        arms = [(g, schedule(group=g, n_total=n_tot)) for g in gl]
        axis = ("dim1 = extras.pingpong_group g = %s -> interior near-plane visits %s "
                "(and %s velocity reversals), at FIXED %d total steps and FIXED %d SLM frames "
                "in every arm"
                % (",".join(str(g) for g in gl),
                   ",".join(str(visits(d)) for _g, d in arms),
                   ",".join(str(reversals(d)) for _g, d in arms),
                   n_tot, len(arms[0][1])))
        why = (
            "THE MEASUREMENT. This is the warm-WGS replication of the grating g-ladder that "
            "produced the campaign's headline number. Holding the step count and the frame "
            "count fixed and sweeping ONLY the interior-visit count is what isolates the "
            "per-visit cost from the per-step cost and from the SLM-write cost: on the grating "
            "the joint fit gave 4.07(8) %% per visit at displacement 0 against a 0.063-0.074 "
            "%%/step background. The g = N/2 arm is the ZERO-VISIT reference and is "
            "frame-for-frame the built-in single triangle, so the ladder carries its own matched "
            "control and needs no cross-day differencing. NOTE the total step count is 40, not "
            "the grating's 80: at the MATCHED 3.00 um stroke the warm producer's kernel table "
            "caps the peak at ~96 rad, which the 80-step ladder's reference arm (g=40, peak 150 "
            "rad) busts. Halving N keeps the matched stroke -- the axis the penalty actually "
            "switches on -- and still gives a LONGER visit-count lever arm (9,4,1,0) than the "
            "grating's own (7,3,1,0).")
    else:
        arms = [(L, schedule(trough=L, peak=peak, n_total=n_tot)) for L in tl]
        axis = ("dim1 = extras.pingpong_trough_leg L = %s at peak %d -> %s velocity reversals, "
                "ALL of them MID-FLIGHT at nonzero displacement, with exactly ONE departure "
                "from and ONE arrival at the plane in every arm (interior visits %s), at FIXED "
                "%d total steps and FIXED %d SLM frames"
                % (",".join(str(L) for L in tl), peak,
                   ",".join(str(reversals(d)) for _L, d in arms),
                   ",".join(str(visits(d)) for _L, d in arms),
                   n_tot, len(arms[0][1])))
        why = (
            "THE CONTROL (F1). The g-ladder axis is DEGENERATE -- per cycle the departures "
            "from 0, the arrivals at 0 and the peak reversals all scale together -- so it "
            "cannot by itself say whether the cost is the DIRECTION CHANGE or the VISIT to "
            "the plane. This schedule holds the plane visits fixed at one departure + one "
            "arrival and sweeps only the mid-flight reversal count. On the grating the answer "
            "was unambiguous: reversals at >=12 um are FREE (-0.010 +- 0.005 %% each) while a "
            "visit at displacement 0 costs 4.07 %%. Repeating it here says whether that "
            "separation is a property of the atoms or of the global grating.")
    return (
        "WARM-WGS AXIAL PING-PONG of the LIVE SPOTS ONLY -- %s LADDER. %s; dim2 = release time "
        "%s us. Fixed stroke step_size_z=%g rad = %.2f um, step_period_ms=%g. Unfilled sites "
        "%s. %s "
        "MECHANISM OF EXPRESSION: extras.disp_schedule on the PINGPONG path -- the server key "
        "added for pingponggrating on 2026-08-11 and extended to the warm 3-D / model pingpong "
        "protocol on 2026-08-29 (campaigns/ppg/turnmech/server_patch_pingpong_disp_schedule.py). "
        "The built-in trajectory is ONE triangle, which touches displacement 0 only at its "
        "endpoints and therefore has ZERO interior visits, so without the schedule this "
        "protocol cannot express the measurement at all. The list is built PER SHOT by "
        "rearrange_callbacks._fold_ppg_pingpong_group from the scalar extras.pingpong_group / "
        "pingpong_trough_leg (a list-valued swept axis breaks the lab-side scan grid) and "
        "echoed as ppg_pp_group / ppg_pp_cycles / ppg_pp_reversals / ppg_pp_frames. Every "
        "schedule ENDS at displacement 0, so every atom is back in its source trap and LIVE "
        "img2 detection stays valid -- no offline re-detection. "
        "PHASE CORRECTION UNDER A SCHEDULE (verified on the live server before submitting): the "
        "per-atom defocus correction is computed as -depth_piston_corr * disp_idx * step_z, i.e. "
        "FROM the schedule itself, so each moving atom is corrected against its OWN commanded "
        "axial displacement in each frame and stationary ghosts correctly receive none; a "
        "uniform schedule was checked to reproduce the built-in triangle element for element in "
        "xy, phase and inter_z. %s %s"
        % ("VISIT" if mode == "ladder" else "MID-FLIGHT REVERSAL", axis,
           ",".join("%g" % t for t in times_us), step_fixed, step_fixed * UM_PER_RAD, period_ms,
           ("KEPT AND STATIONARY (ghost_fraction=1.0) as bystander probes"
            if keep_ghosts else "DROPPED (ghost_fraction=0.0)"),
           why, _WHY, _cfg(cool, corr, wgs_iters, img_det, img_pid, defocus)))


def description(mode, steps, ns, ns_fixed, step_fixed, period_ms, times_us, defocus, cool,
                corr, wgs_iters, img_det, img_pid, keep_ghosts, gl=None, tl=None,
                peak=TROUGH_PEAK, n_tot=LADDER_TOTAL_STEPS):
    if mode in ("ladder", "trough"):
        return _desc_ladder(mode, gl or LADDER_GROUPS, tl or TROUGH_LEGS, peak, n_tot,
                            step_fixed, period_ms, times_us, defocus, cool, corr, wgs_iters,
                            img_det, img_pid, keep_ghosts)
    if mode == "compare":
        return _desc_compare(steps, ns_fixed, period_ms, times_us, defocus, cool, corr,
                             wgs_iters, img_det, img_pid, keep_ghosts)
    return _desc_heat(ns, step_fixed, period_ms, times_us, defocus, cool, corr, wgs_iters,
                      img_det, img_pid, keep_ghosts)


def main():
    ap = argparse.ArgumentParser(
        description="Warm-WGS axial ping-pong of the live spots only (+ RnR thermometry).")
    ap.add_argument("--mode", default="compare",
                    choices=("compare", "heat", "ladder", "trough"),
                    help="ladder = THE MEASUREMENT (near-plane visit count, fixed steps + "
                         "frames); trough = the F1 control (mid-flight reversals only); "
                         "compare = stroke axis; heat = loss + heating curves (nsteps axis)")
    ap.add_argument("--groups", default=None,
                    help="mode ladder: comma list of pingpong_group g (default %s)"
                         % ",".join(str(x) for x in LADDER_GROUPS))
    ap.add_argument("--troughs", default=None,
                    help="mode trough: comma list of pingpong_trough_leg L (default %s)"
                         % ",".join(str(x) for x in TROUGH_LEGS))
    ap.add_argument("--peak", type=int, default=TROUGH_PEAK,
                    help="mode trough: the oscillation peak displacement (default %d)"
                         % TROUGH_PEAK)
    ap.add_argument("--total-steps", type=int, default=LADDER_TOTAL_STEPS,
                    help="ladder/trough: TOTAL steps per shot (default %d)"
                         % LADDER_TOTAL_STEPS)
    ap.add_argument("--step-rad", default=None,
                    help="comma list of per-step strokes in rad of PV Z4 (mode compare), or a "
                         "single value (mode heat)")
    ap.add_argument("--step-um", default=None,
                    help="same, in MICRONS (converted at %.3f um/rad)" % UM_PER_RAD)
    ap.add_argument("--nsteps", default=None,
                    help="comma list (mode heat) or the single fixed value (mode compare)")
    ap.add_argument("--period", type=float, default=0.696, help="step_period_ms")
    ap.add_argument("--times-us", default=None, help="comma list of release times in us")
    ap.add_argument("--reps", type=int, default=None, help="pass count (ybStartScan rep)")
    ap.add_argument("--cool", action="store_true",
                    help="leave post-motion cooling ON (hides the effect; for a control only)")
    ap.add_argument("--drop-ghosts", action="store_true",
                    help="ghost_fraction=0.0 -- DROP the unfilled sites (default keeps them)")
    ap.add_argument("--wgs-iters", type=int, default=WGS_ITERS,
                    help="WGS iterations per frame (default %d -- the validated value)"
                         % WGS_ITERS)
    ap.add_argument("--corr", type=float, default=DEPTH_PISTON_CORR,
                    help="depth_piston_corr (default %+.2f, the measured ridge)"
                         % DEPTH_PISTON_CORR)
    ap.add_argument("--img-det", type=float, default=None,
                    help="override Imag399.FreqDetuning in MHz (default: inherit ByPattern)")
    ap.add_argument("--img-pid", default=None, help="override 'img1,img2' PIDSet")
    ap.add_argument("--abort-limit-ms", type=float, default=ABORT_LIMIT_MS,
                    help="server hard wall-clock shot ceiling (default %(default)s; "
                         "the server default 5000 cancels every warm-3-D shot)")
    ap.add_argument("--label-extra", default="",
                    help="suffix appended to the auto label (e.g. a stroke-chunk tag)")
    ap.add_argument("--note", default="",
                    help="text appended to the auto description (provenance)")
    ap.add_argument("--z-max", type=float, default=None,
                    help="override wgs3d_z_max (rad). Use when SPLITTING the stroke axis "
                         "across jobs so each job pins its own worst stroke, not the grid's.")
    ap.add_argument("--no-symmetric-reuse", action="store_true",
                    help="solve every frame instead of replaying the outbound half")
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    a = ap.parse_args()

    def _lst(s, cast=float):
        return None if not s else [cast(x) for x in str(s).split(",") if str(x).strip()]

    steps = _lst(a.step_rad)
    if a.step_um:
        um = _lst(a.step_um)
        steps = [v / UM_PER_RAD for v in um]
    ns = _lst(a.nsteps, int)
    times_us = _lst(a.times_us)
    img_pid = tuple(_lst(a.img_pid) or (None, None)) if a.img_pid else (None, None)

    gl = _lst(a.groups, int) or LADDER_GROUPS
    tl = _lst(a.troughs, int) or TROUGH_LEGS

    seq_name, g = build(mode=a.mode, step_rad=steps, nsteps=ns, period_ms=a.period,
                        times_us=times_us, cool=a.cool, keep_ghosts=not a.drop_ghosts,
                        wgs_iters=a.wgs_iters, corr=a.corr, img_det=a.img_det,
                        img_pid=img_pid, groups=gl, troughs=tl, peak=a.peak,
                        total_steps=a.total_steps, abort_limit_ms=a.abort_limit_ms,
                        z_max=a.z_max, symmetric_reuse=not a.no_symmetric_reuse)

    steps_eff = steps or STEP_RAD_COMPARE
    ns_eff = ns or NSTEPS_HEAT
    ns_fixed = int((ns or [NSTEPS_COMPARE])[0])
    step_fixed = (float(steps[0]) if (a.mode in ("heat", "ladder", "trough") and steps)
                  else STEP_RAD_HEAT)
    times_eff = times_us or (TIMES_US_COMPARE if a.mode == "compare" else DEF_TIMES_US)
    desc = description(a.mode, steps_eff, ns_eff, ns_fixed, step_fixed, a.period, times_eff,
                       DEFOCUS, a.cool, a.corr, a.wgs_iters, a.img_det, img_pid,
                       not a.drop_ghosts, gl=gl, tl=tl, peak=a.peak, n_tot=a.total_steps)

    print("seq=%s  mode=%s  nseq=%d  period=%g ms  wgs_iters=%d  corr=%+.2f  ghosts=%s"
          % (seq_name, a.mode, g.nseq(), a.period, a.wgs_iters, a.corr,
             "DROPPED" if a.drop_ghosts else "KEPT"))
    if a.mode == "compare":
        print("  nsteps=%d (%d steps)   step_size_z (rad / um / peak um):" % (ns_fixed,
                                                                             2 * ns_fixed))
        for s in steps_eff:
            print("    %6.2f rad  %6.2f um/step   peak %7.1f um"
                  % (s, s * UM_PER_RAD, s * ns_fixed * UM_PER_RAD))
    elif a.mode in ("ladder", "trough"):
        print("  step_size_z=%g rad = %.2f um   TOTAL steps %d per shot"
              % (step_fixed, step_fixed * UM_PER_RAD, a.total_steps))
        print("  %-6s %7s %8s %10s %7s %11s"
              % ("arm", "frames", "visits", "reversals", "peak", "peak um"))
        arms = (gl if a.mode == "ladder" else tl)
        for v in arms:
            d = (schedule(group=v, n_total=a.total_steps) if a.mode == "ladder"
                 else schedule(trough=v, peak=a.peak, n_total=a.total_steps))
            print("  %-6s %7d %8d %10d %7d %11.1f"
                  % (("g=%d" % v) if a.mode == "ladder" else ("L=%d" % v),
                     len(d), visits(d), reversals(d), max(d),
                     max(d) * step_fixed * UM_PER_RAD))
        print("  (visits = INTERIOR returns to displacement 0; the endpoints are not counted)")
    else:
        print("  step_size_z=%g rad = %.2f um   nsteps (steps / peak um):"
              % (step_fixed, step_fixed * UM_PER_RAD))
        for n in ns_eff:
            print("    %3d  ->  %3d steps   peak %7.1f um"
                  % (n, 2 * n, step_fixed * n * UM_PER_RAD))
    print("  release times (us) = %s" % times_eff)

    if a.dry_run:
        # A 1-element scan axis collapses to a scalar param, so nsteps/step_size_z can come back
        # as either a scalar or a list; normalise before hashing.
        def _key(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v
        seen = set()
        for i in range(g.nseq()):
            e = g.getseq(i)["rearrange_kwargs"]
            seen.add((_key(e.get("nsteps")), _key(e["extras"].get("step_size_z"))))
        print("  %d distinct (nsteps, step_size_z) cells" % len(seen))
        s0 = g.getseq(0)["rearrange_kwargs"]
        # Print the FULL extras dict, not a hand-kept subset.  The old curated list silently
        # omitted every key added after it was written (wgs3d_iters, wgs3d_beam,
        # abort_limit_ms, wgs3d_symmetric_reuse ...) and still showed the DEAD `wgs_iters`,
        # so a dry-run could not be used to confirm what actually reaches the server.
        print("  protocol=%r  cell0 extras (%d keys):" % (s0.get("protocol"),
                                                          len(s0["extras"])))
        for k in sorted(s0["extras"]):
            print("      %-26s %r" % (k, s0["extras"][k]))
        print("\n--- description ---\n%s" % desc)
        return

    if not a.force:
        raise SystemExit("refusing to submit without --force (use --dry-run to inspect)")
    from yb_start_scan import ybStartScan
    opts = {"rep": a.reps} if a.reps is not None else {}
    # Label MUST contain 'Rearrange': the Analysis tab's on-demand slm_diag sync
    # (run_analysis._maybe_sync_slm_diag) is name-gated.
    label = ("WarmWGSAxialPingpongRearrange%s"
             % {"compare": "Compare", "heat": "Heat",
                "ladder": "Ladder", "trough": "Trough"}[a.mode])
    if a.label_extra:
        label = label + "_" + a.label_extra
    if a.note:
        desc = desc + " " + a.note
    did = ybStartScan(seq_name, g, url=a.url, label=label, description=desc, **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
