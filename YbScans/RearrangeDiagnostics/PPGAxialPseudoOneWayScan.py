"""PPGAxialPseudoOneWayScan.py -- the axial (+z / -z) PSEUDO-ONE-WAY transport scan.

The axial twin of the 2026-08-06 radial one-way campaign, and the workhorse for every survival
scan in the 08-07 axial campaign (smoke test, range-finder, step x nsteps, period sweep, the
return-leg piston null, and the lateral isotropy arm).

WHY PSEUDO-ONE-WAY AND NOT ONE-WAY
----------------------------------
The radial campaign used ``return_trip=False``: the array translates and RESTS displaced. That is
not runnable axially -- the array would rest DEFOCUSED, the WGS bookend would snap it back to the
focal plane, and the atoms would be gone. So the tested move is run as the **RETURN leg** of an
asymmetric triangle::

    out    -z, n_out small qualified steps of s_out (<= out_step_max), period step_period_ms
    hold   hold_ms at the turnaround frame k = n_out  -> LC fully settled, peak faithful
    return +z, THE MOVE UNDER TEST: return_nsteps x return_step_size at return_step_period_ms

Displacement 0 is the stored WGS phase BYTE-FOR-BYTE (the dispatcher short-circuits ``d == 0`` to
``ip_arr``), so the array lands exactly where it started. Two consequences that make this strictly
better than the radial one-way:

* **Live detection stays valid.** No offline re-detection on a shifted grid, no
  ``ppg_transport_analyze`` -- both camera frames see the same array at the same place. (The radial
  rule "NEVER use stored logicals_img2 for a one-way scan" does NOT apply here.)
* **The tested leg makes n steps, not 2n.** Per-step survival is ``S**(1/n)`` after dividing out
  the outward leg's own cost -- see THE OUTWARD-LEG CONTROL below.

THREE TRAPS, confirmed by reading rearrange_actual.py directly (2026-08-07)
--------------------------------------------------------------------------
1. **The dispatcher's displacement index is NON-NEGATIVE; direction lives entirely in the amplitude
   sign** (``phi = ip_transit + (amp_leg * d) * unit_map``). To test a ``+z`` move, BOTH
   ``step_size`` and ``return_step_size`` must be NEGATIVE (out to -D, then back up to 0). A
   positive return amplitude teleports the array to +z at the turnaround.
2. **The peak is continuous only when ``n_out*s_out == n_ret*s_ret``.** ``_fold_ppg_outward_leg``
   guarantees it by construction (``s_out = D/n_out``, mismatch 0.0 in the fold unit tests); the
   analysis asserts ``diag.leg_out_peak == diag.leg_ret_peak``.
3. **``return_nsteps = 0`` rests the SLM on the PEAK frame**, not on WGS -- a documented footgun.
   The fold special-cases the ``n_test = 0`` / ``s_test = 0`` control to a single static WGS write.

THE OUTWARD LEG IS DERIVED, NOT SCANNED
---------------------------------------
``n_out`` depends on BOTH scanned axes (``D = return_nsteps * return_step_size``), which no
ScanGroup product axis can express. ``extras.out_step_max`` switches on the per-shot
``_fold_ppg_outward_leg`` (pyctrl/YbSeqs/rearrange_callbacks.py), which computes
``n_out = ceil(|D|/out_step_max)`` and ``s_out = D/n_out`` and OVERWRITES ``nsteps`` /
``extras.step_size``. ``rk.nsteps`` below is a placeholder that only the dequeue-time warmup call
ever sees. The applied values are echoed to ``extras.ppg_out_step_applied`` /
``ppg_peak_travel`` so they land in ``setup.extra_params`` and ``/slm/results``.

THE OUTWARD-LEG CONTROL (why a single step=0 cell is NOT enough)
----------------------------------------------------------------
``s_out`` is constant but ``n_out`` varies enormously across the grid (a few steps to ~400), so the
``step=0`` control does not subtract the outward leg's own cost the way the radial campaign's did.
Step 1 qualifies the outward leg to be lossless and cold out to the largest ``n_out`` in use, and a
dedicated control scan measures residual survival + heating vs ``n_out`` directly. Analysis divides
by that, per cell, using this shot's own ``ppg_peak_travel``.

PUPIL-SAMPLING CAP (2026-08-07)
-------------------------------
The exact spherical map has NO rho^4 residual (that is the parabolic-Z4 problem), so commanded-map
aberration is zero at any amplitude. The real limit is phase-gradient aliasing,
``grad_px = A*k*NA^2*rho / (512*sqrt(1-NA^2*rho^2))``, validated to 1% against recorded diag
(2.36/4.71/7.07 rad/px measured at 48/96/144 um peak). The alias front reaches the panel corner at
65 um peak, the INSCRIBED CIRCLE at 245 um, and the beam 1/e^2 at 429 um; beam-weighted power lost
is 0.02% at 100 um, 0.44% at 200 um, 1.4% at 245 um, 3.3% at 300 um.

So the commanded peak is kept **<= PEAK_MAX_UM (250 um)**. A product grid cannot drop individual
cells, so the campaign splits the step x nsteps map into a LOW-n scan (full step range) and a
HIGH-n scan (truncated step range) instead of silently running over. :func:`build` REFUSES to build
a grid whose worst cell exceeds the cap unless ``--allow-over-cap`` is passed, and always prints the
worst-cell peak.

Modes (dim 2)
-------------
``nsteps``  return_nsteps      -- survival vs step size x step count (campaign step 3a)
``period``  return_step_period_ms -- the period sweep 1x..8x of 0.696 ms (step 3c)
``corr``    return_depth_piston_corr -- the return-leg piston null (step 2)
``ruler``   return_nsteps at a FIXED known-good return step -- Probe A, commanded vs actual travel

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialPseudoOneWayScan.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialPseudoOneWayScan.py --force
"""

import argparse
import json
import math
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# LOADING DEFOCUS / rearrange focal plane. -5, matching the current production plane (LACScan.py
# and the server's sticky `loading_zernike` = [0,0,0,0,-5]); the 07-31 and 08-06 axial scans used
# -4, so expect a small offset against their cliffs (2.93 um @ 0.696 ms, 3.73 um @ 3.0 ms) that is
# a focal-plane difference, not physics.
#
# NOTE -- defocus is NOT the loading knob. An earlier hypothesis blamed -4-vs-5 for the low loading
# on 2026-08-07 and was REFUTED by direct A/B: both read ~0.07 back to back. The apparent
# "-5 loads better" correlation in the run history was confounded by time (all the -5 runs predated
# the drop). The real cause was the 399 blue-capture detuning -- see BLUE_DETUNING_MHZ below.
DEFOCUS = -5.0

# ---------------------------------------------------------------------------------------------
# LOADING RECOVERY (2026-08-07 jobs 309 + 310). expConfig.py is NOT modified -- these are applied
# as per-scan g() overrides, per the user's instruction.
#
# The 399 BLUE CAPTURE detuning had drifted off its plateau, and not by a little: at the old
# -40/-42 MHz work-point loading read **0.002** (job 309, fixed 0.30 s LoadingTime). The live
# plateau is -46..-50 MHz, centre -48, where the same 0.30 s gives 0.493. CV bottoms there too
# (0.48 vs 12+ on the dead side), so rate and uniformity agree on the same optimum.
#
# That one knob explains the whole night: the collisional-blockade ceiling was always intact
# (job 308 reached 0.60 given ~2 s), but the loading RATE had collapsed, so every scan running at
# expConfig's short LoadingTime sat below the shifted cliff and read 0.07-0.26. Two other
# hypotheses were tested and REFUTED first -- loading defocus (z4 -4 vs -5 both read ~0.07 back to
# back) and a stray override in these modules (a config diff against the last known-good rearrange
# scan, 08-06 20:16 at 0.602, showed only NumImages/NumPerGroup/name differing).
#
# Re-reading the cliff at -48 (job 310) put the knee back where it belongs: 0.215 at 0.10 s,
# 0.392 at 0.15, 0.467 at 0.25, 0.552 at 0.55. 0.25 s is taken as the work-point -- NOT the
# plateau, deliberately: 0.467 is well clear of the ~0.35 usable bar and LoadingTime is the
# dominant cycle-time cost, so buying the last 0.08 of fill would cost 2x the shot time for a
# campaign that has to finish overnight.
BLUE_DETUNING_MHZ = -48.0
# 2026-08-07 07:50, job 390 vs job 386 -- a DIRECT A/B at identical detuning (-48 MHz), bias
# (0.0343 A) and pattern (33x33_feedback11, 1068 sites): LoadingTime 0.25 s -> fill 0.153,
# LoadingTime 0.467 s -> fill 0.378. The 0.25 s here was a leftover from before the blue-capture
# recovery and is what produced the "unexplained global loading decay" chased for most of the
# night: the loading-optimization scans ran at 0.467 while every SURVIVAL scan still commanded
# 0.25, so the two measurements were never of the same rig state. Costs +0.217 s/shot, which buys
# back far more than it spends (2.5x the atoms per shot).
#
# 2026-08-08 21:31 (PHASE 2, jobs 594/595/596) -- all three loading knobs RE-MEASURED from scratch
# at the start of the session, in the unsaturated regime, in the runbook's order (alignment, then
# blue capture, then the cliff). Every one of them had moved:
#   bias-X    job 594: plateau 0.034-0.035, and the x-GRADIENT nulls at 0.03441 (r_x +0.182 at
#             0.034, -0.260 at 0.035). Rate and uniformity pick the same point. -> 0.03441
#   detuning  job 595: plateau -46..-50 MHz, centre -48, CV minimum at the same place. expConfig's
#             -44 sits 13% low on the rising edge. -> -48 CONFIRMED unchanged from phase 1
#   cliff     job 596: knee back at 0.15 s (it was ~2 s during the collapse), ceiling 0.586.
#             0.40 s reaches 0.580 = 99% of it, and is INTERIOR to the plateau so it has drift
#             margin on both sides; 0.55 s buys 0.006 more fill for 0.15 s/shot. -> 0.40
# Resulting state: loading 0.58, CV 0.295, r_x ~= 0. Phase 1 ran this campaign at 0.15-0.28.
BLUE_LOADING_TIME_S = 0.40
# Residual: r_y ~ +0.13-0.19 across every cell of all three scans -- a real vertical gradient that
# bias-X cannot fix. GreenMOT.BiasCoilCurrent.Y is the knob for it (runbook step 7). Not chased
# here because it is a uniformity effect on a per-site CONDITIONAL measurement, but it is the next
# loading improvement available.
BLUE_BIAS_X_A = 0.03441

# LAC (light-assisted collisions) pulse length. 2026-08-09, user request: raise 30 -> 35 ms for the
# remainder of the campaign, as a per-scan g() override -- expConfig is NOT modified.
# The ByPattern 33x33_feedback11 overlay sets LAC.Time = 30e-3 (base is 20e-3), so this g() override
# is what makes 35 ms actually reach the sequence -- g() beats the pattern overlay.
# PROVENANCE BOUNDARY: jobs <= 610 ran at the overlay's 30 ms; 611 onward run at 35 ms.
LAC_TIME_S = 35e-3

# 399 IMAGING DETUNING. expConfig (and the ByPattern 33x33_feedback11 overlay) hold -5 MHz; this
# campaign runs at -1.0 MHz as a g() override -- expConfig is NOT modified.
#
# 2026-08-09, rounds 103 (job 623) + 104 (job 624), after the user repaired the 399 detuning PID
# lock. Sweeping -9.0 .. +2.0 MHz at 0 pushout, the threshold-free separation rises monotonically
# from 5.27 ADU at -9.0 to a clear INTERIOR peak of 7.52 at -1.5 and falls again to 7.07 at +2.0,
# with survival a broad max near 0.0 and loading dead flat at 0.597-0.607 across all 29 cells (so
# it is a detection effect, not a loading artifact). -1.0 sits between the separation peak and the
# survival peak and is level with or ahead of -1.5 on both once the two rounds are averaged.
# Net vs -5.0: separation +9.2% (6.55 -> 7.15 ADU measured within one round), d' 5.13 -> 5.40,
# survival +0.0006 (flat), loading unchanged.
#
# The three earlier detuning rounds of this session (r100/r101/r102, jobs 619-621) are VOID -- they
# ran while that PID lock was faulty, which is why they disagreed with each other at the SAME cell
# and why loading collapsed mid-r102. Do not cite them.
#
# NOTHING ELSE about imaging moved: the PIDset power map (r105) and both 556 cooling maps (r106 X,
# r107 h) were re-measured AT -1.0 and came back flat, so the provenance shift is one parameter
# wide. Power was re-measured rather than inherited on purpose -- a saturation verdict taken at -5
# does not transfer 4 MHz closer to resonance.
#
# PROVENANCE BOUNDARY: jobs <= 627 imaged at -5 MHz; later campaign jobs image at -1.0 MHz.
# Absolute survivals (including the step=0 controls) shift UP across this boundary; d99 is
# normalised per-scan by its own control and is insensitive to it.
#
# 2026-08-09 22:26 RE-PINNED -1.0 -> +4.5. The 399 DRIFTED ~6 MHz over the evening. Round 109
# (job 637, 22:21) found every metric rising monotonically to the +2.0 grid edge where rounds
# 103/104 at 13:39 had a clean interior peak at -1.5; round 110 (job 638) extended to +7.0 and
# found the new optimum as a broad plateau +4.0..+6.0 (d' 4.6-4.7, survival 0.9886-0.9894).
# +4.5 is the survival max and sits interior to that plateau.
#
# This is the same subsystem that failed this morning (the detuning PID lock), now drifting slowly
# rather than collapsing. It was found by chasing a separation decline that loading could not
# explain: separation fell 3x faster than loading (-24% vs -8%) across jobs 634/635, which rules
# out the MOT and the trap; the detuning sweep then separated FREQUENCY from POWER.
#
# NOT a reason to distrust the phase-2 deliverables. d99 was MEASURED to be insensitive to imaging
# detuning -- 1.2% across a deliberate 4 MHz change (jobs 609 vs 629, n >= 5) -- and every curve is
# normalised by its own plateau. Jobs 629-635 ran at -1.0 while the laser drifted through it, which
# cost STATISTICS (d' 4.20 -> ~3.4) but not central values. All runs retain full raw frames, so
# thresholds are re-derivable offline if the drift is ever suspected of biasing the logicals.
#
# CAVEAT: at the new optimum separation is 5.8 ADU vs 7.2 this afternoon and loading 0.546 vs 0.60,
# so ~20% of the decline is NOT recovered by retuning frequency -- 399 power or trap depth has also
# degraded. Re-measure before trusting absolute numbers taken after 2026-08-09 evening.
#
# 2026-08-10 10:12 RE-PINNED +4.5 -> +1.0. Round 111 (job 658) swept -7.0..+6.0, spanning BOTH
# previous optima in ONE scan, and found a broad plateau -1.0..+3.5 (separation 6.5-6.7 ADU) with
# the peak at +1.0 (sep 6.7, d' 5.07, survival 0.9944, loading 0.606).
#
# The 399 has drifted BACK partway overnight: peak -1.5 at 13:39 on 08-09, +4.0..+6.0 at 22:21,
# now -1.0..+3.5. Amplitude of the excursion ~6 MHz out and ~3 MHz back in under 24 h. Treat the
# imaging detuning as a knob that must be RE-MEASURED at the start of every session, not inherited.
#
# CORRECTION of an inference made at 10:04: it was claimed the +4.5 pin was "stale in the wrong
# direction" because the user's job 657 at the expConfig default -5 read separation 6.05 while
# +4.5 had given 5.93 the previous night. That compared DIFFERENT SCAN TYPES on DIFFERENT DAYS and
# was not a valid comparison. This same-scan sweep shows +4.5 (6.3) is in fact clearly BETTER than
# -5 (5.5); the pin was merely off-plateau, not backwards.
#
# Loading was 0.594-0.614 across all 27 cells, i.e. the MOT is at its optimum and bias-X needed no
# re-scan (it was verified at 0.0345 on 08-09, and the campaign pin 0.03441 sits inside that
# plateau).
IMAG_DETUNING_MHZ = 1.0

BASE_PERIOD_MS = 0.696        # the SLM write floor; every period is an integer multiple
OUT_PERIOD_MS = 0.696         # the OUTWARD leg pacing (fast; the outward step is small)
HOLD_MS = 20.0                # dwell at the turnaround so the LC is fully settled before the test
OUT_STEP_MAX = 1.0            # um per outward step -- PROVISIONAL until campaign step 1 lands
PISTON_CORR = 0.617           # rad/um, measured true_defocus null (job 320, 2026-07-29)
RET_PISTON_CORR = None        # None -> inherit PISTON_CORR; set from campaign step 2
PEAK_MAX_UM = 250.0           # pupil-sampling cap (alias front hits the inscribed circle at 245)

# The campaign step grid: 18 magnitudes, max gap 0.5 um, 0.3 um through the 0.696 ms cliff (~2.93)
# and the 2.784 ms cliff (~3.73). Reaches 0 survival on both. Same grid for both periods.
STEP_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5,
             3.8, 4.1, 4.4, 4.7, 5.0, 5.4, 5.8, 6.2]
NSTEPS_GRID = [0, 1, 2, 5, 10, 20, 30, 40, 50]
PERIOD_MULTS = [1, 2, 3, 4, 5, 6, 7, 8]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json(pattern=None, phase_path=None):
    """Both camera frames are the SAME array at the SAME place -- the pseudo-one-way triangle ends
    at displacement 0, which IS the stored WGS phase. Name = the phase-file BASENAME (what the
    detection registry + expConfig ByPattern are keyed by)."""
    it = {"name": pattern or PATTERN, "base_phase_path": phase_path or PHASE_PATH,
          "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _worst_peak(steps, nsteps_list):
    """Largest commanded peak travel |n*s| over the product grid, in um."""
    if not steps or not nsteps_list:
        return 0.0
    return max(abs(float(n) * float(s)) for s in steps for n in nsteps_list)


def plan(direction=+1, steps=None, nsteps_list=None, out_step_max=OUT_STEP_MAX, n_out_min=0):
    """Per-cell schedule preview: n_out, s_out, frames. Mirrors _fold_ppg_outward_leg exactly."""
    steps = STEP_GRID if steps is None else steps
    nsteps_list = NSTEPS_GRID if nsteps_list is None else nsteps_list
    rows = []
    for s in steps:
        for n in nsteps_list:
            D = abs(float(n) * float(s))
            if D == 0:
                rows.append((s, n, 0.0, 0, 0.0, 1))
                continue
            n_out = max(1, int(math.ceil(D / out_step_max)), int(n_out_min))
            rows.append((s, n, D, n_out, D / n_out, n_out + n + 1))
    return rows


def build(direction=+1, mode="nsteps", steps=None, nsteps_list=None, period_mults=None,
          corr_list=None, ruler_step=None, hold_list=None, const_nout=True,
          fixed_nsteps=50, period_ms=BASE_PERIOD_MS,
          out_step_max=OUT_STEP_MAX, hold_ms=HOLD_MS, piston_corr=PISTON_CORR,
          ret_piston_corr=RET_PISTON_CORR, allow_over_cap=False, label_extra="",
          img_pid=(0.80, 1.00), xcool=None, hcool=None, img_det=IMAG_DETUNING_MHZ,
          inherit_loading=False, pattern=None, phase_path=None):
    """Build (do NOT submit) the 2-D pseudo-one-way ScanGroup. Returns ``(seq_name, g, meta)``.

    ``direction`` is the direction of the TESTED (return) move: +1 = +z, -1 = -z. The commanded
    amplitudes are the NEGATIVE of that on both legs (out to -D, back to 0), per trap 1.
    """
    _bootstrap()
    from scan_group import ScanGroup

    sgn = -1.0 if direction > 0 else +1.0      # tested +z  =>  both legs commanded negative
    steps = list(STEP_GRID if steps is None else steps)
    nsteps_list = list(NSTEPS_GRID if nsteps_list is None else nsteps_list)
    period_mults = list(PERIOD_MULTS if period_mults is None else period_mults)

    if mode == "hold":
        n_for_cap = [0]
    elif mode in ("nsteps", "ruler"):
        n_for_cap = nsteps_list
    else:
        n_for_cap = [fixed_nsteps]
    peak = _worst_peak([ruler_step] if mode == "ruler" and ruler_step else steps, n_for_cap)
    if peak > PEAK_MAX_UM and not allow_over_cap:
        raise ValueError(
            "pseudo-one-way: worst-cell commanded peak %.1f um exceeds the %.0f um "
            "pupil-sampling cap (alias front reaches the inscribed circle at 245 um). Split the "
            "grid into a low-n scan with the full step range and a high-n scan with a truncated "
            "one, or pass allow_over_cap=True and say so in the run description."
            % (peak, PEAK_MAX_UM))

    # Array override (added 2026-09-01): the 17x17_20um axial series needs a different
    # pattern, and the depth knob spans 56x there vs 1.8x on 33x33_feedback11.
    pattern = pattern or PATTERN
    phase_path = phase_path or PHASE_PATH
    seq_name = "RearrangeCommSeq"
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -----------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = phase_path
    rp.warmup_kwargs.final_phase = phase_path
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

    # ---- rearrange_kwargs -------------------------------------------------------------------

    # LOADING RECOVERY (jobs 309/310): 399 blue capture had drifted off plateau -- the old
    # -40/-42 MHz work-point read 0.002 fill. -48 MHz + 0.25 s restores 0.467. Applied as a
    # per-scan g() override; expConfig.py is NOT modified.
    # 2026-08-31: `inherit_loading` SKIPS this whole block.  The constants below were measured
    # 2026-08-07..09; the 399 blue detuning walks ~6 MHz/day and the GreenMOT bias-X resonance is
    # near-vertical (0.01 A = full loading vs zero, per the runbook).  A session that has just
    # re-tuned loading -- as 08-31 did, `33x33_feedback11 loading recovery 08-31: GreenMOT bias
    # X x Y grid` -- leaves the CURRENT optimum in expConfig / the ByPattern overlay, and g()
    # BEATS that overlay, so pinning 3-week-old constants would silently overwrite a fresh
    # calibration with a stale one.  Inheriting runs the scan in the state the rig is actually in.
    if not inherit_loading:
        g().BlueMOT.FreqDetuning = BLUE_DETUNING_MHZ * 1e6

    # GREENMOT BIAS-X (jobs 375 + 376, 2026-08-07 05:50). expConfig holds 0.040 A (the 06/05
    # optimum); the resonance has DRIFTED and 0.040 is now DEAD -- job 375 measured 0.002
    # loading there against 0.309 at 0.035, and the fine scan 376 put the peak at 0.0345
    # (0.442 loading, monotonic shoulders both sides, full width ~3 mA).
    #
    # THIS was the slow loading decline (0.61 -> 0.23 over 90 min). Ruled out first, by
    # measurement rather than assumption: the 399 blue detuning (job 367 re-verified -48 as
    # optimal, curve identical to job 309 four hours earlier), the oven (372.06 vs 372.08 C)
    # and imaging (which was IMPROVING, d-prime 11.3). The MOT coils cooled 19.2 -> 17.3 C
    # over the same window, which drifts the field and hence the cloud position relative to
    # the array -- the runbook flags bias-X as a near-vertical resonance where 0.01 A is the
    # difference between full loading and zero, and this is that failure in the wild.
    if not inherit_loading:
        g().GreenMOT.BiasCoilCurrent.X = BLUE_BIAS_X_A   # 08-08 job 594: rate plateau + x-grad null
        g().BlueMOT.LoadingTime = BLUE_LOADING_TIME_S
        g().LAC.Time = LAC_TIME_S                        # 08-09 user request: 30 -> 35 ms

    # IMAGING (jobs 331 + 340/341/342, 2026-08-07). Pinned EXPLICITLY at the documented
    # ByPattern 33x33_feedback11 values so every descriptor records them; this is NOT a
    # deviation from config.
    #
    # A 3-point A/B inside THIS sequence (static no-motion cells, 20 shots each) showed
    # PIDSet is NOT the lever here: 0.8/1.0 -> sep 4.04 ADU / d' 6.8, 0.9/0.64 -> 3.80/6.4,
    # 0.63/0.43 -> 3.81/6.5. Identical within noise across a 2.3x span of Img2PIDSet.
    #
    # THE REAL GAP IS THE SEQUENCE, NOT THE POWER. The imaging sequence
    # (ImagingPushoutSurvivalSeq, job 331) reaches sep 7.58 ADU / d' 12.6 at the same nominal
    # setpoint, i.e. ~2x this sequence. Prime suspects, both documented and both OUTSIDE the
    # amps-only scope authorised for this campaign: (a) the imaging tool pins a different 556
    # imaging cooling (h at 0.22 MHz / 0.20 vs the pattern's 0.16 / 0.13), and (b) memory
    # gotcha-imaging-pid-held-multiround-rearrange -- the imaging PID locks once at the root
    # BlueMOT and per-pattern PIDSet never reaches the rearrange images.
    #
    # CAMPAIGN IMAGING CONDITION TO QUOTE WITH EVERY RESULT: separation ~3.9 ADU, d' ~6.5,
    # static two-image survival ~0.97, loading ~0.66. Absolute survivals sit on that floor;
    # per-scan control normalization removes it, bare numbers do NOT. If the sequence gap is
    # closed later, every absolute number here shifts up and the campaign should be re-quoted.
    g().BlueMOT.Img1PIDSet = float(img_pid[0])
    g().BlueMOT.Img2PIDSet = float(img_pid[1])

    # 399 imaging detuning -- see IMAG_DETUNING_MHZ above for the measurement and the boundary.
    # g() beats the ByPattern overlay, so this is what actually reaches the image frames.
    if img_det is not None:
        g().Imag399.FreqDetuning = float(img_det) * 1e6

    # 556 COOLING DURING THE IMAGE. Left as None this changes nothing (the pattern overlay is
    # whatever it was); passed explicitly it pins the imaging cooling as a g() override, which is
    # the imaging runbook's "working set W" method -- W lives in g() until it is locked in, and
    # expConfig is not edited.
    #
    # WHY THIS HOOK EXISTS (2026-08-08): the phase-1 campaign recorded the rearrange sequences at
    # sep 3.9-4.3 ADU / d' 6.5-7.4 while ImagingPushoutSurvivalSeq reached 7.58 ADU / d' 12.6 on
    # the SAME array minutes apart, and attributed the gap partly to a different imaging cooling.
    # Its note has the two configurations backwards: h (0.22 MHz, 0.20) is the ByPattern
    # 33x33_feedback11 overlay and h (0.16, 0.13) is the BASE config. So the open question is
    # whether the per-pattern Imag399.Cool556 overlay reaches THIS sequence's image frames at all
    # -- the same class of failure as the gotcha-imaging-pid-held-multiround-rearrange memory.
    # Setting it here bypasses the question entirely: an explicit g() override beats the overlay.
    if xcool is not None:
        g().Imag399.Cool556.X.FreqDetuning = float(xcool[0]) * 1e6
        g().Imag399.Cool556.X.Amp = float(xcool[1])
    if hcool is not None:
        g().Imag399.Cool556.h.FreqDetuning = float(hcool[0]) * 1e6
        g().Imag399.Cool556.h.Amp = float(hcool[1])

    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"

    # OUTWARD leg pacing. With return_step_period_ms set this paces the outward leg ONLY
    # (through the turnaround frame) -- verified in the dispatcher's frame_gap_s construction.
    rk.step_period_ms = float(OUT_PERIOD_MS)
    # PLACEHOLDER ONLY: _fold_ppg_outward_leg overwrites nsteps per shot from the resolved
    # return leg. Only the dequeue-time warmup setup_rearrangement ever sees this value.
    rk.nsteps = 1

    # THE SWITCH that activates the derived outward leg. Magnitude, in the tested leg's own
    # numeraire (um under true_defocus). Consumed by the fold; never reaches the server.
    rk.extras.out_step_max = float(out_step_max)

    # CONSTANT-n_out. Precompute cost AND outward motion both scale with n_frames, which under a
    # plain ceil(|D|/out_step_max) scales with D -- so the per-shot IN-TRAP DELAY would be
    # perfectly correlated with the tested step size, and vacuum loss during precompute would
    # masquerade as transport loss growing with stroke. Pinning n_out to the WORST cell's value
    # makes the delay constant across the grid, demoting that to a common-mode offset the step=0
    # control absorbs. Costs the small-D cells some extra (gentler) outward frames; buys a clean
    # measurement of the axis we actually care about.
    n_out_min = 0
    if const_nout and peak > 0:
        n_out_min = int(math.ceil(peak / float(out_step_max)))
        rk.extras.out_nsteps_min = n_out_min

    # ---- the TESTED (return) leg -------------------------------------------------------------
    if mode == "nsteps":
        # dim 1 tested per-step amplitude, dim 2 tested step count.
        rk.extras.return_step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_nsteps.scan(2, [int(n) for n in nsteps_list])
        rk.extras.return_step_period_ms = float(period_ms)
    elif mode == "period":
        # dim 1 tested per-step amplitude, dim 2 tested PERIOD (integer multiples of the floor).
        rk.extras.return_step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_nsteps = int(fixed_nsteps)
        rk.extras.return_step_period_ms.scan(
            2, [round(m * BASE_PERIOD_MS, 6) for m in period_mults])
    elif mode == "corr":
        # dim 1 tested per-step amplitude, dim 2 the RETURN leg's own piston constant. The
        # outward leg stays pinned at the established 0.617 (it is small-step and known good),
        # so this isolates the tested leg's null. THIS is the campaign-step-2 measurement.
        rk.extras.return_step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_nsteps = int(fixed_nsteps)
        rk.extras.return_step_period_ms = float(period_ms)
        rk.extras.return_depth_piston_corr.scan(2, [float(c) for c in corr_list])
    elif mode == "hold":
        # TRAP-LIFETIME / IN-SHOT-DELAY calibration. NO motion at all (step 0, nsteps 0 -> the
        # fold's static branch, a single WGS write); dim 1 sweeps hold_ms, which the dispatcher
        # honours even at n_frames == 1 (rearrange_actual.py busy-waits to t_ref after the loop
        # precisely for this case). The atoms sit in the WGS traps for hold_ms with nothing else
        # happening -- the exact condition they are in during precompute -- so survival vs hold_ms
        # IS the loss we pay for a long per-shot precompute. Fit tau, then read off the loss at the
        # campaign's worst-case delay and decide between constant-n_out and a pre-precompute.
        rk.extras.return_step_size = 0.0
        rk.extras.return_nsteps = 0
        rk.extras.return_step_period_ms = float(period_ms)
        rk.extras.hold_ms.scan(1, [float(h) for h in hold_list])
    elif mode == "holdsweep":
        # THE TURNAROUND PENALTY, MEASURED DIRECTLY. dim 1 = tested per-step amplitude, dim 2 = the
        # dwell at the turnaround. Everything else is pinned, so d99-vs-hold isolates the settle
        # cost with no cross-geometry or cross-day differencing.
        #
        # WHY THIS EXISTS (2026-08-10). The campaign's turnaround term had only ever been INFERRED,
        # by differencing one-way (20 ms hold, n=50) against round-trip (0-5 ms hold, N=80) runs
        # taken on different days -- two geometries, two hop counts, and day-to-day drift, all
        # varying at once. The only same-protocol hold comparison available is 5 ms (07-30 period
        # sweep, s99 2.983) vs 0 ms (08-04 panel b, d99 2.921) = +2.1%, which sits BELOW the 6.8%
        # spread of the four round-trip measurements at 0.696 ms and therefore cannot be claimed.
        #
        # NOTE `--mode hold` is a DIFFERENT measurement: it pins step 0 and nsteps 0 for a static
        # trap-lifetime/precompute-delay curve. This mode keeps the tested leg MOVING, which is the
        # only way the LC-settle picture can be tested.
        rk.extras.return_step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_nsteps = int(fixed_nsteps)
        rk.extras.return_step_period_ms = float(period_ms)
        rk.extras.hold_ms.scan(2, [float(h) for h in hold_list])
    elif mode == "ruler":
        # PROBE A. The tested leg is a FIXED known-good small step; dim 2 sweeps how MANY of them
        # the return leg takes. Survival peaks where the return travel matches the ACTUAL outward
        # travel, so the peak location measures commanded-vs-delivered displacement in microns.
        # dim 1 is the commanded outward travel D (expressed as the step the outward leg is asked
        # to undo) -- see the caller, which passes `steps` as D/fixed_nsteps.
        rk.extras.return_step_size = sgn * abs(float(ruler_step))
        rk.extras.return_nsteps.scan(2, [int(n) for n in nsteps_list])
        rk.extras.return_step_period_ms = float(period_ms)
    else:
        raise ValueError("unknown mode %r" % (mode,))

    # ---- axial mode, set EXPLICITLY (server extras are STICKY and merge-only) ----------------
    # A cleared flag silently reinterprets step_size as radians of Z4 (0.80x the travel) or as a
    # lateral blaze -- the single most likely way to command defocus and not get it. The analysis
    # asserts diag.axial_step_unit == "um_axial" on every shot.
    rk.extras.depth = True
    rk.extras.true_defocus = True
    rk.extras.depth_piston_corr = float(piston_corr)          # OUTWARD leg
    if mode != "corr" and ret_piston_corr is not None:
        rk.extras.return_depth_piston_corr = float(ret_piston_corr)   # TESTED leg
    # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (deprecated
    # legacy beam-model path) and is sticky, which would pin every point to a stale piston.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0
    # depth_fill_center: left UNSET -> the server's measured beam centroid (-0.0684, +0.0117).
    # return_true_defocus: left UNSET -> the return leg inherits the outward branch (both exact).

    rk.extras.return_trip = True
    if mode not in ("hold", "holdsweep"):   # these SCAN hold_ms; do not overwrite the axis
        rk.extras.hold_ms = float(hold_ms)

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = pattern
    rk.extras.final_pattern = pattern

    # ---- run params ---------------------------------------------------------------------------
    rp.NumPerGroup = 100000                   # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(pattern, phase_path)

    meta = dict(mode=mode, direction=direction, steps=steps, nsteps_list=nsteps_list,
                period_mults=period_mults, fixed_nsteps=fixed_nsteps, period_ms=period_ms,
                out_step_max=out_step_max, hold_ms=hold_ms, piston_corr=piston_corr,
                ret_piston_corr=ret_piston_corr, worst_peak_um=peak, npts=g.nseq(),
                n_out_min=n_out_min, const_nout=bool(const_nout), hold_list=hold_list,
                label_extra=label_extra)
    return seq_name, g, meta


def _desc(meta):
    d = "+z" if meta["direction"] > 0 else "-z"
    if meta["mode"] == "hold":
        return (
            "TRAP-LIFETIME / IN-SHOT-DELAY calibration (pingponggrating, NO MOTION). "
            "step_size = 0 and nsteps = 0, so the fold's static branch gives a SINGLE WGS write "
            "and the atoms simply sit in the loading traps for hold_ms -- which the dispatcher "
            "honours even at n_frames == 1 (it busy-waits to t_ref after the loop precisely for "
            "this case). dim 1 = hold_ms %s ms. "
            "PURPOSE: the pseudo-one-way shots carry a large per-shot PRECOMPUTE delay -- measured "
            "0.5-1.4 ms/frame on the asymmetric-leg path (which bypasses the setup-time uint8 "
            "cache), so ~350-frame shots sit near 500 ms, all of it with the atoms held. This scan "
            "measures what that costs directly. Survival vs hold_ms gives the trap lifetime; the "
            "loss at the campaign's worst-case delay decides whether the main scans run with "
            "CONSTANT n_out (delay equal across the grid -> a common-mode offset the step=0 "
            "control absorbs) or need a server-side pre-precompute. The threat is not the absolute "
            "delay but that under a plain ceil(|D|/out_step_max) the delay is perfectly CORRELATED "
            "with the tested step size, so vacuum loss would masquerade as transport loss growing "
            "with stroke. Array %s, z4 = loading_defocus = %+.0f. %s"
            % (meta.get("hold_list"), PATTERN, DEFOCUS, meta["label_extra"]))
    return (
        "AXIAL PSEUDO-ONE-WAY %s transport (pingponggrating asymmetric legs, true_defocus). "
        "The TESTED move is the RETURN leg: the array first walks OUT in %s in n_out small steps "
        "of <= %.3f um at %.3f ms (qualified lossless + cold by campaign step 1), dwells "
        "hold_ms=%.0f ms at the turnaround so the LC is fully settled, then makes the move under "
        "test back to displacement 0. Displacement 0 IS the stored WGS phase byte-for-byte, so the "
        "array lands exactly where it started and LIVE img1->img2 detection is valid -- unlike the "
        "radial one-way campaign, NO offline re-detection is needed and stored logicals_img2 ARE "
        "usable. The tested leg makes n steps (NOT 2n), so per-step survival is S**(1/n) after "
        "dividing out the outward leg's own cost (measured separately vs n_out). "
        "SIGN: the dispatcher's displacement index is non-negative and direction lives in the "
        "amplitude sign, so BOTH legs are commanded %s to test a %s move. "
        "n_out is DERIVED per shot by _fold_ppg_outward_leg as ceil(|D|/out_step_max) with "
        "s_out = D/n_out, which makes the turnaround exactly continuous "
        "(n_out*s_out == n_ret*s_ret to ~1e-13 um); diag.leg_out_peak / leg_ret_peak record both. "
        "dim1 = tested per-step amplitude (MICRONS, true_defocus exact spherical map), dim2 = %s. "
        "Worst-cell commanded peak %.1f um (pupil-sampling cap %.0f um: the alias front reaches "
        "the inscribed circle at 245 um; beam-weighted loss 0.44%% at 200 um, 1.4%% at 245). "
        "depth_piston_corr=%.3f rad/um on the outward leg%s. depth_fill_frac cleared (it would "
        "override the corr). %s Array %s, z4 = loading_defocus = %+.0f. %s"
        % (d, "-z" if meta["direction"] > 0 else "+z", meta["out_step_max"], OUT_PERIOD_MS,
           meta["hold_ms"], "negative" if meta["direction"] > 0 else "positive", d,
           {"nsteps": "tested step COUNT return_nsteps",
            "period": "tested PERIOD return_step_period_ms (integer multiples of 0.696 ms)",
            "corr": "the RETURN leg's own piston constant return_depth_piston_corr",
            "ruler": "return_nsteps at a FIXED known-good return step (PROBE A: the survival peak "
                     "locates the ACTUAL outward travel, so this measures commanded vs delivered "
                     "displacement)",
            "holdsweep": "the TURNAROUND DWELL hold_ms, with the tested leg still moving -- so "
                         "d99 vs hold isolates the LC settle cost directly, with no cross-geometry "
                         "or cross-day differencing (unlike --mode hold, which is static)",
            "hold": "hold_ms with NO motion at all (static trap-lifetime / precompute-delay "
                    "calibration)"}[meta["mode"]],
           meta["worst_peak_um"], PEAK_MAX_UM, meta["piston_corr"],
           ("" if meta["ret_piston_corr"] is None
            else ", %.3f rad/um on the tested leg" % meta["ret_piston_corr"]),
           ("CONSTANT n_out = %d on EVERY cell (out_nsteps_min), so n_frames -- and therefore the "
            "per-shot precompute + outward-motion DELAY -- is the same for every step size. Under "
            "a plain ceil(|D|/out_step_max) that delay would scale with D and be perfectly "
            "correlated with the tested step, letting vacuum loss during precompute masquerade as "
            "transport loss growing with stroke; pinning n_out demotes it to a common-mode offset "
            "the step=0 control absorbs (small-D cells just take smaller, gentler outward steps). "
            "The static step=0 control does NOT pay the delay, so expect a single uniform "
            "control-vs-data offset, quantified by the hold_ms lifetime scan."
            % meta["n_out_min"]) if meta.get("const_nout") and meta.get("n_out_min")
           else "n_out = ceil(|D|/out_step_max) per cell (NOT constant).",
           PATTERN, DEFOCUS, meta["label_extra"]))


def submit(url=None, reps=None, label=None, **kw):
    seq_name, g, meta = build(**kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    lbl = label or ("PPGAxialPseudoOneWay_%s_%s"
                    % ("pz" if meta["direction"] > 0 else "nz", meta["mode"]))
    did = ybStartScan(seq_name, g, url=url, label=lbl, description=_desc(meta), **opts)
    print("submitted %s -> descriptor id %s (%d pts; worst peak %.1f um; out_step_max %.3f um; "
          "hold %.0f ms; url=%s)"
          % (lbl, did, meta["npts"], meta["worst_peak_um"], meta["out_step_max"],
             meta["hold_ms"], url or "default"))
    return did


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()] if s else None


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Axial pseudo-one-way transport scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--mode", default="nsteps",
                    choices=("nsteps", "period", "corr", "ruler", "hold", "holdsweep"))
    ap.add_argument("--hold-list", default=None,
                    help="hold_ms values for mode=hold (comma-separated)")
    ap.add_argument("--no-const-nout", action="store_true",
                    help="use n_out = ceil(|D|/out_step_max) per cell instead of a constant n_out. "
                         "Constant n_out is the DEFAULT: it makes the per-shot precompute delay "
                         "the same for every step size, so vacuum loss cannot masquerade as "
                         "stroke-dependent transport loss. Only valid once the outward leg is "
                         "qualified flat across the whole 0..out_step_max range at that n.")
    ap.add_argument("--direction", type=int, default=+1, choices=(+1, -1),
                    help="direction of the TESTED move: +1 = +z, -1 = -z")
    ap.add_argument("--steps", default=None, help="tested step MAGNITUDES, um (comma-separated)")
    ap.add_argument("--nsteps-list", default=None, help="tested step COUNTS (comma-separated)")
    ap.add_argument("--period-mults", default=None,
                    help="tested periods as integer multiples of 0.696 ms")
    ap.add_argument("--corr-list", default=None,
                    help="return_depth_piston_corr values, rad/um (mode=corr)")
    ap.add_argument("--ruler-step", type=float, default=None,
                    help="fixed known-good return step in um (mode=ruler)")
    ap.add_argument("--fixed-nsteps", type=int, default=50)
    ap.add_argument("--period", type=float, default=BASE_PERIOD_MS,
                    help="tested-leg period in ms (modes nsteps/corr/ruler)")
    ap.add_argument("--out-step-max", type=float, default=OUT_STEP_MAX)
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR, help="OUTWARD leg depth_piston_corr")
    ap.add_argument("--ret-corr", type=float, default=RET_PISTON_CORR,
                    help="TESTED leg return_depth_piston_corr (default: inherit)")
    ap.add_argument("--allow-over-cap", action="store_true")
    ap.add_argument("--img-pid", type=float, nargs=2, metavar=("IMG1","IMG2"),
                    default=(0.80, 1.00),
                    help="BlueMOT.Img1/Img2PIDSet (V). g() beats the ByPattern overlay, "
                         "whose 33x33_feedback11 value is 0.8/1.0.")
    ap.add_argument("--xcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="pin Imag399.Cool556.X (detuning MHz, amp) as a g() override. Omit to "
                         "leave the pattern overlay alone.")
    ap.add_argument("--hcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="pin Imag399.Cool556.h (detuning MHz, amp) as a g() override. The "
                         "ByPattern 33x33_feedback11 values are h (0.22, 0.20) / X (0.16, 0.23); "
                         "the BASE config is h (0.16, 0.13) / X (0.16, 0.20).")
    ap.add_argument("--img-det", type=float, default=IMAG_DETUNING_MHZ,
                    help="399 imaging FreqDetuning (MHz) as a g() override; default %(default)s, "
                         "measured 2026-08-09 rounds 103/104. Pass the old -5.0 to reproduce a "
                         "pre-boundary run.")
    ap.add_argument("--note", default="", help="appended verbatim to the run description")
    ap.add_argument("--pattern", default=None,
                    help="loading pattern name (default %s)" % PATTERN)
    ap.add_argument("--phase-path", default=None, dest="phase_path",
                    help="server-side phase file (default %s)" % PHASE_PATH)
    ap.add_argument("--inherit-loading", action="store_true",
                    help="do NOT pin the 08-07/08-09 blue-detuning / bias-X / LoadingTime / "
                         "LAC.Time g() overrides; run in whatever state the rig is currently "
                         "calibrated to (use after a fresh loading re-tune)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    kw = dict(direction=args.direction, mode=args.mode, steps=_floats(args.steps),
              nsteps_list=_ints(args.nsteps_list), period_mults=_ints(args.period_mults),
              corr_list=_floats(args.corr_list), ruler_step=args.ruler_step,
              hold_list=_floats(args.hold_list), const_nout=not args.no_const_nout,
              fixed_nsteps=args.fixed_nsteps, period_ms=args.period,
              out_step_max=args.out_step_max, hold_ms=args.hold_ms, piston_corr=args.corr,
              ret_piston_corr=args.ret_corr, allow_over_cap=args.allow_over_cap,
              label_extra=args.note, img_pid=tuple(args.img_pid),
              xcool=tuple(args.xcool) if args.xcool else None,
              hcool=tuple(args.hcool) if args.hcool else None,
              img_det=args.img_det, inherit_loading=args.inherit_loading,
              pattern=args.pattern, phase_path=args.phase_path)

    if args.dry_run:
        _seq, _g, _meta = build(**kw)
        print("seq=%s  nseq=%d  mode=%s  direction=%+d  worst_peak=%.1f um (cap %.0f)"
              % (_seq, _g.nseq(), _meta["mode"], _meta["direction"], _meta["worst_peak_um"],
                 PEAK_MAX_UM))
        print("tested steps [um] = %s" % _meta["steps"])
        print("dim2              = %s"
              % (_meta["nsteps_list"] if _meta["mode"] in ("nsteps", "ruler")
                 else _meta["period_mults"] if _meta["mode"] == "period" else args.corr_list))
        print("out_step_max=%.3f um  out_period=%.3f ms  hold=%.0f ms  corr=%.3f/%s rad/um"
              % (_meta["out_step_max"], OUT_PERIOD_MS, _meta["hold_ms"], _meta["piston_corr"],
                 _meta["ret_piston_corr"]))
        print("const_nout=%s  n_out_min=%d" % (_meta["const_nout"], _meta["n_out_min"]))
        if _meta["mode"] == "hold":
            print("hold_ms axis = %s" % _meta["hold_list"])
        else:
            n_used = (_meta["nsteps_list"] if _meta["mode"] in ("nsteps", "ruler")
                      else [_meta["fixed_nsteps"]])
            # A static-control-only grid (every tested step 0) has no moving cell to schedule --
            # print nothing rather than crashing the whole dry run on min() of an empty sequence.
            moving = [s for s in _meta["steps"] if s > 0]
            probe = ([args.ruler_step] if _meta["mode"] == "ruler"
                     else [min(moving), _meta["steps"][-1]] if moving else [])
            rows = plan(_meta["direction"], probe, n_used, _meta["out_step_max"],
                        _meta["n_out_min"])
            print("schedule (smallest and largest tested step -- s_out must stay <= %.3f um "
                  "and n_out CONSTANT for the delay to be common-mode):" % _meta["out_step_max"])
            for s, n, D, n_out, s_out, nf in rows:
                print("   s %5.2f  n_test %3d  D %6.1f um  n_out %4d  s_out %.4f um  frames %4d"
                      "  motion %6.1f ms"
                      % (s, n, D, n_out, s_out, nf,
                         n_out * OUT_PERIOD_MS + n * args.period + args.hold_ms))
        print("\nDESCRIPTION:\n%s" % _desc(_meta))
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, label=args.label, **kw)
