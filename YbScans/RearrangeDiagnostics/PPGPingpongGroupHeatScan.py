"""PPGPingpongGroupHeatScan.py -- AXIAL ping-pong heating at FIXED total steps and SHRINKING
excursion: does the heat come from the STEPS, or from being 60-80 um off the atomic plane?

THE QUESTION (2026-08-11, user)
  Figure 2(b)'s axial curve is survival over 80 steps as a function of per-step stroke, and the
  07-31 heating campaign showed the matching temperature rise.  Both were taken with the standard
  ping-pong: ONE out-and-back, ``nsteps = 40`` -> 80 steps, so the array walks 40 steps OUT before
  it turns around.  At 1.5-2 um/step that peak excursion is 60-80 um -- far outside the depth of
  field of every calibration the rig owns (imaging, the SLM->camera affine, the true-defocus
  piston null measured near focus), and far enough that the traps at the turnaround are not the
  traps the atoms were prepared in.  So the measured "per-step" heating may not be per-step at
  all: it may be a function of WHERE the array went.

  This scan separates the two by keeping the step count and the per-step stroke EXACTLY as they
  were and shrinking only the excursion: 80 steps of 1.5 um at 3.0 ms, but folded into short
  out-and-back cycles of ``g`` steps each, so the array never leaves +-``g * 1.5`` um.

      g = 5   ->  0 1 2 3 4 5 4 3 2 1 0 1 2 ... 0     8 cycles, peak excursion  7.5 um  (< 10)
      g = 40  ->  0 1 2 ... 40 ... 2 1 0              1 cycle,  peak excursion 60.0 um

  ``g = 40`` is not a separate run: it IS the standard ping-pong, frame for frame (the server's
  built-in triangle is ``0..n`` then ``n-1..0``, which is this schedule at ``g = n``).  So the two
  arms are a matched pair taken in one scan on one array in one imaging state, and the difference
  between them is the excursion and nothing else.

  READING IT.  T(g=5) == T(g=40) => the heating is genuinely PER-STEP and the excursion is
  irrelevant; short-excursion transport buys nothing but it is also not penalised, so the axial
  stroke budget from the 07-31 campaign transfers to any trajectory.  T(g=5) << T(g=40) => most
  of the measured axial heating is a LARGE-EXCURSION effect (out-of-focus traps / a mis-calibrated
  piston far from the plane), the per-step number is an overestimate for realistic moves, and
  fig 2(b)'s axial cliff is not the constraint it looks like.  T(g=5) > T(g=40) is the third
  outcome and is also informative: the price would then be the TURNAROUNDS (16 of them at g=5,
  1 at g=40), each a velocity reversal.

INVARIANTS (held by ``rearrange_callbacks._fold_ppg_pingpong_group``, deliberately)
  * 80 steps and 81 SLM frames per shot at EVERY g, including the s=0 control.  Writes cost
    survival on their own (a zero-motion 400-frame shot measured 0.9495 vs 0.9908 below 300
    frames, job 315), so a varying write count would confound the measurement.
  * per-step stroke unchanged -- every increment is +-1 at the same base amplitude.
  * the schedule ENDS at displacement 0, i.e. on the stored WGS phase byte-for-byte, so every
    atom is back in its source trap and LIVE img2 detection is valid.  (Unlike the one-way
    campaigns, this scan needs NO offline re-detection.)
  * what is NOT held fixed, and cannot be: the turnaround count, ``80/g``.  It is the arm's
    other degree of freedom, and it pushes T the OPPOSITE way from any excursion benefit --
    see "reading it" above.

STRUCTURE (seq ``RearrangeRnRHeatCommSeq``, unchanged from the 07-31 / 08-06 heating campaigns)
      img1 -> Cool556 5 ms (the prepared cold state) -> pingponggrating motion (80 steps)
           -> PostRearrCool with amps 0 (NO post-motion cooling, 0.5 ms dark hold)
           -> release t -> recapture -> img2
  Post-motion cooling is OFF so the release curve probes the POST-MOTION temperature;
  production's 5 ms cool would re-thermalize to ~8 uK and hide the whole effect.  The t=0 column
  is pure motion survival.  t=0.5 us is the RESTORE-MATCHED control: ``RearrangeRnRStep``
  short-circuits on exactly ``Time == 0`` (no AmpSLM toggle, no TTLSampleAndHold re-assert), so
  the t=0 shot never pays the trap restore every t>0 shot pays -- worth ~1% survival after many
  frames.  At 0.5 us the free flight is 19-38 nm against a ~0.6 um waist, so recapture is 1.000
  to far below our resolution.

AXIAL PARAMETRIZATION (matching the campaign that measured the axial cliff)
  scalar ``extras.step_size`` in MICRONS with ``true_defocus=True`` (exact spherical axial phase,
  not the paraxial ANSI Z4 parabola) and ``depth_piston_corr = 0.617`` rad/um, the measured
  true-defocus piston null (job 320, 2026-07-29).  ``z4 = loading_defocus = -4``, the plane those
  axial numbers were measured in.  NOTE the piston null was measured NEAR the plane, which is
  itself part of why the g=40 arm is the suspect one.

STICKY EXTRAS.  The server's ``setup_rearrangement`` extras persist across scans in one backend
session, so every knob that could be inherited from the radial / one-way campaigns (``depth``,
``true_defocus``, ``depth_piston_corr``, ``depth_fill_frac``, ``no_depth_piston``, ``piston``,
``hold_ms``, ``return_trip``) is set EXPLICITLY here.

MECHANISM.  The two-leg ping-pong machinery cannot express a multi-cycle triangle -- each leg is
uniform and the return leg always descends to 0 -- so this uses ``extras.disp_schedule`` (server
key added 2026-08-11 for the soft-start campaign): every frame sits at an integer multiple of one
base amplitude, so a multi-cycle triangle is just a non-monotone index list.  The list is built
PER SHOT by ``_fold_ppg_pingpong_group`` from the scalar ``extras.pingpong_group`` -- a
list-valued swept axis breaks the lab-side scan grid -- and echoed as ``ppg_pp_group`` /
``ppg_pp_nsteps`` / ``ppg_pp_cycles`` / ``ppg_pp_excursion``.

Run (pyctrl backend live)::

    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGPingpongGroupHeatScan.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGPingpongGroupHeatScan.py --force            # data
    python YbScans/RearrangeDiagnostics/PPGPingpongGroupHeatScan.py --step 0 --force   # control
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

DEFOCUS = -4.0                 # z4 == loading_defocus; the plane the axial campaign ran in
AXIAL_PISTON_CORR = 0.617      # rad of uniform phase per UM of axial step (true_defocus null)
POST_COOL_HOLD_MS = 0.5        # dark hold replacing the post-motion cool

STEP_UM = 1.5                  # per-step axial stroke (user-specified for this measurement)
PERIOD_MS = 3.0                # settle time per frame (the fig 2b slow arm)
TOTAL_STEPS = 80               # held fixed across the whole g axis
# 5 = the short-excursion move under test (7.5 um peak, < 10 as asked); 40 = the SAME 80 steps as
# one out-and-back, i.e. the standard ping-pong the 07-31 campaign and fig 2(b) measured.
GROUPS = [5, 40]

# See the docstring: 0 is the byte-clean held-in-traps baseline, 0.5 us the restore-matched one.
DEF_TIMES_US = [0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90]

# IMAGING: INHERIT THE ByPattern OVERLAY BY DEFAULT (both knobs None).
#
# The tempting alternative -- hardcoding a measured working set as g() overrides, as the soft-start
# submitter does -- is a trap for a scan that outlives one session: g() BEATS the overlay, so a
# stale constant silently overrides whatever the array was last calibrated to, and the 399 imaging
# detuning walks ~6 MHz/day.  On 2026-08-11 the two disagreed by 10 MHz (a 02:2x measurement said
# +6.0 while ByPattern held -4.0), and the day's own daily-calibration + imaging-optimization runs
# all ran on the OVERLAY at -4.0 with 5.2-5.7 ADU separation and 0.57-0.60 loading.  Inheriting is
# the honest default: it runs in the state the array was actually calibrated in.
#
# Pass --img-det / --img-pid to override deliberately (e.g. right after re-measuring, before the
# result is written into ByPattern).  DDS Imag399.Amp1/Amp2 are left alone in both cases: a legacy
# amp<1 attenuates AFTER the frozen servo point (gotcha-stale-dds-amps-pid-imaging).
IMAG_DETUNING_MHZ = None
IMG1_PIDSET = None
IMG2_PIDSET = None


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames are the SAME array: the schedule ends at displacement 0, so every atom is back
    in its source trap on the stored WGS phase.  Name = the phase-file BASENAME, which is what the
    detection registry and expConfig ByPattern are keyed by (an alias silently falls back to a
    day-folder grid)."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def schedule(group=None, n_total=TOTAL_STEPS, hold=0, trough=None, peak=10, soft=0):
    """The displacement-index list the per-shot fold will build -- built by calling the FOLD
    itself, so the dry run can never drift from what actually runs."""
    _bootstrap()
    import rearrange_callbacks as rc
    extras = {"step_size": 1.0, "pingpong_nsteps": int(n_total), "pingpong_hold": int(hold),
              "pingpong_soft": int(soft)}
    if trough is not None:
        extras["pingpong_trough_leg"] = int(trough)
        extras["pingpong_peak"] = int(peak)
    else:
        extras["pingpong_group"] = int(group)
    return rc._fold_ppg_pingpong_group({"extras": extras})["extras"]["disp_schedule"]


def _reversals(disp):
    """Interior local extrema of the schedule -- the velocity reversals."""
    return sum(1 for i in range(1, len(disp) - 1)
               if (disp[i] > disp[i - 1] and disp[i] > disp[i + 1])
               or (disp[i] < disp[i - 1] and disp[i] < disp[i + 1]))


def _turnarounds(group, n_total=TOTAL_STEPS):
    """Velocity reversals in the schedule: one per peak (``cycles``) plus one per INTERIOR
    return to 0 (``cycles - 1``). The endpoints are not reversals."""
    c = n_total // (2 * group)
    return 2 * c - 1


def build(step_um=STEP_UM, period_ms=PERIOD_MS, groups=None, n_total=TOTAL_STEPS,
          times_us=None, defocus=DEFOCUS, cool=False, holds=None, troughs=None, peak=10,
          softs=None, img_det=IMAG_DETUNING_MHZ, img_pid=(IMG1_PIDSET, IMG2_PIDSET)):
    """2-D [schedule-shape x ReleaseRecapture.Time] at fixed step / period / total steps.

    dim1 is exactly ONE of (mutually exclusive; checked):
      default        -> ``pingpong_group`` (the excursion / turnaround-count ladder).
      ``holds``      -> ``pingpong_hold`` at the fixed group ``groups[0]``: +h dwells h extra
        frames AT each turnaround, -h the frame-count-matched mid-leg control.
      ``troughs``    -> ``pingpong_trough_leg``: nonzero-trough oscillation 0->peak,
        k x (peak <-> peak-L), peak->0 -- reversal count 2k+1, ALL mid-flight, exactly one
        WGS-frame departure + one arrival at every L. Separates the direction-change penalty
        from anything tied to the 0/WGS frame (first-step / last-step). 81 frames at every L.
      ``softs``      -> ``pingpong_soft`` at the fixed group ``groups[0]``: +1 halves the
        approach speed into/out of every reversal (base amplitude halved, indices doubled),
        -1 the frame-count-matched control paying the same half steps mid-leg, 0 the plain
        triangle. Tests the velocity-kick mechanism."""
    _bootstrap()
    from scan_group import ScanGroup

    groups = [int(x) for x in (groups or GROUPS)]
    times = [float(t) * 1e-6 for t in (times_us or DEF_TIMES_US)]

    g = ScanGroup()
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
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

    # imaging: only override when explicitly asked. g() beats the ByPattern overlay, so setting
    # these pins the imaging state; leaving them unset inherits whatever the array is calibrated to.
    if img_det is not None:
        g().Imag399.FreqDetuning = float(img_det) * 1e6
    if img_pid is not None and img_pid[0] is not None:
        g().BlueMOT.Img1PIDSet = float(img_pid[0])
    if img_pid is not None and img_pid[1] is not None:
        g().BlueMOT.Img2PIDSet = float(img_pid[1])

    g().rearrange_kwargs.extras.n_rounds = 1
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)
    # PLACEHOLDER: the fold overwrites nsteps per shot with the cell's own max(disp) (what the
    # server's setup-time uint8 cache is indexed by).  Only the dequeue-time warmup
    # setup_rearrangement ever sees this value, so it is the LARGEST in the scan.
    if troughs is not None:
        rk.nsteps = int(peak)
    elif softs is not None and any(int(x) for x in softs):
        rk.nsteps = 2 * int(groups[0])
    else:
        rk.nsteps = int(max(groups))

    # AXIAL: scalar step_size in MICRONS under true_defocus.
    rk.extras.step_size = float(step_um)
    rk.extras.depth = True
    rk.extras.true_defocus = True
    rk.extras.depth_piston_corr = AXIAL_PISTON_CORR
    # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (legacy
    # beam-model path) and is sticky, which would pin the piston to a stale value.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True            # EXPLICIT (server default since 2026-07-15)
    rk.extras.piston = 0.0                      # no commanded per-step uniform phase
    rk.extras.hold_ms = 0.0                     # no turnaround dwell (job 424: own loss channel)
    rk.extras.return_trip = False               # the schedule owns the trajectory; also set by
                                                # the fold, stated here so the intent is recorded

    # dim 1: exactly one of the schedule-shape axes. THE SWITCH that activates the fold is
    # pingpong_nsteps; the pingpong_* keys are consumed lab-side and never reach the server
    # (disp_schedule does).
    if sum(x is not None for x in (holds, troughs, softs)) > 1:
        raise ValueError("holds / troughs / softs are mutually exclusive dim-1 axes")
    if troughs is not None:
        rk.extras.pingpong_peak = int(peak)
        rk.extras.pingpong_trough_leg.scan(1, [int(x) for x in troughs])
    elif holds is not None:
        rk.extras.pingpong_group = int(groups[0])
        rk.extras.pingpong_hold.scan(1, [int(h) for h in holds])
    elif softs is not None:
        rk.extras.pingpong_group = int(groups[0])
        rk.extras.pingpong_soft.scan(1, [int(x) for x in softs])
    else:
        rk.extras.pingpong_group.scan(1, groups)
    rk.extras.pingpong_nsteps = int(n_total)

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (production / axial campaign)
    rk.extras.z4 = float(defocus)
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # POST-MOTION cooling OFF unless --cool: amps 0 + a short dark hold, so the release probes the
    # post-motion temperature. cool=True leaves the group UNSET -> ByPattern Cool556 (5 ms).
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(POST_COOL_HOLD_MS) * 1e-3

    # dim 2: the release-recapture thermometer.
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0               # set for faithfulness; UNREAD by the step

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


def _desc_hold(step_um, period_ms, group, holds, n_total, times_us, defocus, cool,
               img_det=None, img_pid=None):
    nt = _turnarounds(group, n_total)
    return (
        "AXIAL PING-PONG: does DWELLING AT THE TURNAROUND buy back the reversal cost? "
        "Fixed g=%d (%d out-and-back cycles of %g um = %g um excursion, %d velocity reversals), "
        "fixed %d steps at %g um/step, step_period_ms=%g. dim1 = extras.pingpong_hold h = %s "
        "EXTRA FRAMES per turnaround (frame count 81 + %d*|h|, i.e. %s); dim2 = release time "
        "%s us. "
        "WHY: the 2026-08-11 g ladder (jobs 823-825) found the cost of a short-excursion axial "
        "ping-pong sits at the VELOCITY REVERSALS, not at the excursion -- survival over 80 steps "
        "of 3 um fell 0.923 -> 0.736 going from 1 turnaround (120 um excursion) to 15 (15 um), "
        "fitting -lnS = 0.074%%/step*80 + 1.69%% per turnaround (R2 0.968), i.e. ONE TURNAROUND "
        "COSTS ~23 STEPS. The 120 um arm being the BEST rules out out-of-focus traps as the "
        "mechanism. The leading candidate is that at a reversal the atom's residual drift "
        "velocity -- which has been following the trap -- is suddenly OPPOSED to the new "
        "direction of travel, so the relative velocity and the energy kick are larger than at any "
        "other hop. That predicts a DWELL should help: pausing lets the atom dephase before the "
        "trap turns. If instead the cost is intrinsic to the reversal geometry, dwelling buys "
        "nothing. "
        "THE SIGNED CONTROL (this is what makes it a measurement): a dwell ADDS FRAMES, and SLM "
        "writes cost survival on their own (a zero-motion 400-frame shot measured 0.9495 vs "
        "0.9908 below 300 frames, job 315), as does the extra in-trap time -- so +h alone is "
        "confounded with its own frame count. h<0 inserts EXACTLY the same number of repeated "
        "frames at interior NON-extremal positions instead (mid-leg pauses), so the +h / -h pair "
        "is matched in frame count, step count, total travel and step-size multiset and differs "
        "ONLY in WHERE the dwell sits: S(+h)/S(-h) is the turnaround effect with no model in "
        "between. Same construction as the +m/-m pair in _fold_ppg_soft_start. "
        "MECHANISM OF EXPRESSION: extras.disp_schedule (server key added 2026-08-11) -- a dwell "
        "is just a REPEATED displacement index, i.e. a frame that rewrites the same phase. Built "
        "per shot by rearrange_callbacks._fold_ppg_pingpong_group / _apply_pingpong_hold from the "
        "scalar extras.pingpong_hold (a list-valued swept axis breaks the lab-side scan grid), "
        "echoed as ppg_pp_hold / ppg_pp_frames / ppg_pp_group / ppg_pp_cycles. Every schedule "
        "still ends at displacement 0 (the stored WGS phase byte-for-byte) and still takes "
        "exactly %d steps, so LIVE img2 detection is valid and no offline re-detection is needed. "
        "SEQ RearrangeRnRHeatCommSeq: img1 -> Cool556 5 ms -> motion -> %s -> release -> img2; "
        "t=0 is pure motion survival, t=0.5 us the restore-matched reference. "
        "AXIAL MAP: true_defocus=True (step in um, exact spherical), depth_piston_corr=%g rad/um "
        "(job 320 null), depth_fill_frac cleared, hold_ms=0 (the SERVER-side turnaround dwell is "
        "deliberately NOT used -- this scan expresses the dwell in the schedule so it is "
        "frame-counted and controllable). IMAGING: %s Array %s, z4 = loading_defocus = %g."
        % (group, n_total // (2 * group), step_um, group * step_um, nt, n_total, step_um,
           period_ms, ",".join("%+d" % h for h in holds), nt,
           "/".join(str(81 + nt * abs(h)) for h in holds),
           ",".join("%g" % t for t in times_us), n_total,
           ("post-motion cooling ON (hides the effect)" if cool else
            "NO post-motion cooling (PostRearrCool amps 0 + %g ms dark hold)" % POST_COOL_HOLD_MS),
           AXIAL_PISTON_CORR, _img_note(img_det, img_pid), PATTERN, defocus))


def _desc_trough(step_um, period_ms, troughs, peak, n_total, times_us, defocus, cool,
                 img_det=None, img_pid=None):
    revs = [_reversals(schedule(trough=L, peak=peak, n_total=n_total)) for L in troughs]
    return (
        "AXIAL PING-PONG TURNAROUND MECHANISM: are the reversals expensive PER SE, or only the "
        "ones that visit the 0/WGS frame (first step off rest / last step back into it)? "
        "dim1 = extras.pingpong_trough_leg L = %s: schedule 0->%d, then k oscillations "
        "%d<->%d-L, then %d->0, at fixed %d total steps -> reversal count 2k+1 = %s, ALL of "
        "them MID-FLIGHT at nonzero displacement, with exactly ONE departure from and ONE "
        "arrival at the pristine-WGS frame at every L. dim2 = release time %s us. "
        "step_size=%g um/step, step_period_ms=%g, %d SLM frames per shot at every L (81-frame "
        "invariant kept: SLM writes cost survival on their own, job 315). "
        "WHY: the 2026-08-11 g ladder (jobs 823-825, recreated 2026-08-19) found "
        "-lnS = 0.074%%/step*80 + 1.69%% per turnaround, but that axis is DEGENERATE -- per "
        "cycle the departures from 0, arrivals at 0, peak reversals and WGS-frame visits all "
        "scale together. This schedule holds the WGS-frame visits fixed at 1+1 and sweeps only "
        "the mid-flight reversal count: a per-reversal slope matching the ladder's 1.69%% kills "
        "the first-step/last-step/WGS-frame hypotheses (penalty = direction change per se); a "
        "vanishing slope localizes the penalty at the zero-frame events. Turnaround-dwell scans "
        "(h=+1/+2/+4 vs mid-leg control, 2026-08-11 17:15) already showed DWELLING does not "
        "heal it. "
        "MECHANISM: extras.disp_schedule via rearrange_callbacks._fold_ppg_pingpong_group "
        "(scalar extras.pingpong_trough_leg + pingpong_peak=%d; echoed as ppg_pp_trough_leg / "
        "ppg_pp_peak / ppg_pp_reversals / ppg_pp_cycles). Every schedule ends at displacement 0 "
        "so LIVE img2 detection is valid. "
        "SEQ RearrangeRnRHeatCommSeq: img1 -> Cool556 5 ms -> motion -> %s -> release -> img2. "
        "AXIAL MAP: true_defocus=True, depth_piston_corr=%g rad/um (job 320 null), "
        "depth_fill_frac cleared, hold_ms=0. IMAGING: %s Array %s, z4 = loading_defocus = %g."
        % (",".join(str(x) for x in troughs), peak, peak, peak, peak, n_total,
           ",".join(str(r) for r in revs), ",".join("%g" % t for t in times_us),
           step_um, period_ms, n_total + 1, peak,
           ("post-motion cooling ON" if cool else
            "NO post-motion cooling (PostRearrCool amps 0 + %g ms dark hold)"
            % POST_COOL_HOLD_MS),
           AXIAL_PISTON_CORR, _img_note(img_det, img_pid), PATTERN, defocus))


def _desc_soft(step_um, period_ms, group, softs, n_total, times_us, defocus, cool,
               img_det=None, img_pid=None):
    nt = _turnarounds(group, n_total)
    return (
        "AXIAL PING-PONG SOFT TURNAROUND: does HALVING THE APPROACH SPEED into/out of every "
        "velocity reversal buy back the ~1.69%%/turnaround cost? Fixed g=%d (%d reversals), "
        "%d total steps at %g um/step, step_period_ms=%g. dim1 = extras.pingpong_soft = %s: "
        "+1 halves the base amplitude and doubles the indices so the two steps flanking every "
        "reversal are HALF steps (approach speed into and out of each extremum halved; +2 "
        "frames per reversal); -1 the frame-count-, travel- and step-multiset-MATCHED control "
        "paying the same half steps spread mid-leg; 0 the plain triangle (81 frames). +1/-1 "
        "both %d frames. dim2 = release time %s us. "
        "WHY: the leading mechanism for the per-reversal cost (g ladder jobs 823-825, "
        "recreated 2026-08-19; dwell scan 08-11 17:15 showed pausing does NOT heal it) is a "
        "velocity kick -- the atom's residual drift velocity, following the trap, suddenly "
        "opposed at the reversal, predicting the cost scales with approach speed. If S(+1) >> "
        "S(-1) the kick picture holds and half-speed turnarounds are the practical mitigation; "
        "if S(+1) == S(-1) the cost is intrinsic to the reversal geometry (e.g. the LC "
        "transition), not the kinematics. "
        "MECHANISM: extras.disp_schedule via rearrange_callbacks._fold_ppg_pingpong_group "
        "(scalar extras.pingpong_soft; echoed as ppg_pp_soft / ppg_pp_reversals / "
        "ppg_pp_frames; step_size rewritten to the halved base for +-1). Every schedule ends "
        "at 0 so LIVE img2 detection is valid. "
        "SEQ RearrangeRnRHeatCommSeq: img1 -> Cool556 5 ms -> motion -> %s -> release -> img2. "
        "AXIAL MAP: true_defocus=True, depth_piston_corr=%g rad/um (job 320 null), "
        "depth_fill_frac cleared, hold_ms=0. IMAGING: %s Array %s, z4 = loading_defocus = %g."
        % (group, nt, n_total, step_um, period_ms,
           ",".join("%+d" % s for s in softs), n_total + 1 + 2 * nt,
           ",".join("%g" % t for t in times_us),
           ("post-motion cooling ON" if cool else
            "NO post-motion cooling (PostRearrCool amps 0 + %g ms dark hold)"
            % POST_COOL_HOLD_MS),
           AXIAL_PISTON_CORR, _img_note(img_det, img_pid), PATTERN, defocus))


def _desc(step_um, period_ms, groups, n_total, times_us, defocus, cool,
          img_det=None, img_pid=None):
    what = ("ZERO-STROKE CONTROL (step_size=0: all %d frames still written, same dark time in "
            "trap, no motion)" % (n_total + 1) if float(step_um) == 0.0 else
            "step_size=%g um/step" % step_um)
    return (
        "AXIAL SHORT-EXCURSION PING-PONG heating: atom TEMPERATURE by release-and-recapture at "
        "FIXED total steps (%d) and fixed per-step stroke, sweeping only the EXCURSION. "
        "dim1 = extras.pingpong_group g = %s steps per out-and-back cycle -> peak excursion "
        "g*%g = %s um; dim2 = release time %s us. %s, step_period_ms=%g, %d steps -> %d SLM "
        "frames per shot at EVERY g. "
        "THE QUESTION: fig 2(b) and the 07-31 heating campaign both used the standard ping-pong "
        "(ONE out-and-back, nsteps=40 -> 80 steps), which walks the array 60-80 um off the "
        "atomic plane before it turns around -- outside the depth of field of every calibration "
        "the rig owns (imaging, the SLM->camera affine, the true-defocus piston null measured "
        "near the plane). So the measured 'per-step' axial heating may be a function of WHERE the "
        "array went rather than of the steps. This scan holds the step count and the stroke fixed "
        "and folds the same 80 steps into %s short cycles, so the array never leaves +-g*%g um. "
        "g=%d IS the standard ping-pong frame for frame (the server's built-in triangle is 0..n "
        "then n-1..0 = this schedule at g=n), so the arms are a matched pair on one array in one "
        "imaging state and the ONLY difference is the excursion. T(g=5)==T(g=40) => the heating "
        "is genuinely per-step; T(g=5)<<T(g=40) => most of the axial heating is a large-excursion "
        "effect and the per-step number overestimates realistic moves; T(g=5)>T(g=40) => the "
        "price is the TURNAROUNDS (velocity reversals: %s at g=%s respectively), which is the "
        "one thing this axis cannot hold fixed. "
        "SEQ RearrangeRnRHeatCommSeq: img1 -> Cool556 5 ms (prepared cold state) -> "
        "pingponggrating motion -> %s -> release -> img2. t=0 column = pure motion survival "
        "(RearrangeRnRStep adds nothing at t=0); t=0.5 us is the RESTORE-MATCHED control (the "
        "t=0 shot skips the AmpSLM toggle / TTLSampleAndHold re-assert that every t>0 shot pays, "
        "~1%% of survival after many frames, while 0.5 us of free flight is 19-38 nm against a "
        "~0.6 um waist). "
        "MECHANISM: the two-leg ping-pong knobs cannot express a multi-cycle triangle (each leg "
        "is uniform and the return leg always descends to 0), so this uses extras.disp_schedule "
        "(server key added 2026-08-11): every frame sits at an integer multiple of one base "
        "amplitude, so a multi-cycle triangle is a non-monotone index list. Built PER SHOT by "
        "rearrange_callbacks._fold_ppg_pingpong_group from the scalar extras.pingpong_group (a "
        "list-valued swept axis breaks the lab-side scan grid) and echoed as ppg_pp_group / "
        "ppg_pp_nsteps / ppg_pp_cycles / ppg_pp_excursion. Every schedule ENDS at displacement 0 "
        "-- the stored WGS phase byte-for-byte -- so every atom is back in its source trap and "
        "LIVE img2 detection is valid; no offline re-detection (unlike the one-way campaigns). "
        "AXIAL MAP: true_defocus=True (step in um, exact spherical, not the paraxial Z4 "
        "parabola), depth_piston_corr=%g rad/um (job 320 null, 2026-07-29), depth_fill_frac "
        "cleared (it would override the corr). hold_ms=0 (job 424: the turnaround dwell is its "
        "own nsteps-independent loss channel). All depth/lateral extras set EXPLICITLY -- server "
        "extras are sticky across scans. "
        "IMAGING: %s "
        "Array %s, z4 = loading_defocus = %g."
        % (n_total, ",".join(str(x) for x in groups), step_um,
           ",".join("%g" % (x * step_um) for x in groups),
           ",".join("%g" % t for t in times_us), what, period_ms, n_total, n_total + 1,
           "/".join(str(n_total // (2 * x)) for x in groups), step_um, max(groups),
           ",".join(str(_turnarounds(x, n_total)) for x in groups),
           ",".join(str(x) for x in groups),
           ("post-motion cooling ON (ByPattern Cool556, production 5 ms) -- NOTE this "
            "re-thermalizes to ~8 uK and hides the effect" if cool else
            "NO post-motion cooling (PostRearrCool amps 0 + %g ms dark hold) so the release "
            "probes the POST-MOTION temperature" % POST_COOL_HOLD_MS),
           AXIAL_PISTON_CORR, _img_note(img_det, img_pid), PATTERN, defocus))


def _description(args, groups, holds, troughs, softs, times):
    """Pick the description that matches the axis actually being swept."""
    pid = tuple(args.img_pid) if args.img_pid else None
    if troughs:
        return _desc_trough(args.step, args.period, troughs, args.peak, args.nsteps, times,
                            args.defocus, args.cool, args.img_det, pid)
    if softs:
        return _desc_soft(args.step, args.period, groups[0], softs, args.nsteps, times,
                          args.defocus, args.cool, args.img_det, pid)
    if holds:
        return _desc_hold(args.step, args.period, groups[0], holds, args.nsteps, times,
                          args.defocus, args.cool, args.img_det, pid)
    return _desc(args.step, args.period, groups, args.nsteps, times, args.defocus, args.cool,
                 args.img_det, pid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=9,
                    help="passes per cell; 9 is where the binomial SEM meets the 0.6%% "
                         "common-mode imaging systematic (default %(default)s)")
    ap.add_argument("--step", type=float, default=STEP_UM,
                    help="per-step axial stroke in um (0 = the zero-stroke control)")
    ap.add_argument("--period", type=float, default=PERIOD_MS, help="step_period_ms")
    ap.add_argument("--groups", default=None,
                    help="comma-separated steps per out-and-back cycle (default %s)"
                         % ",".join(str(x) for x in GROUPS))
    ap.add_argument("--holds", default=None,
                    help="comma-separated turnaround DWELLS in extra frames, at the fixed group "
                         "--groups[0]. +h = dwell h frames AT each turnaround; -h = the "
                         "frame-count-matched CONTROL that pays the same extra frames mid-leg. "
                         "Given, this REPLACES the group axis on dim1.")
    ap.add_argument("--troughs", default=None,
                    help="comma-separated NONZERO-TROUGH oscillation legs L: schedule 0->peak, "
                         "k x (peak <-> peak-L), peak->0, reversal count 2k+1 all mid-flight. "
                         "Given, this REPLACES the group axis on dim1.")
    ap.add_argument("--peak", type=int, default=10,
                    help="peak displacement for --troughs (default %(default)s)")
    ap.add_argument("--softs", default=None,
                    help="comma-separated pingpong_soft values at the fixed group --groups[0]: "
                         "+1 = half-speed approach into/out of every reversal, -1 = the "
                         "frame-count-matched mid-leg control, 0 = plain triangle. Given, this "
                         "REPLACES the group axis on dim1.")
    ap.add_argument("--nsteps", type=int, default=TOTAL_STEPS,
                    help="TOTAL steps per shot, held fixed across the g axis (default %(default)s)")
    ap.add_argument("--times", default=None,
                    help="comma-separated release times in US (default %s)"
                         % ",".join(str(t) for t in DEF_TIMES_US))
    ap.add_argument("--defocus", type=float, default=DEFOCUS)
    ap.add_argument("--cool", action="store_true",
                    help="restore production post-motion cooling (hides the effect; diagnostics)")
    ap.add_argument("--img-det", type=float, default=IMAG_DETUNING_MHZ,
                    help="399 imaging FreqDetuning (MHz) as a g() override. DEFAULT: unset -> "
                         "inherit ByPattern, i.e. image in the state the array was calibrated "
                         "in. Pass this only to pin a freshly re-measured value.")
    ap.add_argument("--img-pid", type=float, nargs=2, metavar=("IMG1", "IMG2"),
                    default=None,
                    help="BlueMOT.Img1/Img2PIDSet g() override. DEFAULT: unset -> inherit "
                         "ByPattern.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    groups = ([int(x) for x in args.groups.split(",") if x.strip()] if args.groups
              else list(GROUPS))
    times = ([float(x) for x in args.times.split(",") if x.strip()] if args.times
             else list(DEF_TIMES_US))
    troughs = ([int(x) for x in args.troughs.split(",") if x.strip()] if args.troughs else None)
    softs = ([int(x) for x in args.softs.split(",") if x.strip()] if args.softs else None)
    holds = ([int(x) for x in args.holds.split(",") if x.strip()] if args.holds else None)
    if troughs is None:
        bad = [x for x in groups if x < 1 or args.nsteps % (2 * x)]
        if bad:
            raise SystemExit("groups %s do not divide %d steps into whole out-and-back cycles "
                             "(need nsteps %% (2g) == 0, else the move ends displaced)"
                             % (bad, args.nsteps))

    seq_name, g = build(step_um=args.step, period_ms=args.period, groups=groups,
                        n_total=args.nsteps, times_us=times, defocus=args.defocus,
                        cool=args.cool, holds=holds, troughs=troughs, peak=args.peak,
                        softs=softs, img_det=args.img_det,
                        img_pid=tuple(args.img_pid) if args.img_pid else None)
    n = g.nseq()
    print("  imaging   = %s" % _img_note(args.img_det,
                                         tuple(args.img_pid) if args.img_pid else None))
    label = args.label or (
        "PPGAxialPingpongTrough_s%gum_p%gms_pk%d" % (args.step, args.period, args.peak)
        if troughs else
        "PPGAxialPingpongSoft_s%gum_p%gms_g%d" % (args.step, args.period, groups[0])
        if softs else
        "PPGAxialPingpongHold_s%gum_p%gms_g%d" % (args.step, args.period, groups[0])
        if holds else
        "PPGAxialPingpongGroup_s%gum_p%gms%s" % (args.step, args.period,
                                                 "_CTRL" if args.step == 0 else ""))

    axis = ("troughs" if troughs else "softs" if softs else "holds" if holds else "groups")
    dim1 = troughs or softs or holds or groups
    print("seq=%s  nseq=%d  (%d %s x %d release times)  step=%g um  period=%g ms  z4=%g"
          % (seq_name, n, len(dim1), axis, len(times), args.step, args.period, args.defocus))
    scheds = []
    if troughs:
        for L in troughs:
            d = schedule(trough=L, peak=args.peak, n_total=args.nsteps)
            scheds.append(d)
            print("  L=%-2d  frames=%3d steps=%3d max_disp=%-3d reversals=%-3d (all nonzero: %s)"
                  "  disp[:16]=%s"
                  % (L, len(d), len(d) - 1, max(d), _reversals(d),
                     all(d[i] for i in range(1, len(d) - 1)
                         if (d[i] > d[i-1] and d[i] > d[i+1]) or
                            (d[i] < d[i-1] and d[i] < d[i+1])), d[:16]))
    elif softs:
        x = groups[0]
        for s in softs:
            d = schedule(x, args.nsteps, soft=s)
            scheds.append(d)
            print("  soft=%+d  frames=%3d steps=%3d max_disp=%-3d reversals=%-3d  %s  "
                  "disp[:16]=%s"
                  % (s, len(d), len(d) - 1, max(d), _reversals(d),
                     ("HALF-SPEED turnarounds" if s > 0 else
                      "CONTROL: half steps MID-LEG" if s < 0 else "plain triangle"), d[:16]))
    elif holds:
        x = groups[0]
        for h in holds:
            d = schedule(x, args.nsteps, hold=h)
            scheds.append(d)
            nt = _turnarounds(x, args.nsteps)
            print("  hold=%+d  frames=%3d (81 + %d*%d)  steps=%3d  %s  disp[:16]=%s"
                  % (h, len(d), nt, abs(h), args.nsteps,
                     ("dwell AT the %d turnarounds" % nt) if h > 0 else
                     ("CONTROL: same frames MID-LEG" if h < 0 else "no dwell"), d[:16]))
    else:
        for x in groups:
            d = schedule(x, args.nsteps)
            scheds.append(d)
            print("  g=%-3d cycles=%-2d frames=%3d steps=%3d max_disp=%-3d excursion=%6.2f um  "
                  "turnarounds=%-3d disp[:12]=%s"
                  % (x, args.nsteps // (2 * x), len(d), len(d) - 1, max(d), x * args.step,
                     _turnarounds(x, args.nsteps), d[:12]))
    fr = max(len(d) for d in scheds)
    print("  shots = %d x %d reps = %d   (motion up to %.0f ms/shot; ~%.0f min at 2.5 s/shot)"
          % (n, args.reps, n * args.reps, fr * args.period, n * args.reps * 2.5 / 60.0))

    if args.dry_run:
        s0 = g.getseq(0)["rearrange_kwargs"]
        print("  cell0 extras: %s"
              % {k: s0["extras"].get(k) for k in
                 ("step_size", "pingpong_group", "pingpong_trough_leg", "pingpong_peak",
                  "pingpong_soft", "pingpong_nsteps", "depth", "true_defocus",
                  "depth_piston_corr", "return_trip", "hold_ms", "piston", "z4")})
        print("\nDESCRIPTION:\n%s" % _description(args, groups, holds, troughs, softs, times))
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force (use --dry-run to inspect)")

    _bootstrap()
    from yb_start_scan import ybStartScan
    did = ybStartScan(seq_name, g, url=args.url, label=label,
                      description=_description(args, groups, holds, troughs, softs, times),
                      rep=args.reps)
    print("submitted %s -> descriptor id %s (%d pts)" % (label, did, n))


if __name__ == "__main__":
    main()
