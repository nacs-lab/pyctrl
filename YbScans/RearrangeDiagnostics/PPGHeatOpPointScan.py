"""PPGHeatOpPointScan.py -- atom TEMPERATURE vs nsteps at chosen pingponggrating operating points.

THE QUESTION (2026-07-31 campaign)
  Working hypothesis from the step-size sweeps: below a threshold per-step stroke there is NO
  heating, above it there IS heating and the heating is the loss mechanism.  The step-size scans
  only ever measured SURVIVAL, which cannot separate "the atom was heated and boiled out" from
  "the atom was lost by some other per-step channel".  This scan measures the TEMPERATURE
  directly, by release-and-recapture (R&R) thermometry, at 8 operating points:

      period {0.696, 3.0} ms  x  axis {radial, axial}  x  step {just below, just above} the
      measured 99%-per-step cliff,

  each as a curve in nsteps {5, 10, 20, 30, 40}.  A below-cliff point whose T is flat in nsteps
  and an above-cliff point whose T grows with nsteps is the hypothesis confirmed; a below-cliff
  point that heats, or an above-cliff point that does not, refutes it.

STRUCTURE (seq ``RearrangeRnRHeatCommSeq``, unchanged from the 2026-07-22 heating campaign)
      img1 -> Cool556 5 ms (the prepared cold state) -> pingponggrating MOTION
           -> PostRearrCool with amps 0 (i.e. NO post-motion cooling, 0.5 ms dark hold)
           -> release t -> recapture -> img2
  The release curve at each nsteps is the thermometer for the post-motion temperature; the t=0
  column is the pure motion survival (``RearrangeRnRStep`` adds NOTHING at t=0, so it is a
  byte-clean "held in traps" baseline).  Production's 5 ms post-motion cool would re-thermalize
  everything to ~8 uK and hide the effect -- that is why it is off (``--cool`` restores it).

RELEASE TIMES ARE LONGER THAN THE 07-22 CAMPAIGN.  That campaign used 0-40 us, which for a
~8-10 uK prepared state only reaches S ~ 0.4 and leaves the cold end of the curve poorly
constrained.  Default here is 0-90 us (9 points), which takes a cold curve down to S ~ 0.02 and
still keeps 3-4 points on the falling edge of a hot (~40 uK) one.

TWO AXIS PARAMETRIZATIONS -- deliberately different, each matching the campaign that measured
its cliff:
  * ``--axis radial``: lateral (blaze) motion.  ``extras.step_x`` in knm PIXELS, folded to the
    xyz ``step_size`` 3-vector [s, 0, 0] per shot by ``rearrange_callbacks._fold_step_xyz``
    (a list-valued swept axis breaks the lab-side scan grid).  ``depth`` OFF.  +x only: the
    07-22 campaign found +-x symmetric (and +-y NOT -- avoid y here).
  * ``--axis axial``: depth motion.  scalar ``extras.step_size`` in MICRONS with
    ``true_defocus=True`` (exact spherical axial phase, not the paraxial ANSI Z4 parabola) and
    ``depth_piston_corr = 0.617`` rad/um -- the measured true-defocus piston null (job 320,
    2026-07-29).  +z only (job 326: the two directions agree to <6% in critical distance, and
    a round trip traverses both step directions anyway -- the sign only picks which side of
    focus the excursion goes to).

STICKY EXTRAS.  The server's ``setup_rearrangement`` extras are STICKY across scans in one
backend session, so this script sets EVERY knob that differs between the two arms explicitly
(``depth``, ``true_defocus``, ``depth_piston_corr``, ``depth_fill_frac``, ``hold_ms``,
``piston``, ``z4``) in BOTH arms.  Without that, running the axial arm after the radial one (or
vice versa) silently inherits the other's depth mode.

``hold_ms = 0`` ON PURPOSE.  Job 424 showed the nsteps-INDEPENDENT loss channel is entirely the
turnaround dwell, so a dwell would add a fixed loss (and a fixed cooling opportunity) that has
nothing to do with per-step heating.

ARRAY / FOCAL PLANE: 33x33_feedback11 with ``z4 = loading_defocus = -4`` and ``ifEnhanced=True``
(BlueLAC loading) -- the configuration the axial critical-distance numbers (jobs 320-327,
2026-07-29/30) were measured in, and current production.  NOTE this differs from the 07-22
radial campaign (which ran at -5 with plain LAC), which is exactly why ``--mode cliff`` re-finds
both cliffs here rather than reusing 07-22's numbers.

MODES
  ``--mode cliff``  2-D [step magnitude x step_period_ms] at fixed nsteps, release time pinned
                    at 0.  Deliverable: per-step survival S**(1/(2n)) -> the step size where the
                    per-step loss is 1% (the "99% mark"), per period.  Same seq and same config
                    as the heat scans, so the cliff measured IS the cliff of the heat scans.
  ``--mode heat``   2-D [nsteps x ReleaseRecapture.Time] at one fixed (axis, step, period).
                    The temperature-vs-nsteps curve for one operating point.  ``--step 0`` is
                    the no-motion control (frames still written, same dark time in trap).

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGHeatOpPointScan.py --mode cliff --axis radial \
        --steps 0,0.6,0.8,1.0,1.1,1.2,1.3,1.4,1.6,1.8 --nsteps 40 --reps 8 --dry-run
    python YbScans/RearrangeDiagnostics/PPGHeatOpPointScan.py --mode heat --axis axial \
        --step 3.2 --period 0.696 --reps 6 --force
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

DEFOCUS = -4.0                 # z4 == loading_defocus (production; jobs 320-327 context)
AXIAL_PISTON_CORR = 0.617      # rad of uniform phase per UM of axial step (true_defocus null)
HOLD_MS = 0.0                  # NO turnaround dwell (job 424: the dwell is its own loss channel)
POST_COOL_HOLD_MS = 0.5        # dark hold replacing the post-motion cool

DEF_NSTEPS_LIST = [5, 10, 20, 30, 40]
# 0.5 us is the RESTORE-MATCHED CONTROL, not a physics point. RearrangeRnRStep short-circuits
# only on exactly Time == 0, so the t=0 shot skips the whole R&R block -- no AmpSLM toggle, no
# 3 us AOM settle, no TTLSampleAndHold re-assert -- while every t>0 shot gets the trap explicitly
# restored. After many SLM writes that restore matters: S(6 us) was measured ABOVE S(0) by
# 0.008-0.012 (~5 sigma) at nsteps 30-40 even at 0.97 survival, which is unphysical for a release
# curve and was corrupting those cells. At 0.5 us the free flight is only 19-38 nm against a
# ~0.6 um waist (v_rms = 3.8 cm/s at 10 uK, 7.6 at 40 uK), so recapture is 1.000 to far below our
# resolution -- it is the t=0 baseline WITH the restore. Keeping both measures the restore cost
# directly as S(0.5)/S(0) - 1.
DEF_TIMES_US = [0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90]
DEF_CLIFF_PERIODS = [0.696, 3.0]
DEF_CLIFF_NSTEPS = 40


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames are the SAME array (the ping-pong returns every atom to its source site).
    Name = the phase-file BASENAME, which is what the detection registry + expConfig ByPattern
    are keyed by (an alias silently falls back to a day-folder grid)."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _common(g, axis, cool):
    """Everything that is identical in every mode / arm. Returns the rearrange_kwargs handle."""
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

    g().rearrange_kwargs.extras.n_rounds = 1

    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (production / axial campaign)
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN
    rk.extras.no_depth_piston = True            # EXPLICIT (server default since 2026-07-15)
    rk.extras.piston = 0.0                      # no commanded per-step uniform phase
    rk.extras.hold_ms = HOLD_MS

    # ---- depth mode: set in BOTH arms because the server extras are STICKY -----------------
    if axis == "axial":
        rk.extras.depth = True
        rk.extras.true_defocus = True           # step_size is MICRONS under this flag
        rk.extras.depth_piston_corr = AXIAL_PISTON_CORR
        # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (legacy
        # beam-model path) and is sticky, which would pin the piston to a stale value.
        rk.extras.depth_fill_frac = None
    else:
        rk.extras.depth = False
        rk.extras.true_defocus = False
        rk.extras.depth_piston_corr = 0.0
        rk.extras.depth_fill_frac = None

    # ---- POST-MOTION cooling: OFF unless --cool ---------------------------------------------
    # amps 0 => no cooling between motion and release; short dark hold. cool=True leaves the
    # group UNSET -> falls back to ByPattern[pattern].Cool556 (production 5 ms), byte-identical
    # to RearrangeRnRCommSeq.
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(POST_COOL_HOLD_MS) * 1e-3

    rp.NumPerGroup = 100000                     # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return rk


def _set_step(rk, axis, value, swept_dim=None):
    """Program the step magnitude, either fixed (swept_dim None) or as a swept axis.

    radial -> extras.step_x (knm px) + step_y/step_z pinned 0 (folded to [x,0,0] per shot).
    axial  -> scalar extras.step_size (um, true_defocus).
    """
    if axis == "axial":
        if swept_dim is None:
            rk.extras.step_size = float(value)
        else:
            rk.extras.step_size.scan(swept_dim, [float(v) for v in value])
    else:
        if swept_dim is None:
            rk.extras.step_x = float(value)
        else:
            rk.extras.step_x.scan(swept_dim, [float(v) for v in value])
        rk.extras.step_y = 0.0
        rk.extras.step_z = 0.0


def build_cliff(axis, steps, periods=None, nsteps=DEF_CLIFF_NSTEPS, cool=False):
    """2-D [step magnitude x step_period_ms] at release time 0. Returns (seq_name, g)."""
    _bootstrap()
    from scan_group import ScanGroup

    periods = list(periods or DEF_CLIFF_PERIODS)
    g = ScanGroup()
    rk = _common(g, axis, cool)
    _set_step(rk, axis, [float(s) for s in steps], swept_dim=1)
    rk.step_period_ms.scan(2, [float(p) for p in periods])
    rk.nsteps = int(nsteps)
    # release time pinned at 0 -> RearrangeRnRStep adds nothing (pure motion survival).
    g().ReleaseRecapture.Time = 0.0
    g().ReleaseRecapture.Hold = 0
    return "RearrangeRnRHeatCommSeq", g


COOL556_DET_X = 0.135e6        # expConfig Cool556.X.FreqDetuning -- the COOLING setting
COOL556_DET_H = 0.130e6        # expConfig Cool556.h.FreqDetuning
PRECOOL_HEAT_TIME_MS = 2.0     # fixed pulse length when the detuning is the swept axis
# Deltas added to BOTH beams' detuning. 0 = nominal cooling (coldest); -0.135e6 puts X exactly on
# resonance; more negative crosses to the other side of the line and the same pulse ANTI-cools.
# 556 is narrow-line (Gamma ~ 182 kHz) so the nominal 0.135 MHz is already sub-linewidth --
# a few tens of kHz moves the atom a long way across the Doppler profile.
DEF_PRECOOL_DET_DELTAS = [0.0, -0.06e6, -0.105e6, -0.135e6, -0.175e6, -0.215e6]


def build_precool(axis, step, period, precool_ms=None, times_us=None, nsteps=20, cool=False,
                  det_deltas=None, heat_time_ms=PRECOOL_HEAT_TIME_MS):
    """2-D [PreMotionCool.Time x ReleaseRecapture.Time] at one operating point.

    With ``det_deltas`` the dim-1 axis becomes PreMotionCool DETUNING instead of Time, at a fixed
    ``heat_time_ms`` pulse. Time bottoms out at 0 (no cooling), which only reaches the ~17.5 uK
    the atoms already have after imaging -- there is no "less than no cooling". Detuning has no
    such floor: walking the same pulse onto and across resonance turns it into a calibrated
    heater, so the initial-temperature lever arm can be extended well above 17.5 uK without
    touching img1 or changing the sequence length.

    THE DISCRIMINATOR between the two temperature brackets. The release-recapture fit cannot say
    on its own whether the motion loss is energy-SELECTIVE (evaporation over a barrier lowered
    during the step -> survivors are a truncated distribution and the parent is up to ~2x hotter
    than the thermal fit says) or energy-BLIND (an impulsive mechanical ejection -> survivors are
    still thermal and the plain fit is right). Those two make OPPOSITE predictions for how the
    motion loss responds to the INITIAL temperature:

        evaporative   -> S(t0) falls steeply as the pre-motion cooling is spoiled
        impulsive     -> S(t0) barely moves

    Spoiling the cooling is what ``PreMotionCool`` is for: bseq1 applies Cool556 twice (before
    img1 and pre-motion), and only the second one may be touched -- degrading the first would cost
    img1 detection and confound the survival being measured.

    Run it TWICE: once with ``step = 0`` to calibrate initial temperature against the cooling time
    (no motion), and once at an above-cliff step to get loss AND post-motion temperature versus
    that initial temperature. The t=0 column of the second run is the discriminator; the rest of
    the release grid says what the survivors' temperature did.
    """
    _bootstrap()
    from scan_group import ScanGroup

    ms = [float(t) for t in (precool_ms if precool_ms else [0.0, 0.5, 1.0, 2.0, 5.0])]
    times = [float(t) * 1e-6 for t in (times_us or DEF_TIMES_US)]

    g = ScanGroup()
    rk = _common(g, axis, cool)
    _set_step(rk, axis, float(step))
    rk.step_period_ms = float(period)
    rk.nsteps = int(nsteps)
    if det_deltas:
        # dim 1: pre-motion 556 DETUNING at a fixed pulse length. Both beams move together so the
        # X/h balance the cooling was optimized at is preserved; only the sign/size of the Doppler
        # force changes.
        g().PreMotionCool.Time = float(heat_time_ms) * 1e-3
        g().PreMotionCool.X.FreqDetuning.scan(1, [COOL556_DET_X + d for d in det_deltas])
        g().PreMotionCool.h.FreqDetuning.scan(1, [COOL556_DET_H + d for d in det_deltas])
    else:
        # dim 1: how long the pre-motion 556 cooling runs (0 = none -> the hottest prepared state).
        g().PreMotionCool.Time.scan(1, [t * 1e-3 for t in ms])
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0
    return "RearrangeRnRHeatCommSeq", g


def build_heat(axis, step, period, nsteps_list=None, times_us=None, cool=False):
    """2-D [nsteps x ReleaseRecapture.Time] at one operating point. Returns (seq_name, g)."""
    _bootstrap()
    from scan_group import ScanGroup

    ns = [int(n) for n in (nsteps_list or DEF_NSTEPS_LIST)]
    times = [float(t) * 1e-6 for t in (times_us or DEF_TIMES_US)]

    g = ScanGroup()
    rk = _common(g, axis, cool)
    _set_step(rk, axis, float(step))
    rk.step_period_ms = float(period)
    rk.nsteps.scan(1, ns)
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0               # set for faithfulness; UNREAD by the step
    return "RearrangeRnRHeatCommSeq", g


# --------------------------------------------------------------------------------------- #
# descriptions (provenance -- these end up in the run record)
# --------------------------------------------------------------------------------------- #
_UNITS = {"radial": "knm px (lateral/blaze, +x)", "axial": "um (true_defocus, +z)"}


def _desc_cliff(axis, steps, periods, nsteps):
    return (
        "PPG %s CLIFF finder (2026-07-31 heating campaign, seq RearrangeRnRHeatCommSeq with "
        "ReleaseRecapture.Time pinned 0 so the R&R step adds no bytes): step magnitude %s [%s] "
        "x step_period_ms %s, nsteps=%d (2n=%d steps per shot, out+back), hold_ms=0. "
        "Deliverable: per-step survival S**(1/(2n)) -> the step size at 1%% loss per step (the "
        "'99%% mark'), separately per period, which then sets the just-below / just-above "
        "operating points of the temperature-vs-nsteps scans. Run in the SAME seq and the SAME "
        "config as those scans (33x33_feedback11, z4 = loading_defocus = -4, ifEnhanced=True, "
        "no post-motion cooling: PostRearrCool amps 0 + 0.5 ms dark hold) so the cliff measured "
        "here IS their cliff. %s"
        % (axis, ",".join("%g" % s for s in steps), _UNITS[axis],
           ",".join("%g" % p for p in periods), nsteps, 2 * nsteps,
           ("Axial map = exact spherical (true_defocus=True), depth_piston_corr=0.617 rad/um "
            "(job 320 null); depth_fill_frac cleared (it would override the corr)."
            if axis == "axial" else
            "Lateral blaze via extras.step_x -> xyz step_size [s,0,0]; depth OFF, "
            "true_defocus OFF, depth_piston_corr 0 (all set explicitly -- server extras are "
            "sticky across scans).")))


def _desc_heat(axis, step, period, ns, times_us):
    what = ("NO-MOTION CONTROL (step 0: frames still written, same dark time in trap)"
            if float(step) == 0.0 else
            "operating point %g %s" % (step, _UNITS[axis]))
    return (
        "PPG heating: atom TEMPERATURE vs nsteps by release-and-recapture, %s %s, "
        "step_period_ms=%g, nsteps %s x release time %s us. Seq RearrangeRnRHeatCommSeq: img1 -> "
        "Cool556 5 ms (prepared cold state) -> pingponggrating motion -> NO post-motion cooling "
        "(PostRearrCool amps 0 + 0.5 ms dark hold) -> release -> img2, so the release curve "
        "measures the POST-MOTION temperature. t=0 column = pure motion survival (the R&R step "
        "adds nothing at t=0). Release grid extended to 90 us (the 07-22 campaign stopped at 40, "
        "which left a ~8-10 uK curve constrained only down to S ~ 0.4). hold_ms=0 (no turnaround "
        "dwell -- job 424 showed the dwell is its own nsteps-independent loss channel). "
        "Array 33x33_feedback11, z4 = loading_defocus = -4, ifEnhanced=True. %s "
        "PURPOSE: test whether heating is what the step-size cliff is made of -- T flat in "
        "nsteps below the 99%%-per-step step size and rising above it."
        % (axis, what, period, ",".join(str(n) for n in ns),
           ",".join("%g" % t for t in times_us),
           ("Axial: true_defocus=True (step in um), depth_piston_corr=0.617 rad/um."
            if axis == "axial" else
            "Radial: extras.step_x -> xyz step_size [s,0,0], depth/true_defocus OFF.")))


def submit(seq_name, g, label, description, url=None, reps=None):
    from yb_start_scan import ybStartScan
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan(seq_name, g, url=url, label=label, description=description, **opts)
    print("submitted %s -> descriptor id %s (%d pts; url=%s)"
          % (label, did, g.nseq(), url or "default"))
    return did


def main():
    ap = argparse.ArgumentParser(
        description="pingponggrating heating: temperature vs nsteps at chosen operating points.")
    ap.add_argument("--mode", required=True, choices=("cliff", "heat", "precool"))
    ap.add_argument("--axis", required=True, choices=("radial", "axial"))
    ap.add_argument("--steps", default=None,
                    help="cliff mode: comma-separated step magnitudes (knm px / um)")
    ap.add_argument("--periods", default=None,
                    help="cliff mode: comma-separated step_period_ms (default 0.696,3.0)")
    ap.add_argument("--step", type=float, default=None,
                    help="heat mode: the fixed step magnitude (0 = no-motion control)")
    ap.add_argument("--period", type=float, default=None,
                    help="heat mode: the fixed step_period_ms")
    ap.add_argument("--nsteps", type=int, default=DEF_CLIFF_NSTEPS,
                    help="cliff mode: fixed nsteps (default %d)" % DEF_CLIFF_NSTEPS)
    ap.add_argument("--nsteps-list", default=None,
                    help="heat mode: comma-separated nsteps (default %s)"
                         % ",".join(str(n) for n in DEF_NSTEPS_LIST))
    ap.add_argument("--times", default=None,
                    help="heat mode: comma-separated release times in US (default %s)"
                         % ",".join(str(t) for t in DEF_TIMES_US))
    ap.add_argument("--precool-det", default=None, const="", nargs="?",
                    help="precool mode: sweep pre-motion DETUNING instead of Time -- "
                         "comma-separated deltas in MHz added to both beams' cooling detuning "
                         "(0 = nominal cooling, about -0.135 = X on resonance, more negative "
                         "anti-cools). Bare flag uses %s"
                         % ",".join("%g" % (d / 1e6) for d in DEF_PRECOOL_DET_DELTAS))
    ap.add_argument("--precool-heat-ms", type=float, default=PRECOOL_HEAT_TIME_MS,
                    help="precool mode: fixed pulse length when --precool-det sweeps (default %g)"
                         % PRECOOL_HEAT_TIME_MS)
    ap.add_argument("--precool-ms", default=None,
                    help="precool mode: comma-separated PreMotionCool.Time in MS "
                         "(default 0,0.5,1,2,5; 5 ms = the production prepared state)")
    ap.add_argument("--cool", action="store_true",
                    help="KEEP the production 5 ms post-motion cooling (default: off)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    if args.mode == "precool":
        if args.step is None or args.period is None:
            ap.error("--mode precool needs --step and --period")
        ms = ([float(x) for x in args.precool_ms.split(",") if x.strip()]
              if args.precool_ms else [0.0, 0.5, 1.0, 2.0, 5.0])
        times = ([float(x) for x in args.times.split(",") if x.strip()]
                 if args.times else list(DEF_TIMES_US))
        dd = None
        if args.precool_det is not None:
            dd = ([float(x) * 1e6 for x in args.precool_det.split(",") if x.strip()]
                  if args.precool_det else list(DEF_PRECOOL_DET_DELTAS))
        seq_name, g = build_precool(args.axis, args.step, args.period, precool_ms=ms,
                                    times_us=times, nsteps=args.nsteps, cool=args.cool,
                                    det_deltas=dd, heat_time_ms=args.precool_heat_ms)
        label = args.label or ("PPGPrecool_%s_s%g_p%g" % (args.axis, args.step, args.period))
        desc = (
            "PPG heating DISCRIMINATOR: motion loss + post-motion temperature vs the INITIAL "
            "temperature. %s x release time %s us, %s step %g, "
            "step_period_ms=%g, nsteps=%d. Tells apart the two temperature brackets that the "
            "release fit alone cannot: if the motion loss is EVAPORATION over a barrier lowered "
            "during the step, S(t=0) falls steeply as the pre-motion cooling is spoiled and the "
            "survivors are a truncated (much hotter parent) distribution; if it is an IMPULSIVE "
            "mechanical ejection, S(t=0) is nearly independent of the initial temperature and the "
            "plain thermal fit is already right. PreMotionCool is a dedicated config group so "
            "only the PRE-MOTION cooling is spoiled -- bseq1's other Cool556 (before img1) is "
            "untouched, otherwise img1 detection would degrade and confound the survival. "
            "Run at step=0 too, to calibrate initial T against cooling time. Array "
            "33x33_feedback11, z4 = loading_defocus = -4, hold_ms=0, no post-motion cooling."
            % (("PreMotionCool detuning delta %s MHz at a fixed %g ms pulse"
                % (",".join("%g" % (d / 1e6) for d in dd), args.precool_heat_ms)) if dd else
               ("PreMotionCool.Time %s ms" % ",".join("%g" % m for m in ms)),
               ",".join("%g" % t for t in times),
               args.axis, args.step, args.period, args.nsteps))
        if args.dry_run:
            print("seq=%s nseq=%d mode=precool axis=%s step=%g period=%g nsteps=%d"
                  % (seq_name, g.nseq(), args.axis, args.step, args.period, args.nsteps))
            if dd:
                print("det deltas (MHz) = %s  at fixed %g ms   -> X detuning %s"
                      % ([d / 1e6 for d in dd], args.precool_heat_ms,
                         ["%.3f" % ((COOL556_DET_X + d) / 1e6) for d in dd]))
            else:
                print("precool (ms) = %s" % ms)
            print("times (us)   = %s" % times)
            print("desc: %s" % desc)
            return
    elif args.mode == "cliff":
        if not args.steps:
            ap.error("--mode cliff needs --steps")
        steps = [float(x) for x in args.steps.split(",") if x.strip()]
        periods = ([float(x) for x in args.periods.split(",") if x.strip()]
                   if args.periods else list(DEF_CLIFF_PERIODS))
        seq_name, g = build_cliff(args.axis, steps, periods=periods,
                                  nsteps=args.nsteps, cool=args.cool)
        label = args.label or ("PPGCliff_%s" % args.axis)
        desc = _desc_cliff(args.axis, steps, periods, args.nsteps)
        if args.dry_run:
            print("seq=%s  nseq=%d  mode=cliff axis=%s" % (seq_name, g.nseq(), args.axis))
            print("steps [%s] = %s" % (_UNITS[args.axis], steps))
            print("periods (ms)  = %s" % periods)
            print("nsteps=%d -> %d steps, %d frames/shot; motion ms = %s"
                  % (args.nsteps, 2 * args.nsteps, 2 * args.nsteps + 1,
                     ["%.1f" % ((2 * args.nsteps + 1) * p) for p in periods]))
            print("desc: %s" % desc)
            return
    else:
        if args.step is None or args.period is None:
            ap.error("--mode heat needs --step and --period")
        ns = ([int(x) for x in args.nsteps_list.split(",") if x.strip()]
              if args.nsteps_list else list(DEF_NSTEPS_LIST))
        times = ([float(x) for x in args.times.split(",") if x.strip()]
                 if args.times else list(DEF_TIMES_US))
        seq_name, g = build_heat(args.axis, args.step, args.period,
                                 nsteps_list=ns, times_us=times, cool=args.cool)
        label = args.label or ("PPGHeat_%s_s%g_p%g" % (args.axis, args.step, args.period))
        desc = _desc_heat(args.axis, args.step, args.period, ns, times)
        if args.dry_run:
            print("seq=%s  nseq=%d  mode=heat axis=%s step=%g period=%g"
                  % (seq_name, g.nseq(), args.axis, args.step, args.period))
            print("nsteps        = %s" % ns)
            print("times (us)    = %s" % times)
            print("motion ms/shot= %s"
                  % ["%.1f" % ((2 * n + 1) * args.period) for n in ns])
            print("desc: %s" % desc)
            return

    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    submit(seq_name, g, label, desc, url=args.url, reps=args.reps)


if __name__ == "__main__":
    main()
