"""PPGAxialHeatScan.py -- release-and-recapture thermometry for the axial campaign.

The thermometry twin of :mod:`PPGAxialPseudoOneWayScan`. Same protocol, same extras, same axial
map, same piston constants -- the ONLY differences are the sequence (``RearrangeRnRHeatCommSeq``,
which adds a release-and-recapture window after the motion) and a second scan axis over the release
time. Keeping the two modules in lockstep matters: a temperature is only attributable to a move if
it is the SAME move the survival scan measured.

WHY NOT PPGHeatRnRScan
----------------------
That module is the older lateral-first one: ``LOADING_DEFOCUS = -5`` (this campaign is -4),
``ifEnhanced=False`` (production is True), and -- the blocker -- it sets NEITHER ``true_defocus``
NOR ``depth_piston_corr``. Server extras are merge-only and sticky, so it would silently inherit
whatever the previous scan left behind and reinterpret ``step_z`` as radians of Z4 (0.80x the
travel). That is the campaign's single most likely failure mode, so this module sets every axial
flag EXPLICITLY, every scan.

Two modes
---------
``pseudo``  The pseudo-one-way move under test (campaign step 3b). dim 1 = tested per-step
            amplitude of the RETURN leg, dim 2 = release time. ``n_out`` is held CONSTANT via
            ``out_nsteps_min`` exactly as in the survival scan, so every cell pays the same
            in-trap precompute delay and the temperature axis is not contaminated by a
            stroke-correlated hold.
``sym``     A plain SYMMETRIC small-step ping-pong at fixed ``nsteps`` (campaign step 1b, the
            OUTWARD-LEG qualification). dim 1 = step magnitude, dim 2 = release time. This is the
            acceptance test for constant-``n_out``: the outward leg must be lossless AND cold
            across the WHOLE range of ``s_out`` the constant-``n_out`` scheme will command
            (0 .. out_step_max), at the n it will command it. A symmetric round trip makes 2n
            steps, so it is the conservative version of the one-way outward leg.

RELEASE GRID and the t=0 caveat are inherited verbatim from the 07-31 / 08-06 campaigns so the
temperatures are directly comparable: ``0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90 us``.
``RearrangeRnRStep`` RETURNS EARLY at exactly t=0 (no ``AmpSLM`` toggle, no ``TTLSampleAndHold``
re-assert, no 3 us AOM settle), so the t=0 shot never pays the trap restore that every t>0 shot
pays -- worth ~1% survival. ``t = 0.5 us`` takes the full branch and IS the restore-only reference;
the pair (0, 0.5) brackets it.

POST-MOTION COOLING IS OFF (``PostRearrCool`` amps 0 + a short dark hold) so the release probes the
POST-MOTION temperature rather than the re-cooled one.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGAxialHeatScan.py --mode sym --steps "0,0.25,0.5,0.75,1.0" --nsteps 250 --dry-run
    python YbScans/RearrangeDiagnostics/PPGAxialHeatScan.py --mode pseudo --force
"""

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PPGAxialPseudoOneWayScan import (  # noqa: E402  -- shared campaign constants
    PATTERN, PHASE_PATH, BAKED_ZERNIKE, MODEL_FILENAME, DEFOCUS, BASE_PERIOD_MS,
    OUT_PERIOD_MS, HOLD_MS, OUT_STEP_MAX, PISTON_CORR, RET_PISTON_CORR, PEAK_MAX_UM,
    BLUE_DETUNING_MHZ, BLUE_LOADING_TIME_S, BLUE_BIAS_X_A, LAC_TIME_S, IMAG_DETUNING_MHZ,
    STEP_GRID, _bootstrap, _image_patterns_json, _worst_peak,
)

HEAT_SEQ = "RearrangeRnRHeatCommSeq"
RELEASE_US = [0, 0.5, 6, 12, 20, 30, 40, 55, 70, 90]
POST_COOL_HOLD_MS = 0.5


def build(mode="pseudo", direction=+1, steps=None, times_us=None, fixed_nsteps=50,
          period_ms=BASE_PERIOD_MS, out_step_max=OUT_STEP_MAX, const_nout=True,
          hold_ms=HOLD_MS, piston_corr=PISTON_CORR, ret_piston_corr=RET_PISTON_CORR,
          post_cool_hold_ms=POST_COOL_HOLD_MS, cool=False, allow_over_cap=False,
          label_extra="", img_pid=(0.80, 1.00)):
    """Build (do NOT submit) the 2-D [step x release] heat ScanGroup -> ``(seq, g, meta)``."""
    _bootstrap()
    from scan_group import ScanGroup

    sgn = -1.0 if direction > 0 else +1.0
    steps = list(STEP_GRID if steps is None else steps)

    # *** UNITS: ReleaseRecapture.Time is in SECONDS. ***
    # The grid is written in MICROSECONDS (that is how every release grid in this lab is quoted)
    # and MUST be converted here, exactly as PPGOneWayHeatScan and PPGHeatRnRScan both do
    # (`times = [float(t) * 1e-6 for t in times_us]`). Job 312 was submitted without this
    # conversion on 2026-08-07 and therefore commanded release windows of 0.5-90 SECONDS: the scan
    # ran for 5 minutes without producing a single shot. The assert below makes the failure loud
    # instead of slow -- a bare number in this list is microseconds, and anything that would imply
    # a >1 ms release is a units error, not a physics choice.
    times_us_list = list(RELEASE_US if times_us is None else times_us)
    if max(times_us_list) > 1000.0:
        raise ValueError(
            "release times look like they are already in seconds (max %g). This list is in "
            "MICROSECONDS; the 1e-6 conversion happens here." % max(times_us_list))
    times = [float(t) * 1e-6 for t in times_us_list]

    peak = _worst_peak(steps, [fixed_nsteps])
    if peak > PEAK_MAX_UM and not allow_over_cap:
        raise ValueError("axial heat: worst-cell peak %.1f um exceeds the %.0f um "
                         "pupil-sampling cap; trim the step grid or lower nsteps."
                         % (peak, PEAK_MAX_UM))

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

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


    # LOADING RECOVERY (jobs 309/310): 399 blue capture had drifted off plateau -- the old
    # -40/-42 MHz work-point read 0.002 fill. -48 MHz + 0.25 s restores 0.467. Applied as a
    # per-scan g() override; expConfig.py is NOT modified.
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
    g().GreenMOT.BiasCoilCurrent.X = BLUE_BIAS_X_A   # 08-08 job 594: rate plateau AND x-gradient null
    g().BlueMOT.LoadingTime = BLUE_LOADING_TIME_S
    g().LAC.Time = LAC_TIME_S                        # 08-09 user request: 30 -> 35 ms
    # 399 imaging detuning -- 08-09 rounds 103/104; see IMAG_DETUNING_MHZ in the pseudo-one-way
    # module for the measurement and the provenance boundary. g() beats the ByPattern overlay.
    # NOTE for thermometry specifically: 3b's 9.5-10.5 uK baselines (jobs 604/605) were measured at
    # -5 MHz. Release-recapture reads survival-vs-release-time, so a change in imaging fidelity
    # shifts the whole curve's normalisation -- any NEW thermometry must be baselined at -1.0 and
    # must not be compared point-by-point against those two runs.
    g().Imag399.FreqDetuning = IMAG_DETUNING_MHZ * 1e6

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

    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"

    n_out_min = 0
    if mode == "pseudo":
        # Identical to PPGAxialPseudoOneWayScan mode=nsteps at a FIXED return_nsteps.
        rk.step_period_ms = float(OUT_PERIOD_MS)          # outward leg pacing
        rk.nsteps = 1                                     # placeholder; the fold overwrites it
        rk.extras.out_step_max = float(out_step_max)
        if const_nout and peak > 0:
            n_out_min = int(math.ceil(peak / float(out_step_max)))
            rk.extras.out_nsteps_min = n_out_min
        rk.extras.return_step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_nsteps = int(fixed_nsteps)
        rk.extras.return_step_period_ms = float(period_ms)
        rk.extras.return_trip = True
        if ret_piston_corr is not None:
            rk.extras.return_depth_piston_corr = float(ret_piston_corr)
    elif mode == "sym":
        # Plain symmetric ping-pong: 2*nsteps steps, both legs identical. NO out_step_max, so the
        # fold stays inert and nsteps/step_size are exactly what is set here.
        rk.step_period_ms = float(period_ms)
        rk.nsteps = int(fixed_nsteps)
        rk.extras.step_size.scan(1, [sgn * abs(float(s)) for s in steps])
        rk.extras.return_trip = True
    else:
        raise ValueError("unknown mode %r" % (mode,))

    # ---- axial mode, EXPLICIT every scan (sticky-extras unit trap) --------------------
    rk.extras.depth = True
    rk.extras.true_defocus = True
    rk.extras.depth_piston_corr = float(piston_corr)
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0
    rk.extras.hold_ms = float(hold_ms)

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- dim 2: the release window ----------------------------------------------------
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0

    # POST-MOTION cooling OFF so the release probes the post-motion temperature.
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(post_cool_hold_ms) * 1e-3

    rp.NumPerGroup = 100000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    meta = dict(mode=mode, direction=direction, steps=steps, times=times_us_list,
                fixed_nsteps=fixed_nsteps, period_ms=period_ms, out_step_max=out_step_max,
                hold_ms=hold_ms, piston_corr=piston_corr, ret_piston_corr=ret_piston_corr,
                worst_peak_um=peak, n_out_min=n_out_min, npts=g.nseq(), cool=cool,
                label_extra=label_extra)
    return HEAT_SEQ, g, meta


def _desc(meta):
    d = "+z" if meta["direction"] > 0 else "-z"
    if meta["mode"] == "sym":
        head = (
            "AXIAL OUTWARD-LEG THERMOMETRY -- the acceptance test for CONSTANT n_out. "
            "SYMMETRIC pingponggrating (both legs identical, 2*nsteps = %d steps per shot) at "
            "fixed nsteps=%d, sweeping the step magnitude over the FULL range that the "
            "constant-n_out scheme will command on the outward leg (s_out = D/n_out spans "
            "~0 .. %.2f um across the main grid). The outward leg is only admissible if it is "
            "lossless AND cold CONSISTENTLY across that whole range at this n -- if temperature "
            "rises at the large-s_out end, the small-D and large-D cells of the main scan would "
            "arrive at the tested move in different thermal states and the measured cliff would "
            "be contaminated. A symmetric round trip is the CONSERVATIVE proxy for the one-way "
            "outward leg (twice the steps)."
            % (2 * meta["fixed_nsteps"], meta["fixed_nsteps"], meta["out_step_max"]))
    else:
        head = (
            "AXIAL PSEUDO-ONE-WAY %s THERMOMETRY at fixed return_nsteps=%d. The tested move is the "
            "RETURN leg (out in %s in small qualified steps, %0.f ms settle hold at the "
            "turnaround, then the move under test back to displacement 0 = the stored WGS phase). "
            "dim 1 = tested per-step amplitude in MICRONS. n_out held CONSTANT at %d "
            "(out_nsteps_min) so every cell pays the same in-trap precompute delay -- otherwise "
            "the delay scales with D, is perfectly correlated with the tested step, and vacuum "
            "loss would masquerade as stroke-dependent heating. Same protocol/extras/piston as "
            "the matching survival scan, so T is attributable to the SAME move."
            % (d, meta["fixed_nsteps"], "-z" if meta["direction"] > 0 else "+z",
               meta["hold_ms"], meta["n_out_min"]))
    return (
        head +
        " dim 2 = ReleaseRecapture.Time %s us. RELEASE GRID starts 0 AND 0.5 us on purpose: "
        "RearrangeRnRStep returns early at exactly t=0 (no AmpSLM toggle, no TTLSampleAndHold "
        "re-assert, no 3 us AOM settle), so the t=0 shot does NOT pay the trap restore every t>0 "
        "shot pays (~1%% survival, 07-31 caveat #4); t=0.5 us takes the full branch and IS the "
        "restore-only reference. POST-MOTION COOLING %s so the release probes the post-motion "
        "temperature. true_defocus=True (step in MICRONS, exact spherical map) with "
        "depth_piston_corr=%.3f rad/um%s -- every axial flag set EXPLICITLY because server extras "
        "are merge-only and sticky, and a cleared flag would silently reinterpret the step as "
        "radians of Z4 (0.80x the travel). depth_fill_frac cleared (it would override the corr). "
        "Worst-cell peak %.1f um (cap %.0f). Array %s, z4 = loading_defocus = %+.0f. %s"
        % (meta["times"], "ON (production Cool556)" if meta["cool"] else "OFF (amps 0)",
           meta["piston_corr"],
           ("" if meta["ret_piston_corr"] is None
            else ", %.3f rad/um on the tested leg" % meta["ret_piston_corr"]),
           meta["worst_peak_um"], PEAK_MAX_UM, PATTERN, DEFOCUS, meta["label_extra"]))


def submit(url=None, reps=None, label=None, **kw):
    seq, g, meta = build(**kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    lbl = label or ("PPGAxialHeat_%s_%s" % (meta["mode"],
                                            "pz" if meta["direction"] > 0 else "nz"))
    did = ybStartScan(seq, g, url=url, label=lbl, description=_desc(meta), **opts)
    print("submitted %s -> id %s (%d pts = %d steps x %d release; worst peak %.1f um; n_out_min %d)"
          % (lbl, did, meta["npts"], len(meta["steps"]), len(meta["times"]),
             meta["worst_peak_um"], meta["n_out_min"]))
    return did


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()] if s else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Axial release-and-recapture thermometry.")
    ap.add_argument("--mode", default="pseudo", choices=("pseudo", "sym"))
    ap.add_argument("--direction", type=int, default=+1, choices=(+1, -1))
    ap.add_argument("--steps", default=None, help="step MAGNITUDES in um (comma-separated)")
    ap.add_argument("--times", default=None, help="release times in us (comma-separated)")
    ap.add_argument("--nsteps", type=int, default=50,
                    help="pseudo: return_nsteps; sym: nsteps each way")
    ap.add_argument("--period", type=float, default=BASE_PERIOD_MS)
    ap.add_argument("--out-step-max", type=float, default=OUT_STEP_MAX)
    ap.add_argument("--no-const-nout", action="store_true")
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR)
    ap.add_argument("--ret-corr", type=float, default=RET_PISTON_CORR)
    ap.add_argument("--cool", action="store_true")
    ap.add_argument("--allow-over-cap", action="store_true")
    ap.add_argument("--img-pid", type=float, nargs=2, metavar=("IMG1","IMG2"),
                    default=(0.80, 1.00),
                    help="BlueMOT.Img1/Img2PIDSet (V); default = the documented ByPattern "
                         "33x33_feedback11 values. A/B jobs 340-342 showed PIDSet is not "
                         "the lever in this sequence.")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    kw = dict(mode=args.mode, direction=args.direction, steps=_floats(args.steps),
              times_us=_floats(args.times), fixed_nsteps=args.nsteps, period_ms=args.period,
              out_step_max=args.out_step_max, const_nout=not args.no_const_nout,
              hold_ms=args.hold_ms, piston_corr=args.corr, ret_piston_corr=args.ret_corr,
              cool=args.cool, allow_over_cap=args.allow_over_cap, label_extra=args.note,
              img_pid=tuple(args.img_pid))

    if args.dry_run:
        _seq, _g, _meta = build(**kw)
        print("seq=%s  nseq=%d  mode=%s  dir=%+d  nsteps=%d  worst_peak=%.1f um  n_out_min=%d"
              % (_seq, _g.nseq(), _meta["mode"], _meta["direction"], _meta["fixed_nsteps"],
                 _meta["worst_peak_um"], _meta["n_out_min"]))
        print("steps [um]   = %s" % _meta["steps"])
        print("release [us] = %s" % _meta["times"])
        print("\nDESCRIPTION:\n%s" % _desc(_meta))
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, label=args.label, **kw)
