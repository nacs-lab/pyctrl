"""AxialLoadingTimeScan.py -- BlueMOT.LoadingTime sweep to recover the loading rate.

Loading fell from ~0.54 (2026-08-06 21:30) to 0.07-0.26 overnight. Detection is healthy throughout
(occupied-vs-empty site intensity separation held at 6.5-7.1 ADU, and 94% of sites still load at
least once), so atoms ARE being trapped -- there are simply far fewer of them. That is a loading
RATE problem, not thresholds, not the grid, and not the 556-unlock signature (which reads ~0
loading with NO intensity separation at all).

Phase 0a of the loading runbook: the ``BlueMOT.LoadingTime`` curve. More blue-MOT capture time
delivers more atoms to the green MOT and hence to the tweezers, up to the collisional-blockade
ceiling (~0.50-0.60 fill) beyond which nothing improves and the extra time is pure cycle cost.

**expConfig.py IS NOT TOUCHED** (explicit user instruction). The override lives in this scan's
``g()`` block, and the chosen value is then applied the same way in the campaign scan modules.

Read BOTH axes at every point, per the runbook -- the array mean hides a gradient:
  * rate       -- mean ``logicals_img1`` fill fraction
  * uniformity -- per-site CV and the x/y spatial gradient, plus shot-to-shot std

Pick the PLATEAU CENTRE, not the raw max: the cliff drifts, and a work-point perched on the knee
degrades as soon as anything moves. Longer is not free -- LoadingTime is the dominant cycle-time
cost, so the right choice is the shortest time that reaches the plateau.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/AxialLoadingTimeScan.py --dry-run
    python YbScans/RearrangeDiagnostics/AxialLoadingTimeScan.py --force
"""

import argparse
import os
import sys

# scan_bootstrap lives in YbScans/, one level up from RearrangeDiagnostics/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scan_bootstrap  # noqa: E402
scan_bootstrap.bootstrap()

PATTERN_PHASE = "phase/33x33_feedback11.pt"
DEFOCUS = -5.0                       # production loading plane (LACScan + the server's sticky value)
TIMES_S = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


DETUNINGS_MHZ = [-36, -38, -40, -42, -44, -46, -48, -50, -52]
# GreenMOT bias-X: bracket the 06/05 optimum 0.040 A by +-0.010 at 2.5 mA, so both
# shoulders fall off. Razor-sharp -- a 0.01 A error is the difference between full and zero.
BIASES_X = [0.030, 0.0325, 0.035, 0.0375, 0.040, 0.0425, 0.045, 0.0475, 0.050]
WORKPOINT_TIME_S = 0.30          # unsaturated at the CURRENT config (job 308: 0.394 fill)


def build(times=None, defocus=DEFOCUS, knob="time", detunings=None,
          fixed_time=WORKPOINT_TIME_S, fixed_detuning=None, biases=None, fixed_bias=None):
    from scan_group import ScanGroup
    times = list(times or TIMES_S)
    g = ScanGroup()

    # Bias-X is the razor-sharp alignment resonance (~3 mA FWHM) and it DRIFTS between sessions,
    # so a time or detuning sweep taken at a stale bias is measuring a partly-detuned MOT. Pin it
    # to the value re-centred at the start of the session whenever one is known.
    if fixed_bias is not None and knob != "bias":
        g().GreenMOT.BiasCoilCurrent.X = float(fixed_bias)

    if knob == "time":
        # THE ONLY SWEPT KNOB. Everything else stays at expConfig defaults so this is a clean
        # 1-D read of the capture curve -- and nothing is left changed on the apparatus after.
        g().BlueMOT.LoadingTime.scan(1, times)
        if fixed_detuning is not None:
            # Re-reading the cliff AT the recovered detuning: blue capture is the one knob
            # that MOVES the cliff, so the knee must be re-found after changing it (06/05:
            # -40 -> -44 pulled full rate from ~0.27 s to ~0.21 s). This is what lets the
            # LoadingTime stay short and the shot time low.
            g().BlueMOT.FreqDetuning = float(fixed_detuning) * 1e6
    elif knob == "bias":
        # GREENMOT BIAS COIL -- the razor-sharp alignment knob. The 06/05 runbook: bias-X is a
        # near-vertical resonance (dead <= 0.037 A, full at 0.040), tolerance ~+-0.005 A, and a
        # ~0.01 A drift collapses loading to zero. It shifts the green-MOT cloud relative to the
        # tweezer array, so it sets BOTH peak rate and uniformity.
        #
        # WHY HERE, 2026-08-07 05:45: loading declined monotonically 0.61 -> 0.23 over ~90 min with
        # the 399 blue detuning RE-VERIFIED unchanged (job 367: -48 still optimal, curve identical
        # to job 309 four hours earlier), the oven rock-steady (372.06 vs 372.08 C) and imaging
        # IMPROVING (d' 11.3). Meanwhile the MOT coils cooled 19.2 -> 17.3 C, which drifts the
        # field and hence the cloud position -- the one mechanism consistent with all of that.
        g().BlueMOT.LoadingTime = float(fixed_time)
        if fixed_detuning is not None:
            g().BlueMOT.FreqDetuning = float(fixed_detuning) * 1e6
        g().GreenMOT.BiasCoilCurrent.X.scan(1, [float(v) for v in (biases or BIASES_X)])
    elif knob == "detuning":
        # BLUE CAPTURE -- the RATE lever, and the one knob that MOVES THE LOADING CLIFF.
        # Job 308 showed the ceiling (~0.60) is intact but now needs ~2 s to reach, where the
        # 06/05 optimized config hit full rate by ~0.21 s: the cliff has marched right by ~10x.
        # Alignment and cooldown do NOT move the cliff (06/05) -- blue capture does. The runbook's
        # plateau is -44 to -48 MHz with the -40 default on the rising edge, so a drift here is
        # exactly the shape observed.
        #
        # Swept at a SHORT LoadingTime (%.2f s, ~0.39 fill on job 308) because the whole game is
        # to tune in the UNSATURATED regime -- at a long LoadingTime everything saturates at the
        # collisional-blockade ceiling and every detuning looks equally good.
        g().BlueMOT.LoadingTime = float(fixed_time)
        g().BlueMOT.FreqDetuning.scan(1, [float(d) * 1e6 for d in
                                          (detunings or DETUNINGS_MHZ)])
    else:
        raise ValueError("unknown knob %r" % (knob,))

    rp = g.runp()
    rp.NumPerGroup = 100000          # upper bound; --reps sets the pass count
    rp.NumImages = 1                 # loading-rate readout only; no survival needed
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.Scramble = 1                  # randomize order so run-start warmup does not bias low times
    rp.loading_phase = PATTERN_PHASE
    rp.loading_defocus = float(defocus)
    return "TweezerLoadingSeq", g, times


DESC = (
    "BlueMOT.LoadingTime sweep %s s to recover the loading rate, which fell from ~0.54 "
    "(08-06 21:30) to 0.07-0.26 overnight. Detection is HEALTHY across that drop -- occupied-vs-"
    "empty site intensity separation held at 6.5-7.1 ADU and 94%% of sites still load at least "
    "once -- so atoms are being trapped and there are simply far fewer of them: a loading RATE "
    "problem, not thresholds/grid, and not the 556-unlock signature (that reads ~0 loading with NO "
    "separation). Phase 0a of the loading runbook. expConfig.py is NOT modified; the override is "
    "in this scan's g() block only. NOTE a defocus hypothesis was tested and REFUTED first: z4=-4 "
    "and z4=-5 both loaded ~0.07-0.12 back to back, so the earlier apparent -5-loads-better "
    "correlation was confounded by time, not focal plane. Read rate AND uniformity (CV, x/y "
    "gradient) at every point and pick the PLATEAU CENTRE, not the raw max -- LoadingTime is the "
    "dominant cycle-time cost, so the right answer is the shortest time that reaches the plateau. "
    "1 image, Scramble on. Array 33x33_feedback11, z4 = %+.0f."
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BlueMOT.LoadingTime recovery sweep.")
    ap.add_argument("--times", default=None, help="comma-separated LoadingTime values in s")
    ap.add_argument("--defocus", type=float, default=DEFOCUS)
    ap.add_argument("--knob", default="time", choices=("time", "detuning", "bias"))
    ap.add_argument("--biases", default=None,
                    help="GreenMOT.BiasCoilCurrent.X values in A (comma-separated)")
    ap.add_argument("--detunings", default=None,
                    help="comma-separated BlueMOT.FreqDetuning values in MHz")
    ap.add_argument("--fixed-detuning", type=float, default=None,
                    help="hold BlueMOT.FreqDetuning at this value (MHz) while sweeping time")
    ap.add_argument("--fixed-time", type=float, default=WORKPOINT_TIME_S,
                    help="LoadingTime held fixed while sweeping detuning (s)")
    ap.add_argument("--fixed-bias", type=float, default=None,
                    help="hold GreenMOT.BiasCoilCurrent.X at this value (A) while sweeping "
                         "time/detuning -- pass the value re-centred this session, since the "
                         "resonance is ~3 mA wide and drifts")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tv = [float(x) for x in args.times.split(",") if x.strip()] if args.times else None
    dv = [float(x) for x in args.detunings.split(",") if x.strip()] if args.detunings else None
    seq, g, times = build(tv, args.defocus, knob=args.knob, detunings=dv,
                          fixed_time=args.fixed_time,
                          fixed_detuning=args.fixed_detuning,
                          biases=[float(x) for x in args.biases.split(",")] if args.biases else None,
                          fixed_bias=args.fixed_bias)
    if args.knob == "detuning":
        desc = (
            "BlueMOT.FreqDetuning (399 BLUE CAPTURE) sweep %s MHz at a FIXED LoadingTime of %.2f s, "
            "to recover the loading RATE. Job 308 measured the LoadingTime curve and found the "
            "ceiling INTACT (~0.60 fill at 2-3 s) but the cliff marched RIGHT by ~10x: full rate "
            "now needs ~2 s where the 2026-06-05 optimized config reached it by ~0.21 s. That is "
            "the blue-capture signature -- alignment and cooldown do NOT move the cliff (06/05), "
            "blue capture does -- and it also explains why every rearrange scan tonight read "
            "0.07-0.26 while this seq reads 0.39-0.60: those scans inherit expConfig's short "
            "LoadingTime (~0.23 s from the 06/05 optimization), which now sits BELOW the shifted "
            "cliff. Same apparatus, different point on a moved curve. "
            "The runbook's plateau is -44 to -48 MHz with the -40 default on the rising edge, so a "
            "drift here produces exactly this shape. Swept in the UNSATURATED regime on purpose "
            "(%.2f s gave 0.394 fill on job 308): at a long LoadingTime everything saturates at the "
            "collisional-blockade ceiling and no detuning discriminates. "
            "IMAGING IS NOT TOUCHED -- this is the loading-side 399 only. expConfig.py is NOT "
            "modified; the override lives in this scan's g() block. Read rate AND uniformity (CV, "
            "x/y gradient) at every point and pick the PLATEAU CENTRE, not the raw max. "
            "1 image, Scramble on. Array 33x33_feedback11, z4 = %+.0f."
            % (dv or DETUNINGS_MHZ, args.fixed_time, args.fixed_time, args.defocus))
    elif args.knob == "bias":
        desc = (
            "GreenMOT.BiasCoilCurrent.X (green-MOT POSITION along x) sweep %s A at a FIXED "
            "LoadingTime of %.2f s. This is the razor-sharp alignment resonance: 06/05 put it at "
            "~3 mA FWHM with a ~0.005 A tolerance, and on 2026-08-07 it was measured to have moved "
            "~6 mA in a single night (0.040 A -> 0.0343 A) tracking a 2 C MOT-coil temperature "
            "change -- so it MUST be re-centred at the start of any multi-hour campaign rather "
            "than assumed. The failure is silent: loading decays smoothly rather than breaking, so "
            "it reads as 'the rig is a bit worse tonight'. Swept in the UNSATURATED regime on "
            "purpose (%.2f s) -- at a long LoadingTime everything saturates at the "
            "collisional-blockade ceiling and no bias value discriminates. Read rate AND "
            "uniformity (CV, x/y gradient) at every point and adopt the PLATEAU CENTRE, not the "
            "raw max, and prefer the side the resonance is drifting toward. Detection health "
            "(occupied-vs-empty ADU separation) is the triage split if this reads flat-zero: a "
            "real rate problem keeps ~6-7 ADU separation, the 556-unlock signature has none. "
            "expConfig.py is NOT modified; the override lives in this scan's g() block. "
            "1 image, Scramble on. Array 33x33_feedback11, z4 = %+.0f."
            % ([float(x) for x in args.biases.split(",")] if args.biases else BIASES_X,
               args.fixed_time, args.fixed_time, args.defocus))
    else:
        desc = DESC % (times, args.defocus)
    if args.dry_run:
        print("seq=%s nseq=%d times=%s reps=%d" % (seq, g.nseq(), times, args.reps))
        print("est shots %d; cycle grows with LoadingTime so expect ~%.1f min"
              % (g.nseq() * args.reps,
                 sum((2.0 + t) * args.reps for t in times) / 60.0))
        print("\n%s" % desc)
    elif not args.force:
        ap.error("refusing to submit without --force")
    else:
        from yb_start_scan import ybStartScan
        lbl = {"time": "AxialLoadingTimeRecovery",
               "detuning": "AxialBlueDetuningRecovery",
               "bias": "AxialGreenBiasXRecovery"}[args.knob]
        did = ybStartScan(seq, g, url=args.url, label=lbl,
                          description=desc, rep=args.reps)
        print("submitted %s -> id %s (%d pts x %d reps)"
              % (lbl, did, g.nseq(), args.reps))
