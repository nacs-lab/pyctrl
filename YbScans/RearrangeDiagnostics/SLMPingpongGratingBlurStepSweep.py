"""SLMPingpongGratingBlurStepSweep.py -- x-direction pingponggrating with an APERTURE-BLUR
sweep, ONE scan per step-period (0.7 and 1.4 ms by default).

Goal: does widening the trap (lower effective NA) let an atom survive a LARGER per-step
move -- i.e. travel farther per step before the survival cliff?  We pingpong the whole
array left-and-back with a phase-only blaze grating (NO model), 50 steps out + 50 back
(two-way; rests on the clean WGS array for imaging), and 2-D sweep:

  * dim 1 -- step_size: 1.0 .. 2.0 knm-px every 0.1 (11 pts).  The per-step transit
             distance == the "distance per step" axis.
  * dim 2 -- aperture_radius_px: the BLUR knob.  Crops every TRANSIT frame's hologram to
             a CIRCULAR pupil of this radius (knm-1024 / SLM px), flat -> DC outside, so
             the effective NA drops and every trap's spot widens ISOTROPICALLY.  Pure
             pupil restriction: NO phase added -> NO defocus (focal plane unchanged; only
             the spot SIZE changes).  Applied to transit frames ONLY -- the disp=0 WGS
             endpoints stay full-NA / clean for imaging.  0 == OFF (full-NA baseline).

=> 11 x len(APERTURE_RADII_PX) points/scan; REPS reps; one scan per step-period.

Approx spot-widening factor ~ 512 / aperture_radius_px (CALIBRATE before trusting -- run
calibrate_aperture_blur.py on the rearrangement machine to map radius -> FWHM for THIS
array's WGS phase; the effective full pupil is a bit larger than 512 so the real factor
runs a little higher).

pingponggrating is phase-only (no model frames): each frame = WGS_initial + step*disp*x-blaze,
then the circular aperture crop on transit frames.  precompute=True bakes the cropped uint8
transit frames at setup time so the hot loop stays write-only even with blur on (needed to
actually hit the 0.7 ms period).  precompute_host=True is set to match the other pingpong
sweeps (it is a no-op for the pingponggrating short-circuit, which has its own uint8 cache).

Loading / phase config mirrors SLMRearrangementScan.py.  PATTERN defaults to
47x47_feedbackwarm4 but is the thing to CONFIRM at queue time (--pattern).

Run (queues BOTH periods on the 47x47_feedbackwarm4 array):
    cd pyctrl && python YbScans/RearrangeDiagnostics/SLMPingpongGratingBlurStepSweep.py
One period only / explicit backend / different pattern:
    python YbScans/.../SLMPingpongGratingBlurStepSweep.py --period 0.7
    python YbScans/.../SLMPingpongGratingBlurStepSweep.py --pattern 33x33_uniform
    python YbScans/.../SLMPingpongGratingBlurStepSweep.py --url tcp://127.0.0.1:1408
Build-only (no submit; prints point counts + sample swept axes):
    python YbScans/.../SLMPingpongGratingBlurStepSweep.py --dry-run

Prereq: the pyctrl backend running at --url + the SLM server reachable, AND the SLM server
running the SLMnet version with the pingponggrating aperture_radius_px patch
(rearrange_actual.py / trap_shape_tools.py).  If the deployed server lacks it, the
dispatcher silently ignores aperture_radius_px and every blur level is IDENTICAL.
"""
import argparse
import os
import sys

# ---- array / phase config (mirrors SLMRearrangementScan.py) ----
PATTERN = "33x33_centered_level_camfb_1068"      # 2026-06-21 camera-feedback 33x33 -> 1068 sites
PHASE_PATH = "phase/33x33_centered_level_camfb_1068.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]                  # camfb array: no baked defocus (z4=-5 added at runtime)
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"

# Patterns not in SLMRearrangementScan._pattern_cfg (resolved here -> phase, baked_zernike).
_LOCAL_PATTERNS = {
    "33x33_centered_level_camfb_1068":
        ("phase/33x33_centered_level_camfb_1068.pt", [0, 0, 0, 0, 0]),
    "33x33_camfb1068_fb1":
        ("phase/33x33_camfb1068_fb1.pt", [0, 0, 0, 0, 0]),
}

NSTEPS = 50                              # 50 out + 50 back (two-way), fixed (no recompile)
REPS = 10
PERIODS_MS = [0.7, 1.4]                  # one scan per period

# dim-1 step-size axis: 0.9 .. 1.7 px every 0.1 (9 pts), lateral x (knm-1024 px).
# Tighter, more informative region around the survival cliff (was 1.0..2.0).
STEP_VALUES = [round(0.9 + 0.1 * i, 1) for i in range(9)]    # [0.9, 1.0, ..., 1.7]
assert len(STEP_VALUES) == 9 and STEP_VALUES[0] == 0.9 and STEP_VALUES[-1] == 1.7

# dim-2 aperture-radius axis (knm-1024 / SLM px).  0 == OFF (full view / full-NA
# baseline).  Detailed sweep from full view down to 384 in 16-px steps (gentle-blur
# regime; 512 = inscribed pupil, corners clipped only).  Approx widening ~ 512/radius:
# 512->~1.0x ... 384->~1.3x.
APERTURE_RADII_PX = [0, 512, 496, 480, 464, 448, 432, 416, 400, 384]

N_PTS = len(STEP_VALUES) * len(APERTURE_RADII_PX)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_group(period_ms, pattern, phase_path, baked_zernike, reps):
    """Build the (step_size x aperture_radius_px) ScanGroup for one step-period."""
    from scan_group import ScanGroup

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (mirrors SLMRearrangementScan) ----
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = phase_path
    rp.warmup_kwargs.final_phase = phase_path
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [float(z) for z in baked_zernike]
    rp.warmup_kwargs.extras.final_phase_zernike = [float(z) for z in baked_zernike]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- pingponggrating: x-direction lateral, two-way, 2-D (step x aperture) sweep ----
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = NSTEPS
    g().rearrange_kwargs.step_period_ms = float(period_ms)
    # dim 1: scalar step_size (lateral x), fastest axis.
    g().rearrange_kwargs.extras.step_size.scan(1, list(STEP_VALUES))
    # dim 2: scalar aperture_radius_px (the blur knob; 0 = off).
    g().rearrange_kwargs.extras.aperture_radius_px.scan(2, list(APERTURE_RADII_PX))
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                 # match loading_defocus (focal plane)
    g().rearrange_kwargs.extras.initial_pattern = pattern
    g().rearrange_kwargs.extras.final_pattern = pattern

    # ---- run params (mirror SLMRearrangementScan; cooling left at config defaults) ----
    rp.NumPerGroup = N_PTS * reps
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    return g


def SLMPingpongGratingBlurStepSweep(url=None, periods=None, pattern=None, dry_run=False,
                                    reps=None):
    """Build + submit one pingponggrating blur x step scan per step-period.  Returns
    descriptor ids (or ScanGroups when ``dry_run``)."""
    _bootstrap()

    periods = list(PERIODS_MS if periods is None else periods)
    reps = REPS if reps is None else int(reps)
    pattern = pattern or PATTERN
    # Resolve phase/baked-zernike: local table first (new arrays), then the production
    # SLMRearrangementScan table, then module defaults.
    phase_path, baked = PHASE_PATH, BAKED_ZERNIKE
    if pattern in _LOCAL_PATTERNS:
        phase_path, baked = _LOCAL_PATTERNS[pattern]
    else:
        try:
            from SLMRearrangementScan import _pattern_cfg  # type: ignore
            cfg = _pattern_cfg(pattern)
            phase_path, baked = cfg["phase_path"], cfg["baked_zernike"]
        except Exception:
            print("WARN: could not resolve %r; using module defaults (%s, baked=%s)"
                  % (pattern, PHASE_PATH, BAKED_ZERNIKE))

    out = []
    if dry_run:
        print("DRY RUN -- no submission.")
        print("  pattern: %s  phase: %s  baked_zernike: %s" % (pattern, phase_path, baked))
        print("  step_size axis (dim1, %d): %s" % (len(STEP_VALUES), STEP_VALUES))
        print("  aperture_radius_px axis (dim2, %d): %s  [0 = off]"
              % (len(APERTURE_RADII_PX), APERTURE_RADII_PX))
        for per in periods:
            g = _build_group(per, pattern, phase_path, baked, reps)
            nseq = g.nseq()
            print("  period %g ms: nseq=%d (expect %d); reps=%d -> %d shots"
                  % (per, nseq, N_PTS, reps, nseq * reps))
            out.append(g)
        print("would queue %d scans (periods %s); ~%d shots total"
              % (len(periods), periods, N_PTS * reps * len(periods)))
        return out

    from yb_start_scan import ybStartScan
    for per in periods:
        g = _build_group(per, pattern, phase_path, baked, reps)
        desc = ("x-direction pingponggrating APERTURE-BLUR sweep on %s: dim1 step_size %s "
                "knm-px (%d pts) x dim2 aperture_radius_px %s (%d pts, 0=off) = %d points; "
                "two-way nsteps=%d, period=%g ms, precompute+precompute_host, %d reps "
                "(-> %d shots). Blur = circular pupil crop on transit frames only (no "
                "defocus); endpoints full-NA. Find the largest step_size that survives per "
                "blur level (survival-vs-per-step-distance)."
                % (pattern, STEP_VALUES, len(STEP_VALUES), APERTURE_RADII_PX,
                   len(APERTURE_RADII_PX), N_PTS, NSTEPS, per, reps, N_PTS * reps))
        did = ybStartScan("RearrangeCommSeq", g, url=url,
                          label="PPGratingBlurStep_p%g" % per,
                          description=desc, rep=reps)
        print("submitted period %g ms -> descriptor id %s (%d pts, %d reps = %d shots)"
              % (per, did, N_PTS, reps, N_PTS * reps))
        out.append(did)
    print("queued %d scans (periods %s); ~%d shots total"
          % (len(out), periods, N_PTS * reps * len(out)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Queue x-direction pingponggrating aperture-blur x step sweeps.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--period", type=float, default=None, action="append",
                    help="step-period (ms) to queue; repeatable. Default: 0.7 and 1.4")
    ap.add_argument("--pattern", default=None,
                    help="loading pattern name (default: %s)" % PATTERN)
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the grid per scan (default %d)" % REPS)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the ScanGroups and print point counts; do NOT submit")
    args = ap.parse_args()
    SLMPingpongGratingBlurStepSweep(url=args.url, periods=args.period,
                                    pattern=args.pattern, dry_run=args.dry_run,
                                    reps=args.reps)
