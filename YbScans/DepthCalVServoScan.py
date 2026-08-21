"""DepthCalVServoScan.py -- 2-D VSLMservo x 556 |mj|=1 push-out: TRAP DEPTH vs SERVO VOLTAGE.

Measures what has so far only been ASSUMED: that VSLMservo is linear in trap depth, and where
(if anywhere) the 532 servo stops delivering the commanded power.

    dim 1: DepthCal.VServo        -- the trap depth during the push-out ONLY
    dim 2: Pushout.Green.Freq     -- the |mj|=1 spectroscopy axis

seq = ``PushoutDepthRampSeq`` (NEW; ``PushoutSurvivalSeq`` is untouched). Per shot:

    load + img1 at Init.VSLMServo -> adiabatic ramp to DepthCal.VServo -> Cool556 + PushoutStep
    at that depth -> ramp back -> img2 at Init.VSLMServo

So loading and imaging always happen at the depth they were optimized for, and only the
spectroscopy sees the test depth. The |mj|=1 dip centre at each voltage converts to a depth via
``run_analysis._trap_depth_from_lightshift`` (delta_nu = 2*(f0 - f_mj1)), with f0 the mj=0 line.

WHY. The 2026-08-10 d99-vs-depth curve put its knee at ~275 uK using ONE measured depth
(470.5 uK at 0.60 V) scaled linearly in V. The two points defining the low side of that knee sit
at 0.15 and 0.25 V -- and ``PPGTransportCoolScan.py:127`` explicitly warns that this servo "may
sit below its regulation floor at 0-0.25" (same pathology as the 399 imaging PID, memory
open-img1pid-servo-floor). A commanded voltage is not a delivered depth. If the splitting stops
tracking V below some voltage, that voltage is the floor and the knee is an artifact.

DEFAULT VOLTAGE LIST spans the verified and the suspect range deliberately: 0.45 and 0.60 are
at/above the lowest value used in any production array (0.39), so they anchor the linear trend;
0.15/0.25/0.35 are below it, where the answer is unknown.

DEFAULT FREQUENCY WINDOW 104.6-107.6 MHz brackets the |mj|=1 dip for those voltages IF the
linear assumption holds (predicted splittings 0.74-2.95 MHz below the mj=0 line at 108.03 MHz).
A dip that pins to the window edge is itself informative -- it means the depth is far from the
linear prediction -- but re-run with --window widened rather than fitting an edge.

PUSH-OUT POWER: keep it LOW. The |mj|=1 line is inhomogeneously broadened by the depth spread
across sites, and over-driving broadens it further and biases the centre (2026-08-10: amp 0.15
gave FWHM 946 kHz vs the 316-539 kHz typical). 0.08 is the default here.

Run:
    cd pyctrl
    python YbScans/DepthCalVServoScan.py --dry-run
    python YbScans/DepthCalVServoScan.py --reps 5
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()

PATTERN = "17x17_20um"
PHASE_PATH = "phase/17x17_20um.pt"

VSERVOS = [0.15, 0.25, 0.35, 0.45, 0.60]
WINDOW = (104.6, 0.1, 107.6)      # MHz colon (lo, step, hi)
PUSH_AMP = 0.08                   # low -- see the docstring
PUSH_TIME = 20e-3                 # the |mj|=1 recipe (Spectrum556Scan mj=1)
# 3.0 ms per direction. BOTH ramps live in ONE basic sequence here, and the NI AO ramp is
# sampled at the full 400 kHz clock -> 400 samples/ms/channel x 21 Dev1 AO channels, against a
# 65,535-sample onboard FIFO = ~7.8 ms of TOTAL ramp time per basic sequence. 2 x 5 ms asked for
# 85,952 and failed (DAQmx -200341, job 764). PPGTransportDepthCommSeq gets away with 2 x 7 ms
# only because its two ramps sit in DIFFERENT basic sequences, each with its own buffer.
# 3 ms is still ~250 radial trap periods (~12 us), i.e. deeply adiabatic.
RAMP_MS = 3.0


def build(vservos=None, window=None, push_amp=PUSH_AMP, push_time=PUSH_TIME,
          ramp_ms=RAMP_MS, loading_phase=None):
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    vs = list(VSERVOS if vservos is None else vservos)
    lo, step, hi = WINDOW if window is None else window
    freqs = [v * 1e6 for v in matlab_colon(float(lo), float(step), float(hi))]

    g = ScanGroup()

    # ---- push-out: the |mj|=1 recipe, weak so the line is not power-broadened ----
    g().Pushout.Green.Amp = float(push_amp)
    g().Pushout.Time = float(push_time)

    # ---- the two swept axes ----
    g().DepthCal.VServo.scan(1, [float(v) for v in vs])   # trap depth during the push-out
    g().Pushout.Green.Freq.scan(2, freqs)                 # |mj|=1 spectroscopy

    g().DepthCal.RampMs = float(ramp_ms)

    rp = g.runp()
    rp.NumPerGroup = float(len(vs) * len(freqs))
    rp.NumImages = 2
    rp.Scramble = 1        # so a slow drift cannot masquerade as a voltage dependence
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = loading_phase or PHASE_PATH
    # loading_defocus deliberately NOT set -> comes from ByPattern SLM.Loading.Defocus (-6.5)
    return g, vs, freqs


def DepthCalVServoScan(url=None, reps=5, **kw):
    from yb_start_scan import ybStartScan
    g, vs, freqs = build(**kw)
    desc = (
        "TRAP DEPTH vs VSLMservo: 2-D DepthCal.VServo %s V x Pushout.Green.Freq %.2f-%.2f MHz "
        "(%d pts) on the |mj|=1 line, seq PushoutDepthRampSeq. Load + img1 at Init.VSLMServo, "
        "adiabatically ramp the trap to the test voltage (ramp_to smootherstep, %.1f ms), cool + "
        "push out AT that depth (Green.Amp %.2f, %.0f ms -- deliberately weak: amp 0.15 gave "
        "FWHM 946 kHz vs the 316-539 typical), ramp back, img2 at the loading depth. Each "
        "voltage's dip centre -> depth via _trap_depth_from_lightshift. PURPOSE: the d99-vs-depth "
        "knee at ~275 uK rests on ONE measured depth (470.5 uK at 0.60 V) scaled linearly in V, "
        "and its low-side points (0.15/0.25 V) sit in the range PPGTransportCoolScan flags as "
        "possibly below the 532 servo's regulation floor. 0.45/0.60 V anchor the trend (>= the "
        "0.39 V lowest production value); 0.15-0.35 V test it. Array %s."
        % ([round(v, 2) for v in vs], freqs[0] / 1e6, freqs[-1] / 1e6, len(freqs),
           kw.get("ramp_ms", RAMP_MS), kw.get("push_amp", PUSH_AMP),
           kw.get("push_time", PUSH_TIME) * 1e3, PATTERN))
    did = ybStartScan("PushoutDepthRampSeq", g, url=url, rep=reps,
                      label="DepthCalVServo_%dV_x_%df" % (len(vs), len(freqs)),
                      description=desc)
    print("submitted DepthCalVServoScan -> id %s (%d V x %d freq = %d cells, reps=%s)"
          % (did, len(vs), len(freqs), len(vs) * len(freqs), reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--vservos", type=float, nargs="+", default=None)
    ap.add_argument("--window", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=None,
                    help="mj=1 sweep colon in MHz (default %s)" % (WINDOW,))
    ap.add_argument("--push-amp", type=float, default=PUSH_AMP)
    ap.add_argument("--push-time", type=float, default=PUSH_TIME)
    ap.add_argument("--ramp-ms", type=float, default=RAMP_MS)
    ap.add_argument("--loading-phase", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    kw = dict(vservos=a.vservos, window=a.window, push_amp=a.push_amp,
              push_time=a.push_time, ramp_ms=a.ramp_ms, loading_phase=a.loading_phase)
    if a.dry_run:
        g, vs, freqs = build(**kw)
        print("seq=PushoutDepthRampSeq  cells=%d  (%d V x %d freq)" % (g.nseq(), len(vs), len(freqs)))
        print("  VServo (V) = %s" % [round(v, 2) for v in vs])
        print("  freq (MHz) = %.2f .. %.2f step %.2f" %
              (freqs[0] / 1e6, freqs[-1] / 1e6, (freqs[1] - freqs[0]) / 1e6))
        print("  push Green.Amp=%.3f time=%.0f ms   ramp=%.1f ms/direction"
              % (a.push_amp, a.push_time * 1e3, a.ramp_ms))
        s0 = g.getseq(0)
        print("  cell0 DepthCal=%s  Pushout=%s" % (s0.get("DepthCal"), s0.get("Pushout")))
        raise SystemExit(0)
    DepthCalVServoScan(url=a.url, reps=a.reps, **kw)
