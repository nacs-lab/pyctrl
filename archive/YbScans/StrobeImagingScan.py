"""StrobeImagingScan.py -- strobe-imaging twin of CoolingScan (submit-only).

Optimises STROBE imaging on ``StrobeImagingPushoutSeq`` (the two survival images are StrobeImag399Step;
fidelity from img1, survival from img1->img2), mirroring CoolingScan on ImagingPushoutSurvivalSeq.

Modes (each on the SAME seq; --pulse-time / --recool-time are FIXED per submission -> the round-runner's
OUTER loop varies them, they can NEVER be .scan() axes because n_cycles=round(exposure/period) is a
Python loop count):

  amps      -- 2-D Imag399.Amp1 (dim1) x Amp2 (dim2): the two 399 imaging-beam amplitudes.
  x2d       -- 2-D Imag399.Strobe.Cool556.X.{FreqDetuning(dim1, MHz), Amp(dim2)}: the strobe RECOOL X beam.
  h2d       -- 2-D Imag399.Strobe.Cool556.h.{FreqDetuning, Amp}: the strobe RECOOL h beam.
  detuning  -- 1-D Imag399.FreqDetuning (MHz): the 399 imaging line.
  beamtest  -- fire StrobeBeamTestSeq (NO atoms) to characterize the AOM response on a scope.

Collect window: set by Orca.ExposureTime (base or ByPattern) -- the strobe train + the camera frame both
follow it. To AMPLIFY strobe-imaging loss prefer raising Orca.ExposureTime (more strobe dose) over the
PushouthX hold; --pushout-time here is a (dark, beams-off) hold for a trap-lifetime cross-check.

Run (pyctrl backend live at --url; reps drive passes):
    cd pyctrl
    python YbScans/StrobeImagingScan.py amps --pulse-time 800e-9 --recool-time 5e-6 --reps 4
    python YbScans/StrobeImagingScan.py x2d  --pulse-time 800e-9 --recool-time 50e-6 --reps 2
    python YbScans/StrobeImagingScan.py beamtest
"""

import argparse
import os
import sys


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _consts():
    from seq_config import SeqConfig
    from consts import Consts
    if not SeqConfig.get().consts:
        SeqConfig.load_real()
    return Consts()


def _set_fixed(g, c, *, pulse_time, recool_time, pushout_time, amp1=None, amp2=None, det_hz=None,
               fix_amps=True, fix_x=True, fix_h=True, x_det=None, x_amp=None, h_det=None, h_amp=None):
    """Pin the always-fixed strobe params (timing, pushout hold, 399 amps/detuning, the non-swept
    recool beam). A param that will be SWEPT must be left unset (ScanGroup refuses to .scan() a
    fixed param). Recool X/h are addressed by FreqDetuning (the step adds Resonance556mj0Freq)."""
    st = g().Imag399.Strobe
    st.BeamPulseTime = float(pulse_time)
    st.RecoolTime = float(recool_time)
    g().Pushout.Time = float(pushout_time)
    if det_hz is not None:
        g().Imag399.FreqDetuning = float(det_hz)
    if fix_amps:
        g().Imag399.Amp1 = float(amp1 if amp1 is not None else c.Imag399.Amp1)
        g().Imag399.Amp2 = float(amp2 if amp2 is not None else c.Imag399.Amp2)
    if fix_x:
        st.Cool556.X.FreqDetuning = float(x_det if x_det is not None else c.Imag399.Strobe.Cool556.X.FreqDetuning)
        st.Cool556.X.Amp = float(x_amp if x_amp is not None else c.Imag399.Strobe.Cool556.X.Amp)
    if fix_h:
        st.Cool556.h.FreqDetuning = float(h_det if h_det is not None else c.Imag399.Strobe.Cool556.h.FreqDetuning)
        st.Cool556.h.Amp = float(h_amp if h_amp is not None else c.Imag399.Strobe.Cool556.h.Amp)


def _runp(g, loading_phase=None, loading_defocus=-5):
    rp = g.runp()
    rp.NumPerGroup = 4000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    if loading_phase:
        rp.loading_phase = loading_phase
        rp.loading_defocus = loading_defocus


def build_amps(a1, a2, *, pulse_time, recool_time, pushout_time, det_hz):
    _bootstrap()
    from scan_group import ScanGroup
    c = _consts(); g = ScanGroup()
    _set_fixed(g, c, pulse_time=pulse_time, recool_time=recool_time, pushout_time=pushout_time,
               det_hz=det_hz, fix_amps=False, fix_x=True, fix_h=True)
    g().Imag399.Amp1.scan(1, [float(a) for a in a1])
    g().Imag399.Amp2.scan(2, [float(a) for a in a2])
    _runp(g)
    return g


def build_cool2d(beam, det_mhz, amps, *, pulse_time, recool_time, pushout_time, amp1, amp2, det_hz,
                 fix_det=None, fix_amp=None):
    """2-D strobe-recool sweep of one beam: FreqDetuning (dim1, from MHz) x Amp (dim2)."""
    _bootstrap()
    from scan_group import ScanGroup
    c = _consts(); g = ScanGroup()
    # pin the OTHER recool beam (at its running optimum if given)
    fx = (dict(x_det=fix_det, x_amp=fix_amp) if beam == "h" else dict(h_det=fix_det, h_amp=fix_amp))
    _set_fixed(g, c, pulse_time=pulse_time, recool_time=recool_time, pushout_time=pushout_time,
               amp1=amp1, amp2=amp2, det_hz=det_hz, fix_x=(beam != "X"), fix_h=(beam != "h"), **fx)
    node = g().Imag399.Strobe.Cool556.X if beam == "X" else g().Imag399.Strobe.Cool556.h
    node.FreqDetuning.scan(1, [float(d) * 1e6 for d in det_mhz])
    node.Amp.scan(2, [float(a) for a in amps])
    _runp(g)
    return g


def build_detuning(dets_mhz, *, pulse_time, recool_time, pushout_time, amp1, amp2):
    _bootstrap()
    from scan_group import ScanGroup
    c = _consts(); g = ScanGroup()
    _set_fixed(g, c, pulse_time=pulse_time, recool_time=recool_time, pushout_time=pushout_time,
               amp1=amp1, amp2=amp2, fix_amps=True, fix_x=True, fix_h=True)
    g().Imag399.FreqDetuning.scan(1, [float(d) * 1e6 for d in dets_mhz])
    _runp(g)
    return g


def build_beamtest(*, pulse_time, recool_time):
    _bootstrap()
    from scan_group import ScanGroup
    g = ScanGroup()
    g().Imag399.Strobe.BeamPulseTime = float(pulse_time)
    g().Imag399.Strobe.RecoolTime = float(recool_time)
    rp = g.runp(); rp.NumImages = 1; rp.Scramble = 0
    return g


def _submit(seqname, g, url, label, reps):
    _bootstrap()
    from yb_start_scan import ybStartScan
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan(seqname, g, url=url, label=label,
                      description=("Strobe imaging optimization (%s). arXiv:2507.01011 scheme, our regime: "
                                   "two StrobeImag399 survival images; fidelity(img1)+survival(img1->img2)."
                                   % label), **opts)
    print("submitted %s -> id %s (url=%s reps=%s nseq=%d)" % (label, did, url or "default", reps, g.nseq()))
    return did


def main():
    _bootstrap()
    from scan_export import matlab_colon
    ap = argparse.ArgumentParser(description="Submit a StrobeImagingScan variant to the pyctrl backend.")
    ap.add_argument("mode", choices=["amps", "x2d", "h2d", "detuning", "beamtest"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=2)
    # strobe timing (FIXED per submission -- the round-runner varies these across submissions)
    ap.add_argument("--pulse-time", type=float, default=1e-6,
                    help="per-beam 399 pulse (s); beam1 then beam2 each get this (period=2*pulse+recool)")
    ap.add_argument("--recool-time", type=float, default=5e-6, help="556 recool window per cycle (s)")
    ap.add_argument("--pushout-time", type=float, default=0.0, help="PushouthX hold (s); 0 = real survival")
    ap.add_argument("--det", type=float, default=-5.0, help="fixed Imag399.FreqDetuning (MHz) for amps/x2d/h2d")
    ap.add_argument("--amp1", type=float, default=0.18, help="fixed Imag399.Amp1 for cool/detuning modes")
    ap.add_argument("--amp2", type=float, default=0.18, help="fixed Imag399.Amp2 for cool/detuning modes")
    # 2-D amps grid
    ap.add_argument("--a1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.30))
    ap.add_argument("--a2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.22))
    # 2-D recool grid
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.26),
                    help="recool FreqDetuning colon in MHz")
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.28),
                    help="recool Amp colon")
    ap.add_argument("--fix-det", type=float, default=None, help="pin non-swept recool beam FreqDetuning (Hz)")
    ap.add_argument("--fix-amp", type=float, default=None, help="pin non-swept recool beam Amp")
    # 1-D detuning grid
    ap.add_argument("--ddet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(-9.0, 0.5, -1.0),
                    help="detuning mode: Imag399.FreqDetuning colon in MHz")
    ap.add_argument("--loading-phase", default=None, help="server-side WGS phase path for this scan")
    args = ap.parse_args()
    det_hz = args.det * 1e6

    if args.mode == "amps":
        g = build_amps(matlab_colon(*args.a1), matlab_colon(*args.a2), pulse_time=args.pulse_time,
                       recool_time=args.recool_time, pushout_time=args.pushout_time, det_hz=det_hz)
        _submit("StrobeImagingPushoutSeq", g, args.url, "StrobeImaging_amps", args.reps)
    elif args.mode in ("x2d", "h2d"):
        beam = "X" if args.mode == "x2d" else "h"
        g = build_cool2d(beam, matlab_colon(*args.fdet), matlab_colon(*args.famp), pulse_time=args.pulse_time,
                         recool_time=args.recool_time, pushout_time=args.pushout_time, amp1=args.amp1,
                         amp2=args.amp2, det_hz=det_hz, fix_det=args.fix_det, fix_amp=args.fix_amp)
        _submit("StrobeImagingPushoutSeq", g, args.url, "StrobeImaging_%s2d" % beam, args.reps)
    elif args.mode == "detuning":
        g = build_detuning(matlab_colon(*args.ddet), pulse_time=args.pulse_time, recool_time=args.recool_time,
                           pushout_time=args.pushout_time, amp1=args.amp1, amp2=args.amp2)
        _submit("StrobeImagingPushoutSeq", g, args.url, "StrobeImaging_detuning", args.reps)
    else:  # beamtest
        g = build_beamtest(pulse_time=args.pulse_time, recool_time=args.recool_time)
        _submit("StrobeBeamTestSeq", g, args.url, "StrobeBeamTest", args.reps)


if __name__ == "__main__":
    main()
