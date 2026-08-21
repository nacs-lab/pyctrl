"""ImagingPIDSetScan.py -- 2-D 399 imaging-power optimization over the two PID SETPOINTS.

The imaging-power knob under the PID-servo convention (2026-07-18 ND filters; see the
``gotcha-stale-dds-amps-pid-imaging`` memory) is NOT the DDS amps ``Imag399.Amp1/Amp2`` -- those sit
at 1 and the AOM is optically flat from ~0.5 up. The real knob is the pair of servo setpoints
``BlueMOT.Img1PIDSet`` / ``BlueMOT.Img2PIDSet``, written as volts on ``VImg1PIDSet`` (Dev1/21) and
``VImg2PIDSet`` (Dev1/24) by the ROOT ``BlueMOTStep`` (BlueMOTStep.py:25-26,56-57) and then HELD for
the whole shot. This scan sweeps exactly those two.

Readouts per (Img1, Img2) grid cell, both from the SAME survival pair (``NumImages=2``):
  * fidelity / d' / dist -- from img1 (loaded-vs-empty intensity split); rises with imaging power.
  * survival (img1 -> img2) -- falls as more 399 heats the atom. This CAPS the usable setpoint.
The optimum is max d'/fidelity at acceptable survival (the 07-18 gate: fid >= 0.99, surv >= 0.95).

``Pushout.Time = 0`` by default -> a REAL survival pair with no extra hold, matching the 2026-07-15 /
07-18 campaigns ("0-pushout"); the imaging dose is set by ``Orca.ExposureTime`` (base or ByPattern).

Dated history (expConfig ByPattern comments): 07-15 picked 0.40/0.10 @ 50 ms; 07-18 post-ND re-opt
gave 0.3/0.3 (surv 0.964, fid 0.993, d' 3.84) @ 100 ms; 07-19 in-sequence re-opt adopted 0.5/0.5
(0.3 was underpowered, d' 3.4 -> 4.67). Base config today is 0.57/0.41.

Run (pyctrl backend live at --url):
    cd pyctrl
    python YbScans/ImagingPIDSetScan.py --img1 0 0.1 0.6 --img2 0 0.1 0.6 --reps 4
    python YbScans/ImagingPIDSetScan.py --img1 0.2 0.05 0.5 --img2 0.2 0.05 0.5 --reps 4  # refine
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


def build(img1, img2, *, time_s=0.0, loading_phase="phase/33x33_feedback11.pt",
          amp1=None, amp2=None):
    """2-D sweep: ``BlueMOT.Img1PIDSet`` (dim 1) x ``BlueMOT.Img2PIDSet`` (dim 2), both in volts.

    ``img1`` / ``img2`` are lists of setpoints. ``time_s`` is the PushouthX hold (0 = plain survival
    pair). The 399 line and the 556 imaging cooling are left at the config / ByPattern values.

    ``amp1`` / ``amp2`` override the DDS imaging amps ``Imag399.Amp1`` / ``Amp2`` (default: leave at
    the config value, normally 1). **Set the OTHER beam's amp to 0 when sweeping one beam's
    setpoint** -- otherwise the un-swept beam's light floods the image and flattens dist/d' so the
    swept setpoint looks like a dead knob (seen 2026-08-17: beam-1 setpoint 0..3 V read completely
    flat, pedestal fixed at 199.80 ADU, with Amp2 still at 1).
    """
    _bootstrap()
    from scan_group import ScanGroup

    _consts()          # ensure the real expConfig is loaded (Consts() side effect)
    g = ScanGroup()
    g().Pushout.Time = float(time_s)
    if amp1 is not None:
        g().Imag399.Amp1 = float(amp1)
    if amp2 is not None:
        g().Imag399.Amp2 = float(amp2)

    # The two swept setpoints must be left UNSET as fixed params -- ScanGroup refuses to .scan() a
    # param that already has a fixed value.
    g().BlueMOT.Img1PIDSet.scan(1, [float(v) for v in img1])
    g().BlueMOT.Img2PIDSet.scan(2, [float(v) for v in img2])

    rp = g.runp()
    rp.NumPerGroup = 4000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    if loading_phase:
        rp.loading_phase = loading_phase
    return g


def ImagingPIDSetScan(url=None, reps=None, img1=None, img2=None, time_s=0.0,
                      loading_phase="phase/33x33_feedback11.pt", amp1=None, amp2=None):
    """Build + submit the 2-D PID-setpoint grid. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(img1, img2, time_s=time_s, loading_phase=loading_phase, amp1=amp1, amp2=amp2)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan("ImagingPushoutSurvivalSeq", g, url=url, label="ImagingPIDSetScan", **opts)
    print("submitted ImagingPIDSetScan -> descriptor id %s (url=%s, reps=%s, %dx%d grid, hold %.3g s, "
          "amp1=%s amp2=%s)"
          % (did, url or "default", reps, len(img1), len(img2), time_s,
             "cfg" if amp1 is None else amp1, "cfg" if amp2 is None else amp2))
    return did


if __name__ == "__main__":
    _bootstrap()
    from scan_export import matlab_colon

    ap = argparse.ArgumentParser(description="2-D Img1PIDSet x Img2PIDSet imaging-power scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=4, help="passes over the grid (0 = forever)")
    ap.add_argument("--img1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.0, 0.1, 0.6),
                    help="BlueMOT.Img1PIDSet grid in volts (default 0:0.1:0.6)")
    ap.add_argument("--img2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.0, 0.1, 0.6),
                    help="BlueMOT.Img2PIDSet grid in volts (default 0:0.1:0.6)")
    ap.add_argument("--time", dest="time_s", type=float, default=0.0,
                    help="PushouthX hold in s (default 0 = plain survival pair)")
    ap.add_argument("--loading-phase", default="phase/33x33_feedback11.pt",
                    help="server-side WGS phase path for this scan")
    ap.add_argument("--amp1", type=float, default=None,
                    help="override Imag399.Amp1 (default: config, normally 1). Set 0 to kill beam 1 "
                         "while sweeping beam 2's setpoint.")
    ap.add_argument("--amp2", type=float, default=None,
                    help="override Imag399.Amp2 (default: config, normally 1). Set 0 to kill beam 2 "
                         "while sweeping beam 1's setpoint.")
    args = ap.parse_args()

    v1 = list(matlab_colon(args.img1[0], args.img1[1], args.img1[2]))
    v2 = list(matlab_colon(args.img2[0], args.img2[1], args.img2[2]))
    print("Img1PIDSet grid (%d): %s" % (len(v1), ", ".join("%.3g" % v for v in v1)))
    print("Img2PIDSet grid (%d): %s" % (len(v2), ", ".join("%.3g" % v for v in v2)))
    ImagingPIDSetScan(url=args.url, reps=args.reps, img1=v1, img2=v2, time_s=args.time_s,
                      loading_phase=args.loading_phase, amp1=args.amp1, amp2=args.amp2)
