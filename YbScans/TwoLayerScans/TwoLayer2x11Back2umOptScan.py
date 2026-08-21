"""TwoLayer2x11Back2umOptScan.py -- imaging optimization campaign for the back-2um bifocal array.

2026-07-14 campaign: after the pattern moved to 50 ms exposure + the PID-servo imaging scheme
(BlueMOT.Img1/Img2PIDSet, DDS Imag399.Amp1/Amp2 = 1), jointly optimize for BOTH layers at the
layer-focus crossover defocus -6.5:

  --mode setpoints   2-D Img1PIDSet x Img2PIDSet map at 0-pushout (hold 1 ms) = the real 50 ms
                     two-image condition. The "amp scan" of the runbook under the new scheme.
  --mode cool        2-D Imag399.Cool556.<beam>.{FreqDetuning x Amp} at 0-pushout, --beam X|h;
                     pins the other beam and the setpoints via fixed g() leaves.
  --mode confirm     single-point high-rep run at the current working set (per-site stats).

All swept/pinned values ride as g() overrides -- expConfig ByPattern["2x11x11_5um_back2um"]
holds the committed seed (Img1 0.7 / Img2 0.25, cooling copied from 33x33_feedback9) until the
campaign locks in. Detection = the 242-site no-dedup grid (see TwoLayer2x11Back2umImagingScan).

Run (queue-safe: just submits; poll your job id -- other scans may interleave):
    cd pyctrl
    python YbScans/TwoLayer2x11Back2umOptScan.py --mode setpoints --reps 15
    python YbScans/TwoLayer2x11Back2umOptScan.py --mode cool --beam X --img1 W1 --img2 W2 --reps 15
    python YbScans/TwoLayer2x11Back2umOptScan.py --mode confirm --img1 W1 --img2 W2 --reps 150
"""

import argparse
import json
import os
import sys


PHASE_PATH = "phase/2x11x11_5um_back2um.pt"
PATTERN_NAME = "2x11x11_5um_back2um"
PLANES_Z_RAD = [-2.7778, 2.7778]
DEFAULT_DEFOCUS = -6.5      # measured layer-focus crossover (2026-07-14), NOT the geometric -5
HOLD = 0.001                # 0-pushout = the real two-image condition


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _colon(lo, step, hi):
    from scan_export import matlab_colon
    return list(matlab_colon(lo, step, hi))


def build(args):
    _bootstrap()
    from scan_group import ScanGroup
    from seq_config import SeqConfig
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    g = ScanGroup()
    rp = g.runp()
    rp.loading_phase = PHASE_PATH
    rp.loading_defocus = float(args.defocus)
    rp.useScanLongSlmLock = 1
    pat = {"name": PATTERN_NAME, "base_phase_path": PHASE_PATH,
           "order": "col_up", "legacy_zerniked": False, "planes_z_rad": PLANES_Z_RAD}
    # order MUST stay "col_up" (the rearrange-era convention): a different order mismatches the
    # seeded 242-site registry record -> fetch_or_refresh RE-DERIVES via the server, whose default
    # cross-plane dedup (radius 6) MERGES the 4.6-knm-apart layer pairs to 121 sites (seen live
    # 2026-07-15 run 070945: 121-site grid, fake union-survival 0.97).
    rp.imagePatternsJson = json.dumps([pat, pat])
    rp.NumImages = 2
    rp.isInit = 0
    rp.Scramble = 1
    rp.isHC = 0
    rp.isGrid2 = 0

    g().Pushout.Time = HOLD

    # pinned working-set members (fixed leaves; None -> pattern/base value)
    if args.img1 is not None and args.mode != "setpoints":
        g().BlueMOT.Img1PIDSet = float(args.img1)
    if args.img2 is not None and args.mode != "setpoints":
        g().BlueMOT.Img2PIDSet = float(args.img2)
    if args.xcool is not None:
        g().Imag399.Cool556.X.FreqDetuning = float(args.xcool[0]) * 1e6
        g().Imag399.Cool556.X.Amp = float(args.xcool[1])
    if args.hcool is not None:
        g().Imag399.Cool556.h.FreqDetuning = float(args.hcool[0]) * 1e6
        g().Imag399.Cool556.h.Amp = float(args.hcool[1])

    axes = []
    if args.mode == "setpoints":
        v1 = _colon(*args.img1_grid)
        v2 = _colon(*args.img2_grid)
        g().BlueMOT.Img1PIDSet.scan(1, v1)
        g().BlueMOT.Img2PIDSet.scan(2, v2)
        axes = [("Img1PIDSet", v1), ("Img2PIDSet", v2)]
    elif args.mode == "cool":
        det = [v * 1e6 for v in _colon(*args.cdet)]
        amp = _colon(*args.famp)
        beam = args.beam
        getattr(g().Imag399.Cool556, beam).FreqDetuning.scan(1, det)
        getattr(g().Imag399.Cool556, beam).Amp.scan(2, amp)
        axes = [("Cool556.%s.det" % beam, det), ("Cool556.%s.amp" % beam, amp)]
    elif args.mode == "confirm":
        pass
    else:
        raise ValueError(args.mode)

    n_points = 1
    for _, v in axes:
        n_points *= len(v)
    rp.NumPerGroup = int(args.reps) * n_points
    return g, axes, n_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--mode", choices=["setpoints", "cool", "confirm"], required=True)
    ap.add_argument("--defocus", type=float, default=DEFAULT_DEFOCUS)
    ap.add_argument("--reps", type=int, default=15)
    # setpoint grids (lo step hi)
    ap.add_argument("--img1-grid", type=float, nargs=3, default=[0.4, 0.225, 1.3])
    ap.add_argument("--img2-grid", type=float, nargs=3, default=[0.15, 0.1, 0.45])
    # cooling grids (det in MHz)
    ap.add_argument("--beam", choices=["X", "h"], default="X")
    ap.add_argument("--cdet", type=float, nargs=3, default=[0.10, 0.03, 0.24])
    ap.add_argument("--famp", type=float, nargs=3, default=[0.10, 0.05, 0.30])
    # pinned working set
    ap.add_argument("--img1", type=float, default=None)
    ap.add_argument("--img2", type=float, default=None)
    ap.add_argument("--xcool", type=float, nargs=2, default=None, metavar=("DET_MHZ", "AMP"))
    ap.add_argument("--hcool", type=float, nargs=2, default=None, metavar=("DET_MHZ", "AMP"))
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    _bootstrap()
    from yb_start_scan import ybStartScan
    g, axes, n_points = build(args)
    desc = args.description or (
        "2x11x11_5um_back2um imaging optimization (%s) at 50 ms exposure + PID-servo scheme, "
        "defocus %g (layer-focus crossover; both layers read from one image, 242-site no-dedup "
        "grid). Axes: %s. Pins: img1=%s img2=%s xcool=%s hcool=%s. 0-pushout (hold %g s) = real "
        "two-image condition. Campaign 2026-07-14 (prior state at 35 ms/legacy DDS amps: fid "
        "~0.92, d' ~2.8, surv ~0.64)."
        % (args.mode, args.defocus,
           [(n, len(v)) for n, v in axes] or "single point",
           args.img1, args.img2, args.xcool, args.hcool, HOLD))
    label = "B2um_%s%s_z%g" % (args.mode,
                               "_%s" % args.beam if args.mode == "cool" else "",
                               args.defocus)
    did = ybStartScan("ImagingPushoutSurvivalSeq", g, url=args.url, rep=int(args.reps),
                      label=label, description=desc)
    print("[B2umOpt] submitted id=%s mode=%s points=%d reps=%d total=%d"
          % (did, args.mode, n_points, args.reps, int(args.reps) * n_points))
    for n, v in axes:
        print("  axis %s: %s" % (n, ["%.3g" % x for x in v]))
    return did


if __name__ == "__main__":
    main()
