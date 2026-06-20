"""Spectrum556_2x15Scan.py -- 556 |mj|=1 trap-depth scan on the 2-layer 15x15 array.

NEW scan (does not touch any existing scan/seq). Same recipe as
``Spectrum556Scan.build(mj=1)`` (PushoutSurvivalSeq, NumImages=2, push-out Amp 0.10 / 20 ms),
but pointed at the two-layer ``phase/2x15x15_xyoffset_5um.pt`` array and with a WIDER, LOWER
frequency window.

Why a different window: this array runs at VSLMServo 1.9 with only ~445 sites (vs the 47x47's
3.7 / 2164), so the per-trap power -- hence the trap depth and the |mj|=1 differential light
shift -- is roughly 2x larger. The |mj|=1 dip therefore sits well BELOW the 47x47's
104.2-107.2 MHz window (estimate ~103-105 MHz). We sweep 101.0-106.0 MHz (51 pts @ 100 kHz) to
bracket it robustly while staying ~1.7 MHz clear of the mj=0 line (107.775). If the dip lands
near an edge, re-run with the window shifted (``--lo/--hi``).

Detection uses the 2x15x15 pattern (planes_z_rad declared) so both layers' 445 sites are
measured; the per-site |mj|=1 center -> trap depth (Delta-nu = 2*(f0 - f_site), f0 = today's
mj=0 = 107.7753 MHz). Per-layer CV / cross-layer CV / spatial gradient are computed offline
from the result (split by combined-lattice parity = the two layers).

Run (pyctrl backend live at --url):
    cd pyctrl
    python YbScans/Spectrum556_2x15Scan.py                 # 51 pts, rep 10 (~510 shots)
    python YbScans/Spectrum556_2x15Scan.py --reps 20       # more shots / cleaner per-site fits
    python YbScans/Spectrum556_2x15Scan.py --lo 100.5 --hi 105.5   # shift the window
"""

import argparse
import json
import os
import sys


PHASE_PATH = "phase/2x15x15_xyoffset_5um.pt"
PATTERN_NAME = "2x15x15_xyoffset_5um"
PLANES_Z_RAD = [-0.768, 0.768]
DEFAULT_DEFOCUS = -5.0
DEFAULT_LO, DEFAULT_HI, STEP = 101.0, 106.0, 0.1     # MHz; 51 pts @ 100 kHz


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build(lo=DEFAULT_LO, hi=DEFAULT_HI, defocus=DEFAULT_DEFOCUS):
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # |mj|=1 "check trap depth" push-out recipe (== Spectrum556Scan.build(mj=1)).
    g().Pushout.Green.Amp = 0.1
    g().Pushout.Time = 20e-3

    # Swept push-out frequency (MATLAB-exact float64 colon; the 0.1-MHz step is not integer).
    freqs = [v * 1e6 for v in matlab_colon(lo, STEP, hi)]
    g().Pushout.Green.Freq.scan(1, freqs)

    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # 2-layer array: write it + hold the lock + apply its ByPattern overlay (VSLMServo 1.9 etc).
    rp.loading_phase = PHASE_PATH
    rp.loading_defocus = float(defocus)
    rp.useScanLongSlmLock = 1
    # Detection pattern (both frames): planes -> per-layer 3-D extraction (2-D fallback if the
    # server's 3-D build isn't deployed; either way both layers' 445 sites are detected).
    pat = {"name": PATTERN_NAME, "base_phase_path": PHASE_PATH, "order": "col",
           "legacy_zerniked": False, "planes_z_rad": PLANES_Z_RAD}
    rp.imagePatternsJson = json.dumps([pat, pat])
    return g, len(freqs)


def Spectrum556_2x15Scan(url=None, reps=10, lo=DEFAULT_LO, hi=DEFAULT_HI,
                         defocus=DEFAULT_DEFOCUS):
    _bootstrap()
    from yb_start_scan import ybStartScan
    g, npts = build(lo=lo, hi=hi, defocus=defocus)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "Spectrum556_2x15_mj1"
    did = ybStartScan("PushoutSurvivalSeq", g, url=url, label=label, **opts)
    print("submitted %s -> id %s (%d pts %.1f-%.1f MHz @ %g kHz, reps=%s, defocus=%g)"
          % (label, did, npts, lo, hi, STEP * 1e3, reps, defocus))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 2x15x15 556 |mj|=1 trap-depth scan.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=10,
                    help="passes over the sweep (0 = forever); default 10 (~510 shots)")
    ap.add_argument("--lo", type=float, default=DEFAULT_LO, help="sweep start (MHz)")
    ap.add_argument("--hi", type=float, default=DEFAULT_HI, help="sweep end (MHz)")
    ap.add_argument("--defocus", type=float, default=DEFAULT_DEFOCUS)
    args = ap.parse_args()
    Spectrum556_2x15Scan(url=args.url, reps=args.reps, lo=args.lo, hi=args.hi,
                         defocus=args.defocus)
