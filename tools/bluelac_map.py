"""bluelac_map.py -- pooled per-point loading map for the in-context BlueLAC 2D sweep.

Pools shots from one or more runs (identical 7x3 det x amp grids), detects frame-0 loading on
the tri_3013_camfb registry grid (GMM-A, box 9 / sigma 2), and prints load atoms per point:
median / p10 / mean, plus mid-frame fill for the same shots (free, in-context).

Point index mapping is COLUMN-MAJOR over (axis1=det 7, axis2=amp 3): det_idx=(p-1)%7,
amp_idx=(p-1)//7  (gotcha-2d-scan-reshape-column-major).

Usage: python tools/bluelac_map.py --scans 20260719_154331 20260719_HHMMSS
"""
import argparse
import glob
import json
import os
import sys

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
DATA_ROOT = r"D:\OneDrive - Harvard University\Documents - Yb\Data"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "pyctrl", "tools"))

import numpy as np      # noqa: E402
import h5py             # noqa: E402

DETS = [-5.5, -5, -4.5, -4, -3.8, -3.4, -3]
AMPS = [0.13, 0.17, 0.21]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--dets", type=float, nargs="+", default=None)
    ap.add_argument("--amps", type=float, nargs="+", default=None)
    a = ap.parse_args()
    global DETS, AMPS
    if a.dets:
        DETS = a.dets
    if a.amps:
        AMPS = a.amps

    from rearrange2_ground_truth import pattern_grid
    from reanchor_pattern_thresholds import fspecial_gaussian
    from yb_analysis.detection.detect_atom import detect_atom_batch
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params

    mask = fspecial_gaussian(9, 2)
    all_load = []
    all_mid = []
    all_pts = []
    gload = gmid = None
    for scan in a.scans:
        day = scan.split("_")[0]
        dd = os.path.join(DATA_ROOT, day, "data_" + scan)
        sid = os.path.basename(dd)
        cfg = json.load(open(os.path.join(dd, sid + ".json")))
        roi = cfg["roi"]
        nimg = int(cfg["NumImages"])
        imgs = []
        for p in sorted(glob.glob(os.path.join(dd, "*.h5"))):
            with h5py.File(p, "r") as f:
                if "imgs" in f:
                    imgs.append(f["imgs"][:])
        imgs = np.concatenate(imgs, 0).astype(np.float64)
        n = imgs.shape[0] // nimg
        pts = np.array(cfg["Params"], dtype=int)[:n]
        if gload is None:
            gload = pattern_grid("tri_3013_camfb", roi)
            gmid = pattern_grid("kagome_res_2198", roi)
        _, I0 = detect_atom_batch(imgs[0::nimg], gload, np.zeros(gload.shape[0]), mask)
        _, I1 = detect_atom_batch(imgs[1::nimg], gmid, np.zeros(gmid.shape[0]), mask)
        all_load.append(I0)
        all_mid.append(I1)
        all_pts.append(pts)
        print("loaded %s: %d shots" % (scan, n))
    I0 = np.concatenate(all_load, 0)
    I1 = np.concatenate(all_mid, 0)
    pts = np.concatenate(all_pts, 0)

    def logic(I):
        thr, _, params = _fit_run_site_params(I)
        thr = np.asarray(thr, float)
        ok = np.array([pp is not None for pp in params])
        thr = np.where(np.isfinite(thr) & ok, thr, np.nanmedian(thr[ok]))
        return I > thr[None, :]
    L0 = logic(I0)
    L1 = logic(I1)
    load = L0.sum(1)
    midfill = L1.mean(1)

    print("\n det(MHz) amp    n   load_med  load_p10  load_mean   midfill_med")
    best = None
    for p in sorted(set(pts)):
        m = pts == p
        di, ai = (p - 1) % len(DETS), (p - 1) // len(DETS)
        row = (DETS[di], AMPS[ai], m.sum(), np.median(load[m]),
               np.percentile(load[m], 10), load[m].mean(), np.median(midfill[m]))
        print(" %5.1f   %.2f  %3d   %6.0f    %6.0f    %6.0f      %.4f" % row)
        if best is None or row[3] > best[3]:
            best = row
    print("\nBEST by load_med: det %.1f MHz amp %.2f -> med %.0f p10 %.0f" %
          (best[0], best[1], best[3], best[4]))


if __name__ == "__main__":
    main()
