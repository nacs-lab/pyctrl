"""rearrange_control_survival.py -- same-grid conditional survival for an all-stays control run
(INIT=MIDDLE=TARGET), split by scan point. Frames all detect on the SAME pattern grid, so
P(img_{k+1} | img_k) per window is the pure context loss (hold + compute + playback + imaging).

Usage: python tools/rearrange_control_survival.py --scan 20260719_151525 [--pattern tri_3013_camfb]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--pattern", default="tri_3013_camfb")
    a = ap.parse_args()

    from rearrange2_ground_truth import pattern_grid
    from reanchor_pattern_thresholds import fspecial_gaussian
    from yb_analysis.detection.detect_atom import detect_atom_batch
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params

    day = a.scan.split("_")[0]
    dd = os.path.join(DATA_ROOT, day, "data_" + a.scan)
    sid = os.path.basename(dd)
    cfg = json.load(open(os.path.join(dd, sid + ".json")))
    roi = cfg["roi"]
    nimg = int(cfg["NumImages"])
    pts = np.array(cfg.get("Params") or [], dtype=int)
    imgs = []
    for p in sorted(glob.glob(os.path.join(dd, "*.h5"))):
        with h5py.File(p, "r") as f:
            if "imgs" in f:
                imgs.append(f["imgs"][:])
    imgs = np.concatenate(imgs, 0).astype(np.float64)
    n = imgs.shape[0] // nimg
    pts = pts[:n] if pts.size else np.ones(n, dtype=int)
    mask = fspecial_gaussian(9, 2)
    g = pattern_grid(a.pattern, roi)
    L = []
    for k in range(nimg):
        _, I = detect_atom_batch(imgs[k::nimg], g, np.zeros(g.shape[0]), mask)
        thr, _, params = _fit_run_site_params(I)
        thr = np.asarray(thr, float)
        ok = np.array([pp is not None for pp in params])
        thr = np.where(np.isfinite(thr) & ok, thr, np.nanmedian(thr[ok]))
        L.append(I > thr[None, :])
    print("=" * 70)
    print("CONTROL SURVIVAL %s  %d shots x %d frames  pattern=%s" % (sid, n, nimg, a.pattern))
    print("point   n   load_med  " + "  ".join(
        "P(i%d|i%d)" % (k + 2, k + 1) for k in range(nimg - 1)) + "   P(i%d|i1)" % nimg)
    for pt in sorted(set(pts)):
        m = pts == pt
        row = ["%5d  %3d   %6.0f " % (pt, m.sum(), np.median(L[0][m].sum(1)))]
        for k in range(nimg - 1):
            num = (L[k][m] & L[k + 1][m]).sum()
            den = L[k][m].sum()
            row.append("  %.4f " % (num / den))
        row.append("   %.4f" % ((L[0][m] & L[-1][m]).sum() / L[0][m].sum()))
        print("".join(row))
    # all points pooled
    row = ["  ALL  %3d   %6.0f " % (n, np.median(L[0].sum(1)))]
    for k in range(nimg - 1):
        row.append("  %.4f " % ((L[k] & L[k + 1]).sum() / L[k].sum()))
    row.append("   %.4f" % ((L[0] & L[-1]).sum() / L[0].sum()))
    print("".join(row))
    print("=" * 70)


if __name__ == "__main__":
    main()
