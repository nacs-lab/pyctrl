"""rearrange2_runtime_fill.py -- RUNTIME-basis per-frame fill for a 2-round kagome run,
split by scan point. Uses each pattern's registry threshold.mat (the exact basis the runtime
detects with -- validated 2026-07-19 against the final-frame arbiter), on the registry grid,
box 9 / sigma 2.

Usage: python tools/rearrange2_runtime_fill.py --scan 20260719_HHMMSS [--labels 20,30,40,50,70]
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
    ap.add_argument("--labels", default=None, help="comma list naming scan points 1..N")
    ap.add_argument("--axis1", default=None, help="comma list: axis-1 values (2D, column-major)")
    ap.add_argument("--axis2", default=None, help="comma list: axis-2 values (2D)")
    ap.add_argument("--norm", action="store_true",
                    help="per-shot common-mode gain normalization to the registry calibration "
                         "(REQUIRED when a frame was DDS-attenuated; mirrors the runtime flag)")
    a = ap.parse_args()

    from rearrange2_ground_truth import pattern_grid
    from reanchor_pattern_thresholds import fspecial_gaussian
    from yb_analysis.detection.detect_atom import detect_atom_batch
    import yb_analysis.analysis.pattern_registry as reg

    day = a.scan.split("_")[0]
    dd = os.path.join(DATA_ROOT, day, "data_" + a.scan)
    sid = os.path.basename(dd)
    cfg = json.load(open(os.path.join(dd, sid + ".json")))
    roi = cfg["roi"]
    nimg = int(cfg["NumImages"])
    pats = [it["name"] for it in (json.loads(cfg["imagePatternsJson"])
                                  if isinstance(cfg.get("imagePatternsJson"), str)
                                  else cfg.get("imagePatternsJson", []))][:nimg]
    imgs = []
    seqids = []
    for p in sorted(glob.glob(os.path.join(dd, "*.h5"))):
        with h5py.File(p, "r") as f:
            if "imgs" in f:
                imgs.append(f["imgs"][:])
            if "seq_ids" in f:
                seqids.append(f["seq_ids"][:])
    imgs = np.concatenate(imgs, 0).astype(np.float64)
    seqids = np.concatenate(seqids, 0)
    n = imgs.shape[0] // nimg
    pts_all = np.array(cfg.get("Params") or [], dtype=int)
    pts = (np.array([pts_all[s - 1] for s in seqids[:n]])
           if pts_all.size else np.ones(n, dtype=int))
    labels = a.labels.split(",") if a.labels else None
    mask = fspecial_gaussian(9, 2)

    fills = []
    for k in range(nimg):
        g = pattern_grid(pats[k], roi)
        td = reg.load_pattern_thresholds(pats[k])
        thr = np.asarray(td["thresholds"], float)
        _, I = detect_atom_batch(imgs[k::nimg], g, np.zeros(g.shape[0]), mask)
        if a.norm and td.get("gauss_fits"):
            gp = td["gauss_fits"]
            mu_e = np.array([(p["params"][0] if p and len(p.get("params", [])) == 6 else np.nan)
                             for p in gp])
            mu_a = np.array([(p["params"][3] if p and len(p.get("params", [])) == 6 else np.nan)
                             for p in gp])
            good = np.isfinite(mu_e) & np.isfinite(mu_a) & ((mu_a - mu_e) > 0.5)
            ref = (mu_a - mu_e)[good].mean()
            base = np.nan_to_num(mu_e)
            for si in range(I.shape[0]):
                sig = I[si] - base
                # bootstrap loaded set with a relaxed cut (attenuated frame sits below thr)
                sel = good & (sig > 0.25 * ref)
                if sel.sum() >= 30 and sig[sel].mean() > 0:
                    I[si] = base + sig * (ref / sig[sel].mean())
        fills.append((I > thr[None, :]).mean(1))
    print("=" * 78)
    print("RUNTIME-BASIS FILL  %s  %d shots  patterns=%s" % (sid, n, pats))
    ax1 = [x for x in (a.axis1 or "").split(",") if x]
    ax2 = [x for x in (a.axis2 or "").split(",") if x]
    hdr = "point  n  " + "  ".join("%12s" % ("f%d(med/p10)" % k) for k in range(nimg))
    print(hdr)
    for p in sorted(set(pts)):
        m = pts == p
        if ax1 and ax2:
            lab = "%s/%s" % (ax1[(p - 1) % len(ax1)], ax2[(p - 1) // len(ax1)])
        else:
            lab = labels[p - 1] if labels and p - 1 < len(labels) else str(p)
        cells = ["%.4f/%.3f" % (np.median(f[m]), np.percentile(f[m], 10)) for f in fills]
        print("%9s %3d  " % (lab, m.sum()) + "  ".join("%12s" % c for c in cells))
    cells = ["%.4f/%.3f" % (np.median(f), np.percentile(f, 10)) for f in fills]
    print("  ALL %3d  " % n + "  ".join("%12s" % c for c in cells))
    print("=" * 78)


if __name__ == "__main__":
    main()
