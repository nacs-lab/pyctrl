"""rearrange2_ground_truth.py -- GMM-A ground-truth analysis of a multi-frame rearrangement run.

Detects EACH camera frame on its OWN pattern's registry grid (knm -> global affine -> ROI crop),
fits per-site double Gaussians from the run's own intensities (contamination-proof), and reports:
  frame0 (loading grid): loading count/fraction
  frame1 (middle grid):  fill fraction of middle-target sites (round-1 outcome)
  frame2 (final grid):   fill fraction of final-target sites (the headline 2-round number)
plus per-frame d'/fidelity medians. For a 2-frame (1-round) run, frame1 is the final grid.

Usage:
    python tools/rearrange2_ground_truth.py --dir "<data_dir>" [--box 13 --sigma 3.5]
    python tools/rearrange2_ground_truth.py --scan 20260719_042510
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

import numpy as np          # noqa: E402
import h5py                 # noqa: E402
from scipy.stats import norm  # noqa: E402


def gauss_mask(box, sigma):
    ax = np.arange(box) - (box - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    m = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return m / m.sum()


def pattern_grid(name, roi):
    """Registry knm coords -> camera px inside the ROI crop, [y, x] per row."""
    from yb_analysis import config as _cfg
    from yb_analysis.analysis.affine_transform import load_matrix, apply_affine_cropped
    rec = json.load(open(os.path.join(_cfg.PATH_PREFIX, "yb_dashboard_state",
                                      "patterns", name, "record.json")))
    knm = np.asarray(rec["knm"], dtype=float)             # (N,2) [row(y), col(x)] knm
    A = load_matrix()
    return apply_affine_cropped(knm[:, ::-1], A, roi)     # -> cropped-frame [Y,X]


def frame_intensities(imgs, grid, mask):
    from yb_analysis.detection.detect_atom import detect_atom_batch
    thr = np.zeros(grid.shape[0])
    _, inten = detect_atom_batch(imgs, grid, thr, mask)
    return inten


def fit_frame(I):
    """Per-site GMM-A fit on this frame group's own intensities -> (thr, params)."""
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params
    thr, _infid, params = _fit_run_site_params(I)
    return np.asarray(thr, dtype=float), params


def site_stats(params):
    fid, dp = [], []
    for pp in params:
        if pp is None:
            continue
        mu_e, s_e, _, mu_a, s_a, _ = pp
        if not (s_e > 0 and s_a > 0) or mu_a <= mu_e:
            continue
        ts = np.linspace(mu_e - 3 * s_e, mu_a + 3 * s_a, 400)
        infid = np.min(0.5 * (norm.sf(ts, mu_e, s_e) + norm.cdf(ts, mu_a, s_a)))
        fid.append(1.0 - infid)
        dp.append((mu_a - mu_e) / np.sqrt((s_e ** 2 + s_a ** 2) / 2))
    return np.array(fid), np.array(dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--scan")
    ap.add_argument("--box", type=int, default=13)
    ap.add_argument("--sigma", type=float, default=3.5)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    if a.dir:
        data_dir = a.dir
    else:
        day = a.scan.split("_")[0]
        data_dir = os.path.join(DATA_ROOT, day, "data_" + a.scan)

    sid = os.path.basename(data_dir.rstrip("/\\"))
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    nimg = int(cfg["NumImages"])
    roi = cfg["roi"]
    pats = [it["name"] for it in json.loads(cfg["imagePatternsJson"])] \
        if isinstance(cfg.get("imagePatternsJson"), str) else \
        [it["name"] for it in cfg.get("imagePatternsJson", [])]
    if len(pats) < nimg:
        pats = pats + [pats[-1]] * (nimg - len(pats))
    pats = pats[:nimg]

    h5s = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    imgs = []
    for p in h5s:
        with h5py.File(p, "r") as f:
            if "imgs" in f:
                imgs.append(f["imgs"][:])
    imgs = np.concatenate(imgs, axis=0)
    n_shots = imgs.shape[0] // nimg
    imgs = imgs[: n_shots * nimg].astype(np.float64)
    print("=" * 78)
    print("REARRANGE GROUND TRUTH  %s   %d shots x %d frames   patterns=%s"
          % (sid, n_shots, nimg, pats))
    mask = gauss_mask(a.box, a.sigma)

    out = {"scan": sid, "n_shots": n_shots, "frames": []}
    fills = []
    for k in range(nimg):
        g = pattern_grid(pats[k], roi)
        I = frame_intensities(imgs[k::nimg], g, mask)     # (n_shots, M_k)
        thr, params = fit_frame(I)
        okfit = np.array([pp is not None for pp in params])
        # sites with a failed fit: fall back to the median threshold of fitted sites
        thr_eff = np.where(np.isfinite(thr) & okfit, thr, np.nanmedian(thr[okfit]) if okfit.any() else np.nan)
        L = I > thr_eff[None, :]
        fill = L.mean()
        per_shot = L.mean(axis=1)
        fid, dp = site_stats(params)
        fills.append((per_shot, L))
        print("  frame%d %-20s M=%4d  fit_ok=%4d   FILL mean %.4f  median %.4f  best %.4f"
              % (k, pats[k], g.shape[0], okfit.sum(), per_shot.mean(),
                 np.median(per_shot), per_shot.max()))
        print("          d' median %.2f   fidelity median %.5f" %
              (np.median(dp) if dp.size else float("nan"),
               np.median(fid) if fid.size else float("nan")))
        out["frames"].append({"pattern": pats[k], "M": int(g.shape[0]),
                              "fill_mean": float(per_shot.mean()),
                              "fill_median": float(np.median(per_shot)),
                              "fill_best": float(per_shot.max()),
                              "fill_per_shot": [float(x) for x in per_shot],
                              "dprime_median": float(np.median(dp)) if dp.size else None,
                              "fid_median": float(np.median(fid)) if fid.size else None})
    # loading-conditioned final fill: exclude shots where frame-0 loading was below the final target
    if nimg >= 2:
        load_cnt = fills[0][1].sum(axis=1)
        M_final = fills[-1][1].shape[1]
        good = load_cnt >= M_final
        pf = fills[-1][0]
        if good.any():
            print("  final fill | load>=%d atoms: mean %.4f  median %.4f  (n=%d/%d shots)"
                  % (M_final, pf[good].mean(), np.median(pf[good]), good.sum(), n_shots))
            out["final_fill_loadcond"] = float(pf[good].mean())
    print("=" * 78)
    if a.json_out:
        json.dump(out, open(a.json_out, "w"))
    return out


if __name__ == "__main__":
    main()
