"""cell_pooled_imaging.py -- per-CELL imaging metrics that survive a swept IMAGING condition.

Why not the existing tools: ``cell_analyze.py`` / ``bluelac_round._self_threshold_occ`` derive ONE
per-site threshold from the WHOLE run, which is only valid when imaging is fixed across cells. In a
399 detuning / power sweep the brightness itself changes per cell, so a whole-run threshold makes
dim cells read as empty and the comparison becomes circular. The dashboard ``logicals`` are no help
either on a freshly re-loaded pattern (stale per-pattern thresholds -> loading ~0.01).

Method (per cell, dashboard-independent):
  1. per-site OFFSET = that site's whole-run median img1 intensity (a stable per-site pedestal),
     subtracted so all sites can be pooled;
  2. fit a 2-component Gaussian mixture (EM) to the pooled centered img1 residuals OF THAT CELL
     -> mu_empty/sigma_empty, mu_atom/sigma_atom, atom weight w;
  3. dist = mu_a-mu_e, d' = dist/rms(sigma), analytic Gaussian-overlap fidelity
     = 1 - min_t 0.5*(sf(t;e)+cdf(t;a)) (the runbook's honest estimator, not the optimal-cut);
  4. threshold = the two Gaussians' crossing -> occupancy for img1/img2 -> loading = P(img1),
     survival = P(img2|img1) with binomial SEM.
Loading here is a DETECTION-dependent number (that is the point: a bad imaging condition loses
atoms in the labels), so read it together with d'.

Usage (yb_analysis env):  python tools/cell_pooled_imaging.py <data_dir>
"""
import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
sys.path.insert(0, REPO)

import numpy as np                                                   # noqa: E402
import h5py                                                          # noqa: E402
from scipy.special import ndtr                                       # noqa: E402
from yb_analysis.detection.scan_analysis import extract_scan_dims     # noqa: E402


def em2(v, iters=200):
    """2-component 1-D Gaussian EM. Returns (mu_e, s_e, mu_a, s_a, w_atom)."""
    v = np.asarray(v, float)
    xs = np.sort(v)
    half = max(len(v) // 2, 1)
    mu_e, mu_a = xs[:half].mean(), xs[half:].mean()
    s_e = s_a = max(v.std(), 1e-6)
    w = 0.5
    for _ in range(iters):
        pe = (1 - w) * np.exp(-0.5 * ((v - mu_e) / s_e) ** 2) / (s_e + 1e-12)
        pa = w * np.exp(-0.5 * ((v - mu_a) / s_a) ** 2) / (s_a + 1e-12)
        r = pa / (pe + pa + 1e-30)
        wa = r.sum()
        if wa < 1 or len(v) - wa < 1:
            break
        mu_a2 = (r * v).sum() / wa
        mu_e2 = ((1 - r) * v).sum() / (len(v) - wa)
        s_a = np.sqrt((r * (v - mu_a2) ** 2).sum() / wa) + 1e-6
        s_e = np.sqrt(((1 - r) * (v - mu_e2) ** 2).sum() / (len(v) - wa)) + 1e-6
        conv = abs(mu_a2 - mu_a) + abs(mu_e2 - mu_e) < 1e-6
        mu_e, mu_a, w = mu_e2, mu_a2, wa / len(v)
        if conv:
            break
    if mu_a < mu_e:
        mu_e, mu_a = mu_a, mu_e
        s_e, s_a = s_a, s_e
        w = 1 - w
    return mu_e, s_e, mu_a, s_a, w


def overlap_fidelity(mu_e, s_e, mu_a, s_a):
    """1 - min_t 0.5*(P(empty>t) + P(atom<t)) over a fine grid between the two means."""
    ts = np.linspace(mu_e, mu_a, 2001)
    infid = 0.5 * (ndtr(-(ts - mu_e) / s_e) + ndtr((ts - mu_a) / s_a))
    i = int(np.argmin(infid))
    return 1.0 - float(infid[i]), float(ts[i])


def main(d):
    sid = os.path.basename(d.rstrip("/\\"))
    cfg = json.load(open(os.path.join(d, sid + ".json")))
    dims = extract_scan_dims(cfg)
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    with h5py.File(os.path.join(d, sid + ".h5"), "r") as f:
        seq = f["seq_ids"][:]
        # A NumImages=1 scan (e.g. a loading-plane / defocus stack) stores plain
        # ``intensities``/``logicals``; only multi-image scans get the ``_img1`` suffix.
        # Without this fallback every single-image scan dies on a KeyError here.
        key1 = "intensities_img1" if "intensities_img1" in f else "intensities"
        I1 = f[key1][:].astype(float)
        has2 = "intensities_img2" in f
        I2 = f["intensities_img2"][:].astype(float) if has2 else None
    n = min(len(seq), I1.shape[0])
    seq, I1 = seq[:n], I1[:n]
    if has2:
        I2 = I2[:n]
    off = np.median(I1, axis=0)                      # per-site pedestal (whole run)
    C1 = I1 - off[None, :]
    C2 = (I2 - off[None, :]) if has2 else None

    flat = P[seq - 1] - 1
    ncell = int(np.prod([dd["size"] for dd in dims])) if dims else 1
    if not dims:
        flat = np.zeros_like(flat)

    print("POOLED-SELF-THRESHOLD per-cell  scan=%s  n=%d  cells=%d  sites=%d"
          % (sid, n, ncell, I1.shape[1]))
    print("%5s %8s %7s %7s %9s %9s %9s %7s"
          % ("cell", "dist", "dprime", "fid", "load", "surv", "sem", "nshot"))
    rows = []
    for p in range(ncell):
        r = np.where(flat == p)[0]
        if r.size == 0:
            rows.append(None)
            print("%5d %8s" % (p, "(no shots)"))
            continue
        v = C1[r].ravel()
        mu_e, s_e, mu_a, s_a, w = em2(v)
        dist = mu_a - mu_e
        dp = dist / np.sqrt((s_e ** 2 + s_a ** 2) / 2.0)
        fid, thr = overlap_fidelity(mu_e, s_e, mu_a, s_a)
        o1 = C1[r] > thr
        load = float(o1.mean())
        if has2:
            o2 = C2[r] > thr
            loaded = int(o1.sum())
            joint = int((o1 & o2).sum())
            s = joint / max(loaded, 1)
            sem = float(np.sqrt(max(s * (1 - s), 1e-9) / max(loaded, 1)))
        else:
            s, sem = float("nan"), float("nan")
        rows.append({"p": p, "dist": float(dist), "dprime": float(dp), "fid": float(fid),
                     "load": load, "surv": float(s), "sem": sem, "n": int(r.size)})
        print("%5d %8.2f %7.2f %7.4f %9.3f %9.4f %9.4f %7d"
              % (p, dist, dp, fid, load, s, sem, r.size))
    if dims:
        for dd in dims:
            v = [round(x * (1e-6 if abs(x) > 1e4 else 1), 4) for x in dd["values"]]
            print("axis dim=%s size=%d values=%s" % (dd.get("dim"), dd["size"], v))
    good = [r for r in rows if r]
    if good:
        bs = max(good, key=lambda r: (r["surv"] if r["surv"] == r["surv"] else -1))
        bd = max(good, key=lambda r: r["dprime"])
        print("BEST survival cell=%d %.4f+-%.4f (d'=%.2f load=%.3f)"
              % (bs["p"], bs["surv"], bs["sem"], bs["dprime"], bs["load"]))
        print("BEST d'       cell=%d %.2f (surv=%.4f fid=%.4f load=%.3f)"
              % (bd["p"], bd["dprime"], bd["surv"], bd["fid"], bd["load"]))
        print("PICK_JSON " + json.dumps({"rows": good}))


if __name__ == "__main__":
    main(sys.argv[1])
