"""stark_vx_fit.py -- fit the 2D Vx x 616-EOM revival (StarkVxRevival616Scan).

Per Vx group: fit the 308 revival (survival vs 616-EOM freq) to a single Lorentzian PEAK ->
center(Vx). Then fit center-vs-Vx to a parabola; the vertex = the Vx that nulls the residual DC
E-field along x (minimizes the DC Stark shift). Mirrors the FreqPushOut308Scan analysis plots.

Run under the yb_analysis env from the project root:
  python pyctrl/tools/stark_vx_fit.py <scan_id>
Saves stark_vx_revival_<fid>.png (per-Vx revival overlay) + stark_vx_parabola_<fid>.png
(center vs Vx + parabola + vertex) into the scan dir; prints the per-Vx centers + the vertex.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _scan_dir(scan_id):
    sys.path.insert(0, REPO)
    from yb_analysis import config
    fid = scan_id if "_" in scan_id else "%s_%s" % (scan_id[:8], scan_id[8:])
    return os.path.join(config.DATA_DIR, fid[:8], "data_" + fid), fid


def lor_peak(x, y0, A, x0, w):
    return y0 + A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_id")
    a = ap.parse_args()
    dd, fid = _scan_dir(a.scan_id)
    sid = os.path.basename(dd)
    if not os.path.isdir(dd):
        print("no scan dir:", dd); sys.exit(2)

    sys.path.insert(0, REPO)
    from yb_analysis.detection.scan_analysis import extract_scan_dims
    cfg = json.load(open(os.path.join(dd, sid + ".json")))
    dims = extract_scan_dims(cfg)
    # dim1 = EOM freq (Hz), dim2 = Vx (V)
    eom = np.asarray(dims[0]["values"], float)            # Hz
    vx = np.asarray(dims[1]["values"], float)             # V
    s0, s1 = dims[0]["size"], dims[1]["size"]
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    with h5py.File(os.path.join(dd, sid + ".h5"), "r") as f:
        seq_ids = f["seq_ids"][:]
        L1 = f["logicals_img1"][:].astype(bool)
        L2 = f["logicals_img2"][:].astype(bool)
    n = min(len(seq_ids), L1.shape[0]); seq_ids, L1, L2 = seq_ids[:n], L1[:n], L2[:n]
    flat = P[seq_ids - 1] - 1                              # 0-based cell index
    kw_axis = str(dims[1].get("name", "Vx")).split(".")[-1]   # swept electrode axis (Vx|Vy|Vz)

    # array-mean survival per (eom, vx) cell. cell index = j*s0 + i (i=dim1 eom, j=dim2 vx)
    ncell = s0 * s1
    surv = np.full(ncell, np.nan); sem = np.full(ncell, np.nan)
    for p in range(ncell):
        rows = np.where(flat == p)[0]
        if rows.size == 0:
            continue
        l1 = L1[rows]; l2 = L2[rows]
        loaded = l1.sum(axis=0); joint = (l1 & l2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p11 = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)
        surv[p] = np.nanmean(p11)
        sem[p] = np.nanstd(p11) / np.sqrt(np.isfinite(p11).sum())
    S = surv.reshape(s1, s0)                               # [vx, eom]
    E = sem.reshape(s1, s0)
    eom_mhz = eom / 1e6

    centers = []; cerr = []
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, s1))
    for j in range(s1):
        y = S[j]; ye = E[j]; m = np.isfinite(y)
        x0g = eom_mhz[m][np.argmax(y[m])]
        try:
            p, pc = curve_fit(lor_peak, eom_mhz[m], y[m], sigma=np.where(ye[m] > 0, ye[m], 0.05)[:, ],
                              p0=[np.nanmin(y), np.nanmax(y) - np.nanmin(y), x0g, 8],
                              bounds=([0, 0, eom_mhz.min(), 1], [1, 1.5, eom_mhz.max(), 40]), maxfev=8000)
            c = p[2]; ce = float(np.sqrt(np.diag(pc))[2])
        except Exception:
            c, ce = x0g, np.nan
        centers.append(c); cerr.append(ce)
        ax1.errorbar(eom_mhz, y, yerr=ye, fmt="o", ms=3, color=cmap[j], alpha=0.7,
                     label="%s=%+.3f V  c=%.2f MHz" % (kw_axis, vx[j], c))
        xf = np.linspace(eom_mhz.min(), eom_mhz.max(), 400)
        ax1.plot(xf, lor_peak(xf, *p), "-", color=cmap[j], lw=1.2)
        ax1.axvline(c, color=cmap[j], ls=":", lw=0.7)
    ax1.set_xlabel("616-EOM freq [MHz]"); ax1.set_ylabel("survival (P11)")
    ax1.set_title("Stark%sRevival616Scan %s -- revival vs %s" % (kw_axis, sid, kw_axis)); ax1.legend(fontsize=7)
    fig1.tight_layout()
    out1 = os.path.join(dd, "stark_%s_revival_%s.png" % (kw_axis.lower(), fid))
    fig1.savefig(out1, dpi=110); plt.close(fig1)
    centers = np.array(centers); cerr = np.array(cerr)

    # parabola center(Vx) = a*Vx^2 + b*Vx + c ; vertex = -b/2a
    def quad(x, a, b, c):
        return a * x * x + b * x + c
    w = np.where(np.isfinite(cerr) & (cerr > 0), 1.0 / cerr, 1.0)
    pq, pcq = curve_fit(quad, vx, centers, sigma=1.0 / w, absolute_sigma=False)
    aa, bb, cc = pq
    vertex = -bb / (2 * aa)
    ss_res = np.sum((centers - quad(vx, *pq)) ** 2)
    ss_tot = np.sum((centers - centers.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(vx, centers, yerr=np.where(np.isfinite(cerr), cerr, 0), fmt="ko", ms=6, capsize=3,
                 label="fitted centers")
    xf = np.linspace(vx.min(), vx.max(), 300)
    ax2.plot(xf, quad(xf, *pq), "r-", lw=1.5, label="parabola R2=%.4f" % r2)
    ax2.axvline(vertex, color="gray", ls=":", lw=1, label="vertex %s=%.4f V" % (kw_axis, vertex))
    ax2.set_xlabel("%s (V)" % kw_axis); ax2.set_ylabel("fitted 616-EOM center (MHz)")
    ax2.set_title("Stark%sRevival616Scan -- center vs %s\nvertex (E-field null) %s = %.4f V" % (kw_axis, kw_axis, kw_axis, vertex))
    ax2.legend(fontsize=8); fig2.tight_layout()
    out2 = os.path.join(dd, "stark_%s_parabola_%s.png" % (kw_axis.lower(), fid))
    fig2.savefig(out2, dpi=110); plt.close(fig2)

    print("StarkVxRevival616Scan %s | %d shots" % (sid, n))
    for j in range(s1):
        print("  %s=%+.4f V -> center %.3f MHz  (err %.3f)"
              % (kw_axis, vx[j], centers[j], (cerr[j] if np.isfinite(cerr[j]) else float("nan"))))
    print("  parabola: a=%.4g MHz/V^2  b=%.4g MHz/V  R2=%.4f" % (aa, bb, r2))
    print("  VERTEX (E-field-null %s) = %.4f V  (min-Stark center %.3f MHz)" % (kw_axis, vertex, quad(vertex, *pq)))
    print("  saved %s" % out1)
    print("  saved %s" % out2)
    print("STARK_JSON " + json.dumps({"scan_id": fid, "n_shots": int(n), "axis": kw_axis,
          "v": vx.tolist(), "centers_MHz": centers.tolist(), "vertex_V": float(vertex),
          "parab_a_MHz_per_V2": float(aa), "parab_b_MHz_per_V": float(bb), "r2": float(r2),
          "revival_png": out1, "parabola_png": out2}))


if __name__ == "__main__":
    main()
