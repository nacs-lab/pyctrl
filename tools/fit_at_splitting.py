"""fit_at_splitting.py -- array-averaged Autler-Townes splitting from a 556AutlerTownesScan.

Fits a DOUBLE Lorentzian dip to the ARRAY-AVERAGE survival(P11) vs 556 freq and reports the
two dip centers + the AT splitting (= the 308 coupling Rabi -> a beam-alignment / polarization
optimization metric: maximize the splitting). Saves a fit plot next to the scan.

Why not fit_spectrum.py --peaks 2: its auto-seed straddles the GLOBAL minimum, which fails when
the AT doublet is wide + the two dips are unequal depth (both Lorentzians collapse onto one dip ->
negative R^2). This seeds from the two deepest WELL-SEPARATED minima instead.

Reads a finished OR still-growing scan (live h5 fallback, locking off).

Usage (PROJECT ROOT, yb_analysis python):
  python pyctrl/tools/fit_at_splitting.py <scan_id|scan_dir> [--min-sep 0.6] [--no-plot]
e.g.  python pyctrl/tools/fit_at_splitting.py 20260630144109
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import curve_fit

from yb_analysis.analysis.unpack import unpack_scan_logicals
from yb_analysis.analysis.probabilities import prob11_site_resolved


def _resolve_dir(arg):
    if os.path.isdir(arg):
        return arg
    sid = str(arg)
    fid = sid[:8] + "_" + sid[8:]
    base = r"D:\OneDrive - Harvard University\Documents - Yb\Data"
    cand = os.path.join(base, sid[:8], "data_" + fid)
    if os.path.isdir(cand):
        return cand
    raise SystemExit("not a dir and could not resolve scan_id %s -> %s" % (arg, cand))


def _snapshot(scan_dir):
    sid = os.path.basename(scan_dir).replace("data_", "")
    scan = json.load(open(os.path.join(scan_dir, "data_%s.json" % sid)))
    import h5py
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    h5 = os.path.join(scan_dir, "data_%s.h5" % sid)
    last = None
    for _ in range(40):
        try:
            with h5py.File(h5, "r", swmr=True) as f:
                n = min(f["seq_ids"].shape[0], f["logicals_img1"].shape[0], f["logicals_img2"].shape[0])
                return scan, sid, f["seq_ids"][:n], f["logicals_img1"][:n], f["logicals_img2"][:n], n
        except Exception as e:
            last = e; time.sleep(0.25)
    raise RuntimeError("could not read h5: %r" % last)


def _lor(x, A, x0, w):
    return A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def _double_dip(x, y0, A1, x1, w1, A2, x2, w2):
    return y0 - _lor(x, A1, x1, w1) - _lor(x, A2, x2, w2)


def _smooth(y, k=3):
    m = np.isfinite(y)
    ys = np.where(m, y, 0.0); w = m.astype(float)
    num = np.convolve(ys, np.ones(k), "same"); den = np.convolve(w, np.ones(k), "same")
    return np.where(den > 0, num / den, np.nan)


def _seed_two_dips(x, ys, min_sep):
    """Two deepest minima at least `min_sep` MHz apart (-> robust double-dip seed)."""
    finite = np.where(np.isfinite(ys))[0]
    o = finite[np.argsort(ys[finite])]          # ascending survival = deepest first
    d1 = o[0]
    d2 = next((k for k in o[1:] if abs(x[k] - x[d1]) > min_sep), o[1] if len(o) > 1 else o[0])
    return sorted([float(x[d1]), float(x[d2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="scan_id (e.g. 20260630144109) or a scan_dir path")
    ap.add_argument("--min-sep", type=float, default=0.6, help="min dip separation (MHz) for seeding")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    scan_dir = _resolve_dir(args.scan)
    scan, sid, sid_arr, l1, l2, nshot = _snapshot(scan_dir)
    sp, logic1, logic2, reps = unpack_scan_logicals(scan, seq_ids=sid_arr, logicals_img1=l1, logicals_img2=l2)
    sp = np.asarray(sp, float)
    x = sp.reshape(sp.shape[0], -1)[:, 0] / 1e6      # MHz
    order = np.argsort(x); x = x[order]
    M, _ = prob11_site_resolved(logic1, logic2)
    y = np.nanmean(M, axis=0)[order]                 # array-averaged survival

    m = np.isfinite(y)
    xf, yf = x[m], y[m]
    if xf.size < 7:
        raise SystemExit("too few finite points (%d)" % xf.size)
    ys = _smooth(y, 3)
    span = x.max() - x.min()
    y0g = float(np.nanpercentile(yf, 90))
    s1, s2 = _seed_two_dips(x, ys, args.min_sep)
    Ag = max(y0g - float(np.nanmin(yf)), 0.05)

    p0 = [y0g, Ag, s1, 0.5, Ag, s2, 0.5]
    lo = [0, 0, x.min(), 0.05, 0, x.min(), 0.05]
    hi = [1.2, 1.5, x.max(), span, 1.5, x.max(), span]
    try:
        p, _ = curve_fit(_double_dip, xf, yf, p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception as e:
        raise SystemExit("double-dip fit failed: %s" % e)
    r2 = 1 - np.sum((yf - _double_dip(xf, *p)) ** 2) / np.sum((yf - yf.mean()) ** 2)

    # order the two components by center
    comps = sorted([(p[2], abs(p[3])), (p[5], abs(p[6]))], key=lambda c: c[0])
    c1, w1 = comps[0]; c2, w2 = comps[1]
    split = abs(c2 - c1)

    print("AT scan %s | %d shots, %d pts, window %.2f-%.2f MHz, loading ~%.2f"
          % (sid, nshot, xf.size, x.min(), x.max(), float(np.nanmean(M))))
    print("  dip1 = %.4f MHz (FWHM %.0f kHz) | dip2 = %.4f MHz (FWHM %.0f kHz)"
          % (c1, w1 * 1e3, c2, w2 * 1e3))
    print("  AT SPLITTING = %.4f MHz   R^2 = %.3f%s"
          % (split, r2, "   *** LOW R^2 -- check the plot / --min-sep ***" if r2 < 0.7 else ""))
    out = {"scan_id": sid, "n_shots": nshot, "dip1_MHz": c1, "dip2_MHz": c2,
           "fwhm1_kHz": w1 * 1e3, "fwhm2_kHz": w2 * 1e3, "splitting_MHz": split, "r_squared": r2}
    print("JSON " + json.dumps(out))

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xg = np.linspace(x.min(), x.max(), 400)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        ax.plot(x, y, "ko", ms=4, label="array-avg survival")
        ax.plot(xg, _double_dip(xg, *p), "-", color="C1", lw=2, label="double Lorentzian  R$^2$=%.3f" % r2)
        ax.plot(xg, p[0] - _lor(xg, p[1], p[2], p[3]), "--", color="C1", lw=0.8, alpha=0.6)
        ax.plot(xg, p[0] - _lor(xg, p[4], p[5], p[6]), "--", color="C1", lw=0.8, alpha=0.6)
        for c in (c1, c2):
            ax.axvline(c, color="C3", ls=":", lw=0.8)
        ax.annotate("", xy=(c1, 0.55), xytext=(c2, 0.55), arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
        ax.text((c1 + c2) / 2, 0.57, "AT splitting = %.3f MHz" % split, ha="center", fontsize=11, color="0.2")
        ax.set_xlabel("556 freq [MHz]"); ax.set_ylabel("survival (P11)")
        ax.set_title("556 Autler-Townes (array-avg)  %s\nd1 %.4f / d2 %.4f MHz, split %.3f MHz, %d shots"
                     % (sid, c1, c2, split, nshot), fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(scan_dir, "at_avg_doublefit_%s.png" % sid)
        fig.savefig(png, bbox_inches="tight")
        print("  saved %s" % png)


if __name__ == "__main__":
    main()
