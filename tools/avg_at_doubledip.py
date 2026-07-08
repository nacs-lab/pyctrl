"""avg_at_doubledip.py -- array-averaged Autler-Townes spectrum with a ROBUST double-dip fit.

fit_spectrum.py --peaks 2 uses an auto-seed that often fails on the AT doublet. This
uses the same robust recipe as tmp_at_gaussian2d.py's per-site fit (smooth -> seed the two
dips from the two lowest points >=0.6 MHz apart -> bounded curve_fit) on the loaded-weighted
array-average P11(freq), and reports the two centers + AT splitting.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/avg_at_doubledip.py <scan_id> [<scan_id> ...]
Saves avg_at_doubledip_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined


def _lor_dip(x, A, x0, w):
    return A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def _double_dip(x, y0, A1, x01, w1, A2, x02, w2):
    return y0 - _lor_dip(x, A1, x01, w1) - _lor_dip(x, A2, x02, w2)


def _smooth(y, k=3):
    m = np.isfinite(y); ys = np.where(m, y, 0.0); w = m.astype(float)
    num = np.convolve(ys, np.ones(k), "same"); den = np.convolve(w, np.ones(k), "same")
    return np.where(den > 0, num / den, np.nan)


def build(scan_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    c = load_combined(scan_ids)
    freq = c["freq_hz"] / 1e6                       # MHz
    tot_load = np.nansum(c["n_load"], axis=0)
    tot_surv = np.nansum(c["n_surv"], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(tot_load > 0, tot_surv / tot_load, np.nan)
    g = np.isfinite(y)
    xf, yf = freq[g], y[g]
    span = xf.max() - xf.min()

    # robust seed (reference recipe): two lowest smoothed points >=0.6 MHz apart
    ys = _smooth(y, 3)
    y0g = float(np.nanpercentile(ys[g], 85))
    fin = np.where(np.isfinite(ys))[0]
    o = fin[np.argsort(ys[fin])]
    d1 = o[0]
    d2 = next((k for k in o[1:] if abs(freq[k] - freq[d1]) > 0.6), o[1])
    xa, xb = sorted([freq[d1], freq[d2]])
    Ag = max(y0g - float(np.nanmin(ys)), 0.05)
    popt, _ = curve_fit(_double_dip, xf, yf,
                        p0=[y0g, Ag, xa, 0.8, Ag, xb, 0.8],
                        bounds=([0, 0, xf.min(), 0.1, 0, xf.min(), 0.1],
                                [1.2, 1.5, xf.max(), span, 1.5, xf.max(), span]), maxfev=40000)
    pred = _double_dip(xf, *popt)
    sst = np.sum((yf - yf.mean()) ** 2)
    r2 = 1 - np.sum((yf - pred) ** 2) / sst if sst > 0 else 0
    c1, w1, c2, w2 = popt[2], popt[3], popt[5], popt[6]
    (c1, w1), (c2, w2) = sorted([(c1, w1), (c2, w2)])
    split = abs(c2 - c1)

    xg = np.linspace(xf.min(), xf.max(), 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq, y, "o", ms=4, color="k", label="array-avg survival")
    ax.plot(xg, _double_dip(xg, *popt), "-", color="C3", lw=1.8,
            label="double Lorentzian  R2=%.3f" % r2)
    for cc in (c1, c2):
        ax.axvline(cc, color="C3", ls="--", lw=0.8)
    ax.set_xlabel("556 freq [MHz]"); ax.set_ylabel("survival (P11)")
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Array-avg AT double-dip  %s\npeaks %.3f / %.3f MHz  split %.3f MHz  "
                 "FWHM %.0f/%.0f kHz  R2 %.3f  |  %d shots"
                 % (tag, c1, c2, split, w1 * 1e3, w2 * 1e3, r2, c["total_shots"]), fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "avg_at_doubledip_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("scans %s | %d shots" % (c["scan_ids"], c["total_shots"]))
    print("  peaks %.4f / %.4f MHz | split %.4f MHz | FWHM %.0f/%.0f kHz | R2 %.3f"
          % (c1, c2, split, w1 * 1e3, w2 * 1e3, r2))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+")
    a = ap.parse_args()
    build(a.scan_ids)
