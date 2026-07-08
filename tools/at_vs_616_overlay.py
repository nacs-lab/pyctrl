"""at_vs_616_overlay.py -- overlay array-avg AT spectra vs 616-EOM freq + summary table.

Stacks the array-averaged Autler-Townes survival spectra for a set of scans (each at a
different 616-EOM freq) with a vertical offset, ordered by 616 freq, so the change in the
DOUBLET SYMMETRY and SPLITTING with 616 detuning is visible at a glance. Fits each with a
robust double-Lorentzian dip (same recipe as avg_at_doubledip.py) and prints/annotates a
summary table (616 freq | left/right dip | splitting | left/right FWHM | R2).

Usage (PROJECT ROOT, yb_analysis python) -- pass "sid:freqMHz" pairs, any order:
  at_vs_616_overlay.py 20260703175204:234.5 20260703174828:235.0 ... --out DIR
Saves at_vs_616_overlay.png (+ a text table) into --out (default = first scan's dir).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined


def _lor(x, A, x0, w):
    return A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def _dbl(x, y0, A1, x01, w1, A2, x02, w2):
    return y0 - _lor(x, A1, x01, w1) - _lor(x, A2, x02, w2)


def _smooth(y, k=3):
    m = np.isfinite(y); ys = np.where(m, y, 0.0); w = m.astype(float)
    return np.where(np.convolve(w, np.ones(k), "same") > 0,
                    np.convolve(ys, np.ones(k), "same") / np.maximum(np.convolve(w, np.ones(k), "same"), 1e-9), np.nan)


def _avg(sid):
    c = load_combined([sid])
    f = c["freq_hz"] / 1e6
    tl = np.nansum(c["n_load"], 0); ts = np.nansum(c["n_surv"], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(tl > 0, ts / tl, np.nan)
    return f, y, c["total_shots"]


def _fit(f, y):
    from scipy.optimize import curve_fit
    g = np.isfinite(y); xf, yf = f[g], y[g]; span = xf.max() - xf.min()
    ys = _smooth(y, 3); y0g = float(np.nanpercentile(ys[g], 85))
    fin = np.where(np.isfinite(ys))[0]; o = fin[np.argsort(ys[fin])]
    d1 = o[0]; d2 = next((k for k in o[1:] if abs(f[k] - f[d1]) > 0.6), o[1])
    xa, xb = sorted([f[d1], f[d2]]); Ag = max(y0g - float(np.nanmin(ys)), 0.05)
    p, _ = curve_fit(_dbl, xf, yf, p0=[y0g, Ag, xa, 0.8, Ag, xb, 0.8],
                     bounds=([0, 0, xf.min(), 0.1, 0, xf.min(), 0.1],
                             [1.2, 1.5, xf.max(), span, 1.5, xf.max(), span]), maxfev=40000)
    pred = _dbl(xf, *p); sst = np.sum((yf - yf.mean()) ** 2)
    r2 = 1 - np.sum((yf - pred) ** 2) / sst if sst > 0 else 0
    # order by center
    comps = sorted([(p[2], p[3], p[1]), (p[5], p[6], p[4])])
    (cL, wL, AL), (cR, wR, AR) = comps
    return dict(y0=p[0], cL=cL, wL=wL, AL=AL, cR=cR, wR=wR, AR=AR,
                split=cR - cL, r2=r2, popt=p)


def build(pairs, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = []
    for sid, fr in pairs:
        f, y, ns = _avg(sid)
        fit = _fit(f, y)
        items.append(dict(sid=sid, f616=fr, f=f, y=y, ns=ns, fit=fit))
    items.sort(key=lambda d: d["f616"])
    if out_dir is None:
        c0 = load_combined([items[0]["sid"]]); out_dir = c0["primary_dir"]

    # ---- overlay (offset-stacked by 616 freq) ----
    fig, ax = plt.subplots(figsize=(9, 10))
    dy = 1.15
    cmap = plt.get_cmap("viridis")
    n = len(items)
    for i, it in enumerate(items):
        off = i * dy
        col = cmap(i / max(n - 1, 1))
        xg = np.linspace(it["f"][np.isfinite(it["y"])].min(), it["f"][np.isfinite(it["y"])].max(), 400)
        ax.plot(it["f"], it["y"] + off, "o", ms=3, color=col)
        ax.plot(xg, _dbl(xg, *it["fit"]["popt"]) + off, "-", color=col, lw=1.6)
        for cc in (it["fit"]["cL"], it["fit"]["cR"]):
            ax.plot([cc, cc], [off + 0.02, off + 0.30], color=col, ls="--", lw=0.7)
        ax.text(it["f"].min() - 0.05, off + 0.55,
                "616=%.1f  split %.2f MHz  R2 %.2f" % (it["f616"], it["fit"]["split"], it["fit"]["r2"]),
                fontsize=8, va="center", ha="right", color=col)
    ax.set_xlabel("556 freq [MHz]")
    ax.set_ylabel("survival (P11)  +  offset by 616 freq")
    ax.set_title("Autler-Townes doublet vs 616-EOM freq (offset-stacked, low 616 at bottom)\n"
                 "symmetry + splitting evolve with 616 detuning", fontsize=10)
    ax.set_xlim(items[0]["f"].min() - 1.6, items[0]["f"].max() + 0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "at_vs_616_overlay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")

    # ---- summary table (print + txt) ----
    hdr = "%-8s %-9s %-9s %-9s %-10s %-10s %-6s %-7s" % (
        "616[MHz]", "dipL[MHz]", "dipR[MHz]", "split", "fwhmL[kHz]", "fwhmR[kHz]", "R2", "shots")
    lines = [hdr, "-" * len(hdr)]
    for it in items:
        ft = it["fit"]
        lines.append("%-8.1f %-9.3f %-9.3f %-9.3f %-10.0f %-10.0f %-6.3f %-7d" % (
            it["f616"], ft["cL"], ft["cR"], ft["split"], ft["wL"] * 1e3, ft["wR"] * 1e3, ft["r2"], it["ns"]))
    table = "\n".join(lines)
    print(table)
    txt = os.path.join(out_dir, "at_vs_616_table.txt")
    open(txt, "w").write(table + "\n")
    print("\n  saved", out)
    print("  saved", txt)
    return out, txt, table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="sid:freqMHz pairs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = [(p.split(":")[0], float(p.split(":")[1])) for p in a.pairs]
    build(pairs, out_dir=a.out)
