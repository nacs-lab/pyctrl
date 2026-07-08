"""persite_fit_gallery.py -- show a grid of individual per-site Lorentzian fits.

Sanity-check the per-site spectroscopy fits behind persite_fit_centers.py:
pick a spread of sites (lowest / median / highest fitted center) and plot each
site's survival-vs-freq points with its fitted dip curve and center, so you can
judge whether the fits (and any spatial pattern) are real. Accepts one OR MORE
scan ids (POOLED, same grid+freq axis) for more reps/point.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_fit_gallery.py <scan_id> [<scan_id> ...] [--minr2 0.4] [--n 12]
    python pyctrl/tools/persite_fit_gallery.py <sid> --sites 12,45,900   # explicit sites
Saves persite_fit_gallery_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined, _bootstrap_root


def build(scan_ids, minr2=0.4, n=12, sites=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _bootstrap_root()
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian

    c = load_combined(scan_ids)
    p11 = c["p11"]; nl = c["n_load"]; freq = c["freq_hz"]
    n_sites = c["n_sites"]
    fmin, fmax = freq.min(), freq.max(); et = 0.03 * (fmax - fmin)
    xs, ys = c["xs"], c["ys"]

    def fit_site(i):
        y = p11[i]; g = np.isfinite(y)
        if g.sum() < 6 or np.nansum(nl[i]) < 20:
            return None
        f = fit_lorentzian(freq[g], y[g], None, mode="dip")   # UNWEIGHTED
        return f

    if sites:
        pick = [s for s in sites if 0 <= s < n_sites]
    else:
        fits_all = {}
        for i in range(n_sites):
            f = fit_site(i)
            if not f:
                continue
            cc = f["center"]; r2 = f["r_squared"]
            if r2 < minr2 or cc < fmin + et or cc > fmax - et:
                continue
            fits_all[i] = f
        if not fits_all:
            raise SystemExit("no good fits")
        idx = np.array(sorted(fits_all))
        cen = np.array([fits_all[i]["center"] for i in idx])
        csrt = idx[np.argsort(cen)]
        k = max(1, n // 3)
        pick = (list(csrt[:k])
                + list(csrt[len(csrt) // 2 - k // 2: len(csrt) // 2 - k // 2 + k])
                + list(csrt[-k:]))
        pick = list(dict.fromkeys(int(i) for i in pick))[:n]

    ncol = 4; nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, i in enumerate(pick):
        ax = axes[j // ncol][j % ncol]; ax.axis("on")
        y = p11[i]; g = np.isfinite(y)
        ax.plot(freq[g] / 1e6, y[g], "o", ms=3, color="k")
        f = fit_site(i)
        if f:
            ax.plot(f["x_fit"] / 1e6, f["y_fit"], "-", color="C3", lw=1.3)
            ax.axvline(f["center"] / 1e6, color="C3", ls="--", lw=0.8)
            ttl = ("site %d @(%.0f,%.0f)\nc=%.3f MHz  w=%.0fkHz  R2=%.2f"
                   % (i, xs[i], ys[i], f["center"] / 1e6, f["width"] / 1e3, f["r_squared"]))
        else:
            ttl = "site %d @(%.0f,%.0f)\n(no fit)" % (i, xs[i], ys[i])
        ax.set_ylim(-0.05, 1.1); ax.set_title(ttl, fontsize=7.5); ax.tick_params(labelsize=6)
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    fig.suptitle("Per-site fit gallery  %s  (%d shots%s)"
                 % (tag, c["total_shots"],
                    ", %d pooled" % len(c["scan_ids"]) if len(c["scan_ids"]) > 1 else ""),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "persite_fit_gallery_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=130)
    print("scans %s | %d shots pooled, showing sites %s" % (c["scan_ids"], c["total_shots"], list(pick)))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more scan ids to POOL")
    ap.add_argument("--minr2", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sites", default=None, help="comma-sep explicit site indices to show")
    a = ap.parse_args()
    sites = [int(s) for s in a.sites.split(",")] if a.sites else None
    build(a.scan_ids, minr2=a.minr2, n=a.n, sites=sites)
