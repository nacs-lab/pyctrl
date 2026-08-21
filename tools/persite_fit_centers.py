"""persite_fit_centers.py -- site-resolved fitted resonance centers (spatial map).

For one OR MORE 1-D survival-spectroscopy scans (same grid + freq axis; POOLED
for more reps/point), fit EACH site's survival-vs-freq to a single Lorentzian dip
and plot the fitted center as a spatial scatter over the array (x,y from the .mat
initGridLocations), colored by center frequency. Reveals resonance inhomogeneity
across the array (field / trap-depth / light-shift spread).

Fits are UNWEIGHTED: a binomial SEM ~ sqrt(p(1-p)/n) collapses at the P11~1
baseline and over-penalizes baseline shot-noise, tanking R^2 for a clean deep dip
(weighted R2 median ~0.09 vs unweighted ~0.57 on 10-rep data). Sites with a bad
fit (low R^2, edge-pinned center, too few loaded shots) are dropped -> grey x's.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_fit_centers.py <scan_id> [<scan_id> ...] [--minr2 0.4] [--minload 20]
                                               [--mode dip|peak]

NOTE (2026-08-18): on a Revival616 line the per-site fits are NOISE-LIMITED at ~3 loaded
shots/site/point (131/1068 sites pass minr2=0.4, median centre stderr 4.3 MHz > the 1.4 MHz
centre scatter), and that line also sits on a SLOPED baseline that this single-Lorentzian
model does not carry -- so prefer spatial CELL-POOLING for revival lineshape work. See the
08/18 Notion entry.
Saves persite_fit_centers_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined, _bootstrap_root


def build(scan_ids, minr2=0.4, minload=20, mode='dip'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _bootstrap_root()
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian

    c = load_combined(scan_ids)
    p11 = c["p11"]; n_load = c["n_load"]
    n_sites, n_params = c["n_sites"], c["n_params"]
    freq = c["freq_hz"]
    fmin, fmax = freq.min(), freq.max()
    edge_tol = 0.03 * (fmax - fmin)
    xs, ys, have_xy = c["xs"], c["ys"], c["have_xy"]

    centers = np.full(n_sites, np.nan)
    r2s = np.full(n_sites, np.nan)
    for i in range(n_sites):
        y = p11[i]
        good = np.isfinite(y)
        if good.sum() < 6 or np.nansum(n_load[i]) < minload:
            continue
        try:
            fit = fit_lorentzian(freq[good], y[good], None, mode=mode)   # UNWEIGHTED
        except Exception:
            fit = None
        if not fit:
            continue
        cc = float(fit["center"]); r2 = float(fit["r_squared"])
        if r2 < minr2 or cc < fmin + edge_tol or cc > fmax - edge_tol:
            continue
        centers[i] = cc; r2s[i] = r2

    ok = np.isfinite(centers)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise SystemExit("no good per-site fits (try lowering --minr2/--minload)")
    c_mhz = centers / 1e6
    med = np.nanmedian(c_mhz[ok])
    mad = np.nanmedian(np.abs(c_mhz[ok] - med)) or 0.05
    vlo, vhi = med - 1.5 * 1.4826 * mad, med + 1.5 * 1.4826 * mad

    fig, ax = plt.subplots(figsize=(11, 9))
    if (~ok).any():
        ax.scatter(xs[~ok], ys[~ok], s=10, marker="x", color="0.7", linewidths=0.5,
                   label="no fit (%d)" % int((~ok).sum()))
    sc = ax.scatter(xs[ok], ys[ok], c=c_mhz[ok], s=42, cmap="turbo",
                    vmin=vlo, vmax=vhi, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("fitted %s center [MHz]" % mode)
    if have_xy:
        ax.set_xlabel("site x [px]"); ax.set_ylabel("site y [px]")
        ax.set_aspect("equal"); ax.invert_yaxis()
    else:
        ax.set_xlabel("site index"); ax.set_ylabel("(no grid xy)")
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Site-resolved fitted centers  %s\n%d/%d sites fit  |  median %.3f MHz  "
                 "spread(MAD) %.0f kHz  |  %d shots%s"
                 % (tag, n_ok, n_sites, med, 1.4826 * mad * 1e3, c["total_shots"],
                    ("  (%d scans pooled)" % len(c["scan_ids"])) if len(c["scan_ids"]) > 1 else ""))
    if (~ok).any():
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "persite_fit_centers_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=130)
    print("scans %s | %d/%d sites fit (minr2=%.2f, minload=%d), %d shots pooled"
          % (c["scan_ids"], n_ok, n_sites, minr2, minload, c["total_shots"]))
    print("  center median %.4f MHz | 10-90 pct %.4f - %.4f MHz | MAD %.0f kHz"
          % (med, np.nanpercentile(c_mhz[ok], 10), np.nanpercentile(c_mhz[ok], 90),
             1.4826 * mad * 1e3))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more scan ids to POOL (same grid+freq axis)")
    ap.add_argument("--minr2", type=float, default=0.4, help="min per-site fit R^2 to keep")
    ap.add_argument("--minload", type=int, default=20, help="min total loaded shots to fit a site")
    ap.add_argument("--mode", choices=("dip", "peak"), default="dip",
                    help="lineshape: 'dip' (push-out spectrum, default) or 'peak' (a survival "
                         "REVIVAL line, e.g. Revival616Scan). Mirrors fit_spectrum.py --mode.")
    a = ap.parse_args()
    build(a.scan_ids, minr2=a.minr2, minload=a.minload, mode=a.mode)
