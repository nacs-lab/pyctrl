"""persite_fit_contrast.py -- site-resolved fitted CONTRAST (dip depth) spatial map.

Companion to persite_fit_centers.py: for one OR MORE pooled 1-D survival-
spectroscopy scans, fit each site's survival-vs-freq to a single Lorentzian dip
(UNWEIGHTED) and plot the fitted CONTRAST = dip amplitude A (baseline - on-
resonance survival) as a spatial scatter over the array. Shows where the push-out
/ Rydberg line is strong vs weak (e.g. an AOM diffraction-efficiency roll-off
across the array shows up as a contrast gradient).

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_fit_contrast.py <scan_id> [<scan_id> ...] [--minr2 0.4] [--minload 20]
Saves persite_fit_contrast_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined, _bootstrap_root


def build(scan_ids, minr2=0.4, minload=20):
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

    contrast = np.full(n_sites, np.nan)
    for i in range(n_sites):
        y = p11[i]
        good = np.isfinite(y)
        if good.sum() < 6 or np.nansum(n_load[i]) < minload:
            continue
        try:
            fit = fit_lorentzian(freq[good], y[good], None, mode="dip")   # UNWEIGHTED
        except Exception:
            fit = None
        if not fit:
            continue
        cc = float(fit["center"]); r2 = float(fit["r_squared"])
        if r2 < minr2 or cc < fmin + edge_tol or cc > fmax - edge_tol:
            continue
        A = float(fit["params"][1])           # dip amplitude = contrast
        contrast[i] = np.clip(A, 0.0, 1.0)

    ok = np.isfinite(contrast)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise SystemExit("no good per-site fits (try lowering --minr2/--minload)")
    med = np.nanmedian(contrast[ok])
    # contrast in [0,1]; color range 0..robust-high for readability
    vhi = min(1.0, np.nanpercentile(contrast[ok], 98))
    vlo = max(0.0, np.nanpercentile(contrast[ok], 2))

    fig, ax = plt.subplots(figsize=(11, 9))
    if (~ok).any():
        ax.scatter(xs[~ok], ys[~ok], s=10, marker="x", color="0.7", linewidths=0.5,
                   label="no fit (%d)" % int((~ok).sum()))
    sc = ax.scatter(xs[ok], ys[ok], c=contrast[ok], s=42, cmap="magma",
                    vmin=vlo, vmax=vhi, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("fitted contrast (dip depth)")
    if have_xy:
        ax.set_xlabel("site x [px]"); ax.set_ylabel("site y [px]")
        ax.set_aspect("equal"); ax.invert_yaxis()
    else:
        ax.set_xlabel("site index"); ax.set_ylabel("(no grid xy)")
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Site-resolved fitted contrast  %s\n%d/%d sites fit  |  median contrast %.2f  "
                 "|  %d shots%s"
                 % (tag, n_ok, n_sites, med, c["total_shots"],
                    ("  (%d scans pooled)" % len(c["scan_ids"])) if len(c["scan_ids"]) > 1 else ""))
    if (~ok).any():
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "persite_fit_contrast_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=130)
    # spatial trend
    if have_xy:
        cx = np.corrcoef(xs[ok], contrast[ok])[0, 1]
        cy = np.corrcoef(ys[ok], contrast[ok])[0, 1]
    else:
        cx = cy = float("nan")
    print("scans %s | %d/%d sites fit (minr2=%.2f), %d shots pooled"
          % (c["scan_ids"], n_ok, n_sites, minr2, c["total_shots"]))
    print("  contrast median %.3f | 10-90 pct %.3f - %.3f | corr vs x %.3f vs y %.3f"
          % (med, np.nanpercentile(contrast[ok], 10), np.nanpercentile(contrast[ok], 90), cx, cy))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more scan ids to POOL (same grid+freq axis)")
    ap.add_argument("--minr2", type=float, default=0.4)
    ap.add_argument("--minload", type=int, default=20)
    a = ap.parse_args()
    build(a.scan_ids, minr2=a.minr2, minload=a.minload)
