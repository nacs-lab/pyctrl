"""persite_at_splitting.py -- site-resolved Autler-Townes SPLITTING spatial map.

For one OR MORE pooled Autler-Townes survival scans (556 freq swept, 308 on),
fit EACH site's survival-vs-freq to a DOUBLE Lorentzian dip and plot the fitted
AT splitting = |peak2 - peak1| as a spatial scatter over the array. The splitting
is proportional to the 308 (upper-leg) Rabi frequency, so this maps the 308
intensity / coupling uniformity across the array.

Sites where the double fit fails / merges (degenerate) or R^2 is low are dropped
(grey x). Uses the shared _persite_common pooling.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_at_splitting.py <scan_id> [<scan_id> ...] [--minr2 0.5] [--minload 20]
Saves persite_at_splitting_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined, _bootstrap_root


def build(scan_ids, minr2=0.5, minload=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _bootstrap_root()
    from yb_analysis.analysis.fittings.lorentzian import fit_double_lorentzian

    c = load_combined(scan_ids)
    p11 = c["p11"]; n_load = c["n_load"]
    n_sites, n_params = c["n_sites"], c["n_params"]
    freq = c["freq_hz"]
    xs, ys, have_xy = c["xs"], c["ys"], c["have_xy"]

    split = np.full(n_sites, np.nan)   # MHz
    c1 = np.full(n_sites, np.nan); c2 = np.full(n_sites, np.nan)
    for i in range(n_sites):
        y = p11[i]
        good = np.isfinite(y)
        if good.sum() < 8 or np.nansum(n_load[i]) < minload:
            continue
        try:
            f = fit_double_lorentzian(freq[good], y[good], None, mode="dip")
        except Exception:
            f = None
        if not f or f["r_squared"] < minr2:
            continue
        split[i] = f["splitting"] / 1e6
        c1[i], c2[i] = f["centers"][0] / 1e6, f["centers"][1] / 1e6

    ok = np.isfinite(split)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise SystemExit("no good double-Lorentzian fits (try lowering --minr2/--minload)")
    med = np.nanmedian(split[ok])
    mad = np.nanmedian(np.abs(split[ok] - med)) or 0.05
    vlo, vhi = max(0, med - 2 * 1.4826 * mad), med + 2 * 1.4826 * mad

    fig, ax = plt.subplots(figsize=(11, 9))
    if (~ok).any():
        ax.scatter(xs[~ok], ys[~ok], s=10, marker="x", color="0.7", linewidths=0.5,
                   label="no 2-dip fit (%d)" % int((~ok).sum()))
    sc = ax.scatter(xs[ok], ys[ok], c=split[ok], s=42, cmap="turbo",
                    vmin=vlo, vmax=vhi, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("AT splitting [MHz]")
    if have_xy:
        ax.set_xlabel("site x [px]"); ax.set_ylabel("site y [px]")
        ax.set_aspect("equal"); ax.invert_yaxis()
    else:
        ax.set_xlabel("site index")
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Site-resolved AT splitting  %s\n%d/%d sites (2-dip fit)  |  median %.3f MHz  "
                 "MAD %.0f kHz  |  %d shots%s"
                 % (tag, n_ok, n_sites, med, 1.4826 * mad * 1e3, c["total_shots"],
                    ("  (%d pooled)" % len(c["scan_ids"])) if len(c["scan_ids"]) > 1 else ""))
    if (~ok).any():
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "persite_at_splitting_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=130)
    if have_xy:
        cx = np.corrcoef(xs[ok], split[ok])[0, 1]; cy = np.corrcoef(ys[ok], split[ok])[0, 1]
    else:
        cx = cy = float("nan")
    print("scans %s | %d/%d sites 2-dip fit (minr2=%.2f), %d shots"
          % (c["scan_ids"], n_ok, n_sites, minr2, c["total_shots"]))
    print("  AT splitting median %.4f MHz | 10-90 pct %.4f - %.4f | MAD %.0f kHz | corr x %.3f y %.3f"
          % (med, np.nanpercentile(split[ok], 10), np.nanpercentile(split[ok], 90),
             1.4826 * mad * 1e3, cx, cy))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more AT scan ids to POOL")
    ap.add_argument("--minr2", type=float, default=0.5)
    ap.add_argument("--minload", type=int, default=20)
    a = ap.parse_args()
    build(a.scan_ids, minr2=a.minr2, minload=a.minload)
