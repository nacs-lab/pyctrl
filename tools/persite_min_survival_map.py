"""persite_min_survival_map.py -- spatial map of the FREQUENCY of lowest survival per site.

Fit-free alternative to persite_fit_centers.py: for each site, take the swept
frequency at which its survival P11 is LOWEST (argmin), directly from the data --
no Lorentzian fit (per-site fits are noisy/unreliable at modest reps). Plot that
frequency as a spatial scatter over the array. Optionally smooth the per-site
survival curve first (rolling mean) to beat single-point shot noise.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_min_survival_map.py <scan_id> [--minload 20] [--smooth 3]
Saves persite_min_survival_map_<scan_id>.png into the scan's data folder.
"""
import argparse
import os
import sys

import numpy as np


def _bootstrap_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


def _smooth(a, w):
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def build(scan_id, minload=20, smooth=3, depth_min=0.15):
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    _bootstrap_root()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from yb_analysis.analysis.run_analysis import _resolve_scan_dir_from_id, _site_grid_xy
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals

    scan_dir = _resolve_scan_dir_from_id(scan_id)
    bundle = load_scan_from_path(str(scan_dir))
    scan = bundle.get("Scan") or {}
    scan_params, logic1, logic2, reps = unpack_scan_logicals(
        scan, seq_ids=bundle.get("seq_ids"), mat_path=bundle.get("mat_path"),
        logicals_img1=bundle.get("logicals_img1"),
        logicals_img2=bundle.get("logicals_img2"))
    n_sites, n_params, _ = logic1.shape
    sp = np.asarray(scan_params, float)
    freq = (sp[:, 0] if sp.ndim == 2 else sp).astype(float)
    order = np.argsort(freq); freq = freq[order]
    fmhz = freq / 1e6
    nl = np.nansum(logic1.astype(float), 2)[:, order]
    p11 = np.where(nl > 0, np.nansum((logic1 & logic2).astype(float), 2)[:, order] / nl, np.nan)

    xs, ys = _site_grid_xy(scan)
    have_xy = len(xs) == n_sites
    xs = np.asarray(xs, float) if have_xy else np.arange(n_sites, float)
    ys = np.asarray(ys, float) if have_xy else np.zeros(n_sites)

    min_freq = np.full(n_sites, np.nan)   # freq [MHz] of lowest survival
    depth = np.full(n_sites, np.nan)      # baseline - min (dip depth)
    for i in range(n_sites):
        y = p11[i]
        if np.nansum(nl[i]) < minload or not np.isfinite(y).any():
            continue
        ys_ = _smooth(np.where(np.isfinite(y), y, np.nan), smooth)
        if not np.isfinite(ys_).any():
            continue
        j = int(np.nanargmin(ys_))
        base = np.nanpercentile(y, 90)   # off-resonance baseline
        min_freq[i] = fmhz[j]
        depth[i] = base - np.nanmin(ys_)

    # only keep sites with a real dip (depth above threshold) for the color map
    have_dip = np.isfinite(min_freq) & (depth >= depth_min)
    n_ok = int(have_dip.sum())
    if n_ok == 0:
        raise SystemExit("no sites with a dip >= %.2f (try lowering --depth-min)" % depth_min)

    med = np.nanmedian(min_freq[have_dip])
    mad = np.nanmedian(np.abs(min_freq[have_dip] - med)) or 0.05
    vlo, vhi = med - 2 * 1.4826 * mad, med + 2 * 1.4826 * mad

    fig, ax = plt.subplots(figsize=(11, 9))
    weak = np.isfinite(min_freq) & ~have_dip
    if weak.any():
        ax.scatter(xs[weak], ys[weak], s=10, marker="x", color="0.75", linewidths=0.5,
                   label="no clear dip (%d)" % int(weak.sum()))
    sc = ax.scatter(xs[have_dip], ys[have_dip], c=min_freq[have_dip], s=42, cmap="turbo",
                    vmin=vlo, vmax=vhi, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("freq of lowest survival [MHz]")
    if have_xy:
        ax.set_xlabel("site x [px]"); ax.set_ylabel("site y [px]")
        ax.set_aspect("equal"); ax.invert_yaxis()
    else:
        ax.set_xlabel("site index")
    tot = int(np.nansum(reps))
    ax.set_title("Freq of lowest survival per site  %s\n%d/%d sites w/ dip>=%.2f  |  "
                 "median %.3f MHz  MAD %.0f kHz  |  smooth=%d  |  %d shots"
                 % (scan_id, n_ok, n_sites, depth_min, med, 1.4826 * mad * 1e3, smooth, tot))
    if weak.any():
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(str(scan_dir), "persite_min_survival_map_%s.png" % scan_id)
    fig.savefig(out, dpi=130)
    print("scan %s | %d/%d sites with dip>=%.2f (minload=%d, smooth=%d)"
          % (scan_id, n_ok, n_sites, depth_min, minload, smooth))
    print("  argmin-freq median %.4f MHz | 10-90 pct %.4f - %.4f | MAD %.0f kHz"
          % (med, np.nanpercentile(min_freq[have_dip], 10),
             np.nanpercentile(min_freq[have_dip], 90), 1.4826 * mad * 1e3))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_id")
    ap.add_argument("--minload", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=3, help="rolling-mean window over freq (pts)")
    ap.add_argument("--depth-min", type=float, default=0.15, dest="depth_min",
                    help="min dip depth (baseline - min) to color a site")
    a = ap.parse_args()
    build(a.scan_id, minload=a.minload, smooth=a.smooth, depth_min=a.depth_min)
