"""imaging_verify_figs.py -- the three figures an imaging campaign owes its Notion entry.

Run on the high-stat verify scan at the adopted working set (stack several same-config runs to
tighten the per-site tails):

    python pyctrl/tools/imaging_verify_figs.py <scan_dir> [<scan_dir> ...]

Writes into the FIRST scan dir, path stamped bottom-left (feedback-stamp-permanent-fig-path):
  persite_fidelity.png   site-resolved analytic Gaussian-overlap fidelity on the array geometry
  persite_survival.png   site-resolved P(img2|img1) on the same geometry
  avg_histogram.png      pooled empty/atom histogram, dark peak centred on 0, cut at the valley

Conventions that are NOT cosmetic (see references/imaging-optimization.md "Notion log"):
  * fidelity is the ANALYTIC Gaussian-overlap per site, never the pooled optimal-cut estimator and
    never a site split at its own threshold (that is degenerate and returns ~1.0).
  * the histogram x-axis is intensity MINUS THAT SITE'S OWN EMPTY-PEAK MEDIAN, so every site's dark
    distribution lands on 0. Colouring by each site's stored threshold instead makes the empty and
    atom populations appear to overlap, because those thresholds sit at different distances above
    their own empty peaks and pooling smears one sharp cut into a band.
  * the histogram is cut at the VALLEY (minimum of the smoothed total between the two modes), which
    is the honest decision boundary; the live per-site thresholds are drawn beside it for comparison.
  * y is peak-normalized to 1 with ONE shared normalization, so the loading weight is preserved.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import persite_imaging as pi   # noqa: E402  (reuses the EM split + overlap infidelity)


def _load(d):
    import h5py
    sid = os.path.basename(d.rstrip("/\\"))
    cfg = json.load(open(os.path.join(d, sid + ".json")))
    with h5py.File(os.path.join(d, sid + ".h5"), "r") as f:
        I1 = f["intensities_img1"][:].astype(float)
        L1 = f["logicals_img1"][:].astype(bool)
        L2 = f["logicals_img2"][:].astype(bool) if "logicals_img2" in f else None
    thr = np.asarray(cfg["initThresholds"], float).ravel()
    x = np.asarray(cfg["initGridLocationsX"], float).ravel()
    y = np.asarray(cfg["initGridLocationsY"], float).ravel()
    return sid, I1, L1, L2, thr, x, y


def _stamp(fig, path):
    fig.text(0.005, 0.005, path, fontsize=5.5, color="0.45", ha="left", va="bottom")


def _map_fig(plt, gx, gy, v, title, label, out, lo=None, hi=None, flag_below=None,
             lognorm=False, flag_above=None):
    """Array map of a per-site quantity.

    The bulk sits within a fraction of a percent of 1, so the colour scale is set on the BULK
    (percentile-based) and any site below ``flag_below`` is ringed in red and labelled with its
    index instead of being crushed into the bottom colour -- those few sites are the reason to
    look at the map at all.
    """
    ok = np.isfinite(v)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=160)
    kw = {}
    if lognorm:
        from matplotlib.colors import LogNorm
        kw["norm"] = LogNorm(vmin=lo, vmax=hi)
    else:
        kw["vmin"] = lo if lo is not None else np.nanpercentile(v, 2)
        kw["vmax"] = hi if hi is not None else np.nanmax(v)
    # viridis throughout: LOW = dark. On the survival map low is bad; on the log-infidelity map low
    # is good, which is why that figure says "dark = better" in its title.
    sc = ax.scatter(gx[ok], gy[ok], c=v[ok], s=16, cmap="viridis", **kw)
    bad = order = None
    if flag_below is not None:
        bad = np.where(np.isfinite(v) & (v < flag_below))[0]
        lab_txt = "< %.3g  (%d sites)" % (flag_below, bad.size)
        order = bad[np.argsort(v[bad])]
    elif flag_above is not None:
        bad = np.where(np.isfinite(v) & (v > flag_above))[0]
        lab_txt = "> %.3g  (%d sites)" % (flag_above, bad.size)
        order = bad[np.argsort(-v[bad])]
    if bad is not None and bad.size:
        ax.scatter(gx[bad], gy[bad], s=95, facecolors="none", edgecolors="#D62728", lw=1.3,
                   label=lab_txt)
        for i in order[:12]:
            ax.annotate("s%d %.3g" % (i, v[i]), (gx[i], gy[i]), fontsize=5.5,
                        color="#D62728", xytext=(4, 4), textcoords="offset points")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.colorbar(sc, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("camera x (px)")
    ax.set_ylabel("camera y (px)")
    ax.set_title(title, fontsize=9)
    _stamp(fig, out)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main(dirs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sid0 = None
    I1s, L1s, L2s = [], [], []
    thr = gx = gy = None
    for d in dirs:
        sid, I1, L1, L2, thr, gx, gy = _load(d)
        sid0 = sid0 or sid
        I1s.append(I1)
        L1s.append(L1)
        if L2 is not None:
            L2s.append(L2)
    I1 = np.vstack(I1s)
    L1 = np.vstack(L1s)
    L2 = np.vstack(L2s) if len(L2s) == len(I1s) else None
    n_shots, n_sites = I1.shape
    out_dir = dirs[0]
    tag = "%s  (%d shots, %d sites%s)" % (
        sid0.replace("data_", ""), n_shots, n_sites,
        ", %d runs stacked" % len(dirs) if len(dirs) > 1 else "")

    # ---- per-site fidelity + d' (analytic Gaussian overlap, NOT the pooled optimal cut)
    fid = np.full(n_sites, np.nan)
    dpr = np.full(n_sites, np.nan)
    empty_med = np.full(n_sites, np.nan)
    atom_med = np.full(n_sites, np.nan)
    for i in range(n_sites):
        try:
            mu_e, s_e, mu_a, s_a, t, _ = pi._two_gauss_split(I1[:, i])
        except Exception:
            continue
        if not np.isfinite([mu_e, s_e, mu_a, s_a]).all() or s_e <= 0 or s_a <= 0:
            continue
        fid[i] = 1.0 - pi._overlap_infid(mu_e, s_e, mu_a, s_a)
        dpr[i] = (mu_a - mu_e) / np.sqrt((s_e ** 2 + s_a ** 2) / 2)
        empty_med[i] = mu_e
        atom_med[i] = mu_a

    # ---- per-site survival
    sv = np.full(n_sites, np.nan)
    if L2 is not None:
        loaded = L1.sum(axis=0)
        joint = (L1 & L2).sum(axis=0)
        sv = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)

    # Fidelity crowds within ~1e-5 of 1, so a linear fidelity scale is unreadable (the colourbar
    # collapses to an offset like 1e-5+9.999e-1). Plot the INFIDELITY on a log colour scale
    # instead -- dark = better -- which resolves 1e-6 from 1e-2 on one map.
    f_ok = np.isfinite(fid)
    infid = np.where(f_ok, np.maximum(1.0 - fid, 1e-7), np.nan)
    _map_fig(plt, gx, gy, infid,
             "Per-site imaging INFIDELITY (analytic Gaussian overlap; dark = better)\n%s\n"
             "fidelity median %.5f   mean %.5f   %.1f%% of sites >= 0.995"
             % (tag, np.nanmedian(fid), np.nanmean(fid),
                100.0 * np.nanmean(fid[f_ok] >= 0.995)),
             "infidelity  (1 - fidelity)", os.path.join(out_dir, "persite_fidelity.png"),
             lo=1e-7, hi=max(1e-2, float(np.nanmax(infid))), lognorm=True, flag_above=5e-3)

    if np.isfinite(sv).any():
        sem = np.nanstd(sv) / np.sqrt(np.isfinite(sv).sum())
        _map_fig(plt, gx, gy, sv,
                 "Per-site survival  P(img2 | img1)\n%s\n"
                 "array mean %.4f +- %.4f   sites < 0.95: %d   (median per-site d-prime %.2f)"
                 % (tag, np.nanmean(sv), sem, int(np.nansum(sv < 0.95)), np.nanmedian(dpr)),
                 "survival", os.path.join(out_dir, "persite_survival.png"),
                 lo=float(np.nanpercentile(sv, 5)), hi=1.0, flag_below=0.95)

    # ---- average histogram
    x = (I1 - empty_med[None, :]).ravel()
    x = x[np.isfinite(x)]
    bins = np.linspace(np.percentile(x, 0.02), np.percentile(x, 99.98), 130)
    mid = 0.5 * (bins[1:] + bins[:-1])
    tot, _ = np.histogram(x, bins=bins)
    sm = np.convolve(tot.astype(float), np.ones(5) / 5, mode="same")
    sep = float(np.nanmedian(atom_med - empty_med))
    i_e = int(np.argmax(np.where(mid < sep / 3, sm, -1)))
    i_a = int(np.argmax(np.where(mid > sep / 3, sm, -1)))
    i_v = i_e + 1 + int(np.argmin(sm[i_e + 1:i_a]))
    cut = float(mid[i_v])
    lab = x > cut
    a, b = x[lab], x[~lab]
    d = (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    ce, _ = np.histogram(b, bins=bins)
    ca, _ = np.histogram(a, bins=bins)
    norm = float(max(ce.max(), ca.max()))
    w = np.diff(bins)
    live = float(np.nanmedian(thr - empty_med))

    fig, ax = plt.subplots(figsize=(6.6, 4.1), dpi=160)
    ax.bar(mid, ce / norm, width=w, color="#4C72B0", alpha=0.85, label="empty", align="center")
    ax.bar(mid, ca / norm, width=w, color="#C44E52", alpha=0.85, label="atom", align="center")
    ax.axvline(cut, color="0.15", ls="--", lw=1.2,
               label="valley cut %.1f ADU (%.0f%% of separation)" % (cut, 100 * cut / sep))
    ax.axvline(live, color="0.55", ls=":", lw=1.2,
               label="live per-site threshold (median, %.0f%%)" % (100 * live / sep))
    ax.set_yscale("log")
    ax.set_ylim(0.5 / norm, 1.6)
    ax.set_xlabel("site intensity - that site's empty-peak median (ADU)")
    ax.set_ylabel("site-shots (peak normalized to 1)")
    ax.set_title("Average histogram\n%s   separation %.1f ADU, pooled d-prime %.2f"
                 % (tag, sep, d), fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    out = os.path.join(out_dir, "avg_histogram.png")
    _stamp(fig, out)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, "| overlap check: %d empty above cut, %d atoms below"
          % (int((b > cut).sum()), int((a <= cut).sum())))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
