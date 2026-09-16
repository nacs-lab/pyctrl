"""imaging_limiter.py -- ranked loss budget for an imaging verify scan. ANALYST TOOL.

    python pyctrl/tools/imaging_limiter.py <scan_dir> [<scan_dir> ...]

Answers "what is stopping survival here" by attributing the survival deficit (1 - survival) to causes
that can be separated from the run itself, so the analyst spends its time on judgement rather than on
re-deriving the same decomposition. Every line it prints is MEASURED from the h5; the tool never
asserts a mechanism -- naming the physics behind the dim-margin tail is the analyst's call, and
confirming it needs a temperature scan.

What it decomposes, in the order it prints:
  margin tail   survival vs img1 threshold-margin. The dimmest 1% / 5% of loaded atom-shots, their
                survival, their SHARE of all lost atoms, and how many DISTINCT sites they touch --
                the last number is what separates a broad physics tail from a few bad traps.
  worst sites   survival recovered by dropping the worst k sites (3 / 10 / 50), and whether they are
                spatially clustered (median pairwise distance of the worst 3 vs the array scale).
  threshold     survival recomputed with each site's cut moved from the live value to the pooled
                valley fraction, plus the effect on fill -- signed, so the bias direction is explicit.
  common mode   per-shot multiplicative de-trend (the 2026-07-18 memory's correction), its effect on
                d' and on survival.
  within-run    linear trend of the atom signal across the run, as a fraction of the mean.
  spatial       survival and fidelity gradients across x and y.

Costs a few seconds on a 100-shot x 1068-site run; no per-site EM (it splits on the stored labels),
which is what keeps it fast enough to sit inside a live handoff.
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _load(dirs):
    import h5py
    I1s, L1s, L2s, I2s = [], [], [], []
    thr = gx = gy = sid0 = None
    for d in dirs:
        sid = os.path.basename(d.rstrip("/\\"))
        sid0 = sid0 or sid
        cfg = json.load(open(os.path.join(d, sid + ".json")))
        with h5py.File(os.path.join(d, sid + ".h5"), "r") as f:
            I1s.append(f["intensities_img1"][:].astype(float))
            L1s.append(f["logicals_img1"][:].astype(bool))
            L2s.append(f["logicals_img2"][:].astype(bool))
            I2s.append(f["intensities_img2"][:].astype(float) if "intensities_img2" in f else None)
        thr = np.asarray(cfg["initThresholds"], float).ravel()
        gx = np.asarray(cfg["initGridLocationsX"], float).ravel()
        gy = np.asarray(cfg["initGridLocationsY"], float).ravel()
    I2 = np.vstack(I2s) if all(a is not None for a in I2s) else None
    return sid0, np.vstack(I1s), np.vstack(L1s), np.vstack(L2s), I2, thr, gx, gy


def _site_survival(L1, L2):
    loaded = L1.sum(axis=0)
    joint = (L1 & L2).sum(axis=0)
    with np.errstate(invalid="ignore"):
        sv = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)
    return sv, loaded, joint


def main(dirs):
    t0 = time.time()
    sid, I1, L1, L2, I2, thr, gx, gy = _load(dirs)
    n_shots, n_sites = I1.shape
    sv, loaded, joint = _site_survival(L1, L2)
    surv = float(np.nanmean(sv))
    deficit = 1.0 - surv
    n_loaded = int(loaded.sum())
    n_lost = int(n_loaded - joint.sum())

    print("=" * 78)
    print("IMAGING LIMITER BUDGET  %s  (%d shots x %d sites%s)"
          % (sid.replace("data_", ""), n_shots, n_sites,
             ", %d runs stacked" % len(dirs) if len(dirs) > 1 else ""))
    print("  array survival %.5f   deficit %.3f pp   loaded atom-shots %d   lost %d"
          % (surv, 100 * deficit, n_loaded, n_lost))

    # ---- 1. margin tail -------------------------------------------------------
    margin = I1 - thr[None, :]
    at = L1                                   # loaded in img1
    m = margin[at]
    surv_shot = L2[at]
    order = np.argsort(m)
    print("\n  MARGIN TAIL (loaded atom-shots ranked by img1 margin above their own threshold)")
    print("   %8s %9s %10s %12s %14s" % ("quantile", "n", "survival", "share of loss", "distinct sites"))
    site_idx = np.tile(np.arange(n_sites), (n_shots, 1))[at]
    for q in (0.01, 0.05, 0.10):
        k = max(1, int(round(q * m.size)))
        sel = order[:k]
        s_q = float(surv_shot[sel].mean())
        lost_q = int((~surv_shot[sel]).sum())
        share = lost_q / max(n_lost, 1)
        print("   %8s %9d %10.4f %12.1f%% %14d"
              % ("%.0f%%" % (100 * q), k, s_q, 100 * share, len(np.unique(site_idx[sel]))))
    rest = order[int(round(0.05 * m.size)):]
    print("   %8s %9d %10.4f" % ("rest", rest.size, float(surv_shot[rest].mean())))

    # ---- 2. worst sites -------------------------------------------------------
    print("\n  WORST SITES (survival recovered by dropping the worst k)")
    ok = np.isfinite(sv)
    rank = np.argsort(np.where(ok, sv, np.inf))
    for k in (3, 10, 50):
        keep = np.ones(n_sites, bool); keep[rank[:k]] = False
        s_k = float(np.nanmean(sv[keep & ok]))
        print("   drop %3d (%4.1f%% of array) -> %.5f   (+%.3f pp, %4.1f%% of the deficit)"
              % (k, 100.0 * k / n_sites, s_k, 100 * (s_k - surv),
                 100.0 * (s_k - surv) / deficit if deficit > 0 else 0.0))
    w = rank[:3]
    d3 = [np.hypot(gx[a] - gx[b], gy[a] - gy[b]) for i, a in enumerate(w) for b in w[i + 1:]]
    span = float(np.hypot(gx.max() - gx.min(), gy.max() - gy.min()))
    print("   worst 3: %s" % ", ".join("s%d(%.2f)" % (i, sv[i]) for i in w))
    print("   their median pairwise separation %.0f px vs array span %.0f px -> %s"
          % (np.median(d3), span, "CLUSTERED" if np.median(d3) < 0.1 * span else "scattered"))

    # ---- 2b. broad tail vs local defect ---------------------------------------
    # These are DIFFERENT causes and must not be merged: re-run the margin tail with the worst 3 sites
    # excluded. Whatever share of the loss survives that exclusion is genuinely BROAD.
    drop = set(w.tolist())
    keep_shot = ~np.isin(site_idx, list(drop))
    m2 = m[keep_shot]; s2 = surv_shot[keep_shot]
    lost2 = int((~s2).sum())
    o2 = np.argsort(m2)
    k2 = max(1, int(round(0.01 * m2.size)))
    share2 = int((~s2[o2[:k2]]).sum()) / max(lost2, 1)
    print("\n  BROAD TAIL vs LOCAL DEFECT (the two must not be merged)")
    print("   with the worst 3 sites EXCLUDED, the dimmest 1%% still carries %.1f%% of the remaining loss"
          % (100 * share2))
    print("   -> a tail touching %d sites is BROAD (array-wide physics); the %d-site cluster is LOCAL"
          % (len(np.unique(site_idx[order[:max(1, int(round(0.01 * m.size)))]])), len(drop)))
    print("   treat them separately: the broad tail is not a masking candidate, the cluster may be")

    # ---- 3. threshold placement ----------------------------------------------
    emp = np.array([np.median(I1[:, i][~L1[:, i]]) if (~L1[:, i]).any() else np.nan
                    for i in range(n_sites)])
    atm = np.array([np.median(I1[:, i][L1[:, i]]) if L1[:, i].any() else np.nan
                    for i in range(n_sites)])
    sep = atm - emp
    live_frac = float(np.nanmedian((thr - emp) / sep))
    x = (I1 - emp[None, :]).ravel()
    x = x[np.isfinite(x)]
    bins = np.linspace(np.percentile(x, 0.02), np.percentile(x, 99.98), 130)
    mid = 0.5 * (bins[1:] + bins[:-1])
    tot, _ = np.histogram(x, bins=bins)
    sm = np.convolve(tot.astype(float), np.ones(5) / 5, mode="same")
    msep = float(np.nanmedian(sep))
    i_e = int(np.argmax(np.where(mid < msep / 3, sm, -1)))
    i_a = int(np.argmax(np.where(mid > msep / 3, sm, -1)))
    valley = float(mid[i_e + 1 + int(np.argmin(sm[i_e + 1:i_a]))])
    v_frac = valley / msep
    thr2 = emp + v_frac * sep
    # Re-gate BOTH images: the live pipeline scores img1 and img2 with the same per-site threshold, so
    # moving the cut and re-gating only img1 would measure a pipeline that does not exist.
    L1b = I1 > thr2[None, :]
    L2b = (I2 > thr2[None, :]) if I2 is not None else L2
    sv2, loaded2, joint2 = _site_survival(L1b, L2b)
    print("\n  THRESHOLD PLACEMENT")
    print("   live cut at %.0f%% of separation, pooled valley at %.0f%%" % (100 * live_frac, 100 * v_frac))
    print("   survival %.5f -> %.5f (%+.3f pp)   fill %.4f -> %.4f   %s"
          % (surv, float(np.nanmean(sv2)), 100 * (float(np.nanmean(sv2)) - surv),
             L1.mean(), L1b.mean(),
             "live cut is CONSERVATIVE" if np.nanmean(sv2) > surv else "live cut INFLATES survival"))

    # ---- 4. common mode -------------------------------------------------------
    sig = np.array([I1[k][L1[k]].mean() - I1[k][~L1[k]].mean() for k in range(n_shots)])
    cv = sig.std() / sig.mean()
    lag1 = float(np.corrcoef(sig[:-1], sig[1:])[0, 1]) if n_shots > 3 else np.nan
    f = sig / sig.mean()
    I1n = (I1 - emp[None, :]) / f[:, None] + emp[None, :]
    def _dp(I):
        a = I[L1]; b = I[~L1]
        return (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    L1n = I1n > thr[None, :]
    svn, _, _ = _site_survival(L1n, L2)
    def _ac(x, k):
        y = x - x.mean()
        return float(np.sum(y[:-k] * y[k:]) / np.sum(y * y)) if len(y) > k + 2 else float("nan")
    # Is the wobble COMMON to the array or per-site noise? Compare the per-shot spread of the array
    # mean against a typical single site's own spread (the ratio the 2026-07-18 memory used).
    per_site_sd = [I1[:, i][L1[:, i]].std() for i in range(n_sites) if L1[:, i].sum() > 20]
    sig_site = float(np.nanmedian(per_site_sd)) if per_site_sd else float("nan")
    cm_frac = (sig.std() / sig_site) ** 2 if np.isfinite(sig_site) and sig_site > 0 else float("nan")
    corr12 = float("nan")
    if I2 is not None:
        s2 = np.array([I2[k][L1[k]].mean() - I2[k][~L1[k]].mean() for k in range(n_shots)])
        corr12 = float(np.corrcoef(sig, s2)[0, 1])
    print("\n  COMMON-MODE SHOT WOBBLE")
    print("   atom-signal CV %.1f%%   autocorr lag1/2/3 %.2f / %.2f / %.2f"
          % (100 * cv, lag1, _ac(sig, 2), _ac(sig, 3)))
    print("   img1 vs img2 same-shot correlation %.3f -> %s"
          % (corr12, "COMMON-MODE (one gain per shot)" if corr12 > 0.8 else "not strongly common"))
    print("   per-shot array spread is %.2f of a single site's own variance" % cm_frac)
    print("   de-trended: pooled d' %.2f -> %.2f   survival %.5f -> %.5f (%+.3f pp)"
          % (_dp(I1), _dp(I1n), surv, float(np.nanmean(svn)), 100 * (float(np.nanmean(svn)) - surv)))

    # ---- 5. within-run trend + spatial ---------------------------------------
    sh = np.arange(n_shots)
    coef = np.polyfit(sh, sig, 1)
    resid = sig - np.polyval(coef, sh)
    expl = 100 * (1 - resid.var() / sig.var()) if sig.var() > 0 else 0.0
    span = ""
    stf = os.path.join(dirs[0], "shot_time.csv")
    if os.path.isfile(stf):
        import csv
        ts = [float(r["t_unix"]) for r in csv.DictReader(open(stf)) if r.get("event") == "pre"]
        if len(ts) > 1:
            span = "  (run spans %.0f s)" % (ts[-1] - ts[0])
    print("\n  WITHIN-RUN TREND   atom signal drifts %+.1f%% of the mean across the run%s"
          % (100 * coef[0] * n_shots / sig.mean(), span))
    print("   the trend explains %.0f%% of the shot-to-shot variance -> %s"
          % (expl, "drift dominates" if expl > 50 else "the wobble is not a within-run drift"))
    for nm, g in (("x", gx), ("y", gy)):
        k = np.isfinite(sv) & np.isfinite(g)
        gs = np.polyfit(g[k], sv[k], 1)[0] * (g[k].max() - g[k].min())
        print("  SPATIAL GRADIENT   survival across %s: %+.4f" % (nm, gs))
    print("\n  (all lines above are MEASURED; the physics behind the margin tail is not asserted here)")
    print("  elapsed %.1f s" % (time.time() - t0))
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
