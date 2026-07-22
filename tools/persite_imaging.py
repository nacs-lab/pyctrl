"""persite_imaging.py -- self-thresholding per-site imaging analysis for a scan dir.

Independent of the dashboard detection `logicals` (which are UNRELIABLE on a freshly-loaded
pattern until its grid/threshold refits -- a stale/degenerate threshold makes every site read
"atom" -> fake loading 1.0). Here we re-derive per-site occupancy from the raw
`intensities_img1/img2` by fitting a per-site empty/atom split, then report the honest metrics
from the imaging-optimization runbook:

  * loading      -- per-site P(atom in img1), array mean + CV + x/y gradient.
  * d'           -- per-site (mu_atom - mu_empty)/rms(sigma); array median.
  * fidelity     -- per-site analytic Gaussian-overlap infidelity (NOT pooled optimal-cut,
                    NOT split-at-own-threshold); fidelity = 1 - min_t 0.5*(sf_e + cdf_a).
  * survival     -- per-site P(img2=atom | img1=atom), array mean + SEM.

Per-site occupancy split: a robust 2-component 1-D split on that site's img1 intensities via a
Gaussian-mixture-ish EM seeded from the low/high halves, falling back to an Otsu cut. Threshold
is the crossing of the two fitted Gaussians. Requires enough shots/site (>= ~20) to be meaningful;
for a single-point run pool that site's shots across the whole run.

Usage:
  python tools/persite_imaging.py <data_dir> [<data_dir2> ...]   # concatenate same-config runs
"""
import os
import sys
import json
import numpy as np


def _load(d):
    import h5py
    sid = os.path.basename(d.rstrip("/\\"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    with h5py.File(os.path.join(d, sid + ".h5"), "r") as f:
        I1 = f["intensities_img1"][:].astype(float)
        I2 = f["intensities_img2"][:].astype(float) if "intensities_img2" in f else None
    return I1, I2, sid


def _site_xy(d):
    """(x,y) per site from the config Grid, if present."""
    sid = os.path.basename(d.rstrip("/\\"))
    try:
        cfg = json.load(open(os.path.join(d, sid + ".json")))
    except Exception:
        return None, None
    for key in ("gridLocations", "grid_locations", "Grid"):
        g = cfg.get(key)
        if g:
            a = np.asarray(g, float)
            if a.ndim == 2 and a.shape[1] >= 2:
                return a[:, 0], a[:, 1]
    return None, None


def _two_gauss_split(x):
    """Split a 1-D sample into (empty, atom) by a 2-Gaussian EM (seeded low/high halves),
    return (mu_e, s_e, mu_a, s_a, thresh, w_atom). Robust fallback to a median/Otsu cut."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None
    xs = np.sort(x)
    # seed: split at median
    mu_e, mu_a = xs[: xs.size // 2].mean(), xs[xs.size // 2:].mean()
    s_e = s_a = max(xs.std(), 1e-6)
    w = 0.5
    for _ in range(50):
        # E-step
        from math import sqrt, pi
        pe = (1 - w) * np.exp(-0.5 * ((x - mu_e) / s_e) ** 2) / (s_e * sqrt(2 * pi) + 1e-12)
        pa = w * np.exp(-0.5 * ((x - mu_a) / s_a) ** 2) / (s_a * sqrt(2 * pi) + 1e-12)
        r = pa / (pe + pa + 1e-30)
        # M-step
        wa = r.sum()
        if wa < 1 or (x.size - wa) < 1:
            break
        mu_a_new = (r * x).sum() / wa
        mu_e_new = ((1 - r) * x).sum() / (x.size - wa)
        s_a_new = np.sqrt((r * (x - mu_a_new) ** 2).sum() / wa) + 1e-6
        s_e_new = np.sqrt(((1 - r) * (x - mu_e_new) ** 2).sum() / (x.size - wa)) + 1e-6
        w_new = wa / x.size
        if (abs(mu_a_new - mu_a) + abs(mu_e_new - mu_e)) < 1e-4:
            mu_e, s_e, mu_a, s_a, w = mu_e_new, s_e_new, mu_a_new, s_a_new, w_new
            break
        mu_e, s_e, mu_a, s_a, w = mu_e_new, s_e_new, mu_a_new, s_a_new, w_new
    if mu_a < mu_e:  # keep atom = brighter
        mu_e, mu_a = mu_a, mu_e
        s_e, s_a = s_a, s_e
        w = 1 - w
    # threshold = equal-posterior crossing (approx: midpoint weighted by sigmas)
    thr = (mu_e * s_a + mu_a * s_e) / (s_e + s_a)
    return mu_e, s_e, mu_a, s_a, thr, w


def _overlap_infid(mu_e, s_e, mu_a, s_a):
    from scipy.special import ndtr
    ts = np.linspace(min(mu_e, mu_a) - 4 * max(s_e, s_a), max(mu_e, mu_a) + 4 * max(s_e, s_a), 800)
    fp = 1 - ndtr((ts - mu_e) / s_e)   # empty above t
    fn = ndtr((ts - mu_a) / s_a)       # atom below t
    return float(np.min(0.5 * (fp + fn)))


def analyze(dirs):
    I1s, I2s = [], []
    sid0 = None
    for d in dirs:
        I1, I2, sid = _load(d)
        sid0 = sid0 or sid
        I1s.append(I1)
        I2s.append(I2 if I2 is not None else np.full_like(I1, np.nan))
    I1 = np.concatenate(I1s, 0)
    I2 = np.concatenate(I2s, 0)
    nsh, nsit = I1.shape
    x, y = _site_xy(dirs[0])

    occ1 = np.zeros((nsh, nsit), bool)
    occ2 = np.zeros((nsh, nsit), bool)
    dprime = np.full(nsit, np.nan)
    infid = np.full(nsit, np.nan)
    for k in range(nsit):
        r = _two_gauss_split(I1[:, k])
        if r is None:
            continue
        mu_e, s_e, mu_a, s_a, thr, w = r
        occ1[:, k] = I1[:, k] >= thr
        occ2[:, k] = I2[:, k] >= thr
        denom = np.sqrt((s_e ** 2 + s_a ** 2) / 2)
        dprime[k] = (mu_a - mu_e) / denom if denom > 0 else np.nan
        infid[k] = _overlap_infid(mu_e, s_e, mu_a, s_a)

    load_site = occ1.mean(0)
    load = load_site.mean()
    cv = load_site.std() / load if load > 0 else np.nan
    loaded = occ1.sum(0)
    joint = (occ1 & occ2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        surv_site = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)
    surv = np.nanmean(surv_site)
    # per-shot survival SEM
    n_surv = int(np.nansum(loaded))
    surv_sem = np.sqrt(max(surv * (1 - surv), 0) / max(n_surv, 1))
    fid = 1 - infid
    print("=" * 74)
    print("PER-SITE (self-thresholded)  scan=%s  n_shots=%d  n_sites=%d  (dirs=%d)"
          % (sid0.replace("data_", ""), nsh, nsit, len(dirs)))
    print("  LOADING     mean=%.4f  CV=%.3f" % (load, cv))
    print("  d'          median=%.3f  (%.0f%% sites d'>=4)"
          % (np.nanmedian(dprime), 100 * np.nanmean(dprime >= 4)))
    print("  FIDELITY    median=%.5f  mean=%.5f  (%.0f%% sites >=0.995)"
          % (np.nanmedian(fid), np.nanmean(fid), 100 * np.nanmean(fid >= 0.995)))
    print("  SURVIVAL    mean=%.4f +- %.4f (SEM, n_surv=%d)" % (surv, surv_sem, n_surv))
    if x is not None and len(x) == nsit:
        gx = np.corrcoef(x, load_site)[0, 1]
        gy = np.corrcoef(y, load_site)[0, 1]
        print("  GRADIENT    corr(load,x)=%.3f  corr(load,y)=%.3f" % (gx, gy))
    print("  PICK_JSON " + json.dumps({"load": float(load), "cv": float(cv),
          "dprime_med": float(np.nanmedian(dprime)), "fid_med": float(np.nanmedian(fid)),
          "surv": float(surv), "surv_sem": float(surv_sem)}))
    print("=" * 74)
    return {"load": load, "surv": surv, "fid_med": float(np.nanmedian(fid)),
            "dprime_med": float(np.nanmedian(dprime))}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: persite_imaging.py <data_dir> [<data_dir2> ...]")
        sys.exit(2)
    analyze(sys.argv[1:])
