"""rethreshold_verify.py -- re-threshold a single-brightness verify scan from its OWN raw
intensities, then recompute survival + per-site fidelity.

WHY: the PIDSet sweep spanned dim->bright cells, contaminating the live per-site double-Gaussian
threshold (the ~200-shot refit + the per-pattern registry threshold). The stored logicals_img1/img2
of any run taken with that stale threshold give a WRONG survival. A high-rep single-point verify is
a clean CONSTANT-brightness dataset, so re-fitting per-site thresholds on ITS OWN intensities and
re-applying them to both frames gives the honest survival + fidelity (independent of the live logicals).

Per-site fidelity = the analytic Gaussian-overlap infidelity from the img1 fit (the runbook metric,
NOT the pooled optimal-cut) averaged over sites; survival = grand-mean per-site P(img2>thr | img1>thr).

Usage:
    python tools/rethreshold_verify.py <round>          # reads tmp/imaging_state_r<round>.json
    python tools/rethreshold_verify.py --dir <data_dir>
"""
import argparse
import json
import os
import sys

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PYCTRL = os.path.join(REPO, "pyctrl")
STATE_DIR = os.path.join(PYCTRL, "tmp")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", nargs="?", type=int, default=None)
    ap.add_argument("--dir", type=str, default=None)
    a = ap.parse_args()

    if a.dir:
        data_dir = a.dir
    else:
        st = json.load(open(os.path.join(STATE_DIR, "imaging_state_r%d.json" % a.round)))
        data_dir = st["data_dir"]

    sys.path.insert(0, REPO)
    import numpy as np
    import h5py
    from scipy.stats import norm
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params

    sid = os.path.basename(data_dir.rstrip("/\\"))
    with h5py.File(os.path.join(data_dir, sid + ".h5"), "r") as f:
        I1 = f["intensities_img1"][:]
        I2 = f["intensities_img2"][:]
        L1_live = f["logicals_img1"][:].astype(bool)
        L2_live = f["logicals_img2"][:].astype(bool)
    n, M = I1.shape
    print("=" * 74)
    print("RE-THRESHOLD verify  scan_id=%s  n_shots=%d  n_sites=%d" % (sid.replace("data_", ""), n, M))

    # --- live-logical survival (the possibly-contaminated number the sweep watch reported) ---
    def survival(L1, L2):
        loaded = L1.sum(axis=0).astype(float)
        joint = (L1 & L2).sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)
        return p, loaded
    p_live, ld_live = survival(L1_live, L2_live)
    print("  LIVE logicals   : survival %.4f  (mean load %.4f)"
          % (np.nanmean(p_live), (L1_live.mean())))

    # --- re-fit per-site thresholds from THIS run's img1 intensities (clean, single-brightness) ---
    thr, inf_optcut, params = _fit_run_site_params(I1)
    ok = np.array([pp is not None for pp in params])
    print("  re-fit sites    : %d/%d fit ok" % (ok.sum(), M))

    # fresh logicals: same per-site threshold on BOTH frames
    L1 = I1 > thr[None, :]
    L2 = I2 > thr[None, :]
    p_new, ld_new = survival(L1, L2)

    # --- per-site fidelity = analytic Gaussian-overlap from the img1 fit (runbook metric) ---
    fid_sites = []
    dprime_sites = []
    for pp in params:
        if pp is None:
            continue
        mu_e, s_e, _, mu_a, s_a, _ = pp
        if not (s_e > 0 and s_a > 0):
            continue
        # min_t 0.5*(P(N(mu_e,s_e)>t) + P(N(mu_a,s_a)<t)) over a fine grid
        ts = np.linspace(min(mu_e, mu_a) - 3 * max(s_e, s_a),
                         max(mu_e, mu_a) + 3 * max(s_e, s_a), 400)
        infid = np.min(0.5 * (norm.sf(ts, mu_e, s_e) + norm.cdf(ts, mu_a, s_a)))
        fid_sites.append(1.0 - infid)
        dprime_sites.append((mu_a - mu_e) / np.sqrt((s_e ** 2 + s_a ** 2) / 2))
    fid_sites = np.array(fid_sites); dprime_sites = np.array(dprime_sites)

    print("  RE-THRESHOLD    : survival %.4f +/- %.4f (per-site SEM)  (mean load %.4f)"
          % (np.nanmean(p_new), np.nanstd(p_new) / np.sqrt(np.isfinite(p_new).sum()), L1.mean()))
    print("  per-site FIDELITY (analytic Gaussian-overlap):")
    print("       median %.5f  mean %.5f  frac>=0.995 %.3f  (n=%d sites)"
          % (np.median(fid_sites), np.mean(fid_sites),
             (fid_sites >= 0.995).mean(), fid_sites.size))
    print("  per-site d': median %.2f  mean %.2f" % (np.median(dprime_sites), np.mean(dprime_sites)))
    print("=" * 74)

    return {"survival_live": float(np.nanmean(p_live)),
            "survival_rethresh": float(np.nanmean(p_new)),
            "fid_median": float(np.median(fid_sites)),
            "fid_mean": float(np.mean(fid_sites)),
            "dprime_median": float(np.median(dprime_sites))}


if __name__ == "__main__":
    r = main()
    print("PICK_JSON " + json.dumps(r))
