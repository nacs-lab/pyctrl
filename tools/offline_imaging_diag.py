"""offline_imaging_diag.py -- OFFLINE (no runs) imaging-chain diagnostics on a scan's raw frames.

Two questions, both from the 2198 verify data (or any --dir):
  (A) how much does per-shot COMMON-MODE normalization improve per-site fidelity/d'?
      (the atom blob brightness fluctuates ~0.6% shot-to-shot, img1~img2 corr 0.987 -- a global
       gain wobble the servo doesn't hold. Divide each shot's intensities by that shot's global
       loaded-site brightness, re-fit, compare.)
  (B) does a bigger/smaller integration BOX help? recompute masked intensities at several
      (box_size, sigma) from raw imgs, re-fit per-site fidelity, compare.

Per-site fidelity = analytic Gaussian-overlap infidelity (runbook metric), median/mean/worst-5%.
Reads img1 = imgs[0::2] (NumImages=2). Grid from initGridLocationsX/Y in the sidecar json.
"""
import argparse
import json
import os
import sys

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PYCTRL = os.path.join(REPO, "pyctrl")
STATE_DIR = os.path.join(PYCTRL, "tmp")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _fid_stats(params):
    import numpy as np
    from scipy.stats import norm
    fid = []
    dp = []
    for pp in params:
        if pp is None:
            continue
        mu_e, s_e, _, mu_a, s_a, _ = pp
        if not (s_e > 0 and s_a > 0) or (mu_a - mu_e) <= 0.5:
            continue
        ts = np.linspace(min(mu_e, mu_a) - 3 * max(s_e, s_a),
                         max(mu_e, mu_a) + 3 * max(s_e, s_a), 400)
        infid = np.min(0.5 * (norm.sf(ts, mu_e, s_e) + norm.cdf(ts, mu_a, s_a)))
        fid.append(1.0 - infid)
        dp.append((mu_a - mu_e) / np.sqrt((s_e ** 2 + s_a ** 2) / 2))
    fid = np.array(fid)
    dp = np.array(dp)
    w5 = fid[fid <= np.percentile(fid, 5)].mean()
    return dict(n=fid.size, mean=fid.mean(), median=float(__import__("numpy").median(fid)),
                p5=float(__import__("numpy").percentile(fid, 5)), worst5=float(w5),
                dprime_med=float(__import__("numpy").median(dp)))


def _pr(tag, s):
    print("  %-26s n=%d  median %.5f  mean %.5f  p5 %.5f  worst5%% %.5f  d'med %.2f"
          % (tag, s["n"], s["median"], s["mean"], s["p5"], s["worst5"], s["dprime_med"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", nargs="?", type=int, default=None)
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--num-images", type=int, default=2)
    a = ap.parse_args()
    data_dir = a.dir or json.load(open(os.path.join(STATE_DIR, "imaging_state_r%d.json" % a.round)))["data_dir"]

    sys.path.insert(0, REPO)
    import numpy as np
    import h5py
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params, _compute_site_intensities
    from yb_analysis.detection.hist_init import make_mask

    sid = os.path.basename(data_dir.rstrip("/\\"))
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    gy = np.asarray(cfg["initGridLocationsY"], float)
    gx = np.asarray(cfg["initGridLocationsX"], float)
    positions = np.column_stack([gy, gx])          # (M,2) [y,x]
    with h5py.File(os.path.join(data_dir, sid + ".h5"), "r") as f:
        imgs = f["imgs"][0::a.num_images]          # img1 of each shot
        I1_store = f["intensities_img1"][:]
    imgs = np.asarray(imgs)
    print("=" * 90)
    print("OFFLINE imaging diag  scan_id=%s  n_shots=%d  n_sites=%d  img %s"
          % (sid.replace("data_", ""), imgs.shape[0], positions.shape[0], imgs.shape[1:]))

    # ---- baseline: the STORED intensities (current live box/sigma) ----
    thr0, _, p0 = _fit_run_site_params(I1_store)
    base = _fid_stats(p0)
    print("\n(baseline = stored intensities, current live box/sigma)")
    _pr("BASELINE", base)

    # ---- (A) per-shot common-mode normalization on the stored intensities ----
    # global per-shot brightness factor from loaded sites (above the site cut), normalized to 1.
    L = I1_store > thr0[None, :]
    mu_e0 = np.array([pp[0] if pp is not None else np.nan for pp in p0])
    mu_a0 = np.array([pp[3] if pp is not None else np.nan for pp in p0])
    good = np.isfinite(mu_a0) & ((mu_a0 - mu_e0) > 1.0)
    # per-shot atom-mean over loaded good sites (background-subtracted by site empty mean)
    sig = I1_store - np.nan_to_num(mu_e0)[None, :]          # atom signal above empty baseline
    shot_fac = np.full(imgs.shape[0], np.nan)
    for k in range(imgs.shape[0]):
        m = L[k] & good
        if m.sum() > 30:
            shot_fac[k] = sig[k, m].mean()
    shot_fac = shot_fac / np.nanmean(shot_fac)             # normalize to 1
    shot_fac[~np.isfinite(shot_fac)] = 1.0
    # apply: divide the atom-signal part by the shot factor, add empty baseline back
    I1_cm = sig / shot_fac[:, None] + np.nan_to_num(mu_e0)[None, :]
    _, _, p_cm = _fit_run_site_params(I1_cm)
    cm = _fid_stats(p_cm)
    print("\n(A) per-shot COMMON-MODE normalization (divide by shot global brightness):")
    _pr("common-mode normalized", cm)
    print("      -> median %+.4f  mean %+.4f  worst5%% %+.4f  d' %+.2f  vs baseline"
          % (cm["median"] - base["median"], cm["mean"] - base["mean"],
             cm["worst5"] - base["worst5"], cm["dprime_med"] - base["dprime_med"]))

    # ---- (B) box-size / sigma sweep from raw imgs ----
    print("\n(B) integration BOX sweep (recompute masked intensities from raw imgs):")
    configs = [(9, 2.0), (11, 3.0), (13, 3.5), (15, 4.0), (17, 4.5), (19, 5.0)]
    for bs, sg in configs:
        mask = make_mask(bs, sg)
        Ib = _compute_site_intensities(imgs, positions, mask)
        _, _, pb = _fit_run_site_params(Ib)
        s = _fid_stats(pb)
        _pr("box=%2d sigma=%.1f%s" % (bs, sg, "  <- current" if bs == 13 else ""), s)
    print("=" * 90)


if __name__ == "__main__":
    main()
