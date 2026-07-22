"""reanchor_pattern_thresholds.py -- re-anchor per-pattern threshold.mat from fresh run frames.

WHY: the rearrange runtime detector (pyctrl rearrange_runtime._Detector) and the lab's ByPattern
detection score against ``yb_dashboard_state/patterns/<name>/threshold.mat``. After the imaging
migration (ND filters + amps=1 + PIDSet) those stored double-Gaussian fits are stale, so the
runtime posterior collapses (run 20260719_042510 sent n_loaded=162 of ~2070 real atoms to the SLM
server -- rearrangement silently no-oped). This tool re-fits per-site GMM-A params from named runs'
OWN raw frames on the pattern's registry grid (registry site order, box 9 / sigma 2 mask -- the
exact units both consumers use) and rewrites threshold.mat (with a .bak_<ts> backup).

Only the middle/final patterns strictly need this (the lab live-refit only re-saves the frame-0
pattern), but re-anchoring all three makes the round-0 bits honest too.

Usage:
    python tools/reanchor_pattern_thresholds.py \
        --pattern tri_3013_camfb    --frames 20260719_040452:0 20260719_042510:0 \
        --pattern kagome_res_2198   --frames 20260719_035530:1 20260719_035743:1 20260719_042510:1 \
        --pattern kagome_2078_camfb --frames 20260719_040452:1 20260719_042510:2 \
        [--dry-run]
"""
import argparse
import glob
import json
import os
import sys
import time

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
DATA_ROOT = r"D:\OneDrive - Harvard University\Documents - Yb\Data"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, REPO)

import numpy as np              # noqa: E402
import h5py                     # noqa: E402

BOX, SIGMA = 9, 2               # MUST match rearrange_runtime._BOX/_SIGMA + lab boxSize/maskSigma


def fspecial_gaussian(n, sigma):
    siz = (n - 1) / 2.0
    ax = np.arange(-siz, siz + 1)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    return h / h.sum()


def pattern_grid(name, roi):
    from yb_analysis import config as _cfg
    from yb_analysis.analysis.affine_transform import load_matrix, apply_affine_cropped
    rec = json.load(open(os.path.join(_cfg.PATH_PREFIX, "yb_dashboard_state",
                                      "patterns", name, "record.json")))
    knm = np.asarray(rec["knm"], dtype=float)
    return apply_affine_cropped(knm[:, ::-1], load_matrix(), roi)


def run_frames(scan, frame_idx):
    day = scan.split("_")[0]
    data_dir = os.path.join(DATA_ROOT, day, "data_" + scan)
    sid = os.path.basename(data_dir)
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    nimg = int(cfg["NumImages"])
    roi = cfg["roi"]
    imgs = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        with h5py.File(p, "r") as f:
            if "imgs" in f:
                imgs.append(f["imgs"][:])
    imgs = np.concatenate(imgs, axis=0)
    n_shots = imgs.shape[0] // nimg
    return imgs[: n_shots * nimg][frame_idx::nimg].astype(np.float64), roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", action="append", required=True)
    ap.add_argument("--frames", action="append", nargs="+", required=True,
                    help="per --pattern: scan_id:frame_idx tokens")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert len(a.pattern) == len(a.frames)

    from yb_analysis.detection.detect_atom import detect_atom_batch
    from yb_analysis.detection.dynamical_threshold import _fit_run_site_params
    import yb_analysis.analysis.pattern_registry as reg
    mask = fspecial_gaussian(BOX, SIGMA)

    for name, toks in zip(a.pattern, a.frames):
        groups = []
        grid = None
        for tok in toks:
            scan, fi = tok.rsplit(":", 1)
            frames, roi = run_frames(scan, int(fi))
            if grid is None:
                grid = pattern_grid(name, roi)
            _, inten = detect_atom_batch(frames, grid, np.zeros(grid.shape[0]), mask)
            groups.append((tok, inten))
        # pool only groups whose bright-peak scale agrees within 15% of the largest group
        ref = max(groups, key=lambda g: g[1].shape[0])
        ref_p90 = np.percentile(ref[1], 90)
        use = []
        for tok, inten in groups:
            p90 = np.percentile(inten, 90)
            okp = abs(p90 / ref_p90 - 1) < 0.15
            print("  %s  %-24s n=%3d  p90=%8.1f  %s"
                  % (name, tok, inten.shape[0], p90, "pool" if okp else "SKIP (scale mismatch)"))
            if okp:
                use.append(inten)
        I = np.concatenate(use, axis=0)
        thr, infid, params = _fit_run_site_params(I)
        thr = np.asarray(thr, dtype=float).ravel()
        okfit = np.array([p is not None for p in params])
        # failed fits: keep the site usable with the median threshold (posterior falls back site-wise)
        med = np.nanmedian(thr[okfit])
        thr = np.where(np.isfinite(thr) & okfit, thr, med)
        infid = np.asarray(infid, dtype=float).ravel() if infid is not None \
            else np.full(thr.size, np.nan)

        old = reg.load_pattern_thresholds(name)
        if old is not None:
            ot = np.asarray(old["thresholds"], dtype=float).ravel()
            if ot.size == thr.size:
                print("  %s  stored-vs-new threshold median ratio %.3f (old %.1f new %.1f)"
                      % (name, np.median(ot) / max(np.median(thr), 1e-9),
                         np.median(ot), np.median(thr)))
            else:
                print("  %s  stored M=%d != new M=%d" % (name, ot.size, thr.size))

        gs = np.empty(thr.size, dtype=[("params", "O")])
        for s in range(thr.size):
            p = params[s]
            gs[s]["params"] = (np.asarray(p, dtype=float).ravel()
                               if p is not None else np.array([]))
        mat = {"thresholds": thr, "infidelities": infid, "gaussFitsStruct": gs}
        print("  %s  -> M=%d fit_ok=%d pooled_shots=%d  thr median %.1f"
              % (name, thr.size, okfit.sum(), I.shape[0], np.median(thr)))
        if a.dry_run:
            print("  %s  DRY RUN (not written)" % name)
            continue
        p = reg.pattern_threshold_path(name)
        if os.path.isfile(str(p)):
            bak = str(p) + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
            import shutil
            shutil.copy2(str(p), bak)
            print("  %s  backup -> %s" % (name, os.path.basename(bak)))
        reg.save_pattern_thresholds(name, mat)
        print("  %s  WROTE %s" % (name, p))


if __name__ == "__main__":
    main()
