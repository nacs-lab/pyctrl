"""_persite_common.py -- shared loader for per-site survival-spectroscopy tools.

Loads one OR MORE 1-D survival-spectroscopy scans (same site grid + same swept
freq axis), stacks their reps, and returns combined per-site survival arrays.
Used by persite_spectrum_heatmap.py and persite_fit_centers.py so multiple runs
(e.g. several Rydberg spectrum scans at the same amp/field) can be pooled for
more reps/point.

combined = load_combined([sid1, sid2, ...])
  -> dict(freq_hz (nParams, sorted asc), n_load (nSites,nParams),
          n_surv (nSites,nParams), p11 (nSites,nParams), n_sites, n_params,
          total_shots, xs, ys, have_xy, scan_ids)
"""
import os
import sys

import numpy as np


def _bootstrap_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


def _one_scan(scan_id):
    from yb_analysis.analysis.run_analysis import _resolve_scan_dir_from_id, _site_grid_xy
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals

    scan_dir = _resolve_scan_dir_from_id(scan_id)
    if scan_dir is None:
        raise SystemExit("could not resolve scan dir for %s" % scan_id)
    bundle = load_scan_from_path(str(scan_dir))
    scan = bundle.get("Scan") or {}
    sp, l1, l2, reps = unpack_scan_logicals(
        scan, seq_ids=bundle.get("seq_ids"), mat_path=bundle.get("mat_path"),
        logicals_img1=bundle.get("logicals_img1"),
        logicals_img2=bundle.get("logicals_img2"))
    if l2 is None or l1.size == 0:
        raise SystemExit("%s: no img2 logicals -- not a survival scan" % scan_id)
    n_sites, n_params, _ = l1.shape
    freq = np.asarray(sp, float)
    freq = (freq[:, 0] if freq.ndim == 2 else freq).astype(float)
    order = np.argsort(freq)
    freq = freq[order]
    n_load = np.nansum(l1.astype(float), 2)[:, order]           # (nSites,nParams)
    n_surv = np.nansum((l1 & l2).astype(float), 2)[:, order]
    xs, ys = _site_grid_xy(scan)
    return dict(scan_dir=str(scan_dir), freq=freq, n_load=n_load, n_surv=n_surv,
                n_sites=n_sites, n_params=n_params,
                total=int(np.nansum(reps)), xs=xs, ys=ys)


def load_combined(scan_ids):
    """Pool one or more scans (same grid + freq axis) into combined per-site P11."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    _bootstrap_root()
    if isinstance(scan_ids, str):
        scan_ids = [scan_ids]

    base = None
    n_load = n_surv = None
    total = 0
    scan_dirs = []
    for sid in scan_ids:
        d = _one_scan(sid)
        scan_dirs.append(d["scan_dir"])
        if base is None:
            base = d
            n_load = d["n_load"].copy()
            n_surv = d["n_surv"].copy()
            total = d["total"]
            continue
        # validate compatibility
        if d["n_sites"] != base["n_sites"]:
            raise SystemExit("%s: n_sites %d != %d (grid mismatch)"
                             % (sid, d["n_sites"], base["n_sites"]))
        if d["n_params"] != base["n_params"] or not np.allclose(d["freq"], base["freq"], rtol=1e-9, atol=1.0):
            raise SystemExit("%s: freq axis differs from first scan -- cannot combine" % sid)
        n_load = n_load + d["n_load"]
        n_surv = n_surv + d["n_surv"]
        total += d["total"]

    with np.errstate(invalid="ignore", divide="ignore"):
        p11 = np.where(n_load > 0, n_surv / n_load, np.nan)

    xs, ys = base["xs"], base["ys"]
    have_xy = len(xs) == base["n_sites"] and len(ys) == base["n_sites"]
    return dict(
        freq_hz=base["freq"], n_load=n_load, n_surv=n_surv, p11=p11,
        n_sites=base["n_sites"], n_params=base["n_params"], total_shots=total,
        xs=(np.asarray(xs, float) if have_xy else np.arange(base["n_sites"], dtype=float)),
        ys=(np.asarray(ys, float) if have_xy else np.zeros(base["n_sites"], dtype=float)),
        have_xy=have_xy, scan_ids=list(scan_ids), scan_dirs=scan_dirs,
        primary_dir=base["scan_dir"])
