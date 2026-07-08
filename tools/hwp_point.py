"""hwp_point.py -- analyze ONE imaging-beam-2 HWP-orientation point.

Per 399 beam-2 Spectrum399Scan at a given half-waveplate angle, compute + save:
  (1) single-Lorentzian DIP fit -> center, FWHM, R^2, and CONTRAST (= dip depth A in
      survival = baseline y0 minus on-resonance floor y0-A).
  (2) img1 atom-vs-empty intensity SEPARATION histogram -> pooled d', per-site d'
      median, empty/atom mu+sigma, delta-mu.

Physics intent (2026-06-28): tuning the beam-2 HWP for photon-collection efficiency.
Optimized -> separation (d', delta-mu) UP (more photons collected) while spectrum
CONTRAST stays ~constant (same number of photons scattered).

Run under the yb_analysis env FROM THE PROJECT ROOT:
  python pyctrl/tools/hwp_point.py <scan_id> --angle 26
Saves fit_spectrum_<fid>.png + intensity_separation_<fid>.png into the scan dir and
prints a one-line summary + an HWP_JSON line.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _scan_dir(scan_id):
    sys.path.insert(0, REPO)
    from yb_analysis import config
    fid = scan_id if "_" in scan_id else "%s_%s" % (scan_id[:8], scan_id[8:])
    return os.path.join(config.DATA_DIR, fid[:8], "data_" + fid), fid


def fit_dip(scan_id, fid, dd, angle, ref_hz):
    """Single-Lorentzian dip; returns metrics incl. contrast = depth A. Saves figure."""
    sys.path.insert(0, REPO)
    from yb_analysis.analysis.run_analysis import analyze_scan
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian

    d = analyze_scan(scan_id, include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False)
    x = np.asarray(d["sweep"]["values"][0], float)
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    fit = fit_lorentzian(x, y, ye, mode="dip")
    if fit is None:
        raise SystemExit("Lorentzian dip fit failed")
    y0, A, x0, w = fit["params"]              # baseline, depth, center, width
    center, fwhm, r2 = fit["center"], abs(fit["width"]), fit["r_squared"]
    contrast = float(A)                       # survival dip depth (baseline - floor)
    span = x.max() - x.min()
    edge = (center <= x.min() + 0.02 * span or center >= x.max() - 0.02 * span)

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.errorbar(x / 1e6, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2, label="array-avg survival", zorder=5)
    ax.plot(fit["x_fit"] / 1e6, fit["y_fit"], "-", color="C3", lw=1.5, label="1 Lorentzian  R2=%.3f" % r2)
    ax.axvline(center / 1e6, color="C3", ls="--", lw=0.9)
    ax.set_xlabel("push-out freq [MHz]"); ax.set_ylabel("survival (P11)")
    ax.set_title("HWP %s deg -- 399 beam-2 spectrum (scan %s)\ncenter %.2f MHz, FWHM %.1f MHz, contrast %.3f, R2 %.3f"
                 % (angle, fid, center / 1e6, fwhm / 1e6, contrast, r2))
    ax.legend(); fig.tight_layout()
    out = os.path.join(dd, "fit_spectrum_%s.png" % fid)
    fig.savefig(out, dpi=110); plt.close(fig)
    return dict(n_shots=d.get("n_shots"), center_MHz=center / 1e6, fwhm_MHz=fwhm / 1e6,
                r2=float(r2), contrast=contrast, baseline=float(y0), floor=float(y0 - A),
                edge_pinned=bool(edge), fit_png=out,
                delta_kHz=((center - ref_hz) / 1e3) if ref_hz else None)


def separation(dd, fid, angle):
    sid = os.path.basename(dd)
    with h5py.File(os.path.join(dd, sid + ".h5"), "r") as f:
        I1 = f["intensities_img1"][:]
        L1 = f["logicals_img1"][:].astype(bool)
    nsite = I1.shape[1]
    I = I1.ravel(); L = L1.ravel()
    i0, i1 = I[~L], I[L]
    m0, sd0, m1, sd1 = i0.mean(), i0.std(), i1.mean(), i1.std()
    den = np.sqrt((sd0**2 + sd1**2) / 2)
    dprime = (m1 - m0) / den if den > 0 else float("nan")
    dps = []
    for s in range(nsite):
        a = I1[:, s][~L1[:, s]]; b = I1[:, s][L1[:, s]]
        if a.size >= 5 and b.size >= 5:
            dd_ = np.sqrt((a.std()**2 + b.std()**2) / 2)
            if dd_ > 0:
                dps.append((b.mean() - a.mean()) / dd_)
    dps = np.array(dps)
    dpmed = float(np.median(dps)) if dps.size else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(i0.min(), i1.min()); hi = np.percentile(i1, 99.5)
    bins = np.linspace(lo, hi, 80)
    ax.hist(i0, bins=bins, alpha=0.6, color="tab:blue", label="empty (n=%d)" % i0.size, density=True)
    ax.hist(i1, bins=bins, alpha=0.6, color="tab:red", label="atom (n=%d)" % i1.size, density=True)
    ax.set_xlabel("img1 site intensity [ADU]"); ax.set_ylabel("density")
    ax.set_title("HWP %s deg -- atom vs empty img1 (scan %s)\npooled d'=%.2f, per-site d' median=%.2f, dmu=%.1f ADU"
                 % (angle, fid, dprime, dpmed, m1 - m0))
    ax.legend(); fig.tight_layout()
    out = os.path.join(dd, "intensity_separation_%s.png" % fid)
    fig.savefig(out, dpi=110); plt.close(fig)
    return dict(empty_mu=float(m0), empty_sd=float(sd0), atom_mu=float(m1), atom_sd=float(sd1),
                dmu=float(m1 - m0), dprime_pooled=float(dprime), dprime_persite_med=dpmed, sep_png=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_id")
    ap.add_argument("--angle", required=True, help="HWP orientation in deg (label)")
    ap.add_argument("--ref", type=float, default=310e6)
    a = ap.parse_args()

    dd, fid = _scan_dir(a.scan_id)
    if not os.path.isdir(dd):
        print("no scan dir:", dd); sys.exit(2)

    f = fit_dip(a.scan_id, fid, dd, a.angle, a.ref)
    s = separation(dd, fid, a.angle)
    out = dict(angle=a.angle, scan_id=fid, **f, **s)
    print("HWP %s deg | center %.2f MHz  FWHM %.1f MHz  R2 %.3f  CONTRAST %.3f (base %.2f floor %.2f) | "
          "d' pooled %.2f persite %.2f  dmu %.1f ADU  (atom %.1f / empty %.1f)"
          % (a.angle, f["center_MHz"], f["fwhm_MHz"], f["r2"], f["contrast"], f["baseline"], f["floor"],
             s["dprime_pooled"], s["dprime_persite_med"], s["dmu"], s["atom_mu"], s["empty_mu"]))
    print("HWP_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
