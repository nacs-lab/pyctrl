"""avg_spectrum.py -- array-averaged survival spectrum for one OR MORE pooled scans.

Companion to the per-site tools (uses the SAME _persite_common pooling), so the
array-average spectrum matches the pooled per-site maps exactly. Averages P11
over all sites (loaded-weighted) at each swept freq, fits a single Lorentzian
dip, and saves a plot. Use it to caption/precede a pooled spatial map.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/avg_spectrum.py <scan_id> [<scan_id> ...] [--ref 143.184e6]
Saves avg_spectrum_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined, _bootstrap_root


def build(scan_ids, ref_hz=None, xlabel="556 freq [MHz]"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _bootstrap_root()
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian

    c = load_combined(scan_ids)
    freq = c["freq_hz"]; n_load = c["n_load"]; n_surv = c["n_surv"]
    # loaded-weighted array average P11 per freq (pool sites)
    tot_load = np.nansum(n_load, axis=0)
    tot_surv = np.nansum(n_surv, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(tot_load > 0, tot_surv / tot_load, np.nan)
        ye = np.where(tot_load > 0, np.sqrt(np.clip(y * (1 - y), 1e-6, None) / np.clip(tot_load, 1, None)), np.nan)
    good = np.isfinite(y)
    fit = fit_lorentzian(freq[good], y[good], ye[good], mode="dip")
    if fit is None:
        raise SystemExit("Lorentzian fit failed")
    center = fit["center"]; fwhm = fit["width"]; r2 = fit["r_squared"]

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.errorbar(freq / 1e6, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2,
                label="array-avg survival", zorder=5)
    ax.plot(fit["x_fit"] / 1e6, fit["y_fit"], "-", color="C3", lw=1.5,
            label="Lorentzian  R2=%.3f" % r2)
    ax.axvline(center / 1e6, color="C3", ls="--", lw=0.9)
    if ref_hz:
        ax.axvline(ref_hz / 1e6, color="k", ls=":", lw=0.9, label="ref %.3f MHz" % (ref_hz / 1e6))
    ax.set_xlabel(xlabel); ax.set_ylabel("survival (P11)")
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Array-avg spectrum  %s\ncenter %.4f MHz  FWHM %.0f kHz  R2 %.3f  |  %d shots%s"
                 % (tag, center / 1e6, fwhm / 1e3, r2, c["total_shots"],
                    ("  (%d pooled)" % len(c["scan_ids"])) if len(c["scan_ids"]) > 1 else ""),
                 fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "avg_spectrum_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("scans %s | center %.4f MHz  FWHM %.0f kHz  R2 %.3f  %d shots"
          % (c["scan_ids"], center / 1e6, fwhm / 1e3, r2, c["total_shots"]))
    if ref_hz:
        print("  delta from ref %.4f MHz = %+.1f kHz" % (ref_hz / 1e6, (center - ref_hz) / 1e3))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more scan ids to POOL")
    ap.add_argument("--ref", type=float, default=None)
    ap.add_argument("--xlabel", default="556 freq [MHz]")
    a = ap.parse_args()
    build(a.scan_ids, ref_hz=a.ref, xlabel=a.xlabel)
