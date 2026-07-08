"""persite_spectrum_heatmap.py -- per-site survival heatmap (sites x swept freq).

For one OR MORE 1-D survival-spectroscopy scans (same grid + freq axis, e.g.
several Rydberg spectrum runs at the same amp/field) build the site-resolved
spectrum: P11 survival for EACH site vs the swept frequency, as a 2D heatmap
(rows = sites sorted by their dip center, cols = freq). Multiple scans are POOLED
(reps summed per site x freq) for more shots/point.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/persite_spectrum_heatmap.py <scan_id> [<scan_id> ...] [--ref 143.184e6] [--sort dip|index]
Saves persite_spectrum_<primary_scan_id>[_combN].png into the first scan's data folder.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined


def build(scan_ids, ref_hz=None, sort="dip"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = load_combined(scan_ids)
    p11 = c["p11"]                       # (nSites, nParams)
    n_sites, n_params = c["n_sites"], c["n_params"]
    freq_mhz = c["freq_hz"] / 1e6

    order = np.arange(n_sites)
    if sort == "dip":
        dip_idx = np.full(n_sites, np.nan)
        for i in range(n_sites):
            row = p11[i]
            if np.isfinite(row).any():
                dip_idx[i] = np.nanargmin(row)
        order = np.argsort(np.where(np.isfinite(dip_idx), dip_idx, np.inf))
    p11_sorted = p11[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [freq_mhz.min(), freq_mhz.max(), 0, n_sites]
    im = ax.imshow(p11_sorted, aspect="auto", origin="lower", extent=extent,
                   cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax); cb.set_label("survival P11")
    ax.set_xlabel("556 freq [MHz]")
    ax.set_ylabel("site (sorted by dip center)" if sort == "dip" else "site index")
    if ref_hz:
        ax.axvline(ref_hz / 1e6, color="w", ls="--", lw=0.8, alpha=0.7,
                   label="ref %.3f MHz" % (ref_hz / 1e6))
        ax.legend(loc="upper right", fontsize=8)
    tag = " + ".join(c["scan_ids"]) if len(c["scan_ids"]) > 1 else c["scan_ids"][0]
    ax.set_title("Per-site survival spectrum  %s\n%d sites x %d freq pts, %d shots%s"
                 % (tag, n_sites, n_params, c["total_shots"],
                    ("  (%d scans pooled)" % len(c["scan_ids"])) if len(c["scan_ids"]) > 1 else ""))
    fig.tight_layout()
    suffix = "" if len(c["scan_ids"]) == 1 else "_comb%d" % len(c["scan_ids"])
    out = os.path.join(c["primary_dir"], "persite_spectrum_%s%s.png" % (c["scan_ids"][0], suffix))
    fig.savefig(out, dpi=130)
    print("scans %s | %d sites x %d params, %d shots pooled"
          % (c["scan_ids"], n_sites, n_params, c["total_shots"]))
    print("  freq %.3f-%.3f MHz | median site P11 range %.2f-%.2f"
          % (freq_mhz.min(), freq_mhz.max(),
             np.nanmin(np.nanmedian(p11, axis=0)), np.nanmax(np.nanmedian(p11, axis=0))))
    print("  saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_ids", nargs="+", help="one or more scan ids to POOL (same grid+freq axis)")
    ap.add_argument("--ref", type=float, default=None, help="reference freq Hz -> white dashed line")
    ap.add_argument("--sort", choices=["dip", "index"], default="dip")
    a = ap.parse_args()
    build(a.scan_ids, ref_hz=a.ref, sort=a.sort)
