"""Regenerate the 'stable' site mask -- sites whose RELATIVE trap depth (depth/mean)
changed less than a threshold between an OLD and a NEW site-resolved depth measurement.

Feeds the global "analyze only these sites" option: writes the .npy the
yb_analysis.analysis.site_mask 'stable' registry name points at (default
_daily/stable_sites_lt5pct.npy), plus an index list + a spatial map.

Each depth npz must carry ``centers_mj1`` (per-site |mj|=1 Lorentzian center [Hz]),
``good`` (bool), ``grid_x``/``grid_y``. Depth is recomputed from centers with the
supplied per-scan mj=0 f0 (so a stale f0 baked into the npz doesn't matter).

  python pyctrl/tools/build_stable_site_mask.py \
      --old  _feedback33x33/_wide_depths_fb9.npz --old-f0 107.8199e6 \
      --new  _daily/fb9_depths_narrow.npz        --new-f0 107.8737e6 \
      --thresh 0.05 --out _daily/stable_sites_lt5pct.npy

Run under the yb_analysis env from the project root. The MASK_REGISTRY['stable']
path in yb_analysis/analysis/site_mask.py should match --out.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"


def trap_depth_uK(dn):
    """Differential 556 |mj|=1 light-shift -> trap depth [uK] (532 nm, mj1=1/mj0=0,
    theta=0). Only the RATIO across sites matters for the mask, so the absolute
    scale is irrelevant."""
    h, kB = 6.62607015e-34, 1.380649e-23
    al_s, al_t, al_g = 22.4, -7.6, 37.9
    T = -1.0
    ae1 = al_s - al_t * T * 1
    ae0 = al_s - al_t * T * (-2)
    return h * (0.25 * abs(al_g) * (-4 * dn / (ae1 - ae0))) / kB * 1e6


def rel_depth(npz_path, f0):
    d = np.load(npz_path, allow_pickle=True)
    depth = trap_depth_uK(2 * (f0 - d["centers_mj1"]))
    good = d["good"] & np.isfinite(depth) & (depth > 0)
    return depth, good, d["grid_x"], d["grid_y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="OLD depth npz (post-feedback baseline)")
    ap.add_argument("--old-f0", type=float, required=True, help="mj=0 f0 [Hz] for --old")
    ap.add_argument("--new", required=True, help="NEW depth npz (today's measurement)")
    ap.add_argument("--new-f0", type=float, required=True, help="mj=0 f0 [Hz] for --new")
    ap.add_argument("--thresh", type=float, default=0.05, help="rel-depth change ceiling (default 0.05)")
    ap.add_argument("--out", default=os.path.join(ROOT, "_daily", "stable_sites_lt5pct.npy"))
    a = ap.parse_args()

    d_old, g_old, gx, gy = rel_depth(a.old, a.old_f0)
    d_new, g_new, gx2, gy2 = rel_depth(a.new, a.new_f0)
    nS = d_new.size
    if d_old.size != nS:
        raise SystemExit("old (%d) / new (%d) site counts differ" % (d_old.size, nS))

    both = g_old & g_new
    n_old = d_old / d_old[g_old].mean()
    n_new = d_new / d_new[g_new].mean()
    rel = np.full(nS, np.nan)
    rel[both] = n_new[both] - n_old[both]
    stable = both & (np.abs(rel) < a.thresh)

    idx_path = os.path.splitext(a.out)[0] + "_idx.npy"
    png_path = os.path.splitext(a.out)[0] + ".png"
    np.save(a.out, stable)
    np.save(idx_path, np.where(stable)[0])

    print("both-good: %d/%d" % (int(both.sum()), nS))
    print("STABLE (|rel change| < %.0f%%): %d (%.1f%% of both-good)"
          % (100 * a.thresh, int(stable.sum()), 100 * stable.sum() / max(both.sum(), 1)))
    print("excluded: %d changed + %d fit-failed" %
          (int((both & (np.abs(rel) >= a.thresh)).sum()), int((~both).sum())))
    print("stable subset rel-depth CV (new): %.2f%%"
          % (100 * n_new[stable].std() / n_new[stable].mean()))

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(gx[~stable], gy[~stable], s=10, c="#ccc", label="excluded (%d)" % int((~stable).sum()))
    sc = ax.scatter(gx[stable], gy[stable], s=12, c=100 * rel[stable], cmap="coolwarm",
                    vmin=-100 * a.thresh, vmax=100 * a.thresh, label="stable (%d)" % int(stable.sum()))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("grid X (px)"); ax.set_ylabel("grid Y (px)")
    ax.set_title("stable sites |rel depth change| < %.0f%%: %d of %d"
                 % (100 * a.thresh, int(stable.sum()), nS))
    fig.colorbar(sc, ax=ax, label="rel-depth change [%] (new - old)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print("saved %s + %s + %s" % (a.out, idx_path, png_path))


if __name__ == "__main__":
    main()
