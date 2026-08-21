"""defocus_rayleigh.py -- measured Rayleigh range of individual tweezer spots (as imaged).

For each requested site, builds the per-plane CONDITIONAL average crop (only the shots in which
THAT site held an atom), fits a 2-D Gaussian per plane, and then fits the caustic

    w(z)^2 = w0^2 * [1 + ((z - z0)/zR)^2]                       (width fit)
    A(z)   = A0   / [1 + ((z - z0)/zR)^2]                       (peak fit, total counts conserved)

Both are reported: the width fit is the direct definition, the peak fit is the higher-SNR one and
is only valid because the total counts per atom are flat with defocus (verified: <=5% over -10..0),
i.e. the light is redistributed, not lost.

WHAT THIS IS THE RAYLEIGH RANGE *OF*: the imaged atom-fluorescence spot at 399 nm, i.e. the
IMAGING system's depth of focus, with the atom walked through focus by the 532 nm trap's z4. It is
not the 532 trap's own Rayleigh range.

    python tools/defocus_rayleigh.py --scan-id 20260807_102227            # fine 0.1-rad scan
    python tools/defocus_rayleigh.py --scan-id 20260807_104351 --sites 8  # wide scan
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import defocus_stack as ds                     # noqa: E402
import defocus_region_html as dr               # noqa: E402

UM_PER_RAD = ds.UM_PER_RAD


def _gauss2d(xy, a, x0, y0, sx, sy, off):
    x, y = xy
    return (a * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)) + off).ravel()


def _fit_plane(img):
    """(amplitude, sigma_x, sigma_y) px of a 2-D Gaussian on one crop; NaNs if it won't fit."""
    from scipy.optimize import curve_fit
    n = img.shape[0]
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n].astype(float)
    p0 = [max(float(np.nanmax(img)), 1e-3), c, c, 2.0, 2.0, float(np.nanmedian(img))]
    try:
        p, _ = curve_fit(_gauss2d, (x, y), np.nan_to_num(img).ravel(), p0=p0, maxfev=6000,
                         bounds=([0, c - 6, c - 6, 0.5, 0.5, -50],
                                 [1e4, c + 6, c + 6, n / 2.0, n / 2.0, 50]))
    except Exception:  # noqa: BLE001
        return np.nan, np.nan, np.nan
    return float(p[0]), float(abs(p[3])), float(abs(p[4]))


def _caustic_w(z, w0, z0, zR):
    return w0 * np.sqrt(1.0 + ((z - z0) / zR) ** 2)


def _lorentz(z, a0, z0, zR, off):
    return a0 / (1.0 + ((z - z0) / zR) ** 2) + off


def _zbin(z, vol, cnt, width):
    """Average the per-plane conditional crops into ``width``-rad bins.

    A single trap gets only ~5 atom images per 0.1-rad plane, which is far too few for a per-plane
    2-D Gaussian fit (widths scatter 1.0-2.6 px, amplitudes 5-60 ADU -- the caustic fit then has
    nothing to grip). Binning to ~1 rad pools ~50 atom images per point, which is what makes w(z)
    measurable, and 1 rad is still fine against a Rayleigh range of a few rad."""
    edges = np.arange(z.min(), z.max() + width, width)
    idx = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    zb, vb, nb = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any() or cnt[m].sum() == 0:
            continue
        w = cnt[m].astype(float)
        zb.append(float((z[m] * w).sum() / w.sum()))
        vb.append((vol[m] * w[:, None, None]).sum(axis=0) / w.sum())
        nb.append(int(w.sum()))
    return np.asarray(zb), np.asarray(vb), np.asarray(nb)


def analyse(scan_id=None, n_sites=4, half=12, kcut=4.0, zbin=1.0, box=None, log=print):
    import h5py
    from scipy.optimize import curve_fit

    root = ds._data_root()
    scan_dir, fid = ds._find_scan(root, scan_id)
    sc = ds._sidecar(scan_dir, fid)
    z = np.asarray(ds._sweep_values(sc), float)
    points = np.asarray(sc["Params"], int)
    nimg = int(sc.get("NumImages", 1) or 1)

    with h5py.File(os.path.join(scan_dir, "data_%s.h5" % fid), "r") as f:
        imgs = f["imgs"]
        nshot = min(imgs.shape[0] // nimg, len(points))
        rows, cols, _ = ds._grid(sc, np.asarray(imgs[0], np.float32))
        occ = dr._occupancy(imgs, points, z, rows, cols, nimg)
        if box is not None:      # the traps shown in the 3-D render, by their pixel box
            r0, r1, c0, c1 = box
            inbox = np.nonzero((rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1))[0]
            sites = inbox[np.argsort(occ[inbox])[::-1]][:n_sites]
        else:
            sites = np.argsort(occ)[::-1][:n_sites]            # best-loading traps
        log("scan %s | %d shots | sites %s (occupancy %s)"
            % (fid, nshot, sites.tolist(), np.round(occ[sites], 2).tolist()))
        ridx, cidx, _ = ds._crop_index(rows, cols, imgs.shape[1:], half)
        keep = np.nonzero((np.round(rows) >= half) & (np.round(rows) < imgs.shape[1] - half) &
                          (np.round(cols) >= half) & (np.round(cols) < imgs.shape[2] - half))[0]
        pos = {int(s): i for i, s in enumerate(keep)}
        rowi = [pos[int(s)] for s in sites if int(s) in pos]

        acc = np.zeros((len(rowi), len(z), 2 * half + 1, 2 * half + 1))
        cnt = np.zeros((len(rowi), len(z)), np.int64)
        for k in range(nshot):
            p = int(points[k]) - 1
            if p < 0 or p >= len(z):
                continue
            try:
                fr = np.asarray(imgs[k * nimg], np.float32)
            except Exception:  # noqa: BLE001
                break
            cr = fr[ridx, cidx] - float(np.median(fr))
            sel = ds._loaded_mask(cr.reshape(cr.shape[0], -1).sum(axis=1), kcut)
            for j, i in enumerate(rowi):
                if sel[i]:
                    acc[j, p] += cr[i]
                    cnt[j, p] += 1
            if (k + 1) % 200 == 0:
                log("  %d/%d shots" % (k + 1, nshot))

    out = []
    for j, i in enumerate(rowi):
        vol = acc[j] / np.maximum(cnt[j], 1)[:, None, None]
        zb, vb, nb = _zbin(z, vol, cnt[j], zbin)
        amp = np.full(len(zb), np.nan)
        w = np.full(len(zb), np.nan)
        for p in range(len(zb)):
            a, sx, sy = _fit_plane(vb[p])
            amp[p] = a
            w[p] = 2.0 * np.sqrt(sx * sy) if np.isfinite(sx) else np.nan   # 1/e^2 radius, px
        good = np.isfinite(amp) & np.isfinite(w) & (amp > 0)
        thr = 3.0 * 1.4826 * np.nanmedian(np.abs(amp[good] - np.nanmedian(amp[good])))
        sig = good & (amp > max(thr, 0.2 * np.nanmax(amp[good])))
        res = {"site": int(sites[j]), "n_planes": int(sig.sum()), "z": zb, "amp": amp, "w": w,
               "sig": sig, "shots": nb, "occ": float(occ[int(sites[j])])}
        # width fit. Primary form is the LINEARISED caustic, w^2 = A + B (z - z0)^2, i.e. a
        # parabola in z -- a least-squares polyfit with no starting guess to get wrong, from which
        # w0 = sqrt(A) at the vertex and zR = sqrt(A/B). curve_fit on the sqrt form is the check.
        try:
            m = sig & np.isfinite(w)
            c2, c1, c0f = np.polyfit(zb[m], w[m] ** 2, 2)
            if c2 <= 0:
                raise ValueError("width does not open upward (c2=%.3g)" % c2)
            z0 = -c1 / (2 * c2)
            w0sq = c0f - c1 ** 2 / (4 * c2)
            if w0sq <= 0:
                raise ValueError("negative waist^2")
            res["w0"], res["z0_w"] = np.sqrt(w0sq), z0
            res["zR_w"] = np.sqrt(w0sq / c2)
        except Exception as e:  # noqa: BLE001
            res["w0"] = res["z0_w"] = res["zR_w"] = np.nan
            log("  site %d width fit failed: %s" % (sites[j], e))
        try:
            pa, _ = curve_fit(_lorentz, zb[good], amp[good],
                              p0=[np.nanmax(amp[good]),
                                  zb[int(np.nanargmax(np.where(good, amp, -np.inf)))], 3.0, 0.0],
                              maxfev=20000)
            res["a0"], res["z0_a"], res["zR_a"] = pa[0], pa[1], abs(pa[2])
        except Exception as e:  # noqa: BLE001
            res["a0"] = res["z0_a"] = res["zR_a"] = np.nan
            log("  site %d peak fit failed: %s" % (sites[j], e))
        out.append(res)
    return fid, scan_dir, z, out


def report(fid, scan_dir, z, res, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perm = os.path.join(scan_dir, "analysis_defocus_stack")
    if not os.path.isdir(perm):
        os.makedirs(perm)
    fig, axs = plt.subplots(2, len(res), figsize=(4.2 * len(res), 7.4), squeeze=False)
    log("\n%-6s %8s %8s %10s %10s %9s %9s"
        % ("site", "w0(px)", "z0(rad)", "zR_w(rad)", "zR_w(um)", "zR_A(rad)", "zR_A(um)"))
    for k, r in enumerate(res):
        s = r["sig"]
        zb = r["z"]
        ax = axs[0][k]
        ax.plot(zb[s], r["w"][s], "o", ms=4, label="1/e^2 radius (binned)")
        if np.isfinite(r["zR_w"]):
            zz = np.linspace(z.min(), z.max(), 400)
            ax.plot(zz, _caustic_w(zz, r["w0"], r["z0_w"], r["zR_w"]), "-",
                    label="w0=%.2f px, zR=%.2f rad" % (r["w0"], r["zR_w"]))
        ax.set_title("site %d  (occ %.2f, %d bins)" % (r["site"], r["occ"], r["n_planes"]),
                     fontsize=9)
        ax.set_xlabel("z4 (rad)")
        ax.set_ylabel("1/e^2 radius (px)")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
        ax = axs[1][k]
        g = np.isfinite(r["amp"])
        ax.plot(zb[g], r["amp"][g], "o", ms=4, label="Gaussian amplitude (binned)")
        if np.isfinite(r["zR_a"]):
            zz = np.linspace(z.min(), z.max(), 400)
            ax.plot(zz, _lorentz(zz, r["a0"], r["z0_a"], r["zR_a"], 0.0), "-",
                    label="zR=%.2f rad" % r["zR_a"])
        ax.set_xlabel("z4 (rad)")
        ax.set_ylabel("peak (ADU)")
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
        log("%-6d %8.2f %8.2f %10.2f %10.2f %9.2f %9.2f"
            % (r["site"], r["w0"], r["z0_w"], r["zR_w"], r["zR_w"] * UM_PER_RAD,
               r["zR_a"], r["zR_a"] * UM_PER_RAD))
    fig.suptitle("Measured axial response per trap -- scan %s (1 rad = %.2f um)" % (fid, UM_PER_RAD))
    fig.text(0.005, 0.005, perm.replace("\\", "/"), fontsize=6, color="0.35")
    fig.tight_layout()
    p = os.path.join(perm, "rayleigh_%s.png" % fid)
    fig.savefig(p, dpi=125)
    plt.close(fig)
    log("fig -> %s" % p)
    import shutil
    shutil.copyfile(p, os.path.join(ds.REPO, "tmp", "claude_display2.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan-id", default=None)
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--half", type=int, default=12)
    ap.add_argument("--zbin", type=float, default=1.0, help="z4 bin width (rad) for the fits")
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("R0", "R1", "C0", "C1"),
                    help="restrict to traps in this pixel box (e.g. the rendered patch)")
    a = ap.parse_args()
    fid, scan_dir, z, res = analyse(scan_id=a.scan_id, n_sites=a.sites, half=a.half,
                                    zbin=a.zbin, box=a.box)
    report(fid, scan_dir, z, res)


if __name__ == "__main__":
    main()
