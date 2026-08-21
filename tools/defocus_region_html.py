"""defocus_region_html.py -- interactive 3-D volume of a FEW tweezer spots vs loading defocus.

Crops a small xy box around an N x N patch of neighbouring traps, keeps the WHOLE z4 range, averages
the frames of each plane, and writes one standalone plotly page (plotly.js inlined, no network).
Rotate / zoom / slice the spots directly.

Voxel budget is the only real constraint (a browser volume renders comfortably to a few 100k
voxels, and plotly serializes x/y/z/value per voxel): 2 traps x 51 planes x ~85 px ~ 370k. Use
``--zstride``/``--traps`` to trade axial sampling against field of view.

    # yb_analysis env; safe mid-scan
    python tools/defocus_region_html.py                          # -> tmp/claude_display.html
    python tools/defocus_region_html.py --traps 3 --zstride 3 --scan-id 20260807_102227
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import defocus_stack as ds                     # noqa: E402  (scan lookup + grid orientation)

UM_PER_RAD = ds.UM_PER_RAD
OUT = os.path.join(ds.REPO, "tmp", "claude_display.html")


def _occupancy(imgs, points, z, rows, cols, nimg, sample=90, half=10, kcut=4.0, window=3.0):
    """Per-site loaded fraction measured NEAR FOCUS ONLY.

    The patch must be chosen by WHICH TRAPS ACTUALLY LOAD, not by geometry: a geometrically-perfect
    N x N block can contain traps that never hold an atom, and the render then shows one lonely
    spot. But occupancy is itself DEFOCUS-CONFOUNDED -- a defocused atom is spread out and fails
    detection, so a site looks empty simply for being off-plane. Averaging over the whole sweep
    therefore measures "how much of the sweep is in focus", not "does this trap load" (on the
    -30..+10 sweep it reads ~0.06 vs ~0.19 near focus). So: locate the best plane from the sampled
    shots first, then score sites using only shots within ``window`` rad of it."""
    h, w = imgs.shape[1:]
    inb = np.nonzero((np.round(rows) >= half) & (np.round(rows) < h - half) &
                     (np.round(cols) >= half) & (np.round(cols) < w - half))[0]
    ridx, cidx, _ = ds._crop_index(rows, cols, (h, w), half)
    ks = np.unique(np.linspace(0, min(imgs.shape[0] // nimg, len(points)) - 1,
                               sample).astype(int))
    masks, zs = [], []
    for k in ks:
        try:
            fr = np.asarray(imgs[int(k) * nimg], np.float32)
        except Exception:  # noqa: BLE001
            break
        cr = fr[ridx, cidx] - float(np.median(fr))
        masks.append(ds._loaded_mask(cr.reshape(cr.shape[0], -1).sum(axis=1), kcut))
        zs.append(float(z[int(points[int(k)]) - 1]))
    if not masks:
        return np.zeros(len(rows))
    M = np.asarray(masks, float)
    zs = np.asarray(zs)
    per_shot = M.mean(axis=1)
    z_best = float(zs[int(np.argmax(per_shot))])        # the in-focus plane, from the data
    near = np.abs(zs - z_best) <= window
    occ = M[near].mean(axis=0) if near.any() else M.mean(axis=0)
    full = np.zeros(len(rows))          # scatter back to FULL site indexing (rows/cols order)
    full[inb] = occ
    return full


def _patch(rows, cols, traps, margin, occ=None):
    """Pixel box around a FULL ``traps`` x ``traps`` block of neighbouring traps.

    Not simply "the sites nearest the array centre": this array has a hole at its exact centre
    (the physical DC spot), so that rule straddles the gap and returns a box 4x too wide. Instead
    pick the seed site whose neighbourhood is DENSEST within the block radius, which lands on real,
    contiguous traps."""
    P = np.c_[rows, cols]
    D = np.hypot(P[:, 0][:, None] - P[:, 0], P[:, 1][:, None] - P[:, 1])
    np.fill_diagonal(D, np.inf)
    pitch = float(np.median(D.min(axis=1)))
    # a square BLOCK anchored at the seed (seed .. seed + (traps-1) pitches in each axis), not a
    # radius -- a radius that covers the 2x2 diagonal also swallows the next ring.
    span = (traps - 1) * pitch
    tol = 0.35 * pitch
    inr = ((rows[None, :] >= rows[:, None] - tol) & (rows[None, :] <= rows[:, None] + span + tol))
    inc = ((cols[None, :] >= cols[:, None] - tol) & (cols[None, :] <= cols[:, None] + span + tol))
    block = inr & inc
    n_in = block.sum(axis=1)
    full = n_in == traps * traps                        # complete blocks only
    if not full.any():
        full = n_in == n_in.max()
    if occ is None:
        rc, cc = float(np.median(rows)), float(np.median(cols))
        dist = (rows - rc) ** 2 + (cols - cc) ** 2
        best = np.lexsort((dist, ~full))[0]
    else:
        # rank complete blocks by their WEAKEST trap (so no dead site is included), then by total
        occ_in = np.where(block, occ[None, :], np.nan)
        worst = np.nanmin(np.where(block, occ[None, :], np.inf), axis=1)
        total = np.nansum(np.where(block, occ_in, 0.0), axis=1)
        best = np.lexsort((-total, -np.where(full, worst, -1)))[0]
    order = np.nonzero(block[best])[0]
    r0 = int(max(rows[order].min() - margin, 0)); r1 = int(rows[order].max() + margin)
    c0 = int(max(cols[order].min() - margin, 0)); c1 = int(cols[order].max() + margin)
    return r0, r1, c0, c1, order, pitch


def build(scan_id=None, traps=2, zstride=2, margin=16, out=OUT, max_shots=None, log=print):
    import h5py
    root = ds._data_root()
    scan_dir, fid = ds._find_scan(root, scan_id)
    sc = ds._sidecar(scan_dir, fid)
    z = np.asarray(ds._sweep_values(sc), float)
    points = np.asarray(sc["Params"], int)
    nimg = int(sc.get("NumImages", 1) or 1)

    with h5py.File(os.path.join(scan_dir, "data_%s.h5" % fid), "r") as f:
        imgs = f["imgs"]
        nshot = min(imgs.shape[0] // nimg, len(points))
        if max_shots:
            nshot = min(nshot, int(max_shots))
        if nshot == 0:
            raise SystemExit("no frames yet in %s" % fid)
        rows, cols, how = ds._grid(sc, np.asarray(imgs[0], np.float32))
        occ = _occupancy(imgs, points, z, rows, cols, nimg)
        r0, r1, c0, c1, order, pitch = _patch(rows, cols, traps, margin, occ=occ)
        log("  occupancy: array mean %.2f | chosen block %s"
            % (occ.mean(), np.round(occ[order], 2).tolist()))
        zsel = np.arange(0, len(z), zstride)
        keepz = {int(p): i for i, p in enumerate(zsel)}
        acc = np.zeros((len(zsel), r1 - r0 + 1, c1 - c0 + 1))
        cnt = np.zeros(len(zsel), np.int64)
        log("scan %s | %d shots on disk | grid %s | patch rows %d-%d cols %d-%d (%d traps) | "
            "%d of %d planes (stride %d) -> %d voxels"
            % (fid, nshot, how, r0, r1, c0, c1, len(order), len(zsel), len(z), zstride,
               len(zsel) * (r1 - r0 + 1) * (c1 - c0 + 1)))
        for k in range(nshot):
            p = int(points[k]) - 1
            i = keepz.get(p)
            if i is None:
                continue
            try:
                frame = np.asarray(imgs[k * nimg], np.float32)
            except Exception as e:  # noqa: BLE001 - writer mid-append
                log("  stopped at shot %d (%s)" % (k, e))
                break
            acc[i] += frame[r0:r1 + 1, c0:c1 + 1] - float(np.median(frame))
            cnt[i] += 1
        vol = acc / np.maximum(cnt, 1)[:, None, None]

    zz = z[zsel]
    um = zz * UM_PER_RAD
    # --- render conditioning ------------------------------------------------------------------
    # A raw volume renders as fog: per-pixel read noise is ~6 ADU and a plane averages only ~10
    # shots (~2 ADU residual), while a spot peaks at ~10-25 ADU. Rendering from the raw max with a
    # low isomin therefore paints every noise voxel and BURIES the fainter spots -- the 2-D slices
    # show 3-4 traps where the volume showed one. So: smooth by ~half a PSF width (kills isolated
    # single-pixel noise, leaves the ~3 px spots), reference the threshold to the measured noise
    # sigma rather than to the brightest voxel, and scale on a robust percentile so one hot pixel
    # cannot set the colour range.
    from scipy.ndimage import gaussian_filter
    v = np.asarray(vol, float) - np.median(vol)
    v = gaussian_filter(v, sigma=(0.6, 0.8, 0.8))
    noise = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    v[v < 0] = 0
    vmax = float(np.percentile(v, 99.98)) or 1.0
    val = np.round(255.0 * np.clip(v / vmax, 0, 1)).astype(np.uint8)
    isomin = int(np.clip(255.0 * (4.0 * noise) / vmax, 12, 120))     # 4 sigma above background
    n_spot_vox = int((val >= isomin).sum())
    log("  render: noise %.2f ADU, scale %.1f ADU (p99.98), isomin %d/255 (=%.1f ADU), "
        "%d voxels above it" % (noise, vmax, isomin, 4.0 * noise, n_spot_vox))
    yy = np.arange(vol.shape[1], dtype=np.float32)      # camera px, patch-relative
    xx = np.arange(vol.shape[2], dtype=np.float32)
    Z, Y, X = np.meshgrid(um.astype(np.float32), yy, xx, indexing="ij")

    import plotly.graph_objects as go
    import plotly.io as pio
    fig = go.Figure(go.Volume(
        x=Z.ravel(), y=Y.ravel(), z=X.ravel(), value=val.ravel(),
        isomin=isomin, isomax=255, opacity=0.35, surface_count=18, colorscale="Inferno",
        # ramp: near-threshold shells stay nearly invisible, the spot cores are solid
        opacityscale=[[0, 0.0], [0.15, 0.03], [0.4, 0.18], [0.7, 0.5], [1, 1.0]],
        colorbar=dict(title="norm. 0-255"),
        hovertemplate="axial %{x:.2f} um<br>y %{y:.0f} px<br>x %{z:.0f} px"
                      "<br>%{value}<extra></extra>"))
    fig.update_layout(
        height=820, margin=dict(l=0, r=0, t=60, b=0),
        title=("%d tweezer spots in 3-D &mdash; scan %s, %d shots, z4 %+0.1f..%+0.1f rad "
               "(%d planes, %.1f rad step). Camera focus FIXED; z4 moves the trap plane "
               "(1 rad = %.3f &micro;m)."
               % (len(order), fid, int(cnt.sum()), zz[0], zz[-1], len(zz),
                  float(np.median(np.diff(zz))) if len(zz) > 1 else 0.0, UM_PER_RAD)),
        scene=dict(xaxis=dict(title="axial z (&micro;m)", range=[float(um.min()), float(um.max())]),
                   yaxis=dict(title="y (px)"), zaxis=dict(title="x (px)"),
                   aspectmode="manual", aspectratio=dict(x=2.2, y=1.0, z=1.0),
                   camera=dict(eye=dict(x=1.9, y=1.3, z=0.9))))
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True,
                       config={"displaylogo": False})
    out = os.path.abspath(out)
    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log("html -> %s (%.1f MB)  peak %.1f ADU/px above background" % (out, os.path.getsize(out) / 1e6,
                                                                    vmax))
    perm = os.path.join(scan_dir, "analysis_defocus_stack")
    if not os.path.isdir(perm):
        os.makedirs(perm)
    np.savez_compressed(os.path.join(perm, "defocus_region_%s.npz" % fid),
                        vol=vol, z4=zz, box=np.asarray([r0, r1, c0, c1]), shots=cnt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan-id", default=None)
    ap.add_argument("--traps", type=int, default=2, help="patch size in traps (N x N)")
    ap.add_argument("--zstride", type=int, default=2, help="keep every Nth defocus plane")
    ap.add_argument("--margin", type=int, default=16, help="px margin around the patch")
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    build(scan_id=a.scan_id, traps=a.traps, zstride=a.zstride, margin=a.margin,
          out=a.out, max_shots=a.max_shots)


if __name__ == "__main__":
    main()
