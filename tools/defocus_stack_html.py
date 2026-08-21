"""defocus_stack_html.py -- self-contained interactive 3-D viewer for a loading-defocus stack.

Reads a (possibly still-RUNNING) ``DefocusStackScan`` and writes one standalone HTML file
(plotly.js inlined -- no network needed) with four linked views:

  1. POOLED PSF volume -- every site x every shot averaged per plane, so this is the
     highest-SNR 3-D picture of "what a tweezer looks like on the atom camera". Rotate/zoom;
     isosurface + volume rendering.
  2. PER-SITE inspector -- a dropdown over a grid of individual traps across the array, each with
     its OWN 3-D volume, so a single spot can be zoomed and inspected (astigmatism, focus offset,
     double-lobe structure).
  3. FIELD CURVATURE -- per-site best-focus z4 (brightness-weighted axial centroid) as a surface +
     map over the array. This is where the array is NOT flat in z.
  4. AXIAL CUTS -- x-z / y-z slices of the pooled volume, plus the axial envelopes.

Conditional averaging (only sites holding an atom, self-thresholded per shot -- Otsu + empty-class
noise floor, see tools/defocus_stack.py) gives the PSF SHAPE; the unconditional average carries the
loading envelope.

    # yb_analysis env; safe to run mid-scan (partial frames are read as far as they exist)
    python tools/defocus_stack_html.py                       # newest swept scan -> tmp/claude_display.html
    python tools/defocus_stack_html.py --scan-id 20260807_102227 --out tmp/claude_display.html
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import defocus_stack as ds                                  # noqa: E402  (shared scan/grid/cut code)

REPO = ds.REPO
UM_PER_RAD = ds.UM_PER_RAD
# Axial sampling for the 3-D views only (the 2-D cuts/curves keep every plane). 0.2 rad still
# oversamples a feature ~10 rad wide, and every plane doubles the page's voxel count.
Z_STRIDE = 2


# =========================================================================== #
# accumulate
# =========================================================================== #
def _site_pitch(rows, cols, sample=200):
    """Median nearest-neighbour site spacing (px) -- the real lattice pitch, measured from the
    grid itself rather than assumed."""
    idx = np.linspace(0, len(rows) - 1, min(sample, len(rows))).astype(int)
    d = np.hypot(rows[idx][:, None] - rows[None, :], cols[idx][:, None] - cols[None, :])
    d[d == 0] = np.inf
    return float(np.median(d.min(axis=1)))


def _region_box(rows, cols, n_traps, shape, margin=18):
    """Pixel box (r0, r1, c0, c1) covering an ``n_traps`` x ``n_traps`` patch of REAL neighbouring
    traps near the array centre -- the sub-array region rendered as one volume, so several spots
    are seen at once with their true spacing (~55 px pitch)."""
    rc, cc = float(np.median(rows)), float(np.median(cols))
    pitch = _site_pitch(rows, cols)
    span = 0.5 * (n_traps - 1) * pitch + margin
    r0 = int(max(np.floor(rc - span), 0)); r1 = int(min(np.ceil(rc + span), shape[0] - 1))
    c0 = int(max(np.floor(cc - span), 0)); c1 = int(min(np.ceil(cc + span), shape[1] - 1))
    return r0, r1, c0, c1


def load_stack(scan_id=None, half=20, site_half=12, n_sites_side=4, kcut=4.0, min_sites=5,
               max_shots=None, region_traps=3, region_bin=2, log=print):
    import h5py
    root = ds._data_root()
    scan_dir, fid = ds._find_scan(root, scan_id)
    sc = ds._sidecar(scan_dir, fid)
    z = np.asarray(ds._sweep_values(sc), float)
    if z is None or not z.size:
        raise SystemExit("%s does not sweep SLM.LoadingDefocus" % fid)
    points = np.asarray(sc["Params"], int)
    nimg = int(sc.get("NumImages", 1) or 1)
    nz, n, m = len(z), 2 * half + 1, 2 * site_half + 1

    with h5py.File(os.path.join(scan_dir, "data_%s.h5" % fid), "r") as f:
        imgs = f["imgs"]
        nshot = min(imgs.shape[0] // nimg, len(points))
        if max_shots:
            nshot = min(nshot, int(max_shots))
        if nshot == 0:
            raise SystemExit("no frames yet in %s" % fid)
        first = np.asarray(imgs[0], np.float32)
        rows, cols, how = ds._grid(sc, first)
        ridx, cidx, nsite = ds._crop_index(rows, cols, first.shape, half)
        # sites that survived the bounds cut, in the same order as the crops
        keep = np.nonzero((np.round(rows) >= half) & (np.round(rows) < first.shape[0] - half) &
                          (np.round(cols) >= half) & (np.round(cols) < first.shape[1] - half))[0]
        log("scan %s: %d planes z4 %+0.2f..%+0.2f | %d shots on disk | grid %s, %d sites"
            % (fid, nz, z[0], z[-1], nshot, how, nsite))

        # the individual traps offered in the per-site inspector: an n x n grid spanning the array
        sel_sites = _pick_sites(rows[keep], cols[keep], n_sites_side)
        sl = slice(half - site_half, half + site_half + 1)

        r0, r1, c0, c1 = _region_box(rows[keep], cols[keep], region_traps, first.shape)
        rb = ((r1 - r0 + 1) // region_bin, (c1 - c0 + 1) // region_bin)
        reg_acc = np.zeros((nz,) + rb); reg_cnt = np.zeros(nz, np.int64)
        log("  sub-array region: rows %d-%d, cols %d-%d -> %dx%d px binned %dx"
            % (r0, r1, c0, c1, rb[0], rb[1], region_bin))

        u_acc = np.zeros((nz, n, n)); u_cnt = np.zeros(nz, np.int64)
        c_acc = np.zeros((nz, n, n)); c_cnt = np.zeros(nz, np.int64)
        s_acc = np.zeros((len(sel_sites), nz, m, m)); s_cnt = np.zeros((len(sel_sites), nz), np.int64)
        sig_acc = np.zeros((nsite, nz)); sig_cnt = np.zeros((nsite, nz), np.int64)
        shots = np.zeros(nz, np.int64)
        sh_pt, sh_load = [], []
        for k in range(nshot):
            p = int(points[k]) - 1
            if p < 0 or p >= nz:
                continue
            try:
                frame = np.asarray(imgs[k * nimg], np.float32)
            except Exception as e:  # noqa: BLE001 - writer is mid-append; stop cleanly
                log("  stopped at shot %d (%s)" % (k, e))
                break
            floor = float(np.median(frame))
            cr = frame[ridx, cidx] - floor
            s = cr.reshape(cr.shape[0], -1).sum(axis=1)
            sel = ds._loaded_mask(s, kcut)
            # sub-array region: UNCONDITIONAL average (each site is occupied ~half the shots, so
            # averaging over shots fills every trap in the patch and shows them side by side).
            sub = (frame[r0:r0 + rb[0] * region_bin, c0:c0 + rb[1] * region_bin] - floor)
            reg_acc[p] += sub.reshape(rb[0], region_bin, rb[1], region_bin).mean(axis=(1, 3))
            reg_cnt[p] += 1
            u_acc[p] += cr.sum(axis=0); u_cnt[p] += cr.shape[0]
            shots[p] += 1
            sig_acc[sel, p] += s[sel]; sig_cnt[sel, p] += 1
            if sel.sum() >= min_sites:
                c_acc[p] += cr[sel].sum(axis=0); c_cnt[p] += int(sel.sum())
                for j, site in enumerate(sel_sites):
                    if sel[site]:
                        s_acc[j, p] += cr[site, sl, sl]; s_cnt[j, p] += 1
            sh_pt.append(p); sh_load.append(float(sel.mean()))
            if (k + 1) % 200 == 0:
                log("  %d/%d shots" % (k + 1, nshot))

    def _norm(acc, cnt):
        c = np.maximum(cnt, 1)
        out = acc / (c.reshape((-1,) + (1,) * (acc.ndim - 1)))
        out[cnt == 0] = np.nan
        return out

    per_site = _norm(s_acc.reshape(-1, m, m), s_cnt.reshape(-1)).reshape(s_acc.shape)
    sig = np.where(sig_cnt > 0, sig_acc / np.maximum(sig_cnt, 1), np.nan)
    return {
        "file_id": fid, "scan_dir": scan_dir, "z4": z, "um": z * UM_PER_RAD,
        "pattern": str(sc.get("calibrationSource", "")).split(":")[-1],
        "cond": _norm(c_acc, c_cnt), "uncond": _norm(u_acc, u_cnt),
        "shots": shots, "n_shots": int(shots.sum()), "n_sites": int(nsite),
        "half": half, "site_half": site_half,
        "region": _norm(reg_acc, reg_cnt), "region_bin": region_bin,
        "region_box": np.asarray([r0, r1, c0, c1], int),
        "per_site": per_site, "site_idx": np.asarray(sel_sites, int),
        "site_rows": rows[keep], "site_cols": cols[keep],
        "site_signal": sig, "loaded_per_shot": np.where(shots > 0, c_cnt / np.maximum(shots, 1), 0),
        "shot_point": np.asarray(sh_pt, int), "shot_loaded": np.asarray(sh_load, float),
    }


def _pick_sites(rows, cols, k):
    """Indices of a k x k grid of individual traps spanning the array (nearest real site to each
    target position), deduplicated -- these are the traps the inspector offers."""
    rt = np.linspace(rows.min(), rows.max(), k + 2)[1:-1]
    ct = np.linspace(cols.min(), cols.max(), k + 2)[1:-1]
    out = []
    for r in rt:
        for c in ct:
            i = int(np.argmin((rows - r) ** 2 + (cols - c) ** 2))
            if i not in out:
                out.append(i)
    return out


def best_focus(z, sig):
    """Per-site best-focus z4 = brightness-weighted axial centroid of the (baseline-subtracted)
    conditional signal. Robust to the noisy per-site curves a partial run gives; NaN when a site
    has too little signal."""
    s = np.array(sig, float)
    good = np.isfinite(s).sum(axis=1) >= max(int(0.5 * len(z)), 3)
    out = np.full(s.shape[0], np.nan)
    s = np.where(np.isfinite(s), s, 0.0)
    base = np.nanpercentile(np.where(s > 0, s, np.nan), 20, axis=1)
    base = np.where(np.isfinite(base), base, 0.0)
    w = np.clip(s - base[:, None], 0, None)
    # keep only the top of each site's curve -- the tails are where loading, not focus, dominates
    w = np.where(w >= 0.5 * w.max(axis=1, keepdims=True), w, 0.0)
    tot = w.sum(axis=1)
    ok = good & (tot > 0)
    out[ok] = (w[ok] * z[None, :]).sum(axis=1) / tot[ok]
    return out


# =========================================================================== #
# HTML
# =========================================================================== #
def _voxels(vol, z_um, half, crop=None, zstride=1):
    """(x, y, z, value) flat arrays for a plotly 3-D volume, in COMPACT dtypes.

    plotly.py base64-encodes numpy arrays, and a volume serializes FOUR arrays of one entry per
    voxel -- at float64 that is 32 B/voxel, which is what blew a first cut of this page up to 81 MB.
    Coordinates go out as float32 and the value as uint8 (the page is a viewer, not the numeric
    record -- the .npz next to the run data is), and the axial axis can be strided: 0.2 rad
    sampling still oversamples a feature ~10 rad wide."""
    v = np.nan_to_num(np.asarray(vol, float), nan=0.0)
    if crop is not None and crop < half:
        s = slice(half - crop, half + crop + 1)
        v = v[:, s, s]
        half = crop
    v = v[::zstride]
    zz = np.asarray(z_um, float)[::zstride]
    v = v - np.percentile(v, 20)
    v[v < 0] = 0
    v = v / (float(v.max()) or 1.0)
    px = np.arange(-half, half + 1, dtype=np.float32)
    Z, Y, X = np.meshgrid(zz.astype(np.float32), px, px, indexing="ij")
    return (Z.ravel(), Y.ravel(), X.ravel(),
            np.round(255.0 * v).astype(np.uint8).ravel(), half)


def _vol_trace(go, vol, z_um, half, name, opacity=0.12, surfaces=17, visible=True,
               crop=None, zstride=1):
    """A plotly Volume of one (nz, n, n) stack. x = axial um, y/z = camera px."""
    x, y, zc, val, _ = _voxels(vol, z_um, half, crop, zstride)
    return go.Volume(x=x, y=y, z=zc, value=val,
                     isomin=20, isomax=255, opacity=opacity, surface_count=surfaces,
                     colorscale="Inferno", name=name, visible=visible,
                     colorbar=dict(title="0-255", len=0.7),
                     hovertemplate="axial %{x:.2f} um<br>y %{y:.0f} px<br>x %{z:.0f} px"
                                   "<br>%{value}<extra></extra>")


def _iso_trace(go, vol, z_um, half, levels=(0.5, 0.2), visible=True, crop=None, zstride=1):
    x, y, zc, val, _ = _voxels(vol, z_um, half, crop, zstride)
    return go.Isosurface(x=x, y=y, z=zc, value=val,
                         isomin=int(255 * min(levels)), isomax=255, surface_count=len(levels),
                         colorscale="Inferno", opacity=0.45, visible=visible,
                         caps=dict(x_show=False, y_show=False, z_show=False),
                         colorbar=dict(title="0-255", len=0.7),
                         hovertemplate="axial %{x:.2f} um<br>y %{y:.0f} px<br>x %{z:.0f} px"
                                       "<extra></extra>")


def _scene(z_um, half, title):
    return dict(xaxis=dict(title="axial z (um) = z4 x %.3f" % UM_PER_RAD,
                           range=[float(z_um.min()), float(z_um.max())]),
                yaxis=dict(title="y (px)", range=[-half, half]),
                zaxis=dict(title="x (px)", range=[-half, half]),
                aspectmode="manual", aspectratio=dict(x=2.0, y=1.0, z=1.0),
                annotations=[], camera=dict(eye=dict(x=1.8, y=1.3, z=0.9)))


def build_html(d, out_path, log=print):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    z, um, half, sh = d["z4"], d["um"], int(d["half"]), int(d["site_half"])
    parts = []
    head = ("<h1 style='font-family:system-ui;margin:18px 0 2px'>Tweezer axial image stack "
            "&mdash; interactive</h1>"
            "<p style='font-family:system-ui;color:#555;max-width:1100px;margin:0 0 6px'>"
            "<b>%s</b> &middot; scan <code>%s</code> &middot; %d planes "
            "(z4 %+0.2f&hellip;%+0.2f rad, %.2f rad step) &middot; %d shots so far &middot; "
            "%d sites/shot. The camera focus is FIXED and z4 moves the TRAP plane, so the axial "
            "axis is the atom's position relative to the imaging focus (1 rad = %.3f &micro;m). "
            "Drag to rotate, scroll to zoom, double-click to reset.</p>"
            % (d["pattern"], d["file_id"], len(z), z[0], z[-1],
               float(np.median(np.diff(z))) if len(z) > 1 else 0.0,
               d["n_shots"], d["n_sites"], UM_PER_RAD))

    # ---- 1. pooled volume (conditional / unconditional, volume / isosurface) -------------- #
    pk = dict(crop=min(half, 14), zstride=Z_STRIDE)
    fig = go.Figure()
    fig.add_trace(_vol_trace(go, d["cond"], um, half, "conditional volume", visible=True, **pk))
    fig.add_trace(_iso_trace(go, d["cond"], um, half, visible=False, **pk))
    fig.add_trace(_vol_trace(go, d["uncond"], um, half, "unconditional volume", visible=False, **pk))
    fig.add_trace(_iso_trace(go, d["uncond"], um, half, visible=False, **pk))
    vis = [[True, False, False, False], [False, True, False, False],
           [False, False, True, False], [False, False, False, True]]
    labels = ["conditional &middot; volume", "conditional &middot; isosurface",
              "unconditional &middot; volume", "unconditional &middot; isosurface"]
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=40, b=0),
        title="1. Pooled PSF: every site &times; every shot averaged per plane (highest SNR)",
        scene=_scene(um, half, ""),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, showactive=True,
                          buttons=[dict(label=l, method="update", args=[{"visible": v}])
                                   for l, v in zip(labels, vis)])])
    parts.append(pio.to_html(fig, include_plotlyjs=True, full_html=False,
                             config={"displaylogo": False}))

    # ---- 2. per-site inspector ------------------------------------------------------------ #
    ps, sidx = d["per_site"], d["site_idx"]
    rows, cols = d["site_rows"], d["site_cols"]
    fig = go.Figure()
    for j in range(ps.shape[0]):
        fig.add_trace(_vol_trace(go, ps[j], um, sh, "site %d" % sidx[j], opacity=0.16,
                                 surfaces=15, visible=(j == 0), zstride=Z_STRIDE))
    btns = []
    for j in range(ps.shape[0]):
        v = [False] * ps.shape[0]
        v[j] = True
        btns.append(dict(label="site %d  (row %d, col %d)" % (sidx[j], rows[sidx[j]], cols[sidx[j]]),
                         method="update", args=[{"visible": v}]))
    fig.update_layout(
        height=660, margin=dict(l=0, r=0, t=40, b=0),
        title="2. ONE trap at a time &mdash; pick a site (conditional average of its own shots)",
        scene=_scene(um, sh, ""),
        updatemenus=[dict(type="dropdown", x=0.0, y=1.12, showactive=True, buttons=btns)])
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                             config={"displaylogo": False}))

    # ---- 2b. sub-array region: several REAL neighbouring traps in one volume --------------- #
    reg = d["region"]
    rb, cb = reg.shape[1], reg.shape[2]
    px_r = (np.arange(rb) - rb / 2.0) * d["region_bin"]
    px_c = (np.arange(cb) - cb / 2.0) * d["region_bin"]
    v = np.nan_to_num(np.asarray(reg, float))[::Z_STRIDE]
    v = v - np.percentile(v, 20)
    v[v < 0] = 0
    v = np.round(255.0 * v / (v.max() or 1.0)).astype(np.uint8)
    ZZ, YY, XX = np.meshgrid((um[::Z_STRIDE]).astype(np.float32),
                             px_r.astype(np.float32), px_c.astype(np.float32), indexing="ij")
    fig = go.Figure(go.Volume(
        x=ZZ.ravel(), y=YY.ravel(), z=XX.ravel(), value=v.ravel(),
        isomin=35, isomax=255, opacity=0.1, surface_count=19, colorscale="Inferno",
        colorbar=dict(title="0-255", len=0.7),
        hovertemplate="axial %{x:.2f} um<br>y %{y:.0f} px<br>x %{z:.0f} px<br>%{value}<extra></extra>"))
    r0, r1, c0, c1 = [int(x) for x in d["region_box"]]
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=40, b=0),
        title=("2b. Zoomed sub-array: real neighbouring traps (rows %d-%d, cols %d-%d, "
               "%dx binned, unconditional) &mdash; several spots at true ~55 px pitch"
               % (r0, r1, c0, c1, d["region_bin"])),
        scene=dict(xaxis=dict(title="axial z (um)",
                              range=[float(um.min()), float(um.max())]),
                   yaxis=dict(title="y (px, rel. patch centre)"),
                   zaxis=dict(title="x (px, rel. patch centre)"),
                   aspectmode="manual", aspectratio=dict(x=2.0, y=1.0, z=1.0),
                   camera=dict(eye=dict(x=1.8, y=1.3, z=0.9))))
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                             config={"displaylogo": False}))

    # ---- 3. field curvature --------------------------------------------------------------- #
    zf = best_focus(z, d["site_signal"])
    ok = np.isfinite(zf)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.52, 0.48],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("best-focus z4 per site (3-D)",
                                        "same, as an array map (px)"))
    fig.add_trace(go.Scatter3d(x=cols[ok], y=rows[ok], z=zf[ok], mode="markers",
                               marker=dict(size=3, color=zf[ok], colorscale="RdBu",
                                           colorbar=dict(title="best-focus<br>z4 (rad)", x=0.45),
                                           cmin=float(np.nanpercentile(zf, 2)),
                                           cmax=float(np.nanpercentile(zf, 98))),
                               hovertemplate="col %{x:.0f}<br>row %{y:.0f}"
                                             "<br>z4 %{z:.2f} rad<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=cols[ok], y=rows[ok], mode="markers",
                             marker=dict(size=5, color=zf[ok], colorscale="RdBu",
                                         showscale=False,
                                         cmin=float(np.nanpercentile(zf, 2)),
                                         cmax=float(np.nanpercentile(zf, 98))),
                             hovertemplate="col %{x:.0f}<br>row %{y:.0f}<extra></extra>"),
                  row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2, title="row (px)")
    fig.update_xaxes(title="col (px)", row=1, col=2)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=60, b=0),
                      title="3. Is the array flat in z? per-site best focus "
                            "(spread %.2f rad = %.2f um p2-p98)"
                            % (np.nanpercentile(zf, 98) - np.nanpercentile(zf, 2),
                               (np.nanpercentile(zf, 98) - np.nanpercentile(zf, 2)) * UM_PER_RAD),
                      scene=dict(xaxis_title="col (px)", yaxis_title="row (px)",
                                 zaxis_title="best-focus z4 (rad)"))
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                             config={"displaylogo": False}))

    # ---- 4. cuts + envelopes -------------------------------------------------------------- #
    C = np.nan_to_num(d["cond"])
    peak = C.reshape(C.shape[0], -1).max(axis=1)
    tot_u = np.nan_to_num(d["uncond"]).reshape(C.shape[0], -1).sum(axis=1)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("x-z cut (pooled)", "y-z cut (pooled)",
                                                       "axial envelopes"))
    fig.add_trace(go.Heatmap(z=C[:, half, :].T, x=z, y=np.arange(-half, half + 1),
                             colorscale="Inferno", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=C[:, :, half].T, x=z, y=np.arange(-half, half + 1),
                             colorscale="Inferno", showscale=False), row=1, col=2)
    for y, lab in ((peak / (peak.max() or 1), "conditional peak (PSF)"),
                   (tot_u / (tot_u.max() or 1), "unconditional sum (loading x bright)"),
                   (d["loaded_per_shot"] / (d["loaded_per_shot"].max() or 1), "loaded sites/shot")):
        fig.add_trace(go.Scatter(x=z, y=y, mode="lines+markers", name=lab,
                                 marker=dict(size=4)), row=1, col=3)
    for c in (1, 2):
        fig.update_xaxes(title="z4 (rad)", row=1, col=c)
        fig.update_yaxes(title="px", row=1, col=c)
    fig.update_xaxes(title="z4 (rad)", row=1, col=3)
    fig.update_layout(height=430, margin=dict(l=40, r=10, t=60, b=40),
                      title="4. Axial cuts and envelopes",
                      legend=dict(orientation="h", y=-0.25))
    parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                             config={"displaylogo": False}))

    body = ("<html><head><meta charset='utf-8'><title>Tweezer axial stack %s</title></head>"
            "<body style='margin:24px;background:#fff'>%s%s"
            "<p style='font-family:system-ui;color:#777;font-size:12px'>%s &middot; "
            "generated from pyctrl/tools/defocus_stack_html.py</p></body></html>"
            % (d["file_id"], head, "<hr style='margin:26px 0'>".join(parts),
               os.path.join(d["scan_dir"], "").replace("\\", "/")))
    out_path = os.path.abspath(out_path)
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    log("html -> %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan-id", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "tmp", "claude_display.html"))
    ap.add_argument("--half", type=int, default=20, help="pooled crop half-width (px)")
    ap.add_argument("--site-half", type=int, default=12, help="per-site crop half-width (px)")
    ap.add_argument("--sites", type=int, default=4, help="per-site inspector grid (N x N traps)")
    ap.add_argument("--kcut", type=float, default=4.0)
    ap.add_argument("--min-sites", type=int, default=5)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--region-traps", type=int, default=3,
                    help="sub-array patch size in TRAPS (N x N) for the zoomed region volume")
    ap.add_argument("--region-bin", type=int, default=2, help="lateral binning in the region volume")
    a = ap.parse_args()
    d = load_stack(scan_id=a.scan_id, half=a.half, site_half=a.site_half, n_sites_side=a.sites,
                   kcut=a.kcut, min_sites=a.min_sites, max_shots=a.max_shots,
                   region_traps=a.region_traps, region_bin=a.region_bin)
    # also drop the numeric stack next to the run data (permanent path, per lab convention)
    perm = os.path.join(d["scan_dir"], "analysis_defocus_stack")
    if not os.path.isdir(perm):
        os.makedirs(perm)
    np.savez_compressed(os.path.join(perm, "defocus_stack_%s.npz" % d["file_id"]),
                        **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})
    build_html(d, a.out)


if __name__ == "__main__":
    main()
