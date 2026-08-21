"""defocus_stack.py -- reconstruct a 3-D image of a tweezer from ONE loading-defocus-swept scan.

Companion to ``YbScans/DefocusStackScan.py``, which sweeps the loading defocus (ANSI z4) PER SHOT
inside a single scan (the run loop rewrites the loading hologram's z4 Zernike before each shot --
``slm_runtime.make_slm_defocus_pre_cb`` -> ``SlmScanSession.set_defocus``).  Every shot therefore
LOADS atoms at its own focal plane, and the frames stack along z.

WHAT THE STACK IS.  The science camera's focal plane is FIXED; z4 moves the TRAP plane.  So the
z-stack is the imaging system's axial response to an atom (a point source) walked through focus,
multiplied by the loading/survival envelope at that plane.  Two products, both built by averaging
per-site crops (~1000 sites x shots per plane, so the SNR is far above one site's):

  * CONDITIONAL stack -- crops of sites that actually hold an atom.  Shape only = the axial PSF.
  * UNCONDITIONAL stack -- every site crop.  Amplitude ~ loading x brightness = the axial
    envelope of the whole load+image chain.

Loaded sites are self-thresholded per shot (median + k*robust-sigma of the crop-sum distribution),
NOT taken from the dashboard logicals: dim defocused shots contaminate the per-pattern threshold
registry (bug-threshold-dim-scan-contamination), so the registry thresholds cannot be trusted
inside this scan.  Re-anchor the registry afterwards with ``reanchor``.

Axial scale: 1 rad of z4 = 0.8 um (rig value, 2026-08-07; the train3d.UM_PER_RAD model says 0.905
-- see the open-rearr-zernike-depth-um-per-rad memory).

    # yb_analysis env
    python tools/defocus_stack.py analyze                      # newest z4stack scan
    python tools/defocus_stack.py analyze --scan-id 20260807_1122xx
    # base python (zmq): bright run at the production plane -> re-anchor thresholds
    python tools/defocus_stack.py reanchor --shots 60 --z4 -5
"""

import argparse
import glob
import json
import os
import sys

URL = "tcp://127.0.0.1:1408"
REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PYCTRL = os.path.join(REPO, "pyctrl")
# Commanded ANSI z4 (rad) -> axial displacement. 0.8 um/rad per the user 2026-08-07 (the
# train3d.UM_PER_RAD model value is 0.905; the rig number wins).
UM_PER_RAD = 0.8
DEFOCUS_PATH = ("SLM", "LoadingDefocus")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _pyctrl_path():
    for p in (PYCTRL, os.path.join(PYCTRL, "lib"), os.path.join(PYCTRL, "YbExptCtrl"),
              os.path.join(PYCTRL, "YbScans"), os.path.join(PYCTRL, "YbSeqs")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _data_root():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from yb_analysis import config
    return config.DATA_DIR


# =========================================================================== #
# scan lookup
# =========================================================================== #
def _find_scan(root, scan_id=None):
    """(scan_dir, file_id). ``scan_id`` may be the 14-digit id or the ``YYYYmmdd_HHMMSS`` file id;
    None -> the newest scan whose sidecar sweeps ``SLM.LoadingDefocus``."""
    if scan_id:
        sid = str(scan_id).replace("_", "")
        fid = "%s_%s" % (sid[:8], sid[8:])
        d = os.path.join(root, sid[:8], "data_%s" % fid)
        if not os.path.isdir(d):
            raise SystemExit("no scan dir %s" % d)
        return d, fid
    for d in sorted(glob.glob(os.path.join(root, "*", "data_*")), reverse=True):
        fid = os.path.basename(d)[5:]
        try:
            if _sweep_values(_sidecar(d, fid)) is not None:
                return d, fid
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit("no defocus-swept scan found under %s" % root)


def _sidecar(scan_dir, file_id):
    with open(os.path.join(scan_dir, "data_%s.json" % file_id), "r") as f:
        return json.load(f)


def _sweep_values(sc):
    """The swept loading-defocus values (per SCAN POINT, 1-based index order), or None."""
    try:
        params = sc["ScanGroup"]["base"]["vars"]["params"]
    except (KeyError, TypeError):
        return None
    for node in (params or []):
        cur = node
        for key in DEFOCUS_PATH:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, list) and cur:
            return [float(v) for v in cur]
    return None


def _grid(sc, mean_img):
    """Per-site (row, col) pixel centres. The sidecar's X/Y naming is ambiguous, so the
    assignment is decided by which one lands on the bright sites."""
    import numpy as np
    gx = np.asarray(sc["initGridLocationsX"], float)
    gy = np.asarray(sc["initGridLocationsY"], float)
    h, w = mean_img.shape

    def score(r, c):
        m = (r > 2) & (r < h - 3) & (c > 2) & (c < w - 3)
        if not m.any():
            return -1.0
        return float(np.mean(mean_img[np.round(r[m]).astype(int), np.round(c[m]).astype(int)]))
    if score(gy, gx) >= score(gx, gy):
        return gy, gx, "row=Y col=X"
    return gx, gy, "row=X col=Y"


def _loaded_mask(s, kcut):
    """Which sites hold an atom, from THIS shot's crop-sum distribution alone.

    Two-stage on purpose. Otsu splits the empty/atom bimodality (the registry thresholds are
    unusable here -- dim defocused shots contaminate them, and a plain median+k*MAD cut collapses
    at the ~50% fill this array runs at, because the median then sits INSIDE the atom population).
    A plane with no atoms has no bimodality, so Otsu would split pure noise; the empty-class noise
    floor (median + kcut*sigma of everything below the Otsu cut) is applied as well and is what
    correctly returns ~nothing there."""
    import numpy as np
    from skimage.filters import threshold_otsu
    try:
        t_otsu = float(threshold_otsu(s))
    except Exception:  # noqa: BLE001 - degenerate distribution
        t_otsu = float(np.median(s))
    lo = s[s <= t_otsu]
    if lo.size:
        med = float(np.median(lo))
        sigma = 1.4826 * float(np.median(np.abs(lo - med))) or 1.0
        t_noise = med + kcut * sigma
    else:
        t_noise = t_otsu
    return s > max(t_otsu, t_noise)


def _crop_index(rows, cols, shape, half):
    """Gather indices for every in-bounds site -> (row_idx, col_idx) broadcastable to
    (nsite, 2*half+1, 2*half+1). Built once, reused for every frame (vectorized crop)."""
    import numpy as np
    h, w = shape
    r = np.round(rows).astype(int)
    c = np.round(cols).astype(int)
    ok = (r >= half) & (r < h - half) & (c >= half) & (c < w - half)
    r, c = r[ok], c[ok]
    d = np.arange(-half, half + 1)
    return r[:, None, None] + d[None, :, None], c[:, None, None] + d[None, None, :], ok.sum()


# =========================================================================== #
# analyze
# =========================================================================== #
def analyze(args):
    import numpy as np
    import h5py

    root = _data_root()
    scan_dir, fid = _find_scan(root, args.scan_id)
    sc = _sidecar(scan_dir, fid)
    zvals = _sweep_values(sc)
    if zvals is None:
        raise SystemExit("%s does not sweep SLM.LoadingDefocus" % fid)
    points = np.asarray(sc["Params"], int)              # per-shot 1-based scan-point index
    nimg = int(sc.get("NumImages", 1) or 1)
    print("scan %s: %d planes z4 %+0.2f..%+0.2f, %d shots planned"
          % (fid, len(zvals), zvals[0], zvals[-1], len(points)))

    half = args.half
    n = 2 * half + 1
    with h5py.File(os.path.join(scan_dir, "data_%s.h5" % fid), "r") as f:
        imgs = f["imgs"]
        nshot = imgs.shape[0] // nimg
        if nshot == 0:
            raise SystemExit("no frames in %s" % fid)
        first = np.asarray(imgs[0], np.float32)
        rows, cols, how = _grid(sc, first)
        ridx, cidx, nsite = _crop_index(rows, cols, first.shape, half)
        print("grid: %s, %d sites in bounds, crop %dx%d px" % (how, nsite, n, n))

        u_acc = np.zeros((len(zvals), n, n), np.float64)
        c_acc = np.zeros((len(zvals), n, n), np.float64)
        u_cnt = np.zeros(len(zvals), np.int64)
        c_cnt = np.zeros(len(zvals), np.int64)
        shots = np.zeros(len(zvals), np.int64)
        # per-shot bookkeeping for the drift check (scrambled order -> shot index is time)
        sh_pt, sh_load, sh_sig = [], [], []
        for k in range(nshot):
            p = int(points[k]) - 1                      # Params is 1-INDEXED
            if p < 0 or p >= len(zvals):
                continue
            frame = np.asarray(imgs[k * nimg], np.float32)
            floor = float(np.median(frame))
            cr = frame[ridx, cidx] - floor              # (nsite, n, n), one gather
            s = cr.reshape(cr.shape[0], -1).sum(axis=1)
            sel = _loaded_mask(s, args.kcut)
            u_acc[p] += cr.sum(axis=0)
            u_cnt[p] += cr.shape[0]
            shots[p] += 1
            if sel.sum() >= args.min_sites:
                c_acc[p] += cr[sel].sum(axis=0)
                c_cnt[p] += int(sel.sum())
            sh_pt.append(p)
            sh_load.append(float(sel.mean()))
            sh_sig.append(float(s[sel].mean()) if sel.any() else np.nan)
            if (k + 1) % 100 == 0:
                print("  %d/%d shots" % (k + 1, nshot), flush=True)

    z = np.asarray(zvals, float)
    have = shots > 0
    U = np.where(u_cnt[:, None, None] > 0, u_acc / np.maximum(u_cnt, 1)[:, None, None], np.nan)
    C = np.where(c_cnt[:, None, None] > 0, c_acc / np.maximum(c_cnt, 1)[:, None, None], np.nan)
    out = {"z4": z, "cond": C.astype(np.float32), "uncond": U.astype(np.float32),
           "shots": shots, "loaded_per_shot": np.where(shots > 0, c_cnt / np.maximum(shots, 1), 0.0),
           "um_per_rad": UM_PER_RAD, "half": half, "file_id": fid, "n_sites": int(nsite),
           "pattern": str(sc.get("calibrationSource", "")).split(":")[-1],
           "shot_point": np.asarray(sh_pt, int), "shot_loaded": np.asarray(sh_load, float),
           "shot_signal": np.asarray(sh_sig, float)}
    print("planes with data: %d/%d, %d..%d shots each"
          % (have.sum(), len(z), shots[have].min() if have.any() else 0,
             shots[have].max() if have.any() else 0))

    perm = os.path.join(scan_dir, "analysis_defocus_stack")
    if not os.path.isdir(perm):
        os.makedirs(perm)
    npz = os.path.join(perm, "defocus_stack_%s.npz" % fid)
    np.savez_compressed(npz, **out)
    print("stack -> %s" % npz)
    _figures(out, perm, args)


# --------------------------------------------------------------------------- #
def _rms_radius(img):
    """Second-moment radius (px) of a background-subtracted crop."""
    import numpy as np
    a = np.array(img, float)
    if not np.isfinite(a).all():
        return np.nan
    a -= np.median(a)
    a[a < 0] = 0
    if a.sum() <= 0:
        return np.nan
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    w = a / a.sum()
    y0, x0 = (w * yy).sum(), (w * xx).sum()
    return float(np.sqrt(max((w * ((yy - y0) ** 2 + (xx - x0) ** 2)).sum() / 2.0, 0.0)))


def _figures(d, perm, args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    z, C, U = d["z4"], np.nan_to_num(d["cond"]), np.nan_to_num(d["uncond"])
    um = z * d["um_per_rad"]
    half = int(d["half"])
    ext = [-half, half, -half, half]
    tot_c = C.reshape(C.shape[0], -1).sum(axis=1)
    tot_u = U.reshape(U.shape[0], -1).sum(axis=1)
    peak_c = C.reshape(C.shape[0], -1).max(axis=1)
    sig = np.asarray([_rms_radius(C[i]) for i in range(C.shape[0])], float)
    ok = np.isfinite(sig) & (d["shots"] > 0)
    i_focus = int(np.nanargmax(np.where(ok, peak_c, -np.inf)))
    stamp = perm.replace("\\", "/")
    title = "%s | scan %s | %d planes, %d sites, %d shots" % (
        d["pattern"], d["file_id"], len(z), int(d["n_sites"]), int(d["shots"].sum()))

    # ---------- 1. overview ----------------------------------------------------------- #
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.45, wspace=0.3)
    for k, (m, ttl) in enumerate(((C[:, half, :].T, "x-z cut"), (C[:, :, half].T, "y-z cut"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="inferno",
                       extent=[z[0], z[-1], -half, half])
        ax.axvline(z[i_focus], color="c", ls="--", lw=0.8)
        ax.set_xlabel("loading defocus z4 (rad)")
        ax.set_ylabel("px")
        ax.set_title("conditional (per-atom) PSF: %s" % ttl, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(z[ok], sig[ok], ".-")
    ax.axvline(z[i_focus], color="c", ls="--", lw=0.8,
               label="brightest z4=%+0.2f (%+0.2f um)" % (z[i_focus], um[i_focus]))
    ax.set_xlabel("z4 (rad)")
    ax.set_ylabel("rms radius (px)")
    ax.set_title("spot size vs defocus", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    for y, lab in ((tot_u, "unconditional (loading x bright)"),
                   (tot_c, "conditional (per atom)"),
                   (d["loaded_per_shot"], "loaded sites/shot"),
                   (peak_c, "conditional peak ADU")):
        m = float(np.nanmax(y)) or 1.0
        ax.plot(z, np.asarray(y, float) / m, ".-", ms=3, lw=0.9, label=lab)
    ax.set_xlabel("z4 (rad)")
    ax.set_ylabel("normalized")
    ax.set_title("axial envelope", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    picks = np.linspace(0, len(z) - 1, 8).round().astype(int)
    vmax = float(np.nanpercentile(C, 99.9))
    for k, i in enumerate(picks):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        ax.imshow(C[i], cmap="inferno", vmin=0, vmax=vmax, extent=ext, origin="lower")
        ax.set_title("z4=%+0.1f (%+0.1f um)" % (z[i], um[i]), fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Tweezer axial image stack -- %s" % title, fontsize=11)
    fig.text(0.005, 0.005, stamp, fontsize=6, color="0.35")
    p1 = os.path.join(perm, "defocus_stack_overview_%s.png" % d["file_id"])
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig -> %s" % p1)

    # ---------- 2. the 3-D picture ----------------------------------------------------- #
    from skimage import measure
    V = C.copy()
    V -= np.median(V)
    V[V < 0] = 0
    if V.max() > 0:
        V /= V.max()
    fig = plt.figure(figsize=(13.5, 6))
    for k, lev in enumerate((0.5, 0.15)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        try:
            verts, faces, _, _ = measure.marching_cubes(np.ascontiguousarray(V), level=lev)
            zz = np.interp(verts[:, 0], np.arange(len(z)), um)
            tri = np.stack([zz[faces], (verts[:, 1] - half)[faces], (verts[:, 2] - half)[faces]],
                           axis=-1)
            mesh = Poly3DCollection(tri, alpha=0.8 if lev > 0.3 else 0.35)
            mesh.set_facecolor(plt.cm.inferno(0.75 if lev > 0.3 else 0.45))
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
        except Exception as e:  # noqa: BLE001
            ax.set_title("marching_cubes failed: %s" % e, fontsize=7)
        ax.set_xlim(um.min(), um.max())
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)
        ax.set_xlabel("axial (um)")
        ax.set_ylabel("y (px)")
        ax.set_zlabel("x (px)")
        ax.set_title("isosurface at %.0f%% of peak" % (100 * lev), fontsize=10)
        ax.view_init(elev=22, azim=-58)
    fig.suptitle("What a tweezer looks like on the atom camera -- %s" % title, fontsize=11)
    fig.text(0.005, 0.005, stamp, fontsize=6, color="0.35")
    p2 = os.path.join(perm, "defocus_stack_3d_%s.png" % d["file_id"])
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig -> %s" % p2)

    # ---------- 3. drift control (scrambled order -> shot index is time) --------------- #
    sp, sl = d["shot_point"], d["shot_loaded"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sl, ".", ms=2)
    if sl.size > 20:
        w = max(sl.size // 20, 1)
        axes[0].plot(np.convolve(sl, np.ones(w) / w, "same"), "-", color="tab:red",
                     label="running mean (%d shots)" % w)
        axes[0].legend(fontsize=7)
    axes[0].set_xlabel("shot index (time)")
    axes[0].set_ylabel("loaded fraction")
    axes[0].set_title("drift control: loading vs time (point order scrambled)", fontsize=9)
    axes[0].grid(alpha=0.3)
    m = sp.size // 2
    for lo, hi, lab in ((0, m, "first half"), (m, sp.size, "second half")):
        env = np.full(len(z), np.nan)
        for p in range(len(z)):
            sel = (sp[lo:hi] == p)
            if sel.any():
                env[p] = sl[lo:hi][sel].mean()
        axes[1].plot(z, env, ".-", ms=3, lw=0.9, label=lab)
    axes[1].set_xlabel("z4 (rad)")
    axes[1].set_ylabel("loaded fraction")
    axes[1].set_title("same envelope, first vs second half of the run", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.text(0.005, 0.005, stamp, fontsize=6, color="0.35")
    fig.tight_layout()
    p3 = os.path.join(perm, "defocus_stack_drift_%s.png" % d["file_id"])
    fig.savefig(p3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig -> %s" % p3)

    tdir = os.path.join(REPO, "tmp")
    if not os.path.isdir(tdir):
        os.makedirs(tdir)
    import shutil
    for src, dst in ((p2, "claude_display1.png"), (p1, "claude_display2.png"),
                     (p3, "claude_display3.png")):
        try:
            shutil.copyfile(src, os.path.join(tdir, dst))
        except Exception as e:  # noqa: BLE001
            print("copy %s failed: %s" % (dst, e))

    print("\nfocus: brightest conditional plane z4 = %+0.2f rad (%+0.2f um); "
          "min rms radius %.2f px at z4 = %+0.2f"
          % (z[i_focus], um[i_focus], np.nanmin(sig[ok]) if ok.any() else float("nan"),
             z[ok][int(np.nanargmin(sig[ok]))] if ok.any() else float("nan")))


# =========================================================================== #
# reanchor
# =========================================================================== #
def reanchor(args):
    """Bright single-plane run at the production defocus, so the per-pattern threshold
    accumulator recovers from the dim defocused shots (bug-threshold-dim-scan-contamination)."""
    _pyctrl_path()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan
    g = ScanGroup()
    rp = g.runp()
    rp.NumPerGroup = float(args.shots)
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = args.phase
    rp.loading_defocus = float(args.z4)
    did = ybStartScan("TweezerLoadingSeq", g, url=args.url, label="z4stack_rethreshold",
                      description=("Bright re-anchor after a loading-defocus stack: %d shots at "
                                   "the production plane z4=%.2f so the per-pattern threshold "
                                   "accumulator recovers from the dim defocused shots "
                                   "(bug-threshold-dim-scan-contamination)."
                                   % (args.shots, args.z4)), rep=int(args.shots))
    print("queued re-anchor: %d shots at z4=%+0.2f -> descriptor %s" % (args.shots, args.z4, did))
    return did


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase_name", choices=("analyze", "reanchor"))
    ap.add_argument("--url", default=URL)
    # analyze
    ap.add_argument("--scan-id", default=None, help="14-digit id or YYYYmmdd_HHMMSS (default: newest swept scan)")
    ap.add_argument("--half", type=int, default=20, help="crop half-width px (site pitch ~55 px)")
    ap.add_argument("--kcut", type=float, default=4.0, help="loaded-site cut in robust sigmas")
    ap.add_argument("--min-sites", type=int, default=5,
                   help="skip a shot's conditional average below this many loaded sites")
    # reanchor
    ap.add_argument("--shots", type=int, default=60)
    ap.add_argument("--z4", type=float, default=-5.0)
    ap.add_argument("--phase", default="phase/33x33_feedback11.pt")
    args = ap.parse_args()
    {"analyze": analyze, "reanchor": reanchor}[args.phase_name](args)


if __name__ == "__main__":
    main()
