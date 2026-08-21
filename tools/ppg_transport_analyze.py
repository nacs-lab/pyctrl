"""ppg_transport_analyze.py -- OFFLINE re-detection for +x one-way grating transport.

Companion to ``YbScans/RearrangeDiagnostics/PPGLateralXTransportOFFLINEDETECT.py``.

Why this exists
---------------
That scan sweeps a 2-D ``(step_size x nsteps)`` grid under ``pingponggrating
return_trip=False``, so every cell leaves the array translated by its OWN
``dx = step_size * nsteps`` knm-px.  The live detection grid is fixed for the whole
scan (``imagePatternsJson`` is resolved once, in ``engine_run.py`` -> scan_prep), so
live img2 occupancy is only correct for the single cell whose ``dx`` matches the
declared frame-1 grid.  Everything else is measured with boxes in the wrong place.

This tool fixes that after the fact: for each scan point it rebuilds the detection
grid shifted by THAT point's ``dx``, re-detects img2 from the saved raw frames, and
recomputes survival.  img1 is re-detected on the unshifted grid (the array has not
moved when frame 0 is taken), so both frames are scored consistently.

Method
------
1. Read the scan sidecar JSON: ``Params`` (1-indexed scan-point id per shot -- see the
   ``gotcha-pyctrl-params-1-indexed-2d-map`` memory), ``ScanGroup`` (the swept values),
   ``roi``, ``boxSize`` / ``maskSigma``.
2. Recover ``(step_size, nsteps)`` per scan point from the descriptor, and hence ``dx``.
3. Build the camera grid per point: registry knm + ``dx`` on the x column -> global
   affine -> minus the crop.  Same code path the live pipeline uses.
4. Re-detect both frames with the SAME Gaussian-weighted box + per-site thresholds the
   live detector uses (``yb_analysis.detection``), so numbers are comparable.
5. Survival = P(atom in img2 at site i | atom in img1 at site i), pooled over sites and
   shots per cell, with a binomial SEM.

Per-step survival ``s = S ** (1/nsteps)`` is reported alongside: if transport loss is
memoryless it collapses onto one curve vs ``step_size``, independent of ``nsteps``.

Usage::

    python pyctrl/tools/ppg_transport_analyze.py --scan-id 20260805194246
    python pyctrl/tools/ppg_transport_analyze.py --scan-id ... --out-dir <dir> --plot
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PYCTRL)
for _p in (_ROOT, _HERE, os.path.join(_PYCTRL, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KNM_PER_CAM_PX = 2.268      # +1 knm-x -> +2.268 camera-X px (validated on real images)
NN_PITCH_KNM = 24.48        # square lattice, axes at 0/90 deg


# ---------------------------------------------------------------- scan discovery

def find_scan_dir(scan_id, prefix=None):
    """``<prefix>/Data/<YYYYMMDD>/data_<YYYYMMDD>_<HHMMSS>`` for a 14-digit scan id."""
    import scan_prep
    return scan_prep.scan_dir(int(scan_id), prefix=prefix)


def load_scan(scan_dir):
    """Load the scan-config sidecar (``data_<stamp>.json``) + the HDF5.

    The scan dir also holds ``analysis_payload.json`` / ``slm_code.json``, which sort
    BEFORE the config alphabetically -- so pick the config by NAME (it mirrors the dir
    name, ``data_<YYYYMMDD>_<HHMMSS>.json``) rather than taking the first match.
    """
    base = os.path.basename(scan_dir.rstrip("\\/"))       # data_<YYYYMMDD>_<HHMMSS>
    want = os.path.join(scan_dir, base + ".json")
    if os.path.isfile(want):
        cfg = json.load(open(want))
    else:
        cands = [p for p in sorted(glob.glob(os.path.join(scan_dir, "data_*.json")))
                 if not os.path.basename(p).startswith("_")]
        if not cands:
            raise SystemExit("no scan-config JSON (data_*.json) in %s" % scan_dir)
        cfg = json.load(open(cands[0]))
    if "roi" not in cfg or "Params" not in cfg:
        raise SystemExit(
            "%s does not look like a scan config (keys: %s) -- expected 'roi' + 'Params'"
            % (want, ", ".join(sorted(cfg.keys())[:8])))
    h5 = sorted(glob.glob(os.path.join(scan_dir, "*.h5")))
    if not h5:
        raise SystemExit("no HDF5 in %s" % scan_dir)
    return cfg, h5[0]


def _walk(obj, key):
    """Yield every value stored under ``key`` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def scan_axes(cfg):
    """Recover the swept ``step_size`` and ``nsteps`` value LISTS from the ScanGroup.

    Returns ``(steps, nsteps)`` as plain lists. Raises if either axis is absent --
    this tool is only meaningful for the 2-D transport scan.
    """
    sg = cfg.get("ScanGroup") or {}

    def _longest(key):
        """Longest LIST stored under ``key``, else the scalar it was fixed at.

        An axis with a single value is NOT swept, so ScanGroup stores it as a scalar
        (e.g. the "long" arm at nsteps=50 only). Treat that as a length-1 axis rather
        than failing -- the cell grid is still well defined.
        """
        best, scalar = None, None
        for v in _walk(sg, key):
            if isinstance(v, list):
                if len(v) > 1 and (best is None or len(v) > len(best)):
                    best = v
                elif len(v) == 1 and scalar is None:
                    scalar = v[0]
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if scalar is None:
                    scalar = v
        if best is not None:
            return best
        return None if scalar is None else [scalar]

    steps = _longest("step_size")
    ns = _longest("nsteps")
    if steps is None or ns is None:
        raise SystemExit(
            "could not recover both step_size and nsteps from ScanGroup "
            "(step_size=%r, nsteps=%r) -- is this a transport scan?" % (steps, ns))
    return [float(s) for s in steps], [int(n) for n in ns]


def point_to_cell(pid, n_steps_axis, n_nsteps_axis):
    """Scan-point id (1-indexed) -> ``(i_step, j_nsteps)``.

    pyctrl flattens a 2-D sweep COLUMN-MAJOR (dim-1 varies fastest) and ``Params`` is
    1-INDEXED -- memories ``gotcha-2d-scan-reshape-column-major`` and
    ``gotcha-pyctrl-params-1-indexed-2d-map``. So on the 0-based index
    ``p = pid - 1``: ``i = p % n_steps``, ``j = p // n_steps``.

    VERIFIED against the real ScanGroup this scan builds: for both arms, every
    ``pid`` in ``1..nseq`` maps to the cell whose ``(step_size, nsteps)`` matches
    ``g.getseq(pid % nseq)`` -- 0 mismatches over 78 + 22 points. (Note ``getseq`` is
    0-based WITH wraparound, i.e. ``getseq(0)`` is the LAST cell and scan point ``p``
    is ``getseq(p)``; that offset is a ScanGroup indexing quirk, NOT part of the
    Params->cell mapping, which is the plain column-major formula above.)
    """
    p = int(pid) - 1
    return p % n_steps_axis, p // n_steps_axis


# ---------------------------------------------------------------- grids

def pattern_knm_xy(pattern, dx=0.0):
    """Pattern site positions as ``[x, y]`` in knm-1024, translated ``dx`` in +x."""
    from yb_analysis.analysis import affine_transform as aff
    from yb_analysis.analysis import pattern_registry as reg
    rec = reg.get_pattern(pattern)
    if rec is None:
        raise SystemExit("pattern %r not in registry" % pattern)
    knm = aff.canonical_knm(rec)
    if knm is None:
        knm = np.asarray(rec["knm"], float)
    knm = np.asarray(knm, float).copy()
    knm[:, 1] += float(dx)          # registry knm is [y, x] -> x is column 1
    return aff._knm_to_xy(knm)      # -> [x, y]


def shifted_grid(pattern, dx, roi):
    """Camera grid (cropped ``[Y, X]``) for ``pattern`` translated ``dx`` knm-px in +x."""
    from yb_analysis.analysis import affine_transform as aff
    A = aff.load_matrix()
    if A is None:
        raise SystemExit("no committed affine")
    return aff.apply_affine_cropped(pattern_knm_xy(pattern, dx), A, roi)


def stored_grid0(cfg):
    """The dx=0 camera grid the scan ACTUALLY detected with, from its own sidecar.

    ``initGridLocationsY/X`` are written per scan, in the same site order as
    ``initThresholds``. Anchoring on them makes the offline re-detection reproducible: the
    alternative (rebuilding from the live pattern registry + the globally committed affine)
    silently depends on mutable state, so the same scan can analyse differently months apart.

    That is not hypothetical -- 2026-08-10, scan 20260810_174745 (33x33_feedback11, one-way):
    analysed right after the run it gave step-0 survival 0.9847 / n_load 15440; re-analysed a
    couple of hours later, after unrelated 17x17 work had moved the shared calibration, the SAME
    raw data gave 0.7730 / 11756. Step 0 is the no-motion control, where the live detector
    measured 0.9815 -- so the offline grid, not the atoms, had moved by 21%.

    Returns ``[Y, X]`` (cropped ROI frame) or None when the sidecar lacks the keys.
    """
    gy, gx = cfg.get("initGridLocationsY"), cfg.get("initGridLocationsX")
    if gy is None or gx is None:
        return None
    gy = np.asarray(gy, float).ravel()
    gx = np.asarray(gx, float).ravel()
    if gy.size != gx.size or gy.size == 0:
        return None
    return np.column_stack([gy, gx])


def dx_shift_camera(pattern, dx, roi):
    """Camera-pixel displacement for a knm-px translation ``dx`` in +x.

    Uses ONLY the affine's linear part (a difference of two mapped grids), which is what sets
    the scale/rotation. The absolute offset -- the part that drifts when the affine is
    re-committed -- cancels, so this stays valid even when the stored grid and the live affine
    disagree about where the array sits.
    """
    if not dx:
        return np.zeros(2)
    g1 = shifted_grid(pattern, dx, roi)
    g0 = shifted_grid(pattern, 0.0, roi)
    return np.asarray(np.mean(g1 - g0, axis=0), float)


def onsensor_mask(grid, roi, margin):
    H, W = float(roi[3]), float(roi[2])
    cy, cx = grid[:, 0], grid[:, 1]
    return ((cy >= margin) & (cy <= H - margin - 1)
            & (cx >= margin) & (cx <= W - margin - 1))


# knm-1024 far-field centre = the PHYSICAL zeroth-order block (see the `slm` skill). A trap
# placed on or near DC is physically removed -- the spot is gone on the real atoms even though
# the hologram contains it. That is why a 33x33 = 1089 design yields 1068 extracted sites: the
# ~21 central ones are eaten by the hole.
DC_KNM = (512.0, 512.0)


def dc_hole_radius(knm_xy, dc=DC_KNM):
    """Conservative DC-hole radius: the closest EXISTING site's distance to DC.

    Everything inside this radius is empty in the loading array *because* the hole removed it,
    so any site transported to r < R_MIN is in territory known to be optically dead. For
    33x33_feedback11 this is 69.28 knm-px.
    """
    d = np.asarray(knm_xy, float) - np.asarray(dc, float)
    return float(np.min(np.hypot(d[:, 0], d[:, 1])))


def outside_dc_mask(knm_xy_shifted, r_min, dc=DC_KNM):
    """True for sites that stay OUTSIDE the zeroth-order hole after the shift.

    Sites that move inside are excluded from BOTH numerator and denominator: they are destroyed
    by the optics, not by transport, and counting them would inflate the measured loss by a
    ``dx``-dependent amount (up to 23 of 1068 = 2.2% at dx = 125) -- a systematic that grows
    with displacement and therefore mimics exactly the effect being measured.
    """
    d = np.asarray(knm_xy_shifted, float) - np.asarray(dc, float)
    return np.hypot(d[:, 0], d[:, 1]) >= float(r_min)


# ---------------------------------------------------------------- detection

def detect(img, grid, thresholds, mask_mat):
    """Boolean occupancy per site, via the production detector."""
    from yb_analysis.detection.detect_atom import detect_atom
    out = detect_atom(img, grid, thresholds, mask_mat)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out).astype(bool).ravel()


def gaussian_mask(box, sigma):
    """The live pipeline's detection weighting mask (data_manager._gaussian_mask)."""
    from yb_analysis.acquisition.data_manager import _gaussian_mask as gm
    return gm(int(box), float(sigma))


# ---------------------------------------------------------------- main analysis

def analyze(scan_id, prefix=None, pattern=None, margin=6, max_shots=None):
    scan_dir = find_scan_dir(scan_id, prefix)
    cfg, h5path = load_scan(scan_dir)

    roi = [float(v) for v in cfg["roi"]]
    n_img = int(cfg.get("NumImages", 2))
    if n_img < 2:
        raise SystemExit("need NumImages>=2 to measure survival, got %d" % n_img)

    pats = json.loads(cfg["imagePatternsJson"]) if isinstance(
        cfg.get("imagePatternsJson"), str) else (cfg.get("imagePatternsJson") or [])
    pattern = pattern or (pats[0]["name"] if pats else None)
    if not pattern:
        raise SystemExit("no pattern declared; pass --pattern")

    steps, nsteps = scan_axes(cfg)
    params = np.asarray(cfg["Params"], dtype=np.int64)
    box = int(cfg.get("boxSize", 11))
    sigma = float(cfg.get("maskSigma", 2.0))
    mask_mat = gaussian_mask(box, sigma)

    thr = np.asarray(cfg.get("initThresholds"), dtype=np.float64).ravel() \
        if cfg.get("initThresholds") is not None else None

    # Grid cache: one per distinct dx (many cells share a dx).
    # dx=0 comes from the scan's OWN stored grid when available (reproducible; matches the
    # thresholds' site order), falling back to the live registry+affine rebuild. See
    # stored_grid0() for why this matters.
    grid0 = stored_grid0(cfg)
    grid_src = "scan sidecar initGridLocations"
    if grid0 is None:
        grid0 = shifted_grid(pattern, 0.0, roi)
        grid_src = "live registry + committed affine"
    else:
        ref = shifted_grid(pattern, 0.0, roi)
        if len(ref) == len(grid0):
            d = float(np.median(np.hypot(*(ref - grid0).T)))
            if d > 1.0:
                print("  NOTE stored grid and live-affine grid differ by %.2f px (median); "
                      "using the stored one" % d)
    n_sites = len(grid0)
    if thr is None or thr.size != n_sites:
        raise SystemExit("initThresholds missing/size-mismatched (%s vs %d sites)"
                         % (None if thr is None else thr.size, n_sites))
    gcache = {0.0: grid0}
    # Zeroth-order hole: conservative radius = the closest existing site's distance to DC.
    knm0 = pattern_knm_xy(pattern, 0.0)
    r_min = dc_hole_radius(knm0)
    dccache = {0.0: outside_dc_mask(knm0, r_min)}

    import h5py
    acc = {}   # (i_step, j_nsteps) -> [n_surv, n_loaded, n_shots, n_offsensor]
    with h5py.File(h5path, "r") as f:
        imgs = f["imgs"]
        # ``Params`` is a LOOKUP TABLE INDEXED BY seq_id, **not** a per-shot sequence.
        # With Scramble=1 the runner visits scan points in a shuffled order, so shot k's
        # scan point is ``Params[seq_ids[k] - 1]`` (seq_id is 1-indexed). Using
        # ``Params[k]`` positionally silently assigns each shot a RANDOM cell -- it
        # agrees with the truth only ~1/n_cells of the time, which washes every cell out
        # to the global mean survival (measured: 0.2-0.7% agreement on these scans).
        seq_ids = f["seq_ids"][:] if "seq_ids" in f else None
        # The image dataset is written incrementally, so on a LIVE scan it can lag the
        # per-shot tables. Take the smallest consistent length.
        n_shots = imgs.shape[0] // n_img
        if seq_ids is not None:
            n_shots = min(n_shots, len(seq_ids))
        if max_shots:
            n_shots = min(n_shots, int(max_shots))
        for sh in range(n_shots):
            if seq_ids is not None:
                sid_k = int(seq_ids[sh]) - 1          # seq_id is 1-indexed
                pid = params[sid_k] if 0 <= sid_k < len(params) else None
            else:
                pid = params[sh] if sh < len(params) else None
            if pid is None:
                continue
            i, j = point_to_cell(pid, len(steps), len(nsteps))
            if i >= len(steps) or j >= len(nsteps):
                continue
            s, n = steps[i], nsteps[j]
            dx = round(s * n, 4)
            if dx not in gcache:
                # grid0 is the scan's own anchor; only the RELATIVE displacement comes from
                # the affine, so a re-committed affine cannot move the absolute registration.
                gcache[dx] = grid0 + dx_shift_camera(pattern, dx, roi)
                dccache[dx] = outside_dc_mask(pattern_knm_xy(pattern, dx), r_min)
            g2 = gcache[dx]
            # Exclude sites that (a) leave the sensor or (b) are transported into the physical
            # zeroth-order hole. Both are instrument effects, not transport loss, and both grow
            # with dx -- so leaving either in would bias exactly the quantity being measured.
            on = onsensor_mask(g2, roi, margin) & onsensor_mask(grid0, roi, margin)
            keep = on & dccache[dx] & dccache[0.0]

            a = imgs[sh * n_img].astype(np.float64)
            b = imgs[sh * n_img + (n_img - 1)].astype(np.float64)
            o1 = detect(a, grid0, thr, mask_mat)
            o2 = detect(b, g2, thr, mask_mat)

            loaded = o1 & keep
            rec = acc.setdefault((i, j), [0, 0, 0, int((~on).sum()), int((~dccache[dx]).sum())])
            rec[0] += int((loaded & o2).sum())
            rec[1] += int(loaded.sum())
            rec[2] += 1

    rows = []
    for (i, j), (nsurv, nload, nsh, noff, ndc) in sorted(acc.items()):
        s, n = steps[i], nsteps[j]
        S = (nsurv / nload) if nload else float("nan")
        sem = (np.sqrt(max(S * (1 - S), 0) / nload) if nload else float("nan"))
        # nsteps=0 is the static no-motion CONTROL: there is no "per step" to speak of
        # (and 1/n would divide by zero), so leave it NaN and read its S as the floor.
        per_step = (S ** (1.0 / n) if (nload and S > 0 and n > 0) else float("nan"))
        rows.append({
            "step_size": s, "nsteps": n, "dx_knm": round(s * n, 4),
            "dx_cam_px": round(s * n * KNM_PER_CAM_PX, 2),
            "dx_pitch": round(s * n / NN_PITCH_KNM, 3),
            "survival": S, "sem": sem, "per_step_survival": per_step,
            "n_loaded": nload, "n_shots": nsh,
            "n_offsensor_sites": noff, "n_dchole_sites": ndc,
        })
    return {"scan_id": int(scan_id), "scan_dir": scan_dir, "pattern": pattern,
            "roi": roi, "steps": steps, "nsteps": nsteps, "n_sites": n_sites,
            "dc_hole_radius_knm": r_min, "rows": rows}


def _erfc_model(x, A, x0, w):
    """Per-step survival vs step size: ``A/2 * erfc((x-x0)/(sqrt(2) w))``.

    Total survival over n steps is ``s(x)**n`` -- the "compounded erfc". The erfc shape
    is the survival probability of a Gaussian-spread quantity against a hard threshold:
    the atom is lost when its (thermal) excursion plus the commanded step carries it out
    of the overlap region between consecutive frames' traps. ``x0`` = per-step cliff
    centre, ``w`` = cliff width, ``A`` = the small-step plateau (the non-transport floor).
    """
    from scipy.special import erfc
    return A * 0.5 * erfc((np.asarray(x, float) - x0) / (np.sqrt(2.0) * w))


def fit_erfc_per_step(rows_for_n, n):
    """Fit ``_erfc_model`` to the per-step survival of one nsteps slice.

    Fits s = S**(1/n) with errors propagated from the binomial SEM on S:
    ``ds = (1/n) S**(1/n - 1) dS``. Returns None if the fit does not converge.
    """
    from scipy.optimize import curve_fit
    rr = [r for r in rows_for_n if r["n_loaded"] > 0 and np.isfinite(r["survival"])]
    if len(rr) < 4 or n <= 0:
        return None
    x = np.array([r["step_size"] for r in rr], float)
    S = np.clip(np.array([r["survival"] for r in rr], float), 1e-9, None)
    sem = np.array([r["sem"] for r in rr], float)
    s = S ** (1.0 / n)
    ds = np.maximum((1.0 / n) * S ** (1.0 / n - 1.0) * sem, 1e-6)
    try:
        p, cov = curve_fit(_erfc_model, x, s, p0=[float(s.max()), 1.45, 0.15],
                           sigma=ds, absolute_sigma=True, maxfev=20000)
    except Exception:  # noqa: BLE001
        return None
    A, x0, w = (float(v) for v in p)
    return {"A": A, "x0": x0, "w": abs(w),
            "err": [float(e) for e in np.sqrt(np.diag(cov))]}


def step_at_survival(fit, n, target=0.99, per_step=True, hi=3.0, normalize=True):
    """Step size where survival hits ``target``.

    ``normalize=True`` (default) divides out the fitted plateau ``A``, so the target is
    on TRANSPORT loss alone: ``s(x)/A = target``. ``A`` is the small-step floor -- atoms
    lost to imaging/loading, not to motion (measured directly here by the step_size=0 and
    nsteps=0 controls, both ~0.994). Without normalizing, an imaging floor of 0.994 eats
    more than half a 1% budget and the "99%" mark would move with imaging quality rather
    than with transport physics.

    ``per_step=True``  -> solve s(x)/A = target        (each individual step that good).
    ``per_step=False`` -> solve (s(x)/A)**n = target   (the WHOLE move that good).
    Returns NaN when the target is unreachable in ``[0, hi]``.
    """
    from scipy.optimize import brentq
    A, x0, w = fit["A"], fit["x0"], fit["w"]
    den = A if (normalize and A > 0) else 1.0
    if per_step:
        f = lambda xx: _erfc_model(xx, A, x0, w) / den - target   # noqa: E731
    else:
        f = lambda xx: (n * np.log(max(_erfc_model(xx, A, x0, w) / den, 1e-12))          # noqa: E731
                        - np.log(target))
    try:
        if f(0.0) > 0 > f(hi):
            return float(brentq(f, 0.0, hi))
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def print_erfc_summary(res, target=0.99):
    """Table of compounded-erfc fits + the step size that holds ``target`` survival."""
    rows = res["rows"]
    ns = sorted({r["nsteps"] for r in rows if r["nsteps"] > 0})
    print("\ncompounded-erfc fit   s(x) = A/2*erfc((x-x0)/(sqrt2 w))   S = s(x)**n")
    print("%6s %7s %7s %7s | %11s %11s %10s"
          % ("nsteps", "A", "x0", "w", "step@s=%.2f" % target,
             "step@S=%.2f" % target, "travel"))
    for n in ns:
        fit = fit_erfc_per_step([r for r in rows if r["nsteps"] == n], n)
        if not fit:
            print("%6d   (fit failed)" % n)
            continue
        xs = step_at_survival(fit, n, target, per_step=True)
        xt = step_at_survival(fit, n, target, per_step=False)
        print("%6d %7.4f %7.4f %7.4f | %11s %11s %10s"
              % (n, fit["A"], fit["x0"], fit["w"],
                 "%.4f" % xs if np.isfinite(xs) else "n/a",
                 "%.4f" % xt if np.isfinite(xt) else "none",
                 "%.1f" % (xt * n) if np.isfinite(xt) else "-"))
    print("  TRANSPORT-ONLY: the fitted plateau A (the imaging/loading floor, ~%.4f here,"
          % np.mean([f["A"] for f in
                     (fit_erfc_per_step([r for r in rows if r["nsteps"] == n], n)
                      for n in ns) if f]))
    print("  independently measured by the step_size=0 / nsteps=0 controls) is DIVIDED OUT,")
    print("  so these are s/A = %.2f and (s/A)**n = %.2f -- loss from MOTION alone."
          % (target, target))
    print("  step@s = each STEP is %.0f%%; step@S = the WHOLE move is %.0f%%."
          % (target * 100, target * 100))
    print("  travel = step@S * nsteps, in knm-px (x2.268 -> camera px, /24.48 -> pitches).")


def print_table(res):
    print("scan %s   pattern=%s   sites=%d   DC-hole radius=%.2f knm px"
          % (res["scan_id"], res["pattern"], res["n_sites"],
             res.get("dc_hole_radius_knm", float("nan"))))
    print("%8s %7s %9s %9s %8s %10s %9s %8s"
          % ("step", "nsteps", "dx_knm", "dx_pitch", "surv", "+/-sem", "per_step", "n_load"))
    for r in sorted(res["rows"], key=lambda r: (r["nsteps"], r["step_size"])):
        flag = "".join([
            "  OFF=%d" % r["n_offsensor_sites"] if r["n_offsensor_sites"] else "",
            "  DC=%d" % r.get("n_dchole_sites", 0) if r.get("n_dchole_sites") else "",
        ])
        print("%8.3f %7d %9.2f %9.3f %8.4f %10.4f %9.4f %8d%s"
              % (r["step_size"], r["nsteps"], r["dx_knm"], r["dx_pitch"],
                 r["survival"], r["sem"], r["per_step_survival"], r["n_loaded"], flag))


def plot(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    ns = sorted({r["nsteps"] for r in rows})
    steps = sorted({r["step_size"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4))

    # --- panels 1-2: vs step size, one curve per nsteps -----------------------------
    for k, n in enumerate(ns):
        rr = sorted([r for r in rows if r["nsteps"] == n], key=lambda r: r["step_size"])
        x = [r["step_size"] for r in rr]
        c = cmap(k / max(1, len(ns) - 1))
        axes[0].errorbar(x, [r["survival"] for r in rr], yerr=[r["sem"] for r in rr],
                         marker="o", ms=4, lw=1.3, color=c, label="n=%d" % n)
        axes[1].plot(x, [r["per_step_survival"] for r in rr], marker="o", ms=4,
                     lw=1.3, color=c, label="n=%d" % n)
        # Compounded-erfc fit: model the PER-STEP survival as a complementary error
        # function in step size, s(x) = A/2 * erfc((x-x0)/(sqrt(2) w)); total survival is
        # then S = s(x)**n ("compounded"). erfc is the natural shape here -- it is the
        # probability that a Gaussian-distributed quantity stays under a threshold, i.e.
        # the atom survives the step unless the (thermally spread) displacement pushes it
        # past the trap-overlap edge. x0 is the per-step cliff centre, w its width.
        fit = fit_erfc_per_step(rr, n)
        if fit:
            xf = np.linspace(min(x), max(x), 200)
            axes[1].plot(xf, _erfc_model(xf, fit["A"], fit["x0"], fit["w"]),
                         "--", lw=1.0, color=c, alpha=.75)
    axes[0].set_xlabel("step_size (knm-px/step)"); axes[0].set_ylabel("survival")
    axes[0].set_title("+x one-way transport: survival vs step size")
    axes[1].set_xlabel("step_size (knm-px/step)")
    axes[1].set_ylabel(r"per-step survival $S^{1/n}$")
    axes[1].set_title("collapse => memoryless; spread => cumulative")

    # --- panel 3: vs nsteps, one curve per step size, LOG y -------------------------
    # Log y because memoryless per-step loss means S = s**n, i.e. a STRAIGHT LINE here
    # (slope = ln s). Curvature away from straight is the signature of n-dependent loss.
    # nsteps=0 is dropped: it is the static control, and it anchors every curve at S~0.99
    # without being part of the n-scaling.
    for k, s in enumerate(steps):
        rr = sorted([r for r in rows if r["step_size"] == s and r["nsteps"] > 0],
                    key=lambda r: r["nsteps"])
        if not rr:
            continue
        c = cmap(k / max(1, len(steps) - 1))
        y = np.array([r["survival"] for r in rr], dtype=float)
        e = np.array([r["sem"] for r in rr], dtype=float)
        good = y > 0                      # a zero would blow up the log axis
        axes[2].errorbar(np.array([r["nsteps"] for r in rr])[good], y[good],
                         yerr=e[good], marker="o", ms=4, lw=1.3, color=c,
                         label="step=%g" % s)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("nsteps")
    axes[2].set_ylabel("survival (log)")
    axes[2].set_title("survival vs nsteps (log y)\nstraight line => memoryless $S=s^{n}$")

    for a in axes:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=7, ncol=2)
    fig.suptitle("scan %s  (OFFLINE re-detection on per-point shifted grids)"
                 % res["scan_id"], fontsize=10)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    perm = os.path.join(res["scan_dir"], "ppg_transport_%s.png" % res["scan_id"])
    fig.text(0.01, 0.01, perm, fontsize=6, ha="left", va="bottom")
    for p in (os.path.join(out_dir, "ppg_transport_%s.png" % res["scan_id"]), perm):
        try:
            fig.savefig(p, dpi=130)
            print("figure -> %s" % p)
        except Exception as e:  # noqa: BLE001
            print("could not save %s (%s)" % (p, e))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-id", required=True, help="14-digit YYYYMMDDHHMMSS")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--target", type=float, default=0.99,
                    help="survival level reported by the erfc summary (default 0.99)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--json", default=None, help="write the row table as JSON")
    args = ap.parse_args()

    res = analyze(args.scan_id, prefix=args.prefix, pattern=args.pattern,
                  margin=args.margin, max_shots=args.max_shots)
    print_table(res)
    try:
        print_erfc_summary(res, target=args.target)
    except Exception as e:  # noqa: BLE001 - the fit must never break the table
        print("erfc summary unavailable (%s)" % str(e)[:120])
    if args.json:
        json.dump(res, open(args.json, "w"), indent=2, default=float)
        print("json -> %s" % args.json)
    if args.plot:
        plot(res, args.out_dir)


if __name__ == "__main__":
    main()
