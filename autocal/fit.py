"""autocal.fit -- fit a pooled calibration spectrum to a Lorentzian (dip or peak).

Prefers the lab's canonical fitter (``yb_analysis.analysis.fittings.lorentzian.fit_lorentzian`` --
the SAME one ``tools/fit_spectrum.py`` uses, so an auto-cal center matches a hand fit). Falls back
to a numpy-only least-squares fit when yb_analysis is not importable (so the pure pooling+fit path
is unit-testable in any env). Returns a normalized dict::

    {"center": float, "fwhm": float, "r2": float, "mode": "dip"|"peak",
     "edge_pinned": bool, "n_points": int, "x_min": float, "x_max": float}

or ``None`` if there are too few finite points to fit.
"""

import math


def fit_spectrum(x, y, sem=None, mode="dip", min_points=5):
    """Fit ``(x, y[, sem])`` to a single Lorentzian. Non-finite points are dropped first."""
    xs, ys, es = _clean(x, y, sem)
    if len(xs) < min_points:
        return None
    res = _fit_canonical(xs, ys, es, mode)
    if res is None:
        res = _fit_local(xs, ys, es, mode)
    if res is None:
        return None
    center, fwhm, r2 = res
    x_min, x_max = min(xs), max(xs)
    span = (x_max - x_min) or 1.0
    edge = (center <= x_min + 0.02 * span) or (center >= x_max - 0.02 * span)
    return {"center": float(center), "fwhm": float(abs(fwhm)), "r2": float(r2),
            "mode": mode, "edge_pinned": bool(edge), "n_points": len(xs),
            "x_min": float(x_min), "x_max": float(x_max)}


def _clean(x, y, sem):
    x = list(x or [])
    y = list(y or [])
    sem = list(sem or [])
    xs, ys, es = [], [], []
    for j in range(min(len(x), len(y))):
        if _finite(x[j]) and _finite(y[j]):
            xs.append(float(x[j]))
            ys.append(float(y[j]))
            es.append(float(sem[j]) if j < len(sem) and _finite(sem[j]) else None)
    if all(e is None for e in es):
        es = None
    return xs, ys, es


def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _fit_canonical(xs, ys, es, mode):
    """Use yb_analysis' fit_lorentzian when available (lazy/guarded)."""
    try:
        import numpy as np
        from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian
    except Exception:  # noqa: BLE001 - yb_analysis not importable in this env
        return None
    try:
        ye = np.asarray(es, float) if es is not None else None
        fit = fit_lorentzian(np.asarray(xs, float), np.asarray(ys, float), ye, mode=mode)
    except Exception:  # noqa: BLE001 - fall back to the local fit on any failure
        return None
    if not fit:
        return None
    return fit["center"], abs(fit["width"]), fit["r_squared"]


def _fit_local(xs, ys, es, mode):
    """numpy-only Lorentzian fit: grid-search (center, width); amplitude+baseline solved linearly.

    Robust and dependency-light (no scipy). For a Lorentzian ``y = b + a / (1 + ((x-c)/hw)^2)``,
    fixing (c, hw) makes ``y`` linear in (b, a), so for each (c, hw) on a grid we solve the 2-param
    least squares in closed form and keep the lowest-residual (c, hw). ``mode='dip'`` expects a<0,
    ``'peak'`` a>0; we don't constrain the sign (the data picks it) but report FWHM = 2*hw."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    w = None
    if es is not None:
        e = np.asarray([v if (v is not None and v > 0) else np.nan for v in es], float)
        if np.isfinite(e).any():
            e = np.where(np.isfinite(e), e, np.nanmax(e[np.isfinite(e)]))
            w = 1.0 / (e * e)
    span = float(x.max() - x.min()) or 1.0
    centers = np.linspace(x.min(), x.max(), 81)
    halfwidths = np.linspace(span / 200.0, span / 2.0, 60)
    best = None
    yc = y - (np.average(y, weights=w) if w is not None else y.mean())
    sst = float(np.sum((w if w is not None else 1.0) * yc * yc)) or 1.0
    for c in centers:
        denom = 1.0 + ((x - c) / halfwidths[:, None]) ** 2
        basis = 1.0 / denom  # shape (n_hw, n_x)
        for i in range(basis.shape[0]):
            g = basis[i]
            # Solve y ~ b*1 + a*g  (weighted), closed form 2x2.
            res = _lstsq2(np.ones_like(x), g, y, w)
            if res is None:
                continue
            b, a, ssr = res
            r2 = 1.0 - ssr / sst
            if best is None or r2 > best[2]:
                best = (c, halfwidths[i], r2, a, b)
    if best is None:
        return None
    c, hw, r2 = best[0], best[1], best[2]
    return float(c), float(2.0 * hw), float(max(min(r2, 1.0), -1.0))


def _lstsq2(g0, g1, y, w):
    """Weighted 2-param least squares for ``y ~ p0*g0 + p1*g1``; returns ``(p0, p1, ssr)``."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    if w is None:
        w = np.ones_like(y)
    a00 = float(np.sum(w * g0 * g0))
    a01 = float(np.sum(w * g0 * g1))
    a11 = float(np.sum(w * g1 * g1))
    b0 = float(np.sum(w * g0 * y))
    b1 = float(np.sum(w * g1 * y))
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-300:
        return None
    p0 = (a11 * b0 - a01 * b1) / det
    p1 = (a00 * b1 - a01 * b0) / det
    resid = y - (p0 * g0 + p1 * g1)
    ssr = float(np.sum(w * resid * resid))
    return p0, p1, ssr
