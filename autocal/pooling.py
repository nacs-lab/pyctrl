"""autocal.pooling -- combine the per-point survival of many short (yielded) runs into one curve.

A background calibration scan is yielded the instant foreground work appears, so any single run
may contribute only a handful of shots. To still get a fittable spectrum we pool runs of the same
``(pattern, cal)`` -- and we do it with INVERSE-VARIANCE weighting of the per-point survival,
which needs only the per-point mean + SEM that ``run_analysis.analyze_scan`` already returns (no
raw-count plumbing) and is the statistically correct way to combine independent repeated
measurements of the same quantity:

    pooled_mean_j = (Sum_i mean_ij / sem_ij^2) / (Sum_i 1 / sem_ij^2)
    pooled_sem_j  = sqrt( 1 / (Sum_i 1 / sem_ij^2) )

The two running sums (``w0 = Sum 1/sem^2``, ``w1 = Sum mean/sem^2``) are ADDITIVE, so the ledger
accumulator just adds each scan's contribution -- it never has to retain every scan's arrays.

Edge cases: a point with no shots in a given run (NaN mean / non-finite SEM) contributes nothing;
an all-survived / all-lost point has SEM 0, which would be infinite weight, so SEM is floored at
``min_sem`` (``DEFAULT_MIN_SEM``). The pure math (``scan_point_weights`` / ``combine_sums``) has no
numpy/yb_analysis dependency; only the on-disk adapter (``scan_to_points``) imports them, lazily.
"""

import math

# A survival SEM floor. ~0.02 corresponds to p(1-p)/n at n~600 near p~0.5; it stops a single
# all-survived short run from dominating the pool with a (spurious) zero-variance point.
DEFAULT_MIN_SEM = 0.02


def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def scan_point_weights(mean, sem, min_sem=DEFAULT_MIN_SEM):
    """One scan's per-point inverse-variance contributions ``(w0, w1)``, aligned to the grid.

    Returns two equal-length lists (same length as ``mean``); a point that is unusable this run
    (non-finite mean or SEM) contributes ``0.0`` to both, so the grid alignment is preserved and a
    later run can fill it in. Returns ``([], [])`` when there is no data at all."""
    mean = list(mean or [])
    sem = list(sem or [])
    n = len(mean)
    if n == 0:
        return [], []
    if len(sem) != n:
        # Defensive: if SEM is missing/short, treat missing entries as the floor.
        sem = (sem + [min_sem] * n)[:n]
    w0 = [0.0] * n
    w1 = [0.0] * n
    for j in range(n):
        m = mean[j]
        s = sem[j]
        if not _finite(m):
            continue
        s = float(s) if _finite(s) else min_sem
        s = max(s, min_sem)
        w = 1.0 / (s * s)
        w0[j] = w
        w1[j] = float(m) * w
    return w0, w1


def combine_sums(w0, w1):
    """Read out pooled ``(mean, sem)`` per point from the running sums.

    A point with zero total weight (never measured) yields ``mean=nan``, ``sem=inf`` -- the fitter
    filters non-finite points. (Not serialized: the ledger stores ``w0``/``w1``, not this output.)"""
    n = len(w0)
    mean = [float("nan")] * n
    sem = [float("inf")] * n
    for j in range(n):
        w = w0[j]
        if w and w > 0:
            mean[j] = w1[j] / w
            sem[j] = math.sqrt(1.0 / w)
    return mean, sem


# =========================================================================== #
# on-disk adapter: a finished scan -> per-point (x, mean, sem) -- GUARDED import
# =========================================================================== #
def scan_to_points(scan_id, reader=None):
    """Turn a finished scan into ``(x, mean, sem, n_shots)`` for pooling.

    ``reader`` (a ``scan_id -> dict`` callable) is injected in tests so no yb_analysis / data files
    are touched. The default reader is the lab's canonical ``analyze_scan`` (yb_analysis env), the
    SAME pipeline ``tools/fit_spectrum.py`` uses: ``sweep.values[0]`` is the swept axis and
    ``summary.survival_mean`` / ``survival_sem`` the per-point P11. Returns ``([],[],[],0)`` if the
    scan has no usable sweep (e.g. a ``--reps 0`` run with no ``config['Params']`` map)."""
    if reader is None:
        reader = _default_reader
    d = reader(scan_id)
    if not d:
        return [], [], [], 0
    sweep = (d.get("sweep") or {}).get("values") or []
    summary = d.get("summary") or {}
    x = list(sweep[0]) if sweep and sweep[0] is not None else []
    mean = list(summary.get("survival_mean") or [])
    sem = list(summary.get("survival_sem") or [])
    n_shots = int(d.get("n_shots") or 0)
    if not x or not mean:
        return [], [], [], n_shots
    return x, mean, sem, n_shots


def _default_reader(scan_id):
    """Default adapter over ``run_analysis.analyze_scan`` (lazy import; yb_analysis env)."""
    from yb_analysis.analysis.run_analysis import analyze_scan
    return analyze_scan(scan_id, include_per_site=False, include_diag_aggregate=False,
                        include_per_iteration=False, sync_slm_diag=False)
