"""pattern_grid.py -- per-pattern grid + threshold resolver (loading-pattern affine migration).

Reads the per-loading-pattern registry that yb_analysis maintains (the SAME files it writes) and
turns a pattern name + camera ROI into a camera-pixel detection grid + per-pattern thresholds. This
is the pyctrl-side READER of the registry; it imports NO yb_analysis code (a pyctrl module that runs
in the engine interpreter), it just reads the JSON / .mat files and ports the tiny affine math.

Files read (paths mirror yb_analysis exactly so we read where it writes):
  * ``<PATH_PREFIX>/yb_dashboard_state/patterns/<name>/record.json``  -- knm trap positions (y,x).
  * ``<PATH_PREFIX>/yb_dashboard_state/affine_transform.json``        -- the global SLM->camera 2x3.
  * ``<PATH_PREFIX>/yb_dashboard_state/patterns/<name>/threshold.mat`` -- per-pattern thresholds.
``PATH_PREFIX`` = ``$YB_PATH_PREFIX`` else the lab default; the patterns dir / affine path honour
``$YB_PATTERNS_DIR`` / ``$YB_AFFINE_PATH`` (same overrides yb_analysis uses).

Everything degrades gracefully to ``None`` when a file is missing or the affine isn't bootstrapped,
so callers (scan-prep sidecar baking, the mid-shot detector) fall back to the day-folder grid.

Design inspired by the MATLAB / yb_analysis original; no brassboard-seq code.
"""

import json
import os

# Mirrors yb_analysis.config.PATH_PREFIX (and scan_prep.DEFAULT_DATA_PREFIX).
DEFAULT_PATH_PREFIX = r"D:\OneDrive - Harvard University\Documents - Yb"


# =========================================================================== #
# paths (mirror pattern_registry._patterns_dir / affine_transform._affine_path)
# =========================================================================== #
def _path_prefix():
    return os.environ.get("YB_PATH_PREFIX", DEFAULT_PATH_PREFIX)


def patterns_dir():
    env = os.environ.get("YB_PATTERNS_DIR")
    if env:
        return env
    return os.path.join(_path_prefix(), "yb_dashboard_state", "patterns")


def affine_path():
    env = os.environ.get("YB_AFFINE_PATH")
    if env:
        return env
    return os.path.join(_path_prefix(), "yb_dashboard_state", "affine_transform.json")


def _record_path(name):
    return os.path.join(patterns_dir(), _sanitize(name), "record.json")


def _pattern_threshold_path(name):
    return os.path.join(patterns_dir(), _sanitize(name), "threshold.mat")


def _sanitize(name):
    """Filesystem-safe pattern name (mirror pattern_registry._sanitize_name)."""
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", (name or "").strip())
    if safe in ("", ".", ".."):
        raise ValueError("invalid pattern name: %r" % (name,))
    return safe


# =========================================================================== #
# registry + affine readers
# =========================================================================== #
def get_pattern_record(name):
    """The full registry record for ``name`` (incl. ``knm``), or None."""
    p = _record_path(name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_affine_matrix():
    """The current global SLM->camera 2x3 affine as a numpy array, or None."""
    p = affine_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    cur = (data or {}).get("current") if isinstance(data, dict) else None
    if not cur or cur.get("A") is None:
        return None
    import numpy as np
    return np.asarray(cur["A"], dtype=np.float64).reshape(2, 3)


def load_pattern_thresholds(name):
    """Per-pattern thresholds (+ infidelities) from ``<name>/threshold.mat``, or None.

    Returns ``{'thresholds': (N,) list, 'infidelities': (N,) list | None}``; mirrors the
    day-folder threshold.mat parsing. None when the file is absent / unreadable."""
    p = _pattern_threshold_path(name)
    if not os.path.isfile(p):
        return None
    try:
        import numpy as np

        def _vec(x):
            return np.asarray(x, dtype=float).ravel().tolist()

        try:
            from scipy.io import loadmat
            d = loadmat(p)
            thr = _vec(d["thresholds"]) if "thresholds" in d else None
            inf = _vec(d["infidelities"]) if "infidelities" in d else None
        except (NotImplementedError, ValueError):
            import h5py
            with h5py.File(p, "r") as f:
                thr = _vec(f["thresholds"]) if "thresholds" in f else None
                inf = _vec(f["infidelities"]) if "infidelities" in f else None
        if thr is None:
            return None
        return {"thresholds": thr, "infidelities": inf}
    except Exception:  # noqa: BLE001 - any read failure -> fall back to the day folder
        return None


# =========================================================================== #
# affine math (ports of affine_transform._knm_to_xy / apply_affine[_cropped])
# =========================================================================== #
def _knm_to_xy(knm):
    """Registry knm is ``[y, x]``; the affine math wants ``[x, y]``. Swap ONCE."""
    import numpy as np
    knm = np.asarray(knm, dtype=np.float64).reshape(-1, 2)
    return knm[:, [1, 0]]


def _apply_affine(knm_xy, A):
    """knm ``[x, y]`` -> absolute camera ``[Y, X]`` via ``[Y,X]^T = A @ [x,y,1]^T``."""
    import numpy as np
    xy = np.asarray(knm_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([xy, np.ones(len(xy))])    # (N,3) [x,y,1]
    return (np.asarray(A, dtype=np.float64).reshape(2, 3) @ hom.T).T   # (N,2) [Y,X]


def _apply_affine_cropped(knm_xy, A, roi):
    """knm ``[x,y]`` -> CROPPED-frame camera ``[Y,X]`` for ``roi = [Xoff, Yoff, W, H]`` (the crop
    offset is applied here, never baked into A)."""
    import numpy as np
    yx = _apply_affine(knm_xy, A)
    xoff, yoff = float(roi[0]), float(roi[1])
    return yx - np.array([yoff, xoff], dtype=np.float64)


# =========================================================================== #
# the convenience the callers use
# =========================================================================== #
def pattern_camera_grid(name, roi):
    """Camera-pixel grid ``(N,2) [Y, X]`` for pattern ``name`` at ``roi = [Xoff, Yoff, W, H]``,
    or None when the registry record / affine is unavailable. Maps the stored knm positions through
    the global affine and the per-scan crop -- exactly what yb_analysis's ``_build_pattern_grids``
    does, so the live monitor and pyctrl agree on the grid."""
    rec = get_pattern_record(name)
    if not rec or not rec.get("knm"):
        return None
    A = load_affine_matrix()
    if A is None:
        return None
    try:
        return _apply_affine_cropped(_knm_to_xy(rec["knm"]), A, roi)
    except Exception:  # noqa: BLE001
        return None


def resolve_pattern_calibration(name, roi):
    """Per-pattern detection calibration for ``name`` at ``roi``, or None.

    Returns ``{'grid': (N,2) [Y,X], 'thresholds': (N,), 'infidelities': (N,) | None, 'n_sites': N}``
    only when BOTH the affine-mapped grid AND per-pattern thresholds resolve with matching site
    counts. Any mismatch / missing piece -> None so the caller uses the day-folder calibration."""
    grid = pattern_camera_grid(name, roi)
    if grid is None or len(grid) == 0:
        return None
    td = load_pattern_thresholds(name)
    if td is None:
        return None
    thr = td["thresholds"]
    if len(thr) != len(grid):
        return None
    inf = td.get("infidelities")
    if inf is not None and len(inf) != len(grid):
        inf = None
    return {"grid": grid, "thresholds": list(thr),
            "infidelities": list(inf) if inf is not None else None,
            "n_sites": len(grid)}
