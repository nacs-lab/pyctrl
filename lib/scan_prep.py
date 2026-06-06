"""scan_prep.py -- write the scan-config the monitor's DataManager reads (JSON, not .mat).

In MATLAB, ``ybBuildScanJob`` writes a ``Scan``-struct ``.mat`` into the dated data dir; the
yb_analysis monitor's ``DataManager`` reads it (via ``scan_id_to_stamps`` -> ``make_scan_dir``)
to learn the scan metadata (frame size, NumImages, isInit/isHC/isGrid2, ...). Without it the
monitor raises "Cannot load <path>" and its ``_process_once`` dies.

pyctrl is a Python backend, so rather than emit a MATLAB ``.mat`` we dump a **JSON** sidecar
(``data_<stamp>.json``) next to where the ``.mat``/``.h5`` live, and the monitor was taught to
prefer that JSON (falling back to ``.mat`` for MATLAB-written scans). Cleaner, no scipy.

For an **isInit scan** (e.g. LACScan) ``DataManager`` reads only ``frameSize`` / ``NumImages`` /
``isInit`` / ``isHC`` / ``isGrid2`` (then creates the HDF5 + saves frames; no detection) -- so
that is what we write, plus ``NumPerGroup`` for display. Path mirrors yb_analysis
``make_scan_dir``: ``<prefix>/Data/<YYYYMMDD>/data_<stamp>/data_<stamp>.json``.

The data root (monitor ``PATH_PREFIX``) is ``$YB_DATA_PREFIX`` if set, else the lab default;
keep it in sync with ``yb_analysis.config.PATH_PREFIX``. Design inspired by the MATLAB original.
"""

import json
import os
import random

# Mirrors yb_analysis.config.PATH_PREFIX (the monitor's data root). Override per-machine with
# $YB_DATA_PREFIX so the backend writes where the monitor reads.
DEFAULT_DATA_PREFIX = r"D:\OneDrive - Harvard University\Documents - Yb"


# =========================================================================== #
# Run-order construction -- ybBuildScanJob's Scan.Params (the production scan
# randomization). The scan loop (run_seq.py / runSeq2.m) just RUNS the order it
# is handed; the stacking + scramble live HERE in the prep layer, exactly as
# MATLAB builds built.Params in ybBuildScanJob.m and hands it to runSeq2.
# =========================================================================== #
def stack(seq, num):
    """Port of ``stack.m`` for a row vector: ``num`` back-to-back copies of ``seq``.

    ``stack([1,2,3], 2) -> [1,2,3,1,2,3]`` -- StackNum passes over the full scan sweep
    (one pass = every scan point exactly once)."""
    out = []
    for _ in range(int(num)):
        out.extend(seq)
    return out


def scramble_groups(seq, group_len, rng):
    """Port of ``scrambleGroups.m``: shuffle the elements of ``seq`` WITHIN each consecutive
    block of ``group_len``, leaving the block boundaries intact (a trailing partial block is
    shuffled too).

    Unlike a single global shuffle (the old ``scramble()`` / runSeq2's ``is_random`` branch),
    each block stays a permutation of the SAME ``group_len`` elements. So when ``seq`` is
    StackNum copies of one full scan sweep (one block = every scan point once), every parameter
    is sampled exactly once per block -- regular revisits, instead of a global randomization
    that can let a point cluster or go unsampled for long stretches."""
    out = list(seq)
    n = len(out)
    if group_len < 1 or n == 0:
        return out
    for start in range(0, n, group_len):
        block = out[start:start + group_len]
        rng.shuffle(block)
        out[start:start + group_len] = block
    return out


def build_scan_order(nseqs, *, stack_num=2, scramble=False, rng=None):
    """Build the production scan run-order -- ybBuildScanJob's ``Scan.Params``.

    Returns a list of 1-based scan-point indices: ``stack_num`` back-to-back passes over
    ``1..nseqs`` (one pass = every point once), each pass independently shuffled when
    ``scramble`` (``scramble_groups``, NOT a global shuffle). This list IS the seq_id -> scan
    point map the live scan curve needs (shot ``i`` ran point ``order[i-1]``).

    ``stack_num`` is the number of passes (MATLAB ``StackNum``); the caller derives it from an
    explicit rep or ``max(ceil(NumPerGroup/nseqs), 2)``. ``rng`` (a ``random.Random``) defaults
    to a fresh PRNG; pass a seeded one for reproducibility / tests."""
    nseqs = int(nseqs)
    if nseqs <= 0 or stack_num < 1:
        return []
    params = stack(list(range(1, nseqs + 1)), stack_num)
    if scramble:
        if rng is None:
            rng = random.Random()
        params = scramble_groups(params, nseqs, rng)
    return params


def _read_threshold_mat(path):
    """Read ``thresholds`` + ``infidelities`` from a MATLAB ``threshold.mat``
    (v7 via scipy, v7.3 via h5py). Returns ``(thresholds, infidelities)`` as
    1-D float lists; either may be ``None`` if absent. Self-contained (lazy
    imports) so this module has no hard numpy/scipy dependency at import time.
    """
    import numpy as np

    def _vec(x):
        return np.asarray(x, dtype=float).ravel().tolist()

    try:
        from scipy.io import loadmat
        d = loadmat(path)
        thr = _vec(d["thresholds"]) if "thresholds" in d else None
        inf = _vec(d["infidelities"]) if "infidelities" in d else None
        return thr, inf
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            thr = _vec(f["thresholds"]) if "thresholds" in f else None
            inf = _vec(f["infidelities"]) if "infidelities" in f else None
            return thr, inf


def load_day_calibration(day_dir):
    """Detection calibration fields for the scan-config JSON, read from the
    day folder — mirrors MATLAB ``ybBuildScanPayload.m`` (the non-isInit
    branch): ``gridLocations.txt`` (header ``Y\\tX``) + ``threshold.mat``
    (``thresholds`` / ``infidelities``).

    Returns a dict with ``initGridLocationsX/Y`` + ``initThresholds`` +
    ``initInfidelities`` (+ ``boxSize`` / ``maskSigma``) when the files are
    present and consistent, else ``{}`` (silently — matching MATLAB's
    arrayConfig fallback; the offline analysis then shows empty per-site
    panels but does not crash). Best-effort: any read error → ``{}``.
    """
    grid_file = os.path.join(day_dir, "gridLocations.txt")
    thresh_file = os.path.join(day_dir, "threshold.mat")
    if not (os.path.isfile(grid_file) and os.path.isfile(thresh_file)):
        return {}
    try:
        import numpy as np
        rows = []
        with open(grid_file, "r") as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", "\t").split("\t")
                # skip a non-numeric header row (the "Y\tX" line)
                try:
                    y, x = float(parts[0]), float(parts[1])
                except (ValueError, IndexError):
                    if ln == 0:
                        continue
                    return {}
                rows.append((y, x))
        grid = np.asarray(rows, dtype=float).reshape(-1, 2)   # [Y, X]
        thr, inf = _read_threshold_mat(thresh_file)
        m = grid.shape[0]
        if m == 0 or thr is None or len(thr) != m:
            return {}   # missing / mismatched -> don't write partial calibration
        out = {
            "initGridLocationsY": grid[:, 0].tolist(),
            "initGridLocationsX": grid[:, 1].tolist(),
            "initThresholds": list(thr),
            "boxSize": 9,        # MATLAB ybBuildScanPayload constants
            "maskSigma": 2,
        }
        if inf is not None and len(inf) == m:
            out["initInfidelities"] = list(inf)
        return out
    except Exception:
        return {}


def resolve_calibration(day_dir, image_patterns=None, roi=None):
    """Per-site detection calibration for the sidecar -- PREFER the per-pattern registry.

    When the scan declares loading pattern(s) and the registry + global affine resolve the
    frame-0 pattern, bake its affine-mapped camera grid + per-pattern thresholds/infidelities into
    ``initGridLocations*`` / ``initThresholds`` / ``initInfidelities`` so the OFFLINE analysis
    (run_analysis) scores with PATTERN thresholds. Falls back to the day-folder calibration
    (``load_day_calibration``) when no pattern is declared or the registry/affine isn't available
    -- so nothing is lost (the day-folder path is preserved). ``roi`` is the full imaging ROI
    ``[Xoff, Yoff, W, H]`` (needed for the affine crop)."""
    if image_patterns and roi is not None:
        spec0 = image_patterns[0] if image_patterns else None
        name = spec0.get("name") if isinstance(spec0, dict) else None
        if name:
            try:
                from pattern_grid import resolve_pattern_calibration
                pc = resolve_pattern_calibration(name, roi)
            except Exception:
                pc = None
            if pc is not None:
                grid = pc["grid"]
                out = {
                    "initGridLocationsY": [float(v) for v in grid[:, 0]],
                    "initGridLocationsX": [float(v) for v in grid[:, 1]],
                    "initThresholds": [float(t) for t in pc["thresholds"]],
                    "boxSize": 9,
                    "maskSigma": 2,
                    "calibrationSource": "pattern:%s" % name,
                }
                if pc.get("infidelities") is not None:
                    out["initInfidelities"] = [float(x) for x in pc["infidelities"]]
                return out
    return load_day_calibration(day_dir)


def _data_root_of(scan_config_json_path):
    """``<prefix>/Data`` from a ``…/Data/<date>/data_<stamp>/data_<stamp>.json`` path."""
    return os.path.dirname(os.path.dirname(os.path.dirname(scan_config_json_path)))


def _scan_id_to_iso(scan_id):
    """14-digit ``YYYYMMDDHHMMSS`` -> ISO-8601 (best-effort; None on malformed)."""
    s = str(int(scan_id))
    if len(s) != 14:
        return None
    return "%s-%s-%sT%s:%s:%s" % (s[0:4], s[4:6], s[6:8], s[8:10], s[10:12], s[12:14])


def _capture_code_snapshot(scan_id, scan_config_json_path, scan_meta):
    """Best-effort per-run code snapshot -> the compact dict for ``cfg['code_snapshot']``.

    Disabled by setting ``$YB_CODE_SNAPSHOT=0``. Returns ``None`` on disable or ANY error --
    snapshotting must never affect the run or the sidecar. Mirrors the SLM server's capture
    (content-addressed blobs + per-run manifest under ``<data_root>/_code_snapshots``)."""
    if os.environ.get("YB_CODE_SNAPSHOT", "1") == "0":
        return None
    try:
        import code_snapshot
        # If THIS run is replaying a prior run's code (a '+code' re-queue), the live disk is NOT
        # what executed -- record an honest pointer to the replayed run instead of re-snapshotting
        # the current tree. scan-prep runs inside the replay context, so this marker is valid.
        replay_src = code_snapshot.active_replay_source()
        if replay_src is not None:
            return {"scan_id": int(scan_id),
                    "replayed_from_scan_id": int(replay_src),
                    "snapshot_dir": "_code_snapshots",
                    "note": "experiment code replayed from run %d's snapshot" % int(replay_src)}
        root = code_snapshot.pyctrl_root()
        seq_name = None
        if isinstance(scan_meta, dict):
            seq_name = scan_meta.get("ScanName")
        git_state = code_snapshot.read_git_state(root)
        return code_snapshot.snapshot_code(
            root, _data_root_of(scan_config_json_path),
            run_id=int(scan_id),
            run_started_iso=_scan_id_to_iso(scan_id),
            seq_name=seq_name,
            git_state=git_state)
    except Exception:  # noqa: BLE001 - provenance is never worth failing a run
        return None


def scan_dir(scan_id, prefix=None):
    """The dated scan dir for ``scan_id`` (14-digit YYYYMMDDHHMMSS), mirroring make_scan_dir:
    ``<prefix>/Data/<YYYYMMDD>/data_<YYYYMMDD>_<HHMMSS>``."""
    prefix = prefix or os.environ.get("YB_DATA_PREFIX", DEFAULT_DATA_PREFIX)
    s = str(int(scan_id))
    if len(s) != 14:
        raise ValueError("scan_id must be 14 digits (YYYYMMDDHHMMSS), got %r" % (scan_id,))
    date, time = s[:8], s[8:]
    return os.path.join(prefix, "Data", date, "data_%s_%s" % (date, time))


def scan_config_path(scan_id, prefix=None):
    """The JSON scan-config path: ``<scan_dir>/data_<stamp>.json``."""
    d = scan_dir(scan_id, prefix=prefix)
    return os.path.join(d, "data_%s.json" % os.path.basename(d)[len("data_"):])


def write_scan_config(scan_id, frame_wh, num_images, *, is_init=0, is_hc=0, is_grid2=0,
                      num_per_group=0, scan_meta=None, params=None, prefix=None,
                      image_patterns=None, roi=None, descriptor=None):
    """Write the JSON scan-config for ``scan_id``; return the path.

    Args:
        scan_id: 14-digit YYYYMMDDHHMMSS (the value frames are stamped with).
        frame_wh: ``(W, H)`` of the imaging ROI (MATLAB ``frameSize = [W, H]``; the monitor
            transposes to ``(H, W)``). ``(0, 0)`` when no camera -> monitor skips HDF5 but
            does not crash.
        num_images / is_init / is_hc / is_grid2 / num_per_group: from the descriptor runp.
        params: the realized run order -- ybBuildScanJob's ``Scan.Params``: a flat list whose
            ``i``-th entry (1-based) is the scan-point index shot ``i`` ran. The monitor's
            ``DataManager`` reads it as ``config['Params']`` to bucket each shot's result onto
            the right scan-curve x-value (``compute_scan_curve``). ``None`` -> no curve (the
            pre-2026-06 behavior). Built by ``build_scan_order``.
        scan_meta: optional dict of extra scan-config fields the monitor's ``DataManager``
            reads for live scan-info -- ``ScanGroup`` (``base.vars`` swept axes + ``base.params``
            fixed/``g()`` overrides), ``ScanName`` (scan title), ``PlotScale``, and the baseline
            ``expConfig`` snapshot (provenance). Built by
            ``scan_summary.scangroup_scan_config``. Merged last; ``None`` -> frame metadata only
            (the pre-2026-06-03 behavior).
    """
    path = scan_config_path(scan_id, prefix=prefix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = int(frame_wh[0]), int(frame_wh[1])
    cfg = {
        "frameSize": [w, h],                 # MATLAB [W, H]
        "NumImages": int(num_images),
        "NumPerGroup": int(num_per_group),
        "isInit": int(is_init),
        "isHC": int(is_hc),
        "isGrid2": int(is_grid2),
        "scan_id": int(scan_id),
        "source": "pyctrl",
    }
    if params is not None:
        # Scan.Params: shot (1-based seq_id) -> scan-point index. The seq_id->point map the
        # live scan curve buckets on (data_manager reads config['Params']).
        cfg["Params"] = [int(x) for x in params]
    # Per-image loading-pattern declaration (loading-pattern affine migration): the SAME
    # imagePatternsJson shape MATLAB's ybLoadingPatternsJson emits. The LIVE monitor reads it to
    # build per-pattern grids + load/refit/save per-pattern thresholds (to BOTH the day folder and
    # <pattern>/threshold.mat); the offline analysis reads it for the pattern-name display.
    if image_patterns:
        cfg["imagePatternsJson"] = json.dumps(image_patterns)
    # The full imaging ROI [Xoff, Yoff, W, H] (camera.current_roi()). The LIVE monitor needs it
    # to map each pattern's knm -> camera pixels: data_manager._build_pattern_grids() bails
    # immediately when config['roi'] is missing, so WITHOUT this the per-pattern grid + threshold
    # registry are never built and the per-pattern thresholds FREEZE (never refit/saved). MATLAB
    # scans carry the ROI too. The rearrange bits score with these pattern thresholds.
    if roi is not None:
        cfg["roi"] = [int(v) for v in list(roi)[:4]]
    # Detection calibration (grid + thresholds + infidelities), baked into the
    # JSON like MATLAB ybBuildScanPayload bakes it into the .mat -- so the
    # OFFLINE analysis (run_analysis) gets per-site maps + thresholds +
    # discrimination for pyctrl runs. PREFER the per-pattern registry (pattern
    # thresholds) when a pattern is declared; else the day-folder calibration.
    # Skipped for isInit scans (no detection yet); graceful when neither exists.
    if not is_init:
        day_dir = os.path.dirname(os.path.dirname(path))   # <prefix>/Data/<YYYYMMDD>
        calib = resolve_calibration(day_dir, image_patterns, roi)
        cfg.update(calib)
    if scan_meta:
        # Scan metadata (ScanGroup/ScanName/PlotScale/expConfig) the DataManager reads; it
        # never overrides the frame-routing keys above (those win on a key clash).
        for k, v in scan_meta.items():
            cfg.setdefault(k, v)
    # Self-contained reconstruction descriptor (SeqPlotter): the exact descriptor JSON
    # (scangroup_to_descriptor's output -- params + sweeps + seq name + runp) so an OFFLINE
    # reconstruction can rebuild the ScanGroup + resolve the seq function with
    # dispatch_descriptor, WITHOUT needing the live queue history. Additive metadata (does
    # NOT affect serialize() bytes / THE ONE RULE). Best-effort -> omitted on failure.
    if descriptor is not None:
        cfg["descriptor"] = descriptor
    # Per-run source-code snapshot (additive provenance; mirrors the SLM server's
    # code_snapshot). Content-addressed blobs + a readable/importable per-run tree under
    # <data_root>/_code_snapshots; the compact result rides in cfg['code_snapshot'] so the
    # offline analysis can show "what code ran" and a re-queue can replay it. STRICTLY
    # best-effort: a failure never affects the run or the rest of the sidecar.
    snap = _capture_code_snapshot(scan_id, path, scan_meta)
    if snap is not None:
        cfg["code_snapshot"] = snap
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, path)                    # atomic publish
    return path
