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
                      num_per_group=0, scan_meta=None, params=None, prefix=None):
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
    if scan_meta:
        # Scan metadata (ScanGroup/ScanName/PlotScale/expConfig) the DataManager reads; it
        # never overrides the frame-routing keys above (those win on a key clash).
        for k, v in scan_meta.items():
            cfg.setdefault(k, v)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, path)                    # atomic publish
    return path
