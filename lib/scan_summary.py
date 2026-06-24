"""scan_summary.py -- scan metadata for the dashboard (queue summary + DataManager config).

Two consumers on the yb_analysis monitor read scan metadata that pyctrl must produce; this
module is the single source for both, ported from MATLAB's two producers:

  * **Queue panel** (``dashboard._q_row`` / ``_q_axes``; Tkinter ``queue_pane``) reads each
    queue entry's ``summary`` dict -- the shape of ``ybScanSummary.m``: ``scan_name`` /
    ``scan_filename``, ``axes`` (``[{dim,name,min,max,npts}]``), ``num_per_group`` /
    ``total_per_group`` / ``num_images``, ``set_params`` / ``default_params``.
    :func:`build_descriptor_summary` builds it from the DESCRIPTOR JSON (the single
    intra-backend payload; see ``scan_export.scangroup_to_descriptor``), which carries the
    same data MATLAB reads off a live ScanGroup:

        fixed params -> ``{"<dotted.path>": value}``                       -> set_params
        sweep axes   -> ``{"<dotted.path>": {"scan": dim, "values": [...]}}`` -> axes
        runp leaves  -> ``descriptor["runp"]``                             -> flags / counts
        seq / label  -> ``descriptor["seq"]`` / ``["label"]``              -> scan_filename / name

  * **Live scan-info** (``DataManager``; the status panel + scan-curve x-axis) reads the
    on-disk scan-config's ``config['ScanGroup']['base']['vars']`` (``extract_scan_dims`` ->
    ``scan_param_path`` + the swept values) and ``config['ScanName']['scanname']``
    (``_extract_scan_title`` -> ``scan_name``). :func:`scangroup_scan_config` builds those
    fields straight off the (post-dispatch) ScanGroup, mirroring ``ybBuildScanJob``'s
    ``Scan.ScanGroup = scangroup.dump()`` + ``Scan.PlotScale`` -- but emits a purpose-built
    ``base.vars`` (the SWEEPS, which in pyctrl live in ``scans[*].vars``, not ``base.vars``)
    so ``extract_scan_dims`` finds them regardless of the internal base/scans split.

Design inspired by the MATLAB originals (ybScanSummary.m, ybBuildScanJob.m); no brassboard-seq
code.
"""

import json
import math

from scan_group import _foreach_nonstruct


# =========================================================================== #
# Surface A -- the queue-panel summary (ybScanSummary.m), from the descriptor
# =========================================================================== #
def build_descriptor_summary(descriptor):
    """Build the ``ybScanSummary``-shaped dict from a descriptor (dict or JSON string).

    Returns a JSON-able dict with ``axes`` / ``set_params`` / ``num_per_group`` /
    ``total_per_group`` / ``scan_name`` / ``scan_filename`` etc. Never raises on a malformed
    descriptor -- returns whatever it could extract (the queue UI degrades gracefully)."""
    desc = _as_dict(descriptor)
    params = desc.get("params") or {}
    runp = desc.get("runp") or {}

    axes = _axes_from_params(params)
    set_params = {path: _scalarize(v) for path, v in params.items()
                  if not _is_sweep(v)}

    num_per_group = int(_num(runp.get("NumPerGroup"), 0))
    num_images = int(_num(runp.get("NumImages"), 0))
    rep = _rep_from_opts(desc)
    # nseqs = product over DISTINCT scan dims. Co-swept params share a dim (same npts, enforced
    # by ScanGroup) -> count that dim's length ONCE, not once per param -- otherwise a 3-param
    # co-sweep on dim 1 reads as an 80x80x80 grid. A genuine multi-dim grid (different dims)
    # still multiplies. Matches ScanGroup.nseq() / the run loop's _build_scan_order.
    by_dim = {}
    for ax in axes:
        by_dim[ax.get("dim", 0)] = max(int(ax.get("npts") or 1), 1)
    nseqs = 1
    for n in by_dim.values():
        nseqs *= n
    if not axes:
        nseqs = 0

    # total_per_group = the number of shots the scan is SET to run ("supposed to do") =
    # nseqs * StackNum. StackNum honors an explicit ``rep`` opt EXACTLY as the run loop does
    # (sequence_runner._build_scan_order): a ``rep >= 1`` is the deliberate pyctrl pass-count
    # override that BYPASSES the NumPerGroup formula -- so a 4-pt sweep with rep=3 is 12 shots,
    # NOT the NumPerGroup-derived StackNum. With no explicit rep (or rep==0 run-forever, which has
    # no finite plan), fall back to ybScanSummary's StackNum = max(ceil(NumPerGroup / nseqs), 2).
    if nseqs > 0 and rep is not None and rep >= 1:
        total_per_group = nseqs * int(rep)
    elif nseqs > 0 and num_per_group > 0:
        stack_num = max(math.ceil(num_per_group / nseqs), 2)
        total_per_group = nseqs * stack_num
    else:
        total_per_group = num_per_group

    label = desc.get("label")
    seq_name = _seq_name(desc.get("seq"))
    scan_name = (str(label) if label else "") or seq_name

    return {
        "axes": axes,
        "num_per_group": num_per_group,
        "num_images": num_images,
        "scramble": bool(_num(runp.get("Scramble"), 0)),
        "is_init": bool(_num(runp.get("isInit"), 0)),
        "is_hc": bool(_num(runp.get("isHC"), 0)),
        "rearrangement": bool(_num(runp.get("Rearrangement"), 0)),
        "nseqs": nseqs,
        "rep": rep,                    # explicit pass-count override (None if unset/forever)
        "total_per_group": total_per_group,
        "scan_filename": scan_name,
        "scan_name": scan_name,
        "set_params": set_params,
        "default_params": {},          # ybScanSummary's whitelist+Consts fill is not ported
        "source": "pyctrl",
    }


def _axes_from_params(params):
    """``[{dim,name,min,max,npts,scale,units}]`` for every sweep entry, ordered by dim."""
    axes = []
    for path, v in params.items():
        if not _is_sweep(v):
            continue
        vals = [float(x) for x in (v.get("values") or []) if _isnum(x)]
        ax = {
            "dim": int(v.get("scan", 0)),
            "name": path,
            "npts": len(v.get("values") or []),
            "scale": 1,
            "units": "",
        }
        if vals:
            ax["min"] = min(vals)
            ax["max"] = max(vals)
        axes.append(ax)
    axes.sort(key=lambda a: a["dim"])
    return axes


# =========================================================================== #
# Surface B -- the DataManager scan-config fields, from the (dispatched) ScanGroup
# =========================================================================== #
def scangroup_scan_config(scangroup, scan_name=None, expconfig=None, description=None,
                          background=False):
    """Build the scan-config fields the monitor's ``DataManager`` reads for live scan-info.

    Returns ``{ScanGroup, ScanName?, description?, background?, PlotScale, expConfig?}`` where
    ``ScanGroup.base.vars`` is
    the ``{params:[dim-struct,...], size:[...]}`` shape ``extract_scan_dims`` expects (the
    swept leaves + per-dim length) and ``ScanGroup.base.params`` is the fixed/``g()``-override
    struct (provenance; MATLAB's ``Scan.ScanName = scangroup.base.params``). ``scan_name`` ->
    ``ScanName.scanname`` (uint16 char codes, as ``_extract_scan_title`` decodes). ``description``
    (the descriptor's free-text run purpose/context) is stamped as a top-level ``description`` key
    only when non-empty -- the analysis dashboard reads it for the run's collapsible note + run
    search. ``background`` (True for a background/calibration-lane run) is stamped as a top-level
    ``background`` key only when True, so the saved scan is explicitly marked as a background run
    (the data is otherwise saved exactly like a normal scan, with its own scan_id + folder).
    ``expconfig`` (the baseline ``SeqConfig.consts`` snapshot) is embedded verbatim for provenance.

    Best-effort: a ScanGroup that doesn't support the query API yields a minimal dict."""
    cfg = {}
    base = {}
    try:
        base["vars"] = _base_vars(scangroup)
    except Exception:  # noqa: BLE001
        base["vars"] = {"params": [], "size": []}
    try:
        base["params"] = _jsonable(scangroup.get_fixed(1))
    except Exception:  # noqa: BLE001
        pass
    cfg["ScanGroup"] = {"version": 1, "base": base}

    try:
        cfg["PlotScale"] = float(_runp_get(scangroup, "PlotScale", 1))
    except Exception:  # noqa: BLE001
        cfg["PlotScale"] = 1.0

    if scan_name:
        cfg["ScanName"] = {"scanname": [ord(c) for c in str(scan_name)]}
    if description:
        cfg["description"] = str(description)
    if background:
        cfg["background"] = True
    if expconfig is not None:
        cfg["expConfig"] = _jsonable(expconfig)
    return cfg


def _base_vars(scangroup):
    """``{params:[dim-struct,...], size:[...]}`` from the non-dummy scan dims of group 1.

    Each dim-struct is the nested param tree for that dimension (e.g.
    ``{"GreenMOT": {"BiasCoilCurrent": {"Y": [..]}}}``) so ``_find_first_numeric`` recovers the
    dotted path; ``size[i]`` is the dim length. Dummy (size-0) dims are skipped."""
    dim_structs, sizes = [], []
    ndim = scangroup.scandim(1)
    for dim in range(1, ndim + 1):
        params, size = scangroup.get_vars(1, dim)
        if not size:
            continue
        dim_structs.append(_jsonable(params))
        sizes.append(int(size))
    return {"params": dim_structs, "size": sizes}


# =========================================================================== #
# helpers
# =========================================================================== #
def _as_dict(descriptor):
    if isinstance(descriptor, (bytes, bytearray)):
        descriptor = descriptor.decode("utf-8", errors="replace")
    if isinstance(descriptor, str):
        try:
            descriptor = json.loads(descriptor)
        except Exception:  # noqa: BLE001
            return {}
    return descriptor if isinstance(descriptor, dict) else {}


def _is_sweep(v):
    """A descriptor sweep value: ``{"scan": dim, "values": [...]}``."""
    return isinstance(v, dict) and "scan" in v and "values" in v


def _rep_from_opts(desc):
    """The explicit ``rep`` pass-count from the descriptor ``opts``, or ``None`` if unset.

    ``rep`` rides in ``descriptor["opts"]`` (``[[key, val], ...]`` from ``scan_export._encode_opts``,
    or a dict), NOT in ``runp`` -- so the StackNum/NumPerGroup math must look here to learn the
    scan's deliberate pass-count override. ``rep == 0`` is run-forever (no finite plan) and is
    returned as ``0``; a non-integer/absent rep -> ``None``."""
    opts = desc.get("opts")
    if not opts:
        return None
    items = opts.items() if isinstance(opts, dict) else opts
    for kv in items:
        try:
            key, val = kv
        except (TypeError, ValueError):
            continue
        if key == "rep":
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def _seq_name(seq):
    if isinstance(seq, str):
        return seq
    if isinstance(seq, dict):
        return str(seq.get("@", "")) or ""
    return ""


def _runp_get(scangroup, name, default):
    """Read a runp leaf (``runp().<name>(default)``), tolerant of absence."""
    rp = scangroup.runp()
    try:
        return getattr(rp, name)(default)
    except Exception:  # noqa: BLE001
        return default


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(x, default):
    return x if _isnum(x) else default


def _scalarize(v):
    """A set-param leaf for display: scalars pass through, handles -> their name."""
    if isinstance(v, dict) and "@" in v:
        return str(v.get("@"))
    return v


def _jsonable(obj):
    """Recursively coerce a params tree (dicts, lists, numpy) to JSON-able Python."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return [_jsonable(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return obj
