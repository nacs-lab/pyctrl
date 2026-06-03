"""scan_export.py -- export an imperatively-built ScanGroup to a descriptor dict.

The INVERSE of ``dispatch_descriptor`` (``YbExptCtrl/dispatch_descriptor.py``). It lets a
pyctrl "scan file" be written MATLAB-style -- build a :class:`ScanGroup` field by field, then
submit it (``YbExptCtrl/yb_start_scan.py`` :func:`ybStartScan`) -- while keeping the SINGLE
intra-backend payload the run loop already consumes: the descriptor JSON. (Option A, chosen
2026-06-02; option B -- shipping a ScanGroup dump as a 2nd job-payload format -- was declined.)

Round-trip contract: ``dispatch_descriptor(scangroup_to_descriptor(g, seq))`` must rebuild a
group whose ``getseq(n)`` enumeration + per-point serialized bytes match ``g``'s. So the
exporter walks the SAME data ``dispatch_descriptor`` writes:

    fixed params  -> ``{"<dotted.path>": value}``                      (decode: _addparam)
    sweep axes    -> ``{"<dotted.path>": {"scan": dim, "values":[...]}}`` (decode: _addscan)
    runp leaves   -> ``descriptor["runp"]`` (plain values; never sweeps)
    seq           -> the resolved name (callable -> ``__name__``; str passes through)
    opts          -> ``[[key, value], ...]`` (callables -> ``{"@": name}`` handles)

Scope (matches the Phase-4 ScanGroup survey + dispatch_descriptor): a SINGLE scan group, fixed
params + 1-D/2-D sweeps. ``groupsize() != 1`` raises -- the descriptor model has no multi-group
form, and neither does ``dispatch_descriptor``.

Authoring DSL reminder (scan_param.py): a sweep is ``g(1).a.b.scan(dim, vals)`` /
``g(1).a.b.scan(vals)`` -- assigning an array with ``=`` stores a FIXED vector, not a sweep.
Use :func:`linspace` / :func:`logspace` (MATLAB endpoint semantics) to build the value array.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import re

from scan_group import ScanGroup, _foreach_nonstruct

# Mirrors yb_analysis/scans/descriptor.py SCHEMA_VERSION (pyctrl's dispatch_descriptor ignores
# it, but emit it for forward-compat with the JSON descriptor validator).
SCHEMA_VERSION = 1

# A MATLAB identifier (same validation dispatch_descriptor uses for seq / handle names).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# =========================================================================== #
# public API
# =========================================================================== #
def scangroup_to_descriptor(scangroup, seq, opts=None, label=None,
                            schema_version=SCHEMA_VERSION):
    """Build a descriptor dict from a single-group :class:`ScanGroup` + a seq.

    Args:
        scangroup: the :class:`ScanGroup` (``groupsize() == 1``).
        seq: the seq function (callable -> its ``__name__``) or its name (str).
        opts: extra run options -- a dict of kwargs (``rep`` / ``random`` / ``tstartwait`` /
            ``pre_cb`` / ``post_cb``) or a list of ``(key, value)`` pairs. Callable values
            are exported as ``{"@": name}`` handles (resolved on the backend by name).
        label: queue-UI label (defaults to the seq name downstream).

    Returns:
        dict -- a descriptor conforming to ``descriptor.schema.json`` (the
        :func:`dispatch_descriptor.dispatch_descriptor` input).
    """
    if not isinstance(scangroup, ScanGroup):
        raise TypeError("scangroup_to_descriptor expects a ScanGroup, got %s"
                        % type(scangroup).__name__)
    if scangroup.groupsize() != 1:
        raise ValueError(
            "scangroup_to_descriptor supports a SINGLE scan group (the descriptor model has "
            "no multi-group form); got groupsize()==%d" % scangroup.groupsize())

    # The merged (base + scan-1) view -- the exact content getseq enumerates.
    full = scangroup._getfullscan(1)

    params = {}
    for value, path in _foreach_nonstruct(full["params"]):
        params[".".join(path)] = _encode_value(value)
    for dim, var in enumerate(full["vars"], start=1):
        if not var or var.get("size", 0) == 0:
            continue                                     # an empty/dummy axis contributes nothing
        for vals, path in _foreach_nonstruct(var["params"]):
            params[".".join(path)] = {"scan": dim, "values": _encode_value(vals)}

    desc = {"schema_version": schema_version, "seq": _seq_name(seq)}
    if params:
        desc["params"] = params
    runp = _export_runp(scangroup.runp())
    if runp:
        desc["runp"] = runp
    enc_opts = _encode_opts(opts)
    if enc_opts:
        desc["opts"] = enc_opts
    if label:
        desc["label"] = str(label)
    return desc


def linspace(start, stop, n):
    """MATLAB ``linspace(start, stop, n)`` -- inclusive endpoints; ``n == 1`` -> ``[stop]``.

    Matches dispatch_descriptor's decode of a ``linspace`` sweep (the STOP-endpoint rule),
    so ``g(1).x.scan(linspace(a, b, n))`` round-trips through the descriptor exactly.
    """
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [float(stop)]
    step = (stop - start) / (n - 1)
    out = [start + step * i for i in range(n)]
    out[-1] = float(stop)
    return out


def logspace(start_exp, stop_exp, n):
    """MATLAB ``logspace(a, b, n)`` -- ``10 ** linspace(a, b, n)``; ``n == 1`` -> ``[10**b]``."""
    return [10.0 ** e for e in linspace(start_exp, stop_exp, n)]


# =========================================================================== #
# value / seq / opts / runp encoders
# =========================================================================== #
def _encode_value(v):
    """Encode a stored ScanGroup value for the descriptor JSON (inverse of _decode_value).

      * callable        -> ``{"@": name}``  (handle; name must be a bare identifier)
      * bool            -> bool             (checked before int)
      * int / float     -> float            (jsondecode yields double; keep the round-trip)
      * str / None      -> passthrough
      * list / tuple    -> list of encoded scalars (a fixed vector OR a sweep value array)
      * numpy array/scalar -> list / float
    """
    if callable(v):
        name = getattr(v, "__name__", None)
        if not name or not _IDENT_RE.match(name):
            raise ValueError(
                "cannot export function-handle value without a simple identifier name: %r" % v)
        return {"@": name}
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return [_encode_scalar(x) for x in v]
    arr = _maybe_numpy_to_list(v)
    if arr is not None:
        return arr
    raise ValueError("unsupported parameter value type for export: %r" % type(v).__name__)


def _encode_scalar(x):
    """Encode one element of a vector / sweep value array (no nested lists allowed)."""
    if callable(x):
        raise ValueError("function handles are not allowed inside a value array")
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x)
    if x is None or isinstance(x, str):
        return x
    sc = _maybe_numpy_to_list(x)
    if isinstance(sc, float) or isinstance(sc, (str,)):
        return sc
    raise ValueError("unsupported array element type for export: %r" % type(x).__name__)


def _maybe_numpy_to_list(v):
    """Return a JSON-able form of a numpy array (list) / scalar (float), else ``None``."""
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(v, np.ndarray):
        return [_encode_scalar(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return float(v.item()) if not isinstance(v.item(), str) else v.item()
    return None


def _seq_name(seq):
    """Resolve the seq to a bare-identifier name (callable -> ``__name__``; str validated)."""
    if callable(seq):
        name = getattr(seq, "__name__", None)
    elif isinstance(seq, str):
        name = seq
    else:
        raise TypeError("seq must be a callable or a name string, got %s" % type(seq).__name__)
    if not name or not _IDENT_RE.match(name):
        raise ValueError("seq name must be a valid identifier, got %r" % (name,))
    return name


def _export_runp(runp):
    """Flatten the runp :class:`DynProps` store into ``{dotted.path: value}`` (no sweeps)."""
    store = runp()                                       # DynProps() -> the underlying dict
    out = {}
    if isinstance(store, dict) and store:
        for value, path in _foreach_nonstruct(store):
            out[".".join(path)] = _encode_value(value)
    return out


def _encode_opts(opts):
    """``{key: value}`` or ``[(key, value), ...]`` -> ``[[key, encoded_value], ...]``."""
    if not opts:
        return []
    items = opts.items() if isinstance(opts, dict) else opts
    out = []
    for kv in items:
        if isinstance(kv, (list, tuple)) and len(kv) == 2:
            key, val = kv
        else:
            raise ValueError("opts entries must be (key, value) pairs, got %r" % (kv,))
        out.append([str(key), _encode_value(val)])
    return out
