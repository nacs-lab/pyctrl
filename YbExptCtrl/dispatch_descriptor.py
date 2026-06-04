"""dispatch_descriptor.py -- build a ScanGroup + resolve a seq from a JSON descriptor.

Transliteration of ``matlab_new/YbExptCtrl/dispatch_descriptor.m``, but deliberately
SIMPLER for the pyctrl scenario-3 run loop. In MATLAB the dispatcher builds a ScanGroup,
hands it to ``ybBuildScanPayload`` and submits the resulting MATLAB byte stream via
``srv.server.submit_job(...)``. pyctrl is BOTH producer and consumer of the queue entry
(scenario 3: new monitor + pyctrl), so the payload IS the descriptor JSON itself -- there
is no ``getByteStreamFromArray`` / ``ybBuildScanPayload`` / ``submit_job`` round-trip and
no ROI side-channel. This function does only the cross-backend part: rebuild a ScanGroup
from ``descriptor.params`` + ``descriptor.runp``, resolve the seq function, auto-derive
``NumPerGroup``, and return ``(scangroup, seq, opts)`` for the run loop (run_seq.py) to
consume exactly as ``runSeq2`` consumes a ScanGroup + func.

Only two things are cross-backend contracts (PYTHON_FRONTEND_PLAN.md Phase 5): the
descriptor JSON (this file's input), and the per-point serialized seq bytes (THE ONE
RULE). The MATLAB payload-byte test (``ybStartScan_refactor_test.m``) is a MATLAB-refactor
check, NOT a pyctrl contract -- so the proprietary payload is not reproduced here.

Differences from dispatch_descriptor.m (justified, see references/runtime-design.md):

  * **No field demangling.** ``json.loads`` keys are already clean ("Cooling.Detuning",
    "@"); MATLAB's ``_0x2E_`` / ``x40_`` demangling is a ``jsondecode`` artifact only.
  * **Seq + function-handle resolution = import-by-convention** (mirrors ``str2func``):
    ``importlib.import_module(name); getattr(mod, name)``. Works because ``YbSeqs`` /
    ``YbSteps`` / ``lib`` are flat on ``sys.path`` and the module name equals the attr
    name (verbatim naming, see references/naming.md). A missing module/attr raises
    :class:`NotMigratedError` -- this makes scenario 3's "only ported seqs are runnable"
    loud instead of silent. No registry. ``seq="auto"`` raises in v1 (mirror).
  * **No ExptServer submit / ROI / payload** -- the run loop owns those.
  * **JSON numbers are float-coerced** to match ``jsondecode`` (which always yields
    ``double``). A bare JSON ``1`` left as a Python ``int`` would tag ``ARG_CONST_INT32``
    and break byte-equality for any param used as a SeqVal operand (Phase 3 finding).
    Booleans stay ``bool`` (MATLAB ``logical``); strings / null pass through.

Sweep / single-element-collapse traps reproduced (Phase 5 findings E):

  * ``linspace [a,b,n]`` with ``n == 1`` yields the **STOP** endpoint ``b`` (MATLAB
    ``linspace(a,b,1) == b``), NOT the START (NumPy's ``linspace(a,b,1) == a``).
    ``logspace`` likewise yields ``10**b``.
  * A single-element **numeric** sweep collapses to a FIXED param. That collapse happens
    downstream in :meth:`ScanGroup._addscan` (``_isarray([x])`` is ``False``), but the
    ``n == 1`` endpoint must be computed here first so the collapsed value is ``b`` not
    ``a``.
  * Sweep detection runs **before** function-handle detection; a dict value that is
    neither a sweep (has ``"scan"``) nor a handle (has ``"@"``) is rejected, mirroring
    MATLAB's ``bad_object_value``.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import importlib
import re
from collections import namedtuple

from scan_group import ScanGroup


# A MATLAB identifier (also the regex dispatch_descriptor.m validates seq names against).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class NotMigratedError(Exception):
    """A descriptor referenced a seq / function name that pyctrl has not ported.

    Raised by the import-by-convention resolver when ``import_module(name)`` fails or the
    module has no attribute named ``name``. In scenario 3 only ported (importable) seqs are
    runnable; this surfaces an attempt to run an un-migrated one rather than failing deep in
    the run loop.
    """


# The dispatcher's product, consumed by the pyctrl run loop. ``seq`` is the resolved
# callable (a YbSeqs function), ``seq_name`` the name it was resolved from (for logging /
# the queue label), ``opts`` the decoded varargin pairs, and ``label`` the queue-UI label
# (falls back to the seq name).
DispatchResult = namedtuple(
    "DispatchResult", ["scangroup", "seq", "seq_name", "opts", "label"])


# =========================================================================== #
# public entry point
# =========================================================================== #
def dispatch_descriptor(desc, seq_resolver=None):
    """Build a :class:`ScanGroup` + resolve the seq from a descriptor.

    Args:
        desc: the descriptor -- either a JSON string (conforming to
            ``yb_analysis/scans/descriptor.schema.json``) or an already-decoded dict.
        seq_resolver: optional ``name -> callable`` override (default:
            :func:`_import_by_convention`). Injected by unit tests so they need not import
            the real YbSeqs.

    Returns:
        :class:`DispatchResult`.

    Raises:
        ValueError: malformed descriptor (mirrors dispatch_descriptor.m's error ids).
        NotMigratedError: a referenced seq / handle name is not importable.
    """
    if seq_resolver is None:
        seq_resolver = _import_by_convention
    if isinstance(desc, (str, bytes, bytearray)):
        import json
        desc = json.loads(desc)
    if not isinstance(desc, dict):
        raise ValueError(
            "dispatch_descriptor: descriptor must decode to a JSON object, got %s"
            % type(desc).__name__)

    # --- Build ScanGroup from descriptor.params + descriptor.runp ----------- #
    g = ScanGroup()
    if "params" in desc and desc["params"] is not None:
        _apply_params(g, desc["params"], seq_resolver)
    if "runp" in desc and desc["runp"] is not None:
        _apply_runp(g.runp(), desc["runp"], seq_resolver)
    _auto_derive_runp(g)

    # --- Resolve seq function (string / {"@": name} -> callable) ------------ #
    seq, seq_name = _resolve_seq(desc, seq_resolver)

    # --- Extra opts: [[key, val], ...] -> list of (key, decoded_val) -------- #
    opts = _unpack_opts(desc.get("opts"), seq_resolver)

    label = desc.get("label") or seq_name
    return DispatchResult(scangroup=g, seq=seq, seq_name=seq_name, opts=opts, label=label)


# =========================================================================== #
# ScanGroup population (mirrors apply_paths / assign_one)
# =========================================================================== #
def _apply_params(g, paths, seq_resolver):
    """Assign each dotted path in ``paths`` into the group's fallback scan ``g()`` (idx 0).

    Fixed values -> ``_addparam``; sweeps -> ``_addscan`` (the MATLAB ``subsasgn`` analog,
    see scan_param.py). A single-element numeric sweep collapses to a fixed param inside
    ``_addscan`` (``_isarray`` returns False for a 1-element list).
    """
    if not isinstance(paths, dict):
        raise ValueError(
            "dispatch_descriptor: params must be a JSON object, got %s"
            % type(paths).__name__)
    for raw_key, spec in paths.items():
        path_parts = tuple(raw_key.split("."))
        try:
            is_sweep, dim, vals = _parse_sweep(spec, seq_resolver)
            if is_sweep:
                g._addscan(0, path_parts, dim, vals)
            else:
                g._addparam(0, path_parts, _decode_value(spec, seq_resolver))
        except Exception as e:  # noqa: BLE001 - re-tag with the offending path (mirror)
            raise ValueError("dispatch_descriptor: assigning %s: %s" % (raw_key, e))


def _apply_runp(runp, paths, seq_resolver):
    """Assign each dotted path in ``paths`` into the runp :class:`DynProps`.

    runp holds plain values (scalars / vectors / cell-of-string / handles), never sweeps --
    a sweep object reaches :func:`_decode_value`, which rejects it as ``bad_object_value``,
    matching MATLAB (``subsasgn`` of ``.scan`` onto a DynProps has no meaning).
    """
    if not isinstance(paths, dict):
        raise ValueError(
            "dispatch_descriptor: runp must be a JSON object, got %s"
            % type(paths).__name__)
    for raw_key, spec in paths.items():
        path_parts = tuple(raw_key.split("."))
        try:
            runp._setpath(path_parts, _decode_value(spec, seq_resolver))
        except Exception as e:  # noqa: BLE001
            raise ValueError("dispatch_descriptor: assigning runp.%s: %s" % (raw_key, e))


def _parse_sweep(spec, seq_resolver):
    """Detect the sweep object shape ``{scan:N, linspace|logspace|values:...}``.

    Returns ``(is_sweep, dim, vals)``. ``vals`` is a float list (numeric sweep) or a
    string list (cell-of-string sweep). Mirrors dispatch_descriptor.m::parse_sweep, plus
    the ``n == 1`` STOP-endpoint rule for linspace/logspace.
    """
    if not isinstance(spec, dict) or "scan" not in spec:
        return False, 0, None
    dim = spec["scan"]
    if isinstance(dim, bool) or not isinstance(dim, (int, float)) or \
            isinstance(dim, float) and not dim.is_integer() or dim < 1:
        raise ValueError("sweep .scan must be a positive integer, got %r" % (dim,))
    dim = int(dim)
    has_lin = "linspace" in spec
    has_log = "logspace" in spec
    has_vals = "values" in spec
    n_set = has_lin + has_log + has_vals
    if n_set != 1:
        raise ValueError(
            "sweep must set exactly one of {linspace, logspace, values}; got %d" % n_set)
    if has_lin:
        v = spec["linspace"]
        if _numlen(v) != 3:
            raise ValueError("linspace must be [start, stop, n], got %d elements" % _numlen(v))
        vals = _maybe_collapse(_linspace(float(v[0]), float(v[1]), v[2]))
    elif has_log:
        v = spec["logspace"]
        if _numlen(v) != 3:
            raise ValueError("logspace must be [start_exp, stop_exp, n], got %d elements"
                             % _numlen(v))
        vals = _maybe_collapse(_logspace(float(v[0]), float(v[1]), v[2]))
    else:
        vals = _decode_value_array(spec["values"])
    return True, dim, vals


def _decode_value(spec, seq_resolver):
    """Convert a JSON-decoded scalar/array/handle into the stored value.

      * ``{"@": name}``           -> resolved callable (mirrors ``str2func``)
      * dict without ``"@"``      -> error (sweeps are handled before this is called)
      * numeric scalar            -> float (jsondecode -> double)
      * numeric / string array    -> float list / string list
      * bool / str / None         -> passthrough (logical / char / [])
    """
    if isinstance(spec, dict):
        if "@" in spec:
            return seq_resolver(_handle_name(spec["@"]))
        raise ValueError(
            "unrecognized object value (fields: %s)" % ",".join(sorted(spec.keys())))
    if isinstance(spec, list):
        return _decode_value_array(spec)
    return _coerce_scalar(spec)


def _decode_value_array(arr):
    """Decode an explicit array: numeric list -> float list; string list -> kept as-is.

    Mirrors dispatch_descriptor.m::decode_value_array (numeric/char/logical pass through;
    object arrays are rejected).
    """
    if not isinstance(arr, list):
        # A bare scalar in a "values" slot: treat as a 1-element sweep value (collapses).
        return _coerce_scalar(arr)
    out = []
    for el in arr:
        if isinstance(el, dict):
            raise ValueError("sweep \"values\" must be a numeric or string array, got object")
        out.append(_coerce_scalar(el))
    return _maybe_collapse(out)


def _auto_derive_runp(g):
    """Fill ``runp.NumPerGroup`` if absent OR <= 0 (mirror auto_derive_runp).

    Heuristic: ``max(nseq * 20, 100)``. NOT byte-affecting -- it feeds the runner's
    grouping / the L1 dump only, never per-seq bytes. The docstring of the MATLAB original
    says "nseq * 20" but the code also applies a 100 floor and the ``<= 0`` sentinel.
    """
    runp = g.runp()
    n_per = None
    try:
        n_per = float(runp.NumPerGroup())
    except Exception:  # noqa: BLE001 - absent field: DynProps read raises without a default
        n_per = None
    if n_per is None or n_per <= 0:
        try:
            nseq = g.nseq()
        except Exception:  # noqa: BLE001
            nseq = 1
        runp._setpath(("NumPerGroup",), float(max(nseq * 20, 100)))


# =========================================================================== #
# seq + opts resolution
# =========================================================================== #
def _resolve_seq(desc, seq_resolver):
    """Resolve ``desc['seq']`` to ``(callable, name)``. Mirrors resolve_seq."""
    if "seq" not in desc:
        raise ValueError("dispatch_descriptor: descriptor must set \"seq\"")
    spec = desc["seq"]
    if isinstance(spec, dict) and "@" in spec:
        name = _handle_name(spec["@"])
        return seq_resolver(name), name
    if not isinstance(spec, str):
        raise ValueError(
            "dispatch_descriptor: seq must be a string or {\"@\": \"Name\"}, got %s"
            % type(spec).__name__)
    if spec == "auto":
        # Reserved for future param-based seq resolution (the hook is here; the resolution
        # table is not implemented for v1). Mirrors auto_not_implemented.
        raise ValueError(
            "dispatch_descriptor: seq=\"auto\" is reserved; pass an explicit seq name")
    if not _IDENT_RE.match(spec):
        raise ValueError(
            "dispatch_descriptor: seq name must be a valid identifier, got %s" % spec)
    return seq_resolver(spec), spec


def _unpack_opts(opts_json, seq_resolver):
    """``[[key, value], ...]`` -> ``[(key, decoded_value), ...]`` (decoded varargin pairs).

    MATLAB flattens to a varargin cell ``{key, val, ...}``; pyctrl keeps ordered pairs so
    the run loop can interpret them. Values are decoded so a ``{"@": name}`` handle in an
    opt resolves to a callable.
    """
    if not opts_json:
        return []
    if not isinstance(opts_json, list):
        raise ValueError(
            "dispatch_descriptor: opts must be a JSON list, got %s"
            % type(opts_json).__name__)
    out = []
    for i, kv in enumerate(opts_json):
        if not isinstance(kv, (list, tuple)) or len(kv) != 2:
            raise ValueError("dispatch_descriptor: opts entry %d must be a [key,value] pair" % i)
        key, val = kv
        out.append((str(key), _decode_value(val, seq_resolver)))
    return out


def _import_by_convention(name):
    """Resolve a name to a callable via ``import_module(name); getattr(mod, name)``.

    The pyctrl analog of ``str2func`` (references/runtime-design.md): flat ``sys.path`` +
    verbatim naming make ``module name == attr name`` the convention. Anything not ported
    raises :class:`NotMigratedError`.

    Live edits: ``import_module`` returns the CACHED module, so the long-lived runner drops
    ported experiment modules from ``sys.modules`` once per job (``seq_reload`` via ``run_job``)
    BEFORE resolving -- this re-import then re-reads edited seq/step files from disk (the
    ``rehash()`` + ``str2func`` analog; transitive over imported steps). Edits to ``lib/`` (the
    framework) still need a runner restart.
    """
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise NotMigratedError(
            "seq/function %r is not migrated to pyctrl (no importable module %r): %s"
            % (name, name, e)) from e
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise NotMigratedError(
            "module %r has no attribute %r -- pyctrl naming convention requires "
            "module name == function name" % (name, name)) from e


# =========================================================================== #
# utilities
# =========================================================================== #
def _coerce_scalar(v):
    """JSON scalar -> stored value: int -> float (match jsondecode double); bool/str/None
    pass through. ``bool`` is checked first because ``isinstance(True, int)`` is True."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def _handle_name(name):
    """Validate a ``{"@": name}`` handle name is a bare identifier (mirror str2func input)."""
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError("function-handle name must be a valid identifier, got %r" % (name,))
    return name


def _maybe_collapse(vals):
    """Collapse a single-element NUMERIC/bool array to its scalar (mirror jsondecode /
    ``linspace(a,b,1)``), so it is stored as a FIXED param exactly as MATLAB does.

    A single-element STRING array is NOT collapsed (MATLAB ``jsondecode(["a"])`` keeps a
    1-element cell -- the "refuted" case in PYTHON_FRONTEND_PLAN.md finding E). Multi-element
    arrays pass through unchanged (a real scan axis / vector param)."""
    if isinstance(vals, list) and len(vals) == 1 and not isinstance(vals[0], str):
        return vals[0]
    return vals


def _numlen(v):
    return len(v) if isinstance(v, (list, tuple)) else 1


def _linspace(a, b, n):
    """MATLAB ``linspace(a, b, n)``: inclusive endpoints; ``n == 1`` -> ``[b]`` (the STOP,
    not the start); ``n <= 0`` -> ``[]``."""
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [float(b)]
    step = (b - a) / (n - 1)
    out = [a + step * i for i in range(n)]
    out[-1] = float(b)   # pin the endpoint exactly (avoid float drift), as MATLAB does
    return out


def _logspace(a, b, n):
    """MATLAB ``logspace(a, b, n)``: ``10 ** linspace(a, b, n)``; ``n == 1`` -> ``[10**b]``."""
    return [10.0 ** e for e in _linspace(a, b, n)]
