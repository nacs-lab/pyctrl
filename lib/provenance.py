"""provenance.py -- param<->channel provenance capture (SeqPlotter Task 4c).

INERT BY DEFAULT. The two hooks below (:func:`on_access`, :func:`on_pulse`) short-circuit
on a ``None`` module-global, so a normal build / ``serialize()`` is byte-identical and pays
only one ``is None`` check per ``DynProps`` leaf read and per ``TimeStep`` pulse. A separate
OFFLINE tool (``tools/provenance_scan.py``, also driven from ``tools/reconstruct_scan.py``)
opens a :class:`ProvenanceSession` around an INSTRUMENTED rebuild, then writes
``<scan>/sequence/xref.json`` -- read by ``yb_analysis/sequence/xref.py`` to light up the
Sequence tab's param<->channel affordance. **Never activate during a real run** (it perturbs
constant types, which is fine for the viewer artifact but not for the byte path).

Why a build-time capture at all (SEQPLOTTER_INTEGRATION_PLAN.md §8): pyctrl bakes each
scanned value as a CONSTANT per point (usevar globals are dormant -- run_seq.py:30), so a
scanned param has no SeqVal/global identity to trace statically; and ``DynProps.get_accessed``
only says a path was *read*, not that it *flows to a channel output*. The accurate answer is
to watch the value flow during the build. Two layers, merged into one ``xref.json``:

  * **value-flow** -- :class:`TaggedFloat` carries the dotted config-param path of every value
    read from ``DynProps`` (``s.C`` / ``Consts()``). Arithmetic and ``SeqVal`` construction
    propagate the tag, so a param folded to a constant is still traced to the channel(s) it
    reaches. Captures SCANNED and FIXED params. Best-effort: a tag is lost through numpy
    ufuncs, ``math.*`` fallbacks, interp tables and other non-:class:`TaggedFloat` paths --
    a lost tag just means a missing xref entry (the feature stays dormant for it), never a
    wrong one.
  * **global-dep** -- a channel pulse ``SeqVal`` that references a runtime global (head
    ``H_GLOBAL``) records that global id, reported as the key ``g(<id>)``. Captures
    runtime-global dependence (e.g. the 616-EOM "from" frequency) that the value-flow can't
    see. Such keys don't match the config param tree, so they surface as documentation in
    ``channel_to_params`` rather than as a clickable param -- honest about the approximate
    channels (pairs with ``manifest.approximate``).
"""

import re

from seq_val import SeqVal
from seq_val import to_string as _sv_to_string
from seq_val import _BINARY_STR, _FUNC_STR, _FUNC2_STR

# Lazy numpy (matches seq_dump's optional-numpy pattern); used only to recognise numpy scalars.
try:  # pragma: no cover - trivial import guard
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# --------------------------------------------------------------------------- #
# TaggedFloat -- a float that carries the dotted param path(s) it came from.
# It is a `float` subclass, so it passes `is_numeric`, serializes as float64, and
# behaves as a number everywhere; only the arithmetic dunders are overridden, to
# propagate the provenance tag (the union of the operands' tags).
# --------------------------------------------------------------------------- #
def _is_scalar_number(o):
    """A real scalar we can fold and tag -- NOT a bool, NOT a SeqVal, NOT an array."""
    if isinstance(o, bool):
        return False
    if isinstance(o, (int, float)):
        return True
    if _np is not None and isinstance(o, (_np.integer, _np.floating)):
        return True
    return False


def _prov_of(o):
    return o._prov if isinstance(o, TaggedFloat) else frozenset()


def _numstr(v):
    """Compact constant rendering for a formula operand (integral -> no decimals)."""
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return "%g" % f


def _expr_of(o):
    return o._expr if isinstance(o, TaggedFloat) else _numstr(o)


class TaggedFloat(float):
    """A ``float`` tagged with the dotted param paths that produced it AND a human
    formula (``_expr``) built up through arithmetic, so the viewer can show e.g.
    ``Resonance399Freq + BlueMOT.FreqDetuning`` for a value that folded to a constant.
    """

    __slots__ = ("_prov", "_expr")

    def __new__(cls, value, prov=(), expr=None):
        obj = float.__new__(cls, value)
        obj._prov = prov if isinstance(prov, frozenset) else frozenset(prov)
        obj._expr = expr if expr is not None else _numstr(value)
        return obj

    # -- binary arithmetic: fold value, union tags, compose the formula; defer
    #    non-numbers (SeqVal/ndarray) so SeqVal.__r*__ / numpy take over. -- #
    @staticmethod
    def _binop(fn, sym):
        def op(self, other):
            if not _is_scalar_number(other):
                return NotImplemented
            res = fn(float(self), float(other))
            return TaggedFloat(res, self._prov | _prov_of(other),
                               "(%s %s %s)" % (self._expr, sym, _expr_of(other)))
        return op

    @staticmethod
    def _rbinop(fn, sym):
        def op(self, other):
            if not _is_scalar_number(other):
                return NotImplemented
            res = fn(float(other), float(self))
            return TaggedFloat(res, self._prov | _prov_of(other),
                               "(%s %s %s)" % (_expr_of(other), sym, self._expr))
        return op

    def __neg__(self):
        return TaggedFloat(-float(self), self._prov, "-%s" % self._expr)

    def __pos__(self):
        return self

    def __abs__(self):
        return TaggedFloat(abs(float(self)), self._prov, "abs(%s)" % self._expr)

    def __index__(self):
        # Allow a tagged integral value to be used as an index (range/list/np.zeros),
        # which a plain float cannot. Non-integral -> the usual TypeError.
        f = float(self)
        if f.is_integer():
            return int(f)
        raise TypeError("TaggedFloat index must be integral, not %r" % f)


# Generate the forward/reflected operators (kept simple: the common build math).
TaggedFloat.__add__ = TaggedFloat._binop(lambda a, b: a + b, "+")
TaggedFloat.__radd__ = TaggedFloat._rbinop(lambda a, b: a + b, "+")
TaggedFloat.__sub__ = TaggedFloat._binop(lambda a, b: a - b, "-")
TaggedFloat.__rsub__ = TaggedFloat._rbinop(lambda a, b: a - b, "-")
TaggedFloat.__mul__ = TaggedFloat._binop(lambda a, b: a * b, "*")
TaggedFloat.__rmul__ = TaggedFloat._rbinop(lambda a, b: a * b, "*")
TaggedFloat.__truediv__ = TaggedFloat._binop(lambda a, b: a / b, "/")
TaggedFloat.__rtruediv__ = TaggedFloat._rbinop(lambda a, b: a / b, "/")
TaggedFloat.__floordiv__ = TaggedFloat._binop(lambda a, b: a // b, "//")
TaggedFloat.__rfloordiv__ = TaggedFloat._rbinop(lambda a, b: a // b, "//")
TaggedFloat.__mod__ = TaggedFloat._binop(lambda a, b: a % b, "%")
TaggedFloat.__rmod__ = TaggedFloat._rbinop(lambda a, b: a % b, "%")
TaggedFloat.__pow__ = TaggedFloat._binop(lambda a, b: a ** b, "^")
TaggedFloat.__rpow__ = TaggedFloat._rbinop(lambda a, b: a ** b, "^")


def tag_value(value, dotted):
    """Wrap a scalar numeric ``value`` as a :class:`TaggedFloat` carrying ``dotted``.

    A leaf read gets ``_expr = dotted`` (the param name); existing tags are unioned.
    Non-scalars (dicts/structs, ``SeqVal``, bools, strings) pass through untouched --
    only numbers flow to channel outputs.
    """
    if isinstance(value, bool):
        return value                              # logical: not a traced numeric
    if isinstance(value, TaggedFloat):
        return TaggedFloat(float(value), value._prov | {dotted}, value._expr)
    if isinstance(value, (int, float)):
        return TaggedFloat(float(value), {dotted}, dotted)
    if _np is not None and isinstance(value, (_np.integer, _np.floating)):
        return TaggedFloat(float(value), {dotted}, dotted)
    return value


# --------------------------------------------------------------------------- #
# Formula rendering -- turn a pulse's value into a readable derivation with PARAM
# NAMES. A folded constant carries its formula in TaggedFloat._expr; a SeqVal
# (e.g. a ramp) is rendered by mirroring SeqVal.to_string but substituting any
# TaggedFloat leaf with its param-name formula instead of the folded number.
# --------------------------------------------------------------------------- #
def _strip_outer(s):
    """Drop one redundant outer paren pair (``(a + b)`` -> ``a + b``)."""
    if s and len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        for i, c in enumerate(s):
            depth += (c == "(") - (c == ")")
            if depth == 0:
                return s[1:-1] if i == len(s) - 1 else s
    return s


def _render_arg(x):
    if isinstance(x, TaggedFloat):
        return x._expr
    if isinstance(x, SeqVal):
        return _render_seqval(x)
    return _numstr(x) if isinstance(x, (int, float)) else str(x)


def _render_seqval(sv):
    head, args = sv.head, sv.args
    if head in _BINARY_STR:
        return "(%s%s%s)" % (_render_arg(args[0]), _BINARY_STR[head], _render_arg(args[1]))
    if head in _FUNC_STR:
        return "%s(%s)" % (_FUNC_STR[head], _render_arg(args[0]))
    if head in _FUNC2_STR:
        return "%s(%s, %s)" % (_FUNC2_STR[head], _render_arg(args[0]), _render_arg(args[1]))
    if head == SeqVal.OP_NOT:
        return "~%s" % _render_arg(args[0])
    if head == SeqVal.OP_SELECT:
        return "ifelse(%s, %s, %s)" % (_render_arg(args[0]), _render_arg(args[1]),
                                       _render_arg(args[2]))
    if head == SeqVal.OP_IDENTITY:
        return _render_arg(args[0])
    if head == SeqVal.H_GLOBAL:
        return "g(%d)" % int(args[0])
    if head == SeqVal.H_MEASURE:
        return "m(%d)" % int(args[0])
    if head == SeqVal.H_ARG:
        return "arg(%d)" % int(args[0])
    return _sv_to_string(sv)                       # interp/xor/etc.: fall back (folded nums)


# A pulse value-function's args: arg(0) = time t into the step; arg(1) = the channel's
# previous value (what a ramp goes FROM). The tick<->second conversion (tick_per_sec) shows
# up as a redundant (X * N) / N round-trip; collapse it so a ramp reads as a ramp.
_TICK_RT = re.compile(r"\(([\w.]+)\s*\*\s*(\d+)\)\s*/\s*\2(?!\d)")


def _cleanup_formula(s):
    if not s:
        return s
    prev = None
    while prev != s:                              # collapse nested (param * N) / N -> param
        prev = s
        s = _TICK_RT.sub(r"\1", s)
    s = re.sub(r"arg\(0\)\s*/\s*\d+", "t", s)     # arg(0)/<scale> == the seconds-time t
    s = s.replace("arg(1)", "from").replace("arg(0)", "t")
    prev = None
    while prev != s:                              # drop redundant parens around a lone token
        prev = s
        s = re.sub(r"\(([\w.]+)\)", r"\1", s)
    return s


def render_expr(value):
    """Readable derivation formula (param names) for a pulse value, or None."""
    try:
        if isinstance(value, TaggedFloat):
            raw = value._expr
        elif isinstance(value, SeqVal):
            raw = _render_seqval(value)
        else:
            return None
        return _strip_outer(_cleanup_formula(raw))
    except Exception:  # noqa: BLE001 - a formula is a hint; never break the build
        return None


_EXPR_MAX = 240


def _pulse_out(e):
    """Serialise one pulse entry for ``xref.json`` (caps a runaway formula)."""
    d = {"channel": e["channel"], "params": sorted(e["params"])}
    ex = e.get("expr")
    if ex:
        d["expr"] = ex if len(ex) <= _EXPR_MAX else ex[:_EXPR_MAX - 3] + "..."
    return d


# --------------------------------------------------------------------------- #
# Channel-name decoration -- replicate dump_output.decorate_channel_name so xref
# keys match the alias-decorated channel labels shown in the .seq / plot, WITHOUT
# importing dump_output (which pulls tools/ onto sys.path).
# --------------------------------------------------------------------------- #
def _decorate_channel(name, inverse_chn_map):
    if not inverse_chn_map:
        return name
    all_names = inverse_chn_map.get(name, [name])
    additional = [n for n in all_names if n != name]
    if not additional:
        return name
    return "%s (%s)" % (", ".join(additional), name)


# --------------------------------------------------------------------------- #
# ProvenanceSession -- accumulates param<->channel edges during one build.
# --------------------------------------------------------------------------- #
class ProvenanceSession:
    """Collects the param<->channel relation for ONE instrumented sequence build."""

    def __init__(self):
        self._ns = {}                  # id(DynProps) -> dotted-path prefix
        self.param_to_channels = {}    # dotted path -> set(channel)
        self.channel_to_params = {}    # channel     -> set(dotted path)
        self.pulses = {}               # pulse id    -> {"channel": str, "params": set}
        self.param_to_pids = {}        # dotted path -> set(pulse id)
        self.time_regions = {}         # dotted path -> [[t0_ms, t1_ms], ...]  (waits)

    def register(self, dynprops, prefix):
        """Map a ``DynProps`` instance to a path prefix (``""`` = bare config namespace).

        Unregistered ``DynProps`` (e.g. ``Consts()`` instances built mid-sequence) default
        to the bare namespace -- correct, since they wrap the same config tree as ``s.C``.
        """
        if dynprops is not None:
            self._ns[id(dynprops)] = prefix

    # -- DynProps leaf-read hook --------------------------------------------- #
    def wrap(self, dynprops, path, value):
        if not path:
            return value
        prefix = self._ns.get(id(dynprops), "")
        dotted = ".".join(str(p) for p in path)
        return tag_value(value, prefix + dotted if prefix else dotted)

    # -- TimeStep pulse hook ------------------------------------------------- #
    def record_pulse(self, toplevel, cid, pulse_id, raw, resolved):
        try:
            name = toplevel.channel_name(cid)
            name = _decorate_channel(name, getattr(toplevel, "inverse_chn_map", None))
        except Exception:  # noqa: BLE001 - a channel we can't name contributes nothing
            return
        paths = set()
        self._collect(raw, paths)
        self._collect(resolved, paths)
        if not paths:
            return
        chan = self.channel_to_params.setdefault(name, set())
        for p in paths:
            chan.add(p)
            self.param_to_channels.setdefault(p, set()).add(name)
        # Per-pulse (region) provenance: the pulse id == the .seq's per-point pid (verified),
        # so the viewer can map a clicked plot point to exactly THIS segment's params and
        # vice-versa (highlight a param's regions). Only param-bearing pulses are recorded.
        pid = int(pulse_id)
        # Formula (param-named): prefer the value that still carries provenance -- a numeric
        # pulse's resolved value is a plain float (the tag is stripped by _resolve_pulse), so
        # the raw (TaggedFloat) carries the formula; a ramp's resolved value is the SeqVal.
        src = resolved if isinstance(resolved, (TaggedFloat, SeqVal)) else (
            raw if isinstance(raw, (TaggedFloat, SeqVal)) else None)
        entry = {"channel": name, "params": set(paths)}
        expr = render_expr(src) if src is not None else None
        if expr:
            entry["expr"] = expr
        self.pulses[pid] = entry
        for p in paths:
            self.param_to_pids.setdefault(p, set()).add(pid)

    # -- Wait/timing hook: a param-driven wait advances time over [t0, t1] with no
    #    channel output, so it maps to a time-axis band rather than a pulse. -- #
    def record_wait(self, t, t0_ticks, t1_ticks):
        paths = set()
        self._collect(t, paths)
        if not paths:
            return
        t0 = float(t0_ticks) * 1e-9                # ticks -> ms (matches seq_parse's x-axis)
        t1 = float(t1_ticks) * 1e-9
        if t1 < t0:
            t0, t1 = t1, t0
        for p in paths:
            self.time_regions.setdefault(p, []).append([t0, t1])

    def _collect(self, value, paths):
        if isinstance(value, TaggedFloat):
            paths.update(value._prov)
        elif isinstance(value, SeqVal):
            self._walk_seqval(value, paths, set())

    def _walk_seqval(self, sv, paths, seen):
        if id(sv) in seen:
            return
        seen.add(id(sv))
        head = sv.head
        if head == SeqVal.H_GLOBAL:
            paths.add("g(%d)" % int(sv.args[0]))
            return
        if head in (SeqVal.H_MEASURE, SeqVal.H_ARG):
            return                                 # measures/args aren't params
        for a in sv.args:
            if isinstance(a, SeqVal):
                self._walk_seqval(a, paths, seen)
            elif isinstance(a, TaggedFloat):
                paths.update(a._prov)
            # interp tables / plain numbers carry no provenance

    # -- result -------------------------------------------------------------- #
    def result(self):
        """The ``xref.json`` ``by_file`` entry: aggregate maps + per-pulse (region) maps.

        ``pulses`` is keyed by the pulse id as a STRING (JSON object keys are strings; the
        ``.seq``'s per-point ``pid`` is the same integer). ``param_to_pids`` is the inverse
        for fast param->region lookup in the viewer.
        """
        return {
            "param_to_channels": {k: sorted(v)
                                  for k, v in sorted(self.param_to_channels.items())},
            "channel_to_params": {k: sorted(v)
                                  for k, v in sorted(self.channel_to_params.items())},
            "pulses": {str(pid): _pulse_out(e) for pid, e in sorted(self.pulses.items())},
            "param_to_pids": {k: sorted(v)
                              for k, v in sorted(self.param_to_pids.items())},
            "time_regions": {k: v for k, v in sorted(self.time_regions.items())},
        }


# --------------------------------------------------------------------------- #
# The inert module-global session + the two hooks called from lib.
# --------------------------------------------------------------------------- #
_session = None


def on_access(dynprops, path, value):
    """``DynProps`` leaf-read hook: tag the value when a session is active, else passthrough."""
    s = _session
    if s is None:
        return value
    return s.wrap(dynprops, path, value)


def on_pulse(toplevel, cid, pulse_id, raw, resolved):
    """``TimeStep`` pulse hook: record param->channel + per-pulse edges (when a session is on)."""
    s = _session
    if s is None:
        return
    s.record_pulse(toplevel, cid, pulse_id, raw, resolved)


def wait_start(seq):
    """``wait()`` pre-hook: the wait's absolute start tick, or None (inert/unresolvable)."""
    if _session is None:
        return None
    try:
        return seq.cur_seq_time.get_val()
    except Exception:  # noqa: BLE001 - dynamic (global/measure) timing -> skip this wait
        return None


def wait_end(seq, t, t0):
    """``wait()`` post-hook: record the param-driven time region [t0, now] (when a session is on)."""
    if _session is None or t0 is None:
        return
    try:
        t1 = seq.cur_seq_time.get_val()
    except Exception:  # noqa: BLE001
        return
    _session.record_wait(t, t0, t1)


def begin(session):
    global _session
    _session = session
    return session


def end():
    global _session
    _session = None


class capture:
    """Context manager: activate a fresh :class:`ProvenanceSession` for a build.

    ``consts_dp``/``globals_dp`` are the sequence's ``s.C`` / ``s.G`` ``DynProps`` so reads
    on the globals context get a ``G.`` prefix (and so never collide with a config param).
    Yields the session; read :meth:`ProvenanceSession.result` after the build.
    """

    def __init__(self, consts_dp=None, globals_dp=None):
        self.session = ProvenanceSession()
        self.session.register(globals_dp, "G.")
        self.session.register(consts_dp, "")

    def __enter__(self):
        return begin(self.session)

    def __exit__(self, *exc):
        end()
        return False
