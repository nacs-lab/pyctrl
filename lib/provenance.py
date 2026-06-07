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

import os
import re
import sys

from seq_val import SeqVal
from seq_val import to_string as _sv_to_string
from seq_val import _BINARY_STR, _FUNC_STR, _FUNC2_STR
from seq_time import SeqTime

import math
try:
    from mat_utils import mat_round
except Exception:  # noqa: BLE001 - fallback (round half away from zero)
    def mat_round(x):
        return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)

# Numeric evaluation of a (possibly SeqVal) timing expression, with runtime globals
# substituted. Sub-sequence OFFSETS are frequently global-dependent (e.g. the 616-EOM
# ramp duration `round((abs((F - g(0))*2e-8*3) + 0.02)*1e12)`), so absolute placement of
# a wait region needs g(id) values (from the run's captured globals.json). Returns None
# on anything we can't resolve to a number (measure/arg heads, unknown op, missing
# global) -- the region is then skipped rather than mis-placed.
_SV_BINOP = {
    SeqVal.OP_ADD: lambda a, b: a + b,
    SeqVal.OP_SUB: lambda a, b: a - b,
    SeqVal.OP_MUL: lambda a, b: a * b,
    SeqVal.OP_DIV: lambda a, b: (a / b) if b else None,
    SeqVal.OP_POW: lambda a, b: a ** b,
    SeqVal.OP_HYPOT: math.hypot,
    SeqVal.OP_ATAN2: math.atan2,
    SeqVal.OP_MAX: max,
    SeqVal.OP_MIN: min,
    SeqVal.OP_MOD: math.fmod,
}
_SV_UNOP = {
    SeqVal.OP_ABS: abs, SeqVal.OP_CEIL: math.ceil, SeqVal.OP_FLOOR: math.floor,
    SeqVal.OP_RINT: lambda x: float(mat_round(x)), SeqVal.OP_SQRT: math.sqrt,
    SeqVal.OP_EXP: math.exp, SeqVal.OP_LOG: math.log, SeqVal.OP_LOG2: math.log2,
    SeqVal.OP_LOG10: math.log10, SeqVal.OP_SIN: math.sin, SeqVal.OP_COS: math.cos,
    SeqVal.OP_TAN: math.tan, SeqVal.OP_IDENTITY: lambda x: x,
}


def _eval_num(v, gmap):
    """Evaluate ``v`` (number / TaggedFloat / SeqVal) to a float, or None if unresolvable."""
    if isinstance(v, SeqVal):
        h, a = v.head, v.args
        if h == SeqVal.H_GLOBAL:
            try:
                gid = int(a[0])
            except Exception:  # noqa: BLE001
                return None
            return None if not gmap else gmap.get(gid)
        if h in (SeqVal.H_MEASURE, SeqVal.H_ARG):
            return None                        # runtime-only, no static value
        ev = [_eval_num(x, gmap) for x in a]
        if any(e is None for e in ev):
            return None
        try:
            if h in _SV_BINOP and len(ev) >= 2:
                return _SV_BINOP[h](ev[0], ev[1])
            if h in _SV_UNOP and len(ev) >= 1:
                return _SV_UNOP[h](ev[0])
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return None                            # unsupported op -> skip
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _abs_ticks(st, gmap=None):
    """Absolute time (in ticks) of a ``SeqTime`` node, following sub-sequence offsets up
    to the root -- the numeric analogue of ``SeqTime.to_string``'s parent/``t_offset``
    walk. ``SeqTime.get_val`` only sums the node's OWN basic-sequence frame, so a wait
    built inside a sub-sequence comes out in that sub-sequence's LOCAL time; we add each
    owning sub-sequence's ``t_offset`` (which lives in its parent's frame, and is often a
    global-dependent SeqVal) until the root. ``None`` if anything can't be resolved.
    """
    total = 0.0
    node = st
    for _ in range(4096):                      # depth guard (pathological cycles)
        if not isinstance(node, SeqTime):
            return None
        try:
            local = _eval_num(node.get_val(), gmap)   # THIS frame's total, globals applied
        except Exception:                      # noqa: BLE001
            return None
        if local is None:
            return None
        total += float(local)
        seq = node.seq
        if seq is None or getattr(seq, "parent", None) is None:
            return total                       # reached the root frame -> absolute
        node = getattr(seq, "t_offset", None)  # ascend into the parent's frame
    return None

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
# Source backtrace capture (B3). Build it HERE, in the offline provenance pass --
# NOT during the live .seq emit. The pulse hook below already fires per pulse with the
# verified .seq pulse_id (that's how click->params works), so we hang the source location
# off it: zero runtime cost (this pass is the engine-free, on-view/after xref build) and no
# reliance on the live obj_counter (which is why the literal "capture in next_obj_id" plan
# had a pulse_id<->obj_id unknown -- moot here). Mirrors MATLAB's debug=1 backtrace, but
# computed offline instead of on the hot path.
# --------------------------------------------------------------------------- #
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))                 # pyctrl/lib
_TOOLS_DIR = os.path.join(os.path.dirname(_LIB_DIR), "tools")         # pyctrl/tools


def _is_framework_file(path):
    """True if ``path`` is pyctrl framework (``lib/``) or the offline harness (``tools/``).
    Those frames bracket the EXPERIMENT call chain (``YbSeqs``/``YbSteps`` ``.add``/``.add_step``):
    the inner ones (the pulse machinery) are skipped, the outer ones (the rebuild driver) stop
    the walk -- so a captured backtrace is exactly the user's experiment frames."""
    try:
        ap = os.path.abspath(path)
    except Exception:  # noqa: BLE001
        return False
    return ap.startswith(_LIB_DIR + os.sep) or ap.startswith(_TOOLS_DIR + os.sep)


def _capture_pulse_frames(max_frames=24):
    """Innermost-first ``[(file, name, line)]`` of the EXPERIMENT call chain that built the
    current pulse. Walks ``f_back`` directly (no source-line lookup -> cheap), skipping the
    leading framework frames (so the leaf is the user's ``.add``/``.add_step`` line) and
    stopping at the first framework frame AFTER the experiment block (drops the rebuild
    driver / runpy noise). ``[]`` if no experiment frame is in scope."""
    frames = []
    try:
        f = sys._getframe(2)                # skip _capture_pulse_frames + record_pulse
    except (ValueError, AttributeError):    # pragma: no cover - no stack introspection
        return frames
    started = False
    while f is not None and len(frames) < max_frames:
        co = f.f_code
        if _is_framework_file(co.co_filename):
            if started:
                break                       # past the experiment block -> stop
        else:
            started = True
            frames.append((co.co_filename, co.co_name, f.f_lineno))
        f = f.f_back
    return frames


# --------------------------------------------------------------------------- #
# ProvenanceSession -- accumulates param<->channel edges during one build.
# --------------------------------------------------------------------------- #
class ProvenanceSession:
    """Collects the param<->channel relation for ONE instrumented sequence build."""

    def __init__(self, capture_bt=True):
        self._ns = {}                  # id(DynProps) -> dotted-path prefix
        self.param_to_channels = {}    # dotted path -> set(channel)
        self.channel_to_params = {}    # channel     -> set(dotted path)
        self.pulses = {}               # pulse id    -> {"channel": str, "params": set}
        self.param_to_pids = {}        # dotted path -> set(pulse id)
        # Source backtrace per pulse id (B3): pid -> [(file, name, line), ...] innermost-first.
        # Captured in record_pulse (offline build only); emitted as result()['backtraces'].
        self.capture_bt = bool(capture_bt)
        self.pulse_bt = {}
        self.time_regions = {}         # dotted path -> [[t0_ms, t1_ms], ...]  (waits)
        # Waits are captured as SeqTime NODES and resolved to ABSOLUTE ms lazily in
        # result() -- a wait's start/end can only be placed once the whole tree is
        # built (its sub-sequence t_offset is known). Resolving at wait-time gave
        # sub-sequence-LOCAL times (bug: GreenMOT.CoolDown.HoldTime landed at ~65ms
        # instead of ~316ms). [(set(paths), t0_node, t1_node), ...]
        self._wait_pending = []
        # Top-level step boundaries: a labeled [start, end] per direct child of the root
        # (the experiment phases: InitStep/BlueMOTStep/GreenMOTStep/...). Same deferred
        # absolute-time resolution as waits. [(label, start_node, end_node), ...] -> steps.
        self._step_pending = []
        self.steps = []                # [{"label": str, "t0": ms, "t1": ms}, ...]
        # Count of wait/step time entries that couldn't be placed for lack of the run's
        # globals (set by _resolve_waits/_resolve_steps) -> result()['pending_globals'].
        self._skipped_waits = 0
        self._skipped_steps = 0

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
        # Source backtrace (B3): capture FIRST, before the param early-returns below, so even
        # a param-less pulse (a constant TTL/voltage set) still records where it was added.
        if self.capture_bt:
            frames = _capture_pulse_frames()
            if frames:
                self.pulse_bt[int(pulse_id)] = frames
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
    def record_wait(self, t, t0_node, t1_node):
        # Defer: stash the param paths + the start/end SeqTime NODES. They're resolved
        # to absolute ms in result(), once the full tree (and sub-sequence offsets) exist.
        paths = set()
        self._collect(t, paths)
        if not paths:
            return
        self._wait_pending.append((paths, t0_node, t1_node))

    def record_step(self, label, t0_node, t1_node):
        # Defer: stash the label + start/end SeqTime NODES; resolved to absolute ms in
        # result() (same reason as waits -- offsets only exist once the tree is built).
        self._step_pending.append((label, t0_node, t1_node))

    def _resolve_steps(self, globals_map=None):
        """Resolve the deferred top-level step nodes to absolute-time ms spans. Idempotent."""
        self.steps = []
        self._skipped_steps = 0
        for label, n0, n1 in self._step_pending:
            a0 = _abs_ticks(n0, globals_map)
            a1 = _abs_ticks(n1, globals_map)
            if a0 is None or a1 is None:
                self._skipped_steps += 1               # unplaceable (global-dependent / dynamic)
                continue
            t0 = a0 * 1e-9
            t1 = a1 * 1e-9
            if t1 < t0:
                t0, t1 = t1, t0
            self.steps.append({"label": label, "t0": t0, "t1": t1})
        self.steps.sort(key=lambda s: (s["t0"], s["t1"]))

    def _resolve_waits(self, globals_map=None):
        """Resolve the deferred wait nodes to absolute-time ms bands. Idempotent.

        ``globals_map`` (``{global_id: value}``) supplies the run's captured globals so
        global-dependent sub-sequence offsets resolve to a number; without it, such
        regions are skipped (no global -> can't place).
        """
        self.time_regions = {}
        self._skipped_waits = 0
        for paths, n0, n1 in self._wait_pending:
            a0 = _abs_ticks(n0, globals_map)
            a1 = _abs_ticks(n1, globals_map)
            if a0 is None or a1 is None:          # dynamic / unplaced -> skip the band
                self._skipped_waits += 1
                continue
            t0 = a0 * 1e-9                          # ticks -> ms (matches seq_parse's x-axis)
            t1 = a1 * 1e-9
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
    def result(self, globals_map=None):
        """The ``xref.json`` ``by_file`` entry: aggregate maps + per-pulse (region) maps.

        ``globals_map`` (optional ``{global_id: value}``) lets wait time_regions resolve
        global-dependent sub-sequence offsets to absolute time.

        ``pulses`` is keyed by the pulse id as a STRING (JSON object keys are strings; the
        ``.seq``'s per-point ``pid`` is the same integer). ``param_to_pids`` is the inverse
        for fast param->region lookup in the viewer.
        """
        self._resolve_waits(globals_map)            # local SeqTime nodes -> absolute ms
        self._resolve_steps(globals_map)            # top-level step boundaries -> absolute ms
        # How many step/wait time entries are still unplaced for lack of the run's globals.
        # Only meaningful when NO globals were supplied (a live-run build): with globals, a
        # remaining skip is a genuinely dynamic time (measure/arg), NOT "pending globals". The
        # viewer uses this to (a) show "N band(s) pending globals" and (b) trigger one rebuild
        # once globals.json lands (dashboard._maybe_autobuild_xref).
        pending_globals = 0 if globals_map else (self._skipped_waits + self._skipped_steps)
        return {
            "param_to_channels": {k: sorted(v)
                                  for k, v in sorted(self.param_to_channels.items())},
            "channel_to_params": {k: sorted(v)
                                  for k, v in sorted(self.channel_to_params.items())},
            "pulses": {str(pid): _pulse_out(e) for pid, e in sorted(self.pulses.items())},
            "param_to_pids": {k: sorted(v)
                              for k, v in sorted(self.param_to_pids.items())},
            "time_regions": {k: v for k, v in sorted(self.time_regions.items())},
            "steps": self.steps,
            # Source backtrace per pulse (B3): {pid(str): [{file, name, line}, ...]}, innermost
            # (the user's .add/.add_step) first. Keyed by the same .seq pid as `pulses`, so the
            # viewer maps a clicked point -> its source location. Denormalized (JSON, not the
            # compact binary .seq backtrace block).
            "backtraces": {
                str(pid): [{"file": fn, "name": nm, "line": ln} for (fn, nm, ln) in frames]
                for pid, frames in sorted(self.pulse_bt.items())},
            # # of step/wait bands still waiting on the run's globals (0 once globals applied).
            "pending_globals": pending_globals,
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
    """``wait()`` pre-hook: the wait's start SeqTime NODE, or None (inert).

    Returns the node (NOT its value) so result() can resolve it to ABSOLUTE time once
    the full tree exists -- ``get_val()`` here would give sub-sequence-local time.
    """
    if _session is None:
        return None
    try:
        return seq.cur_seq_time
    except Exception:  # noqa: BLE001
        return None


def wait_end(seq, t, t0):
    """``wait()`` post-hook: record the param-driven time region [t0, now] (when a session is on)."""
    if _session is None or t0 is None:
        return
    try:
        t1 = seq.cur_seq_time
    except Exception:  # noqa: BLE001
        return
    _session.record_wait(t, t0, t1)


def on_step(parent, cb, start_node, step):
    """``add_custom_step`` post-hook: record a TOP-LEVEL step's [start, end] + label.

    INERT unless a session is active. Only direct children of the root are recorded (the
    experiment phases -- InitStep/BlueMOTStep/...); nested steps would clutter the ruler.
    The label is the step callback's ``__name__``. start/end are SeqTime NODES, resolved to
    absolute ms in result(). Floating (nan) starts and lambdas are skipped.
    """
    if _session is None:
        return
    try:
        if getattr(parent, "parent", None) is not None:
            return                                  # not a top-level step
        end_node = step.cur_seq_time
        if not isinstance(start_node, SeqTime) or not isinstance(end_node, SeqTime):
            return                                  # floating / unplaced -> skip
        label = getattr(cb, "__name__", None)
        if not label or label.startswith("<"):      # drop lambdas / unnamed
            label = None
    except Exception:  # noqa: BLE001
        return
    _session.record_step(label, start_node, end_node)


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

    def __init__(self, consts_dp=None, globals_dp=None, capture_bt=True):
        self.session = ProvenanceSession(capture_bt=capture_bt)
        self.session.register(globals_dp, "G.")
        self.session.register(consts_dp, "")

    def __enter__(self):
        return begin(self.session)

    def __exit__(self, *exc):
        end()
        return False
