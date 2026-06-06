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

from seq_val import SeqVal

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


class TaggedFloat(float):
    """A ``float`` tagged with the set of dotted param paths that produced it."""

    __slots__ = ("_prov",)

    def __new__(cls, value, prov=()):
        obj = float.__new__(cls, value)
        obj._prov = prov if isinstance(prov, frozenset) else frozenset(prov)
        return obj

    # -- binary arithmetic: fold + union tags; defer non-numbers (SeqVal/ndarray). -- #
    @staticmethod
    def _binop(fn):
        def op(self, other):
            if not _is_scalar_number(other):
                return NotImplemented            # let SeqVal.__r*__ / numpy take over
            res = fn(float(self), float(other))
            return TaggedFloat(res, self._prov | _prov_of(other))
        return op

    @staticmethod
    def _rbinop(fn):
        def op(self, other):
            if not _is_scalar_number(other):
                return NotImplemented
            res = fn(float(other), float(self))
            return TaggedFloat(res, self._prov | _prov_of(other))
        return op

    def __neg__(self):
        return TaggedFloat(-float(self), self._prov)

    def __pos__(self):
        return self

    def __abs__(self):
        return TaggedFloat(abs(float(self)), self._prov)

    def __index__(self):
        # Allow a tagged integral value to be used as an index (range/list/np.zeros),
        # which a plain float cannot. Non-integral -> the usual TypeError.
        f = float(self)
        if f.is_integer():
            return int(f)
        raise TypeError("TaggedFloat index must be integral, not %r" % f)


# Generate the forward/reflected operators (kept simple: the common build math).
TaggedFloat.__add__ = TaggedFloat._binop(lambda a, b: a + b)
TaggedFloat.__radd__ = TaggedFloat._rbinop(lambda a, b: a + b)
TaggedFloat.__sub__ = TaggedFloat._binop(lambda a, b: a - b)
TaggedFloat.__rsub__ = TaggedFloat._rbinop(lambda a, b: a - b)
TaggedFloat.__mul__ = TaggedFloat._binop(lambda a, b: a * b)
TaggedFloat.__rmul__ = TaggedFloat._rbinop(lambda a, b: a * b)
TaggedFloat.__truediv__ = TaggedFloat._binop(lambda a, b: a / b)
TaggedFloat.__rtruediv__ = TaggedFloat._rbinop(lambda a, b: a / b)
TaggedFloat.__floordiv__ = TaggedFloat._binop(lambda a, b: a // b)
TaggedFloat.__rfloordiv__ = TaggedFloat._rbinop(lambda a, b: a // b)
TaggedFloat.__mod__ = TaggedFloat._binop(lambda a, b: a % b)
TaggedFloat.__rmod__ = TaggedFloat._rbinop(lambda a, b: a % b)
TaggedFloat.__pow__ = TaggedFloat._binop(lambda a, b: a ** b)
TaggedFloat.__rpow__ = TaggedFloat._rbinop(lambda a, b: a ** b)


def tag_value(value, dotted):
    """Wrap a scalar numeric ``value`` as a :class:`TaggedFloat` carrying ``dotted``.

    Existing tags are preserved (unioned). Non-scalars (dicts/structs, ``SeqVal``,
    bools, strings) pass through untouched -- only numbers flow to channel outputs.
    """
    if isinstance(value, bool):
        return value                              # logical: not a traced numeric
    if isinstance(value, TaggedFloat):
        return TaggedFloat(float(value), value._prov | {dotted})
    if isinstance(value, (int, float)):
        return TaggedFloat(float(value), {dotted})
    if _np is not None and isinstance(value, (_np.integer, _np.floating)):
        return TaggedFloat(float(value), {dotted})
    return value


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
    def record_pulse(self, toplevel, cid, raw, resolved):
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
        """The two sorted maps in the ``xref.json`` ``by_file`` entry shape."""
        return {
            "param_to_channels": {k: sorted(v)
                                  for k, v in sorted(self.param_to_channels.items())},
            "channel_to_params": {k: sorted(v)
                                  for k, v in sorted(self.channel_to_params.items())},
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


def on_pulse(toplevel, cid, raw, resolved):
    """``TimeStep`` pulse hook: record param->channel edges when a session is active."""
    s = _session
    if s is None:
        return
    s.record_pulse(toplevel, cid, raw, resolved)


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
