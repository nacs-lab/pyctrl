"""sub_props.py -- SubProps: a lazy, handle-class proxy into a parent DynProps.

Faithful transliteration of ``matlab_new/lib/SubProps.m``. A SubProps holds the ROOT
``DynProps`` and an ABSOLUTE dotted path; every read/write re-enters the root with that
path prefix (MATLAB ``subsref(self.parent, [self.path, S])``). It NEVER snapshots a
value at creation, so it behaves as a live alias: ``a = c.X; a.f = 1`` mutates ``c.X.f``.

The reconciliation with MATLAB (which resolves a whole ``g.A.B.C(default)`` chain in one
``subsref`` and so knows a trailing call follows): here ``__getattr__`` only EXTENDS the
path (no store touch, no access mark); resolution happens exclusively in ``__call__``
(parens default -> value, marks accessed, persists default), ``__getitem__`` (brace
default -> proxy, no mark, still persists), and the coercion/operator dunders (use as a
value -> resolve with no default, marks). All the real logic lives on the root DynProps;
this class is a thin path-carrying delegator.
"""

import copy
import operator


class SubProps:
    def __init__(self, root, path):
        # Bypass our own __setattr__ for the two real attributes.
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_path", tuple(path))

    # -- lazy navigation: extend the path, never resolve / never mark ---------- #
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return SubProps(self._root, self._path + (name,))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, (dict, list)):
            value = copy.deepcopy(value)        # MATLAB struct/array assignment copies
        self._root._setpath(self._path + (name,), value)

    # -- resolution ------------------------------------------------------------ #
    def __call__(self, *args):
        # parens default: resolve to a value, mark accessed, persist default
        return self._root._resolve(self._path, args, is_brace=False)

    def __getitem__(self, key):
        # brace default: return a SubProps proxy (no mark), still persist default
        args = key if isinstance(key, tuple) else (key,)
        return self._root._resolve(self._path, args, is_brace=True)

    # -- NaN-aware introspection (no access marking) --------------------------- #
    def isfield(self, name):
        return self._root._isfield_at(self._path, name)

    def fieldnames(self):
        return self._root._fieldnames_at(self._path)

    def to_struct(self):
        # Non-aliasing escape hatch: a deep copy of the resolved subtree.
        return copy.deepcopy(self._root._resolve(self._path, (), is_brace=False))

    # -- use-as-value: resolve the path with NO default (raises if unset), mark - #
    def _value(self):
        return self._root._resolve(self._path, (), is_brace=False)

    __hash__ = object.__hash__   # keep identity hash (do NOT resolve in __hash__)

    def __eq__(self, other):
        return self._value() == other

    def __ne__(self, other):
        return self._value() != other

    def __lt__(self, other):
        return self._value() < other

    def __le__(self, other):
        return self._value() <= other

    def __gt__(self, other):
        return self._value() > other

    def __ge__(self, other):
        return self._value() >= other

    def __float__(self):
        return float(self._value())

    def __int__(self):
        return int(self._value())

    def __index__(self):
        return operator.index(self._value())

    def __bool__(self):
        return bool(self._value())

    def __add__(self, o):
        return self._value() + o

    def __radd__(self, o):
        return o + self._value()

    def __sub__(self, o):
        return self._value() - o

    def __rsub__(self, o):
        return o - self._value()

    def __mul__(self, o):
        return self._value() * o

    def __rmul__(self, o):
        return o * self._value()

    def __truediv__(self, o):
        return self._value() / o

    def __rtruediv__(self, o):
        return o / self._value()

    def __neg__(self):
        return -self._value()

    def __pos__(self):
        return +self._value()

    def __abs__(self):
        return abs(self._value())

    def __pow__(self, o):
        return self._value() ** o

    def __rpow__(self, o):
        return o ** self._value()

    def __repr__(self):
        return "SubProps[.%s]" % ".".join(str(p) for p in self._path)
