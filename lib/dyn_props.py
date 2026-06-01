"""dyn_props.py -- DynProps: nested struct with default-value fallback + access tracking.

Faithful transliteration of ``matlab_new/lib/DynProps.m`` (handle class). Backs ``s.C``,
``Consts()``, and ``s.G``. Owns the value store and the access tracker; ``SubProps``
(sub_props.py) is a thin path-carrying proxy that delegates every operation here.

Semantics reproduced from ``DynProps.m`` / ``TestDynProps.m``:
  * Reading a name whose value is a (scalar) struct returns a SubProps and does NOT mark
    accessed; reading a non-struct value returns the value AND marks accessed.
  * Parens default ``c.X(default)`` resolves to a value, marks accessed, and PERSISTS the
    default into the store (struct defaults MERGE -- existing/leftmost wins, NaN-as-missing).
  * Brace default ``c.X{default}`` (Python: ``c.X[default]``) returns a SubProps proxy,
    does NOT mark, but STILL persists the default.
  * ``dp()`` whole-resolve returns the NaN-stripped struct and collapses accessed to True.
  * NaN-as-missing everywhere; a NaN default and an empty default on a missing field error.
  * SubProps is a live handle (mutations write through to the shared store).

Byte-equality fixes (PYTHON_FRONTEND_PLAN.md Phase 3 audit):
  #1 (eager-resolve proxy default args) -- _make_default resolves SubProps/DynProps args.
  #2 (deepcopy nested store) -- __init__ deep-copies the input so default-writes never
     mutate a shared tree (e.g. SeqConfig.consts behind Consts()).
"""

import copy

from sub_props import SubProps


def _is_missing(v):
    # MATLAB isnanobj: a numeric scalar NaN is "missing"; non-numeric never is.
    # np.float64 subclasses float, so this also catches numpy NaN scalars. NaN != NaN.
    return isinstance(v, float) and v != v


def _remove_nanfields(v):
    # Recursively drop NaN-valued keys from a (scalar) struct; values pass through.
    # Builds new dicts (so the returned tree is independent of the live store).
    if not isinstance(v, dict):
        return v
    out = {}
    for k, sv in v.items():
        if _is_missing(sv):
            continue
        out[k] = _remove_nanfields(sv)
    return out


def _merge_struct(a, b):
    # Return (merged, changed). a's values win; keys missing/NaN in a are filled from b;
    # two scalar-struct values recurse. Mirrors DynProps.merge_struct (undefnan=true).
    out = dict(a)
    changed = False
    for k, bv in b.items():
        if k not in out or _is_missing(out[k]):
            out[k] = bv
            changed = True
        else:
            av = out[k]
            if isinstance(av, dict) and isinstance(bv, dict):
                merged, ch = _merge_struct(av, bv)
                if ch:
                    out[k] = merged
                    changed = True
            # else: keep a's value
    return out, changed


def _construct_struct(args):
    # MATLAB construct_struct(varargin): fold args left-to-right into one struct; a dict
    # arg merges whole, a non-dict arg consumes the NEXT arg as its value (name, value).
    res = {}
    i = 0
    n = len(args)
    while i < n:
        v = args[i]
        if not isinstance(v, dict):
            name = v
            i += 1
            v = {name: args[i]}
        res, _ = _merge_struct(res, v)
        i += 1
    return res


def _mark_accessed(acc, path):
    # empty path -> True; an already-True node stays True; else descend, creating sub-dicts.
    if not path:
        return True
    if acc is True:
        return True
    if not isinstance(acc, dict):
        acc = {}
    name = path[0]
    acc[name] = _mark_accessed(acc.get(name, {}), path[1:])
    return acc


def _coerce_arg(x):
    # Fix #1: a default ARGUMENT that is itself a SubProps/DynProps (e.g. Consts().A.B.C)
    # must resolve to its underlying value before being stored/merged -- MATLAB evaluates
    # the argument eagerly to a number; leaving a proxy would persist a proxy and corrupt
    # the store + access tree.
    if isinstance(x, SubProps):
        return x._value()
    if isinstance(x, DynProps):
        return x()
    return x


class DynProps:
    def __init__(self, store=None):
        # Fix #2: deep-copy so default-writes never mutate a shared input tree.
        self._store = copy.deepcopy(store) if store else {}
        self._accessed = {}

    # ----------------------------------------------------------------------- #
    # attribute / call / brace entry points
    # ----------------------------------------------------------------------- #
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return SubProps(self, (name,))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, (dict, list)):
            value = copy.deepcopy(value)        # MATLAB struct/array assignment copies
        self._store[name] = value

    def __call__(self, *args):
        return self._resolve((), args, is_brace=False)

    def __getitem__(self, key):
        args = key if isinstance(key, tuple) else (key,)
        return self._resolve((), args, is_brace=True)

    # ----------------------------------------------------------------------- #
    # store walking
    # ----------------------------------------------------------------------- #
    def _getpath(self, path):
        v = self._store
        for name in path:
            if not isinstance(v, dict) or name not in v:
                return None, False
            v = v[name]
            if _is_missing(v):
                return None, False
        return v, True

    def _setpath(self, path, value):
        if not path:
            self._store = value
            return
        d = self._store
        for name in path[:-1]:
            nxt = d.get(name)
            if not isinstance(nxt, dict):
                nxt = {}
                d[name] = nxt
            d = nxt
        d[path[-1]] = value

    # ----------------------------------------------------------------------- #
    # the resolver -- DynProps.subsref for a <path> followed by ()/{}
    # ----------------------------------------------------------------------- #
    def _resolve(self, path, args, is_brace):
        cur, found = self._getpath(path)
        if found:
            if args:
                default = self._make_default(args)
                if isinstance(cur, dict) and isinstance(default, dict):
                    merged, changed = _merge_struct(cur, default)
                    if changed:
                        self._setpath(path, merged)
                        cur = merged
                # scalar leaf: default ignored
            if is_brace:
                return SubProps(self, path)            # proxy, no mark
            self._mark(path)
            return _remove_nanfields(cur)
        # missing: a default is REQUIRED
        default = self._make_default(args)
        self._setpath(path, default)                  # persist
        if is_brace:
            return SubProps(self, path)               # proxy, no mark (default persisted)
        self._mark(path)
        return _remove_nanfields(default)

    @staticmethod
    def _make_default(args):
        if len(args) == 0:
            raise KeyError("No default value given")
        if len(args) == 1:
            d = _coerce_arg(args[0])                  # fix #1
        else:
            d = _construct_struct(tuple(_coerce_arg(a) for a in args))
        if _is_missing(d):
            raise ValueError("Default value cannot be NaN.")
        return d

    def _mark(self, path):
        self._accessed = _mark_accessed(self._accessed, path)

    # ----------------------------------------------------------------------- #
    # access tracker
    # ----------------------------------------------------------------------- #
    def get_accessed(self):
        return self._accessed

    def clear_accessed(self):
        self._accessed = {}

    # ----------------------------------------------------------------------- #
    # NaN-aware introspection (no access marking)
    # ----------------------------------------------------------------------- #
    def isfield(self, name):
        return self._isfield_at((), name)

    def fieldnames(self):
        return self._fieldnames_at(())

    def _isfield_at(self, path, name):
        v, found = self._getpath(path)
        if not found or not isinstance(v, dict):
            return False
        return name in v and not _is_missing(v[name])

    def _fieldnames_at(self, path):
        v, found = self._getpath(path)
        if not found or not isinstance(v, dict):
            return []
        return [k for k, sv in v.items() if not _is_missing(sv)]

    def getfields(self, *args):
        # Selective extract: optional leading dict = base; each name -> remove_nanfields of
        # the stored value (error if missing/NaN), marking that name only when accessed is
        # still a dict (not collapsed to True). Mirrors DynProps.getfields.
        res = {}
        args = list(args)
        if args and isinstance(args[0], dict):
            res = dict(args[0])
            args = args[1:]
        for name in args:
            if name not in self._store or _is_missing(self._store[name]):
                raise KeyError(name)
            if isinstance(self._accessed, dict):
                self._accessed[name] = True
            res[name] = _remove_nanfields(self._store[name])
        return res

    def to_struct(self):
        # Non-aliasing escape hatch (the documented copy): deep copy of the whole store.
        return copy.deepcopy(self())
