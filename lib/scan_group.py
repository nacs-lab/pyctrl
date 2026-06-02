"""scan_group.py -- ScanGroup: an ordered set of N-dimensional parameter scans.

Faithful transliteration of ``matlab_new/lib/ScanGroup.m`` (a ``handle`` class). This file
is **Phase-4 W2**: the core data model + the authoring DSL only. Materialization/queries
(``getseq``/``getseq_in_scan``/``nseq``/``scansize``/``scandim``/``axisnum``/``get_*``,
the ``getfullscan`` base-merge) are W3; ``usevar`` + ``getseq_*_with_var`` are W8;
``load``/``cat_scans``/``horzcat``/``toscan`` are W7.

Data model (mirrors ScanGroup.m's private properties verbatim):
  * ``_scans``  : list of Scan = ``{'baseidx': int, 'params': dict, 'vars': [Scan1D]}``
  * ``_base``   : the fallback Scan (no ``baseidx``) reachable via ``grp()`` (idx 0)
  * Scan1D      : ``{'size': int, 'params': dict}`` (one scan dimension; size 0 = dummy)
  * ``_runparam``: a DynProps (run parameters; one per group)
  * ``_use_var_base`` / ``_use_var_scans`` : tri-state use-global trees (data kept here so
    ``dump`` matches; the ``usevar`` mutator + expansion are W8)
  * caches: ``_scanscache`` (per-scan dirty flag, parallels ``_scans``), ``_use_var_cache``,
    ``_new_empty_called``

THE KEY SEMANTIC (decision #1 of the Phase-4 audit): a parameter is *swept* iff it lives
under ``scan.vars[dim].params`` and *fixed* iff under ``scan.params`` -- decided at
``.scan()``-call time in ``_addscan``/``_addparam``. No visited tree, no listeners.

Deep-copy boundaries (decision #3 -- Python dicts alias where MATLAB structs copy by
value): every stored value is copied on the way in (``_copyval``), whole-scan assignment
copies the source scan, and ``dump`` returns an independent copy.
"""

import copy

from dyn_props import DynProps
from scan_param import ScanParam

try:                                   # numpy is the array type from Phase 1 onward;
    import numpy as _np                # keep the import soft so pure-stdlib runs still work.
except Exception:                      # pragma: no cover
    _np = None


# -- DEF_* constants as fresh-instance factories (never share a mutable default) -- #
def _def_scan():
    return {"baseidx": 0, "params": {}, "vars": []}


def _def_vars():
    return {"size": 0, "params": {}}


def _def_scancache():
    return {"dirty": True, "params": {}, "vars": {}}


def _def_use_var():
    return {"def": 0, "dims": [], "field": {}}


class ScanGroup:
    def __init__(self):
        self._scans = [_def_scan()]
        self._base = {"params": {}, "vars": []}
        self._runparam = DynProps()
        self._use_var_base = _def_use_var()
        self._use_var_scans = []                # EMPTY_USE_VAR (0x0 struct array)
        self._scanscache = [_def_scancache()]
        self._use_var_cache = []
        self._new_empty_called = False

    # ======================================================================= #
    # public API used by experiments / save-load (W2 subset)
    # ======================================================================= #
    def __call__(self, *args):
        # grp() -> fallback (idx 0); grp(n) -> the n-th scan (n >= 1); grp(:) -> fallback.
        if len(args) == 0:
            return ScanParam(self, 0)
        if len(args) > 1:
            raise ValueError("Too many scan index")
        idx = args[0]
        if isinstance(idx, slice) or idx == ":":
            return ScanParam(self, 0)
        if not _is_pos_int(idx):
            # Don't allow implicitly addressing the fallback with 0 (or NaN/negatives).
            raise ValueError("Scan index must be positive")
        return ScanParam(self, int(idx))

    @property
    def end(self):
        # MATLAB `end` inside grp(end...) -> number of scans.
        return len(self._scans)

    def runp(self):
        return self._runparam

    def groupsize(self):
        return len(self._scans)

    def new_empty(self):
        if (not self._new_empty_called and len(self._scans) == 1
                and self._scans[0] == _def_scan()):
            self._new_empty_called = True
            return 1
        idx = len(self._scans) + 1
        self._scans.append(_def_scan())
        self._scanscache.append(_def_scancache())
        return idx

    def setbase(self, idx, base):
        # Always ensures the scan being set exists + its cache entry is initialized.
        if not _is_nonneg_int(base):
            raise ValueError("Base index must be non-negative integer.")
        base = int(base)
        if base > len(self._scans):
            raise ValueError("Cannot set base to non-existing scan")
        if idx > len(self._scans):
            self._grow_scans(idx)
            self._scans[idx - 1]["baseidx"] = base
            return
        # Fast pass to avoid invalidating anything.
        if self._getbaseidx(idx) == base:
            return
        self._use_var_cache = []
        if base == 0:
            # Back to default -- no possibility of a new loop.
            self._scans[idx - 1]["baseidx"] = base
            self._scanscache[idx - 1]["dirty"] = True
            return
        newbase = base
        # Loop detection.
        visited = [False] * len(self._scans)
        visited[idx - 1] = True
        b = base
        while True:
            if visited[b - 1]:
                raise ValueError("Base index loop detected.")
            visited[b - 1] = True
            b = self._getbaseidx(b)
            if b == 0:
                break
        self._scans[idx - 1]["baseidx"] = newbase
        self._scanscache[idx - 1]["dirty"] = True

    def dump(self):
        # Low-level, class-free representation (the reverse of load, W7). Returns an
        # independent copy (MATLAB struct assignment copies by value).
        return {
            "version": 1,
            "scans": copy.deepcopy(self._scans),
            "base": copy.deepcopy(self._base),
            "runparam": copy.deepcopy(self._runparam()),
            "use_var_base": copy.deepcopy(self._use_var_base),
            "use_var_scans": copy.deepcopy(self._use_var_scans),
        }

    # ======================================================================= #
    # ScanParam-facing implementation (MATLAB methods(Access=?ScanParam))
    # ======================================================================= #
    def _param_size(self, idx, dim):
        if idx == 0:
            scan = self._base
        elif idx > len(self._scans):
            return 1
        else:
            scan = self._scans[idx - 1]
        vars_ = scan["vars"]
        if len(vars_) < dim:
            return 1
        sz = vars_[dim - 1]["size"]
        if sz <= 0:
            sz = 1
        return sz

    def _addparam(self, idx, path, val):
        self._check_noconflict(idx, path, 0)
        val = _copyval(val)
        if idx == 0:
            self._check_param_overwrite(self._base["params"], path, val)
            self._setfield(self._base["params"], path, val)
            self._set_dirty_all()
        else:
            self._check_param_overwrite(self._scans[idx - 1]["params"], path, val)
            self._setfield(self._scans[idx - 1]["params"], path, val)
            self._scanscache[idx - 1]["dirty"] = True
            self._use_var_cache = []

    def _addscan(self, idx, path, dim, vals):
        if not _isarray(vals):
            # Scalar / single value / string -> decays to a fixed parameter.
            self._addparam(idx, path, vals)
            return
        if not _is_pos_int(dim):
            raise ValueError("Scan dimension must be positive integer.")
        dim = int(dim)
        self._check_noconflict(idx, path, dim)
        nvals = _numel(vals)
        vals = _copyval(vals)
        if idx == 0:
            vars_ = self._base["vars"]
            self._set_var(vars_, dim, path, vals, nvals)
            self._set_dirty_all()
        else:
            vars_ = self._scans[idx - 1]["vars"]
            self._set_var(vars_, dim, path, vals, nvals)
            self._scanscache[idx - 1]["dirty"] = True
            self._use_var_cache = []

    def _set_var(self, vars_, dim, path, vals, nvals):
        _ensure_vars(vars_, dim)
        sz = vars_[dim - 1]["size"]
        if sz == 0:
            vars_[dim - 1]["size"] = nvals
        elif sz != nvals:
            raise ValueError("Scan parameter size does not match")
        self._setfield(vars_[dim - 1]["params"], path, vals)

    def _assign_scan(self, idx, B):
        # MATLAB subsasgn `grp(idx) = B` (whole-scan replacement; B is a ScanParam or dict).
        self._use_var_cache = []
        if isinstance(B, ScanParam):
            if B._group is not self:
                raise ValueError("Cannot assign scan from a different group.")
            if B._path:
                raise ValueError("Can only assign a whole scan, not a sub-field.")
            bidx = B._idx
            if bidx == idx:
                return                                  # no-op
            if bidx == 0:
                rscan = self._base
                use_var = self._use_var_base
                rbase = 0
            else:
                rscan = self._scans[bidx - 1]
                if bidx <= len(self._use_var_scans):
                    use_var = self._use_var_scans[bidx - 1]
                else:
                    use_var = _def_use_var()
                rbase = self._getbaseidx(bidx)
            if idx == 0:
                self._base["params"] = copy.deepcopy(rscan["params"])
                self._base["vars"] = copy.deepcopy(rscan["vars"])
                self._use_var_base = copy.deepcopy(use_var)
                self._set_dirty_all()
            else:
                # Through setbase: checks for a loop AND initializes scans/scanscache.
                self.setbase(idx, rbase)
                self._scans[idx - 1]["params"] = copy.deepcopy(rscan["params"])
                self._scans[idx - 1]["vars"] = copy.deepcopy(rscan["vars"])
                self._scanscache[idx - 1]["dirty"] = True
                while len(self._use_var_scans) < idx:
                    self._use_var_scans.append(_def_use_var())
                self._use_var_scans[idx - 1] = copy.deepcopy(use_var)
            return
        if _hasarray(B):
            raise ValueError("Mixing fixed and variable parameters not allowed.")
        # Plain struct (dict) RHS: all fields treated as fixed; clears scan + base index.
        if idx == 0:
            self._base["params"] = copy.deepcopy(B)
            self._base["vars"] = []
            self._set_dirty_all()
        else:
            self._grow_scans(idx)
            self._scans[idx - 1]["params"] = copy.deepcopy(B)
            self._scans[idx - 1]["vars"] = []
            self._scans[idx - 1]["baseidx"] = 0
            self._scanscache[idx - 1]["dirty"] = True

    # ======================================================================= #
    # private helpers
    # ======================================================================= #
    def _getbaseidx(self, idx):
        return self._scans[idx - 1]["baseidx"]

    def _set_dirty_all(self):
        for c in self._scanscache:
            c["dirty"] = True
        self._use_var_cache = []

    def _grow_scans(self, idx):
        # Initialize scans (and their cache entries) up to `idx` with DEF_SCAN.
        while len(self._scans) < idx:
            self._scans.append(_def_scan())
        while len(self._scanscache) < idx:
            self._scanscache.append(_def_scancache())

    def _check_noconflict(self, idx, path, dim):
        # `dim == 0` means a fixed parameter. Initialize a not-yet-existing scan (no
        # conflict possible against an empty scan) and bail.
        if idx == 0:
            scan = self._base
        elif len(self._scans) < idx:
            self._grow_scans(idx)
            return
        else:
            scan = self._scans[idx - 1]
        if dim != 0:
            if _check_field(scan["params"], path):
                raise ValueError("Cannot scan a fixed parameter.")
        for i in range(1, len(scan["vars"]) + 1):
            if dim == i:
                continue
            if _check_field(scan["vars"][i - 1]["params"], path):
                if dim == 0:
                    raise ValueError("Cannot fix a scanned parameter.")
                raise ValueError("Cannot scan a parameter in multiple dimensions.")

    # -- static-ish struct helpers (MATLAB Static, Access=private) ------------- #
    @staticmethod
    def _setfield(obj, path, val):
        # subsasgn over a '.' path; create intermediate scalar structs; mutate in place.
        cur = obj
        for name in path[:-1]:
            nxt = cur.get(name)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[name] = nxt
            cur = nxt
        cur[path[-1]] = val
        return obj

    @staticmethod
    def _check_param_overwrite(obj, path, val):
        # Forbid changing a field's type (struct<->non-struct) and overwriting a struct.
        for name in path:
            if not isinstance(obj, dict):
                raise ValueError("Assignment to field of non-struct not allowed.")
            if name not in obj:
                return                              # creating a new field is fine
            obj = obj[name]
        is_struct = isinstance(val, dict)
        was_struct = isinstance(obj, dict)
        if is_struct and not was_struct:
            raise ValueError("Changing field from non-struct to struct not allowed.")
        elif not is_struct and was_struct:
            raise ValueError("Changing field from struct to non-struct not allowed.")
        elif is_struct and was_struct:
            raise ValueError("Override struct not allowed.")


# =========================================================================== #
# module-level value helpers (MATLAB Static isarray/hasarray + numel/copy)
# =========================================================================== #
def _check_field(obj, path):
    # True iff the dotted `path` is overwritten in `obj` (field exists, or a parent is a
    # non-(scalar-)struct). Mirrors ScanGroup.check_field.
    for name in path:
        if not isinstance(obj, dict):
            return True
        if name not in obj:
            return False
        obj = obj[name]
    return True


def _ensure_vars(vars_, dim):
    while len(vars_) < dim:
        vars_.append(_def_vars())


def _isarray(obj):
    # MATLAB ScanGroup.isarray: scalars are not arrays; a char *row* (Python str) is not an
    # array; a 1-element numeric array decays (isscalar). Cell arrays (no clean Python
    # analog) are not modeled -- a Python list/tuple is the numeric-array analog.
    if isinstance(obj, str):
        return False
    if isinstance(obj, dict):
        return False
    if isinstance(obj, (list, tuple)):
        return len(obj) != 1
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.size != 1
    return False


def _hasarray(obj):
    # isarray(obj) OR any *direct* field is an array (one level, matching ScanGroup).
    if _isarray(obj):
        return True
    if not isinstance(obj, dict):
        return False
    return any(_isarray(v) for v in obj.values())


def _numel(vals):
    if _np is not None and isinstance(vals, _np.ndarray):
        return int(vals.size)
    if isinstance(vals, (list, tuple, str)):
        return len(vals)
    return 1


def _copyval(v):
    # Reproduce MATLAB value-type copy-on-assignment for stored params/arrays.
    if isinstance(v, (dict, list)):
        return copy.deepcopy(v)
    if _np is not None and isinstance(v, _np.ndarray):
        return v.copy()
    return v


def _is_pos_int(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return x > 0
    if isinstance(x, float):
        return x > 0 and x.is_integer()
    return False


def _is_nonneg_int(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return x >= 0
    if isinstance(x, float):
        return x >= 0 and x.is_integer()
    return False
