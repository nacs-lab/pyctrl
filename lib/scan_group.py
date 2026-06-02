"""scan_group.py -- ScanGroup: an ordered set of N-dimensional parameter scans.

Faithful transliteration of ``matlab_new/lib/ScanGroup.m`` (a ``handle`` class). This file
covers **Phase-4 W2** (core data model + authoring DSL), **W3** (materialization +
queries): ``getfullscan`` (base-merge + dirty cache), the column-major
``getseq``/``getseq_in_scan`` expansion, ``nseq``/``scansize``/``scandim``/``axisnum``, and
``get_fixed``/``get_vars``/``get_scan``/``get_scanaxis``, and **W7** (test-only surface for
full TestScanGroup parity): ``cat_scans``/``horzcat``/``toscan`` (concat), ``load``/
``load_v0``/``load_v1``/``validate`` (the inverse of ``dump``), and a minimal
``get_full_use_var`` (what ``cat_scans`` needs). Still TODO: ``usevar`` +
``getseq_*_with_var`` (W8); ``ScanAccessTracker`` (W9).

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
from scan_info import ScanInfo
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
        # grp(2:end) / grp([1, 3]) -> a multi-index ScanParam (concat form, see cat_scans);
        # the MATLAB magic colon `:` maps to the fallback, a bounded range/list does not.
        if len(args) == 0:
            return ScanParam(self, 0)
        if len(args) > 1:
            raise ValueError("Too many scan index")
        idx = args[0]
        if idx == ":":
            return ScanParam(self, 0)
        if isinstance(idx, slice):
            # A full colon `g(:)` (slice(None, None)) -> the fallback (MATLAB magic colon).
            if idx.start is None and idx.stop is None:
                return ScanParam(self, 0)
            # A bounded slice is a 1-based INCLUSIVE range (MATLAB `start:stop`).
            step = idx.step if idx.step is not None else 1
            idxlist = list(range(idx.start, idx.stop + 1, step))
            if not all(_is_pos_int(i) for i in idxlist):
                raise ValueError("Scan index must be positive")
            return ScanParam(self, [int(i) for i in idxlist])
        if isinstance(idx, (list, tuple)):
            if not all(_is_pos_int(i) for i in idx):
                raise ValueError("Scan index must be positive")
            return ScanParam(self, [int(i) for i in idx])
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
    # W3: materialization + queries
    # ======================================================================= #
    def getseq(self, n):
        # The n-th (1-based) sequence parameter across the whole group.
        for scani in range(1, self.groupsize() + 1):
            ss = self.scansize(scani)
            if n <= ss:
                return self.getseq_in_scan(scani, n)
            n = n - ss
        raise ValueError("Sequence index out of bound.")

    def getseq_in_scan(self, scanidx, seqidx):
        # The seqidx-th (1-based) point within scan `scanidx`. THE byte-critical path:
        # a deterministic column-major mixed-radix enumeration -- dimension 1 varies
        # fastest, dummy (size 0) dimensions skipped (ScanGroup.m:278-294).
        scan = self._getfullscan(scanidx)          # already an independent copy
        seq = scan["params"]
        seqidx = seqidx - 1                         # 0-based from here on
        for var in scan["vars"]:
            size = var["size"]
            if size == 0:
                continue
            subidx = seqidx % size                  # 0-based
            seqidx = (seqidx - subidx) // size
            for v, path in _foreach_nonstruct(var["params"]):
                self._setfield(seq, path, v[subidx])    # MATLAB 1-based v(subidx+1)
        return seq

    def nseq(self):
        return sum(self.scansize(i) for i in range(1, self.groupsize() + 1))

    def scansize(self, idx):
        scan = self._getfullscan(idx)
        res = 1
        for var in scan["vars"]:
            sz1d = var["size"]
            if sz1d != 0:
                res = res * sz1d
        return res

    def scandim(self, idx):
        # Number of scan dimensions, including dummy ones.
        return len(self._getfullscan(idx)["vars"])

    def axisnum(self, idx=1, dim=1):
        # Number of scan parameters for scan `idx` along axis `dim` (0 for out-of-bound).
        scan = self._getfullscan(idx)
        if dim > len(scan["vars"]) or scan["vars"][dim - 1]["size"] <= 1:
            return 0
        return sum(1 for _ in _foreach_nonstruct(scan["vars"][dim - 1]["params"]))

    def get_fixed(self, idx):
        if idx == 0:
            raise ValueError("Out of bound scan index.")
        return self._getfullscan(idx)["params"]

    def get_vars(self, idx, dim=1):
        # Returns (params, size) for scan `idx` along `dim` (size 0 = dummy dimension).
        if idx == 0:
            raise ValueError("Out of bound scan index.")
        scan = self._getfullscan(idx)
        var = scan["vars"][dim - 1]
        return var["params"], var["size"]

    def get_scan(self, idx):
        if idx == 0:
            raise ValueError("Out of bound scan index.")
        return ScanInfo(self, idx)

    def get_scanaxis(self, idx, dim, field=1):
        # The scan axis (values + dotted path) for scan `idx` along `dim`. `field` selects
        # the parameter by 1-based index or by dotted-name string. A dummy/out-of-bound
        # dimension falls back to the fixed parameters (matching the size-1 decay).
        if idx == 0:
            raise ValueError("Out of bound scan index.")
        scan = self._getfullscan(idx)
        if len(scan["vars"]) < dim or scan["vars"][dim - 1]["size"] == 0:
            params = scan["params"]                 # deprecated fallback path
        else:
            params = scan["vars"][dim - 1]["params"]
        if isinstance(field, str):
            for v, path in _foreach_nonstruct(params):
                if ".".join(path) == field:
                    return v, field
            raise ValueError("Cannot find scan field")
        # numeric: the `field`-th (1-based) non-struct leaf, in traversal order.
        remaining = int(field)
        for v, path in _foreach_nonstruct(params):
            remaining -= 1
            if remaining == 0:
                return v, ".".join(path)
        raise ValueError("Cannot find scan field")

    # ======================================================================= #
    # W7: concatenation (cat_scans/horzcat/toscan) + load (inverse of dump)
    # ======================================================================= #
    @staticmethod
    def cat_scans(*items):
        # MATLAB `[g1, g2]` / `horzcat` / `toscan`: build a NEW group whose scans are the
        # (base-merged) scans of every input, each with baseidx reset to 0; runparam copied
        # from the first input (ScanGroup.m:1295-1321). Each item is a ScanGroup (all its
        # scans) or a ScanParam (its idx, possibly a list / 0=default).
        res = ScanGroup()
        res._scans = []                              # MATLAB: res.scans(end) = []
        res._scanscache = []
        for item in items:
            if isinstance(item, ScanGroup):
                grp = item
                idxs = range(1, grp.groupsize() + 1)
            elif isinstance(item, ScanParam):
                grp = item._group
                idx = item._idx
                idxs = idx if isinstance(idx, (list, tuple)) else [idx]
            else:
                raise ValueError("Only ScanGroup allowed in concatenation.")
            for j in idxs:
                scan = grp._getfullscan(j)           # already an independent copy
                res._scans.append({"baseidx": 0,
                                   "params": scan["params"],
                                   "vars": scan["vars"]})
                res._use_var_scans.append(copy.deepcopy(grp.get_full_use_var(j)))
        # runparam comes from the FIRST input (its group, if a ScanParam).
        src = items[0]
        if isinstance(src, ScanParam):
            src = src._group
        res._scanscache = [_def_scancache() for _ in res._scans]
        res._runparam = DynProps(copy.deepcopy(src._runparam()))
        return res

    @staticmethod
    def horzcat(*items):
        return ScanGroup.cat_scans(*items)

    def get_full_use_var(self, idx):
        # The use-var tree for scan `idx` with its base chain merged in
        # (ScanGroup.m:1253-1264). Minimal W7 surface: with no usevar set, the merge is a
        # no-op chain down to use_var_base. The full usevar machinery is W8.
        if idx == 0:
            return self._use_var_base
        base = self.get_full_use_var(self._getbaseidx(idx))
        if idx > len(self._use_var_scans):
            return base
        return _merge_use_var(base, self._use_var_scans[idx - 1])

    def validate(self):
        # Faithful no-op: MATLAB ScanGroup.validate (~:1245-1252) is all comments -- `load`
        # calls it expecting loop/conflict/size checks that never happen. Replicated as a
        # no-op for behavior parity (see project_pyctrl_scangroup_latent_bugs #2).
        pass

    @staticmethod
    def load(obj):
        if "version" not in obj:
            raise ValueError("Version missing.")
        if obj["version"] == 1:
            self = ScanGroup._load_v1(obj)
        elif obj["version"] == 0:
            self = ScanGroup._load_v0(obj)
        else:
            raise ValueError("Wrong object version: %r" % (obj["version"],))
        self.validate()
        return self

    @staticmethod
    def _load_v1(obj):
        # Inverse of dump(); deep-copies the payload so the loaded group is independent.
        self = ScanGroup()
        self._scans = copy.deepcopy(obj["scans"])
        self._base = copy.deepcopy(obj["base"])
        self._runparam = DynProps(copy.deepcopy(obj["runparam"]))
        self._scanscache = [_def_scancache() for _ in self._scans]
        if "use_var_base" in obj:
            self._use_var_base = copy.deepcopy(obj["use_var_base"])
            self._use_var_scans = copy.deepcopy(obj["use_var_scans"])
        else:
            # Loading an old payload (no version bump needed -- backward compatible).
            self._use_var_base = _def_use_var()
            self._use_var_scans = []
        return self

    @staticmethod
    def _load_v0(obj):
        # Legacy `p`/`scan` struct-array layout (ScanGroup.m:1516-1549). `obj['p']` is a list
        # of dicts (a MATLAB struct array, uniform fields); a missing/empty value falls back
        # to the first element. Numeric leaves with >1 element become a single dim-1 axis.
        self = ScanGroup()
        self._runparam = DynProps(copy.deepcopy(obj["scan"]))
        p = obj["p"]
        fields = list(p[0].keys()) if p else []
        self._scans = []
        for i in range(len(p)):
            scan = _def_scan()
            vars1d = _def_vars()
            for name in fields:
                val = p[i].get(name)
                if _is_empty(val):
                    val = p[0].get(name)
                sz = _numel(val)
                if sz <= 1:
                    scan["params"][name] = _copyval(val)
                    continue
                if vars1d["size"] == 0:
                    vars1d["size"] = sz
                elif vars1d["size"] != sz:
                    raise ValueError("Scan size mismatch")
                vars1d["params"][name] = _copyval(val)
            if vars1d["size"] != 0:
                scan["vars"] = [vars1d]
            self._scans.append(scan)
        self._scanscache = [_def_scancache() for _ in self._scans]
        return self

    # ======================================================================= #
    # ScanInfo-facing implementation (MATLAB methods(Access=?ScanInfo))
    # ======================================================================= #
    def _try_getfield(self, idx, path, allow_scan):
        # Look up `path` for scan `idx`. Returns (value, 0) for a fixed leaf, (value, dim)
        # for a swept leaf, or (None, -1) when not found as a leaf (ScanGroup.m:1171-1197).
        if idx == 0:
            scan = self._base
        elif len(self._scans) < idx:
            return None, -1
        else:
            scan = self._getfullscan(idx)
        val, res = _get_leaffield(scan["params"], path)
        if res:
            return val, 0
        if not allow_scan:
            return None, -1
        for dim in range(1, len(scan["vars"]) + 1):
            val, res = _get_leaffield(scan["vars"][dim - 1]["params"], path)
            if res:
                return val, dim
        return None, -1

    def _info_fieldnames(self, idx, path):
        # Union of field names under `path`, across fixed + every scan-dim params,
        # first-seen order (ScanGroup.m:711-742).
        res = []

        def add(params):
            p = params
            for name in path:
                if not isinstance(p, dict):
                    raise ValueError("Parameter parent overwriten")
                if name not in p:
                    return
                p = p[name]
            if not isinstance(p, dict):
                return
            for name in p:
                if name not in res:
                    res.append(name)

        scan = self._getfullscan(idx)
        add(scan["params"])
        for var in scan["vars"]:
            add(var["params"])
        return res

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

    def _check_dirty(self, idx):
        # Dirty if this scan -- or any scan up its base chain -- has a dirty cache entry.
        while idx != 0:
            if self._scanscache[idx - 1]["dirty"]:
                return True
            idx = self._scans[idx - 1]["baseidx"]
        return False

    def _getfullscan(self, idx):
        # The scan with its base scan merged in. Returns ``{'params', 'vars'}`` as an
        # INDEPENDENT copy (MATLAB returns a value-copy, so every caller may freely mutate).
        # Caches the merged result keyed by the per-scan dirty flag (ScanGroup.m:1085-1133).
        if idx == 0:
            return {"params": copy.deepcopy(self._base["params"]),
                    "vars": copy.deepcopy(self._base["vars"])}
        if not self._check_dirty(idx):
            c = self._scanscache[idx - 1]
            return {"params": copy.deepcopy(c["params"]),
                    "vars": copy.deepcopy(c["vars"])}
        params = copy.deepcopy(self._scans[idx - 1]["params"])
        vars_ = copy.deepcopy(self._scans[idx - 1]["vars"])
        scan = {"params": params, "vars": vars_}
        base = self._getfullscan(self._getbaseidx(idx))
        # Merge fixed parameters the child hasn't already set (child wins; dim identity
        # tracked by find_scan_dim so a base value never lands on a child-owned path).
        for v, path in _foreach_nonstruct(base["params"]):
            if _find_scan_dim(scan, path) >= 0:
                continue
            self._setfield(params, path, copy.deepcopy(v))
        # Merge variable parameters, keeping each base axis on its ORIGINAL dimension index.
        for scanid in range(1, len(base["vars"]) + 1):
            for v, path in _foreach_nonstruct(base["vars"][scanid - 1]["params"]):
                if _find_scan_dim(scan, path) >= 0:
                    continue
                _ensure_vars(vars_, scanid)
                self._setfield(vars_[scanid - 1]["params"], path, copy.deepcopy(v))
        # Recount sizes with a strict length-consistency check.
        for var in vars_:
            scansize = 0
            for v, path in _foreach_nonstruct(var["params"]):
                nv = _numel(v)
                if nv == 1:
                    raise ValueError("Too few elements to scan.")
                elif scansize == 0:
                    scansize = nv
                elif scansize != nv:
                    raise ValueError("Inconsistent scan size.")
            var["size"] = scansize
        self._scanscache[idx - 1]["params"] = params
        self._scanscache[idx - 1]["vars"] = vars_
        self._scanscache[idx - 1]["dirty"] = False
        return {"params": copy.deepcopy(params), "vars": copy.deepcopy(vars_)}

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


def _foreach_nonstruct(obj):
    # Yield (value, path) for every non-struct LEAF of a nested params dict, depth-first in
    # insertion order. Mirrors ScanGroup.foreach_nonstruct: the top must be a (scalar)
    # struct; an empty sub-struct is skipped (neither descended nor reported); a list/array
    # (a scan axis) is a leaf. Python dicts preserve insertion order, so traversal order
    # matches MATLAB fieldnames order (relevant for get_scanaxis index lookup + the oracle).
    if not isinstance(obj, dict):
        raise ValueError("Object is not a struct.")
    yield from _walk_nonstruct(obj, ())


def _walk_nonstruct(obj, prefix):
    for name, v in obj.items():
        if isinstance(v, dict):
            if v:                              # non-empty struct -> descend
                yield from _walk_nonstruct(v, prefix + (name,))
            # empty struct -> skip (matches MATLAB: not pushed, not reported)
        else:
            yield v, prefix + (name,)


def _get_leaffield(obj, path):
    # (value, True) iff `path` resolves to a non-struct LEAF in `obj`; (None, False) if a
    # name along the path is missing OR the path lands on a (sub-)struct. Raises if a parent
    # along the path is a non-struct (ScanGroup.get_leaffield "Parameter parent overwriten").
    for name in path:
        if not isinstance(obj, dict):
            raise ValueError("Parameter parent overwriten")
        if name not in obj:
            return None, False
        obj = obj[name]
    is_leaf = not isinstance(obj, dict)
    return (obj if is_leaf else None), is_leaf


def _find_scan_dim(scan, path):
    # 0 if `path` is a fixed parameter of `scan`, the 1-based dim if it is a scan parameter,
    # else -1 (ScanGroup.find_scan_dim).
    if _check_field(scan["params"], path):
        return 0
    for i in range(1, len(scan["vars"]) + 1):
        if _check_field(scan["vars"][i - 1]["params"], path):
            return i
    return -1


def _merge_use_var(base, update):
    # Merge `base` into `update`, update wins (ScanGroup.m:1579-1600). MATLAB passes structs
    # by value, so copy `update` before mutating. W7 only exercises the all-default case
    # (base.field is always empty); the field-recursion branch is ported faithfully but
    # never fires here (and replicates MATLAB's whole-field overwrite verbatim).
    update = copy.deepcopy(update)
    if update["def"] == 0:
        update["def"] = base["def"]
    ndims = len(base["dims"])
    while len(update["dims"]) < ndims:
        update["dims"].append(0)
    for i in range(ndims):
        if update["dims"][i] == 0:
            update["dims"][i] = base["dims"][i]
    for fld, val in base["field"].items():
        if fld not in update["field"]:
            update["field"][fld] = copy.deepcopy(val)
        else:
            update["field"] = _merge_use_var(val, update["field"][fld])
    return update


def _is_empty(v):
    # MATLAB isempty: [] / '' / 0x0 struct. Python: None or a zero-length list/tuple/str.
    if v is None:
        return True
    if isinstance(v, (list, tuple, str)):
        return len(v) == 0
    if _np is not None and isinstance(v, _np.ndarray):
        return v.size == 0
    return False


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
