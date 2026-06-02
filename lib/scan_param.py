"""scan_param.py -- ScanParam: a handle into one scan (or the fallback) of a ScanGroup.

Faithful transliteration of ``matlab_new/lib/ScanParam.m`` -- but FOLDING IN the role
that MATLAB plays with a separate ``SubProps`` proxy.

In MATLAB, ``grp(idx)`` returns a ``ScanParam(group, idx)`` and a chained field access
(``grp(idx).a.b``) builds an ``S`` struct array that ``subsref``/``subsasgn`` forward to
the group; the value returned for further chaining is a generic ``SubProps(param, S)``.
Python has no generic ``subsref`` dispatch, so the ported ``SubProps`` (sub_props.py) is
hard-wired to a ``DynProps`` root and cannot also serve a ``ScanParam`` parent. Rather than
invent a second generic proxy, ``ScanParam`` here simply CARRIES the accumulated dotted
``path`` itself: ``grp(idx)`` has ``path=()`` and ``grp(idx).a.b`` is the same class with
``path=('a','b')``. Every authoring operation forwards ``(idx, path, ...)`` to the group,
exactly as MATLAB forwards ``(idx, S)``.

The DSL diverges from MATLAB syntax only where Python has no equivalent (per
PYTHON_FRONTEND_PLAN.md Phase 4):
  * ``grp(idx).a.b = x``            fixed parameter      (MATLAB: same)
  * ``grp(idx).a.b.scan(dim, vals)`` scan axis            (MATLAB: ``.scan(dim) = vals``)
  * ``grp(idx).a.b.scan(vals)``      scan axis, dim 1     (MATLAB: ``.scan(vals)``)
  * ``grp(idx).assign(rhs)``         replace whole scan   (MATLAB: ``grp(idx) = rhs``)
  * ``grp(idx).size(dim)``           scan size along dim  (MATLAB: ``size(grp(idx), dim)``)

Because ``scan``/``assign``/``size``/``usevar``/``toscan`` are real methods, a parameter
literally named one of those is shadowed (unreachable by attribute) -- MATLAB likewise
special-cases ``scan``/``usevar``. Reading a fixed value (``grp(idx).a()``) is Phase-4 W3
(materialize/query); ``usevar`` is W8 and ``toscan`` is W7 -- those raise here.
"""


class ScanParam:
    def __init__(self, group, idx, path=()):
        # Bypass our own __setattr__ for the real attributes.
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_path", tuple(path))

    # -- lazy navigation: extend the dotted path; never resolve / never mutate -- #
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return ScanParam(self._group, self._idx, self._path + (name,))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # grp(idx).<path>.name = value  -> a single fixed parameter (never a scan axis).
        self._group._addparam(self._idx, self._path + (name,), value)

    # -- explicit scan-axis authoring (Python form of MATLAB `.scan(dim) = vals`) - #
    def scan(self, *args):
        if not self._path:
            # MATLAB: scan() directly on grp(idx) -> "Must specify parameter to scan."
            raise ValueError("Must specify parameter to scan.")
        if len(args) == 0:
            raise ValueError("Too few arguments for scan()")
        elif len(args) == 1:
            dim, vals = 1, args[0]
        elif len(args) == 2:
            dim, vals = args
        else:
            raise ValueError("Too many arguments for scan()")
        self._group._addscan(self._idx, self._path, dim, vals)

    # -- whole-scan replacement (Python form of MATLAB `grp(idx) = rhs`) --------- #
    def assign(self, rhs):
        if self._path:
            raise ValueError("Cannot assign to a sub-field; assign() replaces a whole scan.")
        self._group._assign_scan(self._idx, rhs)

    # -- scan size along a dimension (MATLAB `size(param, dim)`) ----------------- #
    def size(self, dim=1):
        return self._group._param_size(self._idx, dim)

    # -- deferred surfaces ------------------------------------------------------- #
    def usevar(self, *args):
        raise NotImplementedError("ScanParam.usevar is Phase-4 W8.")

    def toscan(self):
        raise NotImplementedError("ScanParam.toscan is Phase-4 W7 (cat_scans).")

    def __repr__(self):
        who = "default" if self._idx == 0 else str(self._idx)
        if self._path:
            return "ScanParam<%s>[.%s]" % (who, ".".join(str(p) for p in self._path))
        return "ScanParam<%s>" % who
