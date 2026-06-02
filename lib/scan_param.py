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
special-cases ``scan``/``usevar``. Reading a fixed value (``grp(idx).a()``) resolves the
parameter (Python form of MATLAB ``param_subsref``'s trailing empty ``()``); ``usevar`` (W8)
marks a parameter (or the whole scan) as a runtime variable; ``toscan`` (W7) converts the
param into a one-scan ``ScanGroup`` via ``ScanGroup.cat_scans`` -- the same path as MATLAB's
``[g1, g2]``/``horzcat``.

A ``ScanParam`` ``_idx`` is normally a positive int (one scan) or 0 (the default), but the
multi-index concat form (MATLAB ``g(2:end)`` / ``g([1, 3])``) carries a LIST of indices.
That list form is consumed only by ``cat_scans`` (display-only in MATLAB); field authoring
on a multi-index param is not supported, matching MATLAB.
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

    # -- resolve a fixed/swept value (Python form of MATLAB `grp(idx).a()`) ------ #
    def __call__(self, *args):
        # MATLAB param_subsref only resolves the trailing empty `()`; a default arg
        # (`grp.a(default)`) is "Invalid parameter access syntax", and so is a bare
        # `grp(idx)()` with no field path.
        if args:
            raise ValueError("Invalid parameter access syntax.")
        if not self._path:
            raise ValueError("Invalid parameter access syntax.")
        val, dim = self._group._try_getfield(self._idx, self._path, True)
        if dim < 0:
            raise ValueError("Parameter does not exist yet.")
        return val

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
    # -- mark a parameter (or this whole scan) as a runtime variable ------------- #
    def usevar(self, *args):
        # `grp(idx).usevar(val[, dim])` (scan-level) or `grp(idx).a.b.usevar(...)` (per-field)
        # -- the Python form of MATLAB's `.usevar(...)` (param_subsref). `args` is
        # ``(val,)`` or ``(val, dim)``.
        self._group._param_usevar(self._idx, self._path, *args)

    # -- convert this param into a one-scan ScanGroup (MATLAB `toscan`) ---------- #
    def toscan(self):
        from scan_group import ScanGroup
        return ScanGroup.cat_scans(self)

    def __repr__(self):
        who = "default" if self._idx == 0 else str(self._idx)
        if self._path:
            return "ScanParam<%s>[.%s]" % (who, ".".join(str(p) for p in self._path))
        return "ScanParam<%s>" % who
