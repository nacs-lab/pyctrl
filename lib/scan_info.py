"""scan_info.py -- ScanInfo: a read-only, DynProps-like view of one scan.

Faithful transliteration of ``matlab_new/lib/ScanInfo.m`` (returned by
``ScanGroup.get_scan(idx)``), again FOLDING IN the ``SubProps`` proxy role the way
``scan_param.py`` does -- ``ScanInfo`` carries the accumulated dotted ``path`` itself
rather than handing back a separate proxy object.

MATLAB ``ScanInfo.subsref`` returns ``[value, dim]`` for a field access:
  * a *fixed* leaf            -> ``(value, 0)``
  * a *swept* leaf            -> ``(value, dim)`` with ``dim`` the 1-based scan dimension
  * not found / a sub-tree    -> ``(SubProps(info, S), -1)`` (navigable, not yet resolved)
  * ``info.path(default)``     -> ``(default, 0)`` when not found, else ``(value, dim)``

Python has no ``nargout`` and resolves attribute access eagerly, so -- consistent with the
pyctrl convention that a ``SubProps`` leaf is realized by a trailing ``()`` (see
``sub_props.py`` / the realseq notes) -- navigation accumulates a path and a trailing call
resolves it:

    val, dim = g.get_scan(2).c()        # MATLAB: [val, dim] = g.get_scan(2).c
    val, dim = g.get_scan(2).e(2)       # default form -> (2, 0) if e is unset
    proxy, dim = g.get_scan(2).k()      # not-a-leaf -> (ScanInfo proxy, -1)
    g.get_scan(2).k.fieldnames()        # field names under .k

The resolution / fieldnames logic lives on ``ScanGroup`` (``_try_getfield`` /
``_info_fieldnames``); this class is a thin path-carrying delegator. ``disp``/``subdisp``
are display-only (Phase-4 W7, like DynProps ``test_disp``) and are omitted.
"""


class ScanInfo:
    def __init__(self, group, idx, path=()):
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_path", tuple(path))

    # -- lazy navigation: extend the dotted path (never resolve) --------------- #
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return ScanInfo(self._group, self._idx, self._path + (name,))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # ScanInfo is read-only (MATLAB ScanInfo has no subsasgn).
        raise TypeError("ScanInfo is read-only; mutate the ScanGroup via grp(idx).")

    # -- resolution: (value, dim) per ScanInfo.subsref ------------------------- #
    def __call__(self, *args):
        if len(args) > 1:
            raise ValueError("Wrong number of default value")
        val, dim = self._group._try_getfield(self._idx, self._path, True)
        if args:                                   # info.path(default)
            if dim < 0:
                return args[0], 0
            return val, dim
        if dim < 0:                                # not a leaf -> navigable proxy
            return self, -1
        return val, dim

    # -- introspection (no resolution) ----------------------------------------- #
    def fieldnames(self):
        return self._group._info_fieldnames(self._idx, self._path)

    def subfieldnames(self, path):
        return self._group._info_fieldnames(self._idx, tuple(path))

    def __repr__(self):
        if self._path:
            return "ScanInfo<%d>[.%s]" % (self._idx, ".".join(str(p) for p in self._path))
        return "ScanInfo<%d>" % self._idx
