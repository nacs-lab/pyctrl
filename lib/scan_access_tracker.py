"""scan_access_tracker.py -- ScanAccessTracker: warn about unused scan parameters.

Faithful transliteration of ``matlab_new/lib/ScanAccessTracker.m`` (a ``handle`` class).
It is a post-hoc lint, NOT an enumerator: it CONSUMES the per-sequence ``accessed`` tree the
Phase-3 ``DynProps`` already produces (``get_accessed``), unions it per scan, and once every
sequence of a scan has been collected, warns about any parameter (fixed or swept) that no
sequence in that scan ever touched. Scanning a parameter (e.g. a delay) can legitimately
disable code that leaves other params unused in *some* sequences, so a param counts as used
as long as ANY sequence in the scan accessed it -- hence the per-scan union.

This is pure tree arithmetic over nested dicts:
  * ``_to_bool_struct``  -- collapse a params struct to a bool tree (every non-struct leaf
    -> ``True``); the leaf marks "there is a parameter here".
  * ``_merge_struct``    -- union of two bool/accessed trees; a non-struct on either side
    wins as ``True`` ("accessed at/under here").
  * ``_compute_unused``  -- subtract the ``accessed`` tree from the params tree, leaving only
    the never-accessed parameters.
  * ``_show_unused`` / ``_warn_unused`` -- render the leftover tree as the dotted, indented
    warning message.

PORTER TRAP (project_pyctrl_scangroup_latent_bugs #3): MATLAB ``to_bool_struct`` "returns"
its in-place-mutated input ``v``; a naive port that builds a fresh ``out`` but forgets to
return it yields empty trees. Here we build a fresh dict and return it EXPLICITLY.

The run-loop wiring (``runSeq2.m``) that feeds ``record_access`` live is Phase 5; this class
is exercised in Phase-4 W9 via synthetic ``accessed`` dicts (``TestScanAccessTracker``). The
MATLAB ``FacyOnCleanup`` / ``warning('off','backtrace')`` dance is display-only and dropped;
warnings are emitted via the Python ``warnings`` module.
"""

import warnings


class ScanAccessTracker:
    def __init__(self, sg, warnfixed=True):
        # One scan_info per scan in the group: the (bool) fixed + vars param trees, the union
        # of accessed params, the count of sequences still to run, and a checked flag.
        self._scan_infos = []
        self._scan_index = []          # global (1-based) sequence index -> scan number
        self._warnfixed = warnfixed
        nscans = sg.groupsize()
        for i in range(1, nscans + 1):
            ss = sg.scansize(i)
            self._scan_index.extend([i] * ss)
            fixed = ScanAccessTracker._to_bool_struct(sg.get_fixed(i))
            vars_ = {}
            for j in range(1, sg.scandim(i) + 1):
                params, _ = sg.get_vars(i, j)
                vars_ = ScanAccessTracker._merge_struct(
                    vars_, ScanAccessTracker._to_bool_struct(params))
            self._scan_infos.append({"fixed": fixed, "vars": vars_, "accessed": {},
                                     "seq_left": ss, "checked": False})
        # Whether each global sequence has been collected yet.
        self._collected = [False] * len(self._scan_index)

    # ----------------------------------------------------------------------- #
    # public API (fed by the run loop in Phase 5; by synthetic dicts in tests)
    # ----------------------------------------------------------------------- #
    def record_access(self, idx, accessed):
        # Union `accessed` into this sequence's scan, then mark the sequence collected.
        scan_idx = self._scan_index[idx - 1]
        info = self._scan_infos[scan_idx - 1]
        info["accessed"] = ScanAccessTracker._merge_struct(info["accessed"], accessed)
        self.mark_collected(idx)

    def mark_collected(self, idx):
        # Same effect as record_access with an empty accessed struct; a separate method so a
        # future per-sequence warning can hook here (ScanAccessTracker.m:177-192).
        if self._collected[idx - 1]:
            return
        self._collected[idx - 1] = True
        scan_idx = self._scan_index[idx - 1]
        info = self._scan_infos[scan_idx - 1]
        info["seq_left"] -= 1
        if info["seq_left"] == 0:
            self._process_scan(scan_idx)

    def force_check(self):
        # Check every scan now, in case some sequences will never be run.
        for i in range(1, len(self._scan_infos) + 1):
            self._process_scan(i)

    # ----------------------------------------------------------------------- #
    # internals
    # ----------------------------------------------------------------------- #
    def _process_scan(self, scan_idx):
        info = self._scan_infos[scan_idx - 1]
        if info["checked"]:
            return
        info["checked"] = True
        unused_fixed = ScanAccessTracker._compute_unused(info["fixed"], info["accessed"])
        unused_vars = ScanAccessTracker._compute_unused(info["vars"], info["accessed"])
        if self._warnfixed:
            ScanAccessTracker._warn_unused(unused_fixed, scan_idx, True)
        ScanAccessTracker._warn_unused(unused_vars, scan_idx, False)

    # -- static tree arithmetic (MATLAB Static, Access=private) -------------- #
    @staticmethod
    def _to_bool_struct(v):
        # Collapse a params struct to a bool tree: every non-struct leaf -> True. Build a
        # FRESH dict and return it explicitly (avoid the in-place-return porter trap).
        if not isinstance(v, dict):
            return True
        out = {}
        for name, sv in v.items():
            out[name] = ScanAccessTracker._to_bool_struct(sv)
        return out

    @staticmethod
    def _merge_struct(v, v2):
        # Union of two trees; a non-struct on either side collapses to True.
        if not isinstance(v, dict) or not isinstance(v2, dict):
            return True
        out = dict(v)
        for name, sv2 in v2.items():
            if name not in out:
                out[name] = _deepcopy_tree(sv2)
            else:
                out[name] = ScanAccessTracker._merge_struct(out[name], sv2)
        return out

    @staticmethod
    def _compute_unused(params, accessed):
        # Subtract the accessed tree from the params tree -> only never-accessed params.
        if not isinstance(accessed, dict):
            return {}                       # accessed is a leaf -> everything here used
        if not isinstance(params, dict):
            return params                   # param leaf, accessed went deeper -> still unused
        out = {}
        for name, pv in params.items():
            if name not in accessed:
                out[name] = pv              # not accessed at all -> unused
                continue
            sub = ScanAccessTracker._compute_unused(pv, accessed[name])
            if isinstance(sub, dict) and len(sub) == 0:
                continue                    # fully used -> drop
            out[name] = sub
        return out

    @staticmethod
    def _show_unused(s, indent, needs_dot):
        if not isinstance(s, dict):
            return ""
        parts = []
        prefix = " " * indent
        for i, name in enumerate(s):
            seg = ""
            if i != 0:
                seg += "\n" + prefix
            if needs_dot:
                seg += "." + name
                subindent = indent + len(name) + 1
            else:
                seg += name
                subindent = indent + len(name)
            seg += ScanAccessTracker._show_unused(s[name], subindent, True)
            parts.append(seg)
        return "".join(parts)

    @staticmethod
    def _warn_unused(unused, scan_idx, fixed):
        if len(unused) == 0:
            return
        if fixed:
            msg = "Unused fixed parameters in scan #%d:" % scan_idx
        else:
            msg = "Unused scanning parameters in scan #%d:" % scan_idx
        warnings.warn(msg + "\n  " + ScanAccessTracker._show_unused(unused, 2, False))


def _deepcopy_tree(v):
    # A bool tree is dicts + leaves (True / numbers / lists). Copy structure so a merged tree
    # never aliases the caller's accessed dict (MATLAB struct assignment copies by value).
    if isinstance(v, dict):
        return {k: _deepcopy_tree(sv) for k, sv in v.items()}
    return v
