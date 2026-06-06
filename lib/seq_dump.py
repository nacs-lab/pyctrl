"""seq_dump.py -- per-scan flattened ``.seq`` dump session (auto-dump for SeqPlotter).

Wired into the scan loop (``run_seq.run_scan_group``'s ``on_compile`` hook), this
emits one ``.seq`` per UNIQUE compiled sequence into ``<scan_dir>/sequence/`` and,
at scan end, a ``manifest.json`` mapping every scan point -> its ``.seq`` + the
scanned-axis values. The yb dashboard's Sequence tab reads exactly this layout
(``yb_analysis/sequence/manifest.py``) to step through the sweep and highlight the
scanned parameter.

How the bytes are produced: the engine evaluates the compiled sequence's channel
outputs via ``dump_output.dump_output_branches`` -- ``init_run`` -> loop(``pre_run``
-> ``get_nominal_output`` -> ``post_run``). It **never calls ``start()``**, so it
emits NO FPGA trigger / NI arm / camera frame: reading the nominal output is
hardware-free even on a live rig (it walks the in-memory ``host_seq``). The extra
``init_run`` is reset by the NEXT shot's own ``init_run`` (run_seq2.run_real).

WHEN it fires (load-bearing): the run loop does NOT call ``on_compile`` at compile
time -- it registers it on the seq's ``after_end`` callbacks (run_seq._arm_dump), so
the dump runs during the FIRST shot of each unique seq, after the before_start
callbacks injected this shot's runtime globals and before ``reset_globals`` wipes the
non-persist ones. This matters for global-dependent ramps (e.g. the 616-EOM slow
ramp, runtime_state.register_eom616_persistence): the dump's own ``pre_run`` re-runs
bc_gen reading the CURRENT global, so firing in this window captures the real ~ms
ramp instead of the ~15 s / ~60 MB ramp a 0-valued global builds at compile time.

Dedup (#2): ``on_compile`` runs once per ``seqid`` (the run loop arms it once per
compiled id, and the after_end registration fires it once), so a sweep over a
non-output parameter -- which yields ONE ``seqid`` -- writes ONE ``.seq``; the
manifest points all share it.

Engine-agnostic + unit-testable: byte production is ``dump_fn(seq, seq_name) -> bytes``
(default wraps ``dump_output_branches``); inject a fake in tests. Best-effort
throughout -- a dump or manifest failure logs and NEVER breaks the scan.
"""

import json
import logging
import os
import re

logger = logging.getLogger("pyctrl.seq_dump")

SEQ_SUBDIR = "sequence"
MANIFEST_NAME = "manifest.json"


def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))


def _jsonable(x):
    """Best-effort convert numpy / nested containers to JSON-safe Python."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return [_jsonable(v) for v in x.tolist()]
    except ImportError:
        pass
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def _walk(obj, dotted):
    """Follow a dotted path through dicts / attribute objects; None if absent."""
    node = obj
    for k in dotted.split("."):
        if isinstance(node, dict):
            if k not in node:
                return None
            node = node[k]
        else:
            node = getattr(node, k, None)
        if node is None:
            return None
    return node


def _default_dump_fn(seq, seq_name, pts_per_ramp):
    """Produce ``.seq`` bytes for a generated sequence via the engine (no start)."""
    import dump_output
    inv = getattr(seq, "inverse_chn_map", None)
    return dump_output.dump_output_branches(
        seq.pyseq, pts_per_ramp=pts_per_ramp, seq_name=seq_name, inverse_chn_map=inv)


class SeqDumpSession:
    """Accumulate per-unique-sequence ``.seq`` dumps + write ``manifest.json``."""

    def __init__(self, seq_dir, scangroup, *, scan_id=None, seq_name=None,
                 dump_fn=None, pts_per_ramp=100, datetime_stamp=None, log=None):
        self.seq_dir = seq_dir
        self.scangroup = scangroup
        self.scan_id = scan_id
        self.seq_name = seq_name
        self.pts_per_ramp = pts_per_ramp
        self._dt = datetime_stamp
        self._log = log or (lambda _m: None)
        self._dump_fn = dump_fn
        self.unique = {}            # str(seqid) -> filename
        self._made_dir = False

    # -- helpers -------------------------------------------------------------
    def _ensure_dir(self):
        if not self._made_dir:
            os.makedirs(self.seq_dir, exist_ok=True)
            self._made_dir = True

    def _seq_label(self):
        try:
            import dump_output
            return dump_output.format_seq_name(self.seq_name or "seq", self._dt)
        except Exception:  # noqa: BLE001
            return self.seq_name or "seq"

    def _dump(self, seq, seq_name):
        if self._dump_fn is not None:
            return self._dump_fn(seq, seq_name)
        return _default_dump_fn(seq, seq_name, self.pts_per_ramp)

    def _git_state(self):
        try:
            import code_snapshot
            return code_snapshot.read_git_state()
        except Exception:  # noqa: BLE001
            return None

    # -- the run-loop hook ---------------------------------------------------
    def on_compile(self, arg0, seqid, seq):
        """Dump one unique compiled sequence's ``.seq`` (deduped by seqid).

        Despite the name, the run loop does NOT call this at compile time -- it is
        armed on the seq's after_end callbacks (run_seq._arm_dump) and fires during
        the first shot, so a global-dependent ramp reads its injected runtime value
        (see this module's docstring). Idempotent per seqid via ``self.unique``.
        """
        key = str(seqid)
        if key in self.unique:
            return
        try:
            self._ensure_dir()
            data = self._dump(seq, self._seq_label())
            fname = "point_%05d__seqid_%s.seq" % (int(arg0), _safe(key))
            tmp = os.path.join(self.seq_dir, fname + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(self.seq_dir, fname))
            self.unique[key] = fname
            self._log("[seq_dump] wrote %s (seqid=%s, %d bytes)"
                      % (fname, key, len(data)))
        except Exception as exc:  # noqa: BLE001 - never break the scan
            logger.warning("seq_dump on_compile failed (seqid=%s): %s", seqid, exc)
            self._log("[seq_dump] dump failed for seqid=%s: %s" % (seqid, exc))

    # -- scanned-axis extraction (best-effort, group 1) ----------------------
    def _scanned_axes(self):
        axes = []
        try:
            ndim = int(self.scangroup.scandim(1))
        except Exception:  # noqa: BLE001
            return axes
        for dim in range(1, ndim + 1):
            try:
                nfields = int(self.scangroup.axisnum(1, dim))
            except Exception:  # noqa: BLE001
                nfields = 0
            for field in range(1, nfields + 1):
                try:
                    values, path = self.scangroup.get_scanaxis(1, dim, field)
                    axes.append({"dim": dim, "path": path,
                                 "values": _jsonable(values)})
                except Exception:  # noqa: BLE001
                    continue
        return axes

    # -- end of scan ---------------------------------------------------------
    def finalize(self):
        """Write ``manifest.json``; return the manifest dict (or None on failure)."""
        try:
            self._ensure_dir()
            axes = self._scanned_axes()
            paths = [a["path"] for a in axes]
            try:
                n_total = int(self.scangroup.nseq())
            except Exception:  # noqa: BLE001
                n_total = 0
            points = []
            for n in range(1, n_total + 1):
                try:
                    seqid, seqparam, _ = self.scangroup.getseq_with_var(n)
                except Exception:  # noqa: BLE001
                    continue
                scanned = {}
                for p in paths:
                    v = _walk(seqparam, p)
                    if v is not None:
                        scanned[p] = _jsonable(v)
                points.append({"n": n, "seqid": _jsonable(seqid),
                               "file": self.unique.get(str(seqid)),
                               "scanned": scanned})
            manifest = {
                "scan_id": self.scan_id,
                "seq": self.seq_name,
                "scanned_axes": axes,
                "points": points,
                "unique_seqs": dict(self.unique),
                "git": self._git_state(),
            }
            tmp = os.path.join(self.seq_dir, MANIFEST_NAME + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, os.path.join(self.seq_dir, MANIFEST_NAME))
            self._log("[seq_dump] manifest: %d point(s), %d unique seq(s)"
                      % (len(points), len(self.unique)))
            return manifest
        except Exception as exc:  # noqa: BLE001
            logger.warning("seq_dump finalize failed: %s", exc)
            return None
