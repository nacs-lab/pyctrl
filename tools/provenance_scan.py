#!/usr/bin/env python3
"""provenance_scan.py -- build a scan's param<->channel ``xref.json`` OFFLINE (Task 4c).

The producer side of the Sequence tab's param<->channel affordance (reader:
``yb_analysis/sequence/xref.py``). It rebuilds each unique compiled point with the
:mod:`provenance` capture active (an INERT-by-default lib hook -- never on during a live
run) and writes ``<scan>/sequence/xref.json`` keyed by the ``.seq`` filename:

    {"scan_id": "...",
     "by_file": {"point_00001__seqid_1.seq": {"param_to_channels": {...},
                                              "channel_to_params": {...}}, ...}}

CRUCIAL DESIGN POINT -- run on the LIVE ``lib`` (NOT the run's snapshot ``lib``):
the provenance hooks live in the live ``lib/{dyn_props,time_step,provenance}.py``; the
run's frozen code snapshot has neither. So unlike ``reconstruct_scan.py`` (which prepends
the snapshot ``lib`` for byte-faithful WAVEFORMS), this tool builds with the live ``lib``
so the hooks fire. It needs NO engine -- it walks the build (channel map + pulse tree),
never ``generate()``s -- so it runs in any interpreter with the live ``lib`` on path
(base python OR the engine venv). The per-point parameter VALUES are still the run's: they
come from the scan's own ``descriptor`` (in the ``.json`` sidecar), not from live config.

FIDELITY NOTE: because the EXPERIMENT code (YbSeqs/YbSteps/YbScans) used here is the LIVE
code, an xref for a scan whose sequence code changed since the run reflects the current
code's param<->channel structure (the run's *values* are exact; the *wiring* is current).
For recent scans live == the run. This is acceptable for a dormant viewer affordance; a
snapshot-faithful build would need the hooks compiled into the snapshot ``lib``.

Two ways it runs:
  * ``tools/reconstruct_scan.py`` spawns it as a fresh subprocess after writing the ``.seq``
    files (a fresh process picks up the live ``lib``, not reconstruct's snapshot ``lib``).
  * directly, to (re)build ``xref.json`` for a scan that already has ``.seq`` + a descriptor.

Emits a single ``XREF_RESULT:{json}`` line on stdout. Best-effort throughout.
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_TOOLS)

RESULT_PREFIX = "XREF_RESULT:"
SEQ_SUBDIR = "sequence"
XREF_NAME = "xref.json"
# Schema/format version stamped into xref.json so the viewer auto-rebuilds older artifacts.
# Bump when the producer's output changes in a way the UI should pick up:
#   1 aggregate maps · 2 + per-pulse regions · 3 + derivation formulas (cleaned)
#   4 + wait/timing regions (time_regions)
# Keep in lock-step with SEQ_XREF_VERSION in yb_analysis/plotting/static/dashboard.js.
XREF_VERSION = 4
_DEFAULT_TICK = 10 ** 12                              # config.yml tick_per_sec (1 ps); engine-free fallback
_SIDECAR_RE = re.compile(r"^data_\d{8}_\d{6}\.json$")
# LIVE experiment + lib dirs (NOT the snapshot) so the provenance hooks are loaded.
_LIVE_DIRS = ("lib", "YbSteps", "YbSeqs", "YbScans", "YbRearrangement", "YbExptCtrl")


def _bootstrap_live_path():
    """Put the LIVE ``lib`` (hooks) + experiment dirs + pyctrl root first on ``sys.path``."""
    add = [_PYCTRL, _TOOLS] + [os.path.join(_PYCTRL, d) for d in _LIVE_DIRS]
    for p in add:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _find_sidecar(scan_dir):
    base = os.path.join(scan_dir, os.path.basename(scan_dir) + ".json")
    if os.path.exists(base):
        return base
    for f in sorted(glob.glob(os.path.join(scan_dir, "data_*.json"))):
        if _SIDECAR_RE.match(os.path.basename(f)):
            return f
    return None


def _read_tick(default=_DEFAULT_TICK):
    """``tick_per_sec`` from the live ``config.yml`` (a tiny grep; avoids a yaml dep)."""
    try:
        with open(os.path.join(_PYCTRL, "config.yml"), encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*tick_per_sec\s*:\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return default


# --------------------------------------------------------------------------- #
# Per-point capture (shared with reconstruct_scan via subprocess). Engine-free.
# --------------------------------------------------------------------------- #
def capture_point_xref(seqfn, seqparam, exp_seq_cls=None):
    """Build one point with provenance active; return ``{param_to_channels, channel_to_params}``.

    A SEPARATE ``ExpSeq`` from any that produces a ``.seq`` (so a waveform build is never
    perturbed by the tagged values). Does NOT call ``generate()`` -- channel names come from
    the build's channel map, no engine needed.
    """
    import provenance
    if exp_seq_cls is None:
        from exp_seq import ExpSeq
        exp_seq_cls = ExpSeq
    sp = exp_seq_cls(seqparam)
    with provenance.capture(consts_dp=sp.C, globals_dp=sp.G) as sess:
        seqfn(sp)
    return sess.result()


def write_xref_json(seq_dir, by_file, *, scan_id=None):
    """Write ``xref.json`` (reader format) atomically. Returns the path, or None on failure."""
    if not by_file:
        return None
    try:
        os.makedirs(seq_dir, exist_ok=True)
        doc = {"scan_id": scan_id, "v": XREF_VERSION, "by_file": by_file}
        tmp = os.path.join(seq_dir, XREF_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        dst = os.path.join(seq_dir, XREF_NAME)
        os.replace(tmp, dst)
        return dst
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# .seq filename map -- so by_file keys match the scan's REAL .seq files.
# --------------------------------------------------------------------------- #
def _existing_seq_names(seq_dir):
    """Map ``str(seqid) -> .seq filename`` from the manifest (preferred) or by globbing."""
    manifest = os.path.join(seq_dir, "manifest.json")
    if os.path.exists(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                uniq = json.load(f).get("unique_seqs") or {}
            if uniq:
                return {str(k): v for k, v in uniq.items()}
        except (OSError, ValueError):
            pass
    out = {}
    for p in sorted(glob.glob(os.path.join(seq_dir, "point_*__seqid_*.seq"))):
        m = re.search(r"__seqid_(.+)\.seq$", os.path.basename(p))
        if m:
            out.setdefault(m.group(1), os.path.basename(p))
    return out


# --------------------------------------------------------------------------- #
# Standalone: build xref for a scan from its descriptor (LIVE code, engine-free).
# --------------------------------------------------------------------------- #
def build_scan_xref(scan_dir):
    """Build ``xref.json`` for ``scan_dir`` from its ``descriptor`` using the LIVE code."""
    scan_dir = os.path.abspath(scan_dir)
    sidecar = _find_sidecar(scan_dir)
    if not sidecar:
        return {"ok": False, "error": "no data_*.json sidecar in %s" % scan_dir}
    with open(sidecar) as f:
        cfg = json.load(f)
    descriptor = cfg.get("descriptor")
    if not descriptor:
        return {"ok": False, "error": "sidecar has no 'descriptor' -- cannot rebuild ScanGroup"}
    scan_id = None if cfg.get("scan_id") is None else str(int(cfg["scan_id"]))

    import seq_manager
    seq_manager.override_tick_per_sec(_read_tick())
    from seq_config import SeqConfig
    SeqConfig.reset()
    SeqConfig.load_real()                            # live expConfig consts (run's values override)
    from dispatch_descriptor import dispatch_descriptor
    disp = dispatch_descriptor(descriptor)
    scangroup, seqfn = disp.scangroup, disp.seq

    seq_dir = os.path.join(scan_dir, SEQ_SUBDIR)
    name_by_seqid = _existing_seq_names(seq_dir)

    n_total = int(scangroup.nseq())
    by_file = {}
    seen = set()
    failed = 0
    for n in range(1, n_total + 1):
        seqid, seqparam, _ = scangroup.getseq_with_var(n)
        key = str(seqid)
        if key in seen:
            continue
        seen.add(key)
        fname = name_by_seqid.get(key) or (
            "point_%05d__seqid_%s.seq" % (n, re.sub(r"[^A-Za-z0-9._-]", "_", key)))
        try:
            by_file[fname] = capture_point_xref(seqfn, seqparam)
        except Exception:  # noqa: BLE001 - one point's failure never aborts the rest
            failed += 1

    path = write_xref_json(seq_dir, by_file, scan_id=scan_id)
    n_edges = sum(len(e.get("channel_to_params", {})) for e in by_file.values())
    return {"ok": True, "n_seq": len(by_file), "n_points": n_total, "failed": failed,
            "n_channels_with_params": n_edges, "wrote": path, "scan_id": scan_id}


def main(argv):
    ap = argparse.ArgumentParser(description="Offline build a scan's param<->channel xref.json.")
    ap.add_argument("--scan-dir", required=True, help="the scan data folder (holds data_*.json)")
    args = ap.parse_args(argv)

    _bootstrap_live_path()
    try:
        result = build_scan_xref(args.scan_dir)
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                  "tb": traceback.format_exc()}

    sys.stdout.write(RESULT_PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
