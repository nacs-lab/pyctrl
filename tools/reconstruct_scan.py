#!/usr/bin/env python3
"""reconstruct_scan.py -- regenerate a scan's .seq waveforms OFFLINE (SeqPlotter Task 4).

Run by the ENGINE python (``.venv-engine``, py3.8 + libnacs) as a SEPARATE, hardware-free
subprocess (``use_dummy_device: true`` -> the Zynq backend opens no socket; ``start()`` is
never called). Spawned by the dashboard's ``/api/sequence/reconstruct`` when a scan has NO
``.seq`` but HAS a code snapshot.

Mechanism (SEQPLOTTER_INTEGRATION_PLAN.md §12.3 / §12.7 Q-A/Q-F):
  1. read the scan's ``data_<stamp>.json`` sidecar -> ``descriptor`` + ``scan_id``.
  2. prepend the run's FULL code snapshot (``_code_snapshots/_runs/<scan_id>/``, INCL ``lib``
     + ``YbExptCtrl``) to ``sys.path`` BEFORE importing any builder module -- a fresh process
     has no live singletons, so replaying ``lib`` too is safe and is the 100%-fidelity enabler
     (the "lib is record-only" rule only protects the LIVE backend).
  3. seed ``expConfig`` from the snapshot (its ``expConfig.py`` / ``config.yml`` are on
     ``sys.path``, snapshot-first).
  4. ``dispatch_descriptor(descriptor)`` -> (ScanGroup, seq function).
  5. per UNIQUE compiled point: ``ExpSeq(seqparam); seqfn(s); generate()``; ``set_global`` the
     captured runtime globals (Q-F, ``sequence/globals.json``) BEFORE the dump's
     ``get_nominal_output``; ``dump_output_branches`` -> ``point_NNNNN__seqid_MMMM.seq``.
  6. write ``manifest.json`` (reusing ``seq_dump.SeqDumpSession``) back into ``<scan>/sequence/``.

A seq with runtime globals that were NOT captured (legacy, pre-Q-F) is flagged
``approximate`` -- its global-dependent channels can't be reproduced faithfully. The driver
MUST NOT write the shared ``runtime_state`` mmap; it ``set_global``s in its OWN pyseq and only
READS captured values.

Emits a single line ``RECONSTRUCT_RESULT:{json}`` on stdout. The engine's own chatter goes to
fd 1, which is redirected to devnull during the build so the result line is clean.
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

RESULT_PREFIX = "RECONSTRUCT_RESULT:"
SEQ_SUBDIR = "sequence"
_SIDECAR_RE = re.compile(r"^data_\d{8}_\d{6}\.json$")
_SNAPSHOT_SUBDIRS = ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans", "YbRearrangement")


def _bootstrap_live_path():
    """Live pyctrl dirs so we can import ``code_snapshot`` for the snapshot lookup.

    Snapshot dirs are prepended LATER (before importing builder modules) so they win; these
    are APPENDED so they're only a fallback.
    """
    for p in (_PYCTRL, os.path.join(_PYCTRL, "lib"),
              os.path.join(_PYCTRL, "YbExptCtrl"), _TOOLS):
        if p not in sys.path:
            sys.path.append(p)


def _find_sidecar(scan_dir):
    base = os.path.join(scan_dir, os.path.basename(scan_dir) + ".json")
    if os.path.exists(base):
        return base
    for f in sorted(glob.glob(os.path.join(scan_dir, "data_*.json"))):
        if _SIDECAR_RE.match(os.path.basename(f)):
            return f
    return None


def _prepend_snapshot(data_root, scan_id):
    """Prepend the run's snapshot tree (incl ``lib``) to ``sys.path``; return the folder or None."""
    import code_snapshot
    folder = code_snapshot.run_folder(data_root, scan_id)
    if not os.path.isdir(folder) or not os.path.isfile(os.path.join(folder, "manifest.json")):
        return None
    add = [folder] + [os.path.join(folder, d) for d in _SNAPSHOT_SUBDIRS]
    add = [d for d in add if os.path.isdir(d)]
    sys.path[0:0] = add                       # snapshot wins over the appended live dirs
    return folder


def _setup_engine(snapshot_folder):
    import seq_manager
    if not seq_manager.engine_available():
        raise RuntimeError("libnacs engine not importable in this interpreter")
    mgr = seq_manager.get()
    cfg_path = None
    if snapshot_folder:
        c = os.path.join(snapshot_folder, "config.yml")
        if os.path.exists(c):
            cfg_path = c
    if cfg_path is None:
        cfg_path = os.path.join(_PYCTRL, "config.yml")
    with open(cfg_path) as f:
        mgr.load_config_string("use_dummy_device: true\n" + f.read())  # hardware-free backends
    try:
        mgr.enable_debug(False)
    except Exception:  # noqa: BLE001
        pass
    from seq_config import SeqConfig
    SeqConfig.reset()
    SeqConfig.load_real()                      # snapshot's expConfig.py is first on sys.path


def _generate(s):
    """``ExpSeq.generate()`` tolerant of the dummy NiDAQ error (mirrors time_seq_dump).

    ``use_dummy_device`` makes every device generic, so ``get_nidaq_channel_info`` raises
    "Device NiDAQ is not a Ni DAQ". The dump only needs the nominal output, not the NI map,
    so swallow that one error.
    """
    import seq_manager
    if s.pyseq is None:
        s.pyseq = seq_manager.create_sequence(s.serialize())
        try:
            info = s.pyseq.get_nidaq_channel_info("NiDAQ")
        except Exception:  # noqa: BLE001
            info = None
        s.ni_channels = [{"chn": int(x[0]), "dev": str(x[1])} for x in (info or [])]
        s.reset_globals(True)
    return s


def _apply_globals(s, entries):
    """``set_global`` the captured runtime globals on the compiled handle, matched by id.

    Returns ``(applied, missing)``. ``missing > 0`` (a global with no captured value) means
    a global-dependent channel can't be reproduced -> the seq is flagged "approximate".
    """
    have = {int(e["id"]): e["value"] for e in (entries or [])}
    applied = missing = 0
    for g in getattr(s, "globals", []):
        gid = int(g["id"])
        if gid in have:
            try:
                s.set_global(gid, float(have[gid]))
                applied += 1
            except Exception:  # noqa: BLE001
                missing += 1
        else:
            missing += 1
    return applied, missing


def reconstruct(scan_dir, pts_per_ramp=100):
    """Regenerate every unique point's ``.seq`` + ``manifest.json`` into ``<scan>/sequence/``."""
    scan_dir = os.path.abspath(scan_dir)
    sidecar = _find_sidecar(scan_dir)
    if not sidecar:
        return {"ok": False, "error": "no data_*.json sidecar in %s" % scan_dir}
    with open(sidecar) as f:
        cfg = json.load(f)
    descriptor = cfg.get("descriptor")
    if not descriptor:
        return {"ok": False, "error": "sidecar has no 'descriptor' -- scan predates "
                "self-contained reconstruction; cannot rebuild the ScanGroup"}
    if cfg.get("scan_id") is None:
        return {"ok": False, "error": "sidecar has no scan_id"}
    scan_id = str(int(cfg["scan_id"]))
    data_root = os.path.dirname(os.path.dirname(scan_dir))         # <DATA>

    snapshot_folder = _prepend_snapshot(data_root, scan_id)
    if snapshot_folder is None:
        return {"ok": False, "error": "no code snapshot for scan %s" % scan_id}

    _setup_engine(snapshot_folder)

    from exp_seq import ExpSeq
    import dump_output
    from dispatch_descriptor import dispatch_descriptor
    from seq_dump import SeqDumpSession

    disp = dispatch_descriptor(descriptor)
    scangroup, seqfn, seq_name = disp.scangroup, disp.seq, disp.seq_name

    # Captured runtime globals (Q-F): {str(seqid): [{id, value, ...}]}.
    seq_dir = os.path.join(scan_dir, SEQ_SUBDIR)
    captured = {}
    gpath = os.path.join(seq_dir, "globals.json")
    if os.path.exists(gpath):
        try:
            captured = json.load(open(gpath)).get("globals") or {}
        except Exception:  # noqa: BLE001
            captured = {}

    os.makedirs(seq_dir, exist_ok=True)
    n_total = int(scangroup.nseq())
    unique = {}                                # str(seqid) -> filename
    approximate = False
    for n in range(1, n_total + 1):
        seqid, seqparam, _ = scangroup.getseq_with_var(n)
        key = str(seqid)
        if key in unique:
            continue                           # dedup: one .seq per unique compiled point
        s = ExpSeq(seqparam)
        seqfn(s)
        _generate(s)
        _applied, missing = _apply_globals(s, captured.get(key))
        if missing:
            approximate = True
        data = dump_output.dump_output_branches(
            s.pyseq, pts_per_ramp=pts_per_ramp, seq_name=seq_name,
            inverse_chn_map=getattr(s, "inverse_chn_map", None))
        fname = "point_%05d__seqid_%s.seq" % (n, re.sub(r"[^A-Za-z0-9._-]", "_", key))
        tmp = os.path.join(seq_dir, fname + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, os.path.join(seq_dir, fname))
        unique[key] = fname
        del s

    # manifest.json -- reuse SeqDumpSession.finalize for an identical schema, then stamp
    # the reconstruction flags the dashboard surfaces.
    sess = SeqDumpSession(seq_dir, scangroup, scan_id=scan_id, seq_name=seq_name)
    sess.unique = unique
    sess._made_dir = True
    manifest = sess.finalize()
    if manifest is not None:
        manifest["reconstructed"] = True
        manifest["approximate"] = approximate
        tmp = os.path.join(seq_dir, "manifest.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp, os.path.join(seq_dir, "manifest.json"))

    return {"ok": True, "n_seq": len(unique),
            "n_points": n_total, "approximate": approximate, "scan_id": scan_id}


def main(argv):
    ap = argparse.ArgumentParser(description="Offline reconstruct a scan's .seq waveforms.")
    ap.add_argument("--scan-dir", required=True, help="the scan data folder (holds data_*.json)")
    ap.add_argument("--pts-per-ramp", type=int, default=100)
    args = ap.parse_args(argv)

    _bootstrap_live_path()

    # The engine prints pulse/debug to the C-level stdout (fd 1). Redirect fd 1 to devnull for
    # the build so the result line is clean; restore before printing it.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull, 1)
    try:
        result = reconstruct(args.scan_dir, pts_per_ramp=args.pts_per_ramp)
    except Exception as exc:  # noqa: BLE001 - report, never crash silently
        result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                  "tb": traceback.format_exc()}
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull)

    sys.stdout.write(RESULT_PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    # libnacs bundles libzmq; loaded-but-never-started, its static dtor asserts at exit and the
    # CRT abort() hangs on Windows. Terminate our own process after flushing (mirrors
    # tests/conftest.py + time_seq_dump.py).
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x0001, False, os.getpid())   # PROCESS_TERMINATE
        k.TerminateProcess(h, rc)
    sys.exit(rc)
