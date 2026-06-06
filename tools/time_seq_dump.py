#!/usr/bin/env python3
"""time_seq_dump.py -- measure the SeqPlotter auto-dump overhead (saved vs unsaved).

For EVERY sequence in ``pyctrl/YbSeqs`` it builds + compiles the sequence on the REAL
libnacs engine, hardware-free (``use_dummy_device: true`` -> the zynq backend opens no
socket; we never call ``start()``), and times two paths:

  * UNSAVED = build + ``generate()``                         (toggle OFF; normal compile)
  * SAVED   = build + ``generate()`` + dump + ``.seq`` write (toggle ON)

The dump is exactly what ``runner.run()`` does per UNIQUE compiled sequence
(``seq_dump.SeqDumpSession.on_compile`` -> ``dump_output.dump_output_branches``:
``init_run`` -> loop(``pre_run`` -> ``get_nominal_output`` -> ``post_run``)).

Each sequence is averaged over ``--n`` iterations (default 50; one warmup discarded).
Heavy sequences (a dump that compiles a very large bytecode -- e.g. a ramp-from-0 because
the dump runs before ``before_start`` sets a global) are still measured but ``gc`` runs
every iteration to bound memory.

Run with the ENGINE python in a maintenance window (no MATLAB / live scan needed -- this is
hardware-free, but it loads libnacs):

    <py38> pyctrl/tools/time_seq_dump.py --n 50 --out C:\\Temp\\seq_dump_timing.json
"""

import argparse
import gc
import inspect
import json
import os
import re
import statistics
import sys
import time
import traceback

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_TOOLS)
for _p in (_PYCTRL, os.path.join(_PYCTRL, "lib"), _TOOLS,
           os.path.join(_PYCTRL, "YbSeqs"), os.path.join(_PYCTRL, "YbSteps"),
           os.path.join(_PYCTRL, "YbExptCtrl")):    # runtime_state / rearrange_runtime live here
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG_PATH = os.path.join(_PYCTRL, "config.yml")
CONFIG_REF = os.path.join(_PYCTRL, "tests", "reference", "config_reference.json")
YBSEQS_DIR = os.path.join(_PYCTRL, "YbSeqs")

# A dump that compiles more than this (bytes) or takes longer than this (s) on the
# warmup is "heavy": still measured, but gc every iteration to bound memory.
_HEAVY_BYTES = 5_000_000
_HEAVY_SECS = 2.0


def discover_seqs():
    out = []
    for fn in sorted(os.listdir(YBSEQS_DIR)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        name = fn[:-3]
        if re.search(r"Seq\d*$", name):    # the YbSeq naming convention (XxxSeq, XxxSeq2, ...)
            out.append(name)
    return out


def build(name):
    """Build a YbSeq (0-arg returns its own ExpSeq; 1-arg takes a configured one)."""
    from exp_seq import ExpSeq
    seqfn = getattr(__import__(name), name)
    if len(inspect.signature(seqfn).parameters) == 0:
        return seqfn()
    s = ExpSeq()
    seqfn(s)
    return s


def generate(s):
    """ExpSeq.generate(), but tolerant of the NiDAQ-type error under use_dummy_device.

    use_dummy_device makes every device a generic dummy, so ``get_nidaq_channel_info``
    (called inside the real ``generate()``) throws "Device NiDAQ is not a Ni DAQ". The dump
    only needs the engine's nominal output (host_seq), not the NI channel map, so we swallow
    that one error and leave ``ni_channels`` empty. Everything else matches ExpSeq.generate()."""
    import seq_manager
    if s.pyseq is None:
        s.pyseq = seq_manager.create_sequence(s.serialize())
        try:
            info = s.pyseq.get_nidaq_channel_info("NiDAQ")
        except Exception:  # noqa: BLE001 - dummy NiDAQ isn't a NiDAQ; ni_channels unused here
            info = None
        s.ni_channels = [{"chn": int(x[0]), "dev": str(x[1])} for x in (info or [])]
        s.reset_globals(True)
    return s


def setup_engine():
    import seq_manager
    if not seq_manager.engine_available():
        return False
    mgr = seq_manager.get()
    with open(CONFIG_PATH) as f:
        cfg = f.read()
    mgr.load_config_string("use_dummy_device: true\n" + cfg)   # hardware-free backends
    try:
        mgr.enable_debug(False)            # quiet the engine's add_debug_printf chatter
    except Exception:
        pass
    from seq_config import SeqConfig
    SeqConfig.reset()
    try:
        SeqConfig.load_real()                  # production expConfig.py
    except Exception:
        SeqConfig.load_real(config_path=CONFIG_REF)
    return True


def _stats(xs):
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean_ms": 1e3 * statistics.fmean(xs),
        "median_ms": 1e3 * statistics.median(xs),
        "std_ms": 1e3 * (statistics.pstdev(xs) if len(xs) > 1 else 0.0),
        "min_ms": 1e3 * xs[0],
        "max_ms": 1e3 * xs[-1],
    }


def time_one(name, n, pts, tmp):
    import seq_manager
    import dump_output
    import compare_seq_bytes

    # --- warmup (not counted): one full build+generate+dump; also sizes the .seq ---
    s = build(name)
    generate(s)
    w0 = time.perf_counter()
    data = dump_output.dump_output_branches(
        s.pyseq, pts_per_ramp=pts, seq_name=name,
        inverse_chn_map=getattr(s, "inverse_chn_map", None))
    warm_dump = time.perf_counter() - w0
    dec = compare_seq_bytes.decode(data)
    info = {
        "name": name,
        "nseq": len(dec["seqs"]),
        "nchn": sum(len(sq["channels"]) for sq in dec["seqs"]),
        "npts": sum(len(ch["points"]) for sq in dec["seqs"] for ch in sq["channels"]),
        "bytes": len(data),
    }
    del s
    gc.collect()

    heavy = len(data) > _HEAVY_BYTES or warm_dump > _HEAVY_SECS
    info["heavy"] = heavy

    seq_manager.new_run()                      # mimic scan start (once per seq)
    fpath = os.path.join(tmp, name + ".seq")
    compile_t, dump_t = [], []
    for i in range(n):
        t0 = time.perf_counter()
        s = build(name)
        generate(s)
        t1 = time.perf_counter()
        data = dump_output.dump_output_branches(
            s.pyseq, pts_per_ramp=pts, seq_name=name,
            inverse_chn_map=getattr(s, "inverse_chn_map", None))
        with open(fpath, "wb") as f:
            f.write(data)
        t2 = time.perf_counter()
        compile_t.append(t1 - t0)              # UNSAVED (compile only)
        dump_t.append(t2 - t1)                 # the auto-dump overhead
        del s
        if heavy or (i & 7) == 0:
            gc.collect()

    saved_t = [c + d for c, d in zip(compile_t, dump_t)]
    info["unsaved"] = _stats(compile_t)
    info["saved"] = _stats(saved_t)
    info["overhead"] = _stats(dump_t)
    info["overhead_pct"] = (info["overhead"]["mean_ms"] / info["unsaved"]["mean_ms"] * 100.0
                            if info["unsaved"]["mean_ms"] else float("nan"))
    return info


def _fmt_table(rows):
    hdr = ("%-26s %4s %5s %6s %9s | %11s %11s %11s %7s" %
           ("sequence", "nseq", "nchn", "npts", "bytes",
            "unsaved ms", "saved ms", "ovhd ms", "ovhd%"))
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "error" in r:
            lines.append("%-26s  ERROR: %s" % (r["name"], r["error"]))
            continue
        flag = " *" if r.get("heavy") else ""
        lines.append(
            "%-26s %4d %5d %6d %9d | %5.1f+-%-4.1f %5.1f+-%-4.1f %6.1f %6.1f%s"
            % (r["name"], r["nseq"], r["nchn"], r["npts"], r["bytes"],
               r["unsaved"]["mean_ms"], r["unsaved"]["std_ms"],
               r["saved"]["mean_ms"], r["saved"]["std_ms"],
               r["overhead"]["mean_ms"], r["overhead_pct"], flag))
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="iterations per sequence (>=50)")
    ap.add_argument("--pts", type=int, default=100, help="pts_per_ramp for the dump")
    ap.add_argument("--seqs", nargs="*", help="subset of seq names (default: all)")
    ap.add_argument("--out", default=os.path.join(_TOOLS, "seq_dump_timing.json"))
    args = ap.parse_args(argv)

    if not setup_engine():
        print("ENGINE UNAVAILABLE: libnacs not importable in this interpreter")
        return 2

    import tempfile
    tmp = tempfile.mkdtemp(prefix="seq_dump_timing_")
    names = args.seqs or discover_seqs()
    print("timing %d sequence(s), n=%d, pts=%d (engine, use_dummy_device)\n"
          % (len(names), args.n, args.pts))

    # The engine prints pulse dumps / debug to the C-level stdout (fd 1). Redirect fd 1 to
    # devnull for the run so it doesn't drown the report; our progress goes to stderr, and we
    # restore fd 1 before printing the final table.
    rows = []
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved_fd = os.dup(1)
    os.dup2(_devnull, 1)
    try:
        for name in names:
            try:
                t = time.perf_counter()
                r = time_one(name, args.n, args.pts, tmp)
                r["wall_s"] = round(time.perf_counter() - t, 1)
                rows.append(r)
                print("  done %-26s unsaved %.1f ms  saved %.1f ms  (+%.1f ms, %.0f%%)  [%.0fs]%s"
                      % (name, r["unsaved"]["mean_ms"], r["saved"]["mean_ms"],
                         r["overhead"]["mean_ms"], r["overhead_pct"], r["wall_s"],
                         "  *HEAVY" if r["heavy"] else ""), file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - record + continue
                rows.append({"name": name, "error": "%s: %s" % (type(exc).__name__, exc),
                             "tb": traceback.format_exc()})
                print("  FAIL %-26s %s" % (name, exc), file=sys.stderr)
            sys.stderr.flush()
    finally:
        os.dup2(_saved_fd, 1)          # restore real stdout for the report
        os.close(_saved_fd)
        os.close(_devnull)

    ok = [r for r in rows if "error" not in r]
    print("\n" + _fmt_table(rows))
    if ok:
        tot_un = sum(r["unsaved"]["mean_ms"] for r in ok)
        tot_sv = sum(r["saved"]["mean_ms"] for r in ok)
        print("\n%-26s %43s %5.1f      %5.1f   (+%.1f ms total, %.0f%%)"
              % ("TOTAL (mean per seq, summed)", "",
                 tot_un, tot_sv, tot_sv - tot_un,
                 (tot_sv - tot_un) / tot_un * 100.0 if tot_un else float("nan")))
        print("  * = heavy dump (large bytecode at the dump's pre_run; see notes)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n": args.n, "pts": args.pts, "results": rows}, f, indent=2)
    print("\nJSON -> %s" % args.out)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    # The libnacs engine bundles libzmq; loaded-but-never-started (we never call start())
    # its static dtor asserts at exit and the CRT abort() hangs on Windows. Skip DLL detach
    # by terminating our own process AFTER all output is flushed (mirrors tests/conftest.py).
    if sys.platform == "win32":
        import ctypes
        sys.stdout.flush()
        sys.stderr.flush()
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x0001, False, os.getpid())   # PROCESS_TERMINATE
        k.TerminateProcess(h, rc)
    sys.exit(rc)
