"""analyze_run_timing.py -- aggregate a run-loop timing CSV into a per-stage report.

Reads the per-shot CSV that ``lib/run_timing.py`` writes (``run_timing_<ts>.csv`` under the
pyctrl log dir) and prints a per-stage breakdown: mean / median / p90 / max ms-per-shot, the
number of shots that paid each stage, and the share of the shot's total. This is the offline
counterpart of the end-of-scan ``scan_summary`` line the runner already logs -- run it after a
scan to localize the gap between the hardware-sequence floor (the ``wait`` stage) and the full
per-shot wall-clock.

Two cuts of the data are shown so a once-per-point cost is not confused with a per-shot one:
  * ALL shots -- the honest per-shot mean (what actually slows the scan).
  * STEADY-STATE only (``compiled=0``) vs FIRST-OF-POINT (``compiled=1``) -- isolates the lazy
    ``compile`` (build+serialize+generate) cost, which fires only on a cache miss. If ``compile``
    is non-zero on a large fraction of shots, the per-point seq cache is not being hit (a
    regression) -- the report flags that.

Usage::

    python pyctrl/tools/analyze_run_timing.py                 # newest CSV in the log dir
    python pyctrl/tools/analyze_run_timing.py <path-to.csv>   # a specific file
    python pyctrl/tools/analyze_run_timing.py --glob          # list available CSVs

No dependencies beyond the stdlib.
"""

import argparse
import csv
import glob
import os
import sys


def _default_log_dir():
    override = os.environ.get("YB_PYCTRL_LOG_DIR")
    if override:
        return override
    # tools/analyze_run_timing.py -> pyctrl -> superproject root.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "log", "pyctrl_log")


def _newest_csv(log_dir):
    files = glob.glob(os.path.join(log_dir, "run_timing_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _floats(rows, col):
    out = []
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stage_cols(rows):
    """The ``*_ms`` stage columns in file order, excluding total/other (handled separately)."""
    if not rows:
        return []
    cols = [c for c in rows[0].keys() if c.endswith("_ms")
            and c not in ("total_ms",)]
    return cols


def _report(rows, title):
    n = len(rows)
    if not n:
        print("  (no shots)")
        return
    totals = _floats(rows, "total_ms")
    grand = sum(totals)
    mean_total = grand / n
    print("  %-14s %9s %9s %9s %9s %7s %6s" % (
        "stage", "mean", "median", "p90", "max", "n_hit", "%tot"))
    print("  " + "-" * 72)
    stats = []
    for col in _stage_cols(rows):
        vals = _floats(rows, col)
        if not vals:
            continue
        s = sorted(vals)
        nz = [v for v in vals if v > 0]
        total = sum(vals)
        stats.append((col[:-3],                       # strip "_ms"
                      total / n,                       # mean over ALL shots
                      _pct(s, 0.5), _pct(s, 0.9), s[-1],
                      len(nz),
                      (total / grand * 100.0) if grand else 0.0))
    stats.sort(key=lambda r: -r[1])                    # by per-shot mean, descending
    for name, mean, med, p90, mx, hits, pct in stats:
        print("  %-14s %9.1f %9.1f %9.1f %9.1f %7d %5.1f%%" % (
            name, mean, med, p90, mx, hits, pct))
    print("  " + "-" * 72)
    print("  %-14s %9.1f ms/shot over %d shots (sum %.1f s)" % (
        "TOTAL", mean_total, n, grand / 1e3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="timing CSV (default: newest in the log dir)")
    ap.add_argument("--glob", action="store_true", help="list available CSVs and exit")
    args = ap.parse_args(argv)

    log_dir = _default_log_dir()
    if args.glob:
        files = sorted(glob.glob(os.path.join(log_dir, "run_timing_*.csv")))
        if not files:
            print("no run_timing_*.csv in %s" % log_dir)
        for f in files:
            print(f)
        return 0

    path = args.csv or _newest_csv(log_dir)
    if not path or not os.path.exists(path):
        print("no timing CSV found (looked in %s) -- run a scan with YB_RUN_TIMING=1 first"
              % log_dir, file=sys.stderr)
        return 1

    rows = _read_rows(path)
    print("=== run-loop timing: %s ===" % os.path.basename(path))

    # If the rows carry per-scan labels (the runner stamps "<name> <id> async=0/1"), report each
    # scan separately -- this is how an async-on vs async-off A/B for the same seq separates.
    labels = [r.get("scan", "") for r in rows]
    distinct = [s for s in dict.fromkeys(labels) if s]
    if len(distinct) > 1:
        for lab in distinct:
            sub = [r for r in rows if r.get("scan", "") == lab]
            print("\n########## scan: %s  (%d shots) ##########" % (lab, len(sub)))
            _report(sub, lab)
        return 0

    print("\n[ALL shots]")
    _report(rows, "all")

    steady = [r for r in rows if r.get("compiled", "0") in ("0", "", "0.0")]
    first = [r for r in rows if r.get("compiled") in ("1", "1.0")]
    if first:
        print("\n[STEADY-STATE shots (compiled=0): %d]" % len(steady))
        _report(steady, "steady")
        print("\n[FIRST-OF-POINT shots (compiled=1): %d -- pay the lazy compile]" % len(first))
        _report(first, "first")
        frac = len(first) / len(rows) * 100.0
        if frac > 50.0:
            print("\n  !! %.0f%% of shots paid a compile -- the per-point seq cache is NOT "
                  "being reused (expected ~1 compile per distinct scan point)." % frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
