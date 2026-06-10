"""run_timing.py -- opt-in per-shot stage timing for the run loop.

A diagnostic to localize where a shot's wall-clock goes. The hardware sequence
(``run_bseq``'s ``wait`` poll-loop) is the irreducible floor (~the programmed seq
length, e.g. ~700 ms); everything else -- the engine per-shot calls
(``init_run``/``pre_run``/``start``/``post_run``/``reset_globals``), the callbacks
(camera capture, AWG/SLM heartbeat, runtime-global injection), the lazy compile, and
the tail pause -- is overhead this module attributes stage by stage so the gap between
"basic-sequence time" and "per-seq time" can be read off directly.

**Zero-cost when OFF (the default).** The fast path is a single ``is None`` check: when
no shot is active (timing disabled), :func:`stage` / :func:`substage` yield immediately
without touching the clock. So instrumenting the byte-critical-adjacent run loop costs
nothing on a normal run. (The run loop is NOT the serialize path, so THE ONE RULE is not
at stake here -- but keeping it inert keeps normal-run behavior byte-for-byte unchanged.)

**Turn it on** either way:
  * env ``YB_RUN_TIMING=1`` before launching the backend (inherited by the PyctrlLauncher
    child), or
  * touch the toggle file ``<log_dir>/RUN_TIMING_ON`` -- picked up at the next shot, so a
    LIVE backend can be switched without a restart (and switched back by deleting it).

The enabled decision is re-evaluated once per shot (in :func:`begin_shot`), which is a
couple of µs against a ~700 ms shot -- negligible.

**Output** (best-effort; a failure here never perturbs a run):
  * one concise log line per shot (all non-zero stages, ms),
  * a CSV row per shot at ``<log_dir>/run_timing_<ts>.csv`` (one column per stage) for
    offline aggregation (``tools/analyze_run_timing.py``),
  * a mean/median/max per-stage summary logged at end-of-scan (:func:`scan_summary`).

Stage taxonomy (the flat top-level stages sum to ~the shot total; ``other`` is the small
unattributed remainder -- loop/index bookkeeping):

    gate          control.check_pause_abort (in-process ZMQ request poll)
    compile       prepare_seq's lazy compile (build+serialize+generate) -- ONLY on a cache
                  miss (first time a distinct scan point is seen); ~0 on a rep/reuse hit
    tstartwait    the per-shot ``tstartwait`` sleep (NI-DAQ driver-timing workaround; 0 by default)
    pre_cb        pre-shot callbacks: AWG recall + SLM ensure_held heartbeat
    before_start  run_real before_start_cbs (runtime-global injection, e.g. EOM616 freq)
    init_run      pyseq.init_run() (engine)
    before_bseq   per-bseq before_bseq_cbs
    pre_run       pyseq.pre_run() (engine)  [per-bseq, summed over bseqs]
    ni_arm        get_nidaq_data + NiDAQRunner.run (NI write/arm)
    start         pyseq.start() (engine; triggers the FPGA)
    wait          the wait poll-loop -- the HARDWARE sequence time (the ~700 ms floor)
    ni_wait       NiDAQRunner.wait()
    after_bseq    per-bseq after_bseq_cbs
    bseq_len      pyseq.cur_bseq_length() (engine)
    post_run      pyseq.post_run() (engine; branch routing)
    after_branch  per-bseq after_branch_cbs
    after_end     run_real after_end_cbs (per-UNIQUE-seq: seq dump + globals capture fire
                  here on the FIRST shot of each seq, then never again)
    reset_globals seq.reset_globals(False) (engine)
    tail_pause    the tail wall-clock pad (usually ~0 -- the HW wait already covers it)
    post_cb       post-shot callbacks (the standard frame-capture publish)

Informational sub-stages (NOT summed into the flat total; they break a top-level stage
down further):
    cam_read      frame_capture: reading NumImages frames off the camera buffer (the part
                  of post_cb that can stall on the camera/DCAM)
    cam_store     frame_capture: to_store_array + store_imgs* + seq_finish (in-process)

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import contextlib
import os
import time

# --- canonical column order (CSV + log line). Top-level stages sum to ~total. -------- #
STAGES = [
    "gate", "compile", "tstartwait", "pre_cb",
    "before_start", "init_run",
    "before_bseq", "pre_run", "ni_arm", "start", "wait", "ni_wait",
    "after_bseq", "bseq_len", "post_run", "after_branch",
    "after_end", "reset_globals", "tail_pause", "post_cb",
]
# Sub-stages refine a top-level stage; excluded from the accounted sum to avoid double counts.
SUBSTAGES = ["cam_read", "cam_store"]

# Per-shot accumulator (single-threaded run loop -> a module global is sufficient). ``d``
# is None whenever timing is OFF, which is the fast-path sentinel :func:`stage` checks.
_CUR = {"d": None, "t0": 0.0, "point": None}
# Cross-shot collection, summarized + cleared by :func:`scan_summary`.
_SHOTS = []
# A per-scan label stamped on every shot row (so several scans in one backend session -- e.g. an
# async-on vs async-off A/B -- separate cleanly in the CSV / analyzer). Set by the runner.
_SCAN = {"label": ""}
# Resolved lazily: the CSV path + whether its header has been written.
_CSV = {"path": None, "header_done": False, "resolved": False}
_LOG = {"fn": None}


# =========================================================================== #
# enable / config
# =========================================================================== #
def _truthy(v):
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def _log_dir():
    """``<project_root>/log/pyctrl_log`` (honoring ``YB_PYCTRL_LOG_DIR``)."""
    override = os.environ.get("YB_PYCTRL_LOG_DIR")
    if override:
        return override
    # __file__ = <root>/pyctrl/lib/run_timing.py -> three dirnames to the superproject root.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "log", "pyctrl_log")


def is_enabled():
    """True iff env ``YB_RUN_TIMING`` is truthy OR the ``RUN_TIMING_ON`` toggle file exists.

    Re-evaluated each call (cheap: an env read + at most one ``os.path.exists``) so a live
    backend can be switched by touching/deleting ``<log_dir>/RUN_TIMING_ON``. Fully
    swallows errors -> treated as disabled."""
    try:
        if _truthy(os.environ.get("YB_RUN_TIMING", "")):
            return True
        return os.path.exists(os.path.join(_log_dir(), "RUN_TIMING_ON"))
    except Exception:  # noqa: BLE001 - any probe failure -> disabled
        return False


def set_log(fn):
    """Install the run-loop ``log`` sink (e.g. the runner's ``print`` wrapper). Optional --
    when unset, summaries go to ``print``."""
    _LOG["fn"] = fn


def set_scan_label(label):
    """Stamp every subsequent shot row with ``label`` (e.g. ``"PushoutSurvival <id> async=1"``),
    so multiple scans in one backend session separate in the CSV / analyzer. Set per scan."""
    _SCAN["label"] = "" if label is None else str(label)


def log_dir():
    """The resolved pyctrl log dir (where the CSV + the RUN_TIMING_ON / async toggles live)."""
    return _log_dir()


def _emit(msg):
    fn = _LOG["fn"]
    try:
        (fn or print)(msg)
    except Exception:  # noqa: BLE001 - logging must never break a run
        pass


# =========================================================================== #
# per-shot API (called from run_seq.py / run_seq2.py / frame_capture.py)
# =========================================================================== #
@contextlib.contextmanager
def stage(name):
    """Time a top-level stage, accumulating into the active shot. No-op when timing is OFF.

    Accumulates (``+=``) so a stage entered more than once in a shot (the C.RESTART retry
    loop, or a per-bseq stage summed over multiple basic sequences) totals correctly."""
    cur = _CUR["d"]
    if cur is None:                       # fast path: timing OFF / no active shot
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cur[name] = cur.get(name, 0.0) + (time.perf_counter() - t0)


@contextlib.contextmanager
def substage(name):
    """Like :func:`stage` but the result is excluded from the accounted flat sum (it refines
    a top-level stage rather than adding to it). No-op when timing is OFF."""
    cur = _CUR["d"]
    if cur is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cur[name] = cur.get(name, 0.0) + (time.perf_counter() - t0)


def mark(name, value):
    """Record a non-timing scalar on the active shot (e.g. ``compiled=1``). No-op when OFF."""
    cur = _CUR["d"]
    if cur is not None:
        cur["#" + name] = value


def begin_shot(point=None):
    """Open a shot accumulator iff timing is enabled (re-evaluated here). ``point`` is the
    scan-point index, recorded for the CSV. When disabled, leaves the sentinel None so every
    :func:`stage` in the shot is a no-op."""
    if is_enabled():
        _CUR["d"] = {}
        _CUR["t0"] = time.perf_counter()
        _CUR["point"] = point
    else:
        _CUR["d"] = None


def end_shot():
    """Close the active shot: emit a log line + a CSV row, and stash it for the scan summary.
    No-op when no shot is active. Best-effort -- never raises into the run loop."""
    cur = _CUR["d"]
    if cur is None:
        return
    _CUR["d"] = None
    try:
        total = time.perf_counter() - _CUR["t0"]
        cur["#total"] = total
        cur["#point"] = _CUR["point"]
        cur["#scan"] = _SCAN["label"]
        accounted = sum(v for k, v in cur.items()
                        if not k.startswith("#") and k not in SUBSTAGES)
        cur["#other"] = total - accounted
        _SHOTS.append(cur)
        _emit(_shot_line(len(_SHOTS), cur, total))
        _csv_row(cur)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass


def reset_scan():
    """Drop any stashed shots (call at scan start so a prior aborted scan's data is cleared)."""
    _SHOTS.clear()
    _CUR["d"] = None


def scan_summary(label=None):
    """Log a mean/median/max per-stage summary across the scan's shots, then clear them.

    Called from ``run_scan_group``'s ``finally`` so it fires on the abort/error path too. A
    no-op when nothing was timed (timing off, or zero shots). Best-effort."""
    shots = list(_SHOTS)
    _SHOTS.clear()
    if not shots:
        return
    try:
        _emit(_summary_block(shots, label))
    except Exception:  # noqa: BLE001
        pass


# =========================================================================== #
# formatting helpers
# =========================================================================== #
def _ms(x):
    return x * 1e3


def _shot_line(idx, cur, total):
    pt = cur.get("#point")
    comp = " compile-MISS" if cur.get("#compiled") else ""
    parts = []
    for name in STAGES + SUBSTAGES:
        v = cur.get(name)
        if v:
            parts.append("%s=%.1f" % (name, _ms(v)))
    parts.append("other=%.1f" % _ms(cur.get("#other", 0.0)))
    return "[run_timing] shot %d pt=%s total=%.1fms%s | %s" % (
        idx, pt, _ms(total), comp, " ".join(parts))


def _summary_block(shots, label):
    n = len(shots)
    rows = []
    # Per-stage stats over the shots that have the stage non-zero (so "compile", which fires
    # on a minority of shots, reports its real cost, not a diluted mean). Also report how many
    # shots paid each stage.
    cols = STAGES + SUBSTAGES + ["other"]
    grand_total = sum(s.get("#total", 0.0) for s in shots)
    for name in cols:
        key = name if name != "other" else "#other"
        vals = [s.get(key, 0.0) for s in shots]
        nz = [v for v in vals if v > 0]
        if not nz:
            continue
        mean = sum(vals) / n                       # mean over ALL shots (its per-shot weight)
        med = _median(sorted(vals))
        mx = max(vals)
        hits = len(nz)
        pct = (sum(vals) / grand_total * 100.0) if grand_total else 0.0
        rows.append((name, mean, med, mx, hits, pct))
    rows.sort(key=lambda r: -r[1])                 # by mean-per-shot, descending
    total_mean = grand_total / n
    head = "[run_timing] SCAN SUMMARY%s -- %d shots, mean total %.1f ms/shot" % (
        (" (%s)" % label) if label else "", n, _ms(total_mean))
    lines = [head,
             "  %-14s %9s %9s %9s %7s %6s" % (
                 "stage", "mean_ms", "med_ms", "max_ms", "n_hit", "%tot")]
    for name, mean, med, mx, hits, pct in rows:
        lines.append("  %-14s %9.1f %9.1f %9.1f %7d %5.1f%%" % (
            name, _ms(mean), _ms(med), _ms(mx), hits, pct))
    return "\n".join(lines)


def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# =========================================================================== #
# CSV sink
# =========================================================================== #
def _csv_path():
    if not _CSV["resolved"]:
        _CSV["resolved"] = True
        try:
            d = _log_dir()
            os.makedirs(d, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            _CSV["path"] = os.path.join(d, "run_timing_%s.csv" % ts)
        except Exception:  # noqa: BLE001 - no CSV -> log-line-only
            _CSV["path"] = None
    return _CSV["path"]


def _csv_row(cur):
    path = _csv_path()
    if not path:
        return
    cols = ["scan", "point", "total_ms"] + ["%s_ms" % s for s in STAGES] + \
           ["%s_ms" % s for s in SUBSTAGES] + ["other_ms", "compiled"]
    try:
        with open(path, "a", encoding="utf-8") as fh:
            if not _CSV["header_done"]:
                fh.write(",".join(cols) + "\n")
                _CSV["header_done"] = True
            vals = [_csv_safe(cur.get("#scan", "")),
                    _fmt(cur.get("#point")),
                    "%.3f" % _ms(cur.get("#total", 0.0))]
            for s in STAGES + SUBSTAGES:
                vals.append("%.3f" % _ms(cur.get(s, 0.0)))
            vals.append("%.3f" % _ms(cur.get("#other", 0.0)))
            vals.append("1" if cur.get("#compiled") else "0")
            fh.write(",".join(vals) + "\n")
    except Exception:  # noqa: BLE001 - disk hiccup -> drop the row, keep running
        pass


def _fmt(v):
    return "" if v is None else str(v)


def _csv_safe(v):
    """A CSV-safe single field: drop commas/newlines from a free-text label."""
    return str(v).replace(",", " ").replace("\n", " ").replace("\r", " ")
