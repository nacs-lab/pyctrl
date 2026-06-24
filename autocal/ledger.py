"""autocal.ledger -- the auto-calibration state file: schema, atomic IO, cross-run accumulator.

The ledger is the SINGLE durable source of truth for the calibration system. The controller
reconstructs its entire world from it every tick (plus the live backend + scan data on disk), so
a context reset / restart / crash loses nothing operational -- the loop is idempotent and
catch-up-able. The dashboard reads the same file to render the Auto-Calibrations tab.

Top-level shape (``SCHEMA_VERSION`` = 1)::

    {
      "_version": 1,
      "updated": "<iso8601>",
      "settings": {                       # global lane controls (mirrored on the dashboard)
        "lane_enabled":  true,            # master on/off for the background lane
        "auto_apply":    true,            # apply in-band reversible updates vs propose-only
        "auto_cycle":    true,            # cycle non-home patterns when idle (default ON)
        "home_pattern":  "<name>",        # the pattern foreground always finds on the SLM
        "active_pattern":"<name>",        # what the controller last detected on the SLM
        "cycle_patterns":["<name>", ...]  # patterns eligible for the rotation
      },
      "patterns": { "<name>": <pattern_entry>, ... },
      "alerts":  [ <alert>, ... ],        # open items needing the user
      "runtime": {                        # controller liveness (for the dashboard)
        "controller_alive": bool, "last_tick_ts": <iso>, "lane_state": "<str>"
      }
    }

``<pattern_entry>``::

    { "eligible": bool, "eligibility_reasons": [..],
      "cals": { "<cal_key>": <cal_entry>, ... } }

``<cal_entry>``::

    { "cadence_s": int, "reps_per_run": int, "target_shots": int,
      "ref":      float|None,             # the currently-applied / reference value (Hz, or rate)
      "baseline": float|None,             # accepted baseline for the drift band
      "band":     float|None,             # |latest - baseline| > band  -> out-of-band alert
      "applies_config": bool,             # does an in-band fit auto-write config?
      "config_key":     str|None,         # which expConfig key (per-pattern), if it applies
      "last_run_ts":  <iso>|None,
      "last_fit":     <fit>|None,
      "history":      [ <fit>, ... ],     # capped at HISTORY_CAP
      "accumulator":  <accumulator>,      # cross-run pooling state (see below)
      "status":       "ok|drifting|out_of_band|insufficient|stale|nofit",
      "last_applied": <applied>|None }

``<accumulator>`` (inverse-variance running sums -- additive, so partial yielded runs pool
incrementally without keeping every scan's arrays)::

    { "open": bool, "x": [float],          # the swept grid (grid-checked on every add)
      "w0": [float], "w1": [float],        # per-point Sum(1/sem^2), Sum(mean/sem^2)
      "shots": int, "scan_ids": [str], "updated": <iso>|None }
"""

import json
import os
import tempfile

from . import paths as _paths
from . import pooling as _pooling

SCHEMA_VERSION = 1
HISTORY_CAP = 1000


# =========================================================================== #
# defaults / construction
# =========================================================================== #
def default_ledger(home_pattern=None):
    return {
        "_version": SCHEMA_VERSION,
        "updated": None,
        "settings": {
            "lane_enabled": True,
            "auto_apply": True,
            "auto_cycle": True,
            "home_pattern": home_pattern,
            "active_pattern": home_pattern,
            "cycle_patterns": [home_pattern] if home_pattern else [],
        },
        "patterns": {},
        "alerts": [],
        "runtime": {"controller_alive": False, "last_tick_ts": None, "lane_state": "idle"},
    }


def default_accumulator():
    return {"open": False, "x": [], "w0": [], "w1": [], "shots": 0,
            "scan_ids": [], "updated": None}


def default_cal_entry(cadence_s=3600, reps_per_run=6, target_shots=200,
                      ref=None, baseline=None, band=None,
                      applies_config=False, config_key=None):
    return {
        "cadence_s": int(cadence_s),
        "reps_per_run": int(reps_per_run),
        "target_shots": int(target_shots),
        "ref": ref,
        "baseline": baseline if baseline is not None else ref,
        "band": band,
        "applies_config": bool(applies_config),
        "config_key": config_key,
        "last_run_ts": None,
        "last_fit": None,
        "history": [],
        "accumulator": default_accumulator(),
        "status": "nofit",
        "last_applied": None,
    }


def ensure_pattern(ledger, name):
    """Return ``ledger['patterns'][name]``, creating an empty entry if absent."""
    pats = ledger.setdefault("patterns", {})
    if name not in pats:
        pats[name] = {"eligible": False, "eligibility_reasons": [], "cals": {}}
    return pats[name]


def ensure_cal(ledger, pattern, cal_key, **defaults):
    """Return the cal entry for (pattern, cal_key), creating it from ``defaults`` if absent."""
    pat = ensure_pattern(ledger, pattern)
    cals = pat.setdefault("cals", {})
    if cal_key not in cals:
        cals[cal_key] = default_cal_entry(**defaults)
    return cals[cal_key]


# =========================================================================== #
# atomic IO
# =========================================================================== #
def load(path=None):
    """Load the ledger, or return a fresh default if the file is missing/corrupt.

    A corrupt file is NEVER overwritten implicitly -- we return a default in memory and leave the
    bad file on disk for inspection (the next ``save`` will replace it once the controller has a
    valid state to write)."""
    path = path or _paths.ledger_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError):
        return default_ledger()
    except (ValueError, json.JSONDecodeError):
        return default_ledger()
    if not isinstance(data, dict):
        return default_ledger()
    return _migrate(data)


def save(ledger, path=None, stamp=None):
    """Atomically write the ledger (temp file + ``os.replace``). ``stamp`` sets ``updated``.

    The atomic replace is what makes the dashboard's concurrent reads safe -- it always sees a
    whole file, never a half-written one. ``stamp`` is passed in (the engine forbids
    ``Date.now``-style calls in some contexts; callers own the clock)."""
    path = path or _paths.ledger_path()
    if stamp is not None:
        ledger["updated"] = stamp
    ledger["_version"] = SCHEMA_VERSION
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ledger-", suffix=".json", dir=d or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=True, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def _migrate(data):
    """Forward-compat shim. v1 is the first version; just backfill any missing top-level keys."""
    base = default_ledger()
    for k, v in base.items():
        data.setdefault(k, v)
    data["_version"] = SCHEMA_VERSION
    return data


# =========================================================================== #
# cross-run accumulator (the heart of "be aware it gets interrupted a lot")
# =========================================================================== #
GRID_RTOL = 1e-6


def _grid_matches(x_a, x_b, rtol=GRID_RTOL):
    if len(x_a) != len(x_b):
        return False
    for a, b in zip(x_a, x_b):
        scale = max(abs(a), abs(b), 1.0)
        if abs(a - b) > rtol * scale:
            return False
    return True


def accumulate(acc, x, mean, sem, shots, scan_id, *, min_sem=_pooling.DEFAULT_MIN_SEM,
               stamp=None):
    """Fold one completed (possibly very short) scan into the accumulator. Returns ``(acc, info)``.

    Pooling is inverse-variance and additive: each point contributes ``w0 += 1/sem^2`` and
    ``w1 += mean/sem^2`` (sem floored at ``min_sem``; non-finite / zero-shot points contribute
    nothing). So a scan that was yielded after 4 shots adds its little bit of weight and the next
    partial run adds more -- exactly the "combine scan data across interruptions" requirement.

    Grid handling: the first scan sets the swept grid ``x``; a later scan whose grid matches is
    pooled in; a MISMATCH (the cal definition changed) RESETS the accumulator to the new scan
    rather than mixing incompatible sweeps. ``info`` reports what happened
    (``added`` | ``reset`` | ``empty``)."""
    w0_c, w1_c = _pooling.scan_point_weights(mean, sem, min_sem=min_sem)
    if not w0_c:
        return acc, {"action": "empty", "shots": acc.get("shots", 0)}

    x = [float(v) for v in x]
    if not acc.get("open") or not acc.get("x"):
        action = "reset" if acc.get("scan_ids") else "open"
        acc = {"open": True, "x": x, "w0": list(w0_c), "w1": list(w1_c),
               "shots": int(shots), "scan_ids": [scan_id], "updated": stamp}
        return acc, {"action": action, "shots": acc["shots"]}

    if not _grid_matches(acc["x"], x):
        # The scan grid changed under us -> start a clean accumulation on the new grid.
        acc = {"open": True, "x": x, "w0": list(w0_c), "w1": list(w1_c),
               "shots": int(shots), "scan_ids": [scan_id], "updated": stamp}
        return acc, {"action": "reset", "shots": acc["shots"]}

    acc["w0"] = [a + b for a, b in zip(acc["w0"], w0_c)]
    acc["w1"] = [a + b for a, b in zip(acc["w1"], w1_c)]
    acc["shots"] = int(acc.get("shots", 0)) + int(shots)
    if scan_id not in acc["scan_ids"]:
        acc["scan_ids"].append(scan_id)
    acc["updated"] = stamp
    return acc, {"action": "added", "shots": acc["shots"]}


def accumulator_curve(acc):
    """Read out the pooled ``(x, mean, sem)`` from the accumulator's running sums."""
    if not acc.get("x"):
        return [], [], []
    mean, sem = _pooling.combine_sums(acc["w0"], acc["w1"])
    return list(acc["x"]), mean, sem


def accumulator_ready(acc, target_shots):
    return int(acc.get("shots", 0)) >= int(target_shots)


def reset_accumulator(cal_entry):
    cal_entry["accumulator"] = default_accumulator()
    return cal_entry


# =========================================================================== #
# history + status
# =========================================================================== #
def append_history(cal_entry, fit, cap=HISTORY_CAP):
    hist = cal_entry.setdefault("history", [])
    hist.append(fit)
    if len(hist) > cap:
        del hist[: len(hist) - cap]
    cal_entry["last_fit"] = fit
    return cal_entry


def evaluate_status(cal_entry):
    """Classify a cal from its latest fit + accumulator -> a status string (does not mutate band).

      * ``nofit``        -- never fit.
      * ``insufficient`` -- accumulating, not yet at target_shots.
      * ``out_of_band``  -- |last_fit.center - baseline| > band  (needs the user).
      * ``drifting``     -- moved > band/2 from baseline but still in band (auto-applied if enabled).
      * ``ok``           -- within band/2 of baseline.
    """
    fit = cal_entry.get("last_fit")
    if not fit or fit.get("center") is None:
        acc = cal_entry.get("accumulator", {})
        if acc.get("shots"):
            return "insufficient"
        return "nofit"
    baseline = cal_entry.get("baseline")
    band = cal_entry.get("band")
    if baseline is None or band is None or not band:
        return "ok"
    delta = abs(fit["center"] - baseline)
    if delta > band:
        return "out_of_band"
    if delta > band / 2.0:
        return "drifting"
    return "ok"


# =========================================================================== #
# alerts
# =========================================================================== #
def add_alert(ledger, *, alert_id, severity, pattern, cal, message, action_needed, stamp=None):
    """Append an alert if one with the same ``alert_id`` is not already open (dedup)."""
    alerts = ledger.setdefault("alerts", [])
    for a in alerts:
        if a.get("id") == alert_id:
            a["count"] = int(a.get("count", 1)) + 1
            a["last_ts"] = stamp
            return ledger
    alerts.append({"id": alert_id, "ts": stamp, "last_ts": stamp, "severity": severity,
                   "pattern": pattern, "cal": cal, "message": message,
                   "action_needed": action_needed, "count": 1})
    return ledger


def clear_alert(ledger, alert_id):
    alerts = ledger.get("alerts", [])
    ledger["alerts"] = [a for a in alerts if a.get("id") != alert_id]
    return ledger
