"""autocal.controller -- the deterministic auto-calibration controller.

Two layers, deliberately separated so the dangerous half is small and the smart half is testable:

  * PURE decision logic (no IO, no clock of its own -- timestamps/ids passed in): ``ingest_scan``
    folds a completed scan into the accumulator and, once enough shots have pooled, fits and
    decides apply-vs-flag; ``process_command`` turns a dashboard command into ledger mutations +
    config-write intents; ``plan_submissions`` decides what to enqueue next (home enrollment +
    auto-cycle episodes). These are unit-tested with no engine/backend/hardware.

  * The ``Controller`` class wires those to injected IO adapters (ZMQ queue/submit, SLM
    switch/restore-home, the expConfig writer + oracle re-capture, the analyze_scan reader). Its
    ``tick()`` is the loop body. The IO adapters are the ONLY hardware/config-touching code and are
    wired + validated on-rig in the later phases; in Phase 0 they are injected stubs.

Apply policy (matches the chosen autonomy: "auto-apply in-band, flag the rest"):
  * a cal that ``applies_config`` + global ``auto_apply`` on + fit is clean (R2 >= min_r2, not
    edge-pinned) + status in {ok, drifting}  ->  WRITE the per-pattern config + journal it.
  * out-of-band, edge-pinned, poor fit, or a reference-only cal  ->  no write; raise an alert / keep
    the trend. The user resolves out-of-band via the dashboard (accept-baseline / rollback / manual).
"""

from datetime import datetime

from . import ledger as _ledger
from . import rollback as _rollback
from . import rotation as _rotation
from . import fit as _fit
from . import submit as _submit


# =========================================================================== #
# small time helpers (injectable clock; iso strings on disk)
# =========================================================================== #
def now_iso(clock=None):
    return (clock or datetime.now)().isoformat(timespec="seconds")


def iso_to_epoch(s):
    if s is None:
        return 0.0
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return 0.0


# =========================================================================== #
# PURE: ingest one completed calibration scan
# =========================================================================== #
def ingest_scan(ledger, *, pattern, cal_key, cal_def, x, mean, sem, n_shots, scan_id,
                stamp, fitter=None, auto_apply=True, loading_mean=None):
    """Fold a completed (possibly very short) cal scan into the ledger; fit when ready. PURE.

    Mutates only the cal's accumulator/history/status/last_run_ts (and pattern loading health).
    Returns ``(ledger, outcome)``; the caller executes ``outcome['apply']`` /
    ``outcome['alert']`` (config write + journal / add alert) -- they are returned as intents so
    no config/hardware write happens in this pure function."""
    fitter = fitter or _fit.fit_spectrum
    ce = _ledger.ensure_cal(
        ledger, pattern, cal_key,
        cadence_s=cal_def["cadence_s"], reps_per_run=cal_def["reps_per_run"],
        target_shots=cal_def["target_shots"], band=cal_def["band"],
        applies_config=cal_def["applies_config"], config_key=cal_def["config_key"])
    ce["last_run_ts"] = stamp

    out = {"action": "accumulated", "pattern": pattern, "cal": cal_key, "fit": None,
           "status": None, "apply": None, "alert": None, "progress": None}

    # pattern-level loading health (derived from the cal scans we already run)
    if loading_mean is not None:
        pat = _ledger.ensure_pattern(ledger, pattern)
        pat["loading_mean"] = float(loading_mean)
        pat["loading_ts"] = stamp
        if loading_mean < _rotation.LOADING_MIN_RATE:
            out["alert"] = {"id": "loading::%s" % pattern, "severity": "warn", "pattern": pattern,
                            "cal": _rotation.LOADING_KEY,
                            "message": "loading rate %.3f below %.2f -- SLM/pattern problem?"
                                       % (loading_mean, _rotation.LOADING_MIN_RATE),
                            "action_needed": "check the SLM pattern / loading"}

    acc = ce.get("accumulator") or _ledger.default_accumulator()
    acc, info = _ledger.accumulate(acc, x, mean, sem, n_shots, scan_id, stamp=stamp)
    ce["accumulator"] = acc
    out["progress"] = {"shots": acc.get("shots", 0), "target": ce.get("target_shots")}

    if not _ledger.accumulator_ready(acc, ce.get("target_shots", 200)):
        ce["status"] = "insufficient"
        out["status"] = "insufficient"
        return ledger, out

    # enough shots pooled -> fit the pooled curve
    xs, ym, ys = _ledger.accumulator_curve(acc)
    fit = fitter(xs, ym, ys, mode=cal_def.get("fit_mode", "dip"))
    if not fit:
        ce["status"] = "nofit"
        out["action"] = "fit_failed"
        out["status"] = "nofit"
        out["alert"] = {"id": "fitfail::%s::%s" % (pattern, cal_key), "severity": "warn",
                        "pattern": pattern, "cal": cal_key,
                        "message": "pooled fit failed at %d shots (signal/window?)"
                                   % acc.get("shots", 0),
                        "action_needed": "inspect the cal scan / sweep window"}
        ce["accumulator"] = _ledger.default_accumulator()  # reset; don't pool a bad window forever
        return ledger, out

    fitrec = {"ts": stamp, "center": fit["center"], "fwhm": fit["fwhm"], "r2": fit["r2"],
              "edge_pinned": fit["edge_pinned"], "n_shots": acc.get("shots", 0),
              "scan_ids": list(acc.get("scan_ids", [])), "applied": False}
    _ledger.append_history(ce, fitrec)
    status = _ledger.evaluate_status(ce)
    ce["status"] = status
    out.update({"action": "fit", "fit": fit, "status": status})

    apply, alert = _decide(ce, cal_def, fit, status, auto_apply, pattern, cal_key, scan_id)
    if apply is not None:
        out["apply"] = apply
    if alert is not None and out["alert"] is None:  # don't clobber a loading alert
        out["alert"] = alert

    ce["accumulator"] = _ledger.default_accumulator()  # fresh accumulation for the next cycle
    return ledger, out


def _decide(ce, cal_def, fit, status, auto_apply, pattern, cal_key, scan_id):
    """Decide (apply-intent, alert-intent) for a fresh fit. Returns (apply|None, alert|None)."""
    min_r2 = cal_def.get("min_r2", 0.85)
    if fit["edge_pinned"]:
        return None, {"id": "edge::%s::%s" % (pattern, cal_key), "severity": "warn",
                      "pattern": pattern, "cal": cal_key,
                      "message": "fit center pinned at the sweep edge (%.4f MHz) -- the line moved "
                                 "out of the window; recentre the sweep." % (fit["center"] / 1e6),
                      "action_needed": "recentre the sweep window (user)"}
    if fit["r2"] < min_r2:
        return None, {"id": "lowr2::%s::%s" % (pattern, cal_key), "severity": "info",
                      "pattern": pattern, "cal": cal_key,
                      "message": "fit R2 %.3f below %.2f -- not applied; trending only."
                                 % (fit["r2"], min_r2),
                      "action_needed": None}
    if status == "out_of_band":
        return None, {"id": "oob::%s::%s" % (pattern, cal_key), "severity": "warn",
                      "pattern": pattern, "cal": cal_key,
                      "message": "%s drift %.1f kHz exceeds band %.1f kHz (center %.4f MHz) -- NOT "
                                 "auto-applied; needs you." % (
                                     cal_key, abs(fit["center"] - (ce.get("baseline") or 0)) / 1e3,
                                     (ce.get("band") or 0) / 1e3, fit["center"] / 1e6),
                      "action_needed": "review + accept-baseline / apply manually / investigate"}
    if cal_def.get("applies_config") and auto_apply and status in ("ok", "drifting"):
        return ({"config_key": cal_def["config_key"], "pattern": pattern, "cal": cal_key,
                 "old": ce.get("ref"), "new": fit["center"], "scan_id": scan_id}, None)
    return None, None


def commit_apply(ledger, apply_intent, *, change_id, stamp):
    """After the controller has written config successfully: update the cal's ref/last_applied and
    mark the latest history record applied. PURE (no IO; the caller already wrote + journaled)."""
    p, c = apply_intent["pattern"], apply_intent["cal"]
    ce = ledger.get("patterns", {}).get(p, {}).get("cals", {}).get(c)
    if not ce:
        return ledger
    ce["ref"] = apply_intent["new"]
    ce["last_applied"] = {"ts": stamp, "old": apply_intent["old"], "new": apply_intent["new"],
                          "scan_id": apply_intent.get("scan_id"), "change_id": change_id}
    if ce.get("history"):
        ce["history"][-1]["applied"] = True
    return ledger


# =========================================================================== #
# PURE: interpret a dashboard command
# =========================================================================== #
def process_command(ledger, cmd, *, changes_lookup=None, stamp=None):
    """Turn one dashboard command into ledger mutations + optional config-write intents.

    Returns ``(ledger, intents)``; ``intents`` is a list of ``{kind:'revert', ...}`` config writes
    the controller must execute (the dashboard never writes config itself). Unknown/invalid
    commands are ignored (returned as a 'noop' note) -- a bad command never breaks the loop."""
    intents = []
    t = cmd.get("type")
    a = cmd.get("args") or {}
    s = ledger.setdefault("settings", {})
    try:
        if t == "set_setting":
            if a.get("key") in ("lane_enabled", "auto_apply", "auto_cycle", "home_pattern",
                                "active_pattern", "cycle_patterns"):
                s[a["key"]] = a["value"]
        elif t == "set_baseline":
            ce = _cal(ledger, a["pattern"], a["cal"])
            if ce is not None:
                ce["baseline"] = a.get("value", (ce.get("last_fit") or {}).get("center"))
        elif t == "set_cal_field":
            ce = _cal(ledger, a["pattern"], a["cal"])
            if ce is not None and a.get("field") in ("cadence_s", "band", "reps_per_run",
                                                     "target_shots", "applies_config"):
                ce[a["field"]] = a["value"]
        elif t == "ack_alert":
            _ledger.clear_alert(ledger, a.get("alert_id"))
        elif t == "rerun_cal":
            ce = _cal(ledger, a["pattern"], a["cal"])
            if ce is not None:
                ce["last_run_ts"] = None  # mark overdue so the scheduler re-enrolls it
        elif t == "rollback":
            orig = (changes_lookup or (lambda _id: None))(a.get("change_id"))
            if orig is not None:
                intents.append({"kind": "revert", "orig": orig})
        elif t in ("add_cycle_pattern", "remove_cycle_pattern"):
            cps = s.setdefault("cycle_patterns", [])
            name = a.get("pattern")
            if t == "add_cycle_pattern" and name and name not in cps:
                cps.append(name)
            if t == "remove_cycle_pattern" and name in cps:
                cps.remove(name)
        else:
            return ledger, [{"kind": "noop", "reason": "unknown command %r" % t}]
    except (KeyError, TypeError) as e:
        return ledger, [{"kind": "noop", "reason": "bad command args: %s" % e}]
    return ledger, intents


def _cal(ledger, pattern, cal):
    return ledger.get("patterns", {}).get(pattern, {}).get("cals", {}).get(cal)


# =========================================================================== #
# PURE: what to submit next (home enrollment + auto-cycle episode)
# =========================================================================== #
def plan_submissions(ledger, now_stamp, *, foreground_idle, already_queued, to_epoch=iso_to_epoch):
    """Decide the next background submission. Returns a list of plan dicts (0 or 1 here)::

        {"pattern", "cal_key", "requires_switch", "restore_home_after", "cal_def"}

    Wraps ``rotation.select_next`` and attaches the static cal_def the submit layer needs. Empty
    when nothing is due / the lane is disabled / a foreground scan is running."""
    plan = _rotation.select_next(ledger, now_stamp, to_epoch, foreground_idle=foreground_idle,
                                 already_queued=already_queued)
    if not plan:
        return []
    cal_def = dict(_rotation.CAL_DEFS.get(plan["cal_key"], {}))
    plan["cal_def"] = cal_def
    return [plan]


# =========================================================================== #
# The Controller (IO wiring; adapters injected -- live phases wire real ones)
# =========================================================================== #
class Controller:
    """Composes the pure logic with injected IO adapters. ``tick()`` is one loop iteration.

    Required adapters (all keyword, all injectable for tests / staged bring-up):
      * ``query_backend()``  -> snapshot dict: {'foreground_idle':bool, 'completed':[{scan_id,label,
        n_shots?}], 'queued_labels':[..], 'lane_enabled':bool}
      * ``read_scan(scan_id)`` -> {'x','mean','sem','n_shots','loading_mean'} (pooling.scan_to_points
        + a loading read) or None
      * ``submit(plan)`` -> submit a background descriptor (IO; live only)
      * ``write_config(pattern, key, value)`` -> write the per-pattern config + re-capture oracle
        (IO; live only) -> returns True on success
      * ``switch_pattern(pattern)`` / ``restore_home(home)`` -> SLM writes (IO; live only)
      * ``clock()`` -> a datetime (defaults to datetime.now)

    Phase 0 leaves the IO adapters as injected stubs; this class is exercised by the tests with
    fakes. NOTHING here runs against the live rig until the adapters are the real ones AND the user
    has greenlit the phase.
    """

    def __init__(self, *, query_backend, read_scan, submit=None, write_config=None,
                 switch_pattern=None, restore_home=None, clock=None,
                 ledger_path=None, changes_path=None, commands_path=None):
        self.query_backend = query_backend
        self.read_scan = read_scan
        self._submit = submit
        self._write_config = write_config
        self._switch_pattern = switch_pattern
        self._restore_home = restore_home
        self.clock = clock or datetime.now
        self._ledger_path = ledger_path
        self._changes_path = changes_path
        self._commands_path = commands_path

    # -- helpers ----------------------------------------------------------- #
    def _now(self):
        return now_iso(self.clock)

    def _next_change_id(self, ledger):
        n = int(ledger.setdefault("runtime", {}).get("change_seq", 0)) + 1
        ledger["runtime"]["change_seq"] = n
        return "chg-%06d" % n

    # -- one iteration ----------------------------------------------------- #
    def tick(self, ledger=None):
        """Run one controller iteration. Returns the (mutated, persisted) ledger.

        Order: load -> snapshot backend -> CATCH UP (ingest every newly-completed cal scan, execute
        apply intents) -> process dashboard commands -> plan + submit next -> stamp runtime -> save.
        Every step is wrapped so one failure (a bad scan read, a transient IO error) degrades to a
        logged note instead of crashing the loop -- the next tick retries."""
        stamp = self._now()
        if ledger is None:
            ledger = _ledger.load(self._ledger_path)
        snap = self._safe(self.query_backend) or {}
        ledger.setdefault("runtime", {})

        # CATCH UP: ingest completed background cal scans we haven't processed yet.
        processed = set(ledger["runtime"].get("ingested_scan_ids", []))
        for job in snap.get("completed", []):
            sid = str(job.get("scan_id"))
            parsed = _submit.parse_label(job.get("label"))
            if not parsed or sid in processed:
                continue
            cal_key, pattern = parsed
            cal_def = _rotation.CAL_DEFS.get(cal_key)
            if not cal_def:
                continue
            pts = self._safe(lambda: self.read_scan(sid))
            if not pts or not pts.get("x"):
                continue
            ledger, out = ingest_scan(
                ledger, pattern=pattern, cal_key=cal_key, cal_def=cal_def,
                x=pts["x"], mean=pts["mean"], sem=pts["sem"], n_shots=pts.get("n_shots", 0),
                scan_id=sid, stamp=stamp,
                auto_apply=ledger.get("settings", {}).get("auto_apply", True),
                loading_mean=pts.get("loading_mean"))
            processed.add(sid)
            self._handle_outcome(ledger, out, stamp)
        ledger["runtime"]["ingested_scan_ids"] = list(processed)[-2000:]

        # dashboard commands
        self._consume_commands(ledger, stamp)

        # plan + submit
        if ledger.get("settings", {}).get("lane_enabled", True) and self._submit is not None:
            plans = plan_submissions(
                ledger, stamp, foreground_idle=snap.get("foreground_idle", True),
                already_queued=_queued_pairs(snap))
            for plan in plans:
                self._do_submit(ledger, plan, stamp)

        ledger["runtime"]["last_tick_ts"] = stamp
        ledger["runtime"]["controller_alive"] = True
        _ledger.save(ledger, self._ledger_path, stamp=stamp)
        return ledger

    # -- outcome / command execution (IO) ---------------------------------- #
    def _handle_outcome(self, ledger, out, stamp):
        alert = out.get("alert")
        if alert:
            _ledger.add_alert(ledger, alert_id=alert["id"], severity=alert["severity"],
                              pattern=alert["pattern"], cal=alert["cal"], message=alert["message"],
                              action_needed=alert.get("action_needed"), stamp=stamp)
        apply = out.get("apply")
        if apply and self._write_config is not None:
            ok = self._safe(lambda: self._write_config(apply["pattern"], apply["config_key"],
                                                        apply["new"]))
            if ok:
                cid = self._next_change_id(ledger)
                _rollback.record_change(change_id=cid, pattern=apply["pattern"], cal=apply["cal"],
                                        config_key=apply["config_key"], old=apply["old"],
                                        new=apply["new"], scan_id=apply.get("scan_id"),
                                        kind="apply", ts=stamp, path=self._changes_path)
                commit_apply(ledger, apply, change_id=cid, stamp=stamp)

    def _consume_commands(self, ledger, stamp):
        processed = int(ledger["runtime"].get("commands_processed", 0))
        pending = _rollback.pending_commands(processed, self._commands_path)
        if not pending:
            return
        changes = _rollback.list_changes(self._changes_path)
        by_id = {c.get("change_id"): c for c in changes}
        for cmd in pending:
            ledger, intents = process_command(ledger, cmd, changes_lookup=by_id.get, stamp=stamp)
            for it in intents:
                if it.get("kind") == "revert" and self._write_config is not None:
                    orig = it["orig"]
                    ok = self._safe(lambda: self._write_config(orig["pattern"], orig["config_key"],
                                                               orig["old"]))
                    if ok:
                        cid = self._next_change_id(ledger)
                        rev = _rollback.build_revert(orig, change_id=cid, ts=stamp)
                        _rollback.record_change(**{k: rev[k] for k in (
                            "change_id", "pattern", "cal", "config_key", "old", "new", "scan_id")},
                            kind="revert", reverts=rev.get("reverts"), note=rev.get("note"),
                            ts=stamp, path=self._changes_path)
                        ce = _cal(ledger, orig["pattern"], orig["cal"])
                        if ce is not None:
                            ce["ref"] = orig["old"]
        ledger["runtime"]["commands_processed"] = processed + len(pending)

    def _do_submit(self, ledger, plan, stamp):
        # The home-pattern invariant + restore-home around a pattern switch is enforced in the IO
        # adapters (switch_pattern / restore_home), which are wired + validated on-rig in the
        # auto-cycle phase. Here we just submit the cal and record what was queued.
        if plan.get("requires_switch") and self._switch_pattern is not None:
            if not self._safe(lambda: self._switch_pattern(plan["pattern"])):
                return
        self._safe(lambda: self._submit(plan))
        ledger["runtime"].setdefault("last_submitted", {})[
            "%s::%s" % (plan["pattern"], plan["cal_key"])] = stamp

    # -- robustness -------------------------------------------------------- #
    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - a single failed IO/read must not break the loop
            return None


def _queued_pairs(snap):
    pairs = []
    for lbl in snap.get("queued_labels", []):
        parsed = _submit.parse_label(lbl)
        if parsed:
            pairs.append((parsed[1], parsed[0]))  # (pattern, cal_key)
    return pairs
