"""autocal.rotation -- the calibration menu, per-cal cadence/bands, eligibility, and scheduling.

WHAT to keep a fixed pattern in calibration (the most important measure-only cals, from the
experiment-running runbooks), and WHEN to run each. The rotation is data-driven: ``CAL_DEFS`` is
the default menu, copied into a pattern's ledger entry when it joins the rotation, after which the
agent may retune cadence/reps/bands or add/remove cals per its judgment.

Autonomy boundary baked into the defaults (matches the runbooks + "confirm hard-to-reverse"):
  * ``556mj0`` AUTO-APPLIES the per-pattern mj=0 resonance (small, reversible, in-band; the daily
    calibration line). Everything else is reference/trend only (``applies_config=False``).
  * Anything that would rewrite the hologram (trap-depth feedback), cooling/imaging, or laser power
    is NOT in this menu -- those stay user-gated.
"""

from . import ledger as _ledger

# Each entry: the submittable scan + how to read/judge it. ``module``/``seq``/``build_kwargs`` are
# consumed by autocal.submit; ``fit_mode``/``band``/``applies_config``/``config_key`` by the
# controller. Bands are starting defaults (the agent tunes them per pattern).
CAL_DEFS = {
    "556mj0": {
        "label": "556 mj=0 resonance",
        "module": "Spectrum556Scan",
        "seq": "PushoutSurvivalSeq",
        "build_kwargs": {"mj": 0},
        "fit_mode": "dip",
        "reps_per_run": 6,
        "target_shots": 200,
        "cadence_s": 3600,          # ~1 h: the calibration line, highest priority
        "band": 50e3,               # 50 kHz; few-kHz drift is in-band (ULE linewidth)
        "min_r2": 0.93,             # auto-apply requires a clean fit (runbook: good mj=0 R2 >= 0.95)
        "applies_config": True,
        "config_key": "Resonance556mj0Freq",
        "ref_key": "Resonance556mj0Freq",
        "desc": "556 mj=0 push-out resonance; the daily calibration line (auto-applied in-band).",
    },
    "556mj1": {
        "label": "556 |mj|=1 (trap depth)",
        "module": "Spectrum556Scan",
        "seq": "PushoutSurvivalSeq",
        "build_kwargs": {"mj": 1},
        "fit_mode": "dip",
        "reps_per_run": 7,
        "target_shots": 200,
        "cadence_s": 7200,          # ~2 h: trap-depth / light-shift trend
        "band": 200e3,
        "min_r2": 0.85,
        "applies_config": False,    # trend only; a real shift -> flag (trap-depth feedback is user-gated)
        "config_key": None,
        "ref_key": None,
        "desc": "556 |mj|=1 push-out; trap-depth / light-shift trend + splitting (reference only).",
    },
    "399": {
        "label": "399 imaging reference",
        "module": "Spectrum399Scan",
        "seq": "PushoutSurvivalSeq",
        "build_kwargs": {},
        "fit_mode": "dip",
        "reps_per_run": 3,
        "target_shots": 100,
        "cadence_s": 21600,         # ~6 h: imaging-resonance trend (reference only)
        "band": 5e6,
        "min_r2": 0.85,
        "applies_config": False,
        "config_key": None,
        "ref_key": "Resonance399Freq",
        "desc": "399 1S0->1P1 imaging line; reference/trend only (drifts with trap depth).",
    },
}

# A pattern carries a derived "loading" health metric (loading rate / survival from the cal scans
# it already runs) -- tracked per pattern, NOT a separate scan. The controller updates it on ingest.
LOADING_KEY = "loading"
LOADING_MIN_RATE = 0.15   # below this -> a loading-health alert (SLM/pattern problem)


# =========================================================================== #
# seed a pattern's cal menu into the ledger
# =========================================================================== #
def seed_pattern_cals(ledger, pattern, ref_lookup=None):
    """Ensure ``pattern`` has the default cal menu. ``ref_lookup(config_key) -> value`` seeds each
    cal's ref/baseline from current config (so the first fit is judged against the live value)."""
    for cal_key, d in CAL_DEFS.items():
        ref = None
        if ref_lookup is not None and d.get("ref_key"):
            try:
                ref = ref_lookup(d["ref_key"])
            except Exception:  # noqa: BLE001
                ref = None
        _ledger.ensure_cal(
            ledger, pattern, cal_key,
            cadence_s=d["cadence_s"], reps_per_run=d["reps_per_run"],
            target_shots=d["target_shots"], ref=ref, baseline=ref, band=d["band"],
            applies_config=d["applies_config"], config_key=d["config_key"])
    return ledger


# =========================================================================== #
# eligibility (per-pattern wiring must exist before we calibrate / cycle a pattern)
# =========================================================================== #
def check_eligibility(pattern, checker=None):
    """Return ``(eligible, reasons)``. A pattern is eligible only if its per-pattern wiring is
    present (registry record + thresholds), else calibrating/switching to it is unsafe (the
    runbook's #1 foot-gun). ``checker`` is injected in tests; the default reads the registry."""
    if checker is None:
        checker = _default_wiring_checker
    try:
        info = checker(pattern)
    except Exception as e:  # noqa: BLE001
        return False, ["wiring check failed: %s" % e]
    reasons = []
    if not info.get("record"):
        reasons.append("no registry record.json (grid unknown)")
    if not info.get("threshold"):
        reasons.append("no per-pattern threshold.mat (detection would use day-folder cuts)")
    return (not reasons), reasons


def _default_wiring_checker(pattern):
    """Read the pattern registry to check wiring presence (no yb_analysis import)."""
    import os
    from pattern_grid import _record_path, _pattern_threshold_path  # type: ignore
    return {"record": os.path.exists(_record_path(pattern)),
            "threshold": os.path.exists(_pattern_threshold_path(pattern))}


# =========================================================================== #
# scheduler: what to run next
# =========================================================================== #
def overdue_factor(cal_entry, now_ts, to_epoch):
    """``(now - last_run) / cadence``; >1 means overdue. Never-run -> +inf (run it first)."""
    last = cal_entry.get("last_run_ts")
    cad = cal_entry.get("cadence_s") or 3600
    if last is None:
        return float("inf")
    elapsed = to_epoch(now_ts) - to_epoch(last)
    return elapsed / float(cad)


def select_next(ledger, now_ts, to_epoch, *, foreground_idle=True, already_queued=None):
    """Pick the next calibration to enqueue, honouring the home-pattern invariant.

    Returns a plan dict ``{pattern, cal_key, requires_switch, restore_home_after}`` or ``None``.

    Policy:
      * Always prefer the ACTIVE (== home, normally) pattern's most-overdue eligible cal that is
        not already queued -- no SLM switch needed.
      * Only if every active-pattern cal is fresh (overdue<1) AND ``auto_cycle`` is on AND the rig
        is idle: pick the most-overdue eligible cal of the most-overdue NON-home cycle pattern,
        flagged ``requires_switch`` + ``restore_home_after`` (the controller does the switch + the
        mandatory restore-home; that half is gated to the on-rig phases).
    """
    s = ledger.get("settings", {})
    if not s.get("lane_enabled", True):
        return None
    already_queued = set(already_queued or [])
    active = s.get("active_pattern") or s.get("home_pattern")
    home = s.get("home_pattern")

    def _best_cal(pattern):
        pat = ledger.get("patterns", {}).get(pattern)
        if not pat or not pat.get("eligible", False):
            return None
        best = None
        for cal_key, ce in pat.get("cals", {}).items():
            if (pattern, cal_key) in already_queued:
                continue
            f = overdue_factor(ce, now_ts, to_epoch)
            if best is None or f > best[1]:
                best = (cal_key, f)
        return best

    # 1) the active pattern, most-overdue cal
    if active is not None:
        b = _best_cal(active)
        if b and b[1] >= 1.0:
            return {"pattern": active, "cal_key": b[0], "requires_switch": False,
                    "restore_home_after": False}

    # 2) auto-cycle a non-home pattern only when idle + everything on home is fresh
    if foreground_idle and s.get("auto_cycle", True):
        candidates = []
        for pattern in s.get("cycle_patterns", []):
            if pattern == home or pattern == active:
                continue
            b = _best_cal(pattern)
            if b:
                candidates.append((pattern, b[0], b[1]))
        if candidates:
            candidates.sort(key=lambda c: c[2], reverse=True)
            pattern, cal_key, f = candidates[0]
            if f >= 1.0:
                return {"pattern": pattern, "cal_key": cal_key, "requires_switch": True,
                        "restore_home_after": True}

    # 3) fall through: if the active pattern has anything not-yet-run (factor inf), run it even if
    #    others are fresh (covers the very first pass where last_run_ts is None everywhere).
    if active is not None:
        b = _best_cal(active)
        if b and b[1] == float("inf"):
            return {"pattern": active, "cal_key": b[0], "requires_switch": False,
                    "restore_home_after": False}
    return None
