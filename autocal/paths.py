"""autocal.paths -- filesystem locations for the calibration ledger + journals.

Everything the auto-calibration system persists lives under
``<PATH_PREFIX>/yb_dashboard_state/calibration/`` -- the SAME ``yb_dashboard_state`` root the
pattern registry uses (``lib/pattern_grid.py``), so the dashboard (which already reads that root)
can read our files with no new configuration. ``PATH_PREFIX`` honours ``$YB_PATH_PREFIX`` exactly
like ``pattern_grid`` / ``scan_prep`` do; an explicit ``$YB_AUTOCAL_DIR`` overrides the whole dir
(used by tests to point at a tmp dir).

Files:
  * ``ledger.json``           -- the live state (per-pattern cals, settings, alerts, runtime).
  * ``changes.jsonl``         -- append-only audit/rollback journal of every applied config change.
  * ``commands.jsonl``        -- append-only command queue the dashboard writes and the controller
                                 consumes (rollback requests, setting toggles).
  * ``controller.log``        -- optional human-readable controller log (best-effort).
"""

import os

# Mirrors pattern_grid.DEFAULT_PATH_PREFIX / scan_prep.DEFAULT_DATA_PREFIX.
DEFAULT_PATH_PREFIX = r"D:\OneDrive - Harvard University\Documents - Yb"

LEDGER_NAME = "ledger.json"
CHANGES_NAME = "changes.jsonl"
COMMANDS_NAME = "commands.jsonl"
LOG_NAME = "controller.log"


def path_prefix():
    return os.environ.get("YB_PATH_PREFIX", DEFAULT_PATH_PREFIX)


def calibration_dir():
    """The directory holding all auto-calibration state. ``$YB_AUTOCAL_DIR`` overrides."""
    env = os.environ.get("YB_AUTOCAL_DIR")
    if env:
        return env
    return os.path.join(path_prefix(), "yb_dashboard_state", "calibration")


def ledger_path():
    return os.path.join(calibration_dir(), LEDGER_NAME)


def changes_path():
    return os.path.join(calibration_dir(), CHANGES_NAME)


def commands_path():
    return os.path.join(calibration_dir(), COMMANDS_NAME)


def log_path():
    return os.path.join(calibration_dir(), LOG_NAME)


def ensure_dir():
    """Create the calibration dir if absent. Best-effort: returns the path (raises only on a
    genuinely unwritable filesystem, which the caller should surface, not swallow)."""
    d = calibration_dir()
    os.makedirs(d, exist_ok=True)
    return d
