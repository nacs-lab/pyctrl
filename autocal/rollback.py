"""autocal.rollback -- append-only change journal (audit + one-click rollback) + command queue.

Two append-only JSONL files (append-only so concurrent dashboard reads never see a half state and
a crash can't corrupt prior records):

  * ``changes.jsonl`` -- every config change the controller applies, AND every revert. The "current"
    value of any (pattern, cal, key) is the last event for it; whether a given change was rolled
    back is "does a later ``revert`` event reference its id". This is the "thorough log + history +
    easy rollback" the dashboard surfaces.
  * ``commands.jsonl`` -- requests the DASHBOARD writes for the controller to execute (rollback a
    change, toggle a setting, accept a baseline, ack an alert). The dashboard never writes config
    itself; it only enqueues here, and the controller -- the single writer of config -- consumes
    (tracking how many lines it has processed in ``ledger['runtime']['commands_processed']``).

Records are plain dicts; ids/timestamps are passed in by the caller (the controller owns the clock,
and ids are content/seq based) so this module needs no wall-clock of its own.
"""

import json
import os
import time

from . import paths as _paths


# =========================================================================== #
# low-level append (resilient to a transient Windows/OneDrive lock)
# =========================================================================== #
def _append_jsonl(path, record, retries=5, backoff=0.1):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True)
    last = None
    for attempt in range(retries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            return True
        except OSError as e:  # ERROR_LOCK_VIOLATION (33) etc. -- retry a few times
            last = e
            time.sleep(backoff * (attempt + 1))
    raise last if last else OSError("append failed: %s" % path)


def _read_jsonl(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except (ValueError, json.JSONDecodeError):
                    continue  # skip a torn last line; never raise on a partial read
    except (FileNotFoundError, OSError):
        return []
    return out


# =========================================================================== #
# change journal
# =========================================================================== #
def record_change(*, change_id, pattern, cal, config_key, old, new, scan_id=None,
                   kind="apply", reverts=None, note=None, ts=None, path=None):
    """Append one change (or revert) event. Returns the record dict.

    ``reverts`` (set for a ``kind='revert'`` event) is the ``change_id`` this event undoes -- it is
    what ``reverted_ids`` keys off, so it MUST be persisted on a revert record."""
    rec = {"change_id": change_id, "kind": kind, "ts": ts, "pattern": pattern, "cal": cal,
           "config_key": config_key, "old": old, "new": new, "scan_id": scan_id,
           "reverts": reverts, "note": note}
    _append_jsonl(path or _paths.changes_path(), rec)
    return rec


def list_changes(path=None):
    return _read_jsonl(path or _paths.changes_path())


def reverted_ids(changes):
    """Set of change_ids that a later ``revert`` event refers to (via ``reverts``)."""
    out = set()
    for c in changes:
        if c.get("kind") == "revert" and c.get("reverts") is not None:
            out.add(c["reverts"])
    return out


def build_revert(orig, *, change_id, ts=None, note="dashboard rollback"):
    """Build (do not write) the revert event for an original change: swap new<->old."""
    return {"change_id": change_id, "kind": "revert", "reverts": orig.get("change_id"),
            "ts": ts, "pattern": orig.get("pattern"), "cal": orig.get("cal"),
            "config_key": orig.get("config_key"),
            "old": orig.get("new"), "new": orig.get("old"),
            "scan_id": orig.get("scan_id"), "note": note}


def current_values(changes):
    """Replay the journal -> the net current value per (pattern, cal, config_key)."""
    cur = {}
    for c in changes:
        key = (c.get("pattern"), c.get("cal"), c.get("config_key"))
        cur[key] = c.get("new")
    return cur


# =========================================================================== #
# dashboard -> controller command queue
# =========================================================================== #
def enqueue_command(*, command_id, type, args=None, ts=None, path=None):
    """Append a command the controller will consume. ``type`` in:
    ``rollback`` | ``set_setting`` | ``set_baseline`` | ``ack_alert`` | ``rerun_cal`` | ``set_cal_field``.
    """
    rec = {"command_id": command_id, "ts": ts, "type": type, "args": args or {}}
    _append_jsonl(path or _paths.commands_path(), rec)
    return rec


def read_commands(path=None):
    return _read_jsonl(path or _paths.commands_path())


def pending_commands(processed, path=None):
    """Commands beyond the ``processed`` count (the controller's high-water mark)."""
    cmds = read_commands(path)
    if processed >= len(cmds):
        return []
    return cmds[processed:]
