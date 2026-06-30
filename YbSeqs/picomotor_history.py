"""Append-only history of picomotor-mirror relative moves, for tracing alignment drift.

The piezo picomotor mirrors are driven by a DC voltage kick on an NI channel (``VPicoMotor*``):
hold ``volts`` for ``hold_s`` then ramp back to 0 -- one *relative* move whose direction is the
sign of ``volts`` (see ``PicoMotor308``'s "+2V move twice faster than -2V" note). There is no
absolute position readout, so the only way to know where a mirror sits relative to where it
started is to log every kick. :func:`log_move` appends one JSONL record per move and keeps a
running per-channel cumulative ``volt_seconds`` (signed sum of ``volts*hold_s``) -- a
direction-aware proxy for net displacement you can trace back through.

Records go to ``<project_root>/log/pyctrl_log/picomotor_history.jsonl`` (one persistent file, NOT
per-session -- the whole point is a continuous history across restarts), beside the other pyctrl
logs. Best-effort: a logging failure never breaks a sequence build. Env knob ``YB_PYCTRL_LOG=0``
disables it (matches logging_setup); ``YB_PYCTRL_LOG_DIR`` overrides the dir.

# ponytail: flat JSONL + an in-process running total. If you ever need cross-process-accurate
# cumulative (two backends kicking the same mirror), re-derive the total by replaying the file
# instead of trusting _CUM.
"""

import json
import logging
import os
import threading
import time

_log = logging.getLogger("pyctrl.picomotor")  # INFO also lands in pyctrl_logging_<ts>.log

_LOCK = threading.Lock()
_CUM = {}  # channel -> running signed sum of volts*hold_s this process


def _history_path():
    """``.../pyctrl/YbSeqs/picomotor_history.py`` -> ``<root>/log/pyctrl_log/picomotor_history.jsonl``."""
    log_dir = os.environ.get("YB_PYCTRL_LOG_DIR")
    if not log_dir:
        pyctrl = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.dirname(pyctrl)
        log_dir = os.path.join(root, "log", "pyctrl_log")
    return os.path.join(log_dir, "picomotor_history.jsonl")


def log_move(channel, volts, hold_s, note=""):
    """Record one relative picomotor kick. Returns the new cumulative volt-seconds (or None).

    ``volts``/``hold_s`` are the kick the sequence applies; the signed product ``volts*hold_s``
    is the per-move "amount" and its running sum (per channel) is the traceable net offset.
    """
    if os.environ.get("YB_PYCTRL_LOG", "").strip() == "0":
        return None
    move = float(volts) * float(hold_s)
    try:
        with _LOCK:
            cum = _CUM.get(channel, 0.0) + move
            _CUM[channel] = cum
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "channel": channel,
                "volts": float(volts),
                "hold_s": float(hold_s),
                "move_volt_seconds": move,
                "cum_volt_seconds": cum,
            }
            if note:
                rec["note"] = note
            path = _history_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        _log.info("picomotor %s move %+g V x %g s (cum %+g V.s)%s",
                  channel, volts, hold_s, cum, (" -- " + note) if note else "")
        return cum
    except Exception:  # noqa: BLE001 - history must never break a sequence build
        _log.warning("picomotor history log failed for %s", channel, exc_info=True)
        return None


def demo():
    """Self-check: cumulative is the signed sum; both moves land in the file."""
    import tempfile
    d = tempfile.mkdtemp()
    os.environ["YB_PYCTRL_LOG_DIR"] = d
    _CUM.clear()
    assert log_move("VPicoMotor308v", 2, 0.5) == 1.0
    assert log_move("VPicoMotor308v", -2, 0.5) == 0.0   # back to start
    lines = open(os.path.join(d, "picomotor_history.jsonl"), encoding="utf-8").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["cum_volt_seconds"] == 0.0
    print("ok")


if __name__ == "__main__":
    demo()
