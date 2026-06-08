"""Mirror the pyctrl server's terminal output into organized on-disk log files.

The pyctrl run-loop host (:func:`runner.serve`) emits two independent output streams that
otherwise vanish when the launching terminal closes:

1. ``print()``-based narration -- ``serve()``/``consume_loop()`` route through a ``log=print``
   default, and ``ExptServer`` prints directly (from a *separate* ZMQ worker thread), including
   ``traceback.print_exc()``. ``warnings.warn`` also lands here (on stderr).
2. The stdlib ``logging`` module -- device drivers, ``seq_dump``, ``code_snapshot``, etc. define
   ``pyctrl.*`` loggers but, with no handlers configured, their INFO/DEBUG is dropped (only
   WARNING+ reaches stderr via the ``lastResort`` handler, which the tee below then captures).

:func:`setup_logging` installs a thread-safe tee on ``sys.stdout``/``sys.stderr`` (so the
terminal stays **exactly** as it is today -- we only *copy* to disk) and attaches file handlers
to the root logger. Three files share one per-server-session timestamp, written under
``<project_root>/log/pyctrl_log/`` beside the existing MATLAB ``log/matlab_log/``:

* ``pyctrl_runner_<ts>.log``  -- terminal mirror (tee of stdout+stderr), one timestamp per line.
* ``pyctrl_logging_<ts>.log`` -- logging module, INFO and above.
* ``pyctrl_debug_<ts>.log``   -- logging module, DEBUG only (created lazily; often empty/absent).

Every line in all three carries the same ``YYYY-MM-DD HH:MM:SS,mmm`` stamp so the files
cross-correlate. The whole thing is best-effort and idempotent: any failure leaves the server
running terminal-only, and a second call is a no-op.

Env knobs:
* ``YB_PYCTRL_LOG=0``      -- disable file logging entirely (return ``None``).
* ``YB_PYCTRL_LOG_DIR``    -- override the output directory.
"""

import logging
import os
import sys
import threading
import time

# Match the yb_analysis logging format so timestamps line up across files.
_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"  # asctime appends ",mmm" milliseconds automatically

_INSTALLED = False
_LOCK = threading.Lock()  # shared by both std-stream tees so lines never interleave


def _stamp():
    """``YYYY-MM-DD HH:MM:SS,mmm`` for the current instant (matches logging's asctime)."""
    now = time.time()
    ms = int((now - int(now)) * 1000)
    return "%s,%03d" % (time.strftime(_DATEFMT, time.localtime(now)), ms)


def _project_root():
    """``.../pyctrl/YbExptCtrl/logging_setup.py`` -> ``.../`` (the superproject root)."""
    pyctrl = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(pyctrl)


class _LineTee:
    """Write verbatim to ``stream`` (the terminal) and a timestamped copy to ``fh`` (a file).

    The terminal stream is passed through untouched. The file copy gets a per-line timestamp
    prefix; partial-line writes (``print(..., end="")``) are handled by tracking line-start
    state. A shared lock across both std streams keeps the worker thread from interleaving.
    Disk errors are swallowed so a full/locked disk never breaks the console.
    """

    def __init__(self, stream, fh, lock):
        self._stream = stream
        self._fh = fh
        self._lock = lock
        self._at_line_start = True

    def write(self, s):
        if not s:
            return
        with self._lock:
            self._stream.write(s)
            self._stream.flush()
            try:
                self._fh.write(self._with_stamps(s))
                self._fh.flush()
            except Exception:  # noqa: BLE001 - disk hiccup must never break the terminal
                pass

    def _with_stamps(self, s):
        # Prefix a timestamp at the start of every line, preserving the exact text/newlines.
        out = []
        for ch in s.splitlines(keepends=True):
            if self._at_line_start:
                out.append(_stamp() + " ")
            out.append(ch)
            self._at_line_start = ch.endswith(("\n", "\r"))
        return "".join(out)

    def flush(self):
        with self._lock:
            self._stream.flush()
            try:
                self._fh.flush()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        # isatty/fileno/encoding/etc. -> delegate to the real stream (stay transparent).
        return getattr(self._stream, name)


class _DebugOnly(logging.Filter):
    """Pass only DEBUG records (so the debug file doesn't duplicate the INFO file)."""

    def filter(self, record):  # noqa: A003 - logging API name
        return record.levelno == logging.DEBUG


def setup_logging(level=logging.DEBUG):
    """Install the tee + root-logger file handlers. Best-effort, idempotent.

    Returns a dict of the three resolved paths, or ``None`` if disabled or the log directory
    could not be created. Safe to call more than once (subsequent calls are no-ops).
    """
    global _INSTALLED
    if _INSTALLED:
        return None
    if os.environ.get("YB_PYCTRL_LOG", "").strip() == "0":
        return None

    try:
        log_dir = os.environ.get("YB_PYCTRL_LOG_DIR") or os.path.join(
            _project_root(), "log", "pyctrl_log")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            return None  # unwritable -> terminal-only, never crash

        ts = time.strftime("%Y%m%d_%H%M%S")
        paths = {
            "mirror": os.path.join(log_dir, "pyctrl_runner_%s.log" % ts),
            "logging": os.path.join(log_dir, "pyctrl_logging_%s.log" % ts),
            "debug": os.path.join(log_dir, "pyctrl_debug_%s.log" % ts),
        }

        # (b) Tee stdout + stderr into the mirror file (one shared handle + lock).
        mirror_fh = open(paths["mirror"], "a", encoding="utf-8")
        sys.stdout = _LineTee(sys.stdout, mirror_fh, _LOCK)
        sys.stderr = _LineTee(sys.stderr, mirror_fh, _LOCK)

        # (c) Route the logging module to file(s) only -- no console handler (terminal stays
        #     untouched; the tee already mirrors the WARNING+ lines that reach stderr).
        fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
        root = logging.getLogger()
        root.setLevel(level)

        info_h = logging.FileHandler(paths["logging"], encoding="utf-8", delay=True)
        info_h.setLevel(logging.INFO)
        info_h.setFormatter(fmt)
        root.addHandler(info_h)

        debug_h = logging.FileHandler(paths["debug"], encoding="utf-8", delay=True)
        debug_h.setLevel(logging.DEBUG)
        debug_h.addFilter(_DebugOnly())
        debug_h.setFormatter(fmt)
        root.addHandler(debug_h)

        _INSTALLED = True
        return paths
    except Exception:  # noqa: BLE001 - logging setup must never take the server down
        return None
