"""shot_time.py -- per-shot wall-clock stamping (the RP-N correlation campaign, 2026-07-28).

Records, for EVERY shot of EVERY scan (normal AND rearrangement), the host wall-clock time the
shot ran, keyed by the SAME ``(scan_id, seq_id)`` pair the frames are published under --
``frame_capture`` reads ``int(seq_config.G.seq_id(1))`` and
``rearrange_callbacks._seq_id`` reads ``int(s1.G.seq_id(1))``, i.e. the same field. So an offline
join against the scan's ``.h5`` ``seq_ids`` dataset is exact, with no timestamp matching.

Output: ``<scan_dir>/shot_time.csv``, LONG format (one row per event), because a rearrangement
shot grabs a VARIABLE number of frames:

    scan_id, seq_id, cur_seq_num, arg0, event, frame_idx, t_unix, t_iso

    event = "pre"    -- run_seq pre_cb, immediately before run_real (hardware start)
          = "post"   -- run_seq post_cb, after the shot's frames are in hand
          = "frame"  -- ONE per rearrangement camera grab (rearrange_runtime.grab_one_frame);
                        frame_idx counts 0,1,2,... within the shot. Normal scans read all their
                        frames inside post_cb and emit no "frame" rows (use pre/post instead).

A rearrangement shot lasts ~1 s, so its pre/post bracket is far coarser than the 100 ms
correlation window the campaign needs -- the "frame" rows are what make rearrange scans usable.

WHY a precise clock: on Python 3.12 / Windows ``time.time()`` is ``GetSystemTimeAsFileTime()``
with 15.625 ms resolution (CPython only moved to the precise API in 3.13).
``GetSystemTimePreciseAsFileTime`` (Win8+, via ctypes) gives ~100 ns AND is a WALL clock, so it
is directly comparable across processes with no anchoring -- unlike ``perf_counter``, whose
origin CPython leaves undefined. Falls back to ``time.time()`` where unavailable.

Wholly best-effort: every entry point swallows its own exceptions, and a failure to open the CSV
disables the session rather than raising. A monitoring log must never be able to fail a scan.
Kill-switch: ``YB_SHOT_TIME=0`` (env).

TEMPORARY: added for the RP-N vs fluorescence correlation campaign. See ``campaigns/imaging/rp_ncorr/``.
"""

import os
import threading
import time
from datetime import datetime, timezone

CSV_NAME = "shot_time.csv"

HEADER = ("scan_id", "seq_id", "cur_seq_num", "arg0", "event", "frame_idx", "t_unix", "t_iso")

# --------------------------------------------------------------------------- #
# precise wall clock
# --------------------------------------------------------------------------- #
_PRECISE = None


def _init_precise():
    """Build a ~100 ns wall clock via GetSystemTimePreciseAsFileTime, or None if unavailable."""
    try:
        import ctypes

        k32 = ctypes.windll.kernel32                      # noqa: F821 - Windows only
        fn = k32.GetSystemTimePreciseAsFileTime           # AttributeError pre-Win8
    except Exception:  # noqa: BLE001 - not Windows / too old -> caller falls back
        return None
    import ctypes as _c

    class _FT(_c.Structure):
        _fields_ = [("lo", _c.c_uint32), ("hi", _c.c_uint32)]

    buf = _FT()
    ref = _c.byref(buf)
    # FILETIME is 100 ns ticks since 1601-01-01; 116444736000000000 ticks to the Unix epoch.
    def _now():
        fn(ref)
        return ((buf.hi << 32 | buf.lo) - 116444736000000000) / 1e7

    try:
        _now()
    except Exception:  # noqa: BLE001
        return None
    return _now


def precise_time():
    """Unix seconds. ~100 ns resolution on Windows 8+, else ``time.time()`` (15.6 ms)."""
    global _PRECISE
    if _PRECISE is None:
        _PRECISE = _init_precise() or time.time
    try:
        return _PRECISE()
    except Exception:  # noqa: BLE001 - never let the clock break a shot
        return time.time()


def clock_resolution_note():
    """Human-readable note on which clock this process ended up with (for the run log)."""
    global _PRECISE
    if _PRECISE is None:
        _PRECISE = _init_precise() or time.time
    if _PRECISE is time.time:
        return "time.time() (coarse, %.4g s)" % time.get_clock_info("time").resolution
    return "GetSystemTimePreciseAsFileTime (~1e-7 s)"


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# the per-scan session
# --------------------------------------------------------------------------- #
def enabled():
    """False only when ``YB_SHOT_TIME`` is an explicit falsey value."""
    return os.environ.get("YB_SHOT_TIME", "1").strip().lower() not in ("0", "false", "no", "off")


class ShotTimeSession(object):
    """Appends one CSV row per shot event for a single scan. Best-effort throughout."""

    def __init__(self, path, scan_id):
        self.path = path
        self.scan_id = str(scan_id)
        self._lock = threading.Lock()
        self._fh = None
        self._dead = False
        # Current shot, cached so stamp_frame() needs no seq_config plumbed into the camera layer.
        self._seq_id = -1
        self._cur_seq_num = -1
        self._arg0 = ""
        self._frame_idx = 0
        self._open()

    def _open(self):
        try:
            new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            d = os.path.dirname(self.path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            self._fh = open(self.path, "a", buffering=1)   # line-buffered
            if new:
                self._fh.write(",".join(HEADER) + "\n")
        except Exception:  # noqa: BLE001 - unwritable dir -> silently inert
            self._fh = None
            self._dead = True

    def _row(self, event, seq_id, frame_idx, t):
        if self._dead or self._fh is None:
            return
        try:
            self._fh.write("%s,%d,%d,%s,%s,%s,%.6f,%s\n" % (
                self.scan_id, int(seq_id), int(self._cur_seq_num), self._arg0,
                event, "" if frame_idx is None else int(frame_idx), t, _iso(t)))
        except Exception:  # noqa: BLE001 - one bad row must not kill the scan
            self._dead = True

    # -- run_seq callbacks ------------------------------------------------- #
    def pre(self, seq_id, cur_seq_num, arg0):
        """Stamp the start of a shot and latch its identity for stamp_frame()."""
        t = precise_time()
        with self._lock:
            self._seq_id = int(seq_id)
            self._cur_seq_num = int(cur_seq_num)
            self._arg0 = _fmt_arg0(arg0)
            self._frame_idx = 0
            self._row("pre", self._seq_id, None, t)

    def post(self, seq_id, cur_seq_num, arg0):
        """Stamp the end of a shot (frames in hand)."""
        t = precise_time()
        with self._lock:
            self._cur_seq_num = int(cur_seq_num)
            self._arg0 = _fmt_arg0(arg0)
            self._row("post", int(seq_id), None, t)

    def stamp_frame(self):
        """Stamp ONE rearrangement camera grab against the current shot."""
        t = precise_time()
        with self._lock:
            idx = self._frame_idx
            self._frame_idx += 1
            self._row("frame", self._seq_id, idx, t)

    def finalize(self):
        with self._lock:
            fh, self._fh, self._dead = self._fh, None, True
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass


def _fmt_arg0(arg0):
    """The scan point index as a CSV-safe scalar string (arg0 may be a tuple for 2-D scans)."""
    try:
        if isinstance(arg0, (tuple, list)):
            return "|".join(str(int(x)) for x in arg0)
        return str(int(arg0))
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# module-level active session (so the camera layer can stamp with no plumbing)
# --------------------------------------------------------------------------- #
_ACTIVE = {"session": None}
_ACTIVE_LOCK = threading.Lock()


def begin(scan_id, log=None):
    """Open ``<scan_dir>/shot_time.csv`` for ``scan_id`` and make it the active session.

    Returns the session, or None when disabled / unopenable (callers then skip stamping).
    """
    if not enabled():
        return None
    try:
        from scan_prep import scan_dir
        path = os.path.join(scan_dir(scan_id), CSV_NAME)
        sess = ShotTimeSession(path, scan_id)
    except Exception as e:  # noqa: BLE001 - never fail a scan over a monitoring log
        if log is not None:
            log("[shot_time] disabled (%s)" % e)
        return None
    with _ACTIVE_LOCK:
        _ACTIVE["session"] = sess
    if log is not None:
        log("[shot_time] %s (clock: %s)" % (path, clock_resolution_note()))
    return sess


def end(session=None):
    """Close + clear the active session (idempotent)."""
    with _ACTIVE_LOCK:
        sess = _ACTIVE["session"]
        if session is not None and sess is not session:
            sess = session
        else:
            _ACTIVE["session"] = None
    if sess is not None:
        sess.finalize()


def stamp_frame():
    """Stamp one rearrangement camera grab. No-op when no scan is active. NEVER raises."""
    try:
        sess = _ACTIVE["session"]
        if sess is not None:
            sess.stamp_frame()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# run_scan_group callback factories
# --------------------------------------------------------------------------- #
def make_pre_cb(session, seq_config):
    """A ``run_scan_group`` ``pre_cb``: stamp the shot start.

    seq_id is read EXACTLY as ``frame_capture`` reads it (``int(seq_config.G.seq_id(1))``, before
    the loop bumps it), so the CSV key matches the h5 ``seq_ids`` entry for the same shot.
    """
    def pre_cb(cur_seq_num, arg0):
        try:
            session.pre(_seq_id_of(seq_config), cur_seq_num, arg0)
        except Exception:  # noqa: BLE001
            pass
    return pre_cb


def make_post_cb(session, seq_config):
    """A ``run_scan_group`` ``post_cb``: stamp the shot end (frames in hand)."""
    def post_cb(cur_seq_num, arg0):
        try:
            session.post(_seq_id_of(seq_config), cur_seq_num, arg0)
        except Exception:  # noqa: BLE001
            pass
    return post_cb


def _seq_id_of(seq_config):
    try:
        return int(seq_config.G.seq_id(1))
    except Exception:  # noqa: BLE001 - absent field -> frame_capture's same default
        return 1
