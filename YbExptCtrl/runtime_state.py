"""runtime_state.py -- pyctrl-private persistent runtime state (currently: the 616-EOM freq).

A tiny mmap-backed store for the one bit of cross-shot / cross-scan state a sequence must carry:
the last 616-EOM frequency, used to keep its inter-run ramp short. ``PushoutSurvivalSeq`` & friends
open with a slow ramp that slews the 616 EOM from its LAST value to this run's target; that "last
value" used to live in ``MemoryMap.Data(1).FreqEOM616Old``.

This is **pyctrl-private**: its OWN ``<tempdir>/nacsctl/pyctrl_runtime_state.dat`` (8 bytes, one
little-endian float64 at offset 0), NOT the MATLAB ``nacs_mem_map.dat``. So it carries no
``MemoryMap.m`` byte-layout coupling, and the scenario-3 "no memmap" control boundary is untouched
-- this is benign physical state, read/written only from the DEFERRED ``server_pre_run`` /
``server_post_run`` callbacks, which ``serialize()`` never runs (so the byte path is unaffected).

mmap rather than a JSON file (operator's choice): a single seek + 8-byte read/write on a handle
opened once for the process lifetime -- no per-shot file open/parse. A fresh file is initialised to
NaN ("unset"), so the first run falls back to the target (a 20 ms ramp, no slow first shot);
thereafter the previous run's target is reused, so a constant-freq scan ramps ~0 and an EOM-sweep
ramps only the per-point delta. Without any of this the global stays 0 and the ramp runs ~15 s/shot
(FPGA bytecode ~60 MB).
"""

import logging
import mmap
import os
import struct
import tempfile

logger = logging.getLogger(__name__)

# pyctrl-private store (distinct from MATLAB's nacs_mem_map.dat), in the run-loop artifact dir.
_PATH = os.path.join(tempfile.gettempdir(), "nacsctl", "pyctrl_runtime_state.dat")
_SIZE = 8                       # one float64: FreqEOM616Old at offset 0
_NAN = struct.pack("<d", float("nan"))

_mm = None                      # cached mmap handle (opened once, process lifetime)
_fh = None


def _open():
    """Return the cached mmap, creating/initialising the file (to NaN) on first use."""
    global _mm, _fh
    if _mm is not None:
        return _mm
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    if not os.path.isfile(_PATH) or os.path.getsize(_PATH) < _SIZE:
        with open(_PATH, "wb") as f:        # fresh/short file -> NaN sentinel = "unset"
            f.write(_NAN)
    _fh = open(_PATH, "r+b")
    _mm = mmap.mmap(_fh.fileno(), _SIZE)
    return _mm


def get_eom616_old(default):
    """Persisted last 616-EOM freq, or ``default`` when unset (NaN/0/unreadable).

    Neither NaN (fresh file) nor 0.0 Hz is a valid 616-EOM frequency, so both mean "no value
    yet" -> use the caller's default (this run's target).
    """
    try:
        mm = _open()
        mm.seek(0)
        v = struct.unpack("<d", mm.read(8))[0]
    except Exception as e:  # noqa: BLE001 - any I/O failure -> safe default
        logger.debug("runtime_state read failed (%s); using default", e)
        return float(default)
    if v != v or v == 0.0:      # NaN or 0.0 -> unset
        return float(default)
    return v


def set_eom616_old(value):
    """Persist ``value`` as the last 616-EOM freq (flushed, so it survives a backend restart)."""
    try:
        mm = _open()
        mm.seek(0)
        mm.write(struct.pack("<d", float(value)))
        mm.flush()
    except Exception as e:  # noqa: BLE001 - persistence is best-effort; never kill a shot
        logger.debug("runtime_state write failed: %s", e)


def register_eom616_persistence(s, freq616global, freq_target):
    """Wire a sequence's 616-EOM ramp to start from the LAST run's frequency.

    MemoryMap-free replacement for MATLAB's ``server_pre_run``/``server_post_run`` handling of
    ``FreqEOM616Old``:
      * ``before_start`` (pre-run): inject ``freq616global`` <- the persisted last 616-EOM freq
        (default ``freq_target`` on the first run). Read at ``pre_run``/``bc_gen``, which runs AFTER
        ``before_start`` callbacks, so the BAKED ramp length reflects it (verified: bytecode
        60.8 MB -> 0.2 MB, codegen 6.1 s -> 0.02 s).
      * ``after_end`` (post-run): persist this run's ``freq_target`` as the next run's "old" value.

    Does NOT change ``serialize()`` output (callbacks are not serialized), so byte-equality with
    MATLAB is preserved.
    """
    target = float(freq_target)

    def _pre(s1):
        s1.set_global(freq616global, get_eom616_old(default=target))

    def _post(s1):
        set_eom616_old(target)

    s.reg_before_start(_pre)
    s.reg_after_end(_post)
    return s
