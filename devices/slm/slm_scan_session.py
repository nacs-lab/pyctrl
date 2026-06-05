"""slm_scan_session.py -- scan-long SLM hardware lock + loading-phase write (pyctrl).

Python port of ``matlab_new/YbExptCtrl/SlmScanSession.m``, tuned to the pyctrl rearrangement
spec. Owns the ``slm`` HARDWARE lock for the WHOLE scan (acquired at scan start, held across every
shot, dropped on pause / abort / end) and writes the loading (WGS) phase to the SLM at scan start.

This applies to EVERY scan, not just rearrangement ones: any scan loads atoms into the SLM
pattern and assumes it stays put, so it needs to hold the lock. Rearrangement scans additionally
take a per-shot ``compute`` (GPU) lock and call setup/reload/rearrange (that lives in the seq
callbacks, not here). The server's ``rearrange_actual`` requires the caller hold ``slm`` -- the
scan-long hold under the same ``X-Client-Id`` satisfies that.

Differences from the MATLAB original (per the user spec):
  * **No background heartbeat timer.** The lease is renewed by a per-shot :meth:`keepalive`
    (a single ``/lock/heartbeat`` call, no phase write) so the inter-shot critical path is one
    cheap HTTP round-trip. A shot that outruns the 10 s lease lets it lapse; :meth:`ensure_held`
    on the NEXT shot regrabs + rewrites the WGS phase.
  * **Active, immediate pause drop.** :meth:`on_pause` releases the lock the instant the scan
    pauses (so the SLM can be adjusted); :meth:`on_resume` reacquires + rewrites the phase.
  * **Acquire is mandatory.** :meth:`begin` / :meth:`ensure_held` / :meth:`on_resume`-via-shot
    raise :class:`SlmLockUnavailable` when the lock can't be acquired within the block budget --
    the run is errored (you must not run assuming a pattern you don't actually own).

Lifecycle (driven by the runner / run loop):
    begin()       -- acquire slm (blocking ~5 s), write loading phase, mark held.
    ensure_held() -- per shot: no-op unless the lease lapsed, then regrab + rewrite.
    keepalive()   -- per shot: one heartbeat to renew the lease (no phase write).
    on_pause()    -- active drop.
    on_resume()   -- reacquire + rewrite (best-effort; the next ensure_held enforces).
    done()        -- release (call from the runner's finally for abort safety).

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time

from .slm_client import SlmHTTPError


class SlmLockUnavailable(RuntimeError):
    """The ``slm`` lock could not be acquired within the block budget -- the run must error."""


class SlmScanSession:
    def __init__(self, client, lease_s=10.0, acquire_block_s=5.0,
                 description="yb scan", clock=time.monotonic, log=None):
        self.c = client
        self.lease_s = float(lease_s)             # lock lease (timeout_s); renewed by keepalive
        self.acquire_block_s = float(acquire_block_s)   # wait budget to acquire (block_timeout)
        self.desc = str(description)
        self._clock = clock
        self._log = log or (lambda _m: None)

        self.held = False
        self._last_ok_t = float("-inf")           # last confirmed-held time
        self._last_written = None                 # key of the last-written loading phase

        # loading-pattern declaration (None path -> hold the lock, write nothing).
        self._name = ""
        self._path = ""
        self._zernike = None
        self._legacy = False
        self._baked = None

    # ----------------------------------------------------------------------- #
    # declaration
    # ----------------------------------------------------------------------- #
    def set_loading_pattern(self, name, phase_path, zernike=None,
                            legacy_zerniked=False, baked_zernike=None):
        """Declare the loading phase to write at scan start. ``zernike`` is the (defocus) Zernike
        to ADD on write ([] / None = none). ``legacy_zerniked`` + ``baked_zernike`` describe a
        Zernike still baked into ``phase_path`` that the server strips first."""
        self._name = str(name or "")
        self._path = str(phase_path or "")
        self._zernike = zernike
        self._legacy = bool(legacy_zerniked)
        self._baked = baked_zernike

    def is_held(self):
        return self.held

    # ----------------------------------------------------------------------- #
    # lifecycle
    # ----------------------------------------------------------------------- #
    def begin(self):
        """Acquire the slm lock (mandatory) and write the loading phase. Raises
        :class:`SlmLockUnavailable` if the lock can't be acquired within the block budget."""
        self._acquire(mandatory=True)
        self._write_if_new()

    def ensure_held(self):
        """Per-shot guard. NORMAL path: a timestamp compare, NO SLM comm. If the lease lapsed
        (e.g. a shot outran it), regrab + rewrite the WGS phase before the next shot (the SLM
        could have been touched while we were unlocked). Raises if the regrab fails."""
        if self.held and (self._clock() - self._last_ok_t) < self.lease_s:
            return
        self._acquire(mandatory=True)
        self._last_written = None       # force a rewrite -- the SLM may have changed
        self._write_if_new()

    def keepalive(self):
        """Renew the lease with a SINGLE heartbeat call -- no phase write while we still hold the
        lock (keeps the inter-shot time minimal). A failed heartbeat leaves ``_last_ok_t`` stale
        so the next :meth:`ensure_held` regrabs."""
        if not self.held:
            return
        try:
            self.c.heartbeat("slm")
            self._last_ok_t = self._clock()
        except Exception as e:  # noqa: BLE001 - keepalive is best-effort; ensure_held recovers
            self._log("[SlmScanSession] keepalive failed: %s" % e)

    def on_pause(self):
        """Active, immediate drop so the SLM is free to be adjusted while paused."""
        self._release()

    def on_resume(self):
        """Reacquire + rewrite the loading phase on resume. Best-effort: on failure ``held``
        stays False and the next shot's :meth:`ensure_held` enforces the mandatory regrab."""
        try:
            self._acquire(mandatory=False)
            self._last_written = None
            self._write_if_new()
        except Exception as e:  # noqa: BLE001
            self._log("[SlmScanSession] resume reacquire failed: %s" % e)

    def done(self):
        """Release the lock (idempotent -- safe to call when already dropped)."""
        self._release()

    # ----------------------------------------------------------------------- #
    # internals
    # ----------------------------------------------------------------------- #
    def _acquire(self, mandatory):
        try:
            self.c.acquire_lock("slm", self.desc,
                                timeout_s=self.lease_s,
                                block_timeout=self.acquire_block_s)
            self.held = True
            self._last_ok_t = self._clock()
        except SlmHTTPError as err:
            self.held = False
            self._log("[SlmScanSession] acquire slm lock failed (HTTP %d): %s"
                      % (err.status, err.detail))
            if mandatory:
                raise SlmLockUnavailable(
                    "could not acquire the slm lock within %.1fs (HTTP %d: %s)"
                    % (self.acquire_block_s, err.status, err.detail))
        except Exception as err:  # noqa: BLE001 - connection error / server down
            self.held = False
            self._log("[SlmScanSession] acquire slm lock error: %s" % err)
            if mandatory:
                raise SlmLockUnavailable("could not acquire the slm lock: %s" % err)

    def _release(self):
        if not self.held:
            return
        try:
            self.c.release_lock("slm")
        except Exception as e:  # noqa: BLE001 - release is best-effort
            self._log("[SlmScanSession] release failed: %s" % e)
        self.held = False

    def _write_if_new(self):
        if not self._path or not self.held:
            return                                  # no pattern declared -> hold only, no write
        key = self._pattern_key()
        if key == self._last_written:
            return                                  # unchanged -> assume nothing touched the SLM
        try:
            self.c.write_loading_phase(self._path, self._zernike,
                                       name=self._name or None,
                                       legacy_zerniked=self._legacy,
                                       baked_zernike=self._baked)
            self._last_written = key
            self._last_ok_t = self._clock()
            self._log("[SlmScanSession] wrote loading phase %s" % (self._name or self._path))
        except Exception as e:  # noqa: BLE001
            self._log("[SlmScanSession] write_loading_phase failed: %s" % e)

    def _pattern_key(self):
        return (self._name, _key_list(self._zernike), _key_list(self._baked), self._legacy)


def _key_list(v):
    if v is None:
        return ()
    try:
        return tuple(float(x) for x in v)
    except TypeError:
        return (float(v),)
