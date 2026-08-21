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
  * **No background heartbeat timer.** The lease is renewed on the per-shot path by a single
    ``/lock/heartbeat``. :meth:`ensure_held` is SERVER-AUTHORITATIVE -- it heartbeats every shot
    to BOTH confirm ownership and renew, and regrabs (+ rewrites the WGS phase) the moment the
    heartbeat fails. This requires the server's ``/lock/heartbeat`` to REJECT (HTTP 409) a caller
    that no longer holds the lock (``LockManager.heartbeat_renew`` in slm_server); a fire-and-forget
    heartbeat that always returns ``ok`` would defeat this guard. It never trusts a local timestamp,
    so a server-side release (lease lapse during a long shot / warmup, or a stolen lock) can't
    silently wedge the scan. Because the lease is renewed ONLY while shots run, a pause / hung shot
    / crashed runner lets it lapse within ``lease_s`` and the server auto-releases ``slm`` -- the
    SLM is never wedged for longer than one (short) lease after a stop.

    ``lease_s`` MUST therefore exceed the longest single shot, not the typical one. The renewal
    points are all on the per-shot path, so a shot that blocks inside ``/slm/rearrange`` emits no
    heartbeat for its whole duration; if that exceeds the lease, the server releases the lock
    UNDERNEATH A HEALTHY RUN. What follows is not recoverable by retrying: the next ``ensure_held``
    regrab lands while the same server is still inside a rearrange, hits its busy gate, and 503s.
    That chain killed jobs 313, 343, 344 and 391 (391 at 194/1320 shots, after exhausting a 45 s
    retry budget). Keeping the lease "modest" is therefore in direct tension with surviving the
    slow-shot tail, and the tail wins: an over-long lease costs at most one lease of a stuck SLM
    after a crash, whereas an under-long lease costs the entire run.
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
    # 2026-08-07: raised 15.0 -> 60.0. Measured shot totals on the axial campaign are 3.2 s median
    # but the tail reaches 22 s (before_bseq alone hit 20.2 s), so a 15 s lease lapsed mid-shot on
    # the slow tail -- ~1 lapse/min in the runner log, each one a 409 followed by a regrab that can
    # 503 against the server's own rearrange gate. 60 s clears the worst observed shot by ~3x.
    def __init__(self, client, lease_s=60.0, acquire_block_s=5.0,
                 acquire_retry_s=45.0, acquire_retry_pause_s=0.5,
                 description="yb scan", clock=time.monotonic, log=None):
        self.c = client
        self.lease_s = float(lease_s)             # lock lease (timeout_s); renewed by keepalive
        self.acquire_block_s = float(acquire_block_s)   # wait budget to acquire (block_timeout)
        # Total budget for retrying TRANSIENT acquire failures (503 server-busy / 408). Must
        # comfortably exceed the longest rearrange call, since the server's busy gate 503s for its
        # whole duration: this campaign measures 0.7 s median / 5.3 s p90 with ~400-frame shots, so
        # 45 s is ~8x the p90 and still bounded well below any human-noticeable stall.
        self.acquire_retry_s = float(acquire_retry_s)
        self.acquire_retry_pause_s = float(acquire_retry_pause_s)
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

    def set_defocus(self, z4):
        """Re-point the LOADING DEFOCUS mid-scan and rewrite the phase if it changed.

        This is what makes a per-shot defocus (ANSI z4) sweep possible inside ONE scan: the
        declared pattern is unchanged, only its ``[0 0 0 0 z4]`` Zernike moves, and
        :meth:`_write_if_new` keys on that Zernike (``_pattern_key``) so a changed value forces a
        fresh ``write_loading_phase``. Called from the run loop's per-shot pre_cb BEFORE the
        sequence runs, so the atoms of that shot are loaded at the new plane.

        Returns True iff a write actually went out (unchanged value / no declared pattern -> False).
        """
        if not self._path:
            return False                    # holding the lock only -- nothing declared to rewrite
        z = [0.0, 0.0, 0.0, 0.0, float(z4)] if float(z4) else []
        if _key_list(z) == _key_list(self._zernike):
            return False
        self._zernike = z
        before = self._last_written
        self._write_if_new()
        return self._last_written != before

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
        """Per-shot guard. SERVER-AUTHORITATIVE: one heartbeat that BOTH confirms we still own
        ``slm`` and renews the lease. On heartbeat failure (lapsed lease / server restart / lock
        stolen) regrab + rewrite the WGS phase before the next shot (the SLM could have been
        touched while we were unlocked). A mid-scan regrab that fails within the block budget
        raises :class:`SlmLockUnavailable` -> the run errors (a scan that cannot own the SLM must
        not silently spin). The old purely-local timestamp compare is gone: it could not see a
        server-side release (lease lapse during a long shot / warmup), which silently wedged the
        scan -- the heartbeat is the single source of truth for ownership.

        NOTE: this relies on the server's ``/lock/heartbeat`` REJECTING a caller that no longer
        holds the lock (HTTP 409). A server whose heartbeat is fire-and-forget (always ``ok``)
        breaks the contract: the heartbeat then never raises, this guard never regrabs, and every
        lock-requiring call 423s forever. That server bug was fixed in ``LockManager.heartbeat_renew``
        (slm_server) -- keep both ends in sync."""
        if self.held:
            try:
                self.c.heartbeat("slm")
                self._last_ok_t = self._clock()
                return
            except Exception as e:  # noqa: BLE001 - lost the lock; re-acquire below
                self._log("[SlmScanSession] lost slm lock (heartbeat: %s); re-acquiring" % e)
                self.held = False
        self._acquire(mandatory=True)
        self._last_written = None       # force a rewrite -- the SLM may have changed
        self._write_if_new()

    def keepalive(self):
        """Renew the lease with a SINGLE heartbeat call -- no phase write while we still hold the
        lock (keeps the inter-shot time minimal). A failed heartbeat means we LOST the lock, so
        clear ``held`` -> the next :meth:`ensure_held` deterministically regrabs (rather than
        trusting a stale local timestamp)."""
        if not self.held:
            return
        try:
            self.c.heartbeat("slm")
            self._last_ok_t = self._clock()
        except Exception as e:  # noqa: BLE001 - lost the lock; ensure_held regrabs next shot
            self._log("[SlmScanSession] keepalive failed: %s" % e)
            self.held = False

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
    # HTTP statuses that mean "the server is momentarily unable", NOT "you lost the lock".
    # 503 is the server's busy gate -- it is returned while a rearrange is in flight, and the
    # slm skill documents it as RETRIABLE ("503 'server busy: rearrange' is retriable; the
    # Python client backs off ~0.3 s"). 408 is a blocking-acquire timeout, likewise transient.
    _TRANSIENT_STATUS = (503, 408)

    def _acquire(self, mandatory):
        """Acquire the ``slm`` lock, RETRYING transient server-busy responses.

        WHY THE RETRY EXISTS (2026-08-07). Jobs 313 and 343 both died with
        ``could not acquire the slm lock within 5.0s (HTTP 503: server busy: rearrange (timed out
        after 2.0 s))`` -- 343 after only 4 of 1386 shots. The campaign's pseudo-one-way shots make
        ~300-400 SLM writes and the rearrange call runs 0.7 s median / 5.3 s p90, so a heartbeat or
        acquire issued while one is in flight gets the 503 busy gate. The old code treated that
        single transient 503 as a permanent failure and raised, killing the whole run.

        A 503 says nothing about ownership, so retry it across a budget that comfortably exceeds
        the longest rearrange call. Genuine lock loss (409 / 423) and every other status still fail
        immediately -- a scan that truly cannot own the SLM must not silently spin.
        """
        deadline = self._clock() + self.acquire_retry_s
        attempt = 0
        last = None
        while True:
            attempt += 1
            try:
                self.c.acquire_lock("slm", self.desc,
                                    timeout_s=self.lease_s,
                                    block_timeout=self.acquire_block_s)
                self.held = True
                self._last_ok_t = self._clock()
                if attempt > 1:
                    self._log("[SlmScanSession] slm lock acquired on attempt %d" % attempt)
                return
            except SlmHTTPError as err:
                last = err
                self.held = False
                transient = err.status in self._TRANSIENT_STATUS
                if transient and self._clock() < deadline:
                    self._log("[SlmScanSession] slm acquire HTTP %d (%s) -- transient, retrying "
                              "(attempt %d, %.1fs left)"
                              % (err.status, err.detail, attempt,
                                 deadline - self._clock()))
                    time.sleep(self.acquire_retry_pause_s)
                    continue
                self._log("[SlmScanSession] acquire slm lock failed (HTTP %d): %s"
                          % (err.status, err.detail))
                if mandatory:
                    raise SlmLockUnavailable(
                        "could not acquire the slm lock within %.1fs (HTTP %d: %s)"
                        % (self.acquire_retry_s if transient else self.acquire_block_s,
                           err.status, err.detail))
                return
            except Exception as err:  # noqa: BLE001 - connection error / server down
                last = err
                self.held = False
                if self._clock() < deadline:
                    self._log("[SlmScanSession] slm acquire error (%s) -- retrying" % err)
                    time.sleep(self.acquire_retry_pause_s)
                    continue
                break
        self._log("[SlmScanSession] acquire slm lock error: %s" % last)
        if mandatory:
            raise SlmLockUnavailable("could not acquire the slm lock: %s" % last)

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
