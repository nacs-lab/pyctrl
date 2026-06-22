"""control_channel.py -- gentle pause / abort / start for the pyctrl run loop.

Replaces the memmap ``CheckPauseAbort`` gate (``matlab_new/+archived/scripts/
CheckPauseAbort.m``, called at ``runSeq2.m:305``) with a poll of the ExptServer ZMQ
``SeqRequest`` -- the user-ratified "scenario 3 has NO memmap" resolution
(references/runtime-design.md). The ExptServer ``SeqRequest`` (``NoRequest`` / ``Pause`` /
``Abort``) is the single source of truth; this module is the run-loop-side CONSUMER.

The control source is duck-typed (the hosted ExptServer in production, a fake in tests):

    check_request() -> int        # 0 NoRequest / 1 Pause / 2 Abort (ExptServer.check_request)
    start_scan()    -> scan_id    # clears the request to NoRequest, marks Running
    ack_paused(on)  -> None        # OPTIONAL: reflect actually-parked state (item-7 hook)

``CheckPauseAbort`` semantics reproduced exactly:
  * **Abort precedes Pause** -- "if we are aborting, don't bother pausing."
  * **Pause parks** the run loop in a re-poll spin that breaks out on resume (the request
    clears, via ExptServer.start_seq_serv) OR on abort.
  * **Reached-paused ack** -- ``CheckPauseAbort`` raises ``IsPausedRunSeq=1`` only while
    actually spinning; :meth:`ack_paused` mirrors that. This is the *request-vs-reached*
    must-fix: ExptServer.pause_seq flips State on the bare request (a lie mid-shot); the
    ack corrects it to mean "the runner truly parked."

Must-fix policies folded in (NO-HARDWARE-unit-testable here; the live two-process coherency
test is item-7 territory, see PYTHON_FRONTEND_PLAN.md Phase 5):
  * **single clear-point (clear-at-job-start)** -- ALL control flags (Pause AND Abort) clear
    at START-OF-SCAN: :meth:`begin_scan` calls ``start_scan``, which resets the request to
    NoRequest and marks Running. An abort stops the CURRENT scan via the per-sequence gate
    (:meth:`check_pause_abort`), but a newly-popped job is fresh intent to run, so a stale
    ``Abort``/``Pause`` left from the previous scan must NOT block it. This is the
    runtime-design "clear all control flags at start-of-job" must-fix and the cure for the
    stale-abort wedge (bug-runjob-stale-abortrunseq): the earlier "abort-sticky" begin_scan
    refused to start while an abort was pending, which permanently wedged every future scan
    at 0 iterations because no ZMQ verb clears a stale ``Abort`` from the ``Init`` state.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time
from enum import IntEnum


class SeqRequest(IntEnum):
    """Mirror of ``ExptServer.SeqRequest`` (the wire values ``check_request`` returns)."""
    NoRequest = 0
    Pause = 1
    Abort = 2


class ControlChannel:
    """Run-loop-side wrapper over an ExptServer-like control source (no memmap).

    Args:
        source: the control source (ExptServer or fake) -- see module docstring.
        poll_interval: seconds between re-polls while parked (MATLAB uses 1.0; the
            ExptServer worker thread updates the request concurrently, so the spin sees
            new requests).
        sleep: the sleep function (injected in tests to advance a scripted request
            timeline deterministically).
    """

    def __init__(self, source, poll_interval=1.0, sleep=time.sleep,
                 on_pause=None, on_resume=None, is_background=False):
        self._source = source
        self._poll = poll_interval
        self._sleep = sleep
        # Optional scan-boundary hooks fired when the run loop ENTERS the pause park (on_pause)
        # and when it RESUMES out of it (on_resume, NOT on abort). The rearrangement run wires
        # these to actively drop the scan-long SLM lock on pause and reacquire+rewrite on resume;
        # both are no-ops when no SLM session is active. Best-effort: a hook failure never breaks
        # the pause gate (a failed resume regrab is enforced by the next shot's ensure_held).
        self._on_pause = on_pause
        self._on_resume = on_resume
        # Background (calibration) lane: when True, :meth:`should_yield` lets the run bail out at
        # a shot boundary the moment foreground work is queued (see the run loop's two-tier
        # schedule). A normal (foreground) scan never yields. This is INDEPENDENT of the
        # Pause/Abort SeqRequest -- yielding sets NO control flag (see should_yield).
        self._is_background = bool(is_background)

    # ----------------------------------------------------------------------- #
    # the per-sequence gate (CheckPauseAbort replacement)
    # ----------------------------------------------------------------------- #
    def check_pause_abort(self):
        """Poll the control source at the per-sequence injection point.

        Returns ``True`` to ABORT (stop the run gracefully at this sequence boundary),
        ``False`` to PROCEED. Parks (spins) while the request is ``Pause``; an ``Abort``
        arriving during the pause wins.
        """
        req = self._request()
        if req == SeqRequest.Abort:
            return True               # abort precedes pause -- don't park
        if req == SeqRequest.Pause:
            self.ack_paused(True)     # IsPausedRunSeq = 1 (reached-paused ack)
            self._fire(self._on_pause)  # active drop of the scan-long SLM lock (no-op if none)
            resumed = False
            try:
                while True:
                    req = self._request()
                    if req == SeqRequest.Abort:
                        return True
                    if req != SeqRequest.Pause:
                        resumed = True
                        return False  # resumed (request cleared via start_seq_serv)
                    self._sleep(self._poll)
            finally:
                self.ack_paused(False)  # IsPausedRunSeq = 0 on exit (resume OR abort)
                if resumed:
                    self._fire(self._on_resume)  # reacquire + rewrite the loading phase
        return False

    # ----------------------------------------------------------------------- #
    # scan-boundary control (single clear-point + abort-sticky)
    # ----------------------------------------------------------------------- #
    def begin_scan(self):
        """Start-of-scan single clear-point: clear ALL stale control flags (Pause AND Abort)
        and mark Running, returning the new ``scan_id``.

        A newly-popped job represents fresh intent to run, so a leftover ``Abort``/``Pause``
        from the PREVIOUS scan must not block it -- ``start_scan`` resets the request to
        NoRequest and marks Running (the runtime-design "clear all control flags at
        start-of-job" must-fix; bug-runjob-stale-abortrunseq). An abort issued while THIS scan
        runs still stops it via :meth:`check_pause_abort`; only a stale, pre-job abort is
        cleared here. (Previously this refused to start while an abort was pending, which
        permanently wedged every future scan because no ZMQ verb clears a stale ``Abort``.)

        Returns the new ``scan_id`` (``None`` only if the source's ``start_scan`` itself
        signals a refusal -- ``run_scan_group`` still treats ``None`` as an aborted start).
        """
        return self._source.start_scan()

    def aborting(self):
        """True iff an abort is currently pending (cheap, non-parking check)."""
        return self._request() == SeqRequest.Abort

    # ----------------------------------------------------------------------- #
    # background-lane yield (NOT a SeqRequest -- pure queue-state predicate)
    # ----------------------------------------------------------------------- #
    def should_yield(self):
        """True iff THIS is a background (calibration) run AND foreground work is now queued, so
        the run should bail out cleanly at this shot boundary (the run loop then runs the
        foreground scan and re-queues this one).

        CARDINAL RULE: this NEVER reads or writes the ``SeqRequest`` (Pause/Abort). The yield is
        driven solely by the queue state (``source.has_foreground_work()``), so it cannot fool the
        incoming foreground scan's ``begin_scan``/``start_scan`` -- which clears nothing because
        nothing was set. Returns False for a normal (foreground) run, or if the source predates
        the predicate (getattr-guarded, so old fakes/servers are unaffected)."""
        if not self._is_background:
            return False
        fn = getattr(self._source, "has_foreground_work", None)
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 - a predicate failure must not break the run loop
            return False

    # ----------------------------------------------------------------------- #
    # helpers
    # ----------------------------------------------------------------------- #
    def _request(self):
        return SeqRequest(int(self._source.check_request()))

    def _fire(self, hook):
        """Call a scan-boundary hook best-effort; never let it break the pause gate."""
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001 - a hook failure must not crash the run loop
            pass

    def ack_paused(self, parked):
        """Reflect actually-parked state on the source, if it supports it (item-7 hook).

        No-op when the source has no ``ack_paused`` -- the coarse-status v1 ExptServer does
        not, and the run loop still parks correctly; only get_status granularity differs.
        """
        ack = getattr(self._source, "ack_paused", None)
        if ack is not None:
            ack(bool(parked))
