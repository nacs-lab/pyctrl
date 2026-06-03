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
  * **single clear-point + abort-sticky** -- control flags clear at START-OF-SCAN only, and
    a pending ``Abort`` survives the queue boundary: :meth:`begin_scan` refuses to start a
    job while an abort is pending (so ``start_scan`` cannot clobber a between-scan abort),
    rather than clearing-then-running.

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

    def __init__(self, source, poll_interval=1.0, sleep=time.sleep):
        self._source = source
        self._poll = poll_interval
        self._sleep = sleep

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
            try:
                while True:
                    req = self._request()
                    if req == SeqRequest.Abort:
                        return True
                    if req != SeqRequest.Pause:
                        return False  # resumed (request cleared via start_seq_serv)
                    self._sleep(self._poll)
            finally:
                self.ack_paused(False)  # IsPausedRunSeq = 0 on exit (resume OR abort)
        return False

    # ----------------------------------------------------------------------- #
    # scan-boundary control (single clear-point + abort-sticky)
    # ----------------------------------------------------------------------- #
    def begin_scan(self):
        """Start-of-scan single clear-point. Returns the new ``scan_id``, or ``None`` if an
        abort is pending (abort-sticky: a between-scan abort is honored by refusing to
        start, NOT clobbered by ``start_scan``'s reset).

        ``start_scan`` itself clears Pause/NoRequest and marks Running -- so Pause is the
        only flag the reset clears, satisfying "single clear-point" without losing Abort.
        """
        if self._request() == SeqRequest.Abort:
            return None
        return self._source.start_scan()

    def aborting(self):
        """True iff an abort is currently pending (cheap, non-parking check)."""
        return self._request() == SeqRequest.Abort

    # ----------------------------------------------------------------------- #
    # helpers
    # ----------------------------------------------------------------------- #
    def _request(self):
        return SeqRequest(int(self._source.check_request()))

    def ack_paused(self, parked):
        """Reflect actually-parked state on the source, if it supports it (item-7 hook).

        No-op when the source has no ``ack_paused`` -- the coarse-status v1 ExptServer does
        not, and the run loop still parks correctly; only get_status granularity differs.
        """
        ack = getattr(self._source, "ack_paused", None)
        if ack is not None:
            ack(bool(parked))
