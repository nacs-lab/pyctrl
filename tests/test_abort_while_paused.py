"""abort-while-paused: advance the queue but STAY paused (pyctrl backend).

User report: clicking Abort while a scan is PAUSED (to drop the current item and move on to
the next queued one) un-paused -- the next scan started RUNNING instead of staying paused.

Root cause: ``start_scan`` (the clear-at-job-start point) unconditionally reset the next job to
Running/NoRequest, so the pause was lost across the abort -> next-job boundary.

Fix: a one-shot ``__pause_after_abort`` intent -- ``abort_seq`` records it when the scan being
aborted is PAUSED; the next ``start_scan`` consumes it and begins the next job in Paused/Pause
(the run loop then parks at its first shot until the user hits play). It is one-shot and is also
dropped on idle (``clear_seq_request``), so it can never wedge a future scan
(cf. bug-runjob-stale-abortrunseq).

NO hardware / engine -- ExptServer ZMQ bind only.
    pytest pyctrl/tests/test_abort_while_paused.py
"""
import socket

import pytest

import ExptServer as expt_mod
from ExptServer import ExptServer

pytestmark = pytest.mark.no_hardware


def _free_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "tcp://127.0.0.1:%d" % port


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(expt_mod, "QUEUE_PATH", str(tmp_path / "runner_queue.json"))
    srv = ExptServer(_free_url())
    yield srv
    for teardown in (lambda: srv.stop_worker(),
                     lambda: srv._ExptServer__sock.close(linger=0),
                     lambda: srv._ExptServer__ctx.term()):
        try:
            teardown()
        except Exception:
            pass


# --- the fix: abort WHILE PAUSED -> next job stays paused ----------------------

def test_abort_while_paused_keeps_next_job_paused(server):
    server.start_scan()                       # job 1 -> Running
    server.pause_seq()                        # -> Paused
    assert server.get_status() == "Sequence is paused"

    server.abort_seq()                        # drop job 1, advance, but STAY paused
    assert server.check_request() == server.SeqRequest.Abort.value   # current scan aborts

    server.start_scan()                       # job 2 begins...
    assert server.get_status() == "Sequence is paused"               # ...PAUSED, not running
    assert server.check_request() == server.SeqRequest.Pause.value   # run loop will park
    assert server.is_paused() is False                               # requested, not yet reached


def test_resume_after_abort_while_paused_runs(server):
    # After the carried pause, a normal resume (start_seq_serv) runs the next job.
    server.start_scan()
    server.pause_seq()
    server.abort_seq()
    server.start_scan()                       # next job: Paused/Pause
    assert server.start_seq_serv() == "Sequence should now be running"
    assert server.get_status() == "Sequence is running"
    assert server.check_request() == server.SeqRequest.NoRequest.value


# --- regression: a plain abort (NOT paused) still advances RUNNING -------------

def test_abort_while_running_next_job_runs(server):
    server.start_scan()                       # Running (not paused)
    server.abort_seq()                        # plain abort
    assert server.check_request() == server.SeqRequest.Abort.value

    server.start_scan()                       # next job runs normally
    assert server.get_status() == "Sequence is running"
    assert server.check_request() == server.SeqRequest.NoRequest.value


# --- one-shot: the carry intent is consumed, never sticky ----------------------

def test_pause_carry_is_one_shot(server):
    server.start_scan()
    server.pause_seq()
    server.abort_seq()
    server.start_scan()                       # job 2: Paused (consumes the intent)
    assert server.get_status() == "Sequence is paused"

    server.start_seq_serv()                   # resume job 2 -> Running
    server.start_scan()                       # job 3: must be Running (intent already spent)
    assert server.get_status() == "Sequence is running"
    assert server.check_request() == server.SeqRequest.NoRequest.value


# --- empty-queue leak guard: idle drops a stale carry intent -------------------

def test_idle_clears_pause_carry(server):
    # abort-while-paused with NOTHING queued: the idle loop consumes the abort AND drops the
    # pause-carry, so a much-later submitted scan is NOT silently paused.
    server.start_scan()
    server.pause_seq()
    server.abort_seq()
    server.clear_seq_request()                # IdleScheduler abort-gate (empty queue)

    server.start_scan()                       # a later, freshly-submitted scan
    assert server.get_status() == "Sequence is running"
    assert server.check_request() == server.SeqRequest.NoRequest.value
