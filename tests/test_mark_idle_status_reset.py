"""mark_idle_if_queue_empty: end-of-job idle status reset (bug-pyctrl-status-not-reset-idle).

After a finite scan completes, ``start_scan`` left ``seq_status == Running`` and nothing reset
it, so ``get_status`` reported "Sequence is running" forever (the dashboard's stuck-"running").
``ExptServer.mark_idle_if_queue_empty`` (called by the run loop right after each job) returns the
status to Init iff no work remains queued.

These tests pin the contract that makes it SAFE:
  * status-only -- it must NOT set ``__seq_req`` (an Abort there would poison the next job's
    ``check_request``; cf. bug-runjob-stale-abortrunseq), unlike ``abort_seq``;
  * it resets ONLY from Running (never a Paused or already-idle scan);
  * a still-queued job OR a not-yet-drained descriptor counts as pending (no reset);
  * race-safe vs. a concurrent ``submit_job``: the queue check + reset are atomic under
    ``__queue_lock`` -- the stress test asserts no deadlock and no lost jobs under contention.

NO hardware / engine -- ExptServer ZMQ bind only.
    pytest pyctrl/tests/test_mark_idle_status_reset.py
"""
import json
import socket
import threading
import time

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


# --- core: reset Running -> Init only when the queue is empty -----------------

def test_resets_running_to_init_when_queue_empty(server):
    server.start_scan()                                   # seq_status -> Running
    assert server.get_status() == "Sequence is running"
    assert server.mark_idle_if_queue_empty() is True
    assert server.get_status() == "Sequence is stopped"


def test_keeps_running_when_a_job_is_queued(server):
    server.start_scan()
    server.submit_job(b"{}")                              # work still pending
    assert server.mark_idle_if_queue_empty() is False
    assert server.get_status() == "Sequence is running"


def test_keeps_running_when_a_descriptor_is_queued(server):
    # A not-yet-drained descriptor is pending work too (it becomes a job next loop iter).
    server.start_scan()
    server.submit_scan_descriptor(json.dumps({"seq": "A"}))
    assert server.mark_idle_if_queue_empty() is False
    assert server.get_status() == "Sequence is running"


def test_keeps_running_while_a_job_is_running(server):
    # pop_next_job marks the entry 'running'; an in-flight job must not be reset to idle.
    server.start_scan()
    server.submit_job(b"{}")
    server.pop_next_job()                                 # state -> 'running'
    assert server.mark_idle_if_queue_empty() is False
    assert server.get_status() == "Sequence is running"


# --- safety guarantee 1: never touches the seq request (no abort poison) ------

def test_does_not_set_abort_request(server):
    """Unlike abort_seq, the idle reset must leave __seq_req at NoRequest, or the next job's
    check_request would see a stale Abort and 0-iteration (bug-runjob-stale-abortrunseq)."""
    server.start_scan()                                   # clears request to NoRequest
    no_request = server.SeqRequest.NoRequest.value
    assert server.check_request() == no_request
    server.mark_idle_if_queue_empty()
    assert server.check_request() == no_request           # STILL NoRequest -- unchanged

    # Contrast: abort_seq DOES set the request (this is exactly what we must not do).
    server.start_scan()
    server.abort_seq()
    assert server.check_request() == server.SeqRequest.Abort.value


# --- safety guarantee 2: only resets from Running -----------------------------

def test_does_not_reset_a_paused_scan(server):
    server.start_scan()
    server.pause_seq()                                    # Running -> Paused
    assert server.get_status() == "Sequence is paused"
    assert server.mark_idle_if_queue_empty() is False     # empty queue, but NOT Running
    assert server.get_status() == "Sequence is paused"    # left alone


def test_noop_when_already_idle(server):
    # Fresh server is Init; resetting an already-stopped backend is a harmless no-op.
    assert server.get_status() == "Sequence is stopped"
    assert server.mark_idle_if_queue_empty() is False
    assert server.get_status() == "Sequence is stopped"


# --- race-safety: concurrent submit + consume must not deadlock or lose jobs ---

def test_concurrent_submit_and_mark_idle(server):
    """Hammer submit_job (one thread) against the consume shape pop/start/finish/mark_idle
    (main thread). The queue check + status reset are atomic under __queue_lock, which every
    submit also holds, so this must (a) never deadlock and (b) consume every submitted job."""
    srv = server
    N = 150

    def submitter():
        for _ in range(N):
            srv.submit_job(b"{}")

    t = threading.Thread(target=submitter)
    t.start()

    popped = 0
    t0 = time.time()
    while (popped < N or t.is_alive()) and time.time() - t0 < 20:
        job = srv.pop_next_job()
        if job is None:
            srv.mark_idle_if_queue_empty()                # empty momentarily between submits
            continue
        srv.start_scan()                                  # -> Running (real loop does this)
        srv.finish_job(job["id"], "ok")                   # job leaves __queue
        srv.mark_idle_if_queue_empty()                    # end-of-job reset, race vs. submit
        popped += 1

    t.join(timeout=10)
    assert not t.is_alive(), "submitter/consumer deadlocked"
    assert popped == N, "lost a job under concurrent submit"          # no corruption
    # Queue fully drained and the submitter is done -> status back to idle.
    assert srv.mark_idle_if_queue_empty() in (True, False)            # final call is safe
    assert srv.get_status() == "Sequence is stopped"
