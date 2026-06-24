"""Background (calibration) queue lane on ExptServer.

A background scan is a low-priority REAL scan: it runs only when no foreground scan is
running/queued, yields at a shot boundary when foreground work appears, and (when ``cycle``)
re-queues itself so calibrations cycle round-robin. These tests pin the ExptServer-side
primitives that make that safe:

  * foreground pops (``pop_next_job`` / ``pop_next_descriptor``) EXCLUDE background entries;
  * ``has_foreground_work`` counts queued foreground jobs AND not-yet-dispatched descriptors;
  * ``pop_next_background_*`` only see the background lane;
  * ``requeue_background`` re-queues a clean finish/yield (cycle) at the BACK (round-robin),
    archives an abort/error or a cycle-off scan, in ONE atomic transition (no lost/double job);
  * ``mark_idle_if_queue_empty`` treats a background-only queue as foreground-idle;
  * the global ``set/get_background_enabled`` toggle + the ``set_background_running`` indicator;
  * ``submit_scan_descriptor`` parses the additive ``background``/``cycle`` JSON keys and
    ``link_descriptor_to_job`` carries the lane onto the built job.

NO hardware / engine -- ExptServer ZMQ bind only.
    pytest pyctrl/tests/test_background_queue.py
"""
import json
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


def _bg_job(srv, jid, cycle=True):
    """Create a background JOB directly (the dispatch end-state) and return its id."""
    return srv.submit_job(b"{}", job_id=jid, priority="background", cycle=cycle)


# --- foreground pops exclude background --------------------------------------

def test_pop_next_job_skips_background(server):
    _bg_job(server, 1)
    server.submit_job(b"{}", job_id=2)                   # foreground
    job = server.pop_next_job()
    assert job is not None and job["id"] == 2            # foreground popped, background skipped
    assert server.pop_next_job() is None                 # background NOT returned by the fg pop


def test_pop_next_job_none_when_only_background(server):
    _bg_job(server, 1)
    assert server.pop_next_job() is None
    # but the background pop sees it
    bg = server.pop_next_background_job()
    assert bg is not None and bg["id"] == 1


def test_pop_next_descriptor_skips_background(server):
    server.submit_scan_descriptor(json.dumps({"seq": "Bg", "background": True}))
    server.submit_scan_descriptor(json.dumps({"seq": "Fg"}))
    d = server.pop_next_descriptor()
    assert d is not None and d["descriptor"].find('"Fg"') > 0   # foreground descriptor only
    assert server.pop_next_descriptor() is None
    # the background descriptor is reachable via the background pop
    bd = server.pop_next_background_descriptor()
    assert bd is not None and bd["descriptor"].find('"Bg"') > 0


# --- has_foreground_work -----------------------------------------------------

def test_has_foreground_work_false_with_only_background(server):
    _bg_job(server, 1)
    assert server.has_foreground_work() is False

def test_has_foreground_work_true_with_queued_foreground_job(server):
    server.submit_job(b"{}", job_id=1)
    assert server.has_foreground_work() is True

def test_has_foreground_work_counts_queued_foreground_descriptor(server):
    # A foreground scan submitted while a background job runs sits as a queued DESCRIPTOR
    # (not yet dispatched); the background run must still yield to it.
    server.submit_scan_descriptor(json.dumps({"seq": "Fg"}))
    assert server.has_foreground_work() is True

def test_has_foreground_work_true_while_foreground_running(server):
    server.submit_job(b"{}", job_id=1)
    server.pop_next_job()                                # -> running
    assert server.has_foreground_work() is True

def test_has_foreground_work_ignores_background_descriptor(server):
    server.submit_scan_descriptor(json.dumps({"seq": "Bg", "background": True}))
    assert server.has_foreground_work() is False


# --- requeue_background: cycle / archive -------------------------------------

def test_requeue_ok_cycle_requeues_at_back(server):
    a = _bg_job(server, 1, cycle=True)
    b = _bg_job(server, 2, cycle=True)
    first = server.pop_next_background_job()
    assert first["id"] == a
    assert server.requeue_background(a, "ok") is True
    # A went to the BACK -> the next background pop is B (round-robin), then A again.
    assert server.pop_next_background_job()["id"] == b
    assert server.requeue_background(b, "ok") is True
    assert server.pop_next_background_job()["id"] == a

def test_requeue_yield_cycles_like_ok(server):
    a = _bg_job(server, 1, cycle=True)
    server.pop_next_background_job()
    assert server.requeue_background(a, "yielded") is True
    # still queued (re-runnable), not archived
    assert server.pop_next_background_job()["id"] == a
    assert not any(e["id"] == a for e in server.queue_list()["history"])

def test_requeue_ok_no_cycle_archives_done(server):
    a = _bg_job(server, 1, cycle=False)
    server.pop_next_background_job()
    assert server.requeue_background(a, "ok") is True
    assert server.pop_next_background_job() is None      # not re-queued
    hist = server.queue_list()["history"]
    assert any(e["id"] == a and e["status"] == "ok" for e in hist)

def test_requeue_aborted_archives_error_even_if_cycle(server):
    a = _bg_job(server, 1, cycle=True)
    server.pop_next_background_job()
    assert server.requeue_background(a, "aborted") is True
    assert server.pop_next_background_job() is None      # aborted -> NOT re-queued
    hist = server.queue_list()["history"]
    assert any(e["id"] == a and e["state"] == "error" for e in hist)

def test_requeue_error_status_archives(server):
    a = _bg_job(server, 1, cycle=True)
    server.pop_next_background_job()
    assert server.requeue_background(a, "run error: boom") is True
    assert server.pop_next_background_job() is None
    assert any(e["id"] == a for e in server.queue_list()["history"])

def test_requeue_cycle_enabled_false_archives(server):
    a = _bg_job(server, 1, cycle=True)
    server.pop_next_background_job()
    # explicit global override (cycle_enabled=False) archives even a cyclable ok scan
    assert server.requeue_background(a, "ok", cycle_enabled=False) is True
    assert server.pop_next_background_job() is None


# --- status / idle semantics -------------------------------------------------

def test_mark_idle_resets_with_only_background(server):
    _bg_job(server, 1)
    server.start_scan()                                  # -> Running
    assert server.mark_idle_if_queue_empty() is True     # background-only == foreground-idle
    assert server.get_status() == "Sequence is stopped"

def test_mark_idle_keeps_running_with_foreground_queued(server):
    _bg_job(server, 1)
    server.submit_job(b"{}", job_id=2)                   # foreground pending
    server.start_scan()
    assert server.mark_idle_if_queue_empty() is False
    assert server.get_status() == "Sequence is running"


# --- global toggle + running indicator ---------------------------------------

def test_background_enabled_toggle_roundtrips(server):
    assert server.get_background_enabled() is True       # default on
    server.set_background_enabled(False)
    assert server.get_background_enabled() is False
    server.set_background_enabled(True)
    assert server.get_background_enabled() is True

def test_set_background_running_reflected_in_last_seq_status(server):
    server.set_background_running(True, "Spectrum556Scan_mj0")
    # last_seq_status is a ZMQ verb that reads the private fields; assert via name-mangled access.
    assert server._ExptServer__background_running is True
    assert server._ExptServer__background_name == "Spectrum556Scan_mj0"
    server.set_background_running(False)
    assert server._ExptServer__background_running is False
    assert server._ExptServer__background_name == ""


# --- descriptor lane parse + carry-onto-job ----------------------------------

def test_submit_descriptor_parses_background_keys(server):
    did = server.submit_scan_descriptor(
        json.dumps({"seq": "Bg", "background": True, "cycle": False}))
    e = next(x for x in server.queue_list()["queued"] if x["id"] == did)
    assert e["priority"] == "background"
    assert e["cycle"] is False

def test_normal_descriptor_is_normal_lane(server):
    did = server.submit_scan_descriptor(json.dumps({"seq": "Fg"}))
    e = next(x for x in server.queue_list()["queued"] if x["id"] == did)
    assert e["priority"] == "normal"

def test_link_carries_lane_onto_built_job(server):
    # Full dispatch flow: a background descriptor -> background job (priority + cycle carried).
    did = server.submit_scan_descriptor(
        json.dumps({"seq": "Bg", "background": True, "cycle": True}))
    bd = server.pop_next_background_descriptor()
    assert bd["id"] == did
    jid = server.submit_job(b'{"seq":"Bg"}', job_id=did, priority="background", cycle=True)
    assert server.link_descriptor_to_job(did, jid) is True
    job_row = next(x for x in server.queue_list()["queued"] if x["id"] == jid)
    assert job_row["priority"] == "background"
    assert job_row["cycle"] is True
    assert job_row["kind"] == "job"


# --- persistence: lane survives a reload -------------------------------------

def test_lane_survives_reload(server, tmp_path, monkeypatch):
    _bg_job(server, 1, cycle=True)
    # Reload into a fresh server pointed at the SAME queue file.
    srv2 = ExptServer(_free_url())
    try:
        e = next(x for x in srv2.queue_list()["queued"] if x["id"] == 1)
        assert e["priority"] == "background"
        assert e["cycle"] is True
    finally:
        srv2.stop_worker()
        srv2._ExptServer__sock.close(linger=0)
        srv2._ExptServer__ctx.term()
