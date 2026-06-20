"""ExptServer per-shot health tracker (the dashboard's "shots are failing" banner).

A run can be "Running" yet have EVERY shot error (e.g. a rearrange handoff that keeps failing
``setup_rearrangement``/``rearrange``). That otherwise reads on the dashboard as a bland
"Running / no data". ``ExptServer.record_shot_error`` + ``shot_health`` feed a banner that calls
it out, and ``reset_shot_health`` makes each new scan start clean.

These tests pin the contract the dashboard relies on:
  * record_shot_error accumulates a per-scan total + a bounded recent ring + the last message;
  * shot_health reports server-computed seconds_since_last (clock-skew-free for the consumer);
  * a record_shot_ok more recent than the last error is how the dashboard sees "recovered";
  * reset_shot_health zeroes the scan so a failing scan can't bleed into a healthy next one;
  * record/health never raise (advisory; must not perturb the run loop).

NO hardware / engine -- ExptServer ZMQ bind only.
    pytest pyctrl/tests/test_shot_health.py
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


def test_fresh_server_reports_no_failures(server):
    h = server.shot_health()
    assert h["total"] == 0
    assert h["seconds_since_last"] is None
    assert h["last_message"] == ""
    assert h["errors"] == []


def test_record_accumulates_total_and_last_message(server):
    server.record_shot_error("setup_rearrangement failed: HTTP 400",
                             scan_id="20260609_000001", seq_id=3, kind="setup_rearrangement")
    server.record_shot_error("rearrange call failed: HTTP 503", seq_id=4, kind="rearrange")
    h = server.shot_health()
    assert h["total"] == 2
    assert h["last_message"] == "rearrange call failed: HTTP 503"
    assert h["last_kind"] == "rearrange"
    assert h["seconds_since_last"] is not None and h["seconds_since_last"] >= 0
    assert [e["kind"] for e in h["errors"]] == ["setup_rearrangement", "rearrange"]
    assert h["errors"][0]["seq_id"] == 3


def test_recent_ring_is_bounded(server):
    for i in range(120):
        server.record_shot_error("err %d" % i)
    h = server.shot_health()
    assert h["total"] == 120            # total counts everything
    assert len(h["errors"]) == 10       # shot_health exposes only the last 10


def test_record_ok_lets_consumer_detect_recovery(server):
    server.record_shot_error("rearrange call failed")
    server.record_shot_ok()             # a success AFTER the error -> recovered
    h = server.shot_health()
    assert h["seconds_since_ok"] is not None
    # The dashboard's "recovered" rule: a success more recent than the last error.
    assert h["seconds_since_ok"] <= h["seconds_since_last"]


def test_reset_clears_for_a_new_scan(server):
    server.record_shot_error("boom")
    server.reset_shot_health("20260609_000099")
    h = server.shot_health()
    assert h["total"] == 0
    assert h["seconds_since_last"] is None
    assert h["last_message"] == ""
    assert h["scan_id"] == "20260609_000099"


def test_record_is_best_effort_and_never_raises(server):
    # Odd argument types must not bubble out of the advisory path.
    server.record_shot_error(object(), scan_id=None, seq_id=None, kind=None)
    assert server.shot_health()["total"] == 1
