"""pyctrl id-reuse: a scan carries ONE id (the descriptor id), not desc + job.

pyctrl is both producer and consumer of the queue. Before this change,
``submit_scan_descriptor`` minted a descriptor id (N) and the run loop's
``handle_descriptor_pop`` minted a SECOND id (N+1) for the job, archiving the
descriptor to history -- so a single scan showed two rows / two ids (the
descriptor "ending in 0 s" plus its job). The script (e.g. ``LACScan.py``)
printed N while the dashboard showed N+1.

The run loop now reuses the descriptor's id for the job
(``submit_job(job_id=desc_id)``) and ``link_descriptor_to_job`` DROPS the
descriptor row (its same-id branch) instead of archiving a duplicate. So the id
the script returns is the id the queue/dashboard/tkinter all show -- one id, one
row -- for the pyctrl backend.

This behavior is pyctrl-only by construction: only the pyctrl run loop passes
``job_id``; the default ``submit_job`` path (the MATLAB ".m run-button") mints a
fresh id and archives a distinct-id descriptor, exactly as before.

NO hardware / engine -- ExptServer ZMQ bind only. Run in any pyctrl interpreter:
    pytest pyctrl/tests/test_descriptor_id_reuse.py
"""
import json
import socket

import pytest

import ExptServer as expt_mod
from ExptServer import ExptServer
from runner import handle_descriptor_pop

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
    try:
        srv.stop_worker()
    except Exception:
        pass
    try:
        srv._ExptServer__sock.close(linger=0)
    except Exception:
        pass
    try:
        srv._ExptServer__ctx.term()
    except Exception:
        pass


def _ids(entries):
    return [e["id"] for e in entries]


# --- submit_job(job_id=) primitive -----------------------------------------

def test_submit_job_reuses_given_id(server):
    jid = server.submit_job(b"{}", job_id=7)
    assert jid == 7
    job = server.pop_next_job()
    assert job["id"] == 7


def test_submit_job_advances_counter_past_reused_id(server):
    server.submit_job(b"{}", job_id=50)
    # The next freshly-minted job must NOT collide with the reused 50.
    fresh = server.submit_job(b"{}")
    assert fresh > 50


def test_submit_job_without_id_still_mints(server):
    # MATLAB-path behavior unchanged: no job_id -> counter mint, starting at 1.
    assert server.submit_job(b"{}") == 1
    assert server.submit_job(b"{}") == 2


# --- link_descriptor_to_job same-id drop ------------------------------------

def test_link_same_id_drops_descriptor_without_archiving(server):
    did = server.submit_scan_descriptor(json.dumps({"seq": "A"}))
    # Pop -> 'building', then reuse the id for the job and link.
    server.pop_next_descriptor()
    job_id = server.submit_job(b'{"seq":"A"}', job_id=did)
    assert job_id == did
    assert server.link_descriptor_to_job(did, job_id) is True
    q = server.queue_list()
    # No descriptor anywhere (not archived to history); exactly one job, id == did.
    assert all(e.get("kind") != "descriptor"
               for e in q["queued"] + q["history"])
    assert _ids(q["queued"]) == [did]


def test_link_distinct_id_still_archives(server):
    # Defensive: the MATLAB-style distinct-id path is untouched.
    did = server.submit_scan_descriptor(json.dumps({"seq": "A"}))
    server.pop_next_descriptor()
    job_id = server.submit_job(b'{"seq":"A"}')      # fresh id != did
    assert job_id != did
    assert server.link_descriptor_to_job(did, job_id) is True
    hist = server.queue_list()["history"]
    desc_rows = [e for e in hist if e.get("kind") == "descriptor"]
    assert len(desc_rows) == 1
    assert desc_rows[0]["built_job_id"] == job_id


# --- end to end: submit_scan_descriptor -> handle_descriptor_pop ------------

def test_dispatch_yields_single_id_no_descriptor_row(server):
    did = server.submit_scan_descriptor(
        json.dumps({"seq": "TweezerLoadingSeq", "label": "LACScan"}),
        label="LACScan")
    assert handle_descriptor_pop(server) == 1
    q = server.queue_list()
    # The scan is one queued job carrying the descriptor's id; no descriptor row
    # lingers in queued or history.
    assert _ids(q["queued"]) == [did]
    assert q["queued"][0]["kind"] == "job"
    assert all(e.get("kind") != "descriptor"
               for e in q["queued"] + q["history"])


def test_two_scans_keep_consecutive_single_ids(server):
    d1 = server.submit_scan_descriptor(json.dumps({"seq": "A"}))
    handle_descriptor_pop(server)
    d2 = server.submit_scan_descriptor(json.dumps({"seq": "B"}))
    handle_descriptor_pop(server)
    # Distinct, gap-free ids; each scan is exactly one job row.
    assert d2 == d1 + 1
    queued = server.queue_list()["queued"]
    assert _ids(queued) == [d1, d2]
    assert all(e["kind"] == "job" for e in queued)
