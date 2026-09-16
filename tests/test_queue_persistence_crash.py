"""Queue/job-id persistence must survive an UNCLEAN restart, not just a clean one.

Regression for the real failure seen on 2026-07-22, 2026-08-05 and 2026-09-15: the
computer was hard-reset mid-scan and the backend came up logging

    [ExptServer] warning: could not load ...\nacsctl\runner_queue.json:
        Expecting value: line 1 column 1 (char 0)

i.e. json.load on a ZERO-FILLED file. `os.replace` is atomic against a process
crash, but NTFS journals the rename without necessarily having flushed the data
blocks, so after a power loss the file exists at full length and reads back as
NULs. `__load_queue` treated that as "no state", so `__next_job_id` kept its
constructor default and scan ids restarted at 1 -- reusing ids the operator had
already seen -- and the whole 500-row history was gone.

Three guards, one test each below: flush+fsync before the rename, a `.bak`
rotation, and a separate few-byte id floor that survives losing both JSON files.

NO hardware / engine -- ExptServer ZMQ bind only:
    pytest pyctrl/tests/test_queue_persistence_crash.py
"""
import json
import os
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


def _close(srv):
    for fn in (lambda: srv.stop_worker(),
               lambda: srv._ExptServer__sock.close(linger=0),
               lambda: srv._ExptServer__ctx.term()):
        try:
            fn()
        except Exception:
            pass


@pytest.fixture
def make_server(tmp_path, monkeypatch):
    """Factory for successive ExptServers sharing ONE queue path -- i.e. restarts."""
    qpath = str(tmp_path / "runner_queue.json")
    monkeypatch.setattr(expt_mod, "QUEUE_PATH", qpath)
    created = []

    def _make():
        srv = ExptServer(_free_url())
        created.append(srv)
        return srv

    _make.path = qpath
    yield _make
    for srv in created:
        _close(srv)


def _zero_fill(path):
    """Exactly what the hard reset left behind: right length, all NUL bytes."""
    n = os.path.getsize(path)
    with open(path, "wb") as f:
        f.write(b"\x00" * n)


# --- guard 1: the write itself is durable -----------------------------------

def test_save_fsyncs_before_publishing(make_server, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(expt_mod.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    srv = make_server()
    srv.submit_job(b"{}")
    # The bytes must be on disk BEFORE the rename makes the file visible, or a
    # power loss re-creates the 2026-09-15 zero-filled file.
    assert synced, "queue was published without an fsync"


# --- guard 2: a zero-filled primary falls back to the rotated backup --------

def test_zero_filled_primary_recovers_from_bak(make_server, capsys):
    srv = make_server()
    srv.submit_job(b"{}")           # id 1
    srv.submit_job(b"{}")           # id 2 -- second save rotates the first to .bak
    _close(srv)

    qpath = make_server.path
    assert os.path.exists(qpath + ".bak"), "no backup was rotated aside"
    _zero_fill(qpath)               # the crash

    srv2 = make_server()
    assert "recovered queue state from backup" in capsys.readouterr().out
    # Ids continue -- nothing reuses 1 or 2 ...
    assert srv2.submit_job(b"{}") == 3
    # ... and the queue survives, minus the one generation the .bak is behind
    # (it was written before job 2 was queued). Losing one state transition is
    # the documented cost of the fallback; losing all 500 history rows was the bug.
    assert [e["id"] for e in srv2.queue_list()["queued"]] == [1, 3]


# --- guard 3: losing BOTH json files still keeps ids continuous -------------

def test_id_floor_survives_total_state_loss(make_server, capsys):
    srv = make_server()
    for _ in range(11):
        srv.submit_job(b"{}")       # ids 1..11
    _close(srv)

    qpath = make_server.path
    _zero_fill(qpath)
    _zero_fill(qpath + ".bak")      # worst case: nothing readable survives

    srv2 = make_server()
    out = capsys.readouterr().out
    assert "job ids resume at 12" in out
    # Queue/history are genuinely gone -- but the NEXT id must not reuse 1.
    assert srv2.queue_list()["queued"] == []
    assert srv2.submit_job(b"{}") == 12


# --- guard 4: a stale counter can't shadow the rows that are present --------

def test_next_job_id_never_below_highest_row(make_server):
    srv = make_server()
    srv.submit_job(b"{}", job_id=40)
    _close(srv)

    qpath = make_server.path
    with open(qpath) as f:
        data = json.load(f)
    data["next_job_id"] = 1         # corrupted/stale counter, rows intact
    with open(qpath, "w") as f:
        json.dump(data, f)
    os.remove(qpath + ".bak") if os.path.exists(qpath + ".bak") else None
    os.remove(expt_mod.QUEUE_PATH + ".idfloor")

    srv2 = make_server()
    # High-water mark over the rehydrated rows wins -> no collision with id 40.
    assert srv2.submit_job(b"{}") == 41


# --- unchanged behavior: a CLEAN restart still reloads everything -----------

def test_clean_restart_still_reloads_queue_and_history(make_server):
    srv = make_server()
    jid = srv.submit_job(b"{}")
    srv.pop_next_job()
    srv.finish_job(jid, "ok")
    _close(srv)

    srv2 = make_server()
    q = srv2.queue_list()
    assert [e["id"] for e in q["history"]] == [jid]
    assert srv2.submit_job(b"{}") == jid + 1
