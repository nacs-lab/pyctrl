"""Phase-5 control_channel TWO-PROCESS abort/pause coherency (real ExptServer + a writer proc).

A genuine cross-process integration test: a real :class:`ExptServer` is bound to a localhost port
in THIS process, its :class:`ControlChannel` polls ``check_request`` at the run-loop gate, while a
SEPARATE OS process (``control_writer_helper.py``) issues ``pause_seq`` / ``abort_seq`` /
``start_seq`` verbs over ZMQ. This exercises the real worker-thread-vs-gate coherency the
in-process unit tests (test_control_channel.py) can't: the verbs travel the wire and are applied
by the server's worker thread concurrently with the gate.

NO hardware / engine -- just ZMQ + a subprocess (zmq is present in both interpreters). Each test
binds a fresh free port and isolates the queue file to tmp_path.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

import ExptServer as expt_mod
from ExptServer import ExptServer
from control_channel import ControlChannel, SeqRequest

pytestmark = pytest.mark.no_hardware

_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_writer_helper.py")


def _free_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "tcp://127.0.0.1:%d" % port


def _wait(pred, timeout=4.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _writer(url, script):
    return subprocess.Popen([sys.executable, _HELPER, url, script])


@pytest.fixture
def server(tmp_path, monkeypatch):
    # Isolate the persisted queue so concurrent/other tests don't collide.
    monkeypatch.setattr(expt_mod, "QUEUE_PATH", str(tmp_path / "runner_queue.json"))
    url = _free_url()
    srv = ExptServer(url)
    yield srv, url
    try:
        srv.stop_worker()
    except Exception:
        pass
    # Best-effort socket/context teardown (name-mangled privates) so the port frees promptly.
    for attr in ("_ExptServer__sock",):
        try:
            getattr(srv, attr).close(linger=0)
        except Exception:
            pass
    try:
        srv._ExptServer__ctx.term()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# pause -> resume: the gate parks, then a cross-process start_seq releases it
# --------------------------------------------------------------------------- #
def test_pause_then_resume(server):
    srv, url = server
    control = ControlChannel(srv, poll_interval=0.05)
    assert control.begin_scan() is not None                 # Running
    p = _writer(url, "0.3:pause_seq,0.6:start_seq")
    try:
        assert _wait(lambda: srv.check_request() == SeqRequest.Pause), "pause never landed"
        t0 = time.time()
        aborted = control.check_pause_abort()               # parks until the start_seq lands
        assert aborted is False                             # resumed, NOT aborted
        assert time.time() - t0 >= 0.2                      # it actually parked
        assert srv.check_request() == SeqRequest.NoRequest  # request cleared by start_seq
    finally:
        p.wait(timeout=5)


# --------------------------------------------------------------------------- #
# abort: a cross-process abort_seq makes the gate return True (stop)
# --------------------------------------------------------------------------- #
def test_abort_stops(server):
    srv, url = server
    control = ControlChannel(srv, poll_interval=0.05)
    control.begin_scan()
    p = _writer(url, "0.3:abort_seq")
    try:
        aborted = False
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if control.check_pause_abort():
                aborted = True
                break
            time.sleep(0.05)
        assert aborted is True
        assert srv.check_request() == SeqRequest.Abort
    finally:
        p.wait(timeout=5)


# --------------------------------------------------------------------------- #
# abort precedes pause: an abort arriving WHILE parked wins
# --------------------------------------------------------------------------- #
def test_abort_while_paused_wins(server):
    srv, url = server
    control = ControlChannel(srv, poll_interval=0.05)
    control.begin_scan()
    p = _writer(url, "0.3:pause_seq,0.6:abort_seq")
    try:
        assert _wait(lambda: srv.check_request() == SeqRequest.Pause)
        t0 = time.time()
        aborted = control.check_pause_abort()               # parks on pause; abort breaks it out
        assert aborted is True
        assert time.time() - t0 >= 0.2
    finally:
        p.wait(timeout=5)


# --------------------------------------------------------------------------- #
# clear-at-job-start: a stale abort is cleared by the next begin_scan (no wedge)
# --------------------------------------------------------------------------- #
def test_begin_scan_clears_stale_abort(server):
    srv, url = server
    control = ControlChannel(srv, poll_interval=0.05)
    control.begin_scan()                                    # Running, so abort_seq takes effect
    p = _writer(url, "0.0:abort_seq")
    p.wait(timeout=5)
    assert _wait(lambda: srv.check_request() == SeqRequest.Abort)
    # Single clear-point (clear-at-job-start): the next begin_scan clears the stale abort and
    # starts -- a fresh submission must not be wedged by a prior abort.
    assert control.begin_scan() is not None
    assert srv.check_request() == SeqRequest.NoRequest
