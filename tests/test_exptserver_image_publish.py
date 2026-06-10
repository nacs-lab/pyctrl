"""ExptServer async image persistence -- the single FIFO worker behind publish_shot/stage_frame.

NO hardware/engine (ExptServer ZMQ bind only, like test_shot_health). numpy is needed because the
encode path is the real ``to_store_array``. We assert via the published deque: ``nseq`` counts
finished shots and each ``seq_finish`` appends exactly one ``b''`` separator, so counting those is
a robust check of "how many shots published, in order".
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


def _seps(srv):
    return list(srv.expt_imgs).count(b'')          # one b'' per published shot


def _frame(np, v=0.0):
    return np.full((4, 4), v, dtype=float)


# --------------------------------------------------------------------------- #
# publish_shot: whole-shot (the default capture path)
# --------------------------------------------------------------------------- #
def test_publish_shot_async_publishes_one_shot(server):
    np = pytest.importorskip("numpy")
    server.publish_shot([_frame(np), _frame(np)], 1234, 7)
    server.drain_images()
    assert server.nseq == 1
    assert _seps(server) == 1
    assert server.temp_imgs == []                  # staged + flushed, nothing left


def test_publish_shot_sync_runs_inline_no_drain(server):
    np = pytest.importorskip("numpy")
    server.publish_shot([_frame(np)], 1, 1, async_=False)
    assert server.nseq == 1                         # done already, no drain needed
    assert _seps(server) == 1


def test_fifo_two_shots(server):
    np = pytest.importorskip("numpy")
    server.publish_shot([_frame(np, 1)], 1, 1)
    server.publish_shot([_frame(np, 2)], 1, 2)
    server.drain_images()
    assert server.nseq == 2 and _seps(server) == 2


# --------------------------------------------------------------------------- #
# incremental: stage frame-by-frame then finish (the rearrange path)
# --------------------------------------------------------------------------- #
def test_incremental_stage_then_finish(server):
    np = pytest.importorskip("numpy")
    server.stage_frame(_frame(np, 1), 1, 5)         # img1 (mid-shot)
    server.stage_frame(_frame(np, 2), 1, 5)         # img2 (end)
    server.finish_shot()
    server.drain_images()
    assert server.nseq == 1 and _seps(server) == 1


# --------------------------------------------------------------------------- #
# a persist error drops the shot; the worker survives for the next one
# --------------------------------------------------------------------------- #
def test_persist_error_cancels_shot_and_worker_survives(server):
    np = pytest.importorskip("numpy")
    server.publish_shot([np.zeros(5)], 1, 1)        # 1-D -> to_store_array raises -> seq_cancel
    server.drain_images()
    assert server.nseq == 0 and _seps(server) == 0  # nothing published
    server.publish_shot([_frame(np)], 1, 2)         # worker still alive -> next shot publishes
    server.drain_images()
    assert server.nseq == 1 and _seps(server) == 1
