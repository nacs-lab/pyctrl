"""No client request may be silently dropped by the reply path.

Regression test for the ``finish_recv`` bug: before every reply, the server
"flushed" the socket with bare ``recv(NOBLOCK)`` until empty -- which also
consumed every OTHER client's request that had queued while the handler ran.
Each eaten request left its REQ client hanging until its own timeout (the
monitor's get_imgs stalling ~30 s at scan boundaries / mid-scan dashboard
freezes; ~1-3% of all requests lost under normal dashboard polling load).

The fix drains only the CURRENT message's unconsumed frames (RCVMORE), so
concurrent clients hammering the server must get a reply for EVERY request.
NO hardware/engine -- ExptServer ZMQ bind only, like test_exptserver_image_publish.
"""
import socket
import threading

import pytest
import zmq

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
def server_url(tmp_path, monkeypatch):
    monkeypatch.setattr(expt_mod, "QUEUE_PATH", str(tmp_path / "runner_queue.json"))
    url = _free_url()
    srv = ExptServer(url)
    yield url, srv
    for teardown in (lambda: srv.stop_worker(),
                     lambda: srv._ExptServer__sock.close(linger=0),
                     lambda: srv._ExptServer__ctx.term()):
        try:
            teardown()
        except Exception:
            pass


def _client_worker(ctx, url, n_requests, verbs, timeout_ms, results, idx):
    """One REQ client: n_requests round-trips, records losses (no reply)."""
    lost = 0
    slow = 0
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    try:
        for k in range(n_requests):
            verb = verbs[k % len(verbs)]
            sock.send_string(verb)
            if sock.poll(timeout_ms) == 0:
                lost += 1
                # REQ is now stuck in recv state; reconnect to continue.
                sock.close(linger=0)
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(url)
                continue
            sock.recv_multipart()
    finally:
        sock.close(linger=0)
    results[idx] = (lost, slow)


def test_concurrent_clients_lose_no_requests(server_url):
    """4 clients x 60 mixed requests, all in parallel. Every request must be
    answered. With the old blind drain this loses tens of requests (any
    request queued while another client's reply was being prepared was
    discarded); with the RCVMORE-bounded drain it must be zero."""
    url, srv = server_url
    # Give queue_list some real work so handler occupancy windows exist
    # (mirrors the production 500-entry history the dashboard polls).
    for i in range(200):
        srv.submit_job(b'x' * 200, summary={'seqName': 'fake%d' % i})

    ctx = zmq.Context()
    n_clients = 4
    n_req = 60
    verbs = ["ping", "queue_list", "get_status", "get_num_imgs"]
    results = [None] * n_clients
    threads = [threading.Thread(
        target=_client_worker,
        args=(ctx, url, n_req, verbs[i:] + verbs[:i], 5000, results, i))
        for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    ctx.term()

    assert all(r is not None for r in results), "client thread hung"
    total_lost = sum(r[0] for r in results)
    assert total_lost == 0, (
        "%d requests were never answered (results per client: %s)"
        % (total_lost, results))


def test_multipart_remainder_still_flushed(server_url):
    """The RCVMORE drain must still discard unconsumed frames of the CURRENT
    request so a malformed multipart message cannot desync the stream: a
    submit_job missing its payload gets the error reply, and the socket keeps
    serving follow-up requests."""
    url, _srv = server_url
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    try:
        # Well-formed multi-frame verb first (payload + summary), then pings.
        sock.send_string("submit_job", zmq.SNDMORE)
        sock.send(b'payload-bytes', zmq.SNDMORE)
        sock.send_string('{"seqName": "t"}')
        assert sock.poll(5000), "submit_job got no reply"
        rep = sock.recv_multipart()
        assert rep, "empty submit_job reply"
        for _ in range(5):
            sock.send_string("ping")
            assert sock.poll(5000), "ping lost after multipart traffic"
            assert sock.recv_multipart()[-1] == b"pong"
    finally:
        sock.close(linger=0)
        ctx.term()
