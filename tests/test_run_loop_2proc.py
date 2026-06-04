"""Phase-5 RUN-LOOP two-process abort/pause coherency (real ExptServer + real run loop + a
writer proc).

This is the run-loop-level companion to ``test_control_channel_2proc.py``. That file drives
:class:`ControlChannel` directly against a real :class:`ExptServer`; THIS file drives the actual
scan loop -- ``run_seq.run_scan_group`` and ``sequence_runner.run_job`` -- against the same real
server while a SEPARATE OS process (``control_writer_helper.py``) issues ``pause_seq`` /
``abort_seq`` / ``start_seq`` verbs over ZMQ. It is the "two-process integration test" the Phase-5
plan flags as the only thing the single-process unit tests can't cover: the verbs travel the wire,
are applied by the server's worker thread, and must be honored by the run loop at the per-sequence
gate -- proving the end-to-end control contract, not just its halves.

What it nails down that the unit tests (test_run_seq.py, all-fake control) and the control-channel
2proc test (no run loop) cannot:
  * a cross-process abort STOPS the scan mid-flight and yields ``status == "aborted"`` with a
    PARTIAL shot count (not the full scan, not zero);
  * a cross-process pause PARKS the loop and a later cross-process start RESUMES it to a full,
    ``"ok"`` completion;
  * an abort arriving WHILE parked on a pause wins (abort precedes pause);
  * the "aborted != ok" must-fix end to end: ``run_job`` records an aborted scan in the server
    queue history as ``status='aborted'`` / ``state='error'`` (NOT ``ok``/``done``);
  * the clear-at-job-start fix (bug-runjob-stale-abortrunseq / bug-pyctrl-stale-abort-wedge): a
    stale ``Abort`` left on the server does NOT wedge the next scan at 0 iterations -- ``begin_scan``
    clears it and the fresh scan runs to completion;
  * status coherency: the server reports "running" during the scan and "stopped" after an abort.

NO hardware / engine -- the scan's compile/run seams are injected fakes (a per-shot dwell so the
cross-process verbs land mid-scan); only ZMQ + a subprocess are real. Each test binds a fresh free
port and isolates the queue file to tmp_path.
"""

import os
import socket
import subprocess
import sys
import threading
import time

import pytest

import ExptServer as expt_mod
from ExptServer import ExptServer
from control_channel import ControlChannel, SeqRequest
from dyn_props import DynProps
from run_seq import run_scan_group
from seq_config import SeqConfig
from sequence_runner import run_job

pytestmark = pytest.mark.no_hardware

_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_writer_helper.py")

_DWELL = 0.02          # seconds per shot -- long enough for cross-process verbs to land mid-scan
_BIG = 300             # scan points for abort tests (aborted early -> wall-clock stays small)
_MED = 80              # scan points for pause/resume (runs to completion)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
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
    for attr in ("_ExptServer__sock",):
        try:
            getattr(srv, attr).close(linger=0)
        except Exception:
            pass
    try:
        srv._ExptServer__ctx.term()
    except Exception:
        pass


class _FakeRunSeq:
    """The product of compile_point; run_real reads/writes its ``C.RESTART`` field."""

    def __init__(self, tag):
        self.tag = tag
        self.C = DynProps({})


def _compile_point(seqfn, seqparam):
    return _FakeRunSeq(seqparam["A"]["B"])


class _DwellRunReal:
    """run_real(seq) that sleeps a fixed dwell per shot, so a separate process has time to land
    a control verb while the scan is in flight. Records the shots it ran."""

    def __init__(self, dwell):
        self.dwell = dwell
        self.runs = []

    def __call__(self, seq):
        self.runs.append(seq.tag)
        seq.C.RESTART = 0
        time.sleep(self.dwell)


class _StubScanGroup:
    """Minimal ScanGroup surface run_scan_group / run_job need, with no engine.

    ``runp()`` raises so sequence_runner._build_scan_order falls back to a plain sequential
    order (deterministic 1..n) -- the production scramble/stack prep is exercised elsewhere; here
    we want a predictable shot count to assert "partial" against.
    """

    def __init__(self, n):
        self._n = n

    def nseq(self):
        return self._n

    def getseq_with_var(self, arg0):
        return arg0, {"A": {"B": float(arg0)}}, []

    def runp(self):
        raise RuntimeError("no runp in this stub")


def _seqfn(s):
    return s


def _run_scan_in_thread(server, n, dwell=_DWELL, poll_interval=0.02):
    """Launch run_scan_group on a real ControlChannel over ``server`` in a background thread.

    Returns (thread, holder, run_real). ``holder['res']`` is filled with the result dict when the
    scan finishes. The scan is Running on the server (begin_scan -> start_scan) by the time the
    thread has stepped past its first line.
    """
    holder = {}
    rr = _DwellRunReal(dwell)
    control = ControlChannel(server, poll_interval=poll_interval)

    def target():
        holder["res"] = run_scan_group(
            _seqfn, _StubScanGroup(n), control=control,
            compile_point=_compile_point, run_real=rr, seq_config=SeqConfig())

    t = threading.Thread(target=target)
    t.start()
    return t, holder, rr


# --------------------------------------------------------------------------- #
# 1. cross-process abort stops the scan mid-flight; status coherency around it
# --------------------------------------------------------------------------- #
def test_abort_midscan_stops_with_partial_count(server):
    srv, url = server
    t, holder, rr = _run_scan_in_thread(srv, _BIG)
    try:
        # The run loop marks the server Running synchronously at begin_scan.
        assert _wait(lambda: srv.get_status() == "Sequence is running"), "scan never went Running"
        p = _writer(url, "0.2:abort_seq")
        p.wait(timeout=5)
        t.join(timeout=10)
        assert not t.is_alive(), "run loop did not stop after a cross-process abort"
        res = holder["res"]
        assert res["status"] == "aborted"
        assert 1 <= res["nseq"] < _BIG, "abort should stop mid-scan, not at 0 or full"
        assert res["nseq"] == len(rr.runs)            # counter matches shots actually run
        assert srv.check_request() == SeqRequest.Abort
        assert srv.get_status() == "Sequence is stopped"   # abort_seq returned the server to Init
    finally:
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# 2. cross-process pause parks the loop; a later cross-process start resumes it to full completion
# --------------------------------------------------------------------------- #
def test_pause_then_resume_completes_full_scan(server):
    srv, url = server
    t, holder, rr = _run_scan_in_thread(srv, _MED)
    try:
        assert _wait(lambda: srv.get_status() == "Sequence is running")
        p = _writer(url, "0.2:pause_seq,0.5:start_seq")
        # Coarse get_status reflects the pause REQUEST immediately...
        assert _wait(lambda: srv.get_status() == "Sequence is paused"), "pause request never landed"
        # ...while is_paused() (reached-paused ack) becomes True only once the run loop truly parks
        # at the gate (request-vs-reached must-fix #2).
        assert _wait(lambda: srv.is_paused() is True), "run loop never acked reached-paused"
        p.wait(timeout=5)
        t.join(timeout=10)
        res = holder["res"]
        assert res["status"] == "ok"
        assert res["nseq"] == _MED                    # resumed and finished every shot
        assert srv.check_request() == SeqRequest.NoRequest
        assert srv.is_paused() is False               # reached-paused ack reset on resume
    finally:
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# 3. an abort arriving while parked on a pause wins (abort precedes pause)
# --------------------------------------------------------------------------- #
def test_abort_while_paused_wins(server):
    srv, url = server
    t, holder, rr = _run_scan_in_thread(srv, _BIG)
    try:
        assert _wait(lambda: srv.get_status() == "Sequence is running")
        p = _writer(url, "0.2:pause_seq,0.5:abort_seq")
        assert _wait(lambda: srv.get_status() == "Sequence is paused")
        p.wait(timeout=5)
        t.join(timeout=10)
        res = holder["res"]
        assert res["status"] == "aborted"
        assert 1 <= res["nseq"] < _BIG
        assert srv.check_request() == SeqRequest.Abort
    finally:
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# 4. run_job end to end: an aborted scan is recorded as 'aborted'/'error', NOT 'ok'/'done'
#    (the "aborted != ok" must-fix, across the real queue + real server + cross-process abort).
# --------------------------------------------------------------------------- #
class _Disp:
    def __init__(self, scangroup):
        self.seq = _seqfn
        self.scangroup = scangroup
        self.opts = []
        self.label = "StubScan"
        self.seq_name = "StubScan"


def test_run_job_records_aborted_status(server):
    srv, url = server
    jid = srv.submit_job(b"payload")
    assert srv.pop_next_job()["id"] == jid           # mark it running in the queue

    sg = _StubScanGroup(_BIG)

    def dispatch(_descriptor_json):
        return _Disp(sg)

    def run(seq, scangroup, control, scan_name, **kw):
        # Drive the REAL scan loop with injected compile/run seams + the real control channel
        # run_job built over the real server. kw is the prep-derived run order (empty here ->
        # plain sequential, since the stub's runp() raises).
        return run_scan_group(
            _seqfn, scangroup, control=control, compile_point=_compile_point,
            run_real=_DwellRunReal(_DWELL), seq_config=SeqConfig(), **kw)

    holder = {}

    def target():
        holder["result"] = run_job(srv, "{}", job_id=jid, dispatch=dispatch, run=run)

    t = threading.Thread(target=target)
    t.start()
    try:
        assert _wait(lambda: srv.get_status() == "Sequence is running")
        p = _writer(url, "0.2:abort_seq")
        p.wait(timeout=5)
        t.join(timeout=10)
        assert not t.is_alive()
        assert holder["result"].status == "aborted"
        # The server queue history must reflect the abort, not a false 'ok'.
        hist = srv.queue_list()["history"]
        entry = next(e for e in hist if e["id"] == jid)
        assert entry["status"] == "aborted"
        assert entry["state"] == "error"             # finish_job: non-ok -> 'error', not 'done'
    finally:
        t.join(timeout=5)


# --------------------------------------------------------------------------- #
# 5. request-vs-reached ack: is_paused() (reached) is distinct from get_status (requested).
# --------------------------------------------------------------------------- #
def test_ack_paused_tracks_reached_state(server):
    """Request-vs-reached ack (must-fix #2), at the server contract level. The coarse get_status
    flips to 'paused' on the bare pause REQUEST (a lie mid-shot); is_paused() is the reached truth
    the runner acks only once it has actually parked."""
    srv, _url = server
    assert srv.is_paused() is False
    srv.start_scan()                                  # Running
    srv.pause_seq()                                   # request Pause
    assert srv.get_status() == "Sequence is paused"   # coarse: reflects the REQUEST...
    assert srv.is_paused() is False                   # ...but the runner has not parked yet
    srv.ack_paused(True)                              # runner reaches the gate and parks
    assert srv.is_paused() is True
    srv.start_seq_serv()                              # resume
    assert srv.is_paused() is False                   # reached-paused ack reset on resume
    # abort also un-parks the reached ack
    srv.start_scan(); srv.pause_seq(); srv.ack_paused(True)
    assert srv.is_paused() is True
    srv.abort_seq()
    assert srv.is_paused() is False


# --------------------------------------------------------------------------- #
# 6. clear-at-job-start: a STALE abort does not wedge the next scan (the wedge regression).
#    Reproduces bug-runjob-stale-abortrunseq / bug-pyctrl-stale-abort-wedge at the run-loop level:
#    an Abort left over from a prior scan must be cleared by begin_scan, so a fresh scan runs to
#    full completion instead of returning 0 iterations forever.
# --------------------------------------------------------------------------- #
def test_stale_abort_cleared_by_next_scan(server):
    srv, _url = server
    # Leave a stale Abort on the server, exactly as an abort between scans would (server -> Init,
    # SeqRequest=Abort). No cross-process writer needed: the point is what begin_scan does with a
    # pre-existing abort, and the in-process server verbs reproduce that state faithfully.
    srv.start_scan()                                  # Running
    srv.abort_seq()                                   # Init + SeqRequest=Abort (stale)
    assert srv.check_request() == SeqRequest.Abort

    # A fresh scan: the OLD "abort-sticky" begin_scan would refuse to start (0 iters, wedged);
    # the clear-at-job-start fix clears the stale abort and runs the whole scan.
    rr = _DwellRunReal(0.0)
    res = run_scan_group(
        _seqfn, _StubScanGroup(5), control=ControlChannel(srv),
        compile_point=_compile_point, run_real=rr, seq_config=SeqConfig())
    assert res == {"status": "ok", "nseq": 5}
    assert len(rr.runs) == 5
    assert srv.check_request() == SeqRequest.NoRequest   # cleared at job start
