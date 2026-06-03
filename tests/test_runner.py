"""Phase-5 runner: the scenario-3 run-loop HOST orchestration (YbExptCtrl/runner.py).

NO-HARDWARE: every device/engine/socket dependency is injected, so URL resolution, the
single-backend guard, the descriptor->job dispatch, the consume loop, the camera-command
pump, and the clean-terminate teardown are all exercised with fakes -- no engine load, no
bound socket, no camera. The live ``serve()`` wiring (real ExptServer + engine + pylablib)
is the NEEDS-HARDWARE entry and is not run here.
"""

import signal

import pytest

import runner

pytestmark = pytest.mark.no_hardware


# --------------------------------------------------------------------------- #
# resolve_url -- argv -> $NACS_RUNNER_URL -> DEFAULT_URL
# --------------------------------------------------------------------------- #
class TestResolveUrl:
    def test_argv_wins(self, monkeypatch):
        monkeypatch.setenv("NACS_RUNNER_URL", "tcp://env:1")
        assert runner.resolve_url(["tcp://argv:9"]) == "tcp://argv:9"

    def test_empty_argv_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("NACS_RUNNER_URL", "tcp://env:1")
        assert runner.resolve_url([]) == "tcp://env:1"
        assert runner.resolve_url(["  "]) == "tcp://env:1"   # blank argv ignored

    def test_no_argv_no_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("NACS_RUNNER_URL", raising=False)
        assert runner.resolve_url([]) == runner.DEFAULT_URL


# --------------------------------------------------------------------------- #
# assert_single_backend -- mutual-exclusion guard
# --------------------------------------------------------------------------- #
class TestSingleBackendGuard:
    def test_raises_when_a_backend_answers(self):
        with pytest.raises(RuntimeError):
            runner.assert_single_backend("tcp://x:1", ping=lambda url: True)

    def test_ok_when_nothing_answers(self):
        runner.assert_single_backend("tcp://x:1", ping=lambda url: False)  # no raise


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeQueueServer:
    """An ExptServer-shaped fake for the descriptor/job queue + dummy flag."""

    def __init__(self, descriptors=(), jobs=()):
        self._descs = list(descriptors)        # each {'id','descriptor'}
        self._jobs = list(jobs)                 # each {'id','payload'}
        self._next_job_id = 100
        self.submitted = []                     # [(job_id, payload), ...]
        self.linked = []                        # [(desc_id, job_id), ...]
        self.desc_finished = []                 # [(desc_id, status, msg), ...]
        self.dummy_running = []                 # [flag, ...]
        self.submit_should_raise = False

    def pop_next_descriptor(self):
        return self._descs.pop(0) if self._descs else None

    def submit_job(self, payload):
        if self.submit_should_raise:
            raise RuntimeError("submit boom")
        jid = self._next_job_id
        self._next_job_id += 1
        self.submitted.append((jid, payload))
        return jid

    def link_descriptor_to_job(self, desc_id, job_id):
        self.linked.append((desc_id, job_id))
        return True

    def finish_descriptor(self, desc_id, status, msg=None):
        self.desc_finished.append((desc_id, status, msg))
        return True

    def pop_next_job(self):
        return self._jobs.pop(0) if self._jobs else None

    def set_dummy_running(self, flag):
        self.dummy_running.append(flag)


# --------------------------------------------------------------------------- #
# handle_descriptor_pop -- descriptor -> JSON job (+ link); bad ones marked error
# --------------------------------------------------------------------------- #
class TestHandleDescriptorPop:
    def test_dispatches_and_links(self):
        srv = FakeQueueServer(descriptors=[
            {"id": 1, "descriptor": '{"seq":"A"}'},
            {"id": 2, "descriptor": b'{"seq":"B"}'},   # bytes pass through unchanged
        ])
        n = runner.handle_descriptor_pop(srv)
        assert n == 2
        # JSON string is utf-8 encoded; bytes are forwarded as-is.
        assert srv.submitted == [(100, b'{"seq":"A"}'), (101, b'{"seq":"B"}')]
        assert srv.linked == [(1, 100), (2, 101)]
        assert srv.desc_finished == []

    def test_empty_queue_returns_zero(self):
        srv = FakeQueueServer()
        assert runner.handle_descriptor_pop(srv) == 0
        assert srv.submitted == []

    def test_bad_descriptor_marked_error_and_keeps_draining(self):
        srv = FakeQueueServer(descriptors=[
            {"id": 7, "descriptor": '{"seq":"X"}'},
            {"id": 8, "descriptor": '{"seq":"Y"}'},
        ])
        srv.submit_should_raise = True
        n = runner.handle_descriptor_pop(srv)
        assert n == 0
        assert [d[0] for d in srv.desc_finished] == [7, 8]   # both reported, loop survived
        assert all(d[1] == "error" for d in srv.desc_finished)

    def test_cap_bounds_one_call(self):
        srv = FakeQueueServer(descriptors=[
            {"id": i, "descriptor": "{}"} for i in range(5)])
        assert runner.handle_descriptor_pop(srv, max_per_iter=2) == 2
        assert len(srv.submitted) == 2                        # only 2 drained this call

    def test_pop_failure_aborts_call(self):
        srv = FakeQueueServer()

        def boom():
            raise RuntimeError("pop boom")

        srv.pop_next_descriptor = boom
        assert runner.handle_descriptor_pop(srv) == 0         # graceful, no raise


# --------------------------------------------------------------------------- #
# consume_loop -- job run vs idle, camera pump, stop predicate
# --------------------------------------------------------------------------- #
class TestConsumeLoop:
    def test_runs_a_job_with_engine_run_kwargs(self):
        srv = FakeQueueServer(jobs=[{"id": 42, "payload": b"{}"}])
        calls = []

        def fake_run_job(server, payload, job_id=None, **kw):
            calls.append((payload, job_id, kw))
            return type("R", (), {"status": "ok"})()

        runner.consume_loop(
            srv,
            should_stop=_stop_after(1),
            run_job_fn=fake_run_job,
            dispatch_pop=lambda s: 0,
            run_kwargs={"run": "ENGINE"},
        )
        assert calls == [(b"{}", 42, {"run": "ENGINE"})]
        assert srv.dummy_running == [0]                        # job -> dummy-running off

    def test_empty_queue_steps_idle(self):
        srv = FakeQueueServer()
        steps = []
        idle = type("Idle", (), {"step": lambda self, sleep: steps.append("step")})()
        runner.consume_loop(
            srv, should_stop=_stop_after(1), dispatch_pop=lambda s: 0,
            idle=idle, sleep=lambda dt: None)
        assert steps == ["step"]
        assert srv.dummy_running == [1]                        # idle -> dummy-running on

    def test_camera_pump_invoked_each_iter(self):
        srv = FakeQueueServer()
        seen = []
        runner.consume_loop(
            srv, should_stop=_stop_after(1), dispatch_pop=lambda s: 0,
            handle_camera=lambda server, cam: seen.append(cam), camera="CAM",
            sleep=lambda dt: None)
        assert seen == ["CAM"]

    def test_dispatch_pop_called_before_job_pop(self):
        srv = FakeQueueServer(jobs=[{"id": 1, "payload": b"{}"}])
        order = []
        runner.consume_loop(
            srv, should_stop=_stop_after(1),
            dispatch_pop=lambda s: order.append("disp"),
            run_job_fn=lambda *a, **k: order.append("run") or None,
            sleep=lambda dt: None)
        assert order == ["disp", "run"]

    def test_pop_job_failure_is_survived(self):
        srv = FakeQueueServer()

        def boom():
            raise RuntimeError("pop_next_job boom")

        srv.pop_next_job = boom
        slept = []
        # stop after the first failed pop; the loop must sleep and not raise.
        runner.consume_loop(
            srv, should_stop=_stop_after(1), dispatch_pop=lambda s: 0,
            sleep=lambda dt: slept.append(dt))
        assert slept == [1.0]


# --------------------------------------------------------------------------- #
# handle_camera_cmd -- camera-command pump (fake camera)
# --------------------------------------------------------------------------- #
class FakeCameraServer:
    def __init__(self, cmd):
        self._cmd = cmd
        self.results = []                       # [(connected, roi, error, exposure), ...]

    def get_camera_cmd(self):
        c, self._cmd = self._cmd, None
        return c

    def set_camera_result(self, connected, roi, error="", exposure_time=None):
        self.results.append((connected, list(roi), error, exposure_time))


class FakeCamera:
    def __init__(self, raise_on=None):
        self.raise_on = raise_on
        self.closed = False

    def init(self, roi, exposure):
        if self.raise_on == "init":
            raise RuntimeError("init fail")
        return [1, 2, 3, 4], 0.05

    def apply_settings(self, roi, exposure):
        return [5, 6, 7, 8], 0.02

    def current_roi(self):
        return [9, 9, 9, 9]

    def close(self):
        self.closed = True


class TestHandleCameraCmd:
    def test_no_cmd_no_result(self):
        srv = FakeCameraServer(None)
        runner.handle_camera_cmd(srv, None)
        assert srv.results == []

    def test_close_without_camera_acks(self):
        srv = FakeCameraServer({"cmd": "close"})
        runner.handle_camera_cmd(srv, None)
        assert srv.results == [(False, [0, 0, 0, 0], "", None)]

    def test_close_releases_camera(self):
        srv = FakeCameraServer({"cmd": "close"})
        cam = FakeCamera()
        runner.handle_camera_cmd(srv, cam)
        assert cam.closed is True
        assert srv.results == [(False, [0, 0, 0, 0], "", None)]

    def test_init_without_camera_reports_unavailable(self):
        srv = FakeCameraServer({"cmd": "init", "roi": [0, 0, 10, 10]})
        runner.handle_camera_cmd(srv, None)
        connected, roi, err, _ = srv.results[0]
        assert connected is False and roi == [0, 0, 10, 10] and "unavailable" in err

    def test_init_with_camera_reports_actuals(self):
        srv = FakeCameraServer({"cmd": "init", "roi": [0, 0, 10, 10], "exposure_time": 0.1})
        runner.handle_camera_cmd(srv, FakeCamera())
        assert srv.results == [(True, [1, 2, 3, 4], "", 0.05)]

    def test_init_failure_reports_current_roi(self):
        srv = FakeCameraServer({"cmd": "init", "roi": [0, 0, 10, 10]})
        runner.handle_camera_cmd(srv, FakeCamera(raise_on="init"))
        connected, roi, err, _ = srv.results[0]
        assert connected is False and roi == [9, 9, 9, 9] and "init fail" in err

    def test_apply_settings_with_camera(self):
        srv = FakeCameraServer({"cmd": "apply_settings", "roi": [0, 0, 1, 1]})
        runner.handle_camera_cmd(srv, FakeCamera())
        assert srv.results == [(True, [5, 6, 7, 8], "", 0.02)]


# --------------------------------------------------------------------------- #
# open_camera -- degrades to None when the wrapper / pylablib is absent
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# make_engine_run -- per-scan camera arm + capture post_cb + scan_id plumbing
# --------------------------------------------------------------------------- #
class _ArmCam:
    def __init__(self):
        import numpy as np
        self.started = self.stopped = self.flushed = False
        self._frames = [np.zeros((2, 2), dtype="uint16"), np.zeros((2, 2), dtype="uint16")]

    def flush(self):
        self.flushed = True
        return 0

    def start_video(self, external=True, nframes=None):
        self.started = (external, nframes)

    def stop_video(self):
        self.stopped = True

    def read_frames(self):
        f, self._frames = self._frames, []
        return f


class _StoreServer:
    def __init__(self):
        self.stored = []
        self.finished = 0

    def store_imgs(self, arr, scan_id, seq_id):
        self.stored.append((arr, scan_id, seq_id))

    def seq_finish(self):
        self.finished += 1


class _G:
    def __init__(self, seq_id):
        self._s = seq_id

    def seq_id(self, default=None):
        return self._s


class _SeqCfg:
    def __init__(self, seq_id):
        self.G = _G(seq_id)


class _NumImagesSG:
    def __init__(self, n):
        self._n = n

    def runp(self):
        class _RP:
            def __init__(self, n):
                self._n = n

            def NumImages(self, default=None):
                return self._n
        return _RP(self._n)


class _Ctrl:
    def __init__(self, scan_id):
        self._sid = scan_id

    def begin_scan(self):
        return self._sid

    def check_pause_abort(self):
        return False


class TestMakeEngineRun:
    def test_arms_captures_and_disarms(self, monkeypatch):
        import run_seq
        import seq_manager
        captured = {}

        def fake_rsg(seq, scangroup, control=None, post_cb=(), new_run=None, **kw):
            captured["new_run"] = new_run
            sid = control.begin_scan()              # exercises _ScanIdRecorder
            captured["scan_id_seen"] = sid
            for cb in post_cb:                      # fire the per-shot capture hook once
                cb(0, 1)
            return {"status": "ok", "nseq": 1}

        monkeypatch.setattr(run_seq, "run_scan_group", fake_rsg)
        monkeypatch.setattr(seq_manager, "new_run", lambda: None)

        cam, srv, cfg = _ArmCam(), _StoreServer(), _SeqCfg(seq_id=4)
        run = runner.make_engine_run(srv, cam, cfg)
        res = run(lambda s: s, _NumImagesSG(2), control=_Ctrl(scan_id=555), rep=1)

        assert res["status"] == "ok"
        assert cam.flushed and cam.started == (True, 16) and cam.stopped     # armed + disarmed
        assert captured["new_run"] is seq_manager.new_run                     # engine reset wired
        # 2 frames published, stamped with scan_id 555 (from begin_scan) + seq_id 4
        assert len(srv.stored) == 2 and srv.finished == 1
        assert all(s == 555 and q == 4 for _, s, q in srv.stored)

    def test_no_capture_when_num_images_zero(self, monkeypatch):
        import run_seq
        import seq_manager
        monkeypatch.setattr(run_seq, "run_scan_group",
                            lambda *a, **k: {"status": "ok", "nseq": 1})
        monkeypatch.setattr(seq_manager, "new_run", lambda: None)
        cam, srv = _ArmCam(), _StoreServer()
        run = runner.make_engine_run(srv, cam, _SeqCfg(2))
        run(lambda s: s, _NumImagesSG(0), control=_Ctrl(1))
        assert cam.started is False and srv.stored == []   # NumImages 0 -> no arm/capture

    def test_camera_none_skips_capture(self, monkeypatch):
        import run_seq
        import seq_manager
        monkeypatch.setattr(run_seq, "run_scan_group",
                            lambda *a, **k: {"status": "ok"})
        monkeypatch.setattr(seq_manager, "new_run", lambda: None)
        srv = _StoreServer()
        run = runner.make_engine_run(srv, None, _SeqCfg(2))
        run(lambda s: s, _NumImagesSG(1), control=_Ctrl(1))
        assert srv.stored == []


class TestScanIdRecorderAndNumImages:
    def test_recorder_forwards_and_records(self):
        box = {"id": -1}
        rec = runner._ScanIdRecorder(_Ctrl(scan_id=77), box)
        assert rec.begin_scan() == 77 and box["id"] == 77
        assert rec.check_pause_abort() is False

    def test_recorder_keeps_default_on_none(self):
        box = {"id": -1}

        class _NoneCtrl:
            def begin_scan(self):
                return None

            def check_pause_abort(self):
                return False

        rec = runner._ScanIdRecorder(_NoneCtrl(), box)
        assert rec.begin_scan() is None and box["id"] == -1     # abort-pending -> unchanged

    def test_num_images(self):
        assert runner._num_images(_NumImagesSG(3)) == 3
        assert runner._num_images(object()) == 0                # bad scangroup -> 0


def test_open_camera_none_when_wrapper_absent():
    logs = []
    assert runner.open_camera(log=logs.append) is None       # orca_camera not built yet
    assert logs and "camera" in logs[0].lower()


# --------------------------------------------------------------------------- #
# teardown -- camera released before worker stop
# --------------------------------------------------------------------------- #
def test_teardown_closes_camera_and_stops_worker():
    order = []
    cam = type("C", (), {"close": lambda self: order.append("close")})()
    srv = type("S", (), {"stop_worker": lambda self: order.append("stop")})()
    runner._teardown(srv, cam, log=lambda m: None)
    assert order == ["close", "stop"]                         # camera first, then worker

def test_teardown_tolerates_no_camera():
    order = []
    srv = type("S", (), {"stop_worker": lambda self: order.append("stop")})()
    runner._teardown(srv, None, log=lambda m: None)
    assert order == ["stop"]


# --------------------------------------------------------------------------- #
# signal handlers -- best-effort install, restored after
# --------------------------------------------------------------------------- #
def test_install_signal_handlers_registers_sigint():
    old = signal.getsignal(signal.SIGINT)
    flag = {"hit": False}
    try:
        runner._install_signal_handlers(lambda *a: flag.__setitem__("hit", True))
        h = signal.getsignal(signal.SIGINT)
        assert callable(h)
        h(signal.SIGINT, None)                                # invoking it sets the stop flag
        assert flag["hit"] is True
    finally:
        signal.signal(signal.SIGINT, old)


# --------------------------------------------------------------------------- #
# launch shim -- ybctrl.run_loop.runner bootstraps path + delegates
# --------------------------------------------------------------------------- #
def test_launch_shim_importable_and_delegates(monkeypatch):
    import ybctrl.run_loop.runner as shim

    shim._bootstrap_path()                                    # idempotent; dirs now on path
    import runner as host
    called = {}

    def fake_main(argv=None):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(host, "main", fake_main)
    assert shim.main(["tcp://x:1"]) == 0
    assert called["argv"] == ["tcp://x:1"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stop_after(n):
    """A should_stop() that returns False n times then True (n loop iterations)."""
    state = {"i": 0}

    def stop():
        if state["i"] >= n:
            return True
        state["i"] += 1
        return False

    return stop
