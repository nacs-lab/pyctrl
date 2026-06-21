"""Phase-5 runner: the scenario-3 run-loop HOST orchestration (YbExptCtrl/runner.py).

NO-HARDWARE: every device/engine/socket dependency is injected, so URL resolution, the
single-backend guard, the descriptor->job dispatch, the consume loop, the camera-command
pump, and the clean-terminate teardown are all exercised with fakes -- no engine load, no
bound socket, no camera. The live ``serve()`` wiring (real ExptServer + engine + pylablib)
is the NEEDS-HARDWARE entry and is not run here.
"""

import os
import signal

import pytest

import runner

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _isolate_data_prefix(tmp_path, monkeypatch):
    """Redirect scan-prep writes to a temp dir so tests NEVER touch the real OneDrive data dir."""
    monkeypatch.setenv("YB_DATA_PREFIX", str(tmp_path))


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
        self.submitted_summaries = []           # [summary_or_None, ...] (parallel to submitted)
        self.linked = []                        # [(desc_id, job_id), ...]
        self.desc_finished = []                 # [(desc_id, status, msg), ...]
        self.dummy_running = []                 # [flag, ...]
        self.mark_idle_calls = 0                 # end-of-job idle-status resets
        self.submit_should_raise = False

    def pop_next_descriptor(self):
        return self._descs.pop(0) if self._descs else None

    def submit_job(self, payload, summary=None, job_id=None):
        if self.submit_should_raise:
            raise RuntimeError("submit boom")
        # Mirror the real ExptServer: a given job_id is reused verbatim (pyctrl
        # id-reuse, runner passes job_id=desc_id); otherwise mint from the counter.
        if job_id is None:
            jid = self._next_job_id
            self._next_job_id += 1
        else:
            jid = int(job_id)
        self.submitted.append((jid, payload))
        self.submitted_summaries.append(summary)
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

    def mark_idle_if_queue_empty(self):
        self.mark_idle_calls += 1
        return True


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
        # JSON string is utf-8 encoded; bytes are forwarded as-is. The job REUSES
        # the descriptor's id (id-reuse), so submitted/linked ids match the desc ids.
        assert srv.submitted == [(1, b'{"seq":"A"}'), (2, b'{"seq":"B"}')]
        assert srv.linked == [(1, 1), (2, 2)]
        assert srv.desc_finished == []

    def test_empty_queue_returns_zero(self):
        srv = FakeQueueServer()
        assert runner.handle_descriptor_pop(srv) == 0
        assert srv.submitted == []

    def test_attaches_queue_summary_to_built_job(self):
        # The descriptor's sweep + label become the job row's summary (queue-panel axes/name).
        desc = ('{"seq":"TweezerLoadingSeq","label":"LACScan",'
                '"params":{"GreenMOT.BiasCoilCurrent.Y":{"scan":1,"values":[0.24,0.28,0.32]}},'
                '"runp":{"NumPerGroup":500,"NumImages":1}}')
        srv = FakeQueueServer(descriptors=[{"id": 1, "descriptor": desc}])
        assert runner.handle_descriptor_pop(srv) == 1
        s = srv.submitted_summaries[0]
        assert s is not None
        assert s["scan_name"] == "LACScan"
        assert s["axes"][0]["name"] == "GreenMOT.BiasCoilCurrent.Y"
        assert s["axes"][0]["npts"] == 3 and s["axes"][0]["dim"] == 1

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

    def test_marks_idle_after_a_job_runs(self):
        # End-of-job hook: consume_loop calls server.mark_idle_if_queue_empty() once the job
        # returns, so a finite scan's stuck "running" status returns to idle.
        srv = FakeQueueServer(jobs=[{"id": 5, "payload": b"{}"}])
        runner.consume_loop(
            srv, should_stop=_stop_after(1),
            run_job_fn=lambda *a, **k: type("R", (), {"status": "ok"})(),
            dispatch_pop=lambda s: 0, sleep=lambda dt: None)
        assert srv.mark_idle_calls == 1

    def test_no_idle_reset_on_pure_idle_iteration(self):
        # The reset is end-of-JOB only; a no-job iteration leaves it to the idle state machine
        # (a never-run backend is already Init), so mark_idle is NOT called here.
        srv = FakeQueueServer()
        idle = type("Idle", (), {"step": lambda self, sleep: None})()
        runner.consume_loop(
            srv, should_stop=_stop_after(1), dispatch_pop=lambda s: 0,
            idle=idle, sleep=lambda dt: None)
        assert srv.mark_idle_calls == 0

    def test_idle_reset_failure_never_stops_the_loop(self):
        # A throwing mark_idle_if_queue_empty must be swallowed (status reset is best-effort).
        srv = FakeQueueServer(jobs=[{"id": 9, "payload": b"{}"}])
        srv.mark_idle_if_queue_empty = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        ran = []
        runner.consume_loop(
            srv, should_stop=_stop_after(1),
            run_job_fn=lambda *a, **k: ran.append(1) or type("R", (), {"status": "ok"})(),
            dispatch_pop=lambda s: 0, sleep=lambda dt: None)
        assert ran == [1]                        # job ran; loop survived the reset error


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

    def current_roi(self):
        return [0, 0, 256, 256]


class _StoreServer:
    def __init__(self):
        self.stored = []
        self.finished = 0

    def store_imgs(self, arr, scan_id, seq_id):
        self.stored.append((arr, scan_id, seq_id))

    def seq_finish(self):
        self.finished += 1

    def publish_shot(self, frames, scan_id, seq_id, *, async_=True):
        # mirror ExptServer: stage each (raw) frame then finish the shot
        for f in frames:
            self.store_imgs(f, scan_id, seq_id)
        self.seq_finish()


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

            def useScanLongSlmLock(self, default=0):
                # NO-HARDWARE: these tests exercise camera arm/capture plumbing, not the scan-long
                # SLM session (tested in test_slm_rearrangement). Disable it so make_engine_run does
                # not try to grab the real SLM lock over the network.
                return 0
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
            for cb in post_cb:                      # fire the per-shot capture hook once
                cb(0, 1)
            return {"status": "ok", "nseq": 1}

        reset_calls = {"n": 0}
        monkeypatch.setattr(run_seq, "run_scan_group", fake_rsg)
        monkeypatch.setattr(seq_manager, "new_run",
                            lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1))

        cam, srv, cfg = _ArmCam(), _StoreServer(), _SeqCfg(seq_id=4)
        run = runner.make_engine_run(srv, cam, cfg)
        res = run(lambda s: s, _NumImagesSG(2), control=_Ctrl(scan_id=555), rep=1)

        assert res["status"] == "ok"
        assert cam.flushed and cam.started == (True, 16) and cam.stopped     # armed + disarmed
        # The new_run seam is now a run_timing-wrapped closure (so the engine reset is timed
        # in the bucket-B setup window); assert by BEHAVIOR -- invoking it triggers the reset.
        assert callable(captured["new_run"]) and reset_calls["n"] == 0        # not yet called
        captured["new_run"]()
        assert reset_calls["n"] == 1                                          # engine reset wired
        # 2 frames published with seq_id 4 + a 14-digit YYYYMMDDHHMMSS scan_id (NOT epoch-ms)
        assert len(srv.stored) == 2 and srv.finished == 1
        assert all(q == 4 and len(str(s)) == 14 for _, s, q in srv.stored)

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

    def test_threads_run_order_into_params(self, monkeypatch):
        # The pre-built run order (indices) is persisted as the scan-config's Scan.Params (the
        # seq_id->point map the live scan curve buckets on); NumPerGroup written = len(order).
        import run_seq
        import scan_prep
        import seq_manager
        monkeypatch.setattr(run_seq, "run_scan_group",
                            lambda *a, **k: {"status": "ok", "nseq": 6})
        monkeypatch.setattr(seq_manager, "new_run", lambda: None)
        cap = {}

        def fake_write(scan_id, frame_wh, num_images, *, params=None, num_per_group=0, **kw):
            cap["params"], cap["num_per_group"] = params, num_per_group
            return "captured.json"

        monkeypatch.setattr(scan_prep, "write_scan_config", fake_write)
        run = runner.make_engine_run(_StoreServer(), None, _SeqCfg(seq_id=1))
        order = [1, 2, 3, 1, 2, 3]
        run(lambda s: s, _NumImagesSG(0), control=_Ctrl(1),
            indices=order, rep=1, is_random=False)
        assert cap["params"] == order and cap["num_per_group"] == 6

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


class TestScanPrep:
    def test_writes_json_config(self, tmp_path):
        import json
        import scan_prep
        sid = 20260603001055
        p = scan_prep.write_scan_config(sid, (2100, 1800), 1, is_init=1, num_per_group=500,
                                        prefix=str(tmp_path))
        assert os.path.exists(p) and p.endswith("data_20260603_001055.json")
        cfg = json.loads(open(p).read())
        assert cfg["frameSize"] == [2100, 1800] and cfg["NumImages"] == 1 and cfg["isInit"] == 1
        assert cfg["source"] == "pyctrl"

    def test_write_scan_prep_uses_camera_roi(self):
        # _write_scan_prep pulls frameSize from the camera ROI [x,y,w,h] -> (w,h).
        import scan_prep
        cam = _ArmCam()                                  # current_roi -> [0,0,256,256]
        runner._write_scan_prep(20260603001055, _NumImagesSG(1), cam, 1, log=lambda m: None)
        p = scan_prep.scan_config_path(20260603001055)   # uses YB_DATA_PREFIX (tmp via fixture)
        import json
        assert json.loads(open(p).read())["frameSize"] == [256, 256]


class TestScanIdAndNumImages:
    def test_new_scan_id_is_14_digits(self):
        sid = runner._new_scan_id()
        assert isinstance(sid, int) and len(str(sid)) == 14     # YYYYMMDDHHMMSS

    def test_num_images(self):
        assert runner._num_images(_NumImagesSG(3)) == 3
        assert runner._num_images(object()) == 0                # bad scangroup -> 0


class TestIdleResilience:
    def test_dummy_failure_does_not_crash_backend(self):
        """A keep-alive run that throws must be caught (logged + backed off), not propagated."""
        logs, slept = [], []

        def boom(seq):
            raise RuntimeError("nidaqmx missing")

        class _Srv:
            def dummy_mode(self):
                return "default"

        sched = runner.make_idle(_Srv(), dummy_seq="DUM", run_real=boom,
                                 sleep=lambda dt: slept.append(dt), log=logs.append)
        sched.step(sleep=lambda dt: None)        # default -> run_dummy -> boom (must be caught)
        assert any("dummy run failed" in m for m in logs)
        assert slept == [1.0]                    # 1 s back-off so it doesn't hot-spin


class TestForceDummyOff:
    def test_sets_mode_off(self):
        import threading

        class _Srv:
            def __init__(self):
                setattr(self, "_ExptServer__dummy_mode", "last")
                setattr(self, "_ExptServer__dummy_lock", threading.Lock())

        srv = _Srv()
        runner._force_dummy_off(srv, log=lambda m: None)
        assert getattr(srv, "_ExptServer__dummy_mode") == "off"

    def test_tolerates_missing_attr(self):
        runner._force_dummy_off(object(), log=lambda m: None)   # no raise


def test_open_camera_none_when_wrapper_absent(monkeypatch):
    # Deterministic + hardware-free: simulate the wrapper raising on open (the
    # pylablib-absent / camera-absent path) rather than depending on the test
    # interpreter lacking pylablib (which would touch the real camera under the
    # engine venv). retry_delay=0 keeps it instant.
    import devices.orca as orca_mod
    monkeypatch.setattr(orca_mod, "open_orca_from_config",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cam")))
    logs = []
    assert runner.open_camera(log=logs.append, attempts=2, retry_delay=0) is None
    assert any("camera" in m.lower() for m in logs)


def test_open_camera_retries_then_gives_up(monkeypatch):
    # A contended open that fails-fast must be retried `attempts` times before
    # the backend boots camera-less (Fix for the restart-race DCAM wedge).
    import devices.orca as orca_mod
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("device busy")

    monkeypatch.setattr(orca_mod, "open_orca_from_config", boom)
    logs = []
    assert runner.open_camera(log=logs.append, attempts=3, retry_delay=0) is None
    assert calls["n"] == 3                                    # all attempts used
    assert any("attempt 1/3" in m for m in logs)             # retry logged


def test_open_camera_succeeds_after_transient(monkeypatch):
    # A transient contention that clears on the 2nd try must yield the camera --
    # a clean restart should never drop the camera over a momentary handle race.
    import devices.orca as orca_mod
    calls = {"n": 0}
    sentinel = object()

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("device busy")
        return sentinel

    monkeypatch.setattr(orca_mod, "open_orca_from_config", flaky)
    assert runner.open_camera(attempts=3, retry_delay=0) is sentinel
    assert calls["n"] == 2                                    # stopped retrying on success


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
# launch shim -- launcher.run_loop.runner bootstraps path + delegates
# --------------------------------------------------------------------------- #
def test_launch_shim_importable_and_delegates(monkeypatch):
    import launcher.run_loop.runner as shim

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
