"""Phase-5 sequence_runner: per-job orchestration (run_job) + dummy-mode IdleScheduler.

NO-HARDWARE: a fake ExptServer-like hub + injected dispatch/run drive run_job through all
failure statuses, and the IdleScheduler dummy-mode state machine (off/default/last +
last_fallback_logged) is exercised with fake dummy/last runners. The live ZMQ/camera/engine
hosting loop + entry point are deferred to the hardware/integration step.
"""

import pytest

from dispatch_descriptor import DispatchResult, NotMigratedError
from sequence_runner import IdleScheduler, JobResult, run_job

pytestmark = pytest.mark.no_hardware


class FakeServer:
    def __init__(self, mode="default"):
        self.mode = mode
        self.seq_names = []
        self.finished = []
        self.fallback_calls = []

    def dummy_mode(self):
        return self.mode

    def set_seq_name(self, job_id, name):
        self.seq_names.append((job_id, name))

    def finish_job(self, job_id, status):
        self.finished.append((job_id, status))

    def set_last_fallback_direct(self, on):
        self.fallback_calls.append(bool(on))


class FakeRun:
    def __init__(self, status="ok", raise_exc=None):
        self.calls = []
        self.status = status
        self.raise_exc = raise_exc

    def __call__(self, seqfn, scangroup, control=None, **kw):
        self.calls.append({"seqfn": seqfn, "scangroup": scangroup, "control": control, "kw": kw})
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"status": self.status, "nseq": 5}


def _disp(seq_name="MySeq", opts=()):
    def make(_desc):
        return DispatchResult(scangroup="SG", seq=(lambda s: s),
                              seq_name=seq_name, opts=list(opts), label=seq_name)
    return make


# --------------------------------------------------------------------------- #
# run_job: statuses
# --------------------------------------------------------------------------- #
class TestRunJobStatuses:
    def test_ok(self):
        srv = FakeServer()
        run = FakeRun(status="ok")
        res = run_job(srv, "{}", job_id=7, dispatch=_disp(), run=run,
                      control_factory=lambda s: ("CTL", s))
        assert res == JobResult("ok", "MySeq")
        assert srv.finished == [(7, "ok")]
        assert srv.seq_names == [(7, "MySeq")]
        assert len(run.calls) == 1

    def test_aborted_distinct_from_ok(self):
        srv = FakeServer()
        res = run_job(srv, "{}", job_id=1, dispatch=_disp(), run=FakeRun(status="aborted"),
                      control_factory=lambda s: None)
        assert res.status == "aborted"
        assert srv.finished == [(1, "aborted")]

    def test_descriptor_error_does_not_run(self):
        srv = FakeServer()
        run = FakeRun()

        def bad_dispatch(_desc):
            raise ValueError("malformed")

        res = run_job(srv, "{bad}", job_id=2, dispatch=bad_dispatch, run=run,
                      control_factory=lambda s: None)
        assert res.status.startswith("descriptor error")
        assert srv.finished == [(2, res.status)]
        assert run.calls == []                       # never ran

    def test_not_migrated(self):
        srv = FakeServer()

        def nm_dispatch(_desc):
            raise NotMigratedError("no module FooSeq")

        res = run_job(srv, "{}", job_id=3, dispatch=nm_dispatch, run=FakeRun(),
                      control_factory=lambda s: None)
        assert res.status.startswith("not migrated")
        assert srv.finished == [(3, res.status)]

    def test_run_error(self):
        srv = FakeServer()
        run = FakeRun(raise_exc=RuntimeError("compile boom"))
        res = run_job(srv, "{}", job_id=4, dispatch=_disp(), run=run,
                      control_factory=lambda s: None)
        assert res.status.startswith("run error")
        assert "compile boom" in res.status
        assert srv.finished == [(4, res.status)]

    def test_prep_error_does_not_run(self):
        srv = FakeServer()
        run = FakeRun()

        def on_prep(_disp):
            raise RuntimeError("AWG offline")

        res = run_job(srv, "{}", job_id=5, dispatch=_disp(), run=run,
                      control_factory=lambda s: None, on_prep=on_prep)
        assert res.status.startswith("prep error")
        assert run.calls == []                       # prep failed before the run


# --------------------------------------------------------------------------- #
# run_job: opts mapping + control wiring
# --------------------------------------------------------------------------- #
class TestRunJobWiring:
    def test_opts_map_to_run_kwargs(self):
        srv = FakeServer()
        run = FakeRun()
        cb = lambda n, a: None
        opts = [("rep", 3), ("random", True), ("tstartwait", 0.1),
                ("pre_cb", cb), ("email", "me@x")]   # email ignored (runner/G concern)
        run_job(srv, "{}", dispatch=_disp(opts=opts), run=run, control_factory=lambda s: None)
        kw = run.calls[0]["kw"]
        assert kw["rep"] == 3 and kw["is_random"] is True and kw["tstartwait"] == 0.1
        assert kw["pre_cb"] == [cb] and "email" not in kw

    def test_control_factory_wired(self):
        srv = FakeServer()
        run = FakeRun()
        run_job(srv, "{}", dispatch=_disp(), run=run, control_factory=lambda s: ("CTL", s))
        assert run.calls[0]["control"] == ("CTL", srv)


# --------------------------------------------------------------------------- #
# IdleScheduler: dummy-mode state machine
# --------------------------------------------------------------------------- #
class TestIdleScheduler:
    def _sched(self, mode):
        srv = FakeServer(mode=mode)
        dummies, lasts = [], []
        sched = IdleScheduler(srv, run_dummy=lambda: dummies.append(1),
                              run_last=lambda s: lasts.append(s))
        return srv, sched, dummies, lasts

    def test_off_pauses_only(self):
        srv, sched, dummies, lasts = self._sched("off")
        slept = []
        assert sched.step(sleep=lambda dt: slept.append(dt)) == "off"
        assert slept == [0.1] and dummies == [] and lasts == []

    def test_default_runs_dummy(self):
        srv, sched, dummies, lasts = self._sched("default")
        assert sched.step(sleep=_noop) == "default"
        assert dummies == [1] and lasts == []

    def test_last_without_cache_falls_back_and_logs_once(self):
        srv, sched, dummies, lasts = self._sched("last")
        assert sched.step(sleep=_noop) == "last_fallback"
        assert dummies == [1] and lasts == []
        assert sched.last_fallback_logged is True
        assert srv.fallback_calls == [True]          # logged exactly once
        # a 2nd idle iteration with still-no-cache does NOT re-log
        sched.step(sleep=_noop)
        assert srv.fallback_calls == [True]

    def test_last_with_cache_replays_and_clears_fallback(self):
        srv, sched, dummies, lasts = self._sched("last")
        sched.step(sleep=_noop)                      # no cache -> fallback (logs True)
        sched.cache_last_seq("SEQ")                  # a real job populated the cache
        assert sched.last_fallback_logged is True    # cache_last_seq must NOT reset it
        action = sched.step(sleep=_noop)             # now replays the cached seq
        assert action == "last" and lasts == ["SEQ"]
        assert sched.last_fallback_logged is False
        assert srv.fallback_calls == [True, False]   # recovery cleared the flag once

    def test_default_clears_a_pending_fallback_flag(self):
        srv, sched, dummies, lasts = self._sched("last")
        sched.step(sleep=_noop)                       # last+no-cache -> fallback (True)
        srv.mode = "default"
        sched.step(sleep=_noop)                       # default clears the fallback flag
        assert sched.last_fallback_logged is False
        assert srv.fallback_calls == [True, False]


def _noop(_dt):
    pass
