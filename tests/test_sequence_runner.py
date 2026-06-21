"""Phase-5 sequence_runner: per-job orchestration (run_job) + dummy-mode IdleScheduler.

NO-HARDWARE: a fake ExptServer-like hub + injected dispatch/run drive run_job through all
failure statuses, and the IdleScheduler dummy-mode state machine (off/default/last +
last_fallback_logged) is exercised with fake dummy/last runners. The live ZMQ/camera/engine
hosting loop + entry point are deferred to the hardware/integration step.
"""

import random

import pytest

from dispatch_descriptor import DispatchResult, NotMigratedError
from sequence_runner import IdleScheduler, JobResult, run_job

pytestmark = pytest.mark.no_hardware


class FakeServer:
    def __init__(self, mode="default", seq_req=0):
        self.mode = mode
        self.seq_names = []
        self.finished = []
        self.fallback_calls = []
        self.seq_req = seq_req          # 0 NoRequest / 1 Pause / 2 Abort (idle abort-gate)
        self.cleared = 0

    def dummy_mode(self):
        return self.mode

    def set_seq_name(self, job_id, name):
        self.seq_names.append((job_id, name))

    def finish_job(self, job_id, status):
        self.finished.append((job_id, status))

    def set_last_fallback_direct(self, on):
        self.fallback_calls.append(bool(on))

    def check_request(self):
        return self.seq_req

    def clear_seq_request(self):
        self.seq_req = 0
        self.cleared += 1


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


def _disp(seq_name="MySeq", opts=(), scangroup="SG"):
    def make(_desc):
        return DispatchResult(scangroup=scangroup, seq=(lambda s: s),
                              seq_name=seq_name, opts=list(opts), label=seq_name)
    return make


class _FakeRunp:
    """Mimics a runp DynProps: ``rp.<Field>(default)`` returns the set value or the default."""

    def __init__(self, **fields):
        object.__setattr__(self, "_f", fields)

    def __getattr__(self, name):
        fields = object.__getattribute__(self, "_f")

        def getter(default=None):
            v = fields.get(name)
            return v if v is not None else default
        return getter


class _FakeSG:
    """A ScanGroup stub exposing the run-order query surface (nseq + runp)."""

    def __init__(self, nseqs, **runp_fields):
        self._n = nseqs
        self._rp = _FakeRunp(**runp_fields)

    def nseq(self):
        return self._n

    def runp(self):
        return self._rp


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

    def test_reload_modules_runs_before_dispatch(self):
        # The hot-reload hook must fire BEFORE the dispatcher resolves the seq, else the resolver
        # imports stale (cached) experiment modules.
        order = []
        srv = FakeServer()

        def reloader():
            order.append("reload")

        def disp(_desc):
            order.append("dispatch")
            return _disp()(_desc)

        run_job(srv, "{}", dispatch=disp, run=FakeRun(), control_factory=lambda s: None,
                reload_modules=reloader)
        assert order == ["reload", "dispatch"]

    def test_reload_failure_does_not_kill_the_job(self):
        # A reload that raises is swallowed (stale modules == pre-reload behavior); the job runs.
        srv = FakeServer()
        run = FakeRun(status="ok")

        def boom():
            raise RuntimeError("reload boom")

        res = run_job(srv, "{}", dispatch=_disp(), run=run, control_factory=lambda s: None,
                      reload_modules=boom)
        assert res.status == "ok" and len(run.calls) == 1


# --------------------------------------------------------------------------- #
# run_job: code-snapshot REPLAY wiring (#3). Opt-in -- a descriptor pinning a
# code_snapshot runs its dispatch against that snapshot's experiment code; an
# absent field is a nullcontext (byte-for-byte the prior behavior).
# --------------------------------------------------------------------------- #
class TestCodeSnapshotReplay:
    def test_extract_absent_present_and_garbage(self):
        from sequence_runner import _extract_code_snapshot
        assert _extract_code_snapshot('{"seq":"X"}') is None
        assert _extract_code_snapshot(
            '{"seq":"X","code_snapshot":{"scan_id":1}}') == {"scan_id": 1}
        assert _extract_code_snapshot({"code_snapshot": {"scan_id": 2}}) == {"scan_id": 2}
        assert _extract_code_snapshot("not json") is None
        assert _extract_code_snapshot(123) is None

    def test_no_field_is_nullcontext(self):
        import sys
        from sequence_runner import _snapshot_replay_ctx
        saved = list(sys.path)
        with _snapshot_replay_ctx('{"seq":"X"}', None):
            pass
        assert sys.path == saved

    def test_pinned_descriptor_injects_then_restores(self, tmp_path, monkeypatch):
        import json
        import os
        import sys
        import code_snapshot
        from sequence_runner import _snapshot_replay_ctx
        root = str(tmp_path / "proj")
        data_root = str(tmp_path / "data")
        # Pin the snapshot base under data_root: the production default is now a LOCAL dir off the
        # superproject, but this test asserts the injected path contains data_root (and must not
        # write into the real local snapshot dir).
        monkeypatch.setenv("YB_CODE_SNAPSHOT_DIR", os.path.join(data_root, "_code_snapshots"))
        os.makedirs(os.path.join(root, "YbSeqs"))
        with open(os.path.join(root, "YbSeqs", "ReplayProbe.py"), "w") as f:
            f.write("V = 5\n")
        code_snapshot.snapshot_code(root, data_root, run_id=11)
        desc = json.dumps({"seq": "ReplayProbe",
                           "code_snapshot": {"scan_id": 11, "data_root": data_root}})
        saved = list(sys.path)
        with _snapshot_replay_ctx(desc, None) as active:
            assert active is True
            assert any(p.endswith("YbSeqs") and data_root in p for p in sys.path[:4])
        assert sys.path == saved        # restored exactly

    def test_run_job_with_missing_snapshot_runs_and_restores(self, tmp_path):
        # End to end: a pinned descriptor whose snapshot is absent falls back to live code
        # (the job still completes) and never leaves sys.path mutated.
        import json
        import sys
        desc = json.dumps({"seq": "X",
                           "code_snapshot": {"scan_id": 99, "data_root": str(tmp_path)}})
        saved = list(sys.path)
        res = run_job(FakeServer(), desc, job_id=8, dispatch=_disp(), run=FakeRun(status="ok"),
                      control_factory=lambda s: None)
        assert res.status == "ok"
        assert sys.path == saved


# --------------------------------------------------------------------------- #
# run_job: production run-order (ybBuildScanJob -> Scan.Params), handed to run_scan_group
# as a pre-built/pre-scrambled `indices` list with rep=1, is_random=False.
# --------------------------------------------------------------------------- #
class TestRunOrder:
    def _run(self, sg, opts=(), rng=None):
        srv, run = FakeServer(), FakeRun()
        run_job(srv, "{}", dispatch=_disp(opts=opts, scangroup=sg), run=run,
                control_factory=lambda s: None, rng=rng)
        return run.calls[0]["kw"]

    def test_explicit_rep_builds_stacked_order(self):
        kw = self._run(_FakeSG(3, Scramble=0), opts=[("rep", 2)])
        assert kw["indices"] == [1, 2, 3, 1, 2, 3]       # 2 unscrambled passes
        assert kw["rep"] == 1 and kw["is_random"] is False

    def test_num_per_group_derives_stacknum(self):
        # No explicit rep -> StackNum = max(ceil(NumPerGroup / nseqs), 2) = max(ceil(10/3),2) = 4.
        kw = self._run(_FakeSG(3, Scramble=0, NumPerGroup=10))
        assert kw["indices"] == [1, 2, 3] * 4 and kw["rep"] == 1

    def test_stacknum_floor_is_two(self):
        # NumPerGroup small -> StackNum floored at 2 (MATLAB max(...,2)).
        kw = self._run(_FakeSG(5, Scramble=0, NumPerGroup=1))
        assert kw["indices"] == [1, 2, 3, 4, 5] * 2

    def test_scramble_off_by_default(self):
        # Scramble unset -> OFF; only the scan file's runp turns it on. Deterministic order.
        kw = self._run(_FakeSG(2, NumPerGroup=4))        # StackNum = max(ceil(4/2),2) = 2
        assert kw["indices"] == [1, 2, 1, 2] and kw["is_random"] is False

    def test_runp_scramble_on_scrambles_per_block(self):
        # runp Scramble=1 (set in the scan file) -> each pass independently shuffled, block
        # boundaries intact, and the realized order actually differs from the plain stack.
        kw = self._run(_FakeSG(5, NumPerGroup=10, Scramble=1), rng=random.Random(0))
        order = kw["indices"]
        assert len(order) == 10 and kw["is_random"] is False    # StackNum = max(ceil(10/5),2)=2
        assert sorted(order[0:5]) == [1, 2, 3, 4, 5]            # ...each pass is a full sweep
        assert sorted(order[5:10]) == [1, 2, 3, 4, 5]
        assert order != [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]         # scramble actually reordered

    def test_bare_default_path_derives_stacknum_from_200(self):
        # The common production config: no rep, NumPerGroup unset -> default 200.
        # StackNum = max(ceil(200/17), 2) = 12 -> 12 passes over 17 points = 204 shots.
        kw = self._run(_FakeSG(17))
        assert kw["indices"] == list(range(1, 18)) * 12 and len(kw["indices"]) == 204
        assert kw["rep"] == 1 and kw["is_random"] is False

    def test_explicit_rep_one_is_a_single_pass(self):
        # An explicit rep=1 is honored as a single pass (pyctrl override; no >=2 floor).
        kw = self._run(_FakeSG(3, Scramble=0), opts=[("rep", 1)])
        assert kw["indices"] == [1, 2, 3] and kw["rep"] == 1

    def test_forever_falls_through_to_loop(self):
        # rep=0 (continuous monitor) can't be pre-stacked -> run_scan_group's forever loop.
        kw = self._run(_FakeSG(3, Scramble=0), opts=[("rep", 0)])
        assert kw.get("rep") == 0 and "indices" not in kw

    def test_negative_rep_left_for_run_scan_group_guard(self):
        # rep<0 is NOT swallowed into the NumPerGroup fallback -> run_scan_group raises later.
        kw = self._run(_FakeSG(3, Scramble=0), opts=[("rep", -1)])
        assert "indices" not in kw and kw["rep"] == -1

    def test_unqueryable_group_falls_through(self):
        # A non-ScanGroup stub -> plain opts mapping preserved (no order build).
        kw = self._run("SG", opts=[("rep", 3), ("random", True)])
        assert "indices" not in kw and kw["rep"] == 3 and kw["is_random"] is True


# --------------------------------------------------------------------------- #
# IdleScheduler: dummy-mode state machine
# --------------------------------------------------------------------------- #
class TestIdleScheduler:
    def _sched(self, mode, seq_req=0):
        srv = FakeServer(mode=mode, seq_req=seq_req)
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

    # --- abort gate (must-fix #4): abort during idle silences + consumes the keep-alive --- #
    def test_abort_silences_keepalive_and_consumes_request(self):
        # mode 'default' would fire the DummySeq, but a pending Abort (seq_req=2) suppresses it.
        srv, sched, dummies, lasts = self._sched("default", seq_req=2)
        slept = []
        assert sched.step(sleep=lambda dt: slept.append(dt)) == "aborted"
        assert dummies == [] and lasts == []          # no FPGA keep-alive fired
        assert srv.cleared == 1 and srv.seq_req == 0   # abort consumed (idle has nothing to abort)
        assert slept == [0.1]

    def test_abort_gate_precedes_last_mode(self):
        # The gate wins over every mode, including 'last' (no cached-seq replay while aborting).
        srv, sched, dummies, lasts = self._sched("last", seq_req=2)
        sched.cache_last_seq("SEQ")
        assert sched.step(sleep=_noop) == "aborted"
        assert lasts == [] and dummies == [] and srv.cleared == 1

    def test_no_abort_runs_normally(self):
        # seq_req=0 (NoRequest) -> the gate is transparent; normal mode logic runs.
        srv, sched, dummies, lasts = self._sched("default", seq_req=0)
        assert sched.step(sleep=_noop) == "default"
        assert dummies == [1] and srv.cleared == 0


def _noop(_dt):
    pass
