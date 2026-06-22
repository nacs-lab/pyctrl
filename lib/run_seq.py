"""run_seq.py -- the scan-loop body (port of ``matlab_new/lib/runSeq2.m``).

``runSeq2`` is the per-shot scan loop: it walks the scan points, lazily compiles each
distinct point (constants baked -- the production model is NOT compile-once-sweep-globals),
runs it via ``run_real`` (run_seq2.py), gates pause/abort per sequence, advances the dual
counters, and brackets the run with the config lifecycle. This module ports that loop;
``run_real`` / ``run_bseq`` themselves live in run_seq2.py.

Everything that touches the engine is injected so the orchestration is NO-HARDWARE-testable:

    compile_point(seqfn, seqparam) -> runnable_seq
        prepare_seq's leaf: ``s = ExpSeq(seqparam); seqfn(s); s.generate()``. The default
        builds + generates a real ExpSeq (needs the engine -> only used in scenario 3);
        tests inject a fake returning a stub.
    run_real(seq)                 default: run_seq2.run_real (run the shot).
    control                       a ControlChannel (control_channel.py); ``None`` disables
                                   the gate (e.g. a pure byte/structure smoke run).
    new_run()                     SeqManager.new_run() equivalent (engine reset); default
                                   no-op (wired to the engine in the runner, item 7).
    on_seq_num(n)                 publish CurrentSeqNum (the runner pushes it to ExptServer;
                                   pyctrl has NO memmap, so the counter is process state).
    config_teardown()             the SeqConfig lifecycle bracket's teardown, run in
                                   ``finally`` so it fires on the abort/error path too
                                   (MATLAB ``onCleanup(SeqConfig.reset)``). Default no-op:
                                   pyctrl's ``SeqConfig.reset()`` rebuilds an EMPTY config
                                   (it does NOT re-read expConfig.m the way MATLAB does), so
                                   per-job config reload is the long-lived runner's job
                                   (item 7), not this loop's.

Scan execution model (verified, references/runtime-design.md): production bakes each scanned
value as a CONSTANT and compiles a distinct sequence per distinct point; ``usevar`` (scan-axis
globals) is DORMANT (``getseq_with_var`` returns empty ``vars``), so ``set_global`` is a no-op
for scan axes. THREE distinct memos with distinct scopes (do NOT collapse):
  * ``seqlist[idx]``      -- rep reuse (a slot already compiled in a prior rep short-circuits).
  * ``seq_map[arg0]``     -- intra-pass duplicate-index dedup (stack() always duplicates indices).
  * ``seqid_map[seqid]``  -- compiled-id reuse (ZERO reuse on the production usevar-off path:
                             distinct points -> distinct ids).

Counters advance AFTER a successful shot, NOT on abort: ``CurrentSeqNum`` (process state +
``on_seq_num``) AND ``seq_config.G.seq_id`` (camera/frame routing depends on it).

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time

import run_timing
from seq_config import SeqConfig


def run_scan_group(seqfn, scangroup, indices=None, rep=1, is_random=False,
                   pre_cb=(), post_cb=(), tstartwait=0.0,
                   control=None, compile_point=None, run_real=None,
                   seq_config=None, new_run=None, on_seq_num=None,
                   config_teardown=None, sleep=time.sleep, rng=None,
                   on_compile=None, on_globals=None):
    """Run a ScanGroup as a scan (port of ``runSeq2(func, scangroup, ...)``).

    Returns a result dict ``{"status": "ok"|"aborted"|"yielded", "nseq": <shots completed>}``.
    (The MATLAB graceful abort returns no status, recording ``'ok'`` -- the "aborted != ok"
    must-fix; pyctrl distinguishes them.) ``"yielded"`` is a background (calibration) scan that
    stepped aside for newly-queued foreground work at a shot boundary -- a clean stop like abort,
    but the run loop re-queues it instead of discarding it.
    """
    if rep < 0:
        raise ValueError("Cannot run the sequence by negative times.")
    if compile_point is None:
        compile_point = _default_compile_point
    if run_real is None:
        from run_seq2 import run_real as run_real
    if seq_config is None:
        seq_config = SeqConfig.get()
    if config_teardown is None:
        config_teardown = _noop
    if rng is None:
        import random
        rng = random.Random()

    # The scan indices to run (a full scan = 1..nseq). MATLAB's arglist of scan indices.
    n_total = scangroup.nseq()
    if indices is None:
        indices = list(range(1, n_total + 1))
    nseq = len(indices)

    seqlist = [None] * (nseq + 1)       # 1-based; slot 0 unused
    seq_map = {}                        # arg0 -> first slot (intra-pass dedup)
    seqid_map = {}                      # compiled seqid -> first slot (rep/id reuse)
    counter = {"cur_seq_num": 0}

    seq_config.G.seq_id = 1             # 1-based; advanced after each successful shot
    _publish(on_seq_num, 0)

    # Start-of-scan single clear-point (clear-at-job-start): begin_scan clears stale Pause/Abort
    # and marks Running. A None return is the generic "source refused to start" signal (no longer
    # produced by a stale abort -- that is cleared here -- but still honored as an aborted start).
    if control is not None:
        if control.begin_scan() is None:
            config_teardown()
            return {"status": "aborted", "nseq": 0}

    def prepare_seq(idx):
        """Lazy compile-or-reuse for slot ``idx`` (the three memos)."""
        if seqlist[idx] is not None:
            return                                      # rep reuse
        arg0 = indices[idx - 1]
        prev = seq_map.get(arg0)
        if prev is not None:
            seqlist[idx] = seqlist[prev]                # intra-pass duplicate-index dedup
            return
        seq_map[arg0] = idx
        seqid, seqparam, _seqvars = scangroup.getseq_with_var(arg0)  # usevar dormant -> []
        prev2 = seqid_map.get(seqid)
        if prev2 is not None:
            seqlist[idx] = seqlist[prev2]               # compiled-id reuse (none in prod)
            return
        seqid_map[seqid] = idx
        run_timing.mark("compiled", 1)              # this shot paid a real compile (cache miss)
        seqlist[idx] = compile_point(seqfn, seqparam)
        # After-end hooks: ARM (don't fire) per-unique-sequence work in run_real's
        # after_end window -- after before_start injected this shot's runtime globals
        # and before reset_globals wipes the non-persist ones. Doing it here at
        # compile time would read globals as 0 (e.g. the 616-EOM slow ramp would
        # build the degenerate ~15 s / ~60 MB zero-from ramp). Two independent hooks
        # ride this window (see _arm_after_end_once):
        #   * on_compile  -- the SeqPlotter .seq dump (GATED by the dashboard toggle).
        #   * on_globals  -- runtime-global capture (UNGATED: always armed when given,
        #                    so a never-dumped scan still records its injected globals
        #                    for faithful offline reconstruction).
        if on_compile is not None:
            _arm_after_end_once(seqlist[idx], on_compile, arg0, seqid)
        if on_globals is not None:
            _arm_after_end_once(seqlist[idx], on_globals, arg0, seqid)

    def run_cb(cbs, idx):
        for cb in cbs:
            cb(counter["cur_seq_num"], indices[idx - 1])

    def run_one(idx):
        """The per-shot body (MATLAB nested ``run_seq``).

        Returns a falsy value (``False``) to PROCEED, or a stop-reason sentinel string:
        ``"abort"`` (user abort/pause-then-abort -- discard) or ``"yield"`` (background scan
        stepping aside for foreground work -- re-queue). Both stop at this shot boundary and
        run the same clean teardown; only the run loop's reaction differs."""
        run_timing.begin_shot(indices[idx - 1])         # opt-in per-shot timing (inert when OFF)
        while True:                                     # C.RESTART retry loop
            with run_timing.stage("gate"):
                if control is not None:
                    if control.check_pause_abort():
                        return "abort"                  # gate: user abort at this boundary
                    # Yield AFTER abort (abort precedes yield): a background scan steps aside
                    # for newly-queued foreground work. should_yield never touches SeqRequest,
                    # so the foreground scan's begin_scan sees a clean slate. getattr-guarded:
                    # control is duck-typed (test stubs / older controls may lack should_yield).
                    _yield = getattr(control, "should_yield", None)
                    if _yield is not None and _yield():
                        return "yield"
            with run_timing.stage("compile"):
                prepare_seq(idx)
            if tstartwait > 0:
                with run_timing.stage("tstartwait"):
                    sleep(tstartwait)                   # NI-DAQ driver-timing workaround
            with run_timing.stage("pre_cb"):
                run_cb(pre_cb, idx)
            cur = seqlist[idx]
            # set_global for scan vars: usevar dormant -> empty -> no-op (kept for fidelity).
            run_real(cur)                               # run_seq2.run_real times its own substages
            with run_timing.stage("post_cb"):
                run_cb(post_cb, idx)
            if not _restart(cur):
                break
        counter["cur_seq_num"] += 1                     # AFTER a successful shot, not on abort
        _publish(on_seq_num, counter["cur_seq_num"])
        _bump_seq_id(seq_config)
        run_timing.end_shot()                           # emit the shot's stage line + CSV row
        return False

    run_timing.reset_scan()                             # drop any stale (prior aborted-scan) timing
    try:
        if new_run is not None:
            new_run()                                   # SeqManager.new_run() (engine reset)
        outcome = _scan_loop(run_one, nseq, rep, is_random, rng)
        # outcome: False (ran to completion) | "abort" (user abort) | "yield" (bg stepped aside).
        status = {"abort": "aborted", "yield": "yielded"}.get(outcome, "ok")
        return {"status": status, "nseq": counter["cur_seq_num"]}
    finally:
        run_timing.scan_summary()                       # log mean/median/max per stage (no-op if OFF)
        # End-of-run reset (CurrentSeqNum -> 0). Abort/Pause are NOT cleared here -- the
        # single-clear-point (clear-at-job-start) policy clears them at the next begin_scan, so
        # a stale flag is cured by the next job, not by end-of-run (control_channel.py). Config
        # bracket teardown fires here too (abort/error path included).
        _publish(on_seq_num, 0)
        config_teardown()


def _scan_loop(run_one, nseq, rep, is_random, rng):
    """The rep / random scheduling (port of runSeq2.m:346-412).

    Returns ``False`` if the scan ran to completion, else the stop-reason sentinel ``run_one``
    returned (``"abort"`` or ``"yield"``) -- forwarded verbatim so ``run_scan_group`` can map it
    to a status. A ``rep=0`` forever scan only ever exits via such a sentinel.

    Faithful to runSeq2: ``is_random`` here is runSeq2's OWN global ``randperm``. In the
    production path this branch is NOT used -- ``sequence_runner._build_run_kwargs`` hands a
    pre-built, pre-scrambled order in ``indices`` with ``rep=1, is_random=False`` (the
    randomization lives in the prep layer, ybBuildScanJob's ``scramble_groups`` + ``stack``),
    so this loop just walks that order. The global-shuffle / forever branches remain for the
    rep=0 continuous monitor and for fidelity to runSeq2's varargin behavior."""
    if is_random:
        if rep == 0:
            idx = rng.randint(1, nseq)
            while True:
                stop = run_one(idx)
                if stop:
                    return stop
                idx = rng.randint(1, nseq)
        idxs = list(range(1, nseq + 1)) * rep
        rng.shuffle(idxs)
        for cur in idxs:
            stop = run_one(cur)
            if stop:
                return stop
        return False

    # sequential
    for i in range(1, nseq + 1):
        stop = run_one(i)
        if stop:
            return stop
    if rep == 0:
        while True:                                     # run forever until aborted / yielded
            for i in range(1, nseq + 1):
                stop = run_one(i)
                if stop:
                    return stop
    for _ in range(2, rep + 1):
        for i in range(1, nseq + 1):
            stop = run_one(i)
            if stop:
                return stop
    return False


def _arm_after_end_once(seq, cb, arg0, seqid):
    """Register a best-effort, one-shot ``cb(arg0, seqid, seq)`` in the after_end window.

    Used for both the SeqPlotter ``.seq`` dump (seq_dump.SeqDumpSession.on_compile) and
    the runtime-global capture (seq_dump.GlobalsCaptureSession.on_globals). The callback
    is DEFERRED to the seq's ``after_end`` callbacks rather than fired at compile time:
    after_end runs AFTER the before_start callbacks injected this shot's runtime globals
    (e.g. the persisted 616-EOM frequency, runtime_state.py) and BEFORE ``reset_globals``
    resets the non-persist ones (run_seq2.run_real). So a dump's bc_gen sees the real
    value (short ~ms ramp, not the ~15 s / ~60 MB zero-from ramp), and a global capture
    reads the genuine injected value (not the 0 init).

    A closure flag makes it fire ONCE per unique seqid: the seq object is reused across
    reps and duplicate-index points, and after_end fires every shot, so the flag matches
    the once-per-seqid contract and also avoids re-running if the first attempt fails.
    Because after_end only fires on the SUCCESS path, a shot that errors before after_end
    defers the work to the first successful shot (never acts on a half-run seq). Fully
    guarded: a seq without ``reg_after_end`` (test fakes) or any callback/registration
    error never perturbs a run.
    """
    reg = getattr(seq, "reg_after_end", None)
    if reg is None:
        return
    fired = []

    def _after_end_cb(_root):
        if fired:
            return
        fired.append(True)
        try:
            cb(arg0, seqid, seq)
        except Exception:  # noqa: BLE001 - after-end work never breaks a run
            pass

    try:
        reg(_after_end_cb)
    except Exception:  # noqa: BLE001 - registration must never break a run
        pass


def _default_compile_point(seqfn, seqparam):
    """Production prepare_seq leaf: build + run + generate a real ExpSeq (needs the engine).

    Mirrors ``s = ExpSeq(seqparam); func(s); s.generate()``. ``generate()`` compiles the
    serialized bytes into the engine ``pyseq`` (run_seq2.py's runnable interface). This is
    the only NEEDS-ENGINE seam in this module; tests inject a fake instead.
    """
    from exp_seq import ExpSeq
    s = ExpSeq(seqparam)
    seqfn(s)
    s.generate()
    return s


def _restart(seq):
    """Read ``seq.C.RESTART(0)`` (the AWG-restart retry flag; ``field(default)`` syntax).

    Almost always 0 (the AWG-restart trigger is commented out in MATLAB), so the retry loop
    runs once. Robust to a config object that has no RESTART field yet (-> no retry)."""
    try:
        return bool(seq.C.RESTART(0))
    except Exception:  # noqa: BLE001 - absent/non-DynProps RESTART -> no retry
        return False


def _bump_seq_id(seq_config):
    try:
        cur = seq_config.G.seq_id(1)
    except Exception:  # noqa: BLE001
        cur = 1
    seq_config.G.seq_id = cur + 1


def _publish(on_seq_num, n):
    if on_seq_num is not None:
        on_seq_num(n)


def _noop():
    pass
