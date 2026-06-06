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

from seq_config import SeqConfig


def run_scan_group(seqfn, scangroup, indices=None, rep=1, is_random=False,
                   pre_cb=(), post_cb=(), tstartwait=0.0,
                   control=None, compile_point=None, run_real=None,
                   seq_config=None, new_run=None, on_seq_num=None,
                   config_teardown=None, sleep=time.sleep, rng=None,
                   on_compile=None):
    """Run a ScanGroup as a scan (port of ``runSeq2(func, scangroup, ...)``).

    Returns a result dict ``{"status": "ok"|"aborted", "nseq": <shots completed>}``. (The
    MATLAB graceful abort returns no status, recording ``'ok'`` -- the "aborted != ok"
    must-fix; pyctrl distinguishes them.)
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
        seqlist[idx] = compile_point(seqfn, seqparam)
        # Auto-dump hook: ARM (don't fire) the SeqPlotter .seq dump for this unique
        # compiled sequence. The dump must run DURING the shot, not here at compile
        # time: a global-dependent ramp (e.g. the 616-EOM slow ramp) reads its
        # driving global at bc_gen, and that global is 0 until a before_start
        # callback injects it inside run_real -- dumping at compile would build the
        # degenerate ~15 s / ~60 MB zero-from ramp. So we register the dump on the
        # seq's after_end callbacks (fired after before_start set the global and
        # before reset_globals wipes it), capturing the real per-shot bytecode.
        if on_compile is not None:
            _arm_dump(seqlist[idx], on_compile, arg0, seqid)

    def run_cb(cbs, idx):
        for cb in cbs:
            cb(counter["cur_seq_num"], indices[idx - 1])

    def run_one(idx):
        """The per-shot body (MATLAB nested ``run_seq``). Returns True to ABORT."""
        while True:                                     # C.RESTART retry loop
            if control is not None and control.check_pause_abort():
                return True                             # gate: abort at this boundary
            prepare_seq(idx)
            if tstartwait > 0:
                sleep(tstartwait)                       # NI-DAQ driver-timing workaround
            run_cb(pre_cb, idx)
            cur = seqlist[idx]
            # set_global for scan vars: usevar dormant -> empty -> no-op (kept for fidelity).
            run_real(cur)
            run_cb(post_cb, idx)
            if not _restart(cur):
                break
        counter["cur_seq_num"] += 1                     # AFTER a successful shot, not on abort
        _publish(on_seq_num, counter["cur_seq_num"])
        _bump_seq_id(seq_config)
        return False

    try:
        if new_run is not None:
            new_run()                                   # SeqManager.new_run() (engine reset)
        aborted = _scan_loop(run_one, nseq, rep, is_random, rng)
        return {"status": "aborted" if aborted else "ok", "nseq": counter["cur_seq_num"]}
    finally:
        # End-of-run reset (CurrentSeqNum -> 0). Abort/Pause are NOT cleared here -- the
        # single-clear-point (clear-at-job-start) policy clears them at the next begin_scan, so
        # a stale flag is cured by the next job, not by end-of-run (control_channel.py). Config
        # bracket teardown fires here too (abort/error path included).
        _publish(on_seq_num, 0)
        config_teardown()


def _scan_loop(run_one, nseq, rep, is_random, rng):
    """The rep / random scheduling (port of runSeq2.m:346-412). Returns True if aborted.

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
                if run_one(idx):
                    return True
                idx = rng.randint(1, nseq)
        idxs = list(range(1, nseq + 1)) * rep
        rng.shuffle(idxs)
        for cur in idxs:
            if run_one(cur):
                return True
        return False

    # sequential
    for i in range(1, nseq + 1):
        if run_one(i):
            return True
    if rep == 0:
        while True:                                     # run forever until aborted
            for i in range(1, nseq + 1):
                if run_one(i):
                    return True
    for _ in range(2, rep + 1):
        for i in range(1, nseq + 1):
            if run_one(i):
                return True
    return False


def _arm_dump(seq, on_compile, arg0, seqid):
    """Register a best-effort, one-shot ``.seq`` dump in run_real's after_end window.

    ``on_compile(arg0, seqid, seq)`` is the SeqPlotter auto-dump (seq_dump.py). It is
    DEFERRED to the seq's ``after_end`` callbacks rather than fired at compile time:
    after_end runs AFTER the before_start callbacks injected this shot's runtime
    globals (e.g. the persisted 616-EOM frequency, runtime_state.py) and BEFORE
    ``reset_globals`` resets the non-persist ones (run_seq2.run_real), so the dump's
    bc_gen sees the real value and builds the short (~ms) ramp instead of the
    ~15 s / ~60 MB ramp a zero-valued global yields at compile time.

    A closure flag makes it fire ONCE per unique seqid: the seq object is reused
    across reps and duplicate-index points, and after_end fires every shot, so the
    flag matches the old once-per-seqid contract and also avoids re-running the dump
    every shot if the first attempt fails. Because after_end only fires on the
    SUCCESS path, a shot that errors before after_end defers the dump to the first
    successful shot (never dumps a half-run seq). Fully guarded: a seq without
    ``reg_after_end`` (test fakes) or any dump/registration error never perturbs a run.
    """
    reg = getattr(seq, "reg_after_end", None)
    if reg is None:
        return
    fired = []

    def _dump_cb(_root):
        if fired:
            return
        fired.append(True)
        try:
            on_compile(arg0, seqid, seq)
        except Exception:  # noqa: BLE001 - dumping is never allowed to break a run
            pass

    try:
        reg(_dump_cb)
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
