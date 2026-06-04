"""sequence_runner.py -- the pyctrl scenario-3 run-loop CONSUMER orchestration.

Ports the NO-HARDWARE-testable core of ``matlab_new/YbExptCtrl/SequenceRunner.m``:
  * :func:`run_job`     -- the per-job pipeline (``runJob``), simplified for scenario 3.
  * :class:`IdleScheduler` -- the dummy-mode idle state machine (the main loop's empty-queue
    branch: ``off`` / ``default`` / ``last`` + the ``last_fallback_logged`` cross-iteration
    flag).

**Scenario-3 simplification (references/runtime-design.md).** In MATLAB, ``runJob`` decodes a
proprietary MATLAB byte payload (``getArrayFromByteStream``), ``ScanGroup.load``s it, sets up
the AWG, runs ``ybBuildScanJob`` (ROI / dated dir / camera prep), writes the ``ScanParamsSet``
memmap handshake, flushes stale camera frames, then ``runSeq2``. pyctrl is BOTH producer and
consumer, so the queue payload IS the **descriptor JSON** -- ``run_job`` just
``dispatch_descriptor``s it into a (ScanGroup, seq) and ``run_scan_group``s it. The AWG / ROI /
camera / memmap steps are runtime-camera + yb_analysis concerns (NEEDS-HARDWARE) and ride on
an optional ``on_prep`` hook (default no-op); they are NOT on the byte path. There is **no
memmap** (user-ratified): control + status go over the ExptServer ZMQ verbs.

Failure statuses preserved (a bad job must NEVER tear down the runner):
  ``descriptor error`` (malformed descriptor) · ``not migrated`` (seq not ported) ·
  ``prep error`` (on_prep hook, NEEDS-HARDWARE) · ``run error: <msg>`` (compile/run) ·
  ``aborted`` (graceful abort -- the MATLAB "aborted != ok" must-fix; MATLAB recorded ``ok``).

**Deferred to the hardware/integration step (NEEDS-HARDWARE, maintenance window):** the live
``python -m <module> <url>`` entry point that hosts a pyctrl copy of ``ExptServer.py``, binds
the ZMQ port, runs the ``while True`` consume loop, releases the DCAM camera on SIGTERM/
terminate, captures frames via ``pylablib``, and the startup mutual-exclusion guard. Those
need the real engine + camera + a bound socket; this module is their pure orchestration core.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import math
from collections import namedtuple

from dispatch_descriptor import NotMigratedError, dispatch_descriptor


# run_job's product: the final job status + the resolved seq name (for last-seq metadata).
JobResult = namedtuple("JobResult", ["status", "seq_name"])


def run_job(server, descriptor_json, job_id=None, dispatch=None, run=None,
            control_factory=None, on_prep=None, rng=None):
    """Run one queued descriptor end to end; return :class:`JobResult`.

    Args:
        server: the ExptServer-like hub (``finish_job``/``set_seq_name`` hooks; passed to the
            control channel). May be ``None`` in a pure-orchestration test.
        descriptor_json: the queued descriptor (JSON string or decoded dict).
        job_id: the queue job id, for ``finish_job`` / logging.
        dispatch: ``dispatch_descriptor`` override (tests).
        run: ``run_scan_group`` override (tests) -- the engine-driving run loop.
        control_factory: ``server -> control`` (default :class:`ControlChannel`).
        on_prep: optional NEEDS-HARDWARE hook ``on_prep(DispatchResult)`` for AWG/ROI/camera
            prep; raising it fails the job with ``prep error`` (the runner stays up).
        rng: ``random.Random`` used to scramble the run order (default a fresh PRNG); pass a
            seeded one for reproducibility / tests.
    """
    if dispatch is None:
        dispatch = dispatch_descriptor
    if run is None:
        from run_seq import run_scan_group as run
    if control_factory is None:
        from control_channel import ControlChannel as control_factory

    # --- 1. descriptor -> (ScanGroup, seq, opts). Bad descriptor / un-ported seq fail loud. ---
    try:
        disp = dispatch(descriptor_json)
    except NotMigratedError as e:
        return _fail(server, job_id, "not migrated: %s" % e, None)
    except Exception as e:  # noqa: BLE001 - any malformed descriptor
        return _fail(server, job_id, "descriptor error: %s" % e, None)

    _safe(server, "set_seq_name", job_id, disp.seq_name)

    # --- 2. NEEDS-HARDWARE prep (AWG upload / ROI / stale-frame flush) -- optional hook. ---
    if on_prep is not None:
        try:
            on_prep(disp)
        except Exception as e:  # noqa: BLE001
            return _fail(server, job_id, "prep error: %s" % e, disp.seq_name)

    # --- 3. run the scan (compile-per-point + per-seq gate live inside run_scan_group). ---
    control = control_factory(server) if server is not None else None
    try:
        # scan_name (the descriptor label, e.g. "LACScan") rides to the engine run so scan-prep
        # can stamp ScanName for the dashboard. The default run seam (run_scan_group) is only
        # used in tests, which inject a **kw-tolerant stub; the live engine run accepts it.
        # _build_run_kwargs hands run_scan_group a pre-built, pre-scrambled run order
        # (ybBuildScanJob's Scan.Params), so the scan loop just RUNS the order it is given.
        result = run(disp.seq, disp.scangroup, control=control, scan_name=disp.label,
                     **_build_run_kwargs(disp, rng))
    except Exception as e:  # noqa: BLE001 - a compile/run failure fails THIS job only
        return _fail(server, job_id, "run error: %s" % e, disp.seq_name)

    status = result.get("status", "ok") if isinstance(result, dict) else "ok"
    _finish(server, job_id, status)
    return JobResult(status, disp.seq_name)


class IdleScheduler:
    """The empty-queue dummy-mode branch of the main loop (SequenceRunner.m:121-191).

    Honors the Python-side selector: ``off`` (short pause), ``default`` (canonical DummySeq),
    ``last`` (replay the cached last real seq; fall back to default if none cached). The
    ``last_fallback_logged`` flag is a deliberate CROSS-ITERATION state machine -- it is NOT
    reset when a real job populates ``last_seq`` (so the next idle iteration logs "available"
    and clears the server-side fallback flag exactly once).
    """

    def __init__(self, server, run_dummy, run_last):
        self._server = server
        self._run_dummy = run_dummy      # () -> run the canonical DummySeq once
        self._run_last = run_last        # (last_seq) -> replay it once (scan_id/seq_id = -1)
        self.last_seq = None
        self.last_fallback_logged = False

    def cache_last_seq(self, seq):
        """Record the last successful real seq (do NOT touch ``last_fallback_logged``)."""
        if seq is not None:
            self.last_seq = seq

    def step(self, sleep):
        """One idle iteration; returns the action taken (for logging / tests)."""
        mode = _safe_ret(self._server, "dummy_mode", default="default")
        mode = str(mode) if mode is not None else "default"

        if mode == "off":
            sleep(0.1)
            return "off"

        if mode == "last":
            if self.last_seq is None:
                if not self.last_fallback_logged:
                    self.last_fallback_logged = True
                    _safe(self._server, "set_last_fallback_direct", True)
                self._run_dummy()
                return "last_fallback"
            if self.last_fallback_logged:
                self.last_fallback_logged = False
                _safe(self._server, "set_last_fallback_direct", False)
            self._run_last(self.last_seq)
            return "last"

        # default / unknown -> canonical DummySeq
        if self.last_fallback_logged:
            self.last_fallback_logged = False
            _safe(self._server, "set_last_fallback_direct", False)
        self._run_dummy()
        return "default"


# =========================================================================== #
# helpers
# =========================================================================== #
def _build_run_kwargs(disp, rng):
    """Map descriptor opts -> run_scan_group kwargs, then (production model) replace the
    rep/random knobs with a pre-built, pre-scrambled run ORDER (ybBuildScanJob: ``stack`` +
    ``scramble_groups`` -> ``Scan.Params``), run once (``rep=1, is_random=False``).

    The scramble lives HERE in the prep layer (like ``ybBuildScanJob.m``), NOT in runSeq2's own
    ``is_random`` branch -- ``run_scan_group`` just runs the order it is handed. Two cases fall
    through to the plain opts mapping (no pre-built order):
      * ``rep == 0`` (a run-forever continuous monitor) -- cannot pre-stack an infinite order;
        ``run_scan_group``'s loop runs forever, honoring the ``random`` flag.
      * a ScanGroup that can't be queried (a test stub) -- keep the opts-derived kwargs.
    """
    kw = _opts_to_run_kwargs(disp.opts)
    rep = kw.get("rep")
    if rep is not None and rep <= 0:
        # rep==0: run_scan_group's run-forever loop (continuous monitor). rep<0: leave it for
        # run_scan_group's "Cannot run by negative times" ValueError (do NOT swallow it).
        return kw
    order = _build_scan_order(disp.scangroup, rep, rng)
    if order is None:                            # un-queryable group -> plain opts mapping
        return kw
    kw["indices"] = order
    kw["rep"] = 1
    kw["is_random"] = False
    return kw


def _build_scan_order(scangroup, rep, rng):
    """Build ybBuildScanJob's ``Scan.Params`` from the ScanGroup; ``None`` if unqueryable.

    Number of passes = the explicit ``rep`` opt if given (>=1), else ``max(ceil(NumPerGroup /
    nseqs), 2)`` (MATLAB ``StackNum``, NumPerGroup default 200). An explicit ``rep`` is a
    deliberate pyctrl pass-count override (it bypasses the NumPerGroup formula AND the >=2
    floor, so ``rep=1`` is a single pass -- something MATLAB's ybBuildScanJob never produces).
    Scramble is driven SOLELY by
    the scan file's ``runp.Scramble`` and defaults to **0 (OFF)** -- per-pass scrambling happens
    only when the scan file sets ``g.runp().Scramble = 1`` (consistent with the ScanGroup runp
    default, which shadows ybBuildScanJob's ``scanp.Scramble(1)`` inline fallback)."""
    try:
        nseqs = int(scangroup.nseq())
        rp = scangroup.runp()
    except Exception:  # noqa: BLE001 - a test stub / non-ScanGroup -> fall through
        return None
    if nseqs <= 0:
        return None
    if rep is not None and rep >= 1:
        stack_num = int(rep)
    else:
        npg = _runp_scalar(rp, "NumPerGroup", 200)
        stack_num = max(math.ceil(npg / nseqs), 2)
    scramble = bool(_runp_scalar(rp, "Scramble", 0))     # default OFF; runp opt-in only
    from scan_prep import build_scan_order
    return build_scan_order(nseqs, stack_num=stack_num, scramble=scramble, rng=rng)


def _runp_scalar(rp, name, default):
    """Read a runp leaf (``rp.<name>(default)``), tolerant of absence."""
    try:
        return getattr(rp, name)(default)
    except Exception:  # noqa: BLE001
        return default


def _opts_to_run_kwargs(opts):
    """Map descriptor opts ``[(key, val), ...]`` to run_scan_group kwargs.

    Mirrors the runSeq2 varargin the dispatcher would forward: ``rep`` (number), ``random``
    (flag), ``tstartwait`` (number), ``pre_cb`` / ``post_cb`` (callables). Other opts
    (``email`` / ``scan_id`` / ``scan_struct``) are runner/G-context concerns and are not
    run_scan_group parameters -- ignored here.
    """
    kw = {}
    pre, post = [], []
    for key, val in opts:
        if key == "rep":
            kw["rep"] = int(val)
        elif key == "random":
            kw["is_random"] = bool(val)
        elif key == "tstartwait":
            kw["tstartwait"] = float(val)
        elif key == "pre_cb":
            pre.append(val)
        elif key == "post_cb":
            post.append(val)
    if pre:
        kw["pre_cb"] = pre
    if post:
        kw["post_cb"] = post
    return kw


def _finish(server, job_id, status):
    _safe(server, "finish_job", job_id, status)


def _fail(server, job_id, status, seq_name):
    _finish(server, job_id, status)
    return JobResult(status, seq_name)


def _safe(server, method, *args):
    """Call ``server.method(*args)`` best-effort (a missing hook / failure never aborts)."""
    if server is None:
        return
    fn = getattr(server, method, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass


def _safe_ret(server, method, default=None):
    if server is None:
        return default
    fn = getattr(server, method, None)
    if fn is None:
        return default
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default
