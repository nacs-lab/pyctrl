"""run_loop.py -- the pyctrl run-loop HOST (port of ``SequenceRunner.m``).

This is the long-lived backend process the monitor drives: pyctrl is the runtime, and the
MATLAB stack is a retired backup/reference. It hosts the shared :class:`ExptServer` ZMQ hub,
drains the queue, and runs each scan through the engine -- the Python counterpart of the MATLAB
``SequenceRunner(url)`` function (``matlab_new/YbExptCtrl/SequenceRunner.m``). It is launched
as ``python -m launcher.run_loop.runner <url>`` (see ``launcher/run_loop/runner.py``, which only
bootstraps ``sys.path`` and calls :func:`main` here).

This module was renamed from ``runner.py`` on 2026-07-22, and its camera / engine / SLM / AWG
blocks were split out into ``camera_runtime.py`` / ``engine_run.py`` / ``slm_runtime.py`` /
``awg_runtime.py``. What remains here is the pure run-loop host: URL resolution, the
single-backend guard, descriptor->job dispatch, the consume loop, the idle keep-alive wiring,
and the ExptServer + signal/teardown lifecycle.

Submission paths (verified with the user 2026-06-02): today a scan is started either by a
JSON **descriptor** (``submit_scan_descriptor`` -- the new monitor) or by the **".m run
button"** in a scan file (``ybStartScan`` -> ``submit_job`` with a MATLAB byte-stream
payload). The run-button path needs a live MATLAB, which is not part of the pyctrl runtime, and
its payload is MATLAB-proprietary (``getArrayFromByteStream``) -- so **pyctrl consumes the
descriptor path only**. :func:`handle_descriptor_pop` mirrors MATLAB ``handleDescriptorPop``
but, since pyctrl is BOTH producer and consumer, it dispatches a descriptor into a JSON job
payload it emits and consumes itself (``submit_job`` + ``link_descriptor_to_job``); the main
loop then pops that job and runs it via :func:`run_job.run_job` (which rebuilds the
ScanGroup with ``dispatch_descriptor`` and runs ``run_scan_group``). This reuses the job
queue + UI linkage verbatim while keeping the only two cross-backend contracts intact: the
descriptor JSON and the per-point serialized seq bytes (THE ONE RULE).

NO-HARDWARE testability: the orchestration (:func:`consume_loop`, :func:`handle_descriptor_pop`,
:func:`resolve_url`, :func:`assert_single_backend`) takes every
device/engine/socket dependency as an injected seam, so the full control flow is unit-tested
with fakes and never loads the engine, binds a socket, or opens the camera. :func:`serve`
wires the LIVE seams (real ExptServer, ``run_scan_group`` + ``seq_manager.new_run``, the
pylablib camera) and is the only NEEDS-HARDWARE entry; running it drives the FPGA/NI/camera
to expConfig defaults, so it is gated on a confirmed-safe hardware state.

Clean DCAM release on terminate (run-loop requirement, references/runtime-design.md): the
backend installs SIGTERM/SIGINT handlers and a ``finally`` that closes the camera handle and
stops the ZMQ worker, so the NEXT backend's camera open does not fail. The monitor's
restart-based handoff relies on this self-teardown.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import os
import signal
import sys
import time

from seq_reload import reload_experiment_modules
from run_job import IdleScheduler, run_job
from camera_runtime import make_camera_pump, _await_camera_init
from engine_run import load_configs, make_engine_run

# ZMQ-bind retry fallback URL (mirrors SequenceRunner.m's last-resort default).
DEFAULT_URL = "tcp://127.0.0.1:1408"
# Descriptor-drain cap per loop iteration (mirror handleDescriptorPop MAX_PER_ITER).
MAX_DESC_PER_ITER = 32


# =========================================================================== #
# URL resolution + single-backend guard (startup mutual exclusion)
# =========================================================================== #
def resolve_url(argv):
    """Resolve the bind URL: ``argv[0]`` -> ``$NACS_RUNNER_URL`` -> :data:`DEFAULT_URL`.

    Mirrors SequenceRunner.m's fallback chain (minus ``Consts().MatlabURL``, which is a
    MATLAB-config lookup). ``RunnerLauncher`` / ``PyctrlLauncher`` always pass it explicitly,
    so the monitor and the binding stay in sync.
    """
    if argv:
        url = str(argv[0]).strip()
        if url:
            return url
    env = os.environ.get("NACS_RUNNER_URL", "").strip()
    return env or DEFAULT_URL


def assert_single_backend(url, ping=None):
    """Refuse to start if a backend already answers ``ping`` at ``url`` (mutual exclusion).

    The pyctrl and legacy-MATLAB run loops never run simultaneously
    (references/runtime-design.md). The monitor's restart handoff frees the port before
    spawning us, so normally nothing answers; this is a belt-and-braces guard against a
    second backend silently failing to bind onto a live one. Raises :class:`RuntimeError`
    when a live backend is detected.
    """
    if ping is None:
        ping = _ping
    if ping(url):
        raise RuntimeError(
            "refusing to start: a backend already answers ping at %s "
            "(the two run loops are mutually exclusive -- stop the other backend first)"
            % url)


def _ping(url, timeout_ms=1000):
    """Send one ``ping`` to ``url`` and return True iff we get ``pong`` within the timeout.

    A REQ probe with LINGER 0; the socket is always closed (and the context terminated) so a
    failed probe leaves no half-open socket. Returns False on any error (nothing listening).
    """
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(url)
        sock.send_string("ping")
        if sock.poll(timeout_ms) == 0:
            return False
        return sock.recv_string() == "pong"
    except Exception:  # noqa: BLE001 - any failure means "no live backend"
        return False
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


# =========================================================================== #
# Descriptor pop -> JSON job (mirror handleDescriptorPop)
# =========================================================================== #
def handle_descriptor_pop(server, max_per_iter=MAX_DESC_PER_ITER, log=None,
                          pop_attr="pop_next_descriptor", priority="normal"):
    """Drain queued descriptors into JSON jobs; cap at ``max_per_iter`` per call.

    For each queued descriptor: submit its JSON body as a job payload (pyctrl is producer +
    consumer, so the payload IS the descriptor JSON -- no MATLAB byte stream), REUSING the
    descriptor's id for the job (``submit_job(job_id=desc_id)``) so the scan carries a single
    id -- the one ``submit_scan_descriptor`` returned and the .py scan script printed.
    ``link_descriptor_to_job`` then drops the now-redundant descriptor row instead of
    archiving a duplicate (its same-id branch). A bad descriptor is reported via
    ``finish_descriptor(id, 'error', msg)`` and must NEVER tear down the runner -- the loop
    keeps draining. A ``pop`` failure aborts this call (next iteration retries). Returns the
    number of descriptors dispatched.

    ``pop_attr``/``priority`` select the scheduling lane: the default drains FOREGROUND
    descriptors; :func:`handle_background_descriptor_pop` re-targets it at the background lane
    (``pop_next_background_descriptor`` + ``priority='background'``). ``submit_job`` is stamped
    with ``priority`` so the job is never briefly mis-laned before ``link_descriptor_to_job``
    copies the descriptor's authoritative lane onto it."""
    log = log or _noop_log
    dispatched = 0
    pop = getattr(server, pop_attr, None)
    if pop is None:
        return dispatched
    for _ in range(max_per_iter):
        try:
            desc = pop()
        except Exception as e:  # noqa: BLE001
            log("%s error: %s" % (pop_attr, e))
            return dispatched
        if not desc:
            return dispatched
        desc_id = desc["id"]
        try:
            payload = desc["descriptor"]
            # Carry the queue summary onto the JOB row so the dashboard's queue panel shows
            # axes/reps/scan_name while the scan RUNS. Best-effort.
            summary = _build_summary(payload)
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            # Reuse the descriptor's id for the job so the scan has a SINGLE id (the one the
            # .py script printed); link_descriptor_to_job then drops the descriptor row (its
            # same-id branch) instead of archiving a redundant second row.
            # place_at_descriptor: the built job takes the descriptor's QUEUE SLOT rather
            # than the back, so dispatch never reorders what the operator arranged with the
            # queue's up/down arrows (ExptServer.queue_move is lane-scoped, kind-agnostic).
            job_id = server.submit_job(payload, summary=summary, job_id=desc_id,
                                       priority=priority, place_at_descriptor=desc_id)
            server.link_descriptor_to_job(desc_id, job_id)
            dispatched += 1
        except Exception as e:  # noqa: BLE001 - bad descriptor: mark error, keep draining
            log("descriptor #%s FAILED: %s" % (desc_id, e))
            try:
                server.finish_descriptor(desc_id, "error", str(e))
            except Exception as e2:  # noqa: BLE001
                log("finish_descriptor error after dispatch failure: %s" % e2)
    log("descriptor drain cap (%d) hit; remaining processed next iteration" % max_per_iter)
    return dispatched


def handle_background_descriptor_pop(server, max_per_iter=MAX_DESC_PER_ITER, log=None):
    """Drain queued BACKGROUND (calibration) descriptors into background jobs -- the background
    lane's counterpart to :func:`handle_descriptor_pop`. getattr-guarded inside (a server without
    ``pop_next_background_descriptor`` drains nothing), so it is safe against a coarse server."""
    return handle_descriptor_pop(server, max_per_iter=max_per_iter, log=log,
                                 pop_attr="pop_next_background_descriptor", priority="background")


def _build_summary(descriptor):
    """The ybScanSummary-shaped queue dict from a descriptor (JSON str/bytes/dict). Best-effort
    -> ``None`` (queue UI degrades) on any failure. Used to stamp the built job row."""
    try:
        from scan_summary import build_descriptor_summary
        return build_descriptor_summary(descriptor)
    except Exception:  # noqa: BLE001
        return None


# =========================================================================== #
# The main consume loop (mirror SequenceRunner.m's while true)
# =========================================================================== #
def consume_loop(server, *, should_stop, run_job_fn=None, dispatch_pop=None,
                 dispatch_bg=None, idle=None, handle_camera=None, camera=None,
                 sleep=time.sleep, run_kwargs=None, log=None):
    """Drain the queue until ``should_stop()`` returns True.

    Per iteration (mirrors SequenceRunner.m:101-192): handle a pending camera command, drain
    descriptors into jobs, then pop one job. A job runs via ``run_job_fn``; an empty queue
    advances the idle dummy-mode state machine (:class:`IdleScheduler`).

    Every external dependency is injected:
        should_stop()      -> bool        loop predicate (the signal/terminate flag).
        run_job_fn(server, payload, job_id=, run=, ...) -> JobResult   (default run_job).
        dispatch_pop(server) -> int       descriptor drain (default handle_descriptor_pop).
        idle               IdleScheduler  the off/default/last dummy machine (None disables).
        handle_camera(server, camera)     camera-command pump (None disables).
        camera             the camera handle passed to handle_camera (None when absent).
        run_kwargs         dict of extra kwargs forwarded to run_job_fn (e.g. ``run=`` the
                           engine-wired run_scan_group).
    """
    if run_job_fn is None:
        run_job_fn = run_job
    if dispatch_pop is None:
        dispatch_pop = handle_descriptor_pop
    if dispatch_bg is None:
        dispatch_bg = handle_background_descriptor_pop
    log = log or _noop_log
    run_kwargs = run_kwargs or {}

    while not should_stop():
        if handle_camera is not None:
            try:
                handle_camera(server, camera)
            except Exception as e:  # noqa: BLE001 - a camera-cmd failure never stops the loop
                log("camera command failed: %s" % e)

        # Descriptors are dispatched between jobs (no half-built state). Bad descriptors
        # are marked 'error' inside dispatch_pop and never tear the runner down.
        try:
            dispatch_pop(server)
        except Exception as e:  # noqa: BLE001
            log("descriptor pop failed: %s" % e)

        try:
            job = server.pop_next_job()
        except Exception as e:  # noqa: BLE001
            log("pop_next_job error: %s" % e)
            sleep(1.0)
            continue

        if job is None:
            # Tier 2: a BACKGROUND (calibration) job -- only when the global toggle is on AND no
            # foreground scan is running/queued (both checked in _try_pop_background). It runs via
            # the SAME run_job_fn (run_job reads `background` off the payload, so its control
            # channel yields to foreground work at a shot boundary). Completion is owned HERE:
            # requeue_background re-queues a clean finish/yield to cycle, or archives an error.
            bg = _try_pop_background(server, dispatch_bg)
            if bg is not None:
                _safe_set_dummy_running(server, 0)
                _safe_set_background_running(server, 1, bg.get("seqName", ""))
                try:
                    result = run_job_fn(server, bg["payload"], job_id=bg["id"], **run_kwargs)
                    status = getattr(result, "status", "ok")
                    rq = getattr(server, "requeue_background", None)
                    if rq is not None:
                        try:
                            rq(bg["id"], status)
                        except Exception as e:  # noqa: BLE001
                            log("requeue_background failed: %s" % e)
                    # Background-only queues read as foreground-idle (mark_idle excludes
                    # background), so reset Running -> Init for the operator's "stopped" view.
                    mark_idle = getattr(server, "mark_idle_if_queue_empty", None)
                    if mark_idle is not None:
                        try:
                            mark_idle()
                        except Exception as e:  # noqa: BLE001
                            log("end-of-bg idle-status reset failed: %s" % e)
                    log("bg job #%s finished (%s)" % (bg["id"], status))
                finally:
                    _safe_set_background_running(server, 0)
                continue
            # TRULY idle (no job, no background): if the dummy keep-alive is OFF, release
            # the process-global NI Task. A cached-open Task keeps Dev1's AO channels
            # RESERVED (DAQmx) even between scans, so the dashboard's out-of-band DC set
            # (ni_set_driver.py, a separate process) was refused with -50103 "resource is
            # reserved" until a backend restart. Releasing here is race-free (this thread
            # is the only NiDAQRunner owner) and cheap to undo: the next scan's
            # _get_session sees _session is None and rebuilds. With the keep-alive ON the
            # session is deliberately KEPT -- dummy shots re-drive the NI defaults every
            # cycle, so an out-of-band set would be stomped anyway; the write path tells
            # the operator to turn Dummy off instead.
            _release_ni_when_idle(server, log)
            # Tier 3: the DummySeq keep-alive (off/default/last) when nothing else runs.
            if idle is not None:
                _safe_set_dummy_running(server, 1)
                idle.step(sleep)
            else:
                sleep(0.1)
            continue

        _safe_set_dummy_running(server, 0)
        result = run_job_fn(server, job["payload"], job_id=job["id"], **run_kwargs)
        # End of sequence: a finished finite scan leaves seq_status == Running (start_scan set it
        # at scan begin; nothing resets it). If the queue is now empty, return the status to idle
        # so get_status reports "stopped" (bug-pyctrl-status-not-reset-idle). The server method
        # checks the queue + resets the status ATOMICALLY under __queue_lock, so this cannot race
        # a concurrent submit, and it is status-only (never touches the seq request). getattr-
        # guarded so the NO-HARDWARE consume_loop tests (fake servers) skip it cleanly.
        mark_idle = getattr(server, "mark_idle_if_queue_empty", None)
        if mark_idle is not None:
            try:
                mark_idle()
            except Exception as e:  # noqa: BLE001 - a status reset must never stop the loop
                log("end-of-job idle-status reset failed: %s" % e)
        # 'last'-mode replay caching (the compiled-seq capture) is a documented follow-up;
        # until then IdleScheduler falls back to the canonical DummySeq when no seq is
        # cached. We still surface the resolved name for logging.
        if result is not None:
            log("job #%s finished (%s)" % (job["id"], getattr(result, "status", "?")))


def make_idle(server, dummy_seq=None, run_real=None, sleep=time.sleep, log=None):
    """Build the :class:`IdleScheduler` for the empty-queue dummy keep-alive (live wiring).

    ``run_dummy`` runs the pre-compiled canonical DummySeq once; ``run_last`` replays a cached
    seq (the capture path is a follow-up, so a missing cache falls back to default inside the
    scheduler). Both fire the engine -> NEEDS-HARDWARE; only built inside :func:`serve`.

    A keep-alive run that ERRORS must NEVER tear down the backend (mirrors MATLAB
    ``runDummyOnce``'s try/catch): the run is wrapped, the error logged, and a 1 s back-off
    applied so a persistent failure (e.g. a missing device package) doesn't hot-spin.
    """
    if run_real is None:
        from run_seq2 import run_real as run_real
    log = log or _noop_log

    def _safe_run(seq, what):
        try:
            run_real(seq)
        except Exception as e:  # noqa: BLE001 - a keep-alive failure must not kill the runner
            log("%s run failed: %s (retry next idle)" % (what, e))
            sleep(1.0)

    def run_dummy():
        if dummy_seq is not None:
            _safe_run(dummy_seq, "dummy")

    def run_last(seq):
        _safe_run(seq, "last-seq")

    return IdleScheduler(server, run_dummy=run_dummy, run_last=run_last)


def _compile_dummy():
    """Build + ``generate()`` the canonical DummySeq once for idle replay (engine)."""
    from DummySeq import DummySeq
    s = DummySeq()
    s.generate()
    return s


# =========================================================================== #
# Live entry point (NEEDS-HARDWARE) -- host ExptServer + drive the loop
# =========================================================================== #
def serve(url, *, server_factory=None, with_camera=True, with_idle=True, log=print):
    """Host the ExptServer at ``url`` and run the consume loop until terminated.

    LIVE path: binds the ZMQ port, opens the camera, compiles DummySeq, wires the engine run,
    installs SIGTERM/SIGINT handlers, and runs :func:`consume_loop`. The ``finally`` closes
    the camera (releases the single DCAM handle) and stops the ZMQ worker so the next backend
    can bind/open -- the self-teardown the monitor's restart handoff depends on.

    ⚠ Running this drives the FPGA/NI/camera to expConfig defaults on every shot (and every
    idle DummySeq). Only start it on a confirmed-safe hardware state.
    """
    assert_single_backend(url)
    from logging_setup import setup_logging                  # mirror terminal output -> log files
    _logs = setup_logging()
    if _logs:
        log("[runner] mirroring terminal output to %s" % _logs["mirror"])
    load_configs(log=lambda m: log("[runner] %s" % m))      # expConfig snapshot + engine config.yml
    from seq_config import SeqConfig
    seq_config = SeqConfig.get()                            # the real config activated above
    if server_factory is None:
        from ExptServer import ExptServer as server_factory
    server = server_factory(url)
    _force_dummy_off(server, log)       # boot idle-SAFE: no DummySeq firing until enabled
    log("[runner] ExptServer bound at %s -- entering loop" % url)

    stop = {"flag": False}

    def _request_stop(signum=None, frame=None):  # noqa: ARG001
        stop["flag"] = True

    _install_signal_handlers(_request_stop)
    # Orphan guard: run_monitor (our spawner) owns detection + the .h5 writer.
    # If it dies (e.g. the 2026-07-01 hdf5.dll segfault that killed the parent
    # process), we would otherwise keep firing the scan loop forever with no
    # consumer/saver -- every shot silently lost. Watch YB_PARENT_PID and trip
    # the SAME graceful stop flag on its death so the `finally: _teardown`
    # releases the DCAM handle (a held handle wedges the next backend --
    # bug-pyctrl-orca-restart-race-dcam-wedge) before we hard-exit.
    _start_parent_watchdog(_request_stop, log=log)

    camera = None
    try:
        if with_camera:
            camera = _await_camera_init(server, seq_config=seq_config, log=log)
        idle = None
        if with_idle:
            try:
                dummy_seq = _compile_dummy()
                idle = make_idle(server, dummy_seq=dummy_seq, log=lambda m: log("[runner] %s" % m))
                log("[runner] DummySeq compiled -- idle keep-alive ready (mode OFF until enabled)")
            except Exception as e:  # noqa: BLE001 - idle is optional; loop still serves jobs
                log("[runner] DummySeq compile failed (%s) -- idle disabled" % e)
        run = make_engine_run(server, camera, seq_config,    # engine + camera-arm + capture
                              log=lambda m: log("[runner] %s" % m))

        def _pre_job_reload():
            # Per-job hot-reload so live edits take effect without a restart: ported seq/step
            # modules (rehash()+str2func analog) AND the executable expConfig.py (in place, so
            # SeqConfig identity + runtime globals are preserved). lib/ still needs a restart.
            reload_experiment_modules(log=lambda m: log("[runner] %s" % m))
            from seq_config import SeqConfig
            SeqConfig.load_real(reload=True)

        consume_loop(
            server,
            should_stop=lambda: stop["flag"],
            handle_camera=make_camera_pump() if with_camera else None,
            camera=camera,
            idle=idle,
            run_kwargs={"run": run, "reload_modules": _pre_job_reload},
            log=lambda m: log("[runner] %s" % m),
        )
    finally:
        _teardown(server, camera, log=log)
    # Hard-exit: skip CPython's shutdown, which HANGS on Windows once the libnacs engine + its
    # bundled libzmq are loaded (the DLL-detach wedge -- same one tests dodge). Teardown already
    # released the camera/NI/worker, so an immediate TerminateProcess is safe and is the only
    # way Ctrl+C / a monitor stop actually exits instead of hanging.
    _hard_exit(0)


def _hard_exit(code):
    """Exit NOW, bypassing the engine/libzmq DLL-detach hang on Windows.

    ``os._exit`` still runs ExitProcess -> DLL detach (which wedges), so on Windows we
    TerminateProcess our own handle (skips DLL detach entirely), mirroring the test conftest.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0001, False, os.getpid())   # PROCESS_TERMINATE
        kernel32.TerminateProcess(handle, int(code))
    os._exit(code)


def _pid_alive(pid):
    """True iff process ``pid`` is running. Windows uses OpenProcess + exit-code
    probe (os.kill(pid, 0) is unreliable there); POSIX uses signal 0. Errs on the
    side of 'alive' on any unexpected error so a probe glitch never false-kills a
    healthy run."""
    if pid <= 0:
        return True
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = kernel32.OpenProcess(
                SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False   # gone (or access-denied on a foreign pid, which our own child never is)
            try:
                STILL_ACTIVE = 259
                code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                return bool(ok) and code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(h)
        os.kill(pid, 0)
        return True
    except OSError:
        return False   # ESRCH / no such process
    except Exception:  # noqa: BLE001
        return True    # unknown probe failure -> assume alive, don't false-kill


def _start_parent_watchdog(request_stop, poll_interval_s=3.0, log=None):
    """Daemon thread: call ``request_stop()`` when the pid in ``YB_PARENT_PID``
    dies. Trips the graceful stop flag (NOT a hard exit) so ``serve``'s
    ``finally: _teardown`` releases the camera/NI/ZMQ before the process ends.
    No-op if YB_PARENT_PID is unset/invalid (e.g. a hand-started runner)."""
    import threading
    raw = os.environ.get("YB_PARENT_PID", "")
    try:
        parent_pid = int(raw)
    except (TypeError, ValueError):
        parent_pid = 0
    if parent_pid <= 0:
        return None
    log = log or _noop_log

    def _loop():
        while True:
            time.sleep(poll_interval_s)
            if not _pid_alive(parent_pid):
                try:
                    log("[runner] parent pid %d gone -- stopping backend "
                        "(no consumer/saver alive)" % parent_pid)
                except Exception:  # noqa: BLE001
                    pass
                request_stop()
                return

    t = threading.Thread(target=_loop, name="runner-parent-watchdog", daemon=True)
    t.start()
    return t


def _install_signal_handlers(handler):
    """Install SIGTERM/SIGINT -> ``handler`` (best-effort; SIGTERM may be absent on Win)."""
    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not in main thread / unsupported -> rely on finally


def _teardown(server, camera, log=print):
    """Release the camera (DCAM handle), the NI session, and the ZMQ worker -- clean terminate.

    Camera first: the next backend's camera open fails if we still hold the handle. Release the
    process-global NI Task too (else nidaqmx warns "resources may still be reserved" and the
    card stays reserved for the next backend). Then stop the worker before the socket drops (a
    live worker on a closed socket crashes).
    """
    if camera is not None:
        try:
            camera.close()
            log("[runner] camera closed")
        except Exception as e:  # noqa: BLE001
            log("[runner] camera close failed: %s" % e)
    try:
        from devices.nidaq import NiDAQRunner
        NiDAQRunner.clear_session()              # release the NI Task (no DaqResourceWarning)
    except Exception:  # noqa: BLE001 - nidaqmx absent / nothing to release
        pass
    try:
        server.stop_worker()
    except Exception as e:  # noqa: BLE001
        log("[runner] stop_worker failed: %s" % e)
    log("[runner] teardown complete")


def _force_dummy_off(server, log=print):
    """Default the keep-alive mode to 'off' at boot so the backend does NOT fire DummySeq
    (a full MOT sequence) on the FPGA the instant it binds. The monitor's dummy radios still
    switch it live (ZMQ ``set_dummy_mode``); this just makes the SAFE state the startup default
    for pyctrl. Best-effort -- a coarse ExptServer without the attr is left as-is.
    """
    try:
        with server._ExptServer__dummy_lock:                # owned in-process; pre-clients
            server._ExptServer__dummy_mode = "off"
        log("[runner] keep-alive defaulted OFF (enable from the monitor's Dummy radios)")
    except Exception:  # noqa: BLE001
        pass


def _safe_set_dummy_running(server, flag):
    fn = getattr(server, "set_dummy_running", None)
    if fn is not None:
        try:
            fn(flag)
        except Exception:  # noqa: BLE001
            pass


def _safe_set_background_running(server, flag, name=""):
    fn = getattr(server, "set_background_running", None)
    if fn is not None:
        try:
            fn(flag, name)
        except Exception:  # noqa: BLE001
            pass


def _release_ni_when_idle(server, log=None):
    """Release the cached NI Task while the backend is TRULY idle (consume-loop idle branch).

    An open (even stopped) DAQmx Task keeps its AO channels reserved, blocking the dashboard's
    out-of-band DC set (``ni_set_driver.py``, a separate process) with -50103 "resource is
    reserved". Called ONLY from the consume loop's no-job/no-background branch, so it runs on
    the run-loop thread -- the sole NiDAQRunner owner -- and cannot race a shot.

    Skipped while the dummy keep-alive is ON (mode != 'off'): the very next dummy shot would
    re-arm the session anyway (and re-drive the NI defaults, stomping any out-of-band set), and
    releasing between dummy shots would just force a ~0.7 s channel rebuild every cycle. A
    server without ``dummy_mode`` (fakes / coarse hubs) is treated as dummy-off. Best-effort:
    never raises into the loop; a no-session call is a cheap no-op (the common idle iteration).
    """
    mode_fn = getattr(server, "dummy_mode", None)
    if mode_fn is not None:
        try:
            if mode_fn() != "off":
                return
        except Exception:  # noqa: BLE001 - can't read the mode -> don't touch the session
            return
    try:
        from devices.nidaq import NiDAQRunner
        if not NiDAQRunner.has_session():
            return
        NiDAQRunner.clear_session()
    except Exception:  # noqa: BLE001 - releasing is best-effort; never stop the loop
        return
    if log is not None:
        try:
            log("NI session released (idle, dummy off) -- out-of-band NI writes allowed")
        except Exception:  # noqa: BLE001
            pass


def _try_pop_background(server, dispatch_bg):
    """Pop the next BACKGROUND (calibration) job, or ``None`` if the lane must not run now.

    Returns None when (in order): the global background toggle is off; a foreground scan is
    running or queued (``has_foreground_work`` -- includes not-yet-dispatched foreground
    descriptors); the server predates the background lane; or no background job is queued.
    Otherwise drains background descriptors into jobs (``dispatch_bg``) then pops one. Every
    server hook is getattr-guarded, so a coarse server (no background methods) cleanly yields
    None and the dummy keep-alive runs instead."""
    get_enabled = getattr(server, "get_background_enabled", None)
    if get_enabled is not None:
        try:
            if not get_enabled():
                return None
        except Exception:  # noqa: BLE001
            return None
    has_fg = getattr(server, "has_foreground_work", None)
    if has_fg is not None:
        try:
            if has_fg():
                return None
        except Exception:  # noqa: BLE001
            return None
    pop_bg = getattr(server, "pop_next_background_job", None)
    if pop_bg is None:
        return None
    if dispatch_bg is not None:
        try:
            dispatch_bg(server)
        except Exception:  # noqa: BLE001 - a bad bg descriptor must not stop the loop
            pass
    try:
        return pop_bg()
    except Exception:  # noqa: BLE001
        return None


def _noop_log(_msg):
    pass


def main(argv=None):
    """``python -m`` entry: resolve the URL and serve. Returns the process exit code."""
    if argv is None:
        argv = sys.argv[1:]
    url = resolve_url(argv)
    try:
        serve(url)
    except RuntimeError as e:  # single-backend guard / bind failure
        print("[runner] %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
