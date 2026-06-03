"""runner.py -- the pyctrl scenario-3 run-loop HOST (port of ``SequenceRunner.m``).

This is the long-lived backend process the new monitor drives in **scenario 3** (new
monitor + pyctrl, MATLAB off). It hosts the shared :class:`ExptServer` ZMQ hub, drains the
queue, and runs each scan through the engine -- the Python counterpart of the MATLAB
``SequenceRunner(url)`` function (``matlab_new/YbExptCtrl/SequenceRunner.m``). It is launched
as ``python -m ybctrl.run_loop.runner <url>`` (see ``ybctrl/run_loop/runner.py``, which only
bootstraps ``sys.path`` and calls :func:`main` here).

Submission paths (verified with the user 2026-06-02): today a scan is started either by a
JSON **descriptor** (``submit_scan_descriptor`` -- the new monitor) or by the **".m run
button"** in a scan file (``ybStartScan`` -> ``submit_job`` with a MATLAB byte-stream
payload). The run-button path needs a live MATLAB, which is OFF in scenario 3, and its
payload is MATLAB-proprietary (``getArrayFromByteStream``) -- so **pyctrl consumes the
descriptor path only**. :func:`handle_descriptor_pop` mirrors MATLAB ``handleDescriptorPop``
but, since pyctrl is BOTH producer and consumer, it dispatches a descriptor into a JSON job
payload it emits and consumes itself (``submit_job`` + ``link_descriptor_to_job``); the main
loop then pops that job and runs it via :func:`sequence_runner.run_job` (which rebuilds the
ScanGroup with ``dispatch_descriptor`` and runs ``run_scan_group``). This reuses the job
queue + UI linkage verbatim while keeping the only two cross-backend contracts intact: the
descriptor JSON and the per-point serialized seq bytes (THE ONE RULE).

NO-HARDWARE testability: the orchestration (:func:`consume_loop`, :func:`handle_descriptor_pop`,
:func:`resolve_url`, :func:`assert_single_backend`, :func:`handle_camera_cmd`) takes every
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

from sequence_runner import IdleScheduler, run_job

# ZMQ-bind retry fallback URL (mirrors SequenceRunner.m's last-resort default).
DEFAULT_URL = "tcp://127.0.0.1:1408"
# Descriptor-drain cap per loop iteration (mirror handleDescriptorPop MAX_PER_ITER).
MAX_DESC_PER_ITER = 32
# Seconds to wait for the camera-init command before compiling/serving (mirror the
# MATLAB 15 s camera-init wait); the monitor sends camera_init on startup.
CAMERA_INIT_WAIT_S = 15.0
# Repo root (…/pyctrl/YbExptCtrl/runner.py -> …/pyctrl -> repo) for locating config.yml.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    The two run loops NEVER run simultaneously (the three scenarios are mutually exclusive,
    references/runtime-design.md). The monitor's restart handoff frees the port before
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
def handle_descriptor_pop(server, max_per_iter=MAX_DESC_PER_ITER, log=None):
    """Drain queued descriptors into JSON jobs; cap at ``max_per_iter`` per call.

    For each queued descriptor: submit its JSON body as a job payload (pyctrl is producer +
    consumer, so the payload IS the descriptor JSON -- no MATLAB byte stream) and link the
    descriptor row to the new job id (queue-UI linkage). A bad descriptor is reported via
    ``finish_descriptor(id, 'error', msg)`` and must NEVER tear down the runner -- the loop
    keeps draining. A ``pop_next_descriptor`` failure aborts this call (next iteration
    retries). Returns the number of descriptors dispatched.
    """
    log = log or _noop_log
    dispatched = 0
    for _ in range(max_per_iter):
        try:
            desc = server.pop_next_descriptor()
        except Exception as e:  # noqa: BLE001
            log("pop_next_descriptor error: %s" % e)
            return dispatched
        if not desc:
            return dispatched
        desc_id = desc["id"]
        try:
            payload = desc["descriptor"]
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            job_id = server.submit_job(payload)
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


# =========================================================================== #
# The main consume loop (mirror SequenceRunner.m's while true)
# =========================================================================== #
def consume_loop(server, *, should_stop, run_job_fn=None, dispatch_pop=None,
                 idle=None, handle_camera=None, camera=None, sleep=time.sleep,
                 run_kwargs=None, log=None):
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
            if idle is not None:
                _safe_set_dummy_running(server, 1)
                idle.step(sleep)
            else:
                sleep(0.1)
            continue

        _safe_set_dummy_running(server, 0)
        result = run_job_fn(server, job["payload"], job_id=job["id"], **run_kwargs)
        # 'last'-mode replay caching (the compiled-seq capture) is a documented follow-up;
        # until then IdleScheduler falls back to the canonical DummySeq when no seq is
        # cached. We still surface the resolved name for logging.
        if result is not None:
            log("job #%s finished (%s)" % (job["id"], getattr(result, "status", "?")))


# =========================================================================== #
# Camera command pump (NEEDS-HARDWARE; testable with a fake camera)
# =========================================================================== #
def handle_camera_cmd(server, camera):
    """Execute one pending camera command and report the result back to the server.

    Mirrors SequenceRunner.m ``handleCameraCmd``: pops ``get_camera_cmd()`` (init /
    apply_settings / close) and drives the camera, then ``set_camera_result``. When
    ``camera`` is ``None`` (pylablib absent / camera not opened) an init/apply is reported as
    a failure and a close is acknowledged -- so the monitor's camera pane shows a truthful
    disconnected state instead of hanging.

    The camera object (the pylablib wrapper, a Phase-5 follow-up) must expose:
        init(roi, exposure)           -> (roi, exposure)   open/configure; raise on failure
        apply_settings(roi, exposure) -> (roi, exposure)
        close()                       -> None              release the DCAM handle
        current_roi()                 -> [x, y, w, h]
    """
    try:
        cmd = server.get_camera_cmd()
    except Exception:  # noqa: BLE001
        return
    if not cmd:
        return
    kind = cmd.get("cmd")
    roi = cmd.get("roi") or [0, 0, 4096, 2304]
    exposure = cmd.get("exposure_time")

    if kind == "close":
        if camera is not None:
            try:
                camera.close()
            except Exception:  # noqa: BLE001
                pass
        server.set_camera_result(False, [0, 0, 0, 0], "")
        return

    if kind in ("init", "apply_settings"):
        if camera is None:
            server.set_camera_result(False, roi, "camera unavailable (pylablib not opened)")
            return
        try:
            fn = camera.init if kind == "init" else camera.apply_settings
            actual_roi, actual_exp = fn(roi, exposure)
            server.set_camera_result(True, actual_roi, "", actual_exp)
        except Exception as e:  # noqa: BLE001
            cur = camera.current_roi() if hasattr(camera, "current_roi") else roi
            server.set_camera_result(False, cur, str(e))


def open_camera(roi=None, exposure=None, log=None):
    """Open the Orca camera via the pylablib wrapper, or return ``None`` if unavailable.

    The pylablib capture wrapper (Orca-Quest ``C15550-20UP`` over ``dcamapi.dll``) is a
    separate Phase-5 deliverable; until ``pylablib-lightweight`` is installed and the wrapper
    module exists, this import fails and we degrade gracefully (the backend boots, the camera
    pane shows disconnected, scans needing frames fail loudly at run time). Kept lazy so the
    NO-HARDWARE suite never imports pylablib.
    """
    log = log or _noop_log
    try:
        from orca_camera import OrcaCamera  # Phase-5 follow-up wrapper (not yet built)
    except Exception as e:  # noqa: BLE001 - module/pylablib absent
        log("camera wrapper unavailable (%s) -- backend boots camera-less" % e)
        return None
    try:
        return OrcaCamera(roi=roi, exposure=exposure)
    except Exception as e:  # noqa: BLE001
        log("camera open failed: %s -- backend boots camera-less" % e)
        return None


# =========================================================================== #
# Live engine wiring (NEEDS-HARDWARE)
# =========================================================================== #
def load_configs(log=None):
    """Load BOTH configs the live run needs, before compiling any sequence.

    (1) ``SeqConfig.load_real()`` -- activate the captured real expConfig snapshot
    (channel aliases / defaults) as the SeqConfig singleton, so builds produce correct bytes.
    (2) ``seq_manager.load_config_string(config.yml)`` -- load the engine's channel + timing
    config, WITHOUT which ``tick_per_sec`` / ``generate`` raise "Sequence time unit not
    initialized". Both are required; serve() calls this once at startup.
    """
    log = log or _noop_log
    import seq_manager
    from seq_config import SeqConfig
    SeqConfig.load_real()
    cfg = os.path.join(REPO_ROOT, "matlab_new", "config.yml")
    with open(cfg) as f:
        seq_manager.load_config_string(f.read())
    log("config loaded (expConfig snapshot + engine config.yml=%s)" % cfg)


def make_engine_run(server, camera, seq_config):
    """Build the live ``run`` seam handed to :func:`sequence_runner.run_job`.

    Wraps ``run_scan_group`` with: the engine reset (``seq_manager.new_run``), per-scan camera
    arming (external rising-edge trigger; the seq's Imag399 step pulses ``TTLOrcaTrig``), and a
    per-shot capture ``post_cb`` (:func:`frame_capture.make_capture_post_cb`) that reads
    ``NumImages`` frames and publishes them via ``server.store_imgs`` / ``seq_finish``. The
    scan id (for frame routing) is recorded from ``control.begin_scan`` via
    :class:`_ScanIdRecorder`; the seq id comes from ``seq_config.G.seq_id``.

    ``compile_point`` / ``run_real`` keep their engine defaults. ``config_teardown`` is NOT
    overridden (pyctrl ``SeqConfig.reset()`` would wipe the real config between shots).
    """
    import seq_manager
    from run_seq import run_scan_group

    def run(seq, scangroup, control=None, **opts):
        num_images = _num_images(scangroup)
        post = list(opts.pop("post_cb", []) or [])
        scan_box = {"id": -1}
        if control is not None:
            control = _ScanIdRecorder(control, scan_box)

        armed = False
        if camera is not None and num_images > 0:
            try:
                camera.flush()                                   # drop stale frames (MATLAB flushdata)
                camera.start_video(external=True, nframes=max(num_images * 4, 16))
                armed = True
                from frame_capture import make_capture_post_cb
                post.append(make_capture_post_cb(
                    camera, server, num_images, lambda: scan_box["id"], seq_config))
            except Exception:  # noqa: BLE001 - camera arm failure must not crash the job pre-run
                armed = False
        try:
            return run_scan_group(seq, scangroup, control=control, post_cb=post,
                                  new_run=seq_manager.new_run, **opts)
        finally:
            if armed:
                try:
                    camera.stop_video()
                except Exception:  # noqa: BLE001
                    pass

    return run


def _num_images(scangroup):
    """NumImages for the scan (descriptor runp), default 1; bad/absent -> 0 (no capture)."""
    try:
        return int(scangroup.runp().NumImages(1))
    except Exception:  # noqa: BLE001
        return 0


class _ScanIdRecorder:
    """Wrap a ControlChannel to capture the ``scan_id`` that ``begin_scan`` produces.

    ``run_scan_group`` only calls ``begin_scan`` / ``check_pause_abort`` on the control; this
    forwards both and stashes the (non-None) scan id into ``box['id']`` so the capture post_cb
    can stamp frames with it (MATLAB routes frames by scan_id; a negative id = display-only).
    """

    def __init__(self, inner, box):
        self._inner = inner
        self._box = box

    def begin_scan(self):
        sid = self._inner.begin_scan()
        if sid is not None:
            self._box["id"] = sid
        return sid

    def check_pause_abort(self):
        return self._inner.check_pause_abort()


def make_idle(server, dummy_seq=None, run_real=None):
    """Build the :class:`IdleScheduler` for the empty-queue dummy keep-alive (live wiring).

    ``run_dummy`` runs the pre-compiled canonical DummySeq once; ``run_last`` replays a cached
    seq (the capture path is a follow-up, so a missing cache falls back to default inside the
    scheduler). Both fire the engine -> NEEDS-HARDWARE; only built inside :func:`serve`.
    """
    if run_real is None:
        from run_seq2 import run_real as run_real

    def run_dummy():
        if dummy_seq is not None:
            run_real(dummy_seq)

    def run_last(seq):
        run_real(seq)

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
    load_configs(log=lambda m: log("[runner] %s" % m))      # expConfig snapshot + engine config.yml
    from seq_config import SeqConfig
    seq_config = SeqConfig.get()                            # the real config activated above
    if server_factory is None:
        from ExptServer import ExptServer as server_factory
    server = server_factory(url)
    log("[runner] ExptServer bound at %s -- entering loop" % url)

    stop = {"flag": False}

    def _request_stop(signum=None, frame=None):  # noqa: ARG001
        stop["flag"] = True

    _install_signal_handlers(_request_stop)

    camera = None
    try:
        if with_camera:
            camera = _await_camera_init(server, log=log)
        idle = None
        if with_idle:
            try:
                dummy_seq = _compile_dummy()
                idle = make_idle(server, dummy_seq=dummy_seq)
                log("[runner] DummySeq compiled -- idle keep-alive enabled")
            except Exception as e:  # noqa: BLE001 - idle is optional; loop still serves jobs
                log("[runner] DummySeq compile failed (%s) -- idle disabled" % e)
        run = make_engine_run(server, camera, seq_config)   # engine + camera-arm + capture
        consume_loop(
            server,
            should_stop=lambda: stop["flag"],
            handle_camera=handle_camera_cmd if with_camera else None,
            camera=camera,
            idle=idle,
            run_kwargs={"run": run},
            log=lambda m: log("[runner] %s" % m),
        )
    finally:
        _teardown(server, camera, log=log)


def _await_camera_init(server, wait_s=CAMERA_INIT_WAIT_S, log=print):
    """Open the camera wrapper and pump the monitor's startup ``camera_init`` command.

    Mirrors the MATLAB 15 s camera-init wait: the monitor sends ``camera_init`` (ROI +
    exposure) shortly after we bind, so we pump :func:`handle_camera_cmd` over a short window
    to apply it before the loop proper. The handle is opened eagerly so it exists for
    teardown even if no init arrives. Returns the camera or ``None`` (wrapper unavailable)."""
    camera = open_camera(log=lambda m: log("[runner] %s" % m))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        handle_camera_cmd(server, camera)
        time.sleep(0.3)
        # No camera wrapper yet -> nothing to wait for; the pump above drains any pending
        # init so the camera pane reports a truthful state, then we proceed to serve.
        if camera is None:
            break
    return camera


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
    """Release the camera (DCAM handle) and stop the ZMQ worker -- the clean-terminate path.

    Camera first: the next backend's camera open fails if we still hold the handle. Then stop
    the worker thread before the socket is dropped (a live worker on a closed socket crashes).
    """
    if camera is not None:
        try:
            camera.close()
            log("[runner] camera closed")
        except Exception as e:  # noqa: BLE001
            log("[runner] camera close failed: %s" % e)
    try:
        server.stop_worker()
    except Exception as e:  # noqa: BLE001
        log("[runner] stop_worker failed: %s" % e)
    log("[runner] teardown complete")


def _safe_set_dummy_running(server, flag):
    fn = getattr(server, "set_dummy_running", None)
    if fn is not None:
        try:
            fn(flag)
        except Exception:  # noqa: BLE001
            pass


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
