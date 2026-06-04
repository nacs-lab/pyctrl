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
# Throttle for the consume-loop live camera-status refresh (sensor temp / cooler / trigger).
# The loop iterates ~10 Hz when idle; reading DCAM attributes that often is wasteful, and the
# monitor's camera pane only polls every ~2 s, so refresh status at most this often.
CAMERA_STATUS_REFRESH_S = 2.0
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
            # Carry the queue summary onto the built JOB row so the dashboard's queue panel
            # shows axes/reps/scan_name while the scan RUNS (link_descriptor_to_job archives
            # the descriptor row; the job row takes over visibility). Best-effort.
            summary = _build_summary(payload)
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            job_id = server.submit_job(payload, summary=summary)
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

    A ``close`` releases the DCAM handle; a subsequent ``init`` (the GUI "Connect" after a
    "Disconnect", or the cross-backend handoff that closes the camera) must REOPEN it. The
    pyctrl ``init`` therefore reconnects + re-runs the full OrcaInit config when the handle is
    closed -- mirroring SequenceRunner.m, whose ``init`` does ``imaqreset`` + ``OrcaInit`` to
    recreate ``vid`` from scratch. Without this, Connect-after-Disconnect raised
    ``'NoneType' object has no attribute 'get_roi'`` (operating on the released handle) and the
    camera was stuck disconnected.

    The camera object (the pylablib wrapper) must expose:
        init(roi, exposure)           -> (roi, exposure)   apply ROI/exposure on an open handle
        init_orca(roi, exposure)      -> self              full OrcaInit reconfig (cooling/trigger)
        reconnect()                   -> None              reopen a released DCAM handle
        connected                     -> bool              handle currently open?
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
            if kind == "init" and not getattr(camera, "connected", True):
                # Connect after a close/handoff: reopen the released handle and re-apply the
                # full OrcaInit config (cooling/trigger/outputs), then the requested ROI/exp.
                camera.reconnect()
                camera.init_orca(roi=roi, exposure=exposure)
                actual_roi, actual_exp = camera.current_roi(), camera.get_exposure()
            else:
                fn = camera.init if kind == "init" else camera.apply_settings
                actual_roi, actual_exp = fn(roi, exposure)
            server.set_camera_result(True, actual_roi, "", actual_exp)
            push_camera_status(server, camera)   # immediately surface trigger/cooler/temp too
        except Exception as e:  # noqa: BLE001
            server.set_camera_result(False, _safe_current_roi(camera, roi), str(e))


def _safe_current_roi(camera, fallback):
    """``camera.current_roi()`` or ``fallback`` -- never raise (used on the error path, where
    the handle may be closed and ``current_roi`` would itself crash on the released DCAM)."""
    try:
        return camera.current_roi()
    except Exception:  # noqa: BLE001
        return fallback


def push_camera_status(server, camera):
    """Push the camera's full status (connected/roi/exposure/trigger/cooler/temperature) to
    the server so the monitor + web Camera card show a live, truthful state.

    Best-effort and contract-tolerant: a ``camera`` without :meth:`status` or a ``server``
    without :meth:`set_camera_status` (older fakes / the MATLAB hub) is a no-op, and a failed
    status probe is swallowed -- reporting status must never perturb the run loop."""
    if camera is None:
        return
    status_fn = getattr(camera, "status", None)
    setter = getattr(server, "set_camera_status", None)
    if status_fn is None or setter is None:
        return
    try:
        setter(status_fn())
    except Exception:  # noqa: BLE001 - status reporting is advisory; never raise into the loop
        pass


def make_camera_pump(refresh_s=CAMERA_STATUS_REFRESH_S, monotonic=time.monotonic):
    """Build the consume-loop camera pump: process one pending command, then refresh the live
    status on a throttle.

    The throttle (default :data:`CAMERA_STATUS_REFRESH_S`) keeps sensor temperature / cooler
    readouts current without reading DCAM attributes on every (~10 Hz) loop iteration. Returns
    a ``pump(server, camera)`` closure holding the last-refresh timestamp."""
    state = {"last": 0.0}

    def pump(server, camera):
        handle_camera_cmd(server, camera)
        now = monotonic()
        if now - state["last"] >= refresh_s:
            state["last"] = now
            push_camera_status(server, camera)

    return pump


def open_camera(seq_config=None, log=None):
    """Open + init the Orca from expConfig (the ``OrcaInit.m`` port), or ``None`` if unavailable.

    Reads ``consts.Orca.ROI`` / ``ExposureTime`` from ``seq_config`` and applies the full
    OrcaInit config (cooling, exposure, ROI, external rising-edge trigger, output triggers) via
    :func:`orca_camera.open_orca_from_config`. So the camera defaults to the imaging ROI even if
    the monitor's ``camera_init`` never arrives (e.g. a backend restart under a running monitor).
    The pylablib import is lazy; if it/the camera is absent we degrade gracefully (backend boots
    camera-less, the pane shows disconnected, scans needing frames fail loudly at run time).
    """
    log = log or _noop_log
    try:
        from orca_camera import open_orca_from_config
    except Exception as e:  # noqa: BLE001 - module/pylablib absent
        log("camera wrapper unavailable (%s) -- backend boots camera-less" % e)
        return None
    try:
        return open_orca_from_config(seq_config, log=log)
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


def make_engine_run(server, camera, seq_config, log=None):
    """Build the live ``run`` seam handed to :func:`sequence_runner.run_job`.

    Wraps ``run_scan_group`` with: the engine reset (``seq_manager.new_run``), per-scan camera
    arming (external rising-edge trigger; the seq's Imag399 step pulses ``TTLOrcaTrig``), and a
    per-shot capture ``post_cb`` (:func:`frame_capture.make_capture_post_cb`) that reads
    ``NumImages`` frames and publishes them via ``server.store_imgs`` / ``seq_finish``. The
    scan id (for frame routing + data-dir naming) is a fresh 14-digit ``YYYYMMDDHHMMSS`` stamp
    (:func:`_new_scan_id`, the monitor/MATLAB convention -- NOT ExptServer's epoch-ms); the seq
    id comes from ``seq_config.G.seq_id``.

    ``compile_point`` / ``run_real`` keep their engine defaults. ``config_teardown`` is NOT
    overridden (pyctrl ``SeqConfig.reset()`` would wipe the real config between shots).
    """
    import seq_manager
    from run_seq import run_scan_group
    log = log or _noop_log

    def run(seq, scangroup, control=None, scan_name=None, **opts):
        num_images = _num_images(scangroup)
        post = list(opts.pop("post_cb", []) or [])
        scan_id = _new_scan_id()        # 14-digit YYYYMMDDHHMMSS (the monitor/MATLAB convention)
        # The pre-built run order (sequence_runner._build_run_kwargs) IS ybBuildScanJob's
        # Scan.Params -- persist it so the monitor's scan curve can bucket each shot's result.
        params_order = opts.get("indices")

        # Scan-prep: write the Scan-config .mat the monitor's DataManager reads (best-effort;
        # without it the monitor errors "Cannot load <path>" and its _process_once dies).
        _write_scan_prep(scan_id, scangroup, camera, num_images, log,
                         scan_name=scan_name, seq_config=seq_config, params=params_order)

        armed = False
        if camera is not None and num_images > 0:
            try:
                camera.flush()                                   # drop stale frames (MATLAB flushdata)
                camera.start_video(external=True, nframes=max(num_images * 4, 16))
                armed = True
                from frame_capture import make_capture_post_cb
                post.append(make_capture_post_cb(
                    camera, server, num_images, scan_id, seq_config))
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


def _runp_num(runp, name, default=0):
    """Read a numeric runp flag (``runp.<name>(default)``), tolerant of absence."""
    try:
        return float(getattr(runp, name)(default))
    except Exception:  # noqa: BLE001
        return default


def _write_scan_prep(scan_id, scangroup, camera, num_images, log, *,
                     scan_name=None, seq_config=None, params=None):
    """Write the scan-config the monitor's DataManager reads (best-effort; never crash a job).

    frameSize = the camera ROI (W, H); the rest from the descriptor runp. ``params`` is the
    realized run order (ybBuildScanJob's ``Scan.Params``: shot -> scan-point index) -- persisted
    as ``config['Params']`` so the live scan curve can bucket each shot; when given, the written
    ``NumPerGroup`` is its length (the MATLAB ``Scan.NumPerGroup = length(Scan.Params)``).
    ``scan_meta`` adds the swept axes (``ScanGroup.base.vars``), the fixed/``g()``-override
    params, the scan title (``ScanName``), ``PlotScale`` and the baseline ``expConfig`` snapshot
    (``seq_config.consts``) so the dashboard's live scan-info panel + scan curve populate. A
    write failure is logged but does not fail the run (the monitor will warn until a config
    exists)."""
    try:
        from scan_prep import write_scan_config
        roi = camera.current_roi() if camera is not None else [0, 0, 0, 0]
        rp = scangroup.runp()
        scan_meta = _scan_meta(scangroup, scan_name, seq_config, log)
        num_per_group = len(params) if params is not None else int(_runp_num(rp, "NumPerGroup", 0))
        path = write_scan_config(
            scan_id, (roi[2], roi[3]), num_images,
            is_init=int(_runp_num(rp, "isInit", 0)),
            is_hc=int(_runp_num(rp, "isHC", 0)),
            is_grid2=int(_runp_num(rp, "isGrid2", 0)),
            num_per_group=num_per_group,
            params=params,
            scan_meta=scan_meta)
        log("scan config written: %s" % path)
    except Exception as e:  # noqa: BLE001
        log("scan-config write failed: %s" % e)


def _scan_meta(scangroup, scan_name, seq_config, log):
    """Build the DataManager scan-info fields (ScanGroup/ScanName/PlotScale/expConfig) from the
    dispatched ScanGroup. Best-effort -> ``None`` (frame-metadata-only config) on any failure."""
    try:
        from scan_summary import scangroup_scan_config
        consts = getattr(seq_config, "consts", None) if seq_config is not None else None
        return scangroup_scan_config(scangroup, scan_name=scan_name, expconfig=consts)
    except Exception as e:  # noqa: BLE001
        log("scan-meta build skipped: %s" % e)
        return None


def _build_summary(descriptor):
    """The ybScanSummary-shaped queue dict from a descriptor (JSON str/bytes/dict). Best-effort
    -> ``None`` (queue UI degrades) on any failure. Used to stamp the built job row."""
    try:
        from scan_summary import build_descriptor_summary
        return build_descriptor_summary(descriptor)
    except Exception:  # noqa: BLE001
        return None


def _new_scan_id():
    """A 14-digit ``YYYYMMDDHHMMSS`` scan id -- the monitor's ``scan_id_to_stamps`` / MATLAB
    convention (date 8 + time 6), used to route + name a scan's frames + data dir.

    NOT ``ExptServer.start_scan``'s ``time.time()*1000`` (a 13-digit epoch-ms), which the
    monitor's ``data_manager`` rejects with "scan_id must be 14 digits". One id per scan (per
    run() call), shared across that scan's shots; the per-shot ``seq_id`` comes from seq_config.
    """
    import datetime
    return int(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


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
        consume_loop(
            server,
            should_stop=lambda: stop["flag"],
            handle_camera=make_camera_pump() if with_camera else None,
            camera=camera,
            idle=idle,
            run_kwargs={"run": run},
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


def _await_camera_init(server, seq_config=None, wait_s=CAMERA_INIT_WAIT_S, log=print):
    """Open + init the camera from expConfig, then pump the monitor's startup ``camera_init``.

    The camera is opened with the expConfig defaults (OrcaInit port), so the ROI is correct
    even without a ``camera_init``. We still pump :func:`handle_camera_cmd` over a short window
    (mirrors the MATLAB 15 s wait) so a monitor-supplied ROI/exposure can override. Returns the
    camera or ``None`` (wrapper unavailable)."""
    camera = open_camera(seq_config, log=lambda m: log("[runner] %s" % m))
    # Report the just-opened+configured camera as CONNECTED right away (with its ROI /
    # exposure / trigger / cooler / temperature). Without this the monitor would show
    # "disconnected" until a camera_init arrives -- but in scenario 3 the camera is already
    # open and configured from expConfig, so the truthful state is connected.
    push_camera_status(server, camera)
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
        from nidaq_runner import NiDAQRunner
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
