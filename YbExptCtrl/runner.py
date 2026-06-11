"""runner.py -- the pyctrl scenario-3 run-loop HOST (port of ``SequenceRunner.m``).

This is the long-lived backend process the new monitor drives in **scenario 3** (new
monitor + pyctrl, MATLAB off). It hosts the shared :class:`ExptServer` ZMQ hub, drains the
queue, and runs each scan through the engine -- the Python counterpart of the MATLAB
``SequenceRunner(url)`` function (``matlab_new/YbExptCtrl/SequenceRunner.m``). It is launched
as ``python -m launcher.run_loop.runner <url>`` (see ``launcher/run_loop/runner.py``, which only
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

from seq_reload import reload_experiment_modules
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
# pyctrl package root (…/pyctrl/YbExptCtrl/runner.py -> …/pyctrl) for locating config.yml,
# which now lives inside the submodule (a copy of matlab_new/config.yml) so pyctrl is self-contained.
PYCTRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default loading defocus (ANSI z4) when a scan declares a loading pattern but does NOT set
# ``runp().loading_defocus``. The loading focal plane is a property of the science camera, not the
# pattern: until the camera is moved it is a fixed -5 (the plane the global SLM->camera affine is
# calibrated against). Production SLM scans already set this explicitly; this default keeps every
# other loading-pattern scan at the SAME plane so the single global affine stays valid. Change this
# (and re-bootstrap the affine) if/when the camera focus moves.
DEFAULT_LOADING_DEFOCUS = -5.0

# Every-scan default loading pattern + toggle. Operative defaults live HERE (not in
# expConfig consts, which are governed by the config drift oracle / THE ONE RULE and
# must stay byte-identical to the MATLAB reference) — same home as DEFAULT_LOADING_DEFOCUS
# above. When ALL_SCANS_LOAD_PATTERN is True, a scan that declares no pattern falls back
# to DEFAULT_LOADING_PATTERN_PHASE: it writes that WGS phase + holds the SLM lock for the
# whole scan and detects with the per-pattern threshold registry (so thresholds are never
# shared across patterns / with the day folder). OFF by default because it changes what
# EVERY scan writes to the SLM — flip to True and verify in a hardware window. Per-scan
# override (any scan, no toggle needed): runp().loading_phase / loading_defocus.
DEFAULT_LOADING_PATTERN_PHASE = "phase/33x33_uniform.pt"
ALL_SCANS_LOAD_PATTERN = False


def _loading_defaults(seq_config):
    """(default_phase, all_scans_on) for the no-pattern loading fallback — the module
    constants above. An OPTIONAL ``consts["SLM"]["Loading"]`` (DefaultPhase /
    AllScansLoadPattern) overrides them if some deployment chooses to add it (not set by
    default, to keep the config drift oracle green)."""
    phase, all_on = DEFAULT_LOADING_PATTERN_PHASE, ALL_SCANS_LOAD_PATTERN
    try:
        consts = getattr(seq_config, "consts", None) or {}
        ld = (consts.get("SLM", {}) or {}).get("Loading", {}) or {}
        if ld:
            phase = str(ld.get("DefaultPhase", phase) or phase)
            all_on = bool(ld.get("AllScansLoadPattern", all_on))
    except Exception:  # noqa: BLE001
        pass
    return phase, all_on


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
    consumer, so the payload IS the descriptor JSON -- no MATLAB byte stream), REUSING the
    descriptor's id for the job (``submit_job(job_id=desc_id)``) so the scan carries a single
    id -- the one ``submit_scan_descriptor`` returned and the .py scan script printed.
    ``link_descriptor_to_job`` then drops the now-redundant descriptor row instead of
    archiving a duplicate (its same-id branch). A bad descriptor is reported via
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
            # Carry the queue summary onto the JOB row so the dashboard's queue panel shows
            # axes/reps/scan_name while the scan RUNS. Best-effort.
            summary = _build_summary(payload)
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            # Reuse the descriptor's id for the job so the scan has a SINGLE id (the one the
            # .py script printed); link_descriptor_to_job then drops the descriptor row (its
            # same-id branch) instead of archiving a redundant second row.
            job_id = server.submit_job(payload, summary=summary, job_id=desc_id)
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
        from devices.orca import open_orca_from_config
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
    cfg = os.path.join(PYCTRL_ROOT, "config.yml")
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
        pre = list(opts.pop("pre_cb", []) or [])
        scan_id = _new_scan_id()        # 14-digit YYYYMMDDHHMMSS (the monitor/MATLAB convention)
        # Async image-save toggle (default ON): off via YB_ASYNC_FRAME_SAVE=0 or a runtime
        # <log>/ASYNC_FRAME_SAVE_OFF file (flip between scans, no restart -> A/B). Label the
        # per-shot timing rows with the scan + mode so an async-on vs -off A/B separates cleanly.
        async_save = _async_frame_save_enabled()
        try:
            import run_timing
            run_timing.set_scan_label("%s %s async=%d"
                                      % (scan_name or "seq", scan_id, int(async_save)))
        except Exception:  # noqa: BLE001
            pass
        # The pre-built run order (sequence_runner._build_run_kwargs) IS ybBuildScanJob's
        # Scan.Params -- persist it so the monitor's scan curve can bucket each shot's result.
        params_order = opts.get("indices")

        # Scan-prep: write the Scan-config .mat the monitor's DataManager reads (best-effort;
        # without it the monitor errors "Cannot load <path>" and its _process_once dies).
        _write_scan_prep(scan_id, scangroup, camera, num_images, log,
                         scan_name=scan_name, seq_config=seq_config, params=params_order,
                         seq=seq)
        # Stamp the data-folder id (scan_id) onto the running job so the queue/history shows it
        # (MATLAB fills this via set_job_file_id; pyctrl mints scan_id here, with no job_id in
        # scope, so set_running_job_file_id targets the single running job). Display + a
        # re-queue's key to find this run's code snapshot; best-effort, never fails the run.
        try:
            sid = str(int(scan_id))
            server.set_running_job_file_id("%s_%s" % (sid[:8], sid[8:]))
        except Exception:  # noqa: BLE001
            pass

        # --- Siglent AWG: batch-upload unique waveforms + per-shot active-waveform switch ----- #
        # A scan opts in via runp().AWGs (e.g. ["AWG556"]). setup() walks the ScanGroup and uploads
        # every UNIQUE Gaussian pulse once here; the per-shot pre_cb re-sends the active waveform
        # for this point's AWG.<name>.* scan values (~2 ms, skipped when unchanged). cleanup() in
        # the finally disconnects. Non-AWG scans (AWGs absent/empty) pay nothing.
        # Done BEFORE the scan-long SLM lock (below): the WVDT uploads can take seconds, and doing
        # them first means slm_ses.begin() grabs the lock LAST, with a fresh ~10 s lease entering
        # the per-shot loop (rather than burning the lease during the uploads).
        awg_names = _awg_names(scangroup)
        if awg_names:
            from devices.sigilent_awg import AWGManager
            AWGManager.setup(awg_names, scangroup)

            def _awg_pre_cb(_seq_num, arg0):
                pt = scangroup.getseq(arg0)
                AWGManager.recall_for_seq(pt.get("AWG", {}) if isinstance(pt, dict) else {})
            pre.append(_awg_pre_cb)

        # --- Scan-long SLM session ------------------------------------------------------------ #
        # Hold the slm HARDWARE lock + write the loading (WGS) phase for the WHOLE scan. This
        # applies to EVERY scan (default useScanLongSlmLock=1): any scan loads atoms into the SLM
        # pattern and assumes it stays put, so it must own the lock. begin() raises if the lock
        # can't be acquired within the block budget -> the run errors (run_job catches it).
        # Acquired AFTER the AWG batch-upload (above) so the lease is fresh entering the per-shot
        # loop. Rearrangement scans additionally do an initial setup_rearrangement at dequeue and
        # own their camera frames per shot (so the standard capture post_cb is skipped for them).
        import rearrange_runtime
        is_rearrange = _is_rearrange_scan(scangroup)
        _ld_phase, _ld_all = _loading_defaults(seq_config)   # expConfig SLM.Loading
        # Scan-default SLM pattern for the per-pattern config overlay (expConfig ByPattern):
        # every build in this scan resolves cooling/imaging/VSLMServo against this pattern; a
        # rearrange seq overrides it per bseq via set_pattern. No-op when ByPattern is empty.
        # Cleared in the finally below.
        pat0 = _first_loading_pattern(scangroup.runp(), default_phase=_ld_phase, all_scans=_ld_all)
        import expConfig_helper
        expConfig_helper.set_current_pattern((pat0 or {}).get("name"))
        slm_ses = _make_slm_session(scangroup, scan_id, log,
                                    default_phase=_ld_phase, all_scans=_ld_all)
        # Fresh per-shot health for this scan, so a failing previous scan can't
        # bleed its "shots failing" banner into a healthy new one (and vice
        # versa). Best-effort -- a missing method (older/MATLAB server) is fine.
        try:
            server.reset_shot_health(scan_id)
        except Exception:  # noqa: BLE001
            pass
        if slm_ses is not None:
            slm_client = slm_ses.c
            if is_rearrange:
                _initial_setup_rearrangement(slm_client, scangroup, scan_id, log,
                                             server=server)
            slm_ses.begin()                                      # grab slm lock + write WGS phase
            rearrange_runtime.set_context(rearrange_runtime.ScanContext(  # pat0 resolved above
                session=slm_ses, camera=camera, server=server, client=slm_client,
                scan_id=scan_id, is_rearrange=is_rearrange, n_rounds=_n_rounds(scangroup),
                pattern_name=(pat0 or {}).get("name"),
                log=lambda m: log("[runner] %s" % m)))
        # Capture ownership comes from the SEQ's own declaration (@seq_capabilities(owns_frames=
        # True)), NOT a runp sniff: the seq that does the mid-sequence grab is the source of truth.
        # Still gated on an active SLM session (the rearrange context the seq's callbacks need).
        from seq_capability import has_capability
        seq_owns_frames = has_capability(seq, "owns_frames") and slm_ses is not None

        # Non-rearrange scans renew the scan-long slm lease per shot too (rearrange scans renew via
        # RearrangeCommSeq.pre_run -> ensure_held). Without this the lease lapses ~lease_s into the
        # scan and the server releases slm mid-run. ensure_held is server-authoritative: it
        # heartbeats to confirm+renew and regrabs on loss (erroring the run if it truly can't).
        if slm_ses is not None and not is_rearrange:
            def _slm_pre_cb(_seq_num, _arg0, _ses=slm_ses):
                _ses.ensure_held()
            pre.append(_slm_pre_cb)

        armed = False
        if camera is not None and num_images > 0:
            try:
                camera.flush()                                   # drop stale frames (MATLAB flushdata)
                camera.start_video(external=True, nframes=max(num_images * 4, 16))
                armed = True
                if not seq_owns_frames:
                    # Normal scan: read frames after each shot, then hand them to the ExptServer
                    # persister. async_save=True publishes on the server's FIFO worker (the ~80 ms
                    # encode+store overlaps the next shot's hardware); the kill-switch runs inline.
                    # Rearrangement scans read + store frames mid-shot in their own callbacks.
                    from frame_capture import make_capture_post_cb
                    post.append(make_capture_post_cb(
                        camera, server, num_images, scan_id, seq_config, async_=async_save))
            except Exception:  # noqa: BLE001 - camera arm failure must not crash the job pre-run
                armed = False
        # --- Sequence auto-dump (SeqPlotter), gated by the dashboard toggle ---------------- #
        # When runtime_state's "save sequence dumps" flag is ON, write one flattened .seq per
        # UNIQUE compiled sequence into <scan_dir>/sequence/ + a manifest.json that the dashboard
        # Sequence tab reads. The dump evaluates get_nominal_output WITHOUT start() -> no FPGA
        # trigger / NI arm / camera frame. Wholly best-effort: never affects the run.
        seq_dump_session = _make_seq_dump_session(scan_id, scangroup, scan_name, log)
        seq_on_compile = seq_dump_session.on_compile if seq_dump_session is not None else None
        # --- Runtime-global capture (Q-F), UNGATED ------------------------------------- #
        # ALWAYS on, independent of the dump toggle: record each unique sequence's injected
        # runtime globals (e.g. the 616-EOM "from" frequency) into <scan_dir>/sequence/
        # globals.json so a never-dumped scan stays faithfully reconstructable offline.
        globals_session = _make_globals_session(scan_id, scan_name, log)
        seq_on_globals = globals_session.on_globals if globals_session is not None else None

        # 60 Hz line trigger (scan-wide): wrap the default compile leaf so each compiled ExpSeq
        # waits for the AC-line edge before generate(). Kept HERE (Yb layer), not in
        # lib/run_seq.py, so the framework stays experiment-agnostic / byte-faithful. None ->
        # disabled/unconfigured -> leave compile_point at its engine default (byte-identical).
        lt = _line_trigger_config(scangroup, seq_config, log)
        if lt is not None:
            def _compile_point(seqfn, seqparam, _lt=lt):
                from exp_seq import ExpSeq
                s = ExpSeq(seqparam)
                seqfn(s)
                if getattr(s, "trigger_device", "") == "":   # don't double-enable if the seq did
                    s.enable_global_wait_trigger(_lt["device"], _lt["channel"],
                                                 _lt["raise_"], _lt["timeout"])
                s.generate()
                return s
            opts.setdefault("compile_point", _compile_point)

        try:
            return run_scan_group(seq, scangroup, control=control,
                                  pre_cb=pre, post_cb=post,
                                  new_run=seq_manager.new_run,
                                  on_compile=seq_on_compile,
                                  on_globals=seq_on_globals, **opts)
        finally:
            # Flush any in-flight async image saves BEFORE teardown, so the last shots' frames are
            # published before we stop the camera / release locks. No-op for sync/legacy servers.
            try:
                drain = getattr(server, "drain_images", None)
                if drain is not None:
                    drain()
            except Exception:  # noqa: BLE001 - a drain failure must not break teardown
                pass
            if seq_dump_session is not None:
                try:
                    seq_dump_session.finalize()                  # write manifest.json
                except Exception:  # noqa: BLE001 - dump finalize never fails the run
                    pass
            if globals_session is not None:
                try:
                    globals_session.finalize()                   # write globals.json
                except Exception:  # noqa: BLE001 - globals finalize never fails the run
                    pass
            if armed:
                try:
                    camera.stop_video()
                except Exception:  # noqa: BLE001
                    pass
            if slm_ses is not None:
                try:
                    slm_ses.done()                               # release the scan-long slm lock
                except Exception:  # noqa: BLE001
                    pass
            if awg_names:
                try:
                    from devices.sigilent_awg import AWGManager
                    AWGManager.cleanup()                         # disconnect all AWGs
                except Exception:  # noqa: BLE001
                    pass
            rearrange_runtime.clear_context()
            try:
                import expConfig_helper
                expConfig_helper.set_current_pattern(None)       # drop the per-scan pattern overlay
            except Exception:  # noqa: BLE001
                pass

    return run


def _async_frame_save_enabled():
    """Whether the default capture publishes ASYNC (on the ExptServer worker). Default ON.

    Off when ``YB_ASYNC_FRAME_SAVE`` is a falsey env value OR the runtime toggle file
    ``<log>/ASYNC_FRAME_SAVE_OFF`` exists -- the latter lets you flip async off/on BETWEEN scans
    (no restart) for an A/B, beside the ``RUN_TIMING_ON`` toggle. Any probe failure -> ON."""
    try:
        if os.environ.get("YB_ASYNC_FRAME_SAVE", "1").strip().lower() in (
                "0", "false", "no", "off"):
            return False
        import run_timing
        return not os.path.exists(os.path.join(run_timing.log_dir(), "ASYNC_FRAME_SAVE_OFF"))
    except Exception:  # noqa: BLE001
        return True


def _num_images(scangroup):
    """NumImages for the scan (descriptor runp), default 1; bad/absent -> 0 (no capture)."""
    try:
        return int(scangroup.runp().NumImages(1))
    except Exception:  # noqa: BLE001
        return 0


def _make_seq_dump_session(scan_id, scangroup, scan_name, log):
    """Build a :class:`seq_dump.SeqDumpSession` iff the dashboard "save sequence
    dumps" toggle (runtime_state, offset 8) is ON; else ``None``.

    Best-effort: any failure (toggle off, missing module, bad scan_id) returns
    ``None`` so the auto-dump never affects a run.
    """
    try:
        import runtime_state
        if not runtime_state.get_save_sequence_dumps(False):
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        import os
        from seq_dump import SeqDumpSession, SEQ_SUBDIR
        from scan_prep import scan_dir
        sdir = os.path.join(scan_dir(scan_id), SEQ_SUBDIR)
        dt = None
        try:
            from datetime import datetime
            dt = datetime.strptime(str(int(scan_id)), "%Y%m%d%H%M%S")
        except Exception:  # noqa: BLE001
            dt = None
        sess = SeqDumpSession(sdir, scangroup, scan_id=str(int(scan_id)),
                              seq_name=scan_name or "seq", datetime_stamp=dt, log=log)
        log("[runner] sequence auto-dump ON -> %s" % sdir)
        return sess
    except Exception as exc:  # noqa: BLE001
        try:
            log("[runner] sequence auto-dump setup failed: %s" % exc)
        except Exception:  # noqa: BLE001
            pass
        return None


def _make_globals_session(scan_id, scan_name, log):
    """Build a :class:`seq_dump.GlobalsCaptureSession` (Q-F runtime-global capture).

    UNGATED -- created for EVERY scan, independent of the "save sequence dumps" toggle,
    so a never-dumped scan still records its injected runtime globals (for faithful
    offline reconstruction). Best-effort: any setup failure returns ``None`` so the
    capture never affects a run. ``finalize`` itself skips writing when no globals exist.
    """
    try:
        import os
        from seq_dump import GlobalsCaptureSession, SEQ_SUBDIR
        from scan_prep import scan_dir
        sdir = os.path.join(scan_dir(scan_id), SEQ_SUBDIR)
        return GlobalsCaptureSession(sdir, scan_id=str(int(scan_id)),
                                     seq_name=scan_name or "seq", log=log)
    except Exception as exc:  # noqa: BLE001
        try:
            log("[runner] runtime-global capture setup failed: %s" % exc)
        except Exception:  # noqa: BLE001
            pass
        return None


def _runp_num(runp, name, default=0):
    """Read a numeric runp flag (``runp.<name>(default)``), tolerant of absence."""
    try:
        return float(getattr(runp, name)(default))
    except Exception:  # noqa: BLE001
        return default


def _runp_get(runp, name, default):
    """Read a runp flag (``runp.<name>(default)``, DynProps fallback) WITHOUT coercion,
    tolerant of an absent runp/field. For non-numeric flags (bool/int channel/device str)."""
    if runp is None:
        return default
    try:
        return getattr(runp, name)(default)
    except Exception:  # noqa: BLE001
        return default


# Conservative 60 Hz line-trigger fallback, used ONLY if expConfig consts lacks a ``LineTrigger``
# subtree (older snapshot / a fake seq_config in tests): OFF, so an absent config never silently
# starts gating shots on a line edge. The operative default lives in expConfig consts
# (``consts["LineTrigger"]``, Enable=True); per-scan overrides come from ``runp().LineTrigger*``.
_LINE_TRIGGER_DEFAULTS = {"Enable": False, "Device": "FPGA1", "Channel": None,
                          "Raise": True, "Timeout": 0.02}


def _line_trigger_config(scangroup, seq_config, log=None):
    """Resolve the 60 Hz line-trigger config for this scan, or ``None`` to skip enabling.

    Source of truth is expConfig ``consts["LineTrigger"]`` (Enable/Device/Channel/Raise/Timeout);
    per-scan ``runp().LineTrigger*`` flags win. Returns ``{device, channel, raise_, timeout}``
    when enabled with a real channel, else ``None`` -- disabled, OR enabled-but-no-channel
    (``Channel`` unset), in which case we log once and skip rather than guess a TTL line that
    might be an output. Defensive: any error -> ``None`` (never breaks a run)."""
    cfg = dict(_LINE_TRIGGER_DEFAULTS)
    try:
        consts = getattr(seq_config, "consts", None) or {}
        lt = consts.get("LineTrigger") or {}
        for k in cfg:
            if k in lt:
                cfg[k] = lt[k]
    except Exception:  # noqa: BLE001
        pass
    try:
        rp = scangroup.runp()
    except Exception:  # noqa: BLE001
        rp = None
    if not bool(_runp_get(rp, "LineTriggerEnable", cfg["Enable"])):
        return None
    channel = _runp_get(rp, "LineTriggerChannel", cfg["Channel"])
    if channel is None:
        if log is not None:
            try:
                log("[runner] 60 Hz line trigger enabled but no input channel set "
                    "(consts['LineTrigger']['Channel'] / runp().LineTriggerChannel) -- skipping; "
                    "set it to your line-sync FPGA TTL input line to activate.")
            except Exception:  # noqa: BLE001
                pass
        return None
    return {"device": str(_runp_get(rp, "LineTriggerDevice", cfg["Device"])),
            "channel": int(channel),
            "raise_": bool(_runp_get(rp, "LineTriggerRaise", cfg["Raise"])),
            "timeout": float(_runp_get(rp, "LineTriggerTimeout", cfg["Timeout"]))}


def _awg_names(scangroup):
    """The AWGs this scan activates: ``runp().AWGs`` (e.g. ``["AWG556"]``), [] if unset.

    Mirrors MATLAB ``scanp.AWGs({})`` gating AWGManager.setup. A bare string is wrapped; a
    missing/empty field -> [] so non-AWG scans skip the AWG path entirely.
    """
    try:
        v = scangroup.runp().AWGs([])
    except Exception:  # noqa: BLE001
        return []
    if isinstance(v, str):
        return [v]
    try:
        return [str(x) for x in v if x]
    except TypeError:
        return []


# =========================================================================== #
# Scan-long SLM session helpers (rearrangement scan support)
# =========================================================================== #
def _is_rearrange_scan(scangroup):
    """True iff this scan drives per-shot rearrangement: it loads a rearrangement model
    (``runp().warmup_kwargs.model_filename``) or sets an explicit ``runp().isRearrange``. Used to
    do the dequeue-time setup_rearrangement and to let the seq own its camera frames (skip the
    standard capture post_cb). Defensive -> False on any error."""
    try:
        rp = scangroup.runp()
    except Exception:  # noqa: BLE001
        return False
    try:
        if rp.isfield("isRearrange"):
            return bool(rp.isRearrange(0))
    except Exception:  # noqa: BLE001
        pass
    try:
        wk = rp.warmup_kwargs
        if wk.isfield("model_filename"):
            return bool(str(wk.model_filename("")).strip())
    except Exception:  # noqa: BLE001
        pass
    return False


def _n_rounds(scangroup):
    """Rounds of rearrangement = NumImages - 1 (NumImages = n_rounds + 1); >= 1."""
    try:
        return max(int(_runp_num(scangroup.runp(), "NumImages", 2)) - 1, 1)
    except Exception:  # noqa: BLE001
        return 1


def _make_slm_session(scangroup, scan_id, log, default_phase=None, all_scans=False):
    """Construct (do NOT begin) the :class:`SlmScanSession` for this scan + declare its loading
    pattern. Returns None when ``runp().useScanLongSlmLock`` is disabled (default ON). The caller
    runs ``begin()`` after the optional initial setup_rearrangement, mirroring the user spec order
    (setup -> grab lock -> write WGS phase). ``default_phase``/``all_scans`` come from
    expConfig SLM.Loading (see _loading_defaults): when ``all_scans`` is on, a scan that declares
    no pattern falls back to ``default_phase`` (writes it + holds the lock)."""
    try:
        rp = scangroup.runp()
    except Exception:  # noqa: BLE001
        return None
    if not bool(_runp_num(rp, "useScanLongSlmLock", 1)):
        return None
    from devices.slm import get_client, SlmScanSession
    ses = SlmScanSession(get_client(), description="scan %s" % scan_id,
                         log=lambda m: log("[runner] %s" % m))
    pat = _first_loading_pattern(rp, default_phase=default_phase, all_scans=all_scans)
    if pat is not None:
        ses.set_loading_pattern(pat["name"], pat["phase_path"], pat["zernike"],
                                legacy_zerniked=pat["legacy"], baked_zernike=pat["baked"])
    return ses


def _first_loading_pattern(rp, default_phase=None, all_scans=False):
    """Resolve the img1 loading pattern + loading defocus (port of ybFirstLoadingPattern.m).

    Priority: an explicit ``runp().loading_phase`` (any scan), else a rearrangement scan's
    ``warmup_kwargs.initial_phase`` (+ ``extras.initial_phase_zernike`` baked). The generic
    ``runp().loading_defocus`` (ANSI z4, radians) is layered on top as ``[0 0 0 0 z4]`` (absolute;
    the server strips ``baked`` first). Returns a dict, or None when no pattern is declared (the
    session then holds the lock but writes nothing -- preserving whatever phase is on the SLM).

    When ``all_scans`` is on (expConfig SLM.Loading.AllScansLoadPattern) a scan that declares no
    pattern falls back to ``default_phase`` so EVERY scan writes a known loading hologram."""
    phase = ""
    baked = []
    try:
        phase = str(rp.loading_phase("")).strip()
    except Exception:  # noqa: BLE001
        phase = ""
    if not phase:
        try:
            wk = rp.warmup_kwargs
            phase = str(wk.initial_phase("")).strip()
            if phase and wk.extras.isfield("initial_phase_zernike"):
                baked = [float(x) for x in wk.extras.initial_phase_zernike([])]
        except Exception:  # noqa: BLE001
            phase = phase or ""
    if not phase and all_scans and default_phase:
        phase = str(default_phase).strip()      # every-scan default loading pattern
    if not phase:
        return None
    z4 = _runp_num(rp, "loading_defocus", DEFAULT_LOADING_DEFOCUS)
    zernike = [0.0, 0.0, 0.0, 0.0, float(z4)] if z4 else []
    name = os.path.splitext(os.path.basename(phase.replace("\\", "/")))[0]
    legacy = bool(baked) and any(b != 0 for b in baked)
    return {"name": name, "phase_path": phase.replace("\\", "/"),
            "zernike": zernike, "legacy": legacy, "baked": baked}


def _loading_patterns_json(rp, num_images, default_phase=None, all_scans=False):
    """Per-image loading-pattern declaration (port of ybLoadingPatternsJson.m). One entry per
    camera frame: frame-0 <- ``warmup_kwargs.initial_phase``, final frame <- ``final_phase``, with
    ``extras.*_phase_zernike`` as the baked Zernike to strip. An explicit ``runp().imagePatternsJson``
    wins; failing that, an explicit per-scan ``runp().loading_phase`` (the non-rearrange loading-
    hologram override, e.g. LACScan) becomes a single base-phase entry -- same priority
    ``_first_loading_pattern`` uses to WRITE it, so a scan that loads a pattern this way also
    DECLARES it for detection/thresholds. Each entry: ``{name, base_phase_path, order,
    legacy_zerniked, [baked_zernike]}``. Returns the list, or None when the scan declares no
    loading pattern (legacy day-folder behaviour).

    When ``all_scans`` is on (expConfig SLM.Loading.AllScansLoadPattern) a scan that declares no
    pattern falls back to a single ``default_phase`` entry, so imagePatternsJson is ALWAYS present
    and the monitor uses + updates the per-pattern threshold registry for every scan."""
    def _fallback():
        if all_scans and default_phase:
            return [_pattern_item(str(default_phase), None)]
        return None
    # (1) explicit override wins.
    try:
        explicit = str(rp.imagePatternsJson("")).strip()
    except Exception:  # noqa: BLE001
        explicit = ""
    if explicit:
        try:
            import json
            items = json.loads(explicit)
            if items:
                return items
        except Exception:  # noqa: BLE001
            pass
    # (2) explicit per-scan loading_phase (mirrors _first_loading_pattern's priority): a
    #     non-rearrange scan that overrides the loading hologram via runp().loading_phase (e.g.
    #     LACScan) must DECLARE it for detection/thresholds too, not just write it to the SLM.
    #     The loading defocus (runp().loading_defocus) is re-applied only on the SLM write -- trap
    #     extraction is defocus-independent -- so it is NOT part of this base-phase declaration.
    try:
        lp = str(rp.loading_phase("")).strip()
    except Exception:  # noqa: BLE001
        lp = ""
    if lp:
        return [_pattern_item(lp, None)]
    # (3) synthesise from rearrange warmup_kwargs, else the every-scan default.
    try:
        wk = rp.warmup_kwargs
        ip = str(wk.initial_phase("")).strip()
    except Exception:  # noqa: BLE001
        return _fallback()
    if not ip:
        return _fallback()
    try:
        fp = str(wk.final_phase("")).strip()
    except Exception:  # noqa: BLE001
        fp = ""
    items = [_pattern_item(ip, _baked_zern(wk, "initial_phase_zernike"))]
    if fp and int(num_images) >= 2:
        items.append(_pattern_item(fp, _baked_zern(wk, "final_phase_zernike")))
    return items


def _pattern_item(phase_path, baked):
    path = phase_path.replace("\\", "/")
    name = os.path.splitext(os.path.basename(path))[0]
    z = [float(x) for x in (baked or [])]
    legacy = any(c != 0.0 for c in z)
    it = {"name": name, "base_phase_path": path, "order": "col", "legacy_zerniked": legacy}
    if legacy:
        it["baked_zernike"] = z
    return it


def _baked_zern(wk, field):
    """The baked Zernike list under ``warmup_kwargs.extras.<field>`` if non-zero, else None."""
    try:
        if wk.extras.isfield(field):
            z = [float(x) for x in getattr(wk.extras, field)([])]
            return z if any(c != 0.0 for c in z) else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _initial_setup_rearrangement(client, scangroup, scan_id, log, server=None):
    """Dequeue-time setup_rearrangement: load the model + patterns from ``runp().warmup_kwargs``
    with ``reset_params=True`` (new run + factory-default the sticky cache). Per-shot setup calls
    (in the seq pre_run) then run WITHOUT reset_params so they stay sticky on top of this."""
    import rearrange_runtime
    try:
        rp = scangroup.runp()
    except Exception:  # noqa: BLE001
        return
    args = rearrange_runtime.collect_kwargs(rp.warmup_kwargs)
    args = rearrange_runtime.translate_zernike_zN(args)
    if not args:
        return
    args["reset_params"] = True
    args.setdefault("client_scan_id", str(scan_id))
    # Loading defocus (ANSI z4) -> WGS "loading_zernike": the SERVER adds it to BOTH the initial
    # and final WGS write phases (setup_rearrangement) so reload_rearrange (initial) and the
    # rearrange bookend (final) physically display WGS+defocus during loading -- correct
    # regardless of whether reload runs/no-ops. SEPARATE from the model zernike
    # (rearrange_kwargs.extras.z*/zernike_coeffs, which only touches model frames). Sent on the
    # DEQUEUE setup (when initial/final_phase are stored), never per-shot, so it can't double-stack.
    z4 = _runp_num(rp, "loading_defocus", DEFAULT_LOADING_DEFOCUS)
    if z4:
        extras = args.get("extras")
        if not isinstance(extras, dict):
            extras = {}
            args["extras"] = extras
        extras.setdefault("loading_zernike", [0.0, 0.0, 0.0, 0.0, float(z4)])
    try:
        client.setup_rearrangement(**args)
        log("[runner] initial setup_rearrangement (%d field(s), reset_params)" % len(args))
    except Exception as e:  # noqa: BLE001
        log("[runner] initial setup_rearrangement failed: %s" % e)
        if server is not None:
            try:
                server.record_shot_error(
                    "initial setup_rearrangement failed: %s" % e,
                    scan_id=scan_id, kind="setup_rearrangement")
            except Exception:  # noqa: BLE001
                pass


def _scan_descriptor(scangroup, seq, log):
    """Best-effort descriptor JSON (scangroup_to_descriptor) for self-contained offline
    reconstruction; ``None`` if the group can't be exported (e.g. multi-group)."""
    try:
        from scan_export import scangroup_to_descriptor
        return scangroup_to_descriptor(scangroup, seq)
    except Exception as e:  # noqa: BLE001
        try:
            log("scan descriptor export skipped: %s" % e)
        except Exception:  # noqa: BLE001
            pass
        return None


def _write_scan_prep(scan_id, scangroup, camera, num_images, log, *,
                     scan_name=None, seq_config=None, params=None, seq=None):
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
        # Per-image loading-pattern declaration (port of ybLoadingPatternsJson): drives the live
        # monitor's per-pattern grids/thresholds + the offline analysis's per-pattern calibration.
        # With SLM.Loading.AllScansLoadPattern on, scans that declare none fall back to the default.
        image_patterns = _loading_patterns_json(rp, num_images, *_loading_defaults(seq_config))
        descriptor = _scan_descriptor(scangroup, seq, log) if seq is not None else None
        path = write_scan_config(
            scan_id, (roi[2], roi[3]), num_images,
            is_init=int(_runp_num(rp, "isInit", 0)),
            is_hc=int(_runp_num(rp, "isHC", 0)),
            is_grid2=int(_runp_num(rp, "isGrid2", 0)),
            num_per_group=num_per_group,
            params=params,
            scan_meta=scan_meta,
            image_patterns=image_patterns,
            roi=list(roi),
            descriptor=descriptor)
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
