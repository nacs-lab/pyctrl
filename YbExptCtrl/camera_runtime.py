"""camera_runtime.py -- host-side camera glue over devices/orca.

Opens + inits the Orca at boot (:func:`open_camera`, :func:`_await_camera_init`), drives the
consume-loop command pump + status push (:func:`handle_camera_cmd`, :func:`make_camera_pump`,
:func:`push_camera_status`), and performs the per-scan pattern-resolved exposure sync
(:func:`sync_camera_exposure`, called by engine_run).

Split out of runner.py (now run_loop.py) 2026-07-22.
"""

import os
import time

# Seconds to wait for the camera-init command before compiling/serving (mirror the
# MATLAB 15 s camera-init wait); the monitor sends camera_init on startup.
CAMERA_INIT_WAIT_S = 15.0
# Throttle for the consume-loop live camera-status refresh (sensor temp / cooler / trigger).
# The loop iterates ~10 Hz when idle; reading DCAM attributes that often is wasteful, and the
# monitor's camera pane only polls every ~2 s, so refresh status at most this often.
CAMERA_STATUS_REFRESH_S = 2.0
# Startup camera-open resilience: retry the DCAM open a few times so a restart that briefly races
# the previous backend's handle release (a fail-fast "device busy") recovers instead of booting
# camera-less. A FREE-handle open is ~5 s; the launcher guarantees the handle is free before spawn,
# so these retries only cover a residual transient. Env-overridable for the field.
try:
    CAMERA_OPEN_ATTEMPTS = max(1, int(os.environ.get("YB_CAMERA_OPEN_ATTEMPTS", "3")))
except (TypeError, ValueError):
    CAMERA_OPEN_ATTEMPTS = 3
try:
    CAMERA_OPEN_RETRY_S = float(os.environ.get("YB_CAMERA_OPEN_RETRY_S", "2.0"))
except (TypeError, ValueError):
    CAMERA_OPEN_RETRY_S = 2.0


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


def open_camera(seq_config=None, log=None, attempts=CAMERA_OPEN_ATTEMPTS,
                retry_delay=CAMERA_OPEN_RETRY_S):
    """Open + init the Orca from expConfig (the ``OrcaInit.m`` port), or ``None`` if unavailable.

    Reads ``consts.Orca.ROI`` / ``ExposureTime`` from ``seq_config`` and applies the full
    OrcaInit config (cooling, exposure, ROI, external rising-edge trigger, output triggers) via
    :func:`orca_camera.open_orca_from_config`. So the camera defaults to the imaging ROI even if
    the monitor's ``camera_init`` never arrives (e.g. a backend restart under a running monitor).
    The pylablib import is lazy; if it/the camera is absent we degrade gracefully (backend boots
    camera-less, the pane shows disconnected, scans needing frames fail loudly at run time).

    The open is RETRIED (``attempts`` tries, ``retry_delay`` s apart): a restart can briefly race
    the previous backend's DCAM-handle release, and a contended open may fail-fast with a "device
    busy" error -- a short retry rides that out so a clean restart doesn't drop the camera. (A
    contended open that *hangs* instead of erroring is prevented upstream by the launcher waiting
    for the old backend to fully exit before spawning -- see ``pyctrl_launcher`` / ``port_utils``.)
    Whatever the failure, the backend still boots camera-less and serves, so the controller stays
    responsive and a later ``camera_init`` (the GUI "Connect") can reopen the handle.
    """
    log = log or _noop_log
    try:
        from devices.orca import open_orca_from_config
    except Exception as e:  # noqa: BLE001 - module/pylablib absent
        log("camera wrapper unavailable (%s) -- backend boots camera-less" % e)
        return None
    last = None
    for i in range(max(1, attempts)):
        try:
            return open_orca_from_config(seq_config, log=log)
        except Exception as e:  # noqa: BLE001
            last = e
            if i + 1 < attempts:
                log("camera open attempt %d/%d failed (%s) -- retrying in %.1fs"
                    % (i + 1, attempts, e, retry_delay))
                time.sleep(retry_delay)
    log("camera open failed after %d attempt(s): %s -- backend boots camera-less"
        % (attempts, last))
    return None


# Tolerance for "exposure unchanged". The Orca quantizes the requested exposure to its internal
# readout-time grid (~6.4 us/step: e.g. a 0.035 s request reads back as 0.0350064 s), so the
# slop MUST exceed one quantization step or every job would see a ~6 us mismatch and re-apply
# spuriously. 20 us comfortably covers the quantization while still catching any real change
# (the 35 vs 50 ms moves we care about are 15000 us).
_EXPOSURE_EPS_S = 20e-6


def sync_camera_exposure(camera, seq_config, pattern_name, log=None):
    """Pre-run hook: set the live camera exposure to the resolved (ByPattern-overlaid)
    ``Orca.ExposureTime``, but ONLY when it differs from what the camera currently reports.

    Mirrors the manual dashboard "Apply Settings" (``camera_apply_settings``): the camera is
    inited ONCE at startup from the BASE ``Orca.ExposureTime``, so a per-pattern overlay that
    raises/lowers the exposure (e.g. ``ByPattern["3270_tri"]["Orca"]["ExposureTime"]``) would
    otherwise be ignored by the hardware until a restart. Resolving the overlay here and applying
    iff changed makes the camera follow the active pattern automatically, with no manual step and
    no redundant re-arm when nothing moved. ROI is preserved (only exposure is touched).
    Best-effort: any failure logs and leaves the camera as-is (a run must not die over this)."""
    log = log or _noop_log
    if camera is None or not getattr(camera, "connected", True):
        return
    try:
        import expConfig_helper
        from devices.orca.orca_camera import orca_config_defaults
        # Resolve against the active pattern overlay (base Orca <- ByPattern[pattern]); fall back
        # to the base consts when no pattern / no per-pattern Orca override.
        consts = expConfig_helper.apply_pattern(seq_config.consts, pattern_name)
        target = (consts.get("Orca", {}) or {}).get("ExposureTime")
        if not target:
            target = orca_config_defaults(seq_config)[1]
        if not target:
            return
        target = float(target)
        current = camera.get_exposure()
        if current is not None and abs(float(current) - target) <= _EXPOSURE_EPS_S:
            return                                   # unchanged -> do NOT re-apply
        roi = camera.current_roi()
        actual_roi, actual_exp = camera.apply_settings(roi, target)
        log("Orca exposure synced: %.6gs -> %.6gs (pattern=%s, roi=%s)"
            % (float(current) if current is not None else float("nan"),
               float(actual_exp), pattern_name, actual_roi))
    except Exception as e:  # noqa: BLE001 - never fail a run over an exposure sync
        log("Orca exposure sync skipped (%s)" % e)


def _await_camera_init(server, seq_config=None, wait_s=CAMERA_INIT_WAIT_S, log=print):
    """Open + init the camera from expConfig, then pump the monitor's startup ``camera_init``.

    The camera is opened with the expConfig defaults (OrcaInit port), so the ROI is correct
    even without a ``camera_init``. We still pump :func:`handle_camera_cmd` over a short window
    (mirrors the MATLAB 15 s wait) so a monitor-supplied ROI/exposure can override. Returns the
    camera or ``None`` (wrapper unavailable)."""
    camera = open_camera(seq_config, log=lambda m: log("[runner] %s" % m))
    # Report the just-opened+configured camera as CONNECTED right away (with its ROI /
    # exposure / trigger / cooler / temperature). Without this the monitor would show
    # "disconnected" until a camera_init arrives -- but the camera is already open (pyctrl opens
    # it at startup) and configured from expConfig, so the truthful state is connected.
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


def _noop_log(_msg):
    pass
