"""engine_run.py -- the live per-scan run() seam handed to run_job.run_job.

Builds the per-scan ``run`` closure (:func:`make_engine_run`): engine reset, scan-prep, device
sessions (AWG via :mod:`awg_runtime`, SLM via :mod:`slm_runtime`, camera exposure via
:mod:`camera_runtime`, rearrange context), frame capture, seq auto-dump, line-trigger/TTL-manager
compile wrapping, and teardown ordering. Also :func:`load_configs` (the expConfig snapshot + the
engine config.yml).

Split out of runner.py (now run_loop.py) 2026-07-22. The setup ORDER inside ``run()`` is a
CONTRACT: the AWG upload happens BEFORE the scan-long SLM lock so the slm lease is fresh entering
the shot loop. QICK AWG wiring belongs in ``awg_runtime.py``, called from the AWG block here.
"""

import os

import run_timing
from camera_runtime import sync_camera_exposure
from slm_runtime import (DEFAULT_LOADING_DEFOCUS, _loading_defaults, _first_loading_pattern,
                         _loading_patterns_json, _make_slm_session, _is_rearrange_scan,
                         _n_rounds, _frame_patterns, _initial_setup_rearrangement,
                         _runp_num, _runp_get)
import awg_runtime

# pyctrl package root (…/pyctrl/YbExptCtrl/engine_run.py -> …/pyctrl) for locating config.yml,
# which now lives inside the submodule (a copy of matlab_new/config.yml) so pyctrl is self-contained.
PYCTRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    """Build the live ``run`` seam handed to :func:`run_job.run_job`.

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

    def run(seq, scangroup, control=None, scan_name=None, description=None,
            background=False, **opts):
        num_images = _num_images(scangroup)
        post = list(opts.pop("post_cb", []) or [])
        pre = list(opts.pop("pre_cb", []) or [])
        scan_id = _new_scan_id()        # 14-digit YYYYMMDDHHMMSS (the monitor/MATLAB convention)
        # Async image-save toggle (default ON): off via YB_ASYNC_FRAME_SAVE=0 or a runtime
        # <log>/ASYNC_FRAME_SAVE_OFF file (flip between scans, no restart -> A/B). Label the
        # per-shot timing rows with the scan + mode so an async-on vs -off A/B separates cleanly.
        async_save = _async_frame_save_enabled()
        try:
            run_timing.set_scan_label("%s %s async=%d"
                                      % (scan_name or "seq", scan_id, int(async_save)))
            run_timing.begin_setup_timing()   # open the bucket-B setup window (no-op when OFF)
        except Exception:  # noqa: BLE001
            pass
        # The pre-built run order (run_job._build_run_kwargs) IS ybBuildScanJob's
        # Scan.Params -- persist it so the monitor's scan curve can bucket each shot's result.
        params_order = opts.get("indices")

        # Scan-prep: write the Scan-config .mat the monitor's DataManager reads (best-effort;
        # without it the monitor errors "Cannot load <path>" and its _process_once dies).
        with run_timing.setup_stage("scan_prep"):   # incl. code-snapshot hashing (bucket B)
            _write_scan_prep(scan_id, scangroup, camera, num_images, log,
                             scan_name=scan_name, seq_config=seq_config, params=params_order,
                             seq=seq, description=description, background=background)
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
        awg_names = awg_runtime.awg_names(scangroup)
        if awg_names:
            with run_timing.setup_stage("awg_upload"):   # WVDT batch-upload (bucket B)
                awg_runtime.setup(awg_names, scangroup)
            pre.append(awg_runtime.make_pre_cb(scangroup))

        # --- QICK FPGA_AWG (RFSoC4x2 microwave): batch-upload all programs + arm per shot --------- #
        # A scan opts in via runp().QICK and declares the microwave sequence via g().QICK.* (like the
        # Siglent g().AWG.<name>.*). setup() batch-uploads one program per UNIQUE swept point ONCE and
        # sets external-trigger mode; the per-shot pre_cb ARMS this point's program (stop+start EVERY
        # shot -- the board is one-shot). cleanup() in the finally stops+disconnects. Done here (before
        # the SLM lock, same as Siglent) so the slm lease stays fresh; non-QICK scans pay nothing. The
        # program fires on TTLQickTrig = FPGA1/TTL14, pulsed by the step (RydbergPushoutStep).
        qick_on = awg_runtime.qick_enabled(scangroup)
        if qick_on:
            with run_timing.setup_stage("qick_upload"):  # QICK program batch-upload (bucket B)
                awg_runtime.qick_setup(scangroup)
            pre.append(awg_runtime.make_qick_pre_cb(scangroup))

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
        # Pre-run camera exposure sync: now the per-pattern overlay is active, push the resolved
        # Orca.ExposureTime to the live camera IFF it changed (the camera was inited once at
        # startup from the BASE exposure). No-op when unchanged -> no spurious re-arm. Best-effort.
        sync_camera_exposure(camera, seq_config, (pat0 or {}).get("name"),
                             log=lambda m: log("[runner] %s" % m))
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
                with run_timing.setup_stage("rearrange_setup"):  # model + pattern load (bucket B)
                    _initial_setup_rearrangement(slm_client, scangroup, scan_id, log,
                                                 server=server)
            with run_timing.setup_stage("slm_begin"):            # grab slm lock + write WGS phase
                slm_ses.begin()
            # SINGLE SOURCE OF TRUTH for the rearrange detection grid: pull the SERVER's actual
            # init_grid (the exact array setup_rearrangement derived, that rearrange(bits) scores
            # bits[i] against). The detector maps it through the global affine, so the lab detects
            # in the SAME site order the server scores -> bits[i] corresponds to init_grid[i] BY
            # CONSTRUCTION. No independently re-derived lab grid whose sort (col vs col_up) could
            # desync. Best-effort: on any failure the detector falls back to the registry grid.
            server_grid_knm = None
            if is_rearrange:
                try:
                    server_grid_knm = slm_client.get_rearrange_init_grid()
                    log("[runner] detection grid: server init_grid (%d sites, single source)"
                        % len(server_grid_knm) if server_grid_knm else
                        "[runner] server init_grid unavailable; detection uses registry grid")
                except Exception as e:  # noqa: BLE001
                    log("[runner] fetch server init_grid failed (%s); registry grid" % e)
            rearrange_runtime.set_context(rearrange_runtime.ScanContext(  # pat0 resolved above
                session=slm_ses, camera=camera, server=server, client=slm_client,
                scan_id=scan_id, is_rearrange=is_rearrange, n_rounds=_n_rounds(scangroup),
                pattern_name=(pat0 or {}).get("name"), server_grid_knm=server_grid_knm,
                frame_patterns=_frame_patterns(scangroup, num_images, seq_config,
                                               log=lambda m: log("[runner] %s" % m)),
                loading_defocus=_runp_num(scangroup.runp(), "loading_defocus",
                                          DEFAULT_LOADING_DEFOCUS),
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
                with run_timing.setup_stage("camera_arm"):       # flush stale frames + start_video
                    camera.flush()                               # drop stale frames (MATLAB flushdata)
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
        # --- Per-shot wall-clock stamping (TEMPORARY: RP-N correlation campaign) ----------- #
        # Appended UNCONDITIONALLY, so rearrangement scans are covered too: they take the
        # seq_owns_frames branch above (no default capture post_cb) but still get pre_cb/post_cb.
        # Appended LAST so the "pre" stamp sits closest to run_real (after the awg/slm pre_cbs)
        # and the "post" stamp lands after the capture cb has the frames. Rearrangement's
        # per-frame times come from rearrange_runtime.grab_one_frame -> shot_time.stamp_frame().
        # Wholly best-effort; disable with YB_SHOT_TIME=0.
        import shot_time
        shot_time_session = shot_time.begin(scan_id, log=log)
        if shot_time_session is not None:
            pre.append(shot_time.make_pre_cb(shot_time_session, seq_config))
            post.append(shot_time.make_post_cb(shot_time_session, seq_config))
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
        ttl_mgrs = _ttl_managers_config(scangroup, seq_config, log)
        if lt is not None or ttl_mgrs:
            def _compile_point(seqfn, seqparam, _lt=lt, _ttl_mgrs=ttl_mgrs):
                from exp_seq import ExpSeq
                s = ExpSeq(seqparam)
                seqfn(s)
                if _lt is not None and getattr(s, "trigger_device", "") == "":  # seq may self-enable
                    # Send the COMPLEMENT of the requested edge -- the firmware inverts it, so this
                    # is what makes LineTrigger Raise=True actually fire on the rising edge. See
                    # _MOLECUBE2_TRIG_EDGE_INVERTED for the evidence and how to retire this.
                    wire_raise = (not _lt["raise_"]) if _MOLECUBE2_TRIG_EDGE_INVERTED \
                        else _lt["raise_"]
                    s.enable_global_wait_trigger(_lt["device"], _lt["channel"],
                                                 wire_raise, _lt["timeout"])
                for mgr in _ttl_mgrs:                         # per-channel edge-timing managers
                    s.add_ttl_mgr(*mgr)
                s.generate()
                return s
            opts.setdefault("compile_point", _compile_point)

        # Engine reset (bucket B): runs at the top of run_scan_group's loop, not here -- wrap the
        # seam so its cost is logged alongside the other setup phases when RUN_TIMING is on.
        def _timed_new_run():
            with run_timing.setup_stage("new_run"):
                seq_manager.new_run()

        # Surface an intermittent NI DAC underflow onto the dashboard shot-health chip (same
        # channel as the rearrange-setup failures). The scan stops cleanly (status "ni_error",
        # shots-so-far kept) instead of a hard job crash -- this makes the loss visible.
        def _on_shot_error(message, point):
            log("[runner] %s" % message)
            try:
                server.record_shot_error(message, scan_id=scan_id, kind="ni_dac_underflow")
            except Exception:  # noqa: BLE001
                pass

        try:
            return run_scan_group(seq, scangroup, control=control,
                                  pre_cb=pre, post_cb=post,
                                  new_run=_timed_new_run,
                                  on_compile=seq_on_compile,
                                  on_globals=seq_on_globals,
                                  on_shot_error=_on_shot_error, **opts)
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
            if shot_time_session is not None:
                try:
                    shot_time.end(shot_time_session)             # close shot_time.csv
                except Exception:  # noqa: BLE001 - never fails the run
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
                awg_runtime.cleanup()
            if qick_on:
                awg_runtime.qick_cleanup()           # stop program + disconnect QICK board
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


# The FPGA firmware inverts the wait-trigger edge sense, so pyctrl sends the COMPLEMENT of the
# edge the user asked for (applied at the ``enable_global_wait_trigger`` call in _compile_point,
# NOT in _line_trigger_config -- the resolver keeps physical "which edge do I want" semantics).
#
# Why: molecube2 ``lib/pulser.h`` maps the boolean as ``trig_type = trig_raise ? 1 : 2``, but the
# gateware (molecube-amaranth ``inst_runner.py``) decodes ``trig_lower_edge = trig_type & 1`` and
# then waits for ``trig_ttl == trig_lower_edge`` (TRIG_INIT) followed by ``!=`` (TRIG_ARMED). So
# trig_type 1 ("raise") selects the FALLING edge and 2 selects the RISING edge -- backwards.
#
# Measured 2026-08-16 on scope 192.168.0.27 (CH1 = FPGA1/TTL27, CH2 = the Channel-0 line monitor),
# confirmed at two independent levels, each landing within one 20 us sample of the named edge:
#   pyctrl scan   Raise=1 -> falling   Raise=0 -> rising   (runp value verified in the descriptor)
#   test_trigger  edge=1  -> falling   edge=0  -> rising   (pulser level; no pyctrl, no libnacs)
#
# Retire this by fixing that ternary in molecube2 and rebuilding it on the FPGA; then set this to
# False. Leaving both "fixes" in place would double-invert and silently restore the old behavior.
_MOLECUBE2_TRIG_EDGE_INVERTED = True


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
    # NOTE on ``raise_``: it keeps PHYSICAL semantics here -- True means "I want the rising edge".
    # It is serialized faithfully (ZYNQZYNQ ver-2 trig_type 0x02 vs 0x01 -> bytecode WaitTrigger
    # raise bit), but the firmware decodes the flag BACKWARDS, so _compile_point sends its
    # complement; see _MOLECUBE2_TRIG_EDGE_INVERTED for the mechanism and the scope evidence.
    # Do NOT invert here as well -- that double-inverts and silently restores the bug.
    return {"device": str(_runp_get(rp, "LineTriggerDevice", cfg["Device"])),
            "channel": int(channel),
            "raise_": bool(_runp_get(rp, "LineTriggerRaise", cfg["Raise"])),
            "timeout": float(_runp_get(rp, "LineTriggerTimeout", cfg["Timeout"]))}


_TTL_MGR_FIELDS = ("on_delay", "off_delay", "skip_time", "min_time", "off_val")


def _ttl_managers_config(scangroup, seq_config, log=None):
    """Resolve the per-channel TTL managers for this scan: a list of
    ``(chn, off_delay, on_delay, skip_time, min_time, off_val)`` tuples ready for
    ``ExpSeq.add_ttl_mgr`` (note the arg ORDER: off_delay before on_delay), or ``[]`` to add none.

    Source of truth is expConfig ``consts["TTLManagers"]`` (``{chn: {on_delay, off_delay,
    skip_time, min_time, off_val}}``, times in seconds); a per-scan ``runp().TTLManagers`` dict, if
    present, is merged on top (per channel, per field). All-zero timing entries are dropped (adds
    nothing to the bytes anyway). Defensive: any error -> whatever resolved so far (never breaks a
    run)."""
    merged = {}
    try:
        consts = getattr(seq_config, "consts", None) or {}
        base = consts.get("TTLManagers") or {}
        for chn, params in base.items():
            merged[chn] = dict(params)
    except Exception:  # noqa: BLE001
        pass
    try:
        rp = scangroup.runp()
        ov = _runp_get(rp, "TTLManagers", None)
        if isinstance(ov, dict):
            for chn, params in ov.items():
                merged.setdefault(chn, {}).update(dict(params))
    except Exception:  # noqa: BLE001
        pass
    out = []
    for chn, params in merged.items():
        vals = {k: params.get(k, 0.0) for k in _TTL_MGR_FIELDS}
        # Skip a pure no-op (all timings zero) so it never touches the byte blob.
        if not (vals["on_delay"] or vals["off_delay"] or vals["skip_time"] or vals["min_time"]):
            continue
        out.append((chn, float(vals["off_delay"]), float(vals["on_delay"]),
                    float(vals["skip_time"]), float(vals["min_time"]), bool(vals["off_val"])))
        if log is not None:
            try:
                log("[runner] TTL manager on %s: on_delay=%gus off_delay=%gus "
                    "skip=%gus min=%gus" % (chn, vals["on_delay"] * 1e6, vals["off_delay"] * 1e6,
                                            vals["skip_time"] * 1e6, vals["min_time"] * 1e6))
            except Exception:  # noqa: BLE001
                pass
    return out


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
                     scan_name=None, seq_config=None, params=None, seq=None,
                     description=None, background=False):
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
        scan_meta = _scan_meta(scangroup, scan_name, seq_config, log, description=description,
                               background=background)
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


def _scan_meta(scangroup, scan_name, seq_config, log, description=None, background=False):
    """Build the DataManager scan-info fields (ScanGroup/ScanName/PlotScale/expConfig) from the
    dispatched ScanGroup. ``description`` (the descriptor's free-text run purpose) is stamped as a
    top-level ``description`` key; ``background`` stamps a top-level ``background`` flag so the
    saved scan is explicitly marked as a background/calibration run. Best-effort -> ``None``
    (frame-metadata-only config) on any failure."""
    try:
        from scan_summary import scangroup_scan_config
        consts = getattr(seq_config, "consts", None) if seq_config is not None else None
        return scangroup_scan_config(scangroup, scan_name=scan_name, expconfig=consts,
                                     description=description, background=background)
    except Exception as e:  # noqa: BLE001
        log("scan-meta build skipped: %s" % e)
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


def _noop_log(_msg):
    pass
