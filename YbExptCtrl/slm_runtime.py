"""slm_runtime.py -- SLM loading-pattern resolution + the scan-long SLM session.

Resolves the img1 loading pattern + per-image loading-pattern declaration (ports of
``ybFirstLoadingPattern.m`` / ``ybLoadingPatternsJson.m``), constructs the scan-long
:class:`SlmScanSession`, and does the dequeue-time ``setup_rearrangement`` for a rearrangement
scan.

The operator-edited knobs live HERE (moved from runner.py 2026-07-22): change
:data:`DEFAULT_LOADING_PATTERN_PHASE` / :data:`ALL_SCANS_LOAD_PATTERN` /
:data:`DEFAULT_LOADING_DEFOCUS` in THIS module to change the every-scan loading-pattern
behaviour.

:func:`_runp_num` / :func:`_runp_get` (generic DynProps runp accessors) are parked here because
most of their users live in this module; engine_run imports them from here.

Split out of runner.py (now run_loop.py) 2026-07-22.
"""

import os

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
DEFAULT_LOADING_PATTERN_PHASE = "phase/33x33_feedback9.pt"  # 2026-06-30: was 33x33_uniform; switched to the active trap-depth-flattened array
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
    """Rounds of rearrangement for the scan context. The scan's EXPLICIT declaration
    (``g().rearrange_kwargs.extras.n_rounds`` -- every rearrangement scan sets it) wins;
    fallback is ``NumImages - 1`` (the pure-rearrangement layout, NumImages = n_rounds + 1).
    The explicit value must win because a hybrid science scan has extra post-rearrangement
    frames (e.g. RearrangeSTIRAPScan: NumImages=3 with a single round). >= 1."""
    try:
        v = scangroup.getseq(1)["rearrange_kwargs"]["extras"]["n_rounds"]
        return max(int(v), 1)
    except Exception:  # noqa: BLE001 - not declared -> frame-count fallback
        pass
    try:
        return max(int(_runp_num(scangroup.runp(), "NumImages", 2)) - 1, 1)
    except Exception:  # noqa: BLE001
        return 1


def _frame_patterns(scangroup, num_images, seq_config, log=None):
    """Per-camera-frame pattern NAMES for the multi-round rearrangement detector
    (RearrangeCommSeq2): ``[loading, middle, ..., final]``, one per frame -- so each frame is
    detected with its OWN per-pattern registry grid + thresholds (independent site counts).

    Single source of truth = the SAME per-frame declaration the runner writes for detection /
    thresholds (``_loading_patterns_json``: an explicit ``runp().imagePatternsJson`` when the scan
    set one -- which the two-round SLMRearrangementScan does with [loading, middle, final] -- else
    the synthesized initial/final). Reading the NAMES from there (not from the ScanGroup's
    ``rearrange_kwargs`` param node, whose scannable ``extras`` leaves don't take the ``(default)``
    fallback-call at the group level) guarantees detection matches the declared frames exactly.
    Returns None for a single-frame scan / no declaration -> the single-detector path is unchanged."""
    n = int(num_images)
    if n < 2:
        return None
    try:
        items = _loading_patterns_json(scangroup.runp(), n, *_loading_defaults(seq_config))
    except Exception as e:  # noqa: BLE001 - no declaration -> single-detector fallback
        if log is not None:
            log("[runner] frame-pattern derivation failed (%s); single-detector detection" % e)
        return None
    if not items:
        return None
    names = [str((it or {}).get("name") or "").strip() or None for it in items]
    # Size to the frame count: pad the tail with the last declared name (a scan may declare fewer
    # than num_images entries), truncate any surplus. All-None -> None (nothing declared).
    if not any(names):
        return None
    if len(names) < n:
        names = names + [names[-1]] * (n - len(names))
    return names[:n]


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
