"""rearrange_callbacks.py -- shared per-shot rearrangement callbacks (YbSeqs layer).

Extracted from ``RearrangeCommSeq.py`` / ``RearrangeCommSeq2.py`` so any seq -- including a
hybrid "rearrange, then science" seq (RearrangeSTIRAPSeq) -- composes the same per-shot
machinery instead of copy-pasting it:

    pre_run          -- per-shot camera-buffer resync, scan-long slm lock (ensure_held),
                        per-shot compute (GPU) lock, sticky setup_rearrangement (with the
                        zN / distortion_zN / step_xyz folds), reload_rearrange.
    rearrange_round  -- grab camera frame N, detect per-site probs, rearrange(probs), stage
                        the frame (one rearrangement round).
    verify_frame     -- grab camera frame N, detect bits, update_rearrange(bits), stage the
                        frame. The "check the move" frame of a hybrid seq: reports the
                        rearranged occupancy to the server WITHOUT another rearrange call.
    finalize         -- grab the FINAL frame, publish the aligned N-frame shot with a single
                        finish_shot (or re-publish a failing shot's captured frames under the
                        display-only sentinel); release the compute lock; keepalive the
                        scan-long slm lock.

The seq modules keep their public callback names (``pre_run`` / ``hand_over_slm`` /
``hand_over_slm_2`` / ``post_run``) as thin wrappers over these, so the engine registration
(``reg_before_start`` / ``reg_before_bseq`` / ``reg_after_end``) and the existing tests are
unchanged.

Frame-alignment invariant (unchanged from the originals, load-bearing): a shot either stages
ALL its frames and publishes them with a single ``finish_shot``, or it is cancelled and NO
partial set is persisted. Because the hardware runs every Imag399 step once a shot starts,
every callback that runs must CONSUME its frame even on a failing shot (grab-and-drain) so a
straggler can't shift the frame stream by one on the NEXT shot. A failed shot's captured
frames are re-published under the FAILING sentinel for LIVE DISPLAY ONLY (never persisted).

All handles (camera / ExptServer / SlmScanSession / SlmClient / detectors) come from
:mod:`rearrange_runtime`'s process-global :class:`ScanContext`, set by the runner at scan
start. Lives in YbSeqs (NOT YbExptCtrl) so it hot-reloads per job -- edits need no backend
restart. Nothing here is serialized: no byte-path impact.
"""

import os
import time

import rearrange_runtime

# frame_idx -> captured frame for the CURRENT shot (the failing-shot DISPLAY re-publish; the
# finalizing callback is a separate scope from the round callbacks that grabbed the frames).
# Shots run strictly sequentially (one shot's callbacks at a time), so one slot per frame
# suffices; reset per shot (pre_run + round 0) and cleared in finalize's finally. Display-only
# use; never serialized.
_STASH = {}


def reset_stash():
    """Drop the current shot's stashed frames (called at shot start + finalize teardown)."""
    _STASH.clear()


def stashed(n_frames):
    """Stashed frames ``0..n_frames-1`` in frame order (``None`` where not captured)."""
    return [_STASH.get(i) for i in range(n_frames)]


# =========================================================================== #
# pre_run -- locks + sticky per-shot setup (reg_before_start)
# =========================================================================== #
def pre_run(s1, *, flags=("rearrange_img1_ok",), lock_desc="rearrange compute",
            force_n_rounds=None, frames_label=None):
    """Reset the shot flags, resync the camera buffer, ensure the scan-long slm lock, grab the
    per-shot compute lock, push the sticky per-shot setup_rearrangement, and reload_rearrange.

    Args:
        flags: the per-round/verify ``s1.G`` ok-flags this seq uses, reset False here (the
            ``rearrange_lock_ok`` flag is always reset too). One flag per pre-final frame.
        lock_desc: the compute-lock description shown in the SLM server's lock table.
        force_n_rounds: when set, ``extras.n_rounds`` is defaulted to this value on the
            per-shot setup call (RearrangeCommSeq2 forces 2; the server allocates per-round
            buffers/phases from it). ``None`` leaves it entirely to the scan's declaration.
        frames_label: the "img1/img2" style label for the stale-buffer error message; derived
            from ``len(flags) + 1`` frames when omitted.
    """
    for f in flags:
        setattr(s1.G, f, False)
    s1.G.rearrange_lock_ok = False
    reset_stash()

    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return                          # no rearrangement context -> nothing to do

    if frames_label is None:
        frames_label = "/".join("img%d" % (i + 1) for i in range(len(flags) + 1))

    # Per-shot frame-buffer resync. The Orca free-runs into a circular buffer armed ONCE per
    # scan (runner.py: flush + start_video); within a scan the frames are grabbed one at a
    # time, so a single straggler -- a frame whose readout landed after its grab timed out, a
    # cancelled pair's leftover, or a spurious trigger -- offsets the stream by one and the
    # frames come out SHIFTED for the rest of the scan. pre_run runs BEFORE this shot's
    # Imag399 #1 triggers, so anything buffered now is stale: drop it and surface a nonzero
    # count (the previous shot(s) were misaligned). The per-shot analog of the scan-start flush.
    cam = ctx.camera
    if cam is not None:
        try:
            stale = int(cam.frames_available())
        except Exception:  # noqa: BLE001 - not acquiring / older driver -> assume clean
            stale = 0
        if stale:
            try:
                cam.flush()
            except Exception:  # noqa: BLE001 - best-effort drain
                pass
            ctx.record_error(
                "[pre_run] seq %d: %d stale frame(s) in the camera buffer at shot start -- "
                "flushed; %s on the prior shot(s) were likely misaligned"
                % (_seq_id(s1), stale, frames_label), kind="frame_desync", seq_id=_seq_id(s1))

    # The scan-long slm lock is mandatory: ensure_held re-acquires + rewrites the LOADING phase
    # if the lease lapsed (a shot longer than the lease). A failure raises -> the run errors.
    ctx.session.ensure_held()

    c = ctx.client
    # Per-shot compute (GPU) lock, blocking ~1 s. On miss, scrap THIS shot and retry rather
    # than racing rearrange() against another client.
    try:
        c.acquire_lock("compute", lock_desc, timeout_s=10, block_timeout=1)
        s1.G.rearrange_lock_ok = True
    except Exception as err:  # noqa: BLE001 - contention / timeout -> cancel + retry
        ctx.log("[pre_run] compute lock acquire failed: %s -- cancelling seq for retry" % err)
        _safe(ctx.server, "seq_cancel")
        return

    try:
        c.health()                      # prewarm the connection
    except Exception:  # noqa: BLE001
        pass

    # Per-shot setup_rearrangement from rearrange_kwargs (the SCANNED params); NO reset_params
    # so everything else stays sticky from the initial (dequeue-time) setup call. The scalar
    # folds (extras.z<N> -> zernike_coeffs, extras.distortion_z<N> -> distortion_zernike,
    # extras.step_x/y/z -> step_size) let a scan sweep SCALAR axes for the server's
    # list-valued knobs (a list-valued swept axis breaks the lab-side scan grid/curve); each
    # is a no-op when its keys are absent, and an explicit list already present always wins.
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs)
    args = rearrange_runtime.translate_zernike_zN(args)
    args = _fold_distortion_zernike(args)
    args = _fold_step_xyz(args)
    if force_n_rounds is not None:
        _ensure_extra(args, "n_rounds", int(force_n_rounds))
    args.setdefault("client_scan_id", str(ctx.scan_id))
    try:
        c.setup_rearrangement(**args)
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[pre_run] setup_rearrangement failed: %s" % err,
                         kind="setup_rearrangement", seq_id=_seq_id(s1))
    try:
        c.reload_rearrange()
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[pre_run] reload_rearrange failed: %s" % err,
                         kind="reload_rearrange", seq_id=_seq_id(s1))


# =========================================================================== #
# rearrange_round -- one rearrangement handoff (reg_before_bseq)
# =========================================================================== #
def rearrange_round(s1, frame_idx, *, ok_flag, tag, prior_flag=None,
                    use_frame_pattern=False, record_ok=False,
                    grab_timeout=0.1, drain_timeout=0.2):
    """One rearrangement round on camera frame ``frame_idx`` (0-based): grab the frame, detect
    per-site probabilities, ``rearrange(probs)``, and stage the frame (async persist).

    Args:
        ok_flag: the ``s1.G`` flag set True when this round's rearrange succeeded.
        tag: the log/error tag (the wrapper's public callback name, e.g. "hand_over_slm").
        prior_flag: gate on an earlier round's ok-flag. When that flag is False the shot is
            already failing -- this frame is still DRAINED (never left buffered) + stashed for
            the display re-publish, then the shot is cancelled without a rearrange call.
        use_frame_pattern: detect with this frame's OWN per-pattern registry detector
            (``ctx.detect_probs_for`` + the scan's frame_patterns -- the multi-pattern scan).
            False -> the frame-0 detector (``ctx.detect_probs``, server-grid-anchored -- the
            single-round path).
        record_ok: call ``ctx.record_ok()`` on a healthy rearrange (the single-round seq
            clears the "shots failing" banner here; the multi-round seqs clear it in
            :func:`finalize` once the whole shot published).
    """
    if frame_idx == 0:
        reset_stash()                   # first frame of the shot -> fresh stash
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    # A prior round failed: still consume this frame (it is physically produced), keep it for
    # the failing-shot display re-publish, and cancel so no partial set persists.
    if prior_flag is not None and not getattr(s1.G, prior_flag)(False):
        img, ok, _n = rearrange_runtime.grab_one_frame(ctx.camera, timeout=drain_timeout)
        if ok:
            _STASH[frame_idx] = img
        _safe(ctx.server, "cancel_shot")
        return

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)
    if not ok:
        _report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH[frame_idx] = img

    probs = (ctx.detect_probs_for(_frame_pattern(ctx, frame_idx), img)
             if use_frame_pattern else ctx.detect_probs(img))
    if not probs:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    _do_rearrange_round(ctx, s1, img, probs, tag, ok_flag, record_ok=record_ok)


# =========================================================================== #
# verify_frame -- post-rearrangement check frame, no rearrange call (hybrid seqs)
# =========================================================================== #
def verify_frame(s1, frame_idx, *, ok_flag, prior_flags, tag="verify_frame",
                 use_frame_pattern=True, grab_timeout=0.1, drain_timeout=0.2):
    """Grab + stage the post-rearrangement VERIFY frame and report its occupancy to the SLM
    server via ``update_rearrange(bits)`` -- WITHOUT another rearrange call. The hybrid
    "rearrange, then science" seq's middle frame: survival is later normalized against this
    verified occupancy, and the server's rearrange statistics get their result frame even
    though the shot's FINAL frame is post-science (and must not feed them).

    Sets ``ok_flag`` once the frame is grabbed + staged; an ``update_rearrange`` failure is
    logged but not fatal (matches the pure-rearrangement finalize). Prior-round failure ->
    grab-and-drain + cancel, same as :func:`rearrange_round`."""
    ctx = rearrange_runtime.context()
    if ctx is None:
        return
    if ctx.client is None or not s1.G.rearrange_lock_ok(False):
        return

    if not all(getattr(s1.G, f)(False) for f in prior_flags):
        img, ok, _n = rearrange_runtime.grab_one_frame(ctx.camera, timeout=drain_timeout)
        if ok:
            _STASH[frame_idx] = img
        _safe(ctx.server, "cancel_shot")
        return

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)
    if not ok:
        _report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH[frame_idx] = img

    bits = (ctx.detect_bits_for(_frame_pattern(ctx, frame_idx), img)
            if use_frame_pattern else ctx.detect_bits(img))
    if bits and ctx.client is not None:
        try:
            ctx.client.update_rearrange(bits, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
        except Exception as err:  # noqa: BLE001
            ctx.log("[%s] update_rearrange failed: %s" % (tag, err))
    _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    setattr(s1.G, ok_flag, True)


# =========================================================================== #
# finalize -- final frame + shot publication + teardown (reg_after_end)
# =========================================================================== #
def finalize(s1, *, round_flags, final_frame_idx, tag="post_run",
             update=True, use_frame_pattern=False, record_ok=False, grab_timeout=0.1):
    """Grab the FINAL frame (frame ``final_frame_idx``, 0-based) and publish the shot; release
    the compute lock and keepalive the scan-long slm lock.

    The final frame is PHYSICALLY produced regardless of prior success, so it is ALWAYS
    grabbed (draining it). The shot publishes as one aligned set (stage + single finish_shot)
    only when EVERY flag in ``round_flags`` is True; otherwise the captured frames are
    re-published for DISPLAY ONLY under the failing sentinel (never persisted).

    Args:
        round_flags: the ``s1.G`` ok-flags that must all be True for a persisted shot.
        update: run ``update_rearrange`` on the final frame's detected bits. True for the
            pure-rearrangement seqs, whose final frame IS the rearranged array. MUST be False
            for a hybrid science seq -- its final frame is post-science (e.g. post-pushout),
            so feeding it to the server would poison the rearrange statistics; the hybrid
            reports via :func:`verify_frame` instead.
        use_frame_pattern: detect the final frame with its own per-pattern registry detector
            (``detect_bits_for``) instead of the frame-0 detector (``detect_bits``).
        record_ok: call ``ctx.record_ok()`` on a fully-published shot (clears the dashboard
            "shots failing" banner). The single-round seq records at its rearrange instead.
    """
    ctx = rearrange_runtime.context()
    if ctx is None:
        return
    try:
        all_ok = all(getattr(s1.G, f)(False) for f in round_flags)
        img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)

        if not all_ok:
            # A failing shot: the frames must NOT persist, but the live view should still
            # flash whatever was captured. The shot-health chip is already red via the
            # earlier record_error.
            ctx.publish_failed_shot(stashed(final_frame_idx) + [img if ok else None],
                                    _seq_id(s1))
            return

        if not ok:
            # Every round succeeded but the final frame was lost -> show the captured frames
            # + "no data" (display only) rather than freezing the live view; never persist a
            # partial set. Surplus (desync) vs readout-latency timeout surfaced distinctly;
            # either way the next shot's pre_run flush resyncs the buffer.
            label = "img%d" % (final_frame_idx + 1)
            prior = "/".join("img%d" % (i + 1) for i in range(final_frame_idx))
            if n_seen >= 2:
                ctx.record_error(
                    "[%s] seq %d: %s grab saw %d frames (desynced buffer) -- drained; "
                    "display-only %s" % (tag, _seq_id(s1), label, n_seen, prior),
                    kind="frame_desync", seq_id=_seq_id(s1))
            else:
                ctx.record_error(
                    "[%s] seq %d: %s unavailable (timeout, 0 frames) -- display-only %s"
                    % (tag, _seq_id(s1), label, prior),
                    kind="frame_timeout", seq_id=_seq_id(s1))
            ctx.publish_failed_shot(stashed(final_frame_idx), _seq_id(s1))
            return

        # Success: optionally report the final occupancy to the server, then stage the final
        # frame and publish the whole aligned set with ONE finish_shot (FIFO ordering keeps
        # the frames aligned behind the rounds' staged frames).
        if update and ctx.client is not None:
            bits = (ctx.detect_bits_for(_frame_pattern(ctx, final_frame_idx), img)
                    if use_frame_pattern else ctx.detect_bits(img))
            if bits:
                try:
                    ctx.client.update_rearrange(
                        bits, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
                except Exception as err:  # noqa: BLE001
                    ctx.log("[%s] update_rearrange failed: %s" % (tag, err))
        _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
        _safe(ctx.server, "finish_shot")
        if record_ok:
            ctx.record_ok()             # healthy shot -> clears the "failing" banner promptly
    finally:
        # Release the per-shot compute lock; keepalive (renew) the scan-long slm lock. The
        # scan-long session owns slm and releases it at scan end -- never release slm here.
        if s1.G.rearrange_lock_ok(False) and ctx.client is not None:
            try:
                ctx.client.release_lock("compute")
            except Exception as err:  # noqa: BLE001
                ctx.log("[%s] compute lock release failed: %s" % (tag, err))
        if ctx.session is not None:
            ctx.session.keepalive()
        reset_stash()


# =========================================================================== #
# shared round logic
# =========================================================================== #
def _do_rearrange_round(ctx, s1, img, probs, tag, ok_flag, record_ok=False):
    """Stage ``img`` (async) and call rearrange(probs) for one round; set ``s1.G.<ok_flag>`` on
    success. Includes the abort-tail cleanup (drop the staged frame + the phantom server ledger
    row on a downstream failure) so a failed round never leaves a half-staged shot or a
    desynced ledger."""
    c = ctx.client
    runid = _runid_kwargs(ctx.scan_id, _seq_id(s1))
    # Hand the frame to the ExptServer persister (async, FIFO worker) so the encode+store
    # overlaps rearrange()'s SLM round-trip + the next bseq's hardware instead of blocking the
    # held-atom critical path. A persist failure is handled by the worker (it cancels the
    # shot); the live signal we gate on is the rearrange() result.
    _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    try:
        # Per-site presence PROBABILITIES (floats in [0,1]) in place of the hard bitstring.
        # The SLM server rounds them to 0/1 for now but the floats let it drop low-confidence
        # sites in future.
        r = c.rearrange(probs, **runid)
        if isinstance(r, dict) and r.get("handoff_idle"):
            ctx.log("[%s] seq %d: server idle; cancelling shot, waiting 1 s"
                    % (tag, _seq_id(s1)))
            _safe(ctx.server, "cancel_shot")   # ordered after the staged frame -> drops it
            setattr(s1.G, ok_flag, False)
            time.sleep(1.0)
            return
        setattr(s1.G, ok_flag, True)
        if isinstance(r, dict) and not r.get("ok", True):
            ctx.record_error("[%s] rearrange returned ok=false" % tag,
                             kind="rearrange", seq_id=_seq_id(s1))
        elif record_ok:
            ctx.record_ok()             # healthy shot -> clears the "failing" banner on recovery
    except Exception as err:  # noqa: BLE001
        # rearrange() may have committed server-side before a downstream failure; drop the
        # phantom diag ledger row so the SLM ledger stays aligned with the lab seq_ids.
        try:
            if runid:
                c.cancel_last_shot(**runid)
        except Exception:  # noqa: BLE001 - older server lacks /slm/cancel_last; non-fatal
            pass
        _safe(ctx.server, "cancel_shot")   # drop the staged frame (ordered after it)
        setattr(s1.G, ok_flag, False)
        ctx.record_error("[%s] rearrange call failed: %s" % (tag, err),
                         kind="rearrange", seq_id=_seq_id(s1))


# =========================================================================== #
# SLM-kwarg scalar folds (per-shot setup helpers)
# =========================================================================== #
def _fold_distortion_zernike(args):
    """Fold scalar ``extras.distortion_z<N>`` (ANSI index N, radians) into the
    ``extras.distortion_zernike`` list the SLM-server ``pingponggrating`` dispatcher reads.

    Mirrors ``rearrange_runtime.translate_zernike_zN`` but for the pingponggrating distortion
    knob, and lives HERE (a YbSeqs module, hot-reloaded per job) rather than in YbExptCtrl so
    it needs no backend restart. A scan sweeps a SCALAR axis (e.g.
    ``extras.distortion_z7.scan(...)``) instead of a list-valued ``distortion_zernike`` axis: a
    list-valued swept axis breaks the lab-side live scan curve
    (``scan_analysis._find_first_numeric`` ravels the (npts, coeff_len) list-of-lists ->
    "size 25 vs 5" concat error). We build the full list here, per shot. An explicit
    ``distortion_zernike`` already present wins; ``distortion_zernike`` itself is NOT
    consumed (suffix ``ernike`` is not all-digits)."""
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    prefix = "distortion_z"
    zn = {}
    for key in list(extras.keys()):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            zn[int(key[len(prefix):])] = float(extras.pop(key))
    if not zn:
        return args
    coeffs = [0.0] * (max(zn) + 1)
    for idx, val in zn.items():
        coeffs[idx] = val
    extras.setdefault("distortion_zernike", coeffs)
    return args


def _fold_step_xyz(args):
    """Fold scalar ``extras.step_x`` / ``step_y`` / ``step_z`` into the 3-vector
    ``extras.step_size`` = ``[x, y, z]`` the pingponggrating xyz dispatcher reads.

    Mirrors :func:`_fold_distortion_zernike`. Lets a scan sweep a SCALAR axis
    (e.g. ``extras.step_y.scan(...)``) for a perpendicular (y) or axial (z) move
    instead of a list-valued ``step_size`` axis. A list-valued swept axis breaks
    the lab-side N-D scan grid: a 2-D scan with one ``(npts, 3)`` list-of-lists
    axis plus a scalar axis fails to build ("concatenation axis doesn't match
    along axis 0"), so the scan errors at dequeue before any shot runs. We build
    the 3-vector here, per shot. An explicit ``step_size`` already present wins;
    ``step_x`` / ``step_y`` / ``step_z`` are consumed (popped)."""
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    comps = {}
    for axis, key in (("x", "step_x"), ("y", "step_y"), ("z", "step_z")):
        if key in extras:
            comps[axis] = float(extras.pop(key))
    if not comps:
        return args
    extras.setdefault("step_size", [comps.get("x", 0.0),
                                    comps.get("y", 0.0),
                                    comps.get("z", 0.0)])
    return args


# =========================================================================== #
# helpers
# =========================================================================== #
def _frame_pattern(ctx, idx):
    """The pattern NAME to detect camera frame ``idx`` against (0=loading, ..., last=final),
    from ``ctx.frame_patterns`` when the scan declared it; else None so the detector uses the
    frame-0 / day-folder calibration. Out-of-range -> None (be permissive; the detector falls
    back)."""
    fp = getattr(ctx, "frame_patterns", None)
    if fp and 0 <= idx < len(fp):
        return fp[idx]
    return None


def _report_grab_fail(ctx, s1, tag, which, n_seen):
    """Record a frame-grab failure with the desync-vs-timeout distinction so the dashboard
    shot-health chip lights correctly."""
    if n_seen >= 2:
        ctx.record_error(
            "[%s] seq %d: %s grab saw %d frames (desynced buffer) -- drained + cancelling"
            % (tag, _seq_id(s1), which, n_seen), kind="frame_desync", seq_id=_seq_id(s1))
    else:
        ctx.record_error(
            "[%s] seq %d: %s unavailable (timeout, 0 frames) -- cancelling"
            % (tag, _seq_id(s1), which), kind="frame_timeout", seq_id=_seq_id(s1))


def _ensure_extra(args, name, value):
    """Set ``args['extras'][name] = value`` unless already present (mirror the MATLAB
    ensure_default_extra). Creates the extras dict if absent so the value always reaches the
    server."""
    extras = args.get("extras")
    if not isinstance(extras, dict):
        extras = {}
        args["extras"] = extras
    extras.setdefault(name, value)
    return args


def _seq_id(s1):
    try:
        return int(s1.G.seq_id(1))
    except Exception:  # noqa: BLE001
        return 1


def _runid_kwargs(scan_id, seq_id):
    """Lab-PC run-ID propagation for the SLM diag ledger. ``YB_SLM_DISABLE_RUNID=1`` -> {}
    (legacy body shape) as an emergency rollback, mirroring
    RearrangeCommSeq.m::build_runid_opts."""
    if os.environ.get("YB_SLM_DISABLE_RUNID") == "1":
        return {}
    return {"scan_id": str(scan_id), "seq_id": int(seq_id)}


def _safe(obj, method, *args):
    if obj is None:
        return
    fn = getattr(obj, method, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass
