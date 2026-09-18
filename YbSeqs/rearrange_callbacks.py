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

import contextlib
import os
import time

import rearrange_runtime
import run_timing            # substage timing (no-op when RUN_TIMING is off)


# =========================================================================== #
# per-round timing -- decomposes the run loop's before_bseq stage
# =========================================================================== #
# before_bseq is the sum of the rounds' callbacks and is measured by the run loop as ONE
# number, which cannot say whether a slow round is the LOCAL work (waiting on the camera
# frame, per-site detection) or the SLM server round-trip. This splits it. Lives here rather
# than in run_timing's SUBSTAGES because YbSeqs/ hot-reloads per job while lib/ needs a
# backend restart -- and a restart is exactly what we must not perturb while chasing this.
# One log line per round; arithmetic only, so it cannot fail a shot.
_RT = {}


@contextlib.contextmanager
def _t(key):
    """Accumulate elapsed ms into the current round's ``_RT`` bucket."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _RT[key] = _RT.get(key, 0.0) + (time.perf_counter() - t0) * 1e3


def _rt_log(ctx, tag, frame_idx, seq_id):
    """Emit the round's breakdown; best-effort (diagnostics never break a run)."""
    try:
        if not _RT:
            return
        parts = " ".join("%s=%.1f" % (k, _RT[k])
                         for k in ("grab", "detect", "note", "stage", "http") if k in _RT)
        ctx.log("[rr_timing] %s f%d seq %d | total=%.1f %s"
                % (tag, frame_idx, seq_id, sum(_RT.values()), parts))
    except Exception:  # noqa: BLE001 - never raise out of instrumentation
        pass

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


def stash_frame(frame_idx, img):
    """Record camera frame ``frame_idx`` for the failing-shot display re-publish.

    The public entry point for a seq that grabs its frames with its OWN round callback (e.g.
    Rearrange3DFocusWalkCommSeq, whose rounds post a COMPOSED bit vector) instead of
    :func:`rearrange_round` -- so such a seq still participates in the display re-publish
    without reaching into the module-private stash."""
    if img is not None:
        _STASH[int(frame_idx)] = img


# =========================================================================== #
# no_transit -- the opt-in PURE-IMAGING control arm (2026-09-02)
# =========================================================================== #
def _no_transit(s1):
    """True when this shot is the PURE-IMAGING control: two images with NOTHING between them.

    Set by a scan as ``rearrange_kwargs.extras.no_transit`` (0/1, scannable like any other
    extras axis). It is the in-job denominator a method-comparison run normalises to: no
    ``rearrange()`` call, so zero SLM frames, zero panel writes, zero prefill and no transit
    dwell -- the atoms simply sit in the untouched loading array between img1 and img2.

    It is NOT the server's ``no_move`` (dst := src), which still plays every frame, pays the
    prefill and writes the WGS bookend; that arm measures imaging PLUS dwell PLUS refresh. The
    two together separate those terms (campaigns/rearr/methods CORRECTION 2026-08-13, where the
    imaging maximum had to be reconstructed from a SEPARATE run because no in-job arm carried
    it).

    Everything BEFORE the rearrange call is deliberately left untouched on this arm -- the same
    camera grab, the same detector, the same probability log -- so the control's denominator
    carries the same atom selection the transport arms condition on.

    Absent extra -> False -> every pre-existing seq/scan takes exactly its old path.
    """
    try:
        return bool(int(float(s1.C.rearrange_kwargs.extras.no_transit(0))))
    except Exception:  # noqa: BLE001 - absent/odd extras -> the normal transport path
        return False


def _pop_extra(args, name):
    """Remove ``args['extras'][name]`` if present (lab-side-only extras never reach the server).

    ``collect_kwargs`` bundles EVERY ``extras.*`` leaf into the setup_rearrangement body, so a
    purely lab-side marker has to be dropped here: the server would either reject the unknown
    kwarg or -- worse for an interleaved scan -- fold it into the sticky protocol cache."""
    extras = args.get("extras")
    if isinstance(extras, dict):
        extras.pop(name, None)
    return args


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
    # scan (engine_run.py: flush + start_video); within a scan the frames are grabbed one at a
    # time, so a single straggler -- a frame whose readout landed after its grab timed out, a
    # cancelled pair's leftover, or a spurious trigger -- offsets the stream by one and the
    # frames come out SHIFTED for the rest of the scan. pre_run runs BEFORE this shot's
    # Imag399 #1 triggers, so anything buffered now is stale: drop it and surface a nonzero
    # count (the previous shot(s) were misaligned). The per-shot analog of the scan-start flush.
    cam = ctx.camera
    if cam is not None:
        with run_timing.substage("cam_resync"):
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
                    % (_seq_id(s1), stale, frames_label), kind="frame_desync",
                    seq_id=_seq_id(s1))

    # The scan-long slm lock is mandatory: ensure_held re-acquires + rewrites the LOADING phase
    # if the lease lapsed (a shot longer than the lease). A failure raises -> the run errors.
    with run_timing.substage("slm_hold"):
        ctx.session.ensure_held()

    c = ctx.client
    # Per-shot compute (GPU) lock, blocking ~1 s. On miss, scrap THIS shot and retry rather
    # than racing rearrange() against another client.
    try:
        with run_timing.substage("slm_lock"):
            c.acquire_lock("compute", lock_desc, timeout_s=10, block_timeout=1)
        s1.G.rearrange_lock_ok = True
    except Exception as err:  # noqa: BLE001 - contention / timeout -> cancel + retry
        ctx.log("[pre_run] compute lock acquire failed: %s -- cancelling seq for retry" % err)
        _safe(ctx.server, "seq_cancel")
        return

    with run_timing.substage("slm_health"):
        try:
            c.health()                  # prewarm the connection
        except Exception:  # noqa: BLE001
            pass

    # Per-shot setup_rearrangement from rearrange_kwargs (the SCANNED params); NO reset_params
    # so everything else stays sticky from the initial (dequeue-time) setup call. The scalar
    # folds (extras.z<N> -> zernike_coeffs, extras.distortion_z<N> -> distortion_zernike,
    # extras.step_r/step_theta_deg -> step_x/step_y, extras.step_x/y/z -> step_size) let a scan
    # sweep SCALAR axes for the server's list-valued knobs (a list-valued swept axis breaks the
    # lab-side scan grid/curve); each is a no-op when its keys are absent, and an explicit list
    # already present always wins.
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs)
    args = rearrange_runtime.translate_zernike_zN(args)
    args = _fold_distortion_zernike(args)
    args = _fold_step_polar(args)       # step_r/step_theta_deg -> step_x/step_y (BEFORE the xyz fold)
    args = _fold_step_xyz(args)
    args = _fold_return_step_xyz(args)  # return_step_x/y/z -> return_step_size (same shape rule)
    # LAST: the pseudo-one-way outward leg is DERIVED from the resolved return leg, so it needs
    # every step_size fold above to have already run. No-op without extras.out_step_max.
    args = _fold_ppg_outward_leg(args)
    # Also derived from the resolved scalar step_size (it rewrites it to the base amplitude and
    # emits the non-uniform disp_schedule). No-op without extras.soft_nsteps; mutually exclusive
    # with the pseudo-one-way fold in practice, since that one is a two-leg triangle.
    args = _fold_ppg_soft_start(args)
    # Same disp_schedule machinery, different shape: a SHORT-EXCURSION ping-pong that turns
    # around every `pingpong_group` steps instead of once. No-op without
    # extras.pingpong_nsteps; a loud error if combined with the soft-start fold (both own the
    # schedule).
    args = _fold_ppg_pingpong_group(args)
    # LAB-SIDE ONLY. The pure-imaging control arm is decided here, not on the server, so the
    # marker is stripped before the body is built: the server never sees an unknown kwarg and
    # the rearrange2 stream signature is untouched, which is what lets the control cell
    # interleave shot-to-shot with the transport arms without forcing a producer rebuild.
    _pop_extra(args, "no_transit")
    if force_n_rounds is not None:
        _ensure_extra(args, "n_rounds", int(force_n_rounds))
    args.setdefault("client_scan_id", str(ctx.scan_id))
    # Mirror the scan's imaging-weighted-Hungarian request onto the context so the per-round
    # probability diagnostics can say whether the floats we post can actually bite. The flag
    # itself travels to the server INSIDE these extras (nothing extra is sent).
    _note_prob_hungarian(ctx, args)
    try:
        with run_timing.substage("slm_setup"):
            c.setup_rearrangement(**args)
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[pre_run] setup_rearrangement failed: %s" % err,
                         kind="setup_rearrangement", seq_id=_seq_id(s1))
    try:
        with run_timing.substage("slm_reload"):
            c.reload_rearrange()
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[pre_run] reload_rearrange failed: %s" % err,
                         kind="reload_rearrange", seq_id=_seq_id(s1))


# =========================================================================== #
# rearrange_round -- one rearrangement handoff (reg_before_bseq)
# =========================================================================== #
def rearrange_round(s1, frame_idx, *, ok_flag, tag, prior_flag=None,
                    use_frame_pattern=False, record_ok=False,
                    grab_timeout=0.1, drain_timeout=0.2, min_load=None):
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
        min_load: OPT-IN atom-count gate (None/0 = off, the production default -- only the Dev
            seq passes it): if this round's detected atom count (probs > 0.5) is below it, the
            shot is SKIPPED (frame drained + cancel_shot, no rearrange, ok_flag left False so
            later rounds/finalize cancel too). Keeps thermalization (the sequence still runs)
            without spending SLM playback / polluting stats on shots that can't reach the
            final target.
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

    _RT.clear()
    with _t("grab"):
        img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)
    if not ok:
        _report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH[frame_idx] = img

    with _t("detect"):
        probs = (ctx.detect_probs_for(_frame_pattern(ctx, frame_idx), img)
                 if use_frame_pattern else ctx.detect_probs(img))
    if not probs:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    # Post-hoc visibility of the probability path (prob_hungarian): summarise what this round
    # posts -- how many sites are loaded, how many are MARGINAL, and the total -log(p) the
    # server turns into cost -- plus the surplus against the NEXT frame's pattern (the only
    # regime where the -beta*log(p) term can change the assignment). Throttled + best-effort.
    with _t("note"):
        _note_probs(ctx, tag, frame_idx, probs, use_frame_pattern)

    # PURE-IMAGING control arm (extras.no_transit): stage the frame and mark the round healthy
    # WITHOUT calling rearrange(), so the SLM does literally nothing between the two images.
    # Placed AFTER the grab/detect/log above on purpose -- the control has to carry the same
    # atom selection as the arms it is the denominator for. See :func:`_no_transit`.
    if _no_transit(s1):
        _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
        setattr(s1.G, ok_flag, True)
        if record_ok:
            ctx.record_ok()
        return

    if min_load:
        n_at = sum(1 for p in probs if p > 0.5)
        if n_at < int(min_load):
            # Normal failing-shot abort path (same as grab-fail): record + cancel; ok_flag stays
            # False so later rounds drain their frames and finalize republishes display-only.
            ctx.record_error("[%s] seq %d: low load %d < %d -- shot aborted (min-load gate)"
                             % (tag, _seq_id(s1), n_at, int(min_load)),
                             kind="low_load", seq_id=_seq_id(s1))
            _safe(ctx.server, "cancel_shot")
            return

    _do_rearrange_round(ctx, s1, img, probs, tag, ok_flag, record_ok=record_ok)
    _rt_log(ctx, tag, frame_idx, _seq_id(s1))


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

    _RT.clear()
    with _t("grab"):
        img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)
    if not ok:
        _report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH[frame_idx] = img

    with _t("detect"):
        bits = (ctx.detect_bits_for(_frame_pattern(ctx, frame_idx), img)
                if use_frame_pattern else ctx.detect_bits(img))
    if bits and ctx.client is not None:
        try:
            with _t("http"):
                ctx.client.update_rearrange(bits, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
        except Exception as err:  # noqa: BLE001
            ctx.log("[%s] update_rearrange failed: %s" % (tag, err))
    with _t("stage"):
        _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    setattr(s1.G, ok_flag, True)
    _rt_log(ctx, tag, frame_idx, _seq_id(s1))


# =========================================================================== #
# finalize -- final frame + shot publication + teardown (reg_after_end)
# =========================================================================== #
def finalize(s1, *, round_flags, final_frame_idx, tag="post_run",
             update=True, use_frame_pattern=False, record_ok=False, grab_timeout=0.1,
             bits_fn=None):
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
        bits_fn: optional ``bits -> bits`` map applied to the final frame's detected bitstring
            before ``update_rearrange``. For a seq whose lab-side detection covers only PART of
            the server's init_grid (Rearrange3DFocusWalkCommSeq detects one axial layer of a
            2-layer 3-D grid, so it must pad the other layer's sites with '0'). ``None`` ->
            the bits are posted exactly as detected (every pre-existing caller).
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
        # The pure-imaging control never called rearrange(), so there is no server-side shot for
        # update_rearrange to close: posting the final occupancy would open a phantom ledger row
        # and poison the rearrange statistics. Its frames still stage + publish normally.
        if update and ctx.client is not None and not _no_transit(s1):
            bits = (ctx.detect_bits_for(_frame_pattern(ctx, final_frame_idx), img)
                    if use_frame_pattern else ctx.detect_bits(img))
            if bits and bits_fn is not None:
                bits = bits_fn(bits)
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
    with _t("stage"):
        _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    try:
        # Per-site presence PROBABILITIES (floats in [0,1]) in place of the hard bitstring, one
        # per site of THIS round's init grid, in that grid's order. The server rounds them at
        # 0.5 to build the bitstring (the source set) AND -- when the scan asked for
        # ``extras.prob_hungarian`` -- forwards the raw floats to the protocol as ``site_probs``,
        # where they add ``-beta*log(p)`` to each loaded row of the assignment cost. That term is
        # a per-ROW constant, so it only changes the pairing when atoms are in SURPLUS
        # (n_loaded > n_targets); see the ctx.note_probs summary logged above.
        with _t("http"):
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


def _fold_step_polar(args):
    """Fold scalar ``extras.step_r`` + ``extras.step_theta_deg`` into the scalar
    ``extras.step_x`` / ``step_y`` that :func:`_fold_step_xyz` then folds to the
    ``step_size`` 3-vector.

    Lets a 2-D scan sweep the LATERAL MOVE IN POLAR FORM -- distance on one axis,
    direction on the other -- which is how a 2-D movement-capability map is
    naturally parametrized (and how it is analysed: a native ``[n_r x n_theta]``
    grid). Sweeping ``step_x``/``step_y`` directly cannot express it, because a
    direction requires TWO co-varying scalars, and a list-valued ``step_size``
    axis breaks the lab-side N-D scan grid (see :func:`_fold_step_xyz`).

    ``step_x = r * cos(theta)``, ``step_y = r * sin(theta)``, theta in DEGREES
    measured from the +x axis. Runs BEFORE :func:`_fold_step_xyz` in the per-shot
    chain, so the derived components still become ``step_size``. Both keys are
    consumed (popped). An explicit ``step_x``/``step_y``/``step_size`` already
    present wins (``setdefault``), and the fold is a no-op when ``step_r`` is
    absent -- a bare ``step_theta_deg`` (no radius) is meaningless, so it is
    dropped rather than silently treated as r=0."""
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    if "step_r" not in extras:
        extras.pop("step_theta_deg", None)
        return args
    import math as _math
    r = float(extras.pop("step_r"))
    theta = float(extras.pop("step_theta_deg", 0.0))
    rad = _math.radians(theta)
    extras.setdefault("step_x", r * _math.cos(rad))
    extras.setdefault("step_y", r * _math.sin(rad))
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


def _fold_return_step_xyz(args):
    """Fold scalar ``extras.return_step_x`` / ``return_step_y`` / ``return_step_z``
    into the 3-vector ``extras.return_step_size`` = ``[x, y, z]`` the
    pingponggrating asymmetric-leg dispatcher reads.

    The RETURN-leg twin of :func:`_fold_step_xyz`, and it exists for exactly the
    same reason: a list-valued swept axis breaks the lab-side N-D scan grid, so a
    scan that wants to sweep the return leg's lateral amplitude has to sweep a
    SCALAR. The server requires ``return_step_size`` to have the SAME SHAPE as
    ``step_size`` (both scalar, or both ``[sx, sy, sz]``) and raises a ValueError
    otherwise, so this must run alongside :func:`_fold_step_xyz` -- if the outward
    leg is in xyz mode the return leg must be too.

    An explicit ``return_step_size`` already present wins; the three scalar keys
    are consumed (popped). No-op when none are present (the pure-axial scans pass
    a SCALAR ``return_step_size`` directly and never touch this)."""
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    comps = {}
    for axis, key in (("x", "return_step_x"),
                      ("y", "return_step_y"),
                      ("z", "return_step_z")):
        if key in extras:
            comps[axis] = float(extras.pop(key))
    if not comps:
        return args
    extras.setdefault("return_step_size", [comps.get("x", 0.0),
                                           comps.get("y", 0.0),
                                           comps.get("z", 0.0)])
    return args


def _fold_ppg_outward_leg(args):
    """Derive the pingponggrating OUTWARD leg from the tested RETURN leg -- the
    "pseudo-one-way" move.

    THE MEASUREMENT.  A one-way axial move is not physically runnable: the array
    would rest defocused, the WGS bookend would snap it back, and the atoms would
    be gone.  So the tested move is run as the RETURN leg of an asymmetric
    triangle: the array first walks OUT in ``-d`` in many small, qualified,
    lossless steps, dwells ``hold_ms`` at the turnaround so the LC is fully
    settled, and then makes the move under test in ``+d`` back to displacement 0.
    Displacement 0 is the stored WGS phase byte-for-byte, so the array lands
    exactly where it started and LIVE detection stays valid (no offline
    re-detection, unlike the radial one-way campaign).

    WHY THIS HAS TO BE A PER-SHOT FOLD.  The outward leg's step count depends on
    BOTH scanned axes at once::

        D      = return_nsteps * return_step_size          (the tested travel)
        n_out  = ceil(|D| / out_step_max)
        s_out  = D / n_out                                 (<= out_step_max, signed)

    ``n_out`` is therefore a function of the (step_size, nsteps) CELL, which no
    ScanGroup product axis can express.  Building it here, per shot, is the same
    trick :func:`_fold_step_polar` uses for a direction axis.

    WHY THE DIVISION MUST BE EXACT.  The server's return leg descends displacement
    ``n_ret-1 .. 0`` at its own amplitude, so the turnaround is continuous only
    when ``n_out*s_out == n_ret*s_ret``.  Writing ``s_out = D / n_out`` makes that
    identity hold to ~2*eps*|D| (~1e-13 um at a 300 um peak); the server stamps
    ``diag["leg_out_peak"]`` / ``diag["leg_ret_peak"]`` so the analysis can assert
    it rather than trust it.

    SIGN.  The dispatcher's displacement index is NON-NEGATIVE -- the direction of
    travel lives entirely in the amplitude sign.  So the outward leg must carry
    the SAME sign as the tested leg (both negative to test a ``+z`` move: out to
    ``-D``, then back up to 0).  ``s_out = D / n_out`` inherits that automatically.

    ``return_nsteps == 0`` (or ``return_step_size == 0``) is the STATIC CONTROL:
    it degenerates to ``n_out = 0`` and a single WGS write, NOT the
    "rest on the peak frame" footgun a bare ``return_nsteps=0`` would give.

    Activated only by an explicit positive ``extras.out_step_max`` (magnitude, in
    the tested leg's own numeraire -- microns under ``true_defocus``, knm-px
    lateral).  That key is CONSUMED.  ``nsteps`` and ``extras.step_size`` are
    OVERWRITTEN (they are derived quantities here, not inputs).  For provenance
    the applied values are echoed into extras as ``ppg_out_step_applied`` /
    ``ppg_peak_travel`` -- unknown extras are forwarded verbatim by the server and
    show up in ``setup.extra_params`` / ``/slm/results``.

    Runs LAST in the per-shot fold chain so it sees the final resolved
    ``return_step_size`` (scalar, or the 3-vector :func:`_fold_return_step_xyz`
    just built)."""
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    try:
        out_step_max = float(extras.pop("out_step_max"))
    except (KeyError, TypeError, ValueError):
        return args
    if out_step_max <= 0.0:
        return args

    import math as _math
    s_ret = extras.get("return_step_size", None)
    n_ret = int(extras.get("return_nsteps", 0) or 0)
    vec = isinstance(s_ret, (list, tuple))
    comps = [float(v) for v in s_ret] if vec else [float(s_ret or 0.0)]

    # D_i = n_ret * s_ret_i, per component.  n_out is set by whichever component
    # needs the most steps to stay under out_step_max; the others then take
    # smaller steps, which keeps the direction of the outward leg parallel to the
    # tested one (a pure-axis move has only one nonzero component anyway).
    travel = [n_ret * c for c in comps]
    reach = max(abs(t) for t in travel) if travel else 0.0
    if n_ret <= 0 or reach == 0.0:
        # THE CONTROL. It must pay the same SLM WRITE COUNT as the data cells, not just the same
        # wall-clock delay.
        #
        # Job 315 measured a step=0 (zero-motion) shot at 0.9495 survival for n=400 and 0.9340 for
        # n=500, against 0.9908 at n<=300 -- a 6.6% loss with NO motion whatsoever, of which the
        # measured trap lifetime (tau = 300 s) explains only 0.5%. The cost is the WRITES
        # themselves: every Write_image re-addresses the panel and the liquid crystal re-settles,
        # modulating the traps. It turns sharply worse above ~600 frames.
        #
        # A single-frame control would therefore sit 4-6% ABOVE the data cells for a reason that
        # has nothing to do with transport, and every cliff normalized against it would be biased
        # by that much. So the control instead writes `out_nsteps_min` frames at ZERO amplitude:
        # identical WGS phase every frame (amp = 0 => _phase_for_disp returns the WGS phase for
        # every d), so it makes no motion at all but pays exactly the data cells' write count.
        # `return_nsteps = 0` still means no return frames, and the SLM rests on WGS -- the
        # rest-on-the-peak footgun needs a nonzero amplitude and cannot fire here.
        n_ctrl = 0
        try:
            n_ctrl = max(0, int(extras.pop("out_nsteps_min")))
        except (KeyError, TypeError, ValueError):
            n_ctrl = 0
        args["nsteps"] = n_ctrl
        extras["return_nsteps"] = 0
        extras["step_size"] = [0.0, 0.0, 0.0] if vec else 0.0
        extras["return_step_size"] = [0.0, 0.0, 0.0] if vec else 0.0
        extras["ppg_out_step_applied"] = 0.0
        extras["ppg_peak_travel"] = 0.0
        extras.setdefault("return_trip", True)
        return args

    n_out = int(_math.ceil(reach / out_step_max))
    # CONSTANT-n_out mode (extras.out_nsteps_min). Precompute cost and outward motion time both
    # scale with n_frames, which under the plain ceil() scales with D -- i.e. the per-shot in-trap
    # DELAY would be perfectly correlated with the tested step size, the very axis being measured,
    # so vacuum loss during precompute would masquerade as transport loss growing with stroke.
    # Pinning n_out to the scan's worst-cell value makes n_frames (and therefore the delay)
    # CONSTANT across the grid, demoting that systematic to a common-mode offset the step=0 control
    # already absorbs. s_out = D/n_out then simply shrinks for small-D cells -- strictly gentler
    # transport, and still <= out_step_max because the floor only ever RAISES n_out.
    try:
        n_min = int(extras.pop("out_nsteps_min"))
    except (KeyError, TypeError, ValueError):
        n_min = 0
    n_out = max(1, n_out, n_min)
    s_out = [t / n_out for t in travel]

    args["nsteps"] = n_out
    extras["step_size"] = s_out if vec else s_out[0]
    extras["ppg_out_step_applied"] = max(abs(v) for v in s_out)
    extras["ppg_peak_travel"] = reach
    extras.setdefault("return_trip", True)
    return args


def _fold_ppg_soft_start(args):
    """Build the pingponggrating SOFT-START displacement schedule.

    THE MEASUREMENT.  A one-way transport move of ``N`` steps of stroke ``s`` in
    which the FIRST ``m`` steps are only ``s/k`` -- so the array eases into the
    move instead of taking its first hop at full stroke straight off the static
    WGS phase.  If a fixed FIRST-STEP loss is what depresses the short-period
    99%-per-step stroke (``d99`` falls 1.78 -> 1.39 knm-px going from 3x to 1x the
    0.696 ms SLM write floor, campaign 2026-08-06), softening the entry should buy
    back some of it; if the loss is purely per-step and memoryless, it should not.
    A NEGATIVE ``m`` puts the ``|m|`` small steps at the END instead -- the
    distance-, write-count- and step-multiset-matched CONTROL, so ``+m`` vs ``-m``
    isolates WHERE the small steps sit and nothing else.

    HOW IT IS EXPRESSED.  The server's ping-pong dispatcher writes frame ``k`` at
    phase ``ip + (amp * disp[k]) * unit_map``, i.e. the position of every frame is
    an INTEGER multiple of one base amplitude.  Two step sizes in one leg are
    therefore just a non-uniform integer ``disp`` sequence at the finer base
    amplitude: with ``amp = s/k``, an increment of ``1`` is a soft step and an
    increment of ``k`` is a full one.  This fold emits that sequence as
    ``extras.disp_schedule`` (server support added 2026-08-11) and rewrites
    ``extras.step_size`` to the base amplitude ``s/k`` accordingly.

    WHY THIS HAS TO BE A PER-SHOT FOLD.  ``disp_schedule`` is a LIST whose contents
    depend on the ``(step_size, soft_start_m)`` cell; a list-valued swept axis
    breaks the lab-side scan grid, exactly as for :func:`_fold_step_xyz`.  So the
    scan sweeps the scalar ``extras.soft_start_m`` and the list is built here.

    INVARIANTS the comparison rests on, all held across the whole ``m`` axis:
      * FRAME COUNT is ``soft_nsteps + 1`` for every ``m`` (and for ``s = 0``).
        SLM writes cost survival on their own -- a zero-motion 400-frame shot
        measured 0.9495 vs 0.9908 below 300 frames (job 315) -- so an ``m`` axis
        that changed the write count would confound the very effect it measures.
      * ``m = 0`` reproduces the plain uniform one-way move EXACTLY (increments of
        ``k`` at amplitude ``s/k`` are the same positions as increments of 1 at
        ``s``), so every scan carries its own byte-level baseline.
      * total travel is ``s * (N - |m| * (1 - 1/k))``, the SAME for ``+m`` and
        ``-m``.  It is echoed as ``ppg_soft_travel`` so the offline re-detection
        can shift the img2 grid by this cell's own ``dx`` (one-way rests
        displaced -- live img2 survival is meaningless; see
        ``pyctrl/tools/ppg_transport_analyze.py``).

    Activated only by an explicit ``extras.soft_nsteps`` (the TOTAL step count
    ``N``); ``soft_start_m`` defaults to 0 and ``soft_denom`` (``k``) to 2.  All
    three keys are CONSUMED.  ``nsteps`` and ``extras.step_size`` are OVERWRITTEN
    (derived here, not inputs); ``nsteps`` is set to ``max(disp)`` because that is
    what the server's setup-time uint8 frame cache is indexed by.  Scalar
    (lateral / scalar-axial) ``step_size`` only -- an xyz 3-vector leg is a loud
    error rather than a guess about which axis the schedule refers to.

    For provenance the applied values are echoed into extras as ``ppg_soft_m`` /
    ``ppg_soft_denom`` / ``ppg_soft_nsteps`` / ``ppg_soft_travel`` -- unknown
    extras are forwarded verbatim by the server and land in
    ``setup.extra_params`` / ``/slm/results``.
    """
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    try:
        n_total = int(extras.pop("soft_nsteps"))
    except (KeyError, TypeError, ValueError):
        return args
    m = int(extras.pop("soft_start_m", 0) or 0)
    k = int(extras.pop("soft_denom", 2) or 2)
    if n_total < 0:
        raise ValueError("soft_nsteps must be >= 0, got %d" % n_total)
    if k < 1:
        raise ValueError("soft_denom must be >= 1, got %d" % k)
    if abs(m) > n_total:
        raise ValueError("|soft_start_m| (%d) exceeds soft_nsteps (%d)" % (m, n_total))

    s = extras.get("step_size", 0.0)
    if isinstance(s, (list, tuple)):
        raise ValueError(
            "soft-start needs a SCALAR step_size (which axis would the schedule "
            "refine?); got the xyz vector %r" % (s,))
    s = float(s)

    # Increments in units of the base amplitude s/k: `1` = soft step, `k` = full step.
    n_soft = abs(m)
    incs = [1] * n_soft + [k] * (n_total - n_soft)
    if m < 0:                       # the distance-matched control: small steps LAST
        incs.reverse()
    disp, acc = [0], 0
    for inc in incs:
        acc += inc
        disp.append(acc)

    args["nsteps"] = int(disp[-1])          # setup-cache extent, not the frame count
    extras["step_size"] = s / float(k)
    extras["disp_schedule"] = disp
    # One-way is the only mode the schedule is defined for: the server ignores
    # `return` once a schedule is given, and the array rests wherever the last
    # frame puts it. Stated explicitly so a sticky return_trip=True from an axial
    # scan cannot make the intent ambiguous in the recorded extras.
    extras["return_trip"] = False
    extras["ppg_soft_m"] = m
    extras["ppg_soft_denom"] = k
    extras["ppg_soft_nsteps"] = n_total
    extras["ppg_soft_travel"] = s * (n_total - n_soft * (1.0 - 1.0 / float(k)))
    return args


def _turnaround_idx(disp):
    """Interior indices of ``disp`` that are local extrema -- the velocity reversals. The two
    endpoints are not reversals (the array starts and finishes at rest)."""
    out = []
    for i in range(1, len(disp) - 1):
        a, b, c = disp[i - 1], disp[i], disp[i + 1]
        if (b > a and b > c) or (b < a and b < c):
            out.append(i)
    return out


def _apply_pingpong_hold(disp, hold, extras):
    """Dwell ``|hold|`` extra frames at each turnaround (hold > 0), or the PLACEMENT-MATCHED
    CONTROL that pays the same extra frames away from the turnarounds (hold < 0).

    THE MEASUREMENT.  The ``g`` ladder showed the cost of a short-excursion ping-pong sits at the
    velocity reversals (~1.7 % of survival each at 3 um / 3 ms, ~23x a step).  If that is because
    the atom's residual drift velocity -- which has been following the trap -- is suddenly opposed
    to the new direction of travel, then PAUSING at the reversal lets it dephase before the trap
    turns, and the cost should fall with dwell.  If the cost is instead intrinsic to the reversal
    geometry, dwelling buys nothing.

    A dwell is expressible in the schedule directly: repeating a displacement index is a frame that
    writes the same phase again, i.e. the array standing still for one more period.

    WHY THE SIGN.  A dwell ADDS FRAMES, and SLM writes cost survival on their own (a zero-motion
    400-frame shot measured 0.9495 vs 0.9908 below 300, job 315), as does the extra in-trap time.
    So ``hold = +h`` alone is confounded with its own frame count.  ``hold = -h`` inserts exactly
    the SAME number of repeated frames at NON-extremal positions instead -- mid-leg pauses.  The
    pair is matched in frame count, in step count, in total travel and in step-size multiset, and
    differs only in WHERE the dwell sits, so ``S(+h)/S(-h)`` is the turnaround effect with no model
    in between.  (Same construction, and the same reason, as the ``+m`` / ``-m`` pair in
    :func:`_fold_ppg_soft_start`.)

    Both are echoed via ``ppg_pp_hold`` / ``ppg_pp_frames``; the caller sets ``nsteps`` from
    ``max(disp)``, which a dwell never changes."""
    n = abs(int(hold))
    turns = _turnaround_idx(disp)
    total = n * len(turns)
    if not total:
        return list(disp)

    if hold > 0:
        out = []
        tset = set(turns)
        for i, d in enumerate(disp):
            out.append(d)
            if i in tset:
                out.extend([d] * n)             # stand still at the reversal
        return out

    # THE CONTROL: the same `total` repeated frames, placed at interior NON-extremal indices,
    # spread as evenly as the trajectory allows. Each candidate takes at most one repeat first,
    # then a second pass, so the dwell stays distributed rather than piling into one mid-leg stop.
    cand = [i for i in range(1, len(disp) - 1) if i not in set(turns)]
    if not cand:
        raise ValueError("pingpong_hold control needs interior non-turnaround frames; the "
                         "schedule has none (g too small?)")
    extra = {}
    for k in range(total):                      # k -> cand evenly; collides only if total > cand
        extra.setdefault(cand[k * len(cand) // total], 0)
        extra[cand[k * len(cand) // total]] += 1
    out = []
    for i, d in enumerate(disp):
        out.append(d)
        if extra.get(i):
            out.extend([d] * extra[i])
    return out


def _build_trough_schedule(peak, leg, n_total):
    """Displacement schedule whose interior reversals all sit at NONZERO displacement:
    ``0 -> peak``, then ``k`` oscillations ``peak -> peak-leg -> peak``, then ``peak -> 0``,
    with ``k = (n_total - 2*peak) / (2*leg)``.

    THE MEASUREMENT (2026-08-19).  The g ladder is DEGENERATE in what a "turnaround" is: per
    out-and-back cycle the departures from rest at 0, the arrivals at 0, the peak reversals and
    the visits to the pristine-WGS frame all scale together, so its ~1.69 %/turnaround cost
    (jobs 823-825) cannot say whether the price is the DIRECTION CHANGE itself or something
    special about the 0 frame (first step off the static WGS phase / last step back into it).
    This schedule holds the WGS-frame visits fixed at exactly one departure and one arrival
    regardless of ``k`` while sweeping the reversal count ``2k + 1`` -- all of them at nonzero
    displacement, mid-flight.  A per-reversal cost that matches the ladder's kills the
    first-step / last-step / WGS-frame hypotheses; a vanishing one localizes the penalty at the
    zero-frame visits.

    Every increment is +-1, the schedule ends at 0 (live img2 stays valid) and its length is
    ``n_total + 1`` frames -- the same invariants as the plain group triangle."""
    if not (1 <= leg <= peak - 1):
        raise ValueError("pingpong_trough_leg (%d) must be in [1, peak-1] (peak %d) so the "
                         "trough stays at a NONZERO displacement" % (leg, peak))
    rem = n_total - 2 * peak
    if rem < 0 or rem % (2 * leg):
        raise ValueError(
            "pingpong_nsteps (%d) minus the out-and-back to peak %d leaves %d steps, which is "
            "not a whole number of 2*leg (%d) oscillations -- the schedule would end displaced"
            % (n_total, peak, rem, 2 * leg))
    k = rem // (2 * leg)
    disp = list(range(0, peak + 1))                     # 0 .. peak
    for _ in range(k):
        disp.extend(range(peak - 1, peak - leg - 1, -1))   # peak-1 .. peak-leg
        disp.extend(range(peak - leg + 1, peak + 1))       # peak-leg+1 .. peak
    disp.extend(range(peak - 1, -1, -1))                # peak-1 .. 0
    return disp, k


def _apply_pingpong_soft(disp, soft):
    """Halve the approach speed INTO and OUT OF every interior reversal (soft = +1), or the
    frame-count-matched control that halves the same number of steps MID-LEG (soft = -1).

    THE MEASUREMENT (2026-08-19).  If the per-reversal cost is a velocity kick -- the atom's
    residual drift velocity, which has been following the trap, suddenly opposed by the new
    direction of travel -- it should scale with the approach speed, and halving the last step
    into and the first step out of each reversal should buy a large part of it back.  If the
    cost is intrinsic to the reversal geometry (e.g. the LC transition itself), it should not.

    Expression: every index is DOUBLED and the caller halves ``step_size``, so a full step is an
    increment of 2 and a half step an increment of 1.  ``soft = +1`` replaces, at each interior
    extremum ``p``, the doubled pattern ``... p-2, p, p-2 ...`` by ``... p-2, p-1, p, p-1,
    p-2 ...`` (+2 frames per reversal).  ``soft = -1`` splits the same NUMBER of full mid-leg
    steps into half-step pairs instead, spread evenly -- matched in frame count, step-size
    multiset and total travel, differing only in WHERE the half steps sit (same signed-control
    construction as ``pingpong_hold`` and the soft-start ``+m``/``-m``).

    Returns the new (doubled) schedule; the caller rewrites ``step_size`` and ``nsteps``."""
    turns = _turnaround_idx(disp)
    d2 = [2 * d for d in disp]
    if not turns:
        return d2
    if soft > 0:
        tset = set(turns)
        out = []
        for i, d in enumerate(d2):
            if i in tset:
                mid_in = d - 1 if d2[i - 1] < d else d + 1     # halved approach
                out.extend([mid_in, d, mid_in])                # halved exit is symmetric
            else:
                out.append(d)
        return out
    # CONTROL: split 2*len(turns) full mid-leg steps into half-step pairs, spread evenly.
    n_splits = 2 * len(turns)
    tset = set(turns)
    cand = [i for i in range(1, len(d2)) if i not in tset and (i - 1) not in tset
            and abs(d2[i] - d2[i - 1]) == 2]
    if len(cand) < n_splits:
        raise ValueError("pingpong_soft control needs %d splittable mid-leg steps, schedule "
                         "has %d" % (n_splits, len(cand)))
    chosen = set(cand[j * len(cand) // n_splits] for j in range(n_splits))
    if len(chosen) < n_splits:
        raise ValueError("pingpong_soft control could not spread %d splits over %d candidates "
                         "without collision" % (n_splits, len(cand)))
    out = [d2[0]]
    for i in range(1, len(d2)):
        if i in chosen:
            out.append((d2[i] + d2[i - 1]) // 2)
        out.append(d2[i])
    return out


def _fold_ppg_pingpong_group(args):
    """Build a SHORT-EXCURSION ping-pong displacement schedule: turn around every
    ``pingpong_group`` steps instead of once at the far end.

    THE MEASUREMENT.  The standard ping-pong (``return_trip=True``, ``nsteps = n``)
    is ONE out-and-back: ``2n`` steps whose peak excursion is ``n * step_size``.
    Axially at ``n = 40`` and 1.5-2 um/step that parks the whole array 60-80 um off
    the atomic plane at the turnaround -- far outside the depth of field of every
    calibration the rig owns (imaging, the SLM->camera affine, the trap-depth /
    piston null), and far enough that the transported traps are no longer the traps
    the atoms were prepared in.  This fold keeps the STEP COUNT and the per-step
    stroke exactly as they were and shrinks only the EXCURSION, by folding the same
    total travel into ``N / (2g)`` short out-and-back cycles::

        g = 5,  N = 80:   0 1 2 3 4 5 4 3 2 1 0 1 2 ... 0     (8 cycles, 81 frames)
        g = 40, N = 80:   0 1 2 ... 40 ... 2 1 0             (1 cycle,  81 frames)

    ``g = N/2`` therefore REPRODUCES the standard ping-pong frame for frame (the
    built-in triangle is ``0..n`` then ``n-1..0``, which is exactly this schedule at
    ``g = n``), so a scan that sweeps ``g`` carries its own matched control and the
    comparison needs no cross-day differencing.  Peak excursion is ``g * step_size``.

    WHAT IS HELD FIXED, and why each matters:
      * TOTAL STEPS ``N`` -- the per-step heating/loss term integrates over steps, so
        an axis that changed the step count would confound the very quantity measured.
      * FRAME COUNT ``N + 1`` -- SLM writes cost survival on their own (a zero-motion
        400-frame shot measured 0.9495 vs 0.9908 below 300 frames, job 315).
      * STEP SIZE -- unchanged; every increment is +-1 at the base amplitude.
      * TURNAROUND COUNT is NOT held fixed and cannot be: it is ``N/g`` and is the
        whole point of the axis.  A per-turnaround cost therefore appears as a term
        that GROWS toward small ``g``, opposite in sign to any excursion benefit, and
        the two are separated by the shape of ``T(g)``, not by this fold.

    HOW IT IS EXPRESSED.  The server writes frame ``k`` at
    ``ip + (amp * disp[k]) * unit_map`` with ``disp`` non-negative integers, so a
    multi-cycle triangle is just a non-monotone index list at the SAME base
    amplitude -- ``extras.disp_schedule`` (server key added 2026-08-11).  Under a
    schedule every frame is leg 0, so ``reverse_zernike`` never flips and the
    asymmetric-leg knobs are refused by the server; that is harmless here because
    the axial/lateral maps carry no leg asymmetry in this configuration.

    WHY THIS HAS TO BE A PER-SHOT FOLD.  ``disp_schedule`` is a LIST whose contents
    depend on the ``(pingpong_group, pingpong_nsteps)`` cell, and a list-valued swept
    axis breaks the lab-side N-D scan grid (see :func:`_fold_step_xyz`).  So the scan
    sweeps the SCALAR ``extras.pingpong_group`` and the list is built here.

    ``nsteps`` is OVERWRITTEN with ``g`` -- ``max(disp)``, which is what the server's
    setup-time uint8 frame cache is indexed by (``use_precomputed_uint8`` refuses when
    ``_max_disp > nsteps``).  A short-excursion run therefore needs only ``g + 1``
    unique frames instead of ``n + 1``, which is strictly cheaper to precompute.

    ``N`` MUST BE AN EVEN MULTIPLE OF ``2g`` so the schedule ENDS at displacement 0.
    That is not tidiness: ending at 0 is what puts every atom back in its source trap
    at the WGS phase byte-for-byte, which is what keeps LIVE img2 detection valid (a
    schedule that rests displaced needs the offline re-detection the one-way campaign
    uses -- ``pyctrl/tools/ppg_transport_analyze.py``).  A non-dividing pair is a loud
    error rather than a silently one-way move.

    Activated only by an explicit ``extras.pingpong_nsteps`` (the TOTAL step count
    ``N``); ``pingpong_group`` defaults to ``N/2`` (= the standard single-excursion
    ping-pong).  Both keys are CONSUMED.  Scalar ``step_size`` only -- an xyz 3-vector
    is a loud error rather than a guess about which axis the excursion refers to.
    Applied values are echoed as ``ppg_pp_group`` / ``ppg_pp_nsteps`` /
    ``ppg_pp_cycles`` / ``ppg_pp_excursion`` (the peak excursion in the step's own
    numeraire -- microns under ``true_defocus``, knm-px lateral); unknown extras are
    forwarded verbatim by the server and land in ``setup.extra_params``.
    """
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return args
    try:
        n_total = int(extras.pop("pingpong_nsteps"))
    except (KeyError, TypeError, ValueError):
        return args
    if "disp_schedule" in extras:
        raise ValueError(
            "pingpong-group and soft-start both own extras.disp_schedule; set only one")
    if n_total <= 0 or n_total % 2:
        raise ValueError("pingpong_nsteps must be a positive EVEN step count, got %d"
                         % n_total)
    g = extras.pop("pingpong_group", None)
    g_given = g is not None
    g = n_total // 2 if g is None else int(g)
    if g < 1:
        raise ValueError("pingpong_group must be >= 1, got %d" % g)
    if n_total % (2 * g):
        raise ValueError(
            "pingpong_nsteps (%d) must be a whole number of out-and-back cycles of "
            "pingpong_group (%d), i.e. a multiple of %d -- otherwise the schedule ends "
            "displaced and live img2 detection is invalid"
            % (n_total, g, 2 * g))

    s = extras.get("step_size", 0.0)
    if isinstance(s, (list, tuple)):
        raise ValueError(
            "pingpong-group needs a SCALAR step_size (which axis would the excursion "
            "be along?); got the xyz vector %r" % (s,))
    s = float(s)

    trough_leg = extras.pop("pingpong_trough_leg", None)
    peak = int(extras.pop("pingpong_peak", 10) or 10)
    if trough_leg is not None and int(trough_leg) != 0:
        # nonzero-trough oscillation: reversals mid-flight, exactly one WGS-frame departure
        # and one arrival regardless of the reversal count (see _build_trough_schedule).
        if g_given:
            raise ValueError("pingpong_trough_leg and pingpong_group are mutually exclusive "
                             "schedule shapes; set only one")
        disp, cycles = _build_trough_schedule(peak, int(trough_leg), n_total)
        g = peak                                # cache extent = max displacement
    else:
        trough_leg = 0
        cycles = n_total // (2 * g)
        disp = [0]
        for _ in range(cycles):
            disp.extend(range(1, g + 1))        # out:  1 .. g
            disp.extend(range(g - 1, -1, -1))   # back: g-1 .. 0

    soft = int(extras.pop("pingpong_soft", 0) or 0)
    if soft:
        disp = _apply_pingpong_soft(disp, soft)
        s = s / 2.0                             # a full step is now an increment of 2
        extras["step_size"] = s
        g = 2 * g

    hold = int(extras.pop("pingpong_hold", 0) or 0)
    if hold:
        disp = _apply_pingpong_hold(disp, hold, extras)

    args["nsteps"] = int(g)                     # setup-cache extent, not the frame count
    extras["disp_schedule"] = disp
    extras["ppg_pp_frames"] = len(disp)
    extras["ppg_pp_hold"] = hold
    extras["ppg_pp_soft"] = soft
    extras["ppg_pp_trough_leg"] = int(trough_leg)
    extras["ppg_pp_peak"] = peak if trough_leg else 0
    extras["ppg_pp_reversals"] = len(_turnaround_idx(disp))
    # The server ignores `return` once a schedule is given (every frame is leg 0) and the
    # array rests wherever the last index puts it -- which here is 0, by construction.
    # Stated explicitly so a sticky return_trip=True cannot make the recorded intent
    # ambiguous.
    extras["return_trip"] = False
    extras["ppg_pp_group"] = g
    extras["ppg_pp_nsteps"] = n_total
    extras["ppg_pp_cycles"] = cycles
    # Peak excursion in the step's OWN numeraire.  The grating path carries its stroke in
    # `step_size` (um under true_defocus, knm-px lateral); the warm 3-D / model pingpong path
    # carries a pure-axial stroke in `step_size_z` (rad of PV Z4) with `step_size` left at 0.
    # Reporting `s * g` unconditionally logged a flat 0.0 for every warm-path run, so prefer
    # whichever stroke is actually non-zero and say which one it was.
    _sz = extras.get("step_size_z", 0.0)
    try:
        _sz = float(_sz) if not isinstance(_sz, (list, tuple)) else 0.0
    except (TypeError, ValueError):
        _sz = 0.0
    if s == 0.0 and _sz != 0.0:
        extras["ppg_pp_excursion"] = _sz * g
        extras["ppg_pp_excursion_axis"] = "step_size_z"
    else:
        extras["ppg_pp_excursion"] = s * g
        extras["ppg_pp_excursion_axis"] = "step_size"
    return args


# =========================================================================== #
# helpers
# =========================================================================== #
def _note_prob_hungarian(ctx, args):
    """Copy ``extras.prob_hungarian`` / ``extras.prob_hungarian_beta`` from this shot's
    setup_rearrangement kwargs onto the scan context (diagnostics only -- the values themselves
    still travel to the server inside ``args``). Best-effort: never fails a shot."""
    fn = getattr(ctx, "set_prob_hungarian", None)
    if fn is None:
        return
    extras = args.get("extras") if isinstance(args, dict) else None
    if not isinstance(extras, dict):
        return
    try:
        beta = extras.get("prob_hungarian_beta")
        fn(bool(extras.get("prob_hungarian", False)),
           None if beta is None else float(beta))
    except Exception:  # noqa: BLE001 - a diagnostic must never break the shot
        pass


def _note_probs(ctx, tag, frame_idx, probs, use_frame_pattern):
    """Hand this round's posted probabilities to ``ctx.note_probs`` for the throttled summary.
    The TARGET of round ``frame_idx`` is the pattern of the NEXT camera frame (round 1 moves the
    loading array into the middle array, imaged as frame 1), so that is what the surplus test
    compares against. Best-effort + version-tolerant (an older context has no note_probs)."""
    fn = getattr(ctx, "note_probs", None)
    if fn is None:
        return
    try:
        pat = _frame_pattern(ctx, frame_idx) if use_frame_pattern else ctx.pattern_name
        fn(tag, pat, probs, target_pattern=_frame_pattern(ctx, frame_idx + 1))
    except Exception:  # noqa: BLE001 - a diagnostic must never break the shot
        pass


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
