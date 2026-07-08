"""RearrangeCommSeq2.py -- two-round SLM-rearrangement variant of ``RearrangeCommSeq.py``.

Ground truth: the single-round ``RearrangeCommSeq.py`` (same lock/setup/detect/stage machinery).
This is the TWO-ROUND, THREE-PATTERN extension:

  Init -> MOT -> SLM -> GreenMOT -> LAC -> Imag399 (#1, LOADING pattern)
       -> Cool -> rearrange(round 1) -> Imag399 (#2, MIDDLE pattern)
       -> Cool -> rearrange(round 2) -> Imag399 (#3, FINAL pattern) -> Init

Three camera frames per shot. The scan (SLMRearrangementScan.py, two-round branch) declares three
patterns -- LOADING / MIDDLE / FINAL -- via ``rearrange_kwargs.extras.initial_pattern /
middle_pattern / final_pattern`` and a matching 3-entry ``runp().imagePatternsJson``. Each frame is
DETECTED with its OWN per-pattern registry grid + thresholds (independent site counts / orderings),
so the three patterns may be genuinely different arrays. The caller keeps each round's detected
site count in agreement with what the SLM server scores that round; a mismatch is surfaced (the
detector returns "" / [] on a stale/absent grid and the round is skipped), never a silent
off-by-one.

Pattern-write policy (per the user spec):
  * The LOADING (initial) phase is written once at scan start by :class:`SlmScanSession` (and
    re-written by ``ensure_held`` if the scan-long ``slm`` lock is ever lost). The atoms are then
    MOVED to the middle array by rearrange() round 1 and to the final array by round 2.
  * The MIDDLE and FINAL patterns are ASSUMED already on the SLM (produced by the rearrange calls);
    rounds 2 and 3 add NO SLMStep / no phase write -- they just cool + image as fast as possible.

Frame alignment / abort safety (the load-bearing invariant): a shot either stages ALL THREE frames
and publishes them with a single ``finish_shot``, or it is cancelled and NO partial triple is
persisted. Because the hardware runs all three Imag399 steps once a shot starts, every callback
that runs must CONSUME its frame even on a failing shot (grab-and-drain) so a straggler can't shift
img1/img2/img3 by one on the NEXT shot. A failed round re-publishes the captured frames under the
FAILING sentinel for LIVE DISPLAY ONLY (never persisted).

The BUILD path is unchanged from the byte port (steps/branches/pattern tags); only the deferred
callbacks -- which ``serialize()`` never runs -- carry the runtime logic.
"""

import os
import time

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep

import rearrange_runtime
from seq_capability import seq_capabilities

# Single-slot stashes for the current shot's already-captured frames. The finalizing callback
# (post_run) is a separate scope from the round callbacks that grabbed img1/img2, so those frames
# are stashed here for the failing-shot DISPLAY re-publish. Shots run strictly sequentially (one
# shot's callbacks at a time), so one slot each suffices; pre_run resets them per shot. Display-only
# use; never serialized.
_STASH = {"img1": None, "img2": None}


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def RearrangeCommSeq2(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_img2_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload, n_rounds=2

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 images the LOADING (dense load)
    # pattern; bseq2 the MIDDLE (round-1 target), bseq3 the FINAL (round-2 target). Each bseq's
    # cooling/imaging/VSLMServo resolve from ByPattern[that pattern]. Names from
    # rearrange_kwargs.extras.initial_pattern / middle_pattern / final_pattern (set by the scan);
    # absent -> scan-default / inherit. Tag each bseq HERE before its steps build. No-op when
    # ByPattern is empty.
    _init_pat = s.C.rearrange_kwargs.extras.initial_pattern("")
    if _init_pat:
        s.set_pattern(_init_pat)

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    ifEnhanced = s.C.rearrange_kwargs.extras.ifEnhanced(False)
    if ifEnhanced:
        s.add_step(BlueLACStep, s.C.LAC)
    else:
        s.add_step(LACStep, s.C.LAC)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # First Imag399 (img1, LOADING pattern).
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # Round 1: SLM rearrangement basic sequence (always entered). Imaged at the MIDDLE pattern.
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _mid_pat = s.C.rearrange_kwargs.extras.middle_pattern("")
    if _mid_pat:
        s2.set_pattern(_mid_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> rearrange round 1 (loading -> middle)

    # NI-DAQ keep-alive: reassert one V* channel (VMOTCoil at its current 0) so libnacs emits
    # non-None NI data for this bseq (Cool556/Imag399 touch only TTL+DDS). A physical no-op. See
    # the MATLAB original's note -- we must NOT InitStep between rounds (that zeroes VSLMservo and
    # loses the cooled atoms). ASSUME-WRITTEN policy: no SLMStep / no phase write here.
    s2.add('VMOTCoil', 0)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # Second Imag399 (img2, MIDDLE pattern).
    s2.add_step(Imag399Step, s.C.Imag399)

    # Round 2: second SLM-rearrangement basic sequence. Imaged at the FINAL pattern.
    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s3.set_pattern(_final_pat)

    s3.reg_before_bseq(hand_over_slm_2)   # img2 -> rearrange round 2 (middle -> final)

    # NI-DAQ keep-alive (same reason as s2). ASSUME-WRITTEN: no SLMStep / no phase write.
    s3.add('VMOTCoil', 0)

    s3.add_step(Cool556hXStep, s.C.Cool556)

    # Third Imag399 (img3, FINAL pattern).
    s3.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s3.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img3 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement (sticky -- no reset_params, n_rounds forced to 2), and reload_rearrange."""
    s1.G.rearrange_img1_ok = False
    s1.G.rearrange_img2_ok = False
    s1.G.rearrange_lock_ok = False
    _STASH["img1"] = None
    _STASH["img2"] = None

    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return                          # no rearrangement context -> nothing to do

    # Per-shot frame-buffer resync (identical to single-round). The Orca free-runs into a circular
    # buffer armed ONCE per scan; within a scan the three frames are grabbed one at a time, so a
    # single straggler (a late readout / a cancelled shot's leftover / a spurious trigger) offsets
    # the stream by one and img1/img2/img3 come out SHIFTED for the rest of the scan. pre_run runs
    # BEFORE this shot's Imag399 #1 triggers, so anything buffered now is stale: drop it and surface
    # a nonzero count (the previous shot(s) were misaligned).
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
                "flushed; img1/img2/img3 on the prior shot(s) were likely misaligned"
                % (_seq_id(s1), stale), kind="frame_desync", seq_id=_seq_id(s1))

    # The scan-long slm lock is mandatory: ensure_held re-acquires + rewrites the LOADING phase if
    # the lease lapsed. A failure raises -> the run errors. (Only the loading phase is ever written;
    # middle/final are produced by the rearrange calls and assumed on the SLM.)
    ctx.session.ensure_held()

    c = ctx.client
    # Per-shot compute (GPU) lock, blocking ~1 s. On miss, scrap THIS shot and retry rather than
    # racing rearrange() against another client.
    try:
        c.acquire_lock("compute", "rearrange compute (2 rounds)", timeout_s=10, block_timeout=1)
        s1.G.rearrange_lock_ok = True
    except Exception as err:  # noqa: BLE001 - contention / timeout -> cancel + retry
        ctx.log("[pre_run] compute lock acquire failed: %s -- cancelling seq for retry" % err)
        _safe(ctx.server, "seq_cancel")
        return

    try:
        c.health()                      # prewarm the connection
    except Exception:  # noqa: BLE001
        pass

    # Per-shot setup_rearrangement from rearrange_kwargs (the SCANNED params); NO reset_params so
    # everything else stays sticky from the dequeue-time setup call. Force n_rounds=2 unless the
    # scan set it explicitly (the server allocates per-round buffers/phases from this).
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs)
    args = rearrange_runtime.translate_zernike_zN(args)
    _ensure_extra(args, "n_rounds", 2)
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


def hand_over_slm(s1):
    """Round 1: read img1 (Imag399 #1, LOADING pattern), detect probs, rearrange(round 1), stage
    img1. Sets rearrange_img1_ok on success."""
    _STASH["img1"] = None
    _STASH["img2"] = None
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        _report_grab_fail(ctx, s1, "hand_over_slm", "img1", n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH["img1"] = img

    probs = ctx.detect_probs_for(_frame_pattern(ctx, 0), img)
    if not probs:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    _do_rearrange_round(ctx, s1, img, probs, "hand_over_slm", "rearrange_img1_ok")


def hand_over_slm_2(s1):
    """Round 2: read img2 (Imag399 #2, MIDDLE pattern), detect probs, rearrange(round 2), stage
    img2. Sets rearrange_img2_ok on success.

    Frame alignment: img2 is PHYSICALLY produced whether or not round 1 succeeded. If round 1
    failed we must still CONSUME img2 (grab-and-drain) so it can't straggle into the next shot's
    img1 -- then cancel. Never leave a produced frame buffered."""
    ctx = rearrange_runtime.context()
    if ctx is None:
        return
    if ctx.client is None or not s1.G.rearrange_lock_ok(False):
        return

    # If round 1 didn't rearrange (failing shot), still drain img2 to preserve the frame stream,
    # then cancel so the .h5 never gets a half triple.
    if not s1.G.rearrange_img1_ok(False):
        img2, ok2, _n2 = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.2)
        if ok2:
            _STASH["img2"] = img2       # keep for the post_run failing-display re-publish
        _safe(ctx.server, "cancel_shot")
        return

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        _report_grab_fail(ctx, s1, "hand_over_slm_2", "img2", n_seen)
        _safe(ctx.server, "cancel_shot")
        return
    _STASH["img2"] = img

    probs = ctx.detect_probs_for(_frame_pattern(ctx, 1), img)
    if not probs:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    _do_rearrange_round(ctx, s1, img, probs, "hand_over_slm_2", "rearrange_img2_ok")


def post_run(s1):
    """Finalize: read img3 (Imag399 #3, FINAL pattern), update_rearrange(bits3), stage img3 +
    finish; release the compute lock and keepalive the scan-long slm lock.

    img3 is PHYSICALLY produced regardless of prior-round success, so it is ALWAYS grabbed (draining
    it) -- then published as a full triple only when BOTH rounds succeeded, else re-published for
    DISPLAY ONLY (never persisted, keeping the .h5 in aligned img1/img2/img3 triples)."""
    ctx = rearrange_runtime.context()
    if ctx is None:
        return

    try:
        both_ok = bool(s1.G.rearrange_img1_ok(False)) and bool(s1.G.rearrange_img2_ok(False))
        img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)

        if not both_ok:
            # A failing shot: drain img3 (done above) and re-publish whatever was captured for LIVE
            # DISPLAY ONLY under the failing sentinel; NEVER persist a partial triple. The
            # shot-health chip is already red via the earlier record_error.
            ctx.publish_failed_shot(
                [_STASH.get("img1"), _STASH.get("img2"), img if ok else None], _seq_id(s1))
            return

        if not ok:
            # Both rounds succeeded but img3 was lost -> show the two captured frames + "no data"
            # for img3 (display only); do NOT persist a half triple. Surplus (desync) vs timeout
            # surfaced distinctly; either way the next shot's pre_run flush resyncs.
            if n_seen >= 2:
                ctx.record_error(
                    "[post_run] seq %d: img3 grab saw %d frames (desynced buffer) -- drained; "
                    "display-only img1/img2" % (_seq_id(s1), n_seen),
                    kind="frame_desync", seq_id=_seq_id(s1))
            else:
                ctx.record_error(
                    "[post_run] seq %d: img3 unavailable (timeout, 0 frames) -- display-only "
                    "img1/img2" % _seq_id(s1), kind="frame_timeout", seq_id=_seq_id(s1))
            ctx.publish_failed_shot([_STASH.get("img1"), _STASH.get("img2")], _seq_id(s1))
            return

        # Success: detect the final image (FINAL pattern) + log results, then stage the full triple
        # (img1, img2, img3) and publish it as ONE shot. img1/img2 were staged in their rounds;
        # stage img3 here and finish. FIFO ordering keeps the triple aligned.
        bits3 = ctx.detect_bits_for(_frame_pattern(ctx, 2), img)
        if bits3 and ctx.client is not None:
            try:
                ctx.client.update_rearrange(bits3, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
            except Exception as err:  # noqa: BLE001
                ctx.log("[post_run] update_rearrange failed: %s" % err)
        _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))   # img3
        _safe(ctx.server, "finish_shot")                                  # publish the triple
        ctx.record_ok()                 # healthy shot -> clears the "failing" banner on recovery
    finally:
        # Release the per-shot compute lock; keepalive (renew) the scan-long slm lock. The
        # scan-long session owns slm and releases it at scan end -- never release slm here.
        if s1.G.rearrange_lock_ok(False) and ctx.client is not None:
            try:
                ctx.client.release_lock("compute")
            except Exception as err:  # noqa: BLE001
                ctx.log("[post_run] compute lock release failed: %s" % err)
        if ctx.session is not None:
            ctx.session.keepalive()
        _STASH["img1"] = None
        _STASH["img2"] = None


# =========================================================================== #
# shared round logic
# =========================================================================== #
def _do_rearrange_round(ctx, s1, img, probs, tag, ok_flag):
    """Stage ``img`` (async) and call rearrange(probs) for one round; set ``s1.G.<ok_flag>`` on
    success. Mirrors RearrangeCommSeq.hand_over_slm's stage-then-rearrange-then-reconcile path,
    including the abort-tail cleanup (drop the staged frame + the phantom server ledger row on a
    downstream failure) so a failed round never leaves a half-staged shot or a desynced ledger."""
    c = ctx.client
    runid = _runid_kwargs(ctx.scan_id, _seq_id(s1))
    # Hand the frame to the ExptServer persister (async, FIFO worker) so the encode+store overlaps
    # rearrange()'s SLM round-trip + the next bseq's hardware instead of blocking the held-atom
    # critical path. A persist failure is handled by the worker (it cancels the shot); the live
    # signal we gate on is the rearrange() result.
    _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    try:
        r = c.rearrange(probs, **runid)
        if isinstance(r, dict) and r.get("handoff_idle"):
            ctx.log("[%s] seq %d: server idle; cancelling shot, waiting 1 s" % (tag, _seq_id(s1)))
            _safe(ctx.server, "cancel_shot")   # ordered after the staged frame -> drops it
            setattr(s1.G, ok_flag, False)
            time.sleep(1.0)
            return
        setattr(s1.G, ok_flag, True)
        if isinstance(r, dict) and not r.get("ok", True):
            ctx.record_error("[%s] rearrange returned ok=false" % tag,
                             kind="rearrange", seq_id=_seq_id(s1))
    except Exception as err:  # noqa: BLE001
        # rearrange() may have committed server-side before a downstream failure; drop the phantom
        # diag ledger row so the SLM ledger stays aligned with the lab seq_ids.
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
# helpers
# =========================================================================== #
def _frame_pattern(ctx, idx):
    """The pattern NAME to detect camera frame ``idx`` against (0=loading, 1=middle, 2=final), from
    ``ctx.frame_patterns`` when the scan declared it; else None so the detector uses the frame-0 /
    day-folder calibration. Out-of-range -> None (be permissive; the detector falls back)."""
    fp = getattr(ctx, "frame_patterns", None)
    if fp and 0 <= idx < len(fp):
        return fp[idx]
    return None


def _report_grab_fail(ctx, s1, tag, which, n_seen):
    """Record a frame-grab failure with the desync-vs-timeout distinction (identical wording shape
    to single-round) so the dashboard shot-health chip lights correctly."""
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
    ensure_default_extra). Creates the extras dict if absent so n_rounds always reaches the
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
    """Lab-PC run-ID propagation for the SLM diag ledger. ``YB_SLM_DISABLE_RUNID=1`` -> {} (legacy
    body shape) as an emergency rollback, mirroring RearrangeCommSeq.py."""
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
