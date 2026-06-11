"""RearrangeCommSeq.py -- port of ``matlab_new/YbSeqs/RearrangeCommSeq.m`` (single-round).

The seq BUILD path (the byte-producing part) is the faithful transliteration: a nargin-1 seq with
a SECOND basic sequence for the SLM-rearrangement handoff. ``serialize()`` never runs the deferred
callbacks, so the build path is byte-identical to MATLAB.

The deferred callbacks (``pre_run`` / ``hand_over_slm`` / ``post_run``) carry the per-shot
rearrangement logic, mirroring RearrangeCommSeq.m. They reach the camera / ExptServer / scan-long
SLM session through :mod:`rearrange_runtime` (the pyctrl analog of MATLAB's base workspace, since a
pyctrl callback only gets ``s1``). The scan-long ``slm`` HARDWARE lock + loading-phase write are
owned by :class:`SlmScanSession` for the WHOLE scan (set up by the runner at dequeue); per shot we
take only the ``compute`` (GPU) lock for the rearrange window. The server's ``rearrange_actual``
requires the caller hold ``slm`` -- satisfied by the scan-long hold under the SAME client id.

Per-shot flow (mirrors the user spec / MATLAB):
  pre_run       -- ensure the scan-long slm lock is held; grab the compute lock (cancel+retry on
                   miss); per-shot setup_rearrangement (no reset_params -> sticky); reload_rearrange.
  hand_over_slm -- read img1, detect bits, rearrange(bits), store img1 (between Imag399 #1 and #2).
  post_run      -- read img2, update_rearrange(bits2), store img2 + seq_finish; release the compute
                   lock; keepalive the scan-long slm lock (one heartbeat, no rewrite).
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

# Single-slot stash for the current shot's img1 frame. hand_over_slm captures img1 but post_run
# (a separate callback) is where a failing shot is finalized, and img1 is out of scope there --
# so it's stashed here. Shots run strictly sequentially (one shot's callbacks at a time), so one
# slot suffices; hand_over_slm resets it each shot. Display-only use; never serialized.
_LAST_IMG1 = {"frame": None}


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoff)
def RearrangeCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.rearrange_img1_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # connect, compute lock, per-shot setup + reload

    s.add_step(InitStep, s.C.Init)
    s.add_step(BlueMOTStep, s.C.BlueMOT)
    s.add_step(SLMStep, s.C.SLM)
    s.add_step(GreenMOTStep, s.C.GreenMOT)

    # LAC vs BlueLAC chosen at build time; default (no rearrange_kwargs) -> LAC.
    ifEnhanced = s.C.rearrange_kwargs.extras.ifEnhanced(False)
    if ifEnhanced:
        s.add_step(BlueLACStep, s.C.LAC)
    else:
        s.add_step(LACStep, s.C.LAC)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # First Imag399.
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # Second part: SLM-rearrangement basic sequence (always entered).
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq2 images the FINAL (rearranged
    # target) pattern, so its cooling/imaging/VSLMServo resolve from ByPattern[final]; bseq1 keeps
    # the scan-default (initial) pattern set by the runner. Name from
    # rearrange_kwargs.extras.final_pattern (set by the scan); absent -> bseq2 inherits initial
    # (and the whole thing is a no-op when ByPattern is empty).
    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(hand_over_slm)  # img1 -> bits -> rearrange

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # Second Imag399.
    s2.add_step(Imag399Step, s.C.Imag399)

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img2 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement (sticky -- no reset_params), and reload_rearrange."""
    s1.G.rearrange_img1_ok = False
    s1.G.rearrange_lock_ok = False

    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return                          # no rearrangement context -> nothing to do

    # Per-shot frame-buffer resync. The Orca free-runs into a circular buffer armed ONCE per scan
    # (runner.py: flush + start_video); the run loop only flushes it at scan start. Within a scan
    # img1 (hand_over_slm) and img2 (post_run) are grabbed one frame at a time, so a single
    # straggler -- a frame whose readout landed after its grab timed out, a cancelled pair, or a
    # spurious trigger -- offsets the stream by one and img1/img2 come out SWAPPED (img1 shows the
    # prior shot's sparse post-rearrangement frame, img2 shows this shot's full loading frame) for
    # the rest of the scan. pre_run runs reg_before_start, BEFORE this shot's Imag399 #1 triggers,
    # so anything in the buffer now is stale: drop it and surface a nonzero count (the previous
    # shot(s) were misaligned). This is the per-shot analog of the scan-start flush.
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
                "flushed; img1/img2 on the prior shot(s) were likely misaligned (flipped)"
                % (_seq_id(s1), stale), kind="frame_desync", seq_id=_seq_id(s1))

    # The scan-long slm lock is mandatory: ensure_held re-acquires + rewrites the WGS phase if the
    # 10 s lease lapsed (a shot longer than the lease). A failure raises -> the run errors.
    ctx.session.ensure_held()

    c = ctx.client
    # Per-shot compute (GPU) lock, blocking ~1 s. On miss, scrap THIS shot and retry rather than
    # racing rearrange() against another client.
    try:
        c.acquire_lock("compute", "rearrange compute", timeout_s=10, block_timeout=1)
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
    # everything else stays sticky from the initial (dequeue-time) setup call.
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs)
    args = rearrange_runtime.translate_zernike_zN(args)
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
    """Read img1 (Imag399 #1), detect bits, rearrange(bits), and store img1."""
    # Reset the per-shot img1 stash (post_run re-publishes it for DISPLAY on a failing shot).
    _LAST_IMG1["frame"] = None
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        if n_seen >= 2:
            ctx.record_error(
                "[hand_over_slm] seq %d: img1 grab saw %d frames (desynced buffer) -- drained + "
                "cancelling" % (_seq_id(s1), n_seen), kind="frame_desync", seq_id=_seq_id(s1))
        else:
            ctx.record_error(
                "[hand_over_slm] seq %d: img1 unavailable (timeout, 0 frames) -- cancelling"
                % _seq_id(s1), kind="frame_timeout", seq_id=_seq_id(s1))
        _safe(ctx.server, "cancel_shot")
        return
    # Stash img1 so post_run can re-publish it for DISPLAY ONLY if the shot fails (img1 is captured
    # here but isn't in post_run's scope). Cleared above each shot; success path persists it normally.
    _LAST_IMG1["frame"] = img

    bits = ctx.detect_bits(img)
    if not bits:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    c = ctx.client
    runid = _runid_kwargs(ctx.scan_id, _seq_id(s1))
    # Hand img1 to the ExptServer persister (async): the encode + store runs on the server's single
    # FIFO worker, so it overlaps rearrange()'s SLM round-trip AND the next basic sequence's
    # hardware instead of adding ~100 ms to the held-atom critical path. FIFO + single worker keep
    # (img1 -> img2 -> finish) ordered and temp_imgs single-writer; post_run enqueues img2 +
    # finish_shot. A persist failure is handled by the worker (it cancels the shot), so there is no
    # synchronous result to reconcile here -- the live signal we gate on is the rearrange() result.
    _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))
    try:
        r = c.rearrange(bits, **runid)
        if isinstance(r, dict) and r.get("handoff_idle"):
            ctx.log("[hand_over_slm] seq %d: server idle; cancelling shot, waiting 1 s"
                    % _seq_id(s1))
            _safe(ctx.server, "cancel_shot")   # ordered after the staged img1 -> drops it
            s1.G.rearrange_img1_ok = False
            time.sleep(1.0)
            return
        s1.G.rearrange_img1_ok = True
        if isinstance(r, dict) and not r.get("ok", True):
            ctx.record_error("[hand_over_slm] rearrange returned ok=false",
                             kind="rearrange", seq_id=_seq_id(s1))
        else:
            ctx.record_ok()             # healthy shot -> clears the "failing" banner on recovery
    except Exception as err:  # noqa: BLE001
        # rearrange() may have committed server-side before a downstream failure; drop the
        # phantom diag ledger row so the SLM ledger stays aligned with the lab seq_ids.
        try:
            if runid:
                c.cancel_last_shot(**runid)
        except Exception:  # noqa: BLE001 - older server lacks /slm/cancel_last; non-fatal
            pass
        _safe(ctx.server, "cancel_shot")   # drop the staged img1 (ordered after it)
        ctx.record_error("[hand_over_slm] rearrange call failed: %s" % err,
                         kind="rearrange", seq_id=_seq_id(s1))


def post_run(s1):
    """Read img2 (Imag399 #2), update_rearrange(bits2), store img2 + finish; release the compute
    lock and keepalive the scan-long slm lock."""
    ctx = rearrange_runtime.context()
    if ctx is None:
        return

    # Pair alignment gates on the rearrange() result (rearrange_img1_ok), the live signal. img1's
    # persist is fire-and-forget on the server's FIFO worker; a persist failure is handled there
    # (the worker cancels the shot), and FIFO ordering means img2 + finish_shot below queue AFTER
    # the staged img1, so no join is needed here.
    try:
        if not s1.G.rearrange_img1_ok(False):
            # img1 not rearranged (failing shot) -> the .h5 must NOT get a half pair, but we still
            # want the captured frames to flash by on the live view. Drain the final frame (img2)
            # and re-publish [img1, img2] for DISPLAY ONLY under the failing sentinel (no persist);
            # the shot-health chip already shows "failing" via the earlier record_error.
            img2, ok2, _n2 = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.2)
            ctx.publish_failed_shot(
                [_LAST_IMG1.get("frame"), img2 if ok2 else None], _seq_id(s1))
        else:
            img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
            if not ok:
                # rearrange succeeded but img2 was lost -> show img1 + "no data" for img2 (display
                # only), rather than freezing the live view. Don't persist a half pair. Surface a
                # surplus (desync) distinctly from a readout-latency timeout; either way the next
                # shot's pre_run flush resyncs the buffer.
                if n_seen >= 2:
                    ctx.record_error(
                        "[post_run] seq %d: img2 grab saw %d frames (desynced buffer) -- drained; "
                        "display-only img1" % (_seq_id(s1), n_seen),
                        kind="frame_desync", seq_id=_seq_id(s1))
                else:
                    ctx.record_error(
                        "[post_run] seq %d: img2 unavailable (timeout, 0 frames) -- display-only "
                        "img1" % _seq_id(s1), kind="frame_timeout", seq_id=_seq_id(s1))
                ctx.publish_failed_shot([_LAST_IMG1.get("frame")], _seq_id(s1))
            else:
                bits2 = ctx.detect_bits(img)
                if bits2 and ctx.client is not None:
                    try:
                        ctx.client.update_rearrange(
                            bits2, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
                    except Exception as err:  # noqa: BLE001
                        ctx.log("[post_run] update_rearrange failed: %s" % err)
                _safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))   # img2
                _safe(ctx.server, "finish_shot")                                  # publish the pair
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


# =========================================================================== #
# helpers
# =========================================================================== #
def _seq_id(s1):
    try:
        return int(s1.G.seq_id(1))
    except Exception:  # noqa: BLE001
        return 1


def _runid_kwargs(scan_id, seq_id):
    """Lab-PC run-ID propagation for the SLM diag ledger. ``YB_SLM_DISABLE_RUNID=1`` -> {} (legacy
    body shape) as an emergency rollback, mirroring RearrangeCommSeq.m::build_runid_opts."""
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
