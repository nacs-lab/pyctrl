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
        ctx.log("[pre_run] setup_rearrangement failed: %s" % err)
    try:
        c.reload_rearrange()
    except Exception as err:  # noqa: BLE001
        ctx.log("[pre_run] reload_rearrange failed: %s" % err)


def hand_over_slm(s1):
    """Read img1 (Imag399 #1), detect bits, rearrange(bits), and store img1."""
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    img, ok = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        ctx.log("[hand_over_slm] seq %d: img1 unavailable; cancelling" % _seq_id(s1))
        _safe(ctx.server, "seq_cancel")
        return

    bits = ctx.detect_bits(img)
    if not bits:
        return                          # calibration mismatch -> don't rearrange on a stale grid

    c = ctx.client
    runid = _runid_kwargs(ctx.scan_id, _seq_id(s1))
    try:
        r = c.rearrange(bits, **runid)
        if isinstance(r, dict) and r.get("handoff_idle"):
            ctx.log("[hand_over_slm] seq %d: server idle; cancelling shot, waiting 1 s"
                    % _seq_id(s1))
            _safe(ctx.server, "seq_cancel")
            s1.G.rearrange_img1_ok = False
            time.sleep(1.0)
            return
        _store_img(ctx, img, _seq_id(s1))
        s1.G.rearrange_img1_ok = True
        if isinstance(r, dict) and not r.get("ok", True):
            ctx.log("[hand_over_slm] rearrange returned ok=false")
    except Exception as err:  # noqa: BLE001
        # rearrange() may have committed server-side before a downstream failure; drop the
        # phantom diag ledger row so the SLM ledger stays aligned with the lab seq_ids.
        try:
            if runid:
                c.cancel_last_shot(**runid)
        except Exception:  # noqa: BLE001 - older server lacks /slm/cancel_last; non-fatal
            pass
        ctx.log("[hand_over_slm] rearrange call failed: %s" % err)


def post_run(s1):
    """Read img2 (Imag399 #2), update_rearrange(bits2), store img2 + finish; release the compute
    lock and keepalive the scan-long slm lock."""
    ctx = rearrange_runtime.context()
    if ctx is None:
        return

    try:
        if not s1.G.rearrange_img1_ok(False):
            # hand_over_slm did not store img1 -> drain the final frame + cancel so the .h5 stays
            # in aligned (img1, img2) pairs.
            rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.2)
            _safe(ctx.server, "seq_cancel")
        else:
            img, ok = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
            if not ok:
                ctx.log("[post_run] seq %d: img2 unavailable; cancelling" % _seq_id(s1))
                _safe(ctx.server, "seq_cancel")
            else:
                bits2 = ctx.detect_bits(img)
                if bits2 and ctx.client is not None:
                    try:
                        ctx.client.update_rearrange(
                            bits2, **_runid_kwargs(ctx.scan_id, _seq_id(s1)))
                    except Exception as err:  # noqa: BLE001
                        ctx.log("[post_run] update_rearrange failed: %s" % err)
                _store_img(ctx, img, _seq_id(s1))
                _safe(ctx.server, "seq_finish")
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
def _store_img(ctx, img, seq_id):
    """Store one frame via the ExptServer in the column-major store_imgs wire format."""
    if ctx.server is None:
        return
    try:
        from devices.orca import to_store_array
        ctx.server.store_imgs(to_store_array(img), ctx.scan_id, seq_id)
    except Exception as err:  # noqa: BLE001
        ctx.log("[store_img] failed: %s" % err)


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
