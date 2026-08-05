"""Rearrange3DFocusWalkCommSeq.py -- two-layer (3-D) rearrangement with a mid-shot AXIAL FOCUS WALK.

New seq (2026-07-29). Ground truth for the build path + per-shot machinery is
``RearrangeCommSeq2.py`` (three camera frames, two mid-shot SLM-server handoffs); what changes is
WHAT the two handoffs do and WHAT bit vector they post.

The experiment (see ``YbScans/SLMRearrangement3DFocusWalkScan.py`` for the numbers): a 2-layer
xy-ALIGNED tweezer array whose layers are FAR apart axially (20 um for
``phase/2x11x11_5um_z20um.pt``, i.e. z4 = +-12.53 rad -- many Rayleigh ranges). Only ONE layer can
be imaged at a time; the other contributes a diffuse halo, not resolvable spots. So the loaded
occupancy of BOTH layers is read out in TWO images separated by a phase-only axial walk of the
WHOLE hologram, and only then are the atoms of both layers rearranged into the single (now
in-focus) FRONT layer:

  Init -> MOT -> SLM -> GreenMOT -> LAC -> Imag399 (#1, BACK layer in focus)
       -> Cool -> pingponggrating depth walk (-2*z_half rad, phase-only, no model)
       -> Imag399 (#2, FRONT layer now in focus, same camera pixels)
       -> Cool -> 3-D rearrange(both layers -> FRONT layer) -> Imag399 (#3, final) -> Init

Three camera frames per shot; ONE rearrangement round (the walk is not a round -- the scan declares
``extras.n_rounds = 1`` with ``NumImages = 3``, the hybrid-seq shape RearrangeSTIRAPScan also uses).

WHY THE WALK WORKS, AND WHY THE LAYERS SHARE ONE DETECTION GRID
  The two layers are xy-ALIGNED: the server's 3-D grid derivation returns 242 sites (121 back then
  121 front, layer-major in the declared ``planes_z_rad`` order) whose two halves are the SAME xy to
  <0.2 knm px (verified 2026-07-29 via /eval on the live server). Walking the array axially by
  exactly the layer separation puts the front layer at the plane the back layer occupied, so
  frame #2's atoms land on the SAME camera pixels as frame #1's. One 121-site detection pattern
  (the phase declared with a SINGLE plane, ``planes_z_rad=[z_back]``) therefore serves all three
  frames -- one registry record, one threshold set, and the affine's linear-defocus (dz) term is
  the SAME (and correct) for every frame at any loading carrier, because the walk equals the
  separation. That single-plane record's site order is byte-identical to the matching half of the
  242-site server grid (verified: max|d| = 0.0), which is what makes the splice below sound.

THE POSTED BIT VECTORS (the "store the back and front loading patterns" part)
  ``rearrange(bits)`` scores ``bits[i]`` against ``init_grid[i]`` over all 242 sites, so each call
  posts a COMPOSED vector:

    walk call  (frame #1) : [back probs (121)] + [0]*121        (bits unused by pingponggrating)
    3-D call   (frame #2) : [back probs (121)] + [front probs (121)]
    results    (frame #3) : "0"*121            + front bits (121)   (the back layer is emptied)

  The back half is measured BEFORE the walk and carried in this module's per-shot store; the front
  half is measured after it. Both are detected with the same pattern/grid, so index i of either
  half is the same physical xy at its own layer.

THE TWO PER-SHOT SETUPS (no server-side code change needed)
  The two handoffs need DIFFERENT protocols, and ``protocol`` is server-sticky (not per-call), so
  this seq pushes ``setup_rearrangement`` twice per shot:
    * ``pre_run``           -> ``rearrange_kwargs``  (the WALK: protocol pingponggrating, its
                               nsteps / step_period_ms / depth / step_size / return_trip ...).
    * before the 3-D call  -> ``rearrange_kwargs2`` (protocol rearrange, its own nsteps/period,
                               ``pattern='front_layer'``, and -- the load-bearing bit -- a fresh
                               ``final_phase`` + ``skip_grid_derive=True`` + a NEW
                               ``extras.loading_zernike``, which re-writes ONLY the cached WGS
                               final_phase (the rearrange bookend) at the WALKED carrier while
                               leaving the initial/loading write phase and both derived grids
                               untouched. The bookend then matches the walk's last frame up to an
                               optically inert global piston. Without it the bookend would snap the
                               array back by the full walk (20 um) in one frame and the final image
                               would be out of focus.)
  Neither setup passes ``model_filename`` or ``reset_params``, so the run id does not bump and
  every other param stays sticky from the dequeue warmup setup.

Frame alignment / abort safety is the unchanged invariant (enforced here the same way
``rearrange_callbacks`` does it): a shot either stages ALL THREE frames and publishes them with a
single ``finish_shot``, or it is cancelled and NO partial triple is persisted. Every callback that
runs CONSUMES its frame even on a failing shot so a straggler cannot shift the frame stream on the
next shot; a failed shot's captured frames are re-published under the FAILING sentinel for LIVE
DISPLAY ONLY.

WALK-ONLY MODE (``rearrange_kwargs.extras.walk_only`` truthy, or the scan's --walk-only): do the
load -> image -> walk -> image -> image cycle with NO 3-D rearrangement and NO detection
requirement. Use it (a) as the FIRST run of a new array, to bootstrap the pattern registry record
+ per-site thresholds that the detection above needs, and (b) as the walk's own
survival/heating measurement (frame #3 vs frame #2 with nothing but the walk in between).

The BUILD path is a plain 3-bseq transliteration of RearrangeCommSeq2's (same steps, same
NI keep-alives, same pattern tags); only the deferred callbacks -- which ``serialize()`` never runs
-- carry the logic above. Nothing here touches the byte path.
"""

from BlueLACStep import BlueLACStep
from BlueMOTStep import BlueMOTStep
from consts import Consts
from Cool556hXStep import Cool556hXStep
from GreenMOTStep import GreenMOTStep
from Imag399AmpStep import Imag399AmpStep
from Imag399Step import Imag399Step
from InitStep import InitStep
from LACStep import LACStep
from SLMStep import SLMStep

import rearrange_callbacks
import rearrange_runtime
from seq_capability import seq_capabilities


# Per-shot store for the BACK-layer occupancy measured on frame #1 (needed again on frame #2 to
# compose the full 242-site vector). Shots run strictly sequentially -- one shot's callbacks at a
# time -- so one slot suffices; reset at frame #1. Display/runtime only, never serialized.
_SHOT = {}


# --------------------------------------------------------------------------- #
# Per-frame 399 imaging brightness -- same mechanism + same policy as RearrangeCommSeq2 (the 399
# imaging-power PID locks ONCE at the ROOT BlueMOTStep and HOLDS, so each frame's only brightness
# knob is its own DDS amps; see that file's long policy block and
# yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md). Names here:
#     extras.MidImgAmp1 / MidImgAmp2  -> img2 (FRONT layer, post-walk) only
#     extras.FinImgAmp1 / FinImgAmp2  -> img3 (final, post-rearrangement) only
# Absent -> the frame's own ByPattern/base Amp1/Amp2 and a build byte-identical to the no-knob seq.
# --------------------------------------------------------------------------- #
def _extras_num(s, name, default):
    """``rearrange_kwargs.extras.<name>`` as a float; ``default`` when absent/unresolvable."""
    try:
        return float(getattr(s.C.rearrange_kwargs.extras, name)(default))
    except Exception:  # noqa: BLE001 - absent/odd extras -> default
        return float(default)


def _img_amps(s, n1, n2):
    """The (Amp1, Amp2) override for ONE image, or ``None`` when the scan set NEITHER extra."""
    a1 = _extras_num(s, n1, -1.0)
    a2 = _extras_num(s, n2, -1.0)
    return None if (a1 < 0 and a2 < 0) else (a1, a2)


def _add_imag399(sb, g, amps):
    """This bseq's Imag399: the EXACT pre-existing ``Imag399Step`` call when ``amps`` is None,
    else the amp-overridable twin (identical body, ``a1``/``a2`` replacing g.Amp1/2)."""
    if amps is None:
        return sb.add_step(Imag399Step, g)
    return sb.add_step(Imag399AmpStep, g, amps[0], amps[1])


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def Rearrange3DFocusWalkCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.focus_walk_ok = False
    s.G.rearrange_front_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # locks + per-shot setup (the WALK params) + reload

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 images the BACK layer, bseq2 the
    # FRONT layer (post-walk), bseq3 the rearranged FRONT array. Names from
    # rearrange_kwargs.extras.initial_pattern / middle_pattern / final_pattern; absent -> inherit.
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

    # First Imag399 (img1, BACK layer in focus).
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during the walk + the rearrangement.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # ---- bseq2: the AXIAL FOCUS WALK happens in its before-bseq callback; image the FRONT layer.
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _mid_pat = s.C.rearrange_kwargs.extras.middle_pattern("")
    if _mid_pat:
        s2.set_pattern(_mid_pat)

    s2.reg_before_bseq(walk_to_front)  # img1 -> back-layer probs -> pingponggrating depth walk

    # NI-DAQ keep-alive: re-assert one V* channel (VMOTCoil at its current 0) so libnacs emits
    # non-None NI data for this bseq (Cool556/Imag399 touch only TTL+DDS). Physical no-op. We must
    # NOT InitStep between the images (that zeroes VSLMservo and loses the atoms), and there is no
    # SLMStep / phase write here: the phase on the panel is the WALKED hologram the server left.
    s2.add('VMOTCoil', 0)

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # NI-DAQ minimum-buffer guard: DAQmx FINITE AO rejects a 1-sample buffer (error -200077), and
    # the keep-alive above is this bseq's only NI update time -- re-asserting it after the cooling
    # step gives a second update time (>= 2 samples). Physical no-op.
    s2.add('VMOTCoil', 0)

    # Second Imag399 (img2, FRONT layer in focus after the walk).
    _add_imag399(s2, s.C.Imag399, _img_amps(s, "MidImgAmp1", "MidImgAmp2"))

    # ---- bseq3: the 3-D rearrangement happens in its before-bseq callback; image the result.
    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s3.set_pattern(_final_pat)

    s3.reg_before_bseq(rearrange_to_front)   # img2 -> front probs -> setup#2 -> 3-D rearrange

    s3.add('VMOTCoil', 0)                    # NI keep-alive (same reason as s2)

    s3.add_step(Cool556hXStep, s.C.Cool556)

    s3.add('VMOTCoil', 0)                    # NI minimum-buffer guard (same as s2)

    # Third Imag399 (img3, FINAL rearranged front-layer array).
    _add_imag399(s3, s.C.Imag399, _img_amps(s, "FinImgAmp1", "FinImgAmp2"))

    # Initialisation again (shut down for safety).
    s3.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # img3 -> update_rearrange; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push the per-shot
    setup_rearrangement for the WALK (``rearrange_kwargs``: protocol pingponggrating + its
    nsteps/step_size/...; sticky, no reset_params), and reload_rearrange (writes the LOADING
    hologram at the loading carrier = the start of this shot's walk)."""
    _SHOT.clear()
    rearrange_callbacks.pre_run(
        s1, flags=("focus_walk_ok", "rearrange_front_ok"),
        lock_desc="3-D focus-walk rearrange compute", frames_label="img1/img2/img3")


def walk_to_front(s1):
    """Frame #1 -> BACK-layer occupancy, then the phase-only AXIAL WALK that brings the FRONT
    layer into the camera focus plane.

    The walk is a ``rearrange()`` call whose sticky protocol is ``pingponggrating`` (depth mode,
    ``return=False`` so the panel RESTS on the fully-walked frame). Its bits are unused by that
    protocol but must still be ``len(init_grid)`` long, so the back half is posted for the record
    and the front half is zero-filled. Sets ``focus_walk_ok`` on a successful walk."""
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    _SHOT.clear()
    rearrange_callbacks.reset_stash()   # first frame of the shot -> fresh stash

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        _grab_fail(ctx, s1, "walk_to_front", 0, n_seen)
        return
    rearrange_callbacks.stash_frame(0, img)

    walk_only = _walk_only(s1)
    n_plane = _n_per_plane(s1)

    # BACK-layer per-site probabilities (this frame's own pattern grid = the single-plane record).
    probs_back = ctx.detect_probs_for(_frame_pattern(ctx, 0), img)
    if probs_back:
        _SHOT["back"] = list(probs_back)
        n_plane = len(probs_back)
    elif not walk_only:
        # No calibration yet / stale grid: still WALK (frames #2/#3 must be in focus and the frame
        # stream must stay aligned) but the shot cannot be scored -> flag it failing.
        ctx.record_error(
            "[walk_to_front] seq %d: no per-site probs for pattern %r -- walking anyway, but the "
            "shot cannot be scored (run --walk-only once to bootstrap the pattern thresholds)"
            % (_seq_id(s1), _frame_pattern(ctx, 0)), kind="detect", seq_id=_seq_id(s1))

    _stage(ctx, s1, img)

    vec = _compose(_SHOT.get("back"), None, n_plane)
    if not _server_round(ctx, s1, vec, "walk_to_front", "focus_walk_ok"):
        return
    # A scored shot needs the back half; walk-only does not.
    if not walk_only and "back" not in _SHOT:
        s1.G.focus_walk_ok = False


def rearrange_to_front(s1):
    """Frame #2 -> FRONT-layer occupancy, then the 3-D rearrangement of BOTH layers into the
    (now in-focus) front layer.

    Pushes the second per-shot ``setup_rearrangement`` first (``rearrange_kwargs2``: protocol
    ``rearrange`` + ``pattern='front_layer'`` + the bookend re-write at the walked carrier -- see
    the module docstring), then posts the composed 242-site vector. Walk-only mode stops after
    staging the frame. Sets ``rearrange_front_ok``."""
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return

    # The walk failed: still CONSUME frame #2 (it is physically produced), keep it for the
    # display re-publish, and cancel so no partial triple persists.
    if not s1.G.focus_walk_ok(False):
        img, ok, _n = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.2)
        rearrange_callbacks.stash_frame(1, img if ok else None)
        _cancel(ctx)
        return

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        _grab_fail(ctx, s1, "rearrange_to_front", 1, n_seen)
        return
    rearrange_callbacks.stash_frame(1, img)

    walk_only = _walk_only(s1)
    probs_front = ctx.detect_probs_for(_frame_pattern(ctx, 1), img)
    _stage(ctx, s1, img)

    if walk_only:
        s1.G.rearrange_front_ok = True         # walk-only: the walk IS the whole shot
        return

    probs_back = _SHOT.get("back")
    if not probs_back or not probs_front:
        ctx.record_error(
            "[rearrange_to_front] seq %d: missing %s-layer probs -- no rearrangement this shot"
            % (_seq_id(s1), "back" if not probs_back else "front"),
            kind="detect", seq_id=_seq_id(s1))
        _cancel(ctx)
        return

    # ---- second per-shot setup: protocol rearrange + the bookend at the WALKED carrier --------
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs2)
    args = rearrange_runtime.translate_zernike_zN(args)
    if not args:
        ctx.record_error(
            "[rearrange_to_front] seq %d: rearrange_kwargs2 is empty -- the scan must declare the "
            "3-D rearrangement setup (protocol/nsteps/final_phase/loading_zernike)" % _seq_id(s1),
            kind="setup_rearrangement", seq_id=_seq_id(s1))
        _cancel(ctx)
        return
    args.setdefault("client_scan_id", str(ctx.scan_id))
    try:
        ctx.client.setup_rearrangement(**args)
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[rearrange_to_front] setup_rearrangement failed: %s" % err,
                         kind="setup_rearrangement", seq_id=_seq_id(s1))
        _cancel(ctx)
        return

    vec = _compose(probs_back, probs_front, len(probs_front))
    _server_round(ctx, s1, vec, "rearrange_to_front", "rearrange_front_ok", stage_img=None)


def post_run(s1):
    """Frame #3 -> the final front-layer occupancy: report it to the server (padded with '0' over
    the emptied BACK-layer half of the 242-site grid), publish the aligned triple with one
    ``finish_shot``, release the compute lock, keepalive the scan-long slm lock."""
    walk_only = _walk_only(s1)
    rearrange_callbacks.finalize(
        s1, round_flags=("focus_walk_ok", "rearrange_front_ok"), final_frame_idx=2,
        tag="post_run", update=not walk_only, use_frame_pattern=True, record_ok=True,
        bits_fn=None if walk_only else _pad_back_zeros)


# =========================================================================== #
# helpers
# =========================================================================== #
def _pad_back_zeros(bits):
    """Front-layer bitstring -> the full 242-site vector: '0' for every BACK-layer site (the
    rearrangement moved those atoms out) followed by the measured front-layer bits. The halves are
    layer-major in the declared ``planes_z_rad`` order (back first), matching the server grid."""
    return "0" * len(bits) + bits


def _compose(back, front, n_plane):
    """The full ``len(init_grid)`` == 2 * ``n_plane`` probability vector: back half then front
    half, each ``None`` half zero-filled (the walk call has no front measurement yet; a missing
    back half only happens on the un-scored walk-only path)."""
    n = int(n_plane)
    b = list(back) if back else [0.0] * n
    f = list(front) if front else [0.0] * n
    return [float(v) for v in b] + [float(v) for v in f]


def _server_round(ctx, s1, vec, tag, ok_flag, stage_img=None):
    """POST ``vec`` to ``/slm/rearrange`` (the walk, or the 3-D rearrangement -- whichever protocol
    is currently sticky) and set ``s1.G.<ok_flag>``. Returns True on success.

    Mirrors ``rearrange_callbacks._do_rearrange_round``'s abort tail: on failure drop the phantom
    server ledger row + the staged frame so the SLM ledger stays aligned with the lab seq_ids."""
    runid = rearrange_callbacks._runid_kwargs(ctx.scan_id, _seq_id(s1))
    if stage_img is not None:
        _stage(ctx, s1, stage_img)
    try:
        r = ctx.client.rearrange(vec, **runid)
        if isinstance(r, dict) and not r.get("ok", True):
            ctx.record_error("[%s] rearrange returned ok=false" % tag,
                             kind="rearrange", seq_id=_seq_id(s1))
        setattr(s1.G, ok_flag, True)
        return True
    except Exception as err:  # noqa: BLE001
        try:
            if runid:
                ctx.client.cancel_last_shot(**runid)
        except Exception:  # noqa: BLE001 - older server lacks /slm/cancel_last; non-fatal
            pass
        _cancel(ctx)
        setattr(s1.G, ok_flag, False)
        ctx.record_error("[%s] rearrange call failed: %s" % (tag, err),
                         kind="rearrange", seq_id=_seq_id(s1))
        return False


def _walk_only(s1):
    """``rearrange_kwargs.extras.walk_only`` -- skip the 3-D rearrangement entirely (bootstrap /
    walk-survival mode). Absent -> False (the full flow)."""
    try:
        return bool(s1.C.rearrange_kwargs.extras.walk_only(False))
    except Exception:  # noqa: BLE001
        return False


def _n_per_plane(s1):
    """Sites per axial layer, from the scan's ``focus_walk.n_per_plane`` declaration (a seq-local
    node -- never forwarded to the server). Only used to size an all-zero vector when detection
    produced nothing; a successful detection supersedes it."""
    try:
        return max(int(s1.C.focus_walk.n_per_plane(0)), 0)
    except Exception:  # noqa: BLE001
        return 0


def _frame_pattern(ctx, idx):
    """The detection pattern name for camera frame ``idx`` (all three frames declare the same
    single-plane pattern), or None -> the detector's own fallback."""
    return rearrange_callbacks._frame_pattern(ctx, idx)


def _stage(ctx, s1, img):
    """Hand the frame to the ExptServer persister (async, FIFO) so the encode+store overlaps the
    server round-trip instead of blocking the held-atom critical path."""
    rearrange_callbacks._safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))


def _cancel(ctx):
    rearrange_callbacks._safe(ctx.server, "cancel_shot")


def _grab_fail(ctx, s1, tag, frame_idx, n_seen):
    """Record a frame-grab failure (desync vs timeout) and cancel the shot."""
    rearrange_callbacks._report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
    _cancel(ctx)


def _seq_id(s1):
    return rearrange_callbacks._seq_id(s1)
