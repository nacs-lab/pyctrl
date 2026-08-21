"""TwoLayerLiftCommSeq.py -- assemble a TWO-LAYER array out of one loaded plane and read both
layers out, with the FULL trap array present in every frame.

New seq (2026-08-07).  Ground truth for the build path + per-shot machinery is
``RearrangeCommSeq2.py``; like ``Rearrange3DFocusWalkCommSeq`` the later handoffs push their OWN
``setup_rearrangement``, so each round can run a different protocol / target / nsteps.

THE SHOT -- three rounds, and THREE or FOUR camera frames depending on one scan flag
``rearrange_kwargs.extras.mid_image`` (default 1 = on):

  mid_image = 0  (THE DEMO SHAPE -- 3 frames, maximum survival)
      Imag399 #1  LOADING array
      ROUND 0     loading -> every_other          (bookend: the FULL loading WGS)
      ROUND 1     3-D lift of one half            (bookend: the two-layer WGS)
      Imag399 #2  NEAR layer -- the atoms left behind at the loading plane
      ROUND 2     pingponggrating one-way axial walk of the WHOLE hologram
      Imag399 #3  FAR layer, now at the camera plane

  mid_image = 1  (THE DEBUG SHAPE -- 4 frames)
      ... identical, except an extra Imag399 sits between ROUND 0 and ROUND 1 so the compacted
      array can be seen on its own.  It costs what any extra image costs (detection infidelity +
      the imaging pulse's heating, ~1-3%% on every atom downstream of it) and it is NOT part of
      the measurement -- turn it off once the assembly is debugged.

A SECOND build-time flag, ``rearrange_kwargs.extras.do_walk`` (default 1 = on), drops ROUND 2 and
its frame.  Off, the shot ends on the NEAR-layer frame with the far layer assembled and left 20 um
away, unimaged: the cheap way to ask what the lift costs the atoms that DON'T move, without
spending the walk's frames or a fourth image (and it keeps a debug shot to three frames, which is
what the dashboard shows).  The two flags are independent, so the frame count is 2, 3 or 4.

WHAT THE MID IMAGE ACTUALLY BUYS, AND WHAT ITS ABSENCE COSTS
  With it, round 1's bit vector is MEASURED: img2 is taken on the same 1068-site loading grid the
  round's ``init_grid`` is, so the lift's assignment sees exactly the atoms that survived round 0.
  Without it there is no frame in front of round 1, so the vector is DECLARED by the scan
  (``two_layer.mid_bits`` -- "every every-other site is occupied").  That is sound because round 0
  is supply-saturated (~640 loaded atoms against 536 targets), and it makes the round-1 assignment
  a FIXED geometric permutation: identical every shot, so the transit frame stack is
  shot-independent.  Sites round 0 failed to fill are commanded anyway; an empty trap being moved
  costs nothing.

WHY EVERY BOOKEND IS A FULL-ARRAY WGS.  Every frame is imaged with all 1068 traps present -- the
unloaded ones, the ones the compaction emptied, and (after the lift) the two layers together.
Trap count is therefore constant across the whole shot, so per-trap depth is constant and the
frames are directly comparable with no depth correction.  That is what round 0's bookend choice
buys: the compaction targets the every-other subset but the bookend written afterwards is the FULL
loading array, not an every-other one.

  (The TRANSIT frames are still sparse -- ``_protocol_rearrange`` builds its frames from the paired
  atoms only, so empty sites are absent while atoms are moving.  That is between images and does
  not touch any imaging condition; every atom keeps its own trap throughout.)

THE THREE PER-SHOT SETUPS, AND WHY ALL THREE ARE NEEDED
  ``protocol`` / ``pattern`` / ``target_bits`` / ``nsteps`` / the phases are all SERVER-STICKY, and
  the three rounds disagree about every one of them, so each round pushes its own setup:

    pre_run              <- ``rearrange_kwargs``   round 0: protocol rearrange, pattern
                                                   every_other, target_bits None, initial_phase =
                                                   final_phase = the LOADING WGS.
    before the lift      <- ``rearrange_kwargs2``  round 1: final_phase = the two-layer WGS +
                                                   ``target_grid_planes_z_rad`` (the 3-D opt-in) +
                                                   ``target_bits`` (the mask that makes only the
                                                   near+far sites active) + ``pattern = None``.
    before the walk      <- ``rearrange_kwargs3``  round 2: protocol pingponggrating with
                                                   ``initial_phase`` swapped to the two-layer WGS
                                                   and ``skip_grid_derive``.

  Re-asserting the LOADING phases in round 0's setup is load-bearing, not tidiness: round 2 leaves
  ``_rearrange_initial_phase`` pointing at the two-layer hologram, and ``reload_rearrange`` writes
  whatever that is at the next shot's start.  Without the re-assert the next shot would load into
  the two-layer array.  The grid derives this costs are ~40-75 ms each on the server (measured
  2026-08-07), i.e. ~0.25 s of extra in-trap time per shot against a ~300 s trap lifetime.

ROUND 2 RIDES ON A DIFFERENT HOLOGRAM.  ``pingponggrating`` builds every frame as
``server._rearrange_initial_phase + amp*d*unit_map`` -- the cached WGS INITIAL phase, NOT whatever
is on the panel.  After round 1 the panel holds the two-layer WGS, so the walk would otherwise
translate the loading array.  Swapping ``initial_phase`` with ``skip_grid_derive=True`` is the
documented cheap path for "same lattice, different WGS": no FFT, and frame 0 of the walk IS the
round-1 bookend, so the two are continuous.

WHY THERE IS NO ``update_rearrange`` ON THE FINAL FRAME.  The last frame is detected on the
FAR-layer pattern, whose site count is not the server's ``init_grid`` length, and the shot ends
displaced mid-protocol.  Posting it would fail the length check and would mean nothing to the
server's scorer.  Both layers' survival is computed lab-side from the stored frames.

Frame alignment / abort safety is the unchanged invariant (same enforcement as
``rearrange_callbacks``): a shot either stages ALL its frames and publishes them with a single
``finish_shot``, or it is cancelled and NO partial set is persisted.  Every callback that runs
CONSUMES its frame even on a failing shot so a straggler cannot shift the frame stream on the next
shot; a failed shot's captured frames are re-published under the FAILING sentinel for LIVE DISPLAY
ONLY.

The BUILD path is a plain extension of RearrangeCommSeq2's (same steps, same NI keep-alives, same
pattern tags); only the deferred callbacks -- which ``serialize()`` never runs -- carry the logic
above.  ``mid_image`` is read at BUILD time (it adds/removes a basic sequence), exactly as
``walk_only`` is in the focus-walk seq.  Nothing here touches the byte path.

Scan-declared seq-local nodes (never forwarded to the server):
    ``two_layer.mid_bits``  list -- round 1's declared vector, required when ``mid_image`` is 0.
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


# --------------------------------------------------------------------------- #
# Per-frame 399 imaging brightness -- same mechanism + same policy as RearrangeCommSeq2 (the 399
# imaging-power PID locks ONCE at the ROOT BlueMOTStep and HOLDS, so each frame's only brightness
# knob is its own DDS amps; see that file's policy block and
# yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md).  Names here:
#     extras.MidImgAmp1  / MidImgAmp2   -> the post-compaction frame (mid_image only)
#     extras.LiftImgAmp1 / LiftImgAmp2  -> the NEAR-layer frame
#     extras.FinImgAmp1  / FinImgAmp2   -> the FAR-layer frame
# Absent -> the frame's own ByPattern/base Amp1/Amp2 and a build byte-identical to the no-knob seq.
# The near and far frames look through different amounts of out-of-focus halo from the OTHER
# layer, so they generally do not want the same amps -- that is what these are for.
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


def _mid_image(s):
    """``rearrange_kwargs.extras.mid_image`` -- take a camera frame between the compaction and the
    lift.  Default 1 (on).  Read at BUILD time: it adds/removes a whole basic sequence."""
    return bool(_extras_num(s, "mid_image", 1.0))


def _lift_image(s):
    """``rearrange_kwargs.extras.lift_image`` -- take a camera frame after the LIFT (the NEAR
    layer, at the loading plane).  Default 1 (on).  Read at BUILD time.

    Off, the lift and the walk run back to back with no frame between them, and the shot's last
    image is the FAR layer.  That is the configuration that measures the AXIALLY MOVED atoms:
    survival is compacted-frame -> far-layer frame, i.e. it follows the atoms that went up 20 um
    and came back to the camera plane.  With it ON you instead see the atoms left behind."""
    return bool(_extras_num(s, "lift_image", 1.0))


def _layout(mid, lift_img, walk, split=False):
    """Group the rounds into HANDOFFS -- one per basic sequence, each ending in one image.

    Every round is scored on the frame in front of it, so a round whose image is disabled cannot
    end a handoff: it runs inside the same callback as the round that follows.  The LAST round
    always takes an image regardless of its own flag -- a shot that ends with an un-imaged round
    would have moved atoms with nothing to read them out.

    >>> _layout(True, True, True)     # load / compacted / near / far
    [['compact'], ['lift'], ['walk']]
    >>> _layout(True, False, True)    # load / compacted / far  -- the AXIAL-MOVE measurement
    [['compact'], ['lift', 'walk']]
    >>> _layout(False, True, True)    # load / near / far
    [['compact', 'lift'], ['walk']]
    >>> _layout(True, True, False)    # load / compacted / near -- the walk is off
    [['compact'], ['lift']]
    >>> _layout(False, True, True, split=True)   # load / near / far, lift split in two
    [['compact', 'axial', 'radial'], ['walk']]
    """
    # ``split_lift`` turns the single 3-D lift into TWO rounds: a pure AXIAL hop (far half
    # straight up, xy untouched) then a pure RADIAL rearrange in the far plane.  Neither leg is
    # imaged on its own -- they run inside the same handoff -- so the frame set is unchanged.
    # Motivation: one combined round has to serve a 10 um axial and a ~39 um lateral move with a
    # single nsteps, which pins the axial step ~13x finer than the ~1.25 rad it quantises at.
    # Split, each leg runs at its own rate: axial on the characterised warm-WGS 3-D path, radial
    # at round 0's proven 0.49 px/frame in-plane stroke.
    lift_rounds = ["axial", "radial"] if split else ["lift"]
    rounds = ["compact"] + lift_rounds + (["walk"] if walk else [])
    takes = {"compact": bool(mid), "lift": bool(lift_img),
             "axial": False, "radial": bool(lift_img), "walk": True}
    takes[rounds[-1]] = True                     # the last round always gets read out
    groups, cur = [], []
    for r in rounds:
        cur.append(r)
        if takes[r]:
            groups.append(cur)
            cur = []
    return groups


def _split_lift(s):
    """``rearrange_kwargs.extras.split_lift`` -- run the lift as AXIAL then RADIAL instead of
    one combined 3-D move.  Default 0 (off, unchanged).  Read at BUILD time: it adds a round."""
    return bool(_extras_num(s, "split_lift", 0.0))


def _do_walk(s):
    """``rearrange_kwargs.extras.do_walk`` -- run ROUND 2 (the axial grating walk) and image the
    FAR layer.  Default 1 (on).  Read at BUILD time: it adds/removes a whole basic sequence.

    Off, the shot ends on the NEAR-layer frame: the far layer is assembled and left 20 um away,
    unimaged.  That is the cheap way to measure what the lift costs the atoms that DON'T move --
    the stationary near layer -- without spending the walk's frames or a fourth image, and it
    keeps the shot to three frames so the dashboard shows all of them."""
    return bool(_extras_num(s, "do_walk", 1.0))


def _keepalive_and_cool(sb, s):
    """The between-images boilerplate every non-first bseq needs, in order.

    NI-DAQ keep-alive: re-assert one V* channel (VMOTCoil at its current 0) so libnacs emits
    non-None NI data for this bseq (Cool556/Imag399 touch only TTL+DDS). Physical no-op. We must
    NOT InitStep between the images (that zeroes VSLMservo and loses the atoms), and there is no
    SLMStep / phase write here: the phase on the panel is the bookend the server left.

    Then the cooling step, then the SAME keep-alive again -- DAQmx FINITE AO rejects a 1-sample
    buffer (error -200077) and the first assert is this bseq's only NI update time, so a second
    one is needed to reach >= 2 samples. Physical no-op."""
    sb.add('VMOTCoil', 0)
    sb.add_step(Cool556hXStep, s.C.Cool556)
    sb.add('VMOTCoil', 0)


@seq_capabilities(owns_frames=True)   # grabs + stores its own frames mid-sequence (the handoffs)
def TwoLayerLiftCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.two_layer_compact_ok = False
    s.G.two_layer_lift_ok = False
    s.G.two_layer_walk_ok = False
    s.G.rearrange_lock_ok = False

    mid = _mid_image(s)
    walk = _do_walk(s)

    s.reg_before_start(pre_run)        # locks + round 0's setup + reload

    # Per-bseq SLM pattern (expConfig ByPattern overlay + per-frame detection grid).  Every frame
    # but the last images the SAME 1068-site loading grid (the compaction and the lift move atoms
    # WITHIN it / out of its focal plane, they never change it), so they share one registry record
    # and one threshold set.  Only the last frame needs its own -- the far layer's post-lift xy.
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

    # First Imag399 (LOADING array).
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during the rounds.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # ---- one basic sequence per HANDOFF -------------------------------------------------
    # A bseq's before-callback grabs the frame the PREVIOUS bseq produced, runs this handoff's
    # rounds, and the bseq then takes its own image.  Which rounds share a handoff is decided by
    # which images are enabled (see _layout): a round whose image is off has no frame to be scored
    # on, so it runs back-to-back with the next one inside a single callback.
    handoffs = _layout(mid, _lift_image(s), walk, _split_lift(s))
    cbs = (handoff_0, handoff_1, handoff_2)
    pat_key = {"compact": ("middle_pattern", "MidImgAmp1", "MidImgAmp2"),
               "lift": ("lift_pattern", "LiftImgAmp1", "LiftImgAmp2"),
               "walk": ("final_pattern", "FinImgAmp1", "FinImgAmp2")}
    prev = s
    for i, group in enumerate(handoffs):
        sb = s.new_basic_seq()
        prev.cond_branch(True, sb)
        pat_name, a1, a2 = pat_key[group[-1]]    # the image belongs to this group's LAST round
        _pat = getattr(s.C.rearrange_kwargs.extras, pat_name)("")
        if _pat:
            sb.set_pattern(_pat)
        sb.reg_before_bseq(cbs[i])
        _keepalive_and_cool(sb, s)
        _add_imag399(sb, s.C.Imag399, _img_amps(s, a1, a2))
        prev = sb

    # Initialisation again (shut down for safety).
    prev.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # last frame -> publish the set; release compute; keepalive
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
#
# There is ONE callback per handoff and the handoffs come from the SAME :func:`_layout` the build
# used, so the callbacks can never disagree with the bseq structure about which rounds run where
# or which camera frame each is scored on.
# =========================================================================== #
_ROUND_SPEC = {
    # name       -> (per-shot setup kwargs node, ok-flag).  ``None`` = no setup of its own: round
    #               0 runs on the sticky setup pre_run already pushed.
    "compact": (None, "two_layer_compact_ok"),
    "lift": ("rearrange_kwargs2", "two_layer_lift_ok"),
    # split_lift: the axial hop reuses round 1's node, the radial leg gets its own.
    "axial": ("rearrange_kwargs2", "two_layer_axial_ok"),
    "radial": ("rearrange_kwargs4", "two_layer_radial_ok"),
    "walk": ("rearrange_kwargs3", "two_layer_walk_ok"),
}


def _flag(s1, name, default=1.0):
    # ``default`` matters: the historical flags default ON, but a NEW flag must default OFF or it
    # silently changes every existing scan.  The except-branch has to honour it too.
    try:
        return bool(float(getattr(s1.C.rearrange_kwargs.extras, name)(default)))
    except Exception:  # noqa: BLE001
        return bool(default)


def _handoffs(s1):
    return _layout(_flag(s1, "mid_image"), _flag(s1, "lift_image"), _flag(s1, "do_walk"),
                   _flag(s1, "split_lift", 0.0))


def _round_flags(s1):
    """The ok-flags that must ALL be true for a shot to persist -- exactly the rounds that run."""
    return tuple(_ROUND_SPEC[r][1] for g in _handoffs(s1) for r in g)


def pre_run(s1):
    """Locks + round 0's setup (``rearrange_kwargs``; sticky, no reset_params) + reload.

    ``force_n_rounds=1``: the rounds here are INDEPENDENT setups, not stages of the server's
    multi-round mechanism, so the stage counter must stay off."""
    n_frames = len(_handoffs(s1)) + 1
    rearrange_callbacks.pre_run(
        s1, flags=_round_flags(s1), lock_desc="two-layer lift compute", force_n_rounds=1,
        frames_label="/".join("img%d" % (i + 1) for i in range(n_frames)))


def handoff_0(s1):
    """First handoff: grab the LOADING frame, run this group's rounds."""
    _run_handoff(s1, 0)


def handoff_1(s1):
    """Second handoff (present unless every round shares one)."""
    _run_handoff(s1, 1)


def handoff_2(s1):
    """Third handoff (present only when all three rounds are imaged separately)."""
    _run_handoff(s1, 2)


def post_run(s1):
    """Final frame -> publish the aligned set with one ``finish_shot``, release the compute lock,
    keepalive the scan-long slm lock.

    ``update=False`` in every shape.  When the walk runs, the final frame is on the far-layer
    pattern, whose site count is not the server's ``init_grid`` length.  When it does not, the
    final frame IS on the loading grid and would pass the length check -- but it only sees the
    NEAR half of round 1's target set (the far half is 20 um out of focus), so the server would
    score it as a ~50% run.  Both layers' survival is computed lab-side from the stored frames."""
    rearrange_callbacks.finalize(
        s1, round_flags=_round_flags(s1),
        final_frame_idx=len(_handoffs(s1)), tag="post_run", update=False,
        use_frame_pattern=True, record_ok=True)


def _run_handoff(s1, idx):
    """Run handoff ``idx``: grab + detect camera frame ``idx``, then fire this group's rounds in
    order, each preceded by its own ``setup_rearrangement`` where it has one.

    The FIRST round of the group is scored on the frame just grabbed.  A later round in the same
    group has no frame of its own (its image was disabled), so it is fired on the scan's DECLARED
    vector ``two_layer.mid_bits`` -- see the module docstring.  A missing declaration is a loud
    failing shot, never a guess."""
    groups = _handoffs(s1)
    if idx >= len(groups):
        return
    group = groups[idx]
    prior = None if idx == 0 else _ROUND_SPEC[groups[idx - 1][-1]][1]
    ctx, _img, probs = _grab_detect(s1, idx, "handoff%d" % idx, prior_flag=prior)
    if ctx is None:
        return
    for k, name in enumerate(group):
        kwargs_name, ok_flag = _ROUND_SPEC[name]
        tag = "handoff%d/%s" % (idx, name)
        if k == 0:
            vec = probs                       # scored on the frame in front of this handoff
        else:
            vec = _declared_mid_bits(s1)      # no frame of its own -> the scan's declaration
            if not vec:
                ctx.record_error(
                    "[%s] seq %d: two_layer.mid_bits not declared -- this round has no camera "
                    "frame in front of it (its image is off), so its vector must come from the "
                    "scan" % (tag, _seq_id(s1)), kind="rearrange", seq_id=_seq_id(s1))
                _cancel(ctx)
                return
        if kwargs_name is not None and not _setup(ctx, s1, kwargs_name, tag):
            return
        if not _round(ctx, s1, vec, ok_flag=ok_flag, tag=tag):
            return


# =========================================================================== #
# helpers
# =========================================================================== #
def _declared_mid_bits(s1):
    """``two_layer.mid_bits`` -- round 1's scan-declared vector (used only without the mid image).
    Empty list when absent/malformed, which the caller turns into a loud failing shot."""
    try:
        bits = s1.C.two_layer.mid_bits(None)
    except Exception:  # noqa: BLE001
        return []
    if not bits:
        return []
    try:
        return [float(v) for v in bits]
    except Exception:  # noqa: BLE001
        return []


def _grab_detect(s1, frame_idx, tag, *, prior_flag=None, grab_timeout=0.1, drain_timeout=0.2):
    """Grab camera frame ``frame_idx``, stash + stage it, and detect its per-site probabilities.

    Returns ``(ctx, img, probs)``, or ``(None, None, None)`` when the shot must not continue --
    having already drained the frame and cancelled, so a straggler can never shift the next shot's
    frame stream.  Mirrors :func:`rearrange_callbacks.rearrange_round`'s front half exactly."""
    none = (None, None, None)
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return none
    if not s1.G.rearrange_lock_ok(False):
        return none                     # pre_run couldn't get the compute lock -> skip

    if frame_idx == 0:
        rearrange_callbacks.reset_stash()   # first frame of the shot -> fresh stash

    # A prior round failed: still consume this frame (it is physically produced), keep it for the
    # display re-publish, and cancel so no partial set persists.
    if prior_flag is not None and not getattr(s1.G, prior_flag)(False):
        img, ok, _n = rearrange_runtime.grab_one_frame(ctx.camera, timeout=drain_timeout)
        rearrange_callbacks.stash_frame(frame_idx, img if ok else None)
        _cancel(ctx)
        return none

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=grab_timeout)
    if not ok:
        rearrange_callbacks._report_grab_fail(ctx, s1, tag, "img%d" % (frame_idx + 1), n_seen)
        _cancel(ctx)
        return none
    rearrange_callbacks.stash_frame(frame_idx, img)

    probs = ctx.detect_probs_for(_frame_pattern(ctx, frame_idx), img)
    _stage(ctx, s1, img)
    if not probs:
        ctx.record_error(
            "[%s] seq %d: no per-site probs for pattern %r -- round skipped"
            % (tag, _seq_id(s1), _frame_pattern(ctx, frame_idx)),
            kind="detect", seq_id=_seq_id(s1))
        _cancel(ctx)
        return none
    return ctx, img, probs


def _setup(ctx, s1, kwargs_name, tag):
    """Push the per-shot ``setup_rearrangement`` held in ``s1.C.<kwargs_name>``.  False on
    failure (already recorded + cancelled)."""
    args = rearrange_runtime.collect_kwargs(getattr(s1.C, kwargs_name))
    args = rearrange_runtime.translate_zernike_zN(args)
    if not args:
        ctx.record_error(
            "[%s] seq %d: %s is empty -- the scan must declare this round's setup"
            % (tag, _seq_id(s1), kwargs_name),
            kind="setup_rearrangement", seq_id=_seq_id(s1))
        _cancel(ctx)
        return False
    args.setdefault("client_scan_id", str(ctx.scan_id))
    try:
        ctx.client.setup_rearrangement(**args)
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[%s] setup_rearrangement failed: %s" % (tag, err),
                         kind="setup_rearrangement", seq_id=_seq_id(s1))
        _cancel(ctx)
        return False
    return True


def _round(ctx, s1, vec, *, ok_flag, tag):
    """One ``rearrange(vec)`` call.  Sets ``ok_flag`` and returns True on success; on failure
    drops the phantom server ledger row + cancels the shot (the shared round's abort tail)."""
    runid = rearrange_callbacks._runid_kwargs(ctx.scan_id, _seq_id(s1))
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


def _frame_pattern(ctx, idx):
    """The detection pattern name for camera frame ``idx``, or None -> the detector's fallback."""
    return rearrange_callbacks._frame_pattern(ctx, idx)


def _stage(ctx, s1, img):
    """Hand the frame to the ExptServer persister (async, FIFO) so the encode+store overlaps the
    server round-trip instead of blocking the held-atom critical path."""
    rearrange_callbacks._safe(ctx.server, "stage_frame", img, ctx.scan_id, _seq_id(s1))


def _cancel(ctx):
    rearrange_callbacks._safe(ctx.server, "cancel_shot")


def _seq_id(s1):
    return rearrange_callbacks._seq_id(s1)
