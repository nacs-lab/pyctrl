"""RearrangeSphereFocusStackCommSeq.py -- assemble a 3-D SPHERE, then walk it axially so ONE of
its latitude levels lands on the camera focal plane.

New seq (2026-08-07).  Ground truth for the per-shot machinery is ``RearrangeCommSeq.py`` (one
rearrangement round) and ``Rearrange3DFocusWalkCommSeq.py`` (the two-mid-shot-handoff shape and
the "re-send a phase + ``skip_grid_derive`` to change ONE cached write phase" trick).  What is new
here is the ORDER -- rearrange FIRST, walk SECOND -- and what that order buys.

  Init -> MOT -> SLM(33x33 loading WGS) -> GreenMOT -> LAC -> Imag399 (#1, the loading array)
       -> Cool -> 3-D rearrange(loaded atoms -> the 50 sphere sites, warm 3-D WGS transit,
                                bookend = the sphere WGS)
       -> pingponggrating depth walk of the WHOLE sphere by ``walk`` rad (phase-only, no model)
       -> Cool -> Imag399 (#2, the sphere at that carrier: ONE level is in focus) -> Init

TWO camera frames, ONE rearrangement round.  The scan sweeps the walk amplitude over the sphere's
level depths, so each scan cell images a different slice and the run as a whole is a focus stack
of one rigid object.  See ``YbScans/SLMRearrangeSphere3DFocusStackScan.py`` for every number and
``campaigns/geometry/sphere3d/sphere_geometry.py`` for the sphere itself.

WHY THE WALK COMES AFTER THE ASSEMBLY (and is a walk at all)
  A 20 um-radius sphere is 40 um deep; the imaging depth of field is ~4 um.  So no single frame
  can show more than one level, and the only way to see the others is to move the WHOLE object
  along the optical axis between shots.  Two ways to do that:
    * command the sphere at a per-shot z offset (targets = sphere + delta) -- but then every shot
      assembles a DIFFERENT object, the transit length varies with delta, and per-plane brightness
      inherits that systematic;
    * assemble the SAME sphere every shot and translate it afterwards -- one rigid object, one
      transit cost, and the only thing that changes between cells is a global quadratic phase.
  This seq does the second.  ``pingponggrating`` depth mode is exactly that translation: frame k
  is ``initial_phase + k*step_size*Z4`` written straight to the panel, no model, no bookend, and
  ``return=False`` leaves the panel resting on the fully walked frame -- which is where Imag399
  #2 fires.

THE ONE SUBTLETY: WHAT ``pingponggrating`` WALKS
  It adds its Z4 ramp to the server's cached WGS **initial_phase** -- the LOADING hologram.  Used
  as-is, the first walk frame would replace the just-assembled sphere with the 33x33 array and
  every atom would be lost.  So the mid-shot setup (``rearrange_kwargs2``) re-sends
  ``initial_phase = <the sphere phase>`` together with ``skip_grid_derive=True``: the server
  rebuilds ONLY the cached initial write phase and keeps BOTH derived grids (no mid-shot 4096^2
  FFT, and the 3-D z labels stay intact).  The walk then starts from the hologram that is
  physically on the panel -- the sphere WGS bookend the rearrange call just wrote -- so frame 0 of
  the walk is a no-op continuation rather than a jump.
  Symmetrically, the per-shot setup #1 (``rearrange_kwargs``, pushed in ``pre_run``) re-sends
  ``initial_phase = <the loading phase>`` so the NEXT shot's ``reload_rearrange`` writes the 33x33
  array again.  The two setups hand the "initial phase" role back and forth once per shot; miss
  either half and the scan quietly loads atoms into the wrong hologram.

DETECTION
  Frame #1 is the production 33x33 pattern -- the one with a calibrated affine and per-site
  thresholds -- and its per-site probabilities are what the rearrange call is given.  Frame #2 is
  the sphere: a 50-site 3-D pattern whose registry record this scan declares, but whose per-site
  thresholds must be BOOTSTRAPPED offline from a first run's frames
  (``campaigns/imaging/imaging_det/bootstrap_pattern_thresholds.py``) exactly as the 2-layer array was.  Nothing in
  the shot path depends on that: frame #2's detection is never fed back to the server (the seq
  finalizes with ``update=False``), because the server scores ``result_bits`` against the 1068-site
  init_grid and this scan's final array is a DIFFERENT 50-site grid -- a length mismatch, and a
  meaningless statistic even if it fitted.  The sphere is measured lab-side, from the frames.

Frame alignment / abort safety is the unchanged invariant: a shot either stages BOTH frames and
publishes them with a single ``finish_shot``, or it is cancelled and no partial pair persists.
Every callback that runs CONSUMES its frame even on a failing shot so a straggler cannot shift the
frame stream on the next shot.

ASSEMBLE-ONLY MODE (``rearrange_kwargs.extras.assemble_only`` truthy): skip the walk entirely.
Frame #2 then images the sphere at the loading carrier (its equatorial levels in focus).  Use it
as the FIRST run on a new sphere -- it bootstraps the pattern's thresholds and answers "did the
3-D assembly work at all?" before the walk adds a second unknown.

The BUILD path is a plain 2-bseq transliteration of RearrangeCommSeq's (same steps, same NI
keep-alives, same pattern tags); only the deferred callbacks -- which ``serialize()`` never runs --
carry the logic above.  Nothing here touches the byte path.
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
from ramp_to import ramp_to
from SLMStep import SLMStep

import rearrange_callbacks
import rearrange_runtime
from seq_capability import seq_capabilities


# --------------------------------------------------------------------------- #
# Per-frame 399 imaging brightness -- same mechanism + policy as RearrangeCommSeq2 (the imaging
# PID locks ONCE at the root BlueMOTStep and HOLDS, so each frame's only brightness knob is its
# own DDS amps; see yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md):
#     extras.FinImgAmp1 / FinImgAmp2 -> frame #2 (the sphere) only.
# The sphere is 50 traps against the loading array's 1068, so its per-trap intensity -- and hence
# the imaging exposure it wants -- is a different regime; this is the knob for it.
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
def RearrangeSphereFocusStackCommSeq(s):
    # Per-seq coordination flags (DynProps reads return a bool, not a SubProps).
    s.G.sphere_ok = False
    s.G.rearrange_lock_ok = False

    s.reg_before_start(pre_run)        # locks + per-shot setup #1 (the 3-D rearrange) + reload

    # Per-bseq SLM pattern (expConfig ByPattern overlay): bseq1 images the loading array, bseq2
    # the sphere. Names from rearrange_kwargs.extras.initial_pattern / final_pattern.
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

    # First Imag399 (img1, the 33x33 loading array -> which sites loaded).
    s.add_step(Imag399Step, s.C.Imag399)

    s.add_step(Cool556hXStep, s.C.Cool556)

    # Leave the cooling light on a little during the transit + the walk.
    Freq_Cool556Detuning = s.C.rearrange_kwargs.extras.RearrCoolDet(0.13 * 1e6)
    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq()
    Freq_Cool556 = Freq_Resonance556mj0Freq + Freq_Cool556Detuning
    Amp_Cool556 = s.C.rearrange_kwargs.extras.RearrCoolAmp(0)

    s.add('Freq556MOTX', Freq_Cool556).add('Amp556MOTX', Amp_Cool556)
    s.add('Freq556RydbergMOTh', Freq_Cool556).add('Amp556RydbergMOTh', Amp_Cool556)

    # ---- bseq2: assembly + walk happen in its before-bseq callback; image the result.
    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)

    _final_pat = s.C.rearrange_kwargs.extras.final_pattern("")
    if _final_pat:
        s2.set_pattern(_final_pat)

    s2.reg_before_bseq(assemble_and_walk)   # img1 -> probs -> 3-D rearrange -> axial walk

    # NI-DAQ keep-alive: re-assert one V* channel (VMOTCoil at its current 0) so libnacs emits
    # non-None NI data for this bseq (Cool556/Imag399 touch only TTL+DDS). Physical no-op. We must
    # NOT InitStep between the images (that zeroes VSLMservo and loses the atoms), and there is no
    # SLMStep / phase write here: the phase on the panel is the WALKED sphere the server left.
    s2.add('VMOTCoil', 0)

    # ---- trap-depth handover: transport deep, image at production per-trap depth ----------
    # The 532 servo sets TOTAL power, so the assembled array's ~50 traps sit at ~20x the
    # production per-trap depth while the panel still carries the 1068-trap setpoint -- enough
    # light shift to push the 399 imaging light off resonance. Ramp it down HERE: after the
    # transit and the walk (both of which are safer deep), before the cooling + image. The ramp
    # is slow against the ~85 kHz radial trap frequency, so it is adiabatic: the atoms cool as
    # the trap relaxes rather than spilling. ``sphere.servo`` <= 0 disables the ramp entirely
    # (a scan that wants the loading setpoint straight through).
    _servo = s.C.sphere.servo(0.0)
    if _servo > 0:
        s2.add_step(s.C.sphere.servo_ramp(10e-3)).add('VSLMservo', ramp_to(_servo))

    s2.add_step(Cool556hXStep, s.C.Cool556)

    # NI-DAQ minimum-buffer guard: DAQmx FINITE AO rejects a 1-sample buffer (error -200077), and
    # the keep-alive above is this bseq's only NI update time -- re-asserting it after the cooling
    # step gives a second update time (>= 2 samples). Physical no-op. (The servo ramp already
    # supplies many NI samples when it is enabled; this keeps the no-ramp path legal too.)
    s2.add('VMOTCoil', 0)

    # Second Imag399 (img2, the sphere at this cell's carrier: ONE level in focus).
    _add_imag399(s2, s.C.Imag399, _img_amps(s, "FinImgAmp1", "FinImgAmp2"))

    # Initialisation again (shut down for safety).
    s2.add_step(InitStep, s.C.Init)

    s.reg_after_end(post_run)          # publish the pair; release compute; keepalive slm
    return s


# =========================================================================== #
# Deferred per-shot callbacks (run by the engine; serialize() never runs them).
# =========================================================================== #
def pre_run(s1):
    """Ensure the scan-long slm lock is held, grab the per-shot compute lock, push per-shot setup
    #1 (``rearrange_kwargs``: protocol ``rearrange``, the warm 3-D producer, the sphere as
    ``final_phase``, and -- load-bearing -- ``initial_phase`` back to the LOADING hologram, which
    the previous shot's walk setup replaced), and ``reload_rearrange`` (writes that loading
    hologram at the loading carrier)."""
    rearrange_callbacks.pre_run(
        s1, flags=("sphere_ok",),
        lock_desc="3-D sphere assembly + focus walk compute", frames_label="img1/img2")


def assemble_and_walk(s1):
    """Frame #1 -> loaded occupancy, then the two server handoffs of this shot.

    1. ``rearrange(probs)`` on the sticky setup #1: the warm 3-D WGS transit that carries the
       loaded atoms onto the 50 sphere sites, ending on the sphere WGS bookend.
    2. push setup #2 (``rearrange_kwargs2``: protocol ``pingponggrating`` + depth mode + the
       sphere as ``initial_phase`` with ``skip_grid_derive=True``) and call ``rearrange`` again to
       WALK the assembled sphere axially, resting on the walked frame that Imag399 #2 images.

    The walk's bits are unused by ``pingponggrating`` but must still be ``len(init_grid)`` long,
    so the same probability vector is posted for the record.  Sets ``sphere_ok``."""
    ctx = rearrange_runtime.context()
    if ctx is None or ctx.client is None:
        return
    if not s1.G.rearrange_lock_ok(False):
        return                          # pre_run couldn't get the compute lock -> skip

    rearrange_callbacks.reset_stash()   # first frame of the shot -> fresh stash

    img, ok, n_seen = rearrange_runtime.grab_one_frame(ctx.camera, timeout=0.1)
    if not ok:
        _grab_fail(ctx, s1, "assemble_and_walk", 0, n_seen)
        return
    rearrange_callbacks.stash_frame(0, img)

    probs = ctx.detect_probs_for(_frame_pattern(ctx, 0), img)
    _stage(ctx, s1, img)
    if not probs:
        # No calibration for the LOADING pattern is fatal here: without per-site probabilities the
        # server cannot be told which atoms exist, so there is nothing to assemble.
        ctx.record_error(
            "[assemble_and_walk] seq %d: no per-site probs for pattern %r -- no assembly this "
            "shot (the 33x33 loading pattern must have a calibrated grid + thresholds)"
            % (_seq_id(s1), _frame_pattern(ctx, 0)), kind="detect", seq_id=_seq_id(s1))
        _cancel(ctx)
        return

    # ---- handoff 1: the 3-D assembly (sticky setup #1 from pre_run) -----------------------
    if not _server_round(ctx, s1, list(probs), "assemble", "sphere_ok"):
        return

    if _assemble_only(s1):
        return                          # bootstrap mode: image the sphere where it was built

    # ---- handoff 2: setup #2 (the walk) then the walk itself ------------------------------
    args = rearrange_runtime.collect_kwargs(s1.C.rearrange_kwargs2)
    args = rearrange_runtime.translate_zernike_zN(args)
    if not args:
        ctx.record_error(
            "[assemble_and_walk] seq %d: rearrange_kwargs2 is empty -- the scan must declare the "
            "walk setup (protocol/nsteps/step_size/initial_phase/skip_grid_derive)" % _seq_id(s1),
            kind="setup_rearrangement", seq_id=_seq_id(s1))
        s1.G.sphere_ok = False
        _cancel(ctx)
        return
    args.setdefault("client_scan_id", str(ctx.scan_id))
    try:
        ctx.client.setup_rearrangement(**args)
    except Exception as err:  # noqa: BLE001
        ctx.record_error("[assemble_and_walk] walk setup_rearrangement failed: %s" % err,
                         kind="setup_rearrangement", seq_id=_seq_id(s1))
        s1.G.sphere_ok = False
        _cancel(ctx)
        return

    _server_round(ctx, s1, list(probs), "walk", "sphere_ok")


def post_run(s1):
    """Frame #2 -> the walked sphere: publish the aligned pair with one ``finish_shot``, release
    the compute lock, keepalive the scan-long slm lock.

    ``update=False`` on purpose: the server scores ``result_bits`` per init_grid site (1068, the
    33x33 array) and this frame measures a DIFFERENT 50-site grid, so there is no vector to post.
    The sphere is scored lab-side from the published frames."""
    rearrange_callbacks.finalize(
        s1, round_flags=("sphere_ok",), final_frame_idx=1, tag="post_run",
        update=False, use_frame_pattern=True, record_ok=True)


# =========================================================================== #
# helpers
# =========================================================================== #
def _server_round(ctx, s1, vec, tag, ok_flag):
    """POST ``vec`` to ``/slm/rearrange`` (the assembly, or the walk -- whichever protocol is
    currently sticky) and set ``s1.G.<ok_flag>``.  Returns True on success.

    Mirrors ``rearrange_callbacks._do_rearrange_round``'s abort tail: on failure drop the phantom
    server ledger row + cancel the shot so the SLM ledger stays aligned with the lab seq_ids."""
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


def _assemble_only(s1):
    """``rearrange_kwargs.extras.assemble_only`` -- skip the axial walk (bootstrap / assembly
    check).  Absent -> False (the full flow)."""
    try:
        return bool(s1.C.rearrange_kwargs.extras.assemble_only(False))
    except Exception:  # noqa: BLE001
        return False


def _frame_pattern(ctx, idx):
    """The detection pattern name for camera frame ``idx`` (frame 0 = the loading array, frame 1 =
    the sphere), or None -> the detector's own fallback."""
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
