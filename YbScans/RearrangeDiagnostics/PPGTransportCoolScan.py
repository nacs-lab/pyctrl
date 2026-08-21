"""PPGTransportCoolScan.py -- 556 COOLING DURING radial pingponggrating transport.

THE QUESTION (2026-08-09, user).  Cooling during full SLM rearrangement was tried long ago and
did not help, but the technique was never optimized.  This is the small, clean version of that
test: a fixed radial out-and-back ``pingponggrating`` move chosen to sit ON the survival cliff
(~50% survival with no cooling), with the 556 X+h molasses ON during the move.  Deliverable:
does survival at that one operating point go UP with cooling, and at what (detuning, amp)?

MECHANISM -- nothing new is built.  ``RearrangeCommSeq`` already ends bseq1 with::

    Freq556MOTX / Freq556RydbergMOTh <- Resonance556mj0Freq + extras.RearrCoolDet   (default 0.13 MHz)
    Amp556MOTX  / Amp556RydbergMOTh  <- extras.RearrCoolAmp                         (default 0 = OFF)

and those DDS values are HELD by the FPGA across the bseq1 -> bseq2 handoff, i.e. for the whole
host-side rearrange window (img1 readout + detection + the SLM playback that IS the transport).
bseq2's ``Cool556hXStep`` turns them off again.  Same two channels, same shape as the pre-imaging
``Cool556hXStep``, so ``RearrCoolDet = 0.12 MHz`` + ``RearrCoolAmp = 0.14`` reproduces the
33x33_feedback11 pre-imaging cooling EXACTLY (ByPattern Cool556 X and h are both 0.12/0.14).
The default is ``RearrCoolAmp = 0``, so nothing here changes any default, any step, or any other
scan -- both knobs are set from this scan only.

WHAT IS AND IS NOT GATED.  There is no FPGA timing during the handoff, so the light cannot be
gated to the motion alone: it is on for the ENTIRE window (detect + compute + ~35 ms of writes),
and during the pre-motion part the traps are at FULL depth while during the move they are
shallower (grating diffraction efficiency).  That is why every cooling cell is measured TWICE --
see the control below.

THE CONTROL (dim 3).  ``extras.step_x = 0`` at the SAME nsteps writes the same number of SLM
frames with zero displacement (amp 0 -> every frame is the WGS phase), so it pays the identical
write count, the identical window length and the identical 556 dose but makes NO motion.  Any
survival cost of the light alone therefore appears there, and the transport figure of merit is
the ratio::

    S_transport(det, amp) = S(step = s*, det, amp) / S(step = 0, det, amp)

The ``amp = 0`` row of both slabs is the no-cooling baseline (9 detuning replicates of it, since
detuning is inert at zero amp).

DETUNING SIGN / RANGE.  ``RearrCoolDet`` is an offset from the daily ``Resonance556mj0Freq`` (the
in-trap mj=0 line at FULL depth), the same numeraire ``Cool556.{X,h}.FreqDetuning`` uses, where
the working cooling value is +0.12 MHz.  The trap is SHALLOWER during transport, so the in-trap
line moves and the right offset is not the static one -- hence a wide grid that deliberately
crosses zero onto the wrong (heating) side, so the scan shows harm as well as help.

MODES
  ``--mode calib``   1-D ``step_x`` sweep with the cooling OFF -- the no-cooling survival-vs-step
                     curve at this nsteps, and the STATIONARY COOLING-OFF control (``step_x = 0``:
                     same SLM write count, no motion, no light) that anchors everything else.
                     Predicted from the 07-22 cliff (measured at nsteps=50 = 100 steps, S=0.44 at
                     1.30) re-raised to the 80 steps run here: the operating point 1.225 knm px
                     (= 0.5 um) should land near ~0.74, with 50% nearer 1.30.  Measured, not
                     assumed -- the cliff drifts.
  ``--mode cool``    THE EXPERIMENT.  3-D: dim1 RearrCoolDet x dim2 RearrCoolAmp x dim3 step_x
                     {0, s*}.  ``--step`` defaults to the user-chosen 0.5 um = 1.225 knm px.
  ``--mode confirm`` 2-D: dim1 step_x sweep x dim2 cooling {off, the winning (det, amp)} at high
                     reps -- the money plot (does the whole cliff move?).  Pass ``--det``/``--amp``.

GEOMETRY / CONFIG.  Mirrors the 08-06 radial campaign (``PPGOneWayHeatScan``) so the numbers are
comparable, except the defocus: z4 = loading_defocus = **-5** (user 2026-08-09 -- loads better
than the campaign's -4; the two are moved TOGETHER so the loaded atoms and the transit frames
stay co-planar).  Array 33x33_feedback11, ifEnhanced=True, precompute +
precompute_host, hw_sequence=False, depth/true_defocus OFF (LATERAL, set explicitly -- server
extras are sticky and an axial run leaves depth=True behind), ``hold_ms = 0`` (no turnaround
dwell, per the user spec), ``return_trip`` left at its True default -> nsteps OUT + nsteps BACK.
Scale 2.45 knm px per um.  RETURN-TRIP means the array lands back on the stored WGS phase, so
LIVE detection is valid for every cell -- no offline re-detection (unlike the one-way campaign).

Run::

    python YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode calib --dry-run
    python YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode calib --reps 8 --force
    python YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode cool --step-um 0.5 --force
    python YbScans/RearrangeDiagnostics/PPGTransportCoolScan.py --mode confirm --det 0.06 --amp 0.14 --force
"""

import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -5.0              # user 2026-08-09: -5 loads better than the -4 the 08-06 campaign used.
                            # z4 is set to the SAME value so the loaded atoms and the transit
                            # frames stay co-planar (a z4 != loading_defocus mismatch defocuses
                            # every transit frame and would show up as transport loss).

PERIOD_MS = 0.696           # the SLM write floor -- the fastest pacing that exists
NSTEPS = 40                 # 40 OUT + 40 BACK (return_trip default True) = 80 steps/shot
HOLD_MS = 0.0               # no turnaround pause (user spec); explicit -- extras are sticky

KNM_PER_UM = 2.45           # campaign scale (user-specified, campaigns/ppg/oneway/CAMPAIGN_STATE.md)

# --- mode calib: the no-cooling survival-vs-step curve at nsteps=40 ----------------------
# Predicted from the 07-22 fine cliff (measured at nsteps=50, i.e. 100 steps): per-step loss is
# S**(1/100) there, re-raised to the 80 steps run here -> 1.10 -> 0.94, 1.225 -> 0.74,
# 1.30 -> 0.51, 1.35 -> 0.34, 1.40 -> 0.22.  step_x = 0 is the STATIONARY, COOLING-OFF control
# (same write count, no motion, no light) -- the anchor everything else is normalized to.
CALIB_STEPS = [0.0, 1.10, 1.225, 1.30, 1.35, 1.45]

# --- mode cool: the cooling grid ---------------------------------------------------------
# dim1 -- detuning offset from Resonance556mj0Freq, MHz.  POSITIVE = RED (the cooling side --
# the production Cool556 values are all positive); NEGATIVE = BLUE = heating.
#
# THE SIGNATURE WE WANT (user, 2026-08-09): HARM on one side of the line and HELP on the other.
# A one-sided improvement could be almost anything (a trivial trap/servo artifact, a detection
# shift); an antisymmetric response across a resonance can only be Doppler/sideband cooling
# vs heating, i.e. proof the light is doing thermodynamic work on the atoms.  So the grid must
# STRADDLE the true in-trap line, and that line's position during the move is NOT known: the
# axis zero is the full-depth static mj=0 resonance, while the traps are shallower in transit.
# Hence a range wide on BOTH sides -- far enough blue to bracket the line even if it has moved
# down, far enough red to pass through the optimum and out the other side.
COOL_DETS_MHZ = [-0.35, -0.20, -0.10, 0.00, 0.08, 0.16, 0.25, 0.35, 0.50, 0.70]
# dim2 -- 556 amplitude on BOTH beams.  0 = the no-cooling baseline; 0.14 = the exact
# pre-imaging (ByPattern Cool556 X/h) value the user asked to start from.
COOL_AMPS = [0.0, 0.07, 0.14, 0.25]
DEFAULT_STEP_KNM = 0.5 * KNM_PER_UM   # 1.225 knm px = 0.5 um/step (user-chosen operating point)

# --- mode depth: raise the TRAP DEPTH during transport ------------------------------------
# VSLMservo (the 532 servo setpoint, NI Dev1/8) held across the handoff, so the whole
# rearrange window runs at this depth.  33x33_feedback11's normal value is Init.VSLMServo =
# 1.9, which is IN the grid and is the matched-timing control (it still pays both ramps and
# the hold, so a difference against it is depth, not timing).
#
# 0 and 0.25 are included deliberately (user): 0 is the trap-off extreme and anchors the
# bottom of the curve.  Safety: VSLMservo is a servo SETPOINT, values 0.03-6.0 appear across
# arrays, 0.03 is used deliberately as a 'dropped trap' reference, and SLMTrapModulationStep
# already ramps the depth down on purpose -- there is no note anywhere against low or zero
# values.  CAVEAT (not damage): the 532 servo may sit below its regulation floor at 0-0.25
# (the 399 imaging PID has exactly that pathology, memory open-img1pid-servo-floor), so those
# two cells may not deliver the commanded power and may cost a recovery transient on the way
# back up.  The hold + ramp-down + Cool556 give it time, and the step=0 slab at the same
# voltage absorbs whatever it does -- but do not over-read those two points.
DEPTH_VSERVOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 2.0, 2.25, 2.5, 2.75, 3.0]
DEPTH_RAMP_MS = 5.0      # ~400 radial trap periods (85.7 kHz) -- deeply adiabatic, and well
                         # above the NI >=2-sample floor (bug-pyctrl-ni-dac-underflow-200018)
DEPTH_HOLD_MS = 10.0     # settle at depth after the move, before touching the servo again

# --- mode dir: is the molasses DIRECTIONAL? ----------------------------------------------
# Transport direction, degrees from +x in the knm plane, folded to step_x/step_y per shot by
# rearrange_callbacks._fold_step_polar (extras.step_r + extras.step_theta_deg).
DIR_THETAS_DEG = [0.0, 90.0, 180.0, 270.0]
# dim-2 beam cells, (amp_X, amp_h) as a FRACTION of --amp.  The 556 recool is a TWO-CHANNEL
# molasses -- 556MOTX drives X1+X2 (diagonal, in a plane containing the tweezer axis),
# 556RydbergMOTh drives the horizontal MOT-H beam -- and jobs 641/646 drove both TOGETHER,
# which averages away any anisotropy.  These cells separate them.
DIR_BEAM_CELLS = [("off", 0.0, 0.0), ("both", 1.0, 1.0), ("X only", 1.0, 0.0),
                  ("h only", 0.0, 1.0), ("both weak", 0.25, 0.25)]

# --- mode temp: the thermal-vs-mechanical discriminator ----------------------------------
# Cool556.Time (ms). Production is 5 ms; 0.5 ms leaves the atoms markedly hotter, 15 ms is
# past saturation, so the axis brackets the prepared temperature both ways.
TEMP_TIMES_MS = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

# --- mode confirm ------------------------------------------------------------------------
CONFIRM_STEPS = [0.0, 1.10, 1.225, 1.30, 1.35, 1.45]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both frames = the SAME unshifted array.  Correct for EVERY cell here: the move is a
    round trip, so the array rests on the stored WGS phase when img2 is taken."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _common(g, period_ms, nsteps, precompute=True, polar=False):
    """Everything shared by the three modes: warmup, loading, the lateral pingponggrating
    config, and the run params."""
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.extras.final_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    g().rearrange_kwargs.extras.n_rounds = 1
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)
    rk.nsteps = int(nsteps)
    rk.extras.hold_ms = HOLD_MS          # no turnaround dwell (user spec)
    # return_trip is left at its True default -> out AND back, so the array lands on the stored
    # WGS phase and live detection is valid at every cell.

    # RADIAL only.  step_x/y/z are folded to the xyz 3-vector per shot by
    # rearrange_callbacks._fold_step_xyz -- a list-valued swept axis breaks the lab-side grid.
    #
    # POLAR MODE: _fold_step_polar derives step_x/step_y from step_r + step_theta_deg with
    # setdefault, so a step_y ALREADY in extras wins and the direction axis goes silently inert
    # (every theta would move along +x). Setting it to None is worse -- _fold_step_xyz then calls
    # float(None) and the shot dies. So in polar mode the key is simply not emitted.
    if not polar:
        rk.extras.step_y = 0.0
    rk.extras.step_z = 0.0

    # LATERAL mode, set EXPLICITLY: server extras are sticky across scans and a prior axial run
    # leaves depth=True / true_defocus=True behind.
    rk.extras.depth = False
    rk.extras.true_defocus = False
    rk.extras.depth_piston_corr = 0.0
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0

    # PRECOMPUTE.  True bakes every transit frame's uint8 phase at setup so the hot loop is
    # write-only; False computes each frame inside the loop.  Set EXPLICITLY either way -- server
    # extras are sticky across scans, so an omitted key silently inherits the previous scan's.
    # With it OFF the per-frame compute may not fit inside the 0.696 ms step period, in which case
    # the SERVER's own pacing is what actually ran: check diag paced_total_ms / write_lat_ms, and
    # check that the step_x=0 control and the no-cooling baseline ratio are unchanged before
    # comparing any cooling number across the two settings.
    rk.extras.precompute = bool(precompute)
    rk.extras.precompute_host = bool(precompute)
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True                 # BlueLAC loading (production)
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    rp.NumPerGroup = 100000                     # upper bound; --reps sets the pass count
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1                             # interleaves the controls in time
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return rk


def build_calib(steps=None, period_ms=PERIOD_MS, nsteps=NSTEPS, precompute=True):
    """1-D step_x sweep with cooling OFF -> the no-cooling cliff at this nsteps."""
    _bootstrap()
    from scan_group import ScanGroup
    steps = [float(s) for s in (steps or CALIB_STEPS)]

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute)
    rk.extras.step_x.scan(1, steps)
    rk.extras.RearrCoolAmp = 0.0                # cooling OFF -- explicit, = the seq default
    rk.extras.RearrCoolDet = 0.12e6             # inert at amp 0; pinned for provenance
    return g, {"steps": steps}


def build_cool(step_knm=DEFAULT_STEP_KNM, dets_mhz=None, amps=None,
               period_ms=PERIOD_MS, nsteps=NSTEPS, precompute=True):
    """3-D [RearrCoolDet x RearrCoolAmp x step_x{0, s*}] -- the experiment + its control."""
    _bootstrap()
    from scan_group import ScanGroup
    dets = [float(d) for d in (dets_mhz or COOL_DETS_MHZ)]
    amps = [float(a) for a in (amps or COOL_AMPS)]
    steps = [0.0, float(step_knm)]              # dim3: control, then the operating point

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute)
    rk.extras.RearrCoolDet.scan(1, [d * 1e6 for d in dets])
    rk.extras.RearrCoolAmp.scan(2, amps)
    rk.extras.step_x.scan(3, steps)
    return g, {"dets_mhz": dets, "amps": amps, "steps": steps}


def build_confirm(det_mhz, amp, steps=None, period_ms=PERIOD_MS, nsteps=NSTEPS,
                  precompute=True):
    """2-D [step_x x cooling{off, (det, amp)}] -- does the whole cliff move?

    Both cooling params ride dim 2 together (same axis -> zipped, not producted), so the axis is
    exactly two cells: amp 0 (baseline) and the winning (det, amp).  The detuning value in the
    baseline cell is inert at amp 0 but is pinned to the same number so the two cells differ in
    ONE thing only."""
    _bootstrap()
    from scan_group import ScanGroup
    steps = [float(s) for s in (steps or CONFIRM_STEPS)]

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute)
    rk.extras.step_x.scan(1, steps)
    rk.extras.RearrCoolAmp.scan(2, [0.0, float(amp)])
    rk.extras.RearrCoolDet.scan(2, [float(det_mhz) * 1e6, float(det_mhz) * 1e6])
    return g, {"steps": steps, "det_mhz": float(det_mhz), "amp": float(amp)}


def build_depth(vservos=None, step_knm=DEFAULT_STEP_KNM, ramp_ms=DEPTH_RAMP_MS,
                hold_ms=DEPTH_HOLD_MS, period_ms=PERIOD_MS, nsteps=NSTEPS, precompute=True):
    """2-D [VSLMservo during transport x step_x{0, s*}] -- MAKE THE TRAP DEEPER INSTEAD.

    Every cooling attempt (641/646/649/652) failed to remove energy during the move: over det
    -0.35..+2.00 MHz and amp 0.005..0.60, across four directions and each beam separately, the
    light either harms or does nothing.  This attacks the same loss from the other side -- rather
    than cooling the atom, deepen the trap it rides in, so the same per-step displacement is a
    smaller fraction of the trap and the atom sits further from the escape threshold.  The transit
    traps are the shallow ones (the blaze-grating transit frames are far less efficient than the
    WGS endpoint), which is exactly the regime this addresses.

    Seq ``PPGTransportDepthCommSeq`` ramps ``VSLMservo`` up after img1, holds it across the
    handoff (the FPGA holds NI levels, so the whole rearrange window runs deep), then after the
    move waits ``DepthHoldMs`` and ramps back down BEFORE ``SLMStep`` -- which is load-bearing,
    since ``SLMStep`` writes ``VSLMservo = SLM.VServo`` at t=0 and would otherwise snap the depth
    back before the hold.

    ``1.9`` (= ``Init.VSLMServo`` for this array) is in the grid ON PURPOSE: that cell pays both
    ramps and the hold but moves the depth nowhere, so it is the matched-timing control and the
    comparison against it isolates depth from the ~20 ms of added sequence.
    """
    _bootstrap()
    from scan_group import ScanGroup
    vs = [float(v) for v in (vservos or DEPTH_VSERVOS)]

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute)
    rk.extras.step_x.scan(2, [0.0, float(step_knm)])
    rk.extras.TransportVServo.scan(1, vs)
    rk.extras.DepthRampMs = float(ramp_ms)
    rk.extras.DepthHoldMs = float(hold_ms)
    rk.extras.RearrCoolAmp = 0.0        # transit molasses OFF -- this experiment is not that one
    rk.extras.RearrCoolDet = 0.12e6
    return g, {"vservos": vs, "step": float(step_knm), "ramp_ms": float(ramp_ms),
               "hold_ms": float(hold_ms)}


def build_dir(thetas_deg=None, step_knm=DEFAULT_STEP_KNM, det_mhz=0.12, amp=0.04,
              period_ms=PERIOD_MS, nsteps=NSTEPS, precompute=True):
    """3-D [direction x beam-cell x step_r{0, s*}] -- IS THE MOLASSES DIRECTIONAL?

    THE QUESTION.  Jobs 641/646 found a clean harm resonance and zero cooling benefit while
    driving the X beams and the H beam TOGETHER from one amplitude and one detuning.  The 556
    recool is a two-channel molasses whose channels have different geometry, and the move deposits
    its energy along ONE lateral direction -- so a channel with poor projection on that direction
    cannot damp what the transport heats, and driving both together averages any such anisotropy
    away.  Two independent handles, both needed:

      * dim 1 -- the TRANSPORT DIRECTION (0/90/180/270 deg from +x), via ``extras.step_r`` +
        ``step_theta_deg``, folded to ``step_x``/``step_y`` per shot by ``_fold_step_polar``.  A
        list-valued ``step_size`` axis would break the lab-side grid; that is why the fold exists.
      * dim 2 -- the BEAM CELL (off / both / X only / h only / both-weak) as per-beam amplitudes
        ``extras.RearrCoolAmpX`` / ``RearrCoolAmpH``, which requires ``PPGTransportCoolCommSeq``
        (``RearrangeCommSeq`` has only the single shared amp knob).

    WHY THE OPERATING POINT IS A HARMFUL ONE.  det +0.12 MHz / amp 0.04 is where job 646 measured
    a large, high-SNR effect (ratio 0.617 against a 0.773 baseline, ~-10 sigma).  Harm is the only
    lever this system has produced that is big enough to read a geometry off, so the test uses it
    as the probe: if the harm depends on which way the array moved, the molasses is directional
    and the null is a geometry problem rather than a physics verdict.  The ``both weak`` cell
    (amp x0.25) revisits the harmless regime along the directions +x never tested.

    dim 3 ``step_r = 0`` is the usual control -- same write count, same window, same light, no
    motion -- and is direction-independent by construction, so it also cross-checks the fold.
    """
    _bootstrap()
    from scan_group import ScanGroup
    thetas = [float(t) for t in (thetas_deg or DIR_THETAS_DEG)]

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute, polar=True)   # no step_y -> the fold owns it
    rk.extras.step_theta_deg.scan(1, thetas)

    # dim 2: the beam cell. Both per-beam amps ride dim 2 together (same axis -> zipped), so the
    # axis is exactly len(DIR_BEAM_CELLS) cells rather than a product of two amp axes.
    rk.extras.RearrCoolAmpX.scan(2, [float(amp) * fx for _, fx, _ in DIR_BEAM_CELLS])
    rk.extras.RearrCoolAmpH.scan(2, [float(amp) * fh for _, _, fh in DIR_BEAM_CELLS])
    rk.extras.RearrCoolDetX = float(det_mhz) * 1e6
    rk.extras.RearrCoolDetH = float(det_mhz) * 1e6

    rk.extras.step_r.scan(3, [0.0, float(step_knm)])
    return g, {"thetas_deg": thetas, "cells": [c[0] for c in DIR_BEAM_CELLS],
               "det_mhz": float(det_mhz), "amp": float(amp), "step": float(step_knm)}


def build_temp(times_ms=None, step_knm=DEFAULT_STEP_KNM, period_ms=PERIOD_MS,
               nsteps=NSTEPS, precompute=True):
    """2-D [Cool556.Time x step_x{0, s*}] -- IS THE TRANSPORT LOSS THERMAL AT ALL?

    THE DISCRIMINATOR.  Every cooling scan here assumes the loss it is trying to fix is thermal:
    heat accumulates over the 80 steps until the atom leaves.  The competing hypothesis is
    MECHANICAL -- each step displaces the trap faster than the atom can follow and a fixed
    fraction is simply left behind, in which case the atom's temperature is irrelevant and NO
    cooling scheme, gated or not, can ever help.

    The two make opposite predictions about the PREPARED temperature.  ``Cool556.Time`` sets how
    long the pre-transport molasses runs (5 ms in production), so shortening it hands the same
    move a hotter atom:

      * thermal    -> the transport ratio DEGRADES as the prep gets shorter/hotter.
      * mechanical -> the transport ratio is FLAT in prep time (only the img1 counts move).

    The transit cooling stays OFF here; this measures the move's own temperature sensitivity.
    ``Cool556.Time`` moves every Cool556 in the shot (both bseq1 calls and the bseq2 recool), so
    a short prep also costs img1 quality -- which is exactly why the metric stays the ratio
    against the ``step_x = 0`` slab at the SAME prep time, where that cost divides out."""
    _bootstrap()
    from scan_group import ScanGroup
    times = [float(t) * 1e-3 for t in (times_ms or TEMP_TIMES_MS)]

    g = ScanGroup()
    rk = _common(g, period_ms, nsteps, precompute)
    rk.extras.RearrCoolAmp = 0.0                # transit cooling OFF -- this is not that test
    rk.extras.RearrCoolDet = 0.12e6
    g().Cool556.Time.scan(1, times)
    rk.extras.step_x.scan(2, [0.0, float(step_knm)])
    return g, {"times_ms": [t * 1e3 for t in times], "step": float(step_knm)}


_HEAD = ("RADIAL (+x) pingponggrating out-and-back, nsteps=%d OUT + %d BACK at "
         "step_period_ms=%.3f (the SLM write floor), hold_ms=0 (no turnaround dwell), "
         "return_trip default True so the array lands back on the stored WGS phase and LIVE "
         "detection is valid at every cell. Array 33x33_feedback11, z4 = loading_defocus = -4, "
         "ifEnhanced=True, precompute+precompute_host, lateral (depth/true_defocus explicitly "
         "OFF). Scale 2.45 knm px/um. ")


def _desc(mode, meta, nsteps, period_ms, reps):
    head = _HEAD % (nsteps, nsteps, period_ms)
    if mode == "calib":
        return (head +
                "CALIBRATION / BASELINE LEG of the transport-cooling test: 1-D extras.step_x %s "
                "knm px with the transit cooling OFF (extras.RearrCoolAmp = 0, the "
                "RearrangeCommSeq default). PURPOSE: (a) the no-cooling survival-vs-step curve at "
                "THIS nsteps, so the cooling scan's operating point 1.225 knm px (= 0.5 um) has a "
                "measured baseline rather than an extrapolated one, and (b) step_x = 0 is the "
                "STATIONARY COOLING-OFF control -- same SLM write count, no motion, no light -- "
                "which proves the write train itself is not costing survival before any cooling "
                "is added. Prediction from the 07-22 fine cliff (measured at nsteps=50 = 100 "
                "steps, S=0.44 at 1.30) re-raised to the 80 steps run here: ~0.94 at 1.10, ~0.74 "
                "at 1.225, ~0.51 at 1.30. The cliff drifts, so it is measured, not assumed. "
                "%d reps."
                % (",".join("%g" % s for s in meta["steps"]), reps))
    if mode == "cool":
        s = meta["steps"][1]
        return (head +
                "COOLING DURING TRANSPORT -- the experiment. 3-D: dim1 extras.RearrCoolDet %s MHz "
                "x dim2 extras.RearrCoolAmp %s x dim3 extras.step_x {0, %g} knm px (= %.3f um) = "
                "%d points, %d reps. RearrCoolDet/RearrCoolAmp are the EXISTING RearrangeCommSeq "
                "knobs (default 0.13 MHz / 0, i.e. OFF): at the end of bseq1 they set "
                "Freq/Amp556MOTX and Freq/Amp556RydbergMOTh to Resonance556mj0Freq + det / amp, "
                "and the FPGA HOLDS those across the handoff, so the 556 X+h molasses is on for "
                "the whole host-side rearrange window (img1 readout + detect + the SLM playback "
                "that IS the transport); bseq2's Cool556hXStep turns it off. Same channels and "
                "shape as the pre-imaging Cool556hXStep, so det 0.12 MHz / amp 0.14 reproduces "
                "the 33x33_feedback11 pre-imaging cooling (ByPattern Cool556 X and h both "
                "0.12/0.14) EXACTLY -- that is the starting point the wide grid is centred on. "
                "NO default, step or seq was modified; both knobs are set from this scan only. "
                "dim3 step_x = 0 is the CONTROL: same nsteps, so the same SLM write count, the "
                "same window length and the same 556 dose, but ZERO motion -- it isolates any "
                "survival cost of the light alone (the light cannot be gated to the motion: "
                "there is no FPGA timing during the handoff). The transport figure of merit is "
                "the ratio S(step=s*)/S(step=0) per (det, amp) cell. The amp=0 row is the "
                "no-cooling baseline, replicated across all %d detunings (detuning is inert at "
                "zero amp). DETUNING SIGN: positive = RED = the cooling side (all production "
                "Cool556 values are positive), negative = BLUE = heating. The grid STRADDLES the "
                "line on purpose: the wanted signature is HARM on the blue side and HELP on the "
                "red side, because an antisymmetric response across a resonance can only be "
                "cooling vs heating, whereas a one-sided gain could be a trivial trap/servo or "
                "detection artifact. The axis zero is the FULL-DEPTH static mj=0 resonance while "
                "the traps are shallower in transit, so the true line may sit anywhere inside "
                "this window -- that is why it runs from -0.35 (well blue) to +0.70 (well past "
                "the static +0.12 optimum) rather than being centred on the static value."
                % (",".join("%g" % d for d in meta["dets_mhz"]),
                   ",".join("%g" % a for a in meta["amps"]),
                   s, s / KNM_PER_UM,
                   len(meta["dets_mhz"]) * len(meta["amps"]) * 2, reps,
                   len(meta["dets_mhz"])))
    if mode == "depth":
        return (head +
                "RAISE THE TRAP DEPTH DURING TRANSPORT (seq PPGTransportDepthCommSeq). 2-D: dim1 "
                "extras.TransportVServo %s V x dim2 extras.step_x {0, %g} knm px = %d points, %d "
                "reps. THE SEQ: after img1 the 532 servo VSLMservo is ramped (ramp_to quintic "
                "smootherstep) from Init.VSLMServo to TransportVServo over %g ms; the FPGA HOLDS "
                "that level across the bseq1->bseq2 handoff, so the entire rearrange window "
                "(detect + compute + the SLM playback that IS the transport) runs at the new "
                "depth; after the move it waits %g ms at depth and ramps back BEFORE SLMStep -- "
                "load-bearing, since SLMStep writes VSLMservo = SLM.VServo at t=0 and would "
                "otherwise snap the depth back before the hold. MOTIVATION: cooling during "
                "transport has now failed across det -0.35..+2.00 MHz and amp 0.005..0.60 (jobs "
                "641/646/652), across four directions and each 556 beam separately (job 649) -- "
                "harm or nothing, never help -- while jobs 647 and the 07/22 calorimetry show the "
                "loss IS thermal. So instead of removing energy from the atom, raise the barrier "
                "it has to clear: the transit frames are blaze-grating holograms and far less "
                "efficient than the WGS endpoint, so the moving traps are the shallow ones. "
                "CONTROL: V = %g (= Init.VSLMServo for 33x33_feedback11) is IN the grid and pays "
                "both ramps plus the hold while moving the depth nowhere, so it isolates depth "
                "from the ~%g ms of added sequence; dim2 step_x = 0 is the usual no-motion slab. "
                "The transit molasses is OFF (RearrCoolAmp = 0) -- this is not that experiment. "
                "CAVEAT on the V = 0 and 0.25 cells: VSLMservo is a servo SETPOINT and may sit "
                "below its regulation floor there (the 399 imaging PID has that pathology), so "
                "those two may not deliver the commanded power and may cost a recovery transient; "
                "they are included as the trap-off anchor, not as precision points."
                % (",".join("%g" % v for v in meta["vservos"]), meta["step"],
                   len(meta["vservos"]) * 2, reps, meta["ramp_ms"], meta["hold_ms"],
                   1.9, 2 * meta["ramp_ms"] + meta["hold_ms"]))
    if mode == "dir":
        return (head +
                "IS THE TRANSIT MOLASSES DIRECTIONAL? Seq PPGTransportCoolCommSeq -- a copy of "
                "RearrangeCommSeq whose ONLY change is per-beam transit-cool knobs "
                "(extras.RearrCoolAmpX/AmpH + DetX/DetH, each defaulting to the shared "
                "RearrCoolAmp/Det, so it is byte-identical to the parent when they are unset). "
                "3-D: dim1 transport DIRECTION extras.step_theta_deg %s deg from +x (folded to "
                "step_x/step_y per shot by _fold_step_polar with extras.step_r as the radius) x "
                "dim2 BEAM CELL {%s} (the two per-beam amps ride dim2 together, so the axis is "
                "exactly %d cells) x dim3 extras.step_r {0, %g} knm px = %d points, %d reps. "
                "MOTIVATION: jobs 641 (data_20260809_225051) and 646 (data_20260810_005813) "
                "covered det -0.35..+0.70 MHz x amp 0.005..0.25 and found a clean harm resonance "
                "but ZERO cooling benefit -- both driving 556MOTX (the X1+X2 beams, diagonal in a "
                "plane containing the tweezer axis) and 556RydbergMOTh (the horizontal MOT-H "
                "beam) TOGETHER at a common amp and detuning. A molasses channel with poor "
                "projection on the transport axis cannot damp the degree of freedom the move "
                "heats, and driving both together averages any such anisotropy away, so the "
                "geometry has never been tested. OPERATING POINT det %+.2f MHz / amp %.3f is "
                "deliberately a HARMFUL cell (job 646 measured ratio 0.617 vs a 0.773 baseline, "
                "about -10 sigma): harm is the only large, high-SNR lever this system has "
                "produced, so it is used as the probe -- if it depends on which way the array "
                "moved, the molasses is directional and the null is a geometry problem, not a "
                "physics verdict. The `both weak` cell (amp x0.25) revisits the harmless regime "
                "along the directions +x never tested. dim3 step_r=0 is the usual no-motion "
                "control (same write count, same window, same light), direction-independent by "
                "construction, so it also cross-checks the polar fold."
                % (",".join("%g" % t for t in meta["thetas_deg"]), ", ".join(meta["cells"]),
                   len(meta["cells"]), meta["step"],
                   len(meta["thetas_deg"]) * len(meta["cells"]) * 2, reps,
                   meta["det_mhz"], meta["amp"]))
    if mode == "temp":
        return (head +
                "THERMAL-vs-MECHANICAL DISCRIMINATOR for the transport-cooling campaign, transit "
                "cooling OFF (extras.RearrCoolAmp = 0): 2-D dim1 Cool556.Time %s ms x dim2 "
                "extras.step_x {0, %g} knm px, %d reps. THE QUESTION: every cooling scan assumes "
                "the transport loss is THERMAL (heat accumulates over the 80 steps until the atom "
                "leaves). The competing hypothesis is MECHANICAL -- each step outruns the atom and "
                "a fixed fraction is left behind regardless of temperature, in which case NO "
                "cooling scheme, gated or not, can help. Cool556.Time sets how long the "
                "pre-transport molasses runs (5 ms in production), so a shorter prep hands the "
                "same move a HOTTER atom: thermal => the transport ratio S(move)/S(step=0) "
                "degrades as the prep shortens; mechanical => that ratio is FLAT and only the img1 "
                "counts move. Cool556.Time moves every Cool556 in the shot (both bseq1 calls and "
                "the bseq2 recool), so a short prep also costs img1 quality -- which is why the "
                "metric is the ratio against the step_x=0 slab at the SAME prep time, where that "
                "cost divides out. Motivation: job 641 (data_20260809_225051) found NO cooling "
                "benefit at any of 10 detunings x 3 powers; its lowest power was still ~30x the "
                "integrated dose of the 5 ms recool that works, so the next question is whether "
                "this loss is thermally addressable at all."
                % (",".join("%g" % t for t in meta["times_ms"]), meta["step"], reps))
    return (head +
            "CONFIRMATION LEG of the transport-cooling test: 2-D dim1 extras.step_x %s knm px x "
            "dim2 cooling {OFF, det %g MHz / amp %g} (both cooling params ride dim2 together, so "
            "the axis is exactly two cells and they differ in ONE thing -- the amp), %d reps. "
            "PURPOSE: the winning cell from the cooling scan is a single operating point; this "
            "asks whether the whole survival-vs-step CLIFF moves with cooling on, which is the "
            "proof-of-concept plot. step_x=0 is the no-motion control row."
            % (",".join("%g" % s for s in meta["steps"]),
               meta["det_mhz"], meta["amp"], reps))


def main():
    ap = argparse.ArgumentParser(
        description="556 cooling DURING radial pingponggrating transport (proof of concept).")
    ap.add_argument("--mode",
                    choices=("calib", "cool", "confirm", "temp", "dir", "depth"),
                    default="calib")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP_KNM,
                    help="cool mode: the operating step size in knm px (from the calib fit)")
    ap.add_argument("--step-um", type=float, default=None,
                    help="cool mode: the operating step in um (converted at %g knm px/um)"
                         % KNM_PER_UM)
    ap.add_argument("--steps", default=None, help="calib/confirm: comma-separated knm px")
    ap.add_argument("--dets", default=None, help="cool: comma-separated detunings, MHz")
    ap.add_argument("--amps", default=None, help="cool: comma-separated 556 amps")
    ap.add_argument("--det", type=float, default=None, help="confirm: the winning detuning, MHz")
    ap.add_argument("--amp", type=float, default=None, help="confirm: the winning 556 amp")
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--times", default=None,
                help="temp mode: comma-separated Cool556.Time values, ms")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--vservos", default=None,
                    help="depth mode: comma-separated VSLMservo values")
    ap.add_argument("--ramp-ms", type=float, default=DEPTH_RAMP_MS,
                    help="depth mode: adiabatic ramp length, ms (default %g)"
                         % DEPTH_RAMP_MS)
    ap.add_argument("--hold-ms", type=float, default=DEPTH_HOLD_MS,
                    help="depth mode: hold at depth after the move, ms (default %g)"
                         % DEPTH_HOLD_MS)
    ap.add_argument("--no-precompute", action="store_true",
                    help="compute each transit frame in the loop instead of baking "
                         "them at setup; VERIFY the server diag pacing afterwards")
    ap.add_argument("--url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    steps = ([float(x) for x in args.steps.split(",") if x.strip()] if args.steps else None)
    reps = args.reps if args.reps else {"calib": 8, "cool": 6, "confirm": 10,
                                       "temp": 12, "dir": 10, "depth": 12}[args.mode]

    if args.mode == "calib":
        g, meta = build_calib(steps=steps, period_ms=args.period, nsteps=args.nsteps,
                              precompute=not args.no_precompute)
        label = "PPGTransportCool_calib"
    elif args.mode == "cool":
        step = args.step_um * KNM_PER_UM if args.step_um is not None else args.step
        dets = ([float(x) for x in args.dets.split(",") if x.strip()] if args.dets else None)
        amps = ([float(x) for x in args.amps.split(",") if x.strip()] if args.amps else None)
        g, meta = build_cool(step_knm=step, dets_mhz=dets, amps=amps,
                             period_ms=args.period, nsteps=args.nsteps,
                             precompute=not args.no_precompute)
        label = "PPGTransportCool_s%g" % step
    elif args.mode == "depth":
        step = args.step_um * KNM_PER_UM if args.step_um is not None else args.step
        vs = ([float(x) for x in args.vservos.split(",") if x.strip()]
              if args.vservos else None)
        g, meta = build_depth(vservos=vs, step_knm=step, ramp_ms=args.ramp_ms,
                              hold_ms=args.hold_ms, period_ms=args.period,
                              nsteps=args.nsteps, precompute=not args.no_precompute)
        label = "PPGTransportDepth"
    elif args.mode == "dir":
        step = args.step_um * KNM_PER_UM if args.step_um is not None else args.step
        g, meta = build_dir(step_knm=step,
                            det_mhz=(args.det if args.det is not None else 0.12),
                            amp=(args.amp if args.amp is not None else 0.04),
                            period_ms=args.period, nsteps=args.nsteps,
                            precompute=not args.no_precompute)
        label = "PPGTransportCool_dir"
    elif args.mode == "temp":
        step = args.step_um * KNM_PER_UM if args.step_um is not None else args.step
        times = ([float(x) for x in args.times.split(",") if x.strip()]
                 if args.times else None)
        g, meta = build_temp(times_ms=times, step_knm=step,
                             period_ms=args.period, nsteps=args.nsteps,
                             precompute=not args.no_precompute)
        label = "PPGTransportCool_temp"
    else:
        if args.det is None or args.amp is None:
            ap.error("--mode confirm needs --det and --amp (the winning cooling cell)")
        g, meta = build_confirm(args.det, args.amp, steps=steps,
                                period_ms=args.period, nsteps=args.nsteps,
                                precompute=not args.no_precompute)
        label = "PPGTransportCool_confirm"

    if args.no_precompute:
        label += "_noprecomp"
    desc = _desc(args.mode, meta, args.nsteps, args.period, reps)
    if args.no_precompute:
        desc += (" PRECOMPUTE IS OFF for this run (extras.precompute = precompute_host = False): "
                 "every transit frame is computed inside the motion loop instead of being baked "
                 "at setup. Purpose: raise the fraction of the rearrange window during which the "
                 "traps are actually DISPLACED (job 641 measured 57 ms of motion inside a ~0.3 s "
                 "light-on window = ~19%% duty, which is why an ungated molasses is ~30x "
                 "over-dosed). CHECK BEFORE COMPARING ANYTHING: the SLM server diag for this scan "
                 "(paced_total_ms, write_lat_ms, n_frames) says whether the commanded 0.696 ms "
                 "pacing still HELD -- if per-frame compute overran it, the move itself became "
                 "slower and the transport physics moved with it. The step_x=0 control and the "
                 "no-cooling baseline ratio (0.9897 / 0.780 with precompute ON) are the other "
                 "check: if either moves, this is not a like-for-like comparison.")
    npts = g.nseq()
    print("mode=%s  nseq=%d  nsteps=%d (out+back = %d steps)  period=%g ms  reps=%d"
          % (args.mode, npts, args.nsteps, 2 * args.nsteps, args.period, reps))
    for k, v in meta.items():
        print("  %-10s %s" % (k, v))
    print("shots: %d  (~%.2f h at 3 s/shot)" % (npts * reps, npts * reps * 3.0 / 3600))
    if args.mode == "cool":
        s = meta["steps"][1]
        print("  operating point: %g knm px = %.3f um/step; total stroke %g knm px = %.2f um"
              % (s, s / KNM_PER_UM, s * args.nsteps, s * args.nsteps / KNM_PER_UM))

    if args.dry_run:
        print("DRY RUN -- not submitted.")
        return
    if not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")

    from yb_start_scan import ybStartScan
    # `dir` needs the per-beam transit-cool knobs, which only the new seq has.
    seq_name = {"dir": "PPGTransportCoolCommSeq",
                "depth": "PPGTransportDepthCommSeq"}.get(args.mode,
                                                        "RearrangeCommSeq")
    did = ybStartScan(seq_name, g, url=args.url, label=label,
                      description=desc, rep=reps)
    print("submitted %s [%s] -> descriptor id %s (%d pts, %d reps = %d shots)"
          % (label, seq_name, did, npts, reps, npts * reps))
    return did


if __name__ == "__main__":
    main()
