"""SLM3DCarrierFocusScan.py -- ARE THE COMMANDED AXIAL OFFSETS REAL?  Scan the global carrier
across BOTH layers of the 20 um array and fit the two focus centres; their separation is the
REALIZED layer gap, in the same radians it was commanded in.

WHY THIS REPLACES THE ARRIVAL TEST (SLM3DLayerMoveVerifyScan, cancelled 2026-07-31)
  That scan asked "did the moved sites show up at the other plane, yes or no".  Three problems the
  user pointed out, all fatal to it and all fixed here:

  1. IMAGING TOLERATES +-5 rad of defocus (slightly lower quality, but atoms are still detected).
     A yes/no arrival test therefore has ~10 rad of dead band -- it cannot distinguish "moved
     25.06 rad" from "moved 21 rad", which is exactly the discrepancy worth finding.  The fix is
     the user's: do not threshold, SCAN and FIT THE CENTRE.  A focus response ~10 rad wide whose
     centre is located to a fraction of its width beats a hard edge that does not exist.
  2. ISOLATING THE MOVERS (ghost_fraction=0) CHANGES TRAP DEPTH -- 61 traps carry ~4x the per-site
     power of 242, and imaging is depth-sensitive, so the isolated frame is not imaged under the
     same conditions as the reference.  Worse, imaging the moved state at all requires
     skip_final_phase (the WGS bookend would otherwise snap the movers back), which locks in that
     sparse frame.  This scan sidesteps the whole issue: NOTHING is isolated and NOTHING is moved
     per-site.  The entire array translates together under ``pingponggrating depth`` -- every trap
     present, full density, unchanged depth, and no bookend to skip (grating mode writes
     WGS_initial + k*step*Z4 straight to the panel; there is no model frame and no final-phase
     write to suppress).
  3. LOADING DEFOCUS IS ALREADY OPTIMIZED (aligned to the atom-camera focus) and must not be
     casually re-pointed.  Here it is left exactly at the scan's calibrated value; the thing that
     scans is the WALK, i.e. a phase ramp applied after loading.  As a bonus the same fit
     re-measures where the camera plane actually is (see OUTPUT 3).

THE MEASUREMENT
  ``phase/2x11x11_5um_z20um.pt`` has two 11x11 layers at z4 -+%(half).4f rad, xy-ALIGNED -- the two
  layers land in the SAME 121 camera boxes.  Load both, image (img1 = reference at the loading
  carrier), then walk the whole array by w radians and image again (img2).  As w scans:

      w ~ 0          -> the BACK layer sits at the camera plane   -> img2 occupancy peaks
      w ~ -%(gap).4f -> the FRONT layer sits there instead        -> img2 occupancy peaks again

  So img2 occupancy vs w is a TWO-PEAK curve in one scan, and the peak separation is the realized
  gap.  Both peaks come from the same boxes, the same shot structure, the same trap depth and the
  same detection thresholds, so essentially every systematic is common-mode and cancels out of the
  separation.  Fit each peak's centre (the curve is smooth and roughly symmetric -- the +-5 rad
  imaging tolerance is what gives it width, and width is not the observable).

WHAT IT DOES AND DOES NOT PIN DOWN
  Both the carrier and the layer offset are ANSI Z4 phase, so the separation is measured in radians
  against radians: this tests whether the WGS phase's baked-in per-spot z ACTUALLY displaces the
  focus by what a global Z4 of the same size does -- a genuine and unverified property of a 3-D
  hologram, and the direct form of "are we realizing the steps we claim".  It does NOT convert to
  microns; nothing here is a physical ruler, and a common scale error would cancel.  The micron
  anchor is the separate ``PPGAxialLegDefocusCalibScan`` null, which measures Z4 against the exact
  spherical map at known lambda/NA.  Run both; they answer different halves of the question.

OUTPUTS
  1. peak separation  ->  realized / commanded layer gap  (expect %(gap).4f rad if perfect)
  2. peak widths      ->  the axial imaging depth of field, measured rather than assumed
  3. the BACK peak's own centre -> where the camera plane actually is relative to the calibrated
     loading defocus.  Nominally the back layer is placed at Z_CAM_DEFOCUS %(zcam)+.1f so the peak
     should sit at w = 0; a nonzero centre is the residual focus error of that placement, which is
     worth knowing before any 3-D imaging optimization is trusted.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/SLM3DCarrierFocusScan.py --dry-run
    python YbScans/RearrangeDiagnostics/SLM3DCarrierFocusScan.py --force
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))
BASE_SCAN = os.path.join(PYCTRL, "YbScans", "SLMRearrangement3DFocusWalkScan.py")

# Walk amplitudes w (radians of PV ANSI Z4) applied AFTER loading.  The range brackets both layers
# with margin: the back layer should peak near 0 and the front near -25.06.  Spacing ~1.85 rad is
# comfortably finer than the ~10 rad focus width, which is what makes a centre fit meaningful.
W_MIN = +5.0
W_MAX = -30.0
N_W = 20

WALK_STEPS = 50          # pacing held fixed so only the amplitude changes across the axis
NUM_PER_GROUP = 600      # 20 cells -> 30 shots/cell x 121 sites x 2 layers loaded


def _load_base():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("focuswalk", BASE_SCAN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _amplitudes(n=N_W, w0=W_MIN, w1=W_MAX):
    if n < 2:
        raise ValueError("need >= 2 amplitudes to fit a centre")
    return [w0 + (w1 - w0) * i / float(n - 1) for i in range(n)]


def build(amps=None, walk_steps=WALK_STEPS):
    """Built inline rather than through ``SLMRearrangement3DFocusWalkScan.build()``: that helper
    pins ``step_size`` as a FIXED param (it is a constant of the nominal walk there), and ScanGroup
    refuses to turn a fixed parameter into a swept one.  Everything below is the walk-only branch
    of that scan verbatim -- same array, same carriers, same detection, same imaging params -- with
    ``step_size`` promoted to the scan axis.  Constants are imported from it so the two cannot
    drift apart."""
    m = _load_base()
    from scan_group import ScanGroup

    ws = list(amps if amps else _amplitudes())
    g = ScanGroup()
    rp = g.runp()

    g().focus_walk.n_per_plane = m.N_PER_PLANE

    rp.warmup_kwargs.model_filename = m.MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = m.PHASE_PATH
    rp.warmup_kwargs.final_phase = m.PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [0, 0, 0, 0, 0]
    rp.warmup_kwargs.extras.final_phase_zernike = [0, 0, 0, 0, 0]
    rp.warmup_kwargs.extras.planes_z_rad = list(m.PLANES_Z_RAD)
    rp.warmup_kwargs.extras.dedup_xy_knm = 0          # layers are xy-coincident: ANY dedup merges
    rp.warmup_kwargs.derive_threshold = m.DERIVE_THRESHOLD
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True

    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.nsteps = int(walk_steps)
    rk.step_period_ms = m.WALK_PERIOD_MS
    rk.extras.depth = True
    # The ONLY scanned quantity: the per-step stroke, i.e. the total walk w = nsteps * step_size.
    # nsteps stays fixed so the pacing (and any pacing-dependent loss) is identical in every cell
    # and cannot masquerade as a focus response.
    rk.extras.step_size.scan(1, [w / float(walk_steps) for w in ws])
    rk.extras.return_trip = False                     # rest on the walked frame -- img2/img3 there
    rk.extras.no_depth_piston = True
    rk.extras.depth_fill_frac = m.DEPTH_FILL_FRAC
    rk.extras.piston = 0.0
    rk.extras.precompute = True
    rk.extras.n_rounds = 1
    rk.extras.ifEnhanced = False
    rk.extras.walk_only = True                        # seq: skip setup #2 and the rearrangement
    rk.extras.initial_pattern = m.DETECT_PATTERN
    rk.extras.middle_pattern = m.DETECT_PATTERN
    rk.extras.final_pattern = m.DETECT_PATTERN

    # Array-specific imaging/loading params (no expConfig ByPattern entry for this name yet).
    g().Init.VSLMServo = 0.39
    g().BlueMOT.Img1PIDSet = float(os.environ.get("YB_IMG1PID", "0.625"))
    g().BlueMOT.Img2PIDSet = float(os.environ.get("YB_IMG2PID", "0.35"))
    g().LAC.FreqDetuning = 0.11e6
    g().LAC.Amp = 0.2
    g().LAC.Time = 30e-3
    g().Imag399.Cool556.X.FreqDetuning = 0.14e6
    g().Imag399.Cool556.X.Amp = 0.26
    g().Imag399.Cool556.h.FreqDetuning = 0.13e6
    g().Imag399.Cool556.h.Amp = 0.20
    g().Cool556.X.FreqDetuning = 0.16e6
    g().Cool556.X.Amp = 0.14
    g().Cool556.h.FreqDetuning = 0.16e6
    g().Cool556.h.Amp = 0.12

    rp.loading_defocus = m.LOADING_DEFOCUS            # calibrated -- NOT a knob here
    rp.NumImages = 3
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.isRearrange = 1
    rp.imagePatternsJson = m._image_patterns_json()
    return "Rearrange3DFocusWalkCommSeq", g, m, ws, walk_steps


def _desc(m, ws, walk_steps):
    return (
        "AXIAL REALIZATION via a CARRIER FOCUS SCAN on the 2-layer 20 um array %s (2x11x11, 5 um "
        "pitch, xy-ALIGNED layers at z4 %+.4f / %+.4f rad, nominal gap %.4f rad).  Focus-walk seq "
        "in WALK-ONLY mode (no rearrangement, no model, no per-site motion): load both layers at "
        "the calibrated carrier %+.4f -> img1 (reference) -> pingponggrating depth walk of w rad "
        "(%d steps x w/%d, return=False, no_depth_piston, fill %.3f, precompute, %.3f ms/frame) -> "
        "img2, img3.  THE SCAN AXIS IS w: %s rad.  Because the layers are xy-aligned they share the "
        "same 121 camera boxes, so img2 occupancy vs w is a TWO-PEAK curve -- the BACK layer is at "
        "the camera plane near w=0 and the FRONT layer near w=%.4f -- and the PEAK SEPARATION is "
        "the REALIZED layer gap measured against the same radians it was commanded in.  Everything "
        "else is common-mode between the two peaks (same boxes, same thresholds, same trap depth, "
        "same pacing), so it cancels.  WHY NOT AN ARRIVAL TEST: imaging tolerates ~+-5 rad of "
        "defocus, so a yes/no 'did it arrive' check has a ~10 rad dead band and cannot resolve the "
        "discrepancy worth finding; the centre of a broad response is far better determined than "
        "its edges.  WHY NOTHING IS ISOLATED: dropping non-movers (ghost_fraction=0) would raise "
        "per-trap power ~4x on 61 of 242 sites, and imaging is trap-depth sensitive, so the moved "
        "frame would not be imaged under reference conditions; imaging a moved state at all also "
        "forces skip_final_phase.  A global grating walk avoids all of it -- full density, "
        "unchanged depth, and grating mode writes WGS_initial + k*step*Z4 directly with no model "
        "frame and no bookend.  The calibrated loading defocus is left untouched; only the "
        "post-load ramp scans.  SCOPE: this compares Z4 against Z4 and does NOT convert to microns "
        "(a common scale error cancels) -- the micron anchor is the separate leg_defocus null, "
        "which measures Z4 against the exact spherical map at known lambda/NA.  Secondary outputs: "
        "the peak WIDTHS give the axial depth of field directly, and the BACK peak's centre gives "
        "the residual focus error of the %+.1f rad camera-plane placement (it should sit at w=0)."
        % (m.PHASE_PATH, m.BACK_LAYER_Z, m.FRONT_LAYER_Z, abs(m.WALK_TOTAL_RAD), m.LOADING_DEFOCUS,
           walk_steps, walk_steps, m.DEPTH_FILL_FRAC, m.WALK_PERIOD_MS,
           ["%+.2f" % w for w in ws], m.WALK_TOTAL_RAD, m.Z_CAM_DEFOCUS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--w-min", type=float, default=W_MIN)
    ap.add_argument("--w-max", type=float, default=W_MAX)
    ap.add_argument("--n-w", type=int, default=N_W)
    ap.add_argument("--walk-steps", type=int, default=WALK_STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    amps = _amplitudes(args.n_w, args.w_min, args.w_max)
    seq_name, g, m, ws, walk_steps = build(amps=amps, walk_steps=args.walk_steps)
    g.runp().NumPerGroup = int(args.num_per_group)
    print("seq=%s nseq=%d   array %s   loading carrier %+.4f (back layer %+.4f -> camera plane "
          "%+.1f)" % (seq_name, g.nseq(), m.PHASE_PATH, m.LOADING_DEFOCUS, m.BACK_LAYER_Z,
                      m.Z_CAM_DEFOCUS))
    print("  walk amplitudes w (rad): %s" % ", ".join("%+.2f" % w for w in ws))
    print("  expect img2 occupancy peaks near w = 0.00 (back layer) and w = %+.4f (front layer)"
          % m.WALK_TOTAL_RAD)
    print("  %d steps/walk -> %.4f..%.4f rad/step, %.1f ms per walk"
          % (walk_steps, min(ws) / walk_steps, max(ws) / walk_steps,
             (walk_steps + 1) * m.WALK_PERIOD_MS))
    if args.dry_run:
        seen = sorted({round(g.getseq(i)["rearrange_kwargs"]["extras"]["step_size"] * walk_steps, 4)
                       for i in range(g.nseq())})
        print("  %d distinct total walks: %s" % (len(seen), seen))
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label="SLM3DCarrierFocusRearrangeScan",
                      description=_desc(m, ws, walk_steps), **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
