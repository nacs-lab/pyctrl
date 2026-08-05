"""SLM3DArrayImagingOptScan.py -- 399 imaging power optimization for the 2-layer 20 um array.

Imaging has never been optimized on ``phase/2x11x11_5um_z20um.pt``, and every 3-D measurement
(the focus walk, the layer-move verification, the diagonal rearrangement) reads its answer out
of that detection.  Do this FIRST.

WHY THIS ARRAY NEEDS ITS OWN OPTIMUM
  242 sites vs the production 1068, so at the same total 399 power the per-site intensity is
  ~4.4x higher -- the production ``Imag399.Amp1/Amp2`` are very likely too HOT here, and the
  useful part of the sweep is the low end.  (AOM knee measured 2026-07-16: DDS amp 0.5-1.0 is
  optically FLAT, only <= 0.5 actually attenuates.  So sweep amps, never compute a ratio.)
  On top of that the OTHER layer is 20 um out and contributes a diffuse halo over the whole
  field -- a background pedestal the production array simply does not have, which shifts the
  optimum further.

WHY A PLAIN g() OVERRIDE IS CORRECT HERE (unlike SLMRearrangeImagingOptScan)
  That scan needs per-frame extras (MidImgAmp1/FinImgAmp1) because its three frames image
  DIFFERENT patterns and must be optimized separately.  Here all three frames image the SAME
  121-site grid at the same plane (the walk puts the front layer exactly where the back layer
  was), so one setting should serve all of them and ``g().Imag399.Amp*`` -- which by precedence
  wins in every bseq and moves all frames together -- is exactly what we want.

  Run in WALK-ONLY mode: 2 frames (back layer, then front layer after the walk), no
  rearrangement.  That is enough to score detection on both layers and keeps the shot short.

METRIC -- read d-prime, not brightness.  Judge by per-site d-prime (separation / noise) and the
logicals correlation, NOT absolute ADU or raw-frame max: a dim image with clean separation
beats a bright one with a smeared histogram (yb_skills memory
``gotcha-detection-health-dprime-not-abs-adu``).  Both frames must be scored -- the front-layer
frame sits on the back layer's halo and may well want a different amp than the back frame, in
which case the two-frame compromise is itself the result.

ORDER (yb_skills memory ``feedback-power-before-cooling-per-detuning``): power FIRST, at fixed
detuning/exposure.  Only once the amp is settled is it worth touching cooling or detuning.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/SLM3DArrayImagingOptScan.py --dry-run
    python YbScans/RearrangeDiagnostics/SLM3DArrayImagingOptScan.py --force
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))
BASE_SCAN = os.path.join(PYCTRL, "YbScans", "SLMRearrangement3DFocusWalkScan.py")

# (Amp1, Amp2) pairs, swept together.  Weighted to the LOW end: 242 sites means ~4.4x the
# per-site power of the 1068-site production array at the same amp, and >= 0.5 is optically
# flat anyway, so 1.0 and 0.5 are effectively the same point optically and both are included
# only as the "today's production setting" anchor.
AMP_PAIRS = [(1.0, 1.0), (0.5, 0.5), (0.35, 0.35), (0.25, 0.25), (0.18, 0.18), (0.12, 0.12)]


def _load_base():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("focuswalk", BASE_SCAN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(pairs=None, load_defocus=None, walk_steps=None):
    m = _load_base()
    pairs = list(AMP_PAIRS if pairs is None else pairs)
    # walk_only: 2 frames (back, then front after the walk).  No rearrangement -- this is a
    # detection measurement, and the shorter shot means more statistics per unit time.
    seq_name, g = m.build(walk_only=True, load_defocus=load_defocus, walk_steps=walk_steps)
    g().Imag399.Amp1.scan(1, [p[0] for p in pairs])
    g().Imag399.Amp2.scan(1, [p[1] for p in pairs])
    return seq_name, g, m, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=360)
    ap.add_argument("--amps", default=None,
                    help="comma-separated amp values, applied to BOTH Amp1 and Amp2")
    ap.add_argument("--load-defocus", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    pairs = ([(float(a), float(a)) for a in args.amps.split(",") if a.strip()]
             if args.amps else None)
    seq_name, g, m, pairs = build(pairs=pairs, load_defocus=args.load_defocus)
    g.runp().NumPerGroup = int(args.num_per_group)
    print("seq=%s nseq=%d  amp pairs=%s" % (seq_name, g.nseq(), pairs))
    print("  array %s, back layer at the camera plane (carrier %+.4f), walk %+.4f rad"
          % (m.PHASE_PATH, m.LOADING_DEFOCUS, m.WALK_TOTAL_RAD))
    if args.dry_run:
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    desc = (
        "399 IMAGING POWER OPTIMIZATION for the 2-layer 20 um array %s -- never done for this "
        "pattern, and every 3-D measurement reads its answer out of this detection, so it goes "
        "first.  Focus-walk seq in WALK-ONLY mode: 2 frames, BACK layer in focus at carrier "
        "%+.4f then FRONT layer after the %+.4f rad phase-only axial walk, no rearrangement.  "
        "Sweeps g().Imag399.Amp1/Amp2 together over %s.  A plain g() override is the right knob "
        "here (unlike SLMRearrangeImagingOptScan's per-frame extras) because all frames image "
        "the SAME 121-site grid at the same plane.  Weighted to the LOW end: 242 sites is ~4.4x "
        "the per-site power of the 1068-site production array at equal amp, and the AOM knee "
        "(2026-07-16) makes >= 0.5 optically flat, so production settings are probably too hot "
        "here.  Extra wrinkle this array has and production does not: the out-of-focus layer "
        "sits 20 um away and lays a diffuse halo pedestal across the field, which should push "
        "the optimum lower still and may differ between the two frames.  JUDGE BY PER-SITE "
        "D-PRIME and logicals correlation, not absolute ADU (memory "
        "gotcha-detection-health-dprime-not-abs-adu).  Power first, then cooling/detuning "
        "(memory feedback-power-before-cooling-per-detuning)."
        % (m.PHASE_PATH, m.LOADING_DEFOCUS, m.WALK_TOTAL_RAD,
           [p[0] for p in pairs]))
    did = ybStartScan(seq_name, g, url=args.url,
                      label="SLM3DArrayImagingOptRearrangeScan", description=desc, **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
