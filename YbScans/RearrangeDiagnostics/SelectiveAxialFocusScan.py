"""SelectiveAxialFocusScan.py -- SOME tweezers move axially, the rest do not, in the SAME image.

THE POINT
  The claim to demonstrate is **arbitrary per-site axial control**, not "the array can be
  translated".  A global carrier walk moves everything together and proves nothing about
  addressability -- it would look identical if the per-site z channel were ignored entirely.  Here
  a chosen SUBSET of the 1068 production tweezers is commanded to a depth offset dz while every
  other site is commanded to stay, and both populations are read out of the SAME frame.  At large
  dz the site map is a checkerboard: movers defocused and dark, non-movers sharp.  That image IS
  the demonstration; the numbers below are what makes it quantitative.

WHY THIS RUNS ON THE PRODUCTION 33x33 ARRAY
  Single layer, so nothing else is in the field to confuse a defocus readout; a calibrated affine
  and real per-site thresholds already exist (the 2-layer array has neither -- see
  ``_imaging_det/bootstrap_pattern_thresholds.py``); ~1068 sites per shot, split ~534/534 between
  the two populations, so both arms are measured at full statistics simultaneously.

THE OBSERVABLE AND WHY IT IS A RATIO
  Imaging tolerates roughly +-5 rad of defocus, so occupancy vs commanded dz is a smooth peak
  centred at dz = 0 whose half-width is the axial depth of field DIVIDED BY the realization factor
  alpha (= realized displacement / commanded displacement).  A width alone cannot separate the two.
  So dim 2 runs the SAME protocol twice with only the mover set changed:

    * ``move_idx = ALL``    -- every site moves.  Identical to a global axial shift, so its width
                              is the depth of field itself, measured on these sites with this
                              detection on these shots.
    * ``move_idx = SUBSET`` -- half the sites move.  Same producer, same pacing, same amplitude.

  ``alpha`` is then the RATIO of the two widths, and essentially every systematic (threshold
  placement, imaging quality, loading, shot-to-shot brightness) is common-mode and divides out.
  The subset arm also carries its own internal control: the non-movers in the very same frame must
  stay flat across the whole dz axis.  If they do not, the per-site channel is leaking into sites
  it was never commanded to touch -- which would be a far more interesting result than the
  calibration, and is exactly the failure a global scan cannot see.

WARM PRODUCER, NOT THE MODEL -- this one is load-bearing
  The 3-D SLMnet model quantizes per-spot z onto a 1 um = **1.105 rad** ladder
  (``rearrange_actual.py`` ``Z_UM_MAX``/``round(z*UM_PER_RAD).clamp``), which would round the very
  quantity being measured: a commanded 1.5 rad becomes 1 or 2 ladder units.  Tonight's job 420-vs-433
  comparison also puts the model behind the warm producer above ~2 rad/step.  So ``wgs3d_warm=True``.

  ``nsteps = %(n)d`` with the largest amplitude gives <= %(mx).2f rad/step -- far under both the
  ~2.25 rad per-frame axial cliff and the 4.8 rad/step where tonight's warm runs still showed zero
  per-step loss, so transport cost is negligible and what is left is pure defocus.

TRAP DEPTH (the 2026-07-31 correction)
  ``ghost_fraction = 1.0``: every site keeps a trap, including the ones that do not move.  Dropping
  non-movers would concentrate the same power into fewer traps and change imaging, which is
  depth-sensitive -- and it would also destroy the in-image reference population this measurement
  is built on.  ``skip_final_phase = True`` is unavoidable (the WGS bookend would snap the movers
  back before img2), so the imaged hologram is the producer's frame rather than WGS; that is
  identical for both arms and both populations, and img1 normalizes loading out.

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/SelectiveAxialFocusScan.py --dry-run
    python YbScans/RearrangeDiagnostics/SelectiveAxialFocusScan.py --force
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_2D = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0
PERIOD_MS = 0.696
N_SITES = 1068

# dim 1: commanded axial offset dz (rad of PV ANSI Z4), signed -- the two directions are not
# equivalent (job 428 found the piston null itself direction-dependent) so both are swept.
DZ = [-10.0, -7.0, -5.0, -3.5, -2.0, -1.0, 0.0, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0]

NSTEPS = 30                 # -> <= 0.33 rad/step at the largest |dz|: transport cost negligible
HOLD_MS = 0.0
MOVE_EVERY = 2              # subset arm: every other registry site (~534 movers, ~534 reference)
DEPTH_PISTON_CORR = -0.5    # the ridge measured tonight for the per-site 3-D path (jobs 416-432)
WGS3D_Z_MAX = 12.0          # kernel table half-range; max |dz| is 10 rad, keep a small margin
WGS3D_RADIUS_FRAC = 0.7
NUM_PER_GROUP = 780         # 13 x 2 = 26 cells -> 30 shots/cell


def _bootstrap():
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def _move_sets(every=MOVE_EVERY, n=N_SITES):
    """(all-sites, subset) mover index lists.  The subset is every ``every``-th registry index; the
    registry order is column-major over the 33x33 grid, so stride 2 gives an interleaved pattern
    that is obvious by eye in a per-site map and puts a stationary neighbour next to every mover."""
    return list(range(n)), list(range(0, n, int(every)))


def build(dz=None, nsteps=NSTEPS, every=MOVE_EVERY, period_ms=PERIOD_MS, hold_ms=HOLD_MS):
    _bootstrap()
    from scan_group import ScanGroup

    dzs = [float(v) for v in (dz if dz else DZ)]
    all_idx, sub_idx = _move_sets(every)

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_2D
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

    rk = g().rearrange_kwargs
    rk.extras.n_rounds = 1
    rk.protocol = "pingpong"
    rk.nsteps = int(nsteps)
    rk.step_period_ms = float(period_ms)
    rk.extras.step_size = 0.0            # no lateral motion at all: this is a pure axial test
    rk.extras.direction = 0.0
    rk.extras.depth3d = True
    rk.extras.random_z = False
    rk.extras.oneway = True              # rest at the destination -- img2 is taken there
    rk.extras.skip_final_phase = True    # LOAD-BEARING: the WGS bookend would snap the movers back
    rk.extras.ghost_fraction = 1.0       # every site keeps a trap (density + the reference set)
    # dim 1: the commanded offset, applied as dz/nsteps per frame.
    rk.extras.step_size_z.scan(1, [v / float(nsteps) for v in dzs])
    # dim 2: WHICH sites move.  Everything else is byte-identical between the two arms, so the
    # width ratio is alpha and the systematics divide out.
    rk.extras.move_idx.scan(2, [all_idx, sub_idx])
    rk.extras.piston = 0.0
    rk.extras.depth_piston_corr = float(DEPTH_PISTON_CORR)
    rk.extras.lateral_piston_corr = 0.0
    rk.extras.hold_ms = float(hold_ms)
    rk.extras.wgs3d_warm = True          # NOT the model: its 1.105 rad z ladder would quantize dz
    rk.extras.wgs3d_z_max = float(WGS3D_Z_MAX)
    rk.extras.wgs3d_radius_frac = float(WGS3D_RADIUS_FRAC)
    rk.extras.precompute = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()
    return seq_name, g, dzs, len(all_idx), len(sub_idx)


def _desc(dzs, nsteps, n_all, n_sub, period_ms, hold_ms, every):
    return (
        "SELECTIVE AXIAL CONTROL -- some tweezers move, the rest do not, read out of the SAME "
        "frame.  pingpong depth3d on %s at z4 = loading_defocus = %+.1f, ifEnhanced, precompute, "
        "nsteps=%d, step_period_ms=%.3f, hold_ms=%g, step_size=0 (NO lateral motion -- pure axial). "
        "dim 1 = commanded axial offset dz %s rad of PV ANSI Z4, applied as dz/nsteps per frame "
        "(<= %.2f rad/step at the largest |dz|, far under the ~2.25 rad per-frame cliff and under "
        "the 4.8 rad/step where tonight's warm runs still showed zero per-step loss -- so transport "
        "cost is negligible and what remains is pure defocus).  dim 2 = move_idx, the ONLY "
        "difference between the two arms: ALL %d sites move (equivalent to a global axial shift, so "
        "its width IS the axial depth of field, measured on these sites with this detection on "
        "these shots) versus a SUBSET of %d (every %d-th registry index, interleaved so every mover "
        "has a stationary neighbour).  Occupancy vs dz is a smooth peak centred at 0 whose "
        "half-width is the depth of field DIVIDED BY the realization factor alpha = realized/"
        "commanded, so alpha is the RATIO of the two arms' widths and threshold placement, imaging "
        "quality, loading and shot-to-shot brightness are all common-mode and divide out.  The "
        "subset arm carries its own internal control: the NON-movers in the same frame must stay "
        "flat across the whole dz axis -- if they do not, the per-site channel is leaking into "
        "sites it was never commanded to touch, which a global scan cannot see.  WHY THE WARM "
        "PRODUCER (load-bearing): the 3-D model quantizes per-spot z onto a 1 um = 1.105 rad ladder, "
        "which would round the very quantity being measured (a commanded 1.5 rad becomes 1 or 2 "
        "units), and tonight's job 420-vs-433 comparison also puts it behind the warm producer "
        "above ~2 rad/step; wgs3d_warm=True, z_max %.1f, radius_frac %.2f.  TRAP DEPTH: "
        "ghost_fraction=1.0 so EVERY site keeps a trap -- dropping non-movers would concentrate the "
        "same power into fewer traps and change imaging (which is depth-sensitive) and would also "
        "destroy the in-image reference population.  skip_final_phase=True is unavoidable (the WGS "
        "bookend would snap the movers back before img2), so the imaged hologram is the producer's "
        "frame rather than WGS -- identical for both arms and both populations, and img1 normalizes "
        "loading out.  depth_piston_corr=%.2f (the ridge measured tonight, jobs 416-432).  The "
        "headline figure is the per-site map at large |dz|: movers dark, non-movers sharp."
        % (PATTERN, DEFOCUS, nsteps, period_ms, hold_ms, dzs,
           max(abs(v) for v in dzs) / float(nsteps), n_all, n_sub, every,
           WGS3D_Z_MAX, WGS3D_RADIUS_FRAC, DEPTH_PISTON_CORR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--num-per-group", type=int, default=NUM_PER_GROUP)
    ap.add_argument("--dz", default=None, help="comma-separated commanded offsets (rad)")
    ap.add_argument("--nsteps", type=int, default=NSTEPS)
    ap.add_argument("--every", type=int, default=MOVE_EVERY)
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold", type=float, default=HOLD_MS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    dz = [float(v) for v in args.dz.split(",")] if args.dz else None
    seq_name, g, dzs, n_all, n_sub = build(dz=dz, nsteps=args.nsteps, every=args.every,
                                           period_ms=args.period, hold_ms=args.hold)
    g.runp().NumPerGroup = int(args.num_per_group)
    print("seq=%s nseq=%d" % (seq_name, g.nseq()))
    print("  dz (rad) = %s" % dzs)
    print("  nsteps=%d -> <= %.3f rad/step   period=%g   hold=%g"
          % (args.nsteps, max(abs(v) for v in dzs) / args.nsteps, args.period, args.hold))
    print("  arms: move_idx ALL (%d sites)  vs  SUBSET (%d sites, every %d-th)"
          % (n_all, n_sub, args.every))
    if args.dry_run:
        s0 = g.getseq(0)["rearrange_kwargs"]["extras"]
        mi = s0.get("move_idx")
        print("  cell0: step_size_z=%s  move_idx len=%s  %s"
              % (s0.get("step_size_z"), len(mi) if mi is not None else None,
                 {k: s0.get(k) for k in ("depth3d", "oneway", "skip_final_phase",
                                         "ghost_fraction", "wgs3d_warm", "depth_piston_corr")}))
        lens = sorted({len(g.getseq(i)["rearrange_kwargs"]["extras"]["move_idx"])
                       for i in range(g.nseq())})
        zs = sorted({round(g.getseq(i)["rearrange_kwargs"]["extras"]["step_size_z"]
                           * args.nsteps, 4) for i in range(g.nseq())})
        print("  distinct move_idx sizes: %s" % lens)
        print("  distinct total dz: %s" % zs)
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    did = ybStartScan(seq_name, g, url=args.url, label="SelectiveAxialFocusRearrangeScan",
                      description=_desc(dzs, args.nsteps, n_all, n_sub, args.period,
                                        args.hold, args.every), **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
