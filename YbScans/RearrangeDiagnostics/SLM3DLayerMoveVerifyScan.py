"""SLM3DLayerMoveVerifyScan.py -- DOES A COMMANDED AXIAL STEP ACTUALLY HAPPEN?

Everything measured tonight (jobs 412-434) infers axial displacement from atom LOSS.  That is
circular if what we want to know is whether a commanded step is physically realized: a step
that silently under-delivers looks like a step that is simply gentler.  This scan removes the
inference and reads the displacement OPTICALLY, against a ruler baked into the hologram.

THE RULER.  ``phase/2x11x11_5um_z20um.pt`` (rearrangement computer, generated 2026-07-29):
2 x 11x11 sites, 5 um lateral pitch, layers xy-ALIGNED (stacked along the optical axis), axial
gap 20 um = z4 +-12.5313 rad PV at 0.798 um/rad.  20 um is ~9 Rayleigh ranges, so exactly ONE
layer is ever resolvable -- the other is a diffuse halo, not spots.  "In focus" is therefore a
yes/no observable, not a fit.

THE SHOT (3 camera frames; seq ``Rearrange3DFocusWalkCommSeq``, unchanged)
  1. load both layers at LOADING_DEFOCUS      -> img1: BACK layer in focus
  2. slow global axial WALK of the whole hologram, phase-only (``pingponggrating`` depth,
     parabolic ANSI Z4, ``return=False``), %(walk).4f rad in %(nw)d steps
                                              -> img2: FRONT layer in focus, SAME camera pixels
  3. move a SELECTED SUBSET of BACK-layer sites to the FRONT layer depth, per site
     (``pingpong`` + ``move_idx`` + ``step_size_z``, ``oneway=True`` so they STAY there)
                                              -> img3

Stage 3 is the only departure from ``SLMRearrangement3DFocusWalkScan`` (which instead runs a
full 3-D rearrange of both layers into the front layer).  Here the point is not assembly, it is
verification of a PER-SITE step: the same ``move_idx`` machinery job 434 uses, driven by a step
whose correct size is known independently (the layer gap), with the answer visible as spots
appearing in focus rather than as atoms disappearing.

WHY IT IS UNAMBIGUOUS DESPITE THE LAYERS BEING xy-ALIGNED
  A moved back-layer atom lands on the SAME camera pixel as its front-layer partner, so img3
  occupancy alone cannot say which atom is which.  The conditioning does it, per site per shot,
  offline: keep only sites that are IN ``move_idx``, were LOADED in img1 (back occupied) and
  EMPTY in img2 (front unoccupied).  Any atom seen there in img3 can only be the moved one.
  img2 already gives the per-site front occupancy, so no per-shot dynamic ``move_idx`` is
  needed -- the subset is fixed and the selection happens in analysis.

BOTH LAYERS LOAD -- so the other layer has to be EMPTIED, not just ignored
  A moved back-layer atom lands on the same camera pixel as its front-layer partner, and that
  partner is occupied ~half the time.  Rather than condition it away, the transit runs at
  ``ghost_fraction=0``: every non-mover (the whole front layer, plus the unselected back sites)
  is sent to the off-grid sentinel for the duration, so those atoms are dropped and img3
  contains ONLY the moved atoms.  Arrival is then read directly, with no partner ambiguity.

  The scale axis proves the drop actually happened rather than assuming it:
    * scale 0.0 -> movers commanded nowhere, everything else dropped: img3 at the FRONT plane
      must be EMPTY.  Non-empty here means the drop failed (the front layer survived) and the
      whole measurement is void.
    * scale 0.5 -> movers parked halfway (10 um), out of focus at both planes: img3 also empty.
      Separates "the drop killed everything" from "the move does nothing".
    * scale 0.85 / 1.0 / 1.15 -> arrival, peaking where commanded matches the true gap.
  img1 (back occupancy) and img2 (front occupancy) are still recorded per site, so the
  per-site conditioning is available as a cross-check on top of the direct read.

WHAT THE OUTCOME MEANS
  * arrival probability comparable to the hold baseline  -> the commanded 25.06 rad per-site
    step lands the atom within the depth of field of the front layer: the step is real, and
    the rad -> um scale is right to within the depth of field (~1 Rayleigh range, ~2.3 um out
    of 20, i.e. ~10%).
  * arrival near the false-positive floor -> the atom is NOT arriving where commanded, and
    every per-step number from jobs 412-434 is measuring something other than the displacement
    it claims.
  * ``--move-scale`` sweeps the commanded step around the known-correct value.  The arrival
    probability should PEAK at the scale that matches the true layer gap; the peak position is
    a direct, loss-free measurement of realized/commanded axial step, and is the same quantity
    the leg-swap Z4 calibration on the 07/30 Notion page chases at +-3-4% through atom loss.

The per-site step is split over ``--nsteps`` frames (default %(ns)d -> %(perstep).2f rad/step),
inside the regime job 432 measured as free (lambda = 0.000 +- 0.0005 per step out to 4.8
rad/step), so transport loss should not confound the arrival number.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/SLM3DLayerMoveVerifyScan.py --dry-run
    python YbScans/RearrangeDiagnostics/SLM3DLayerMoveVerifyScan.py --force
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYCTRL = os.path.dirname(os.path.dirname(HERE))
BASE_SCAN = os.path.join(PYCTRL, "YbScans", "SLMRearrangement3DFocusWalkScan.py")

# ---- the subset that moves (fixed; conditioning happens in analysis) ------------------
# Layer-major 242-site grid: indices 0..120 = BACK layer, 121..241 = FRONT layer.  Move a
# CHECKERBOARD of the 11x11 back layer so movers and non-movers are interleaved -- any
# hologram-quality effect from the movers then hits both classes equally.
N_PER_LAYER = 121
GRID = 11
PHASE_PATH = "phase/2x11x11_5um_z20um.pt"
Z_HALF_RAD = 12.531328320802004


def _checkerboard_back():
    return [r * GRID + c for r in range(GRID) for c in range(GRID) if (r + c) % 2 == 0]


MOVE_IDX = _checkerboard_back()          # 61 of the 121 back-layer sites

# --move-scale sweep: the commanded step as a fraction of the KNOWN layer gap.  Arrival should
# PEAK where the commanded step matches the true gap, so the peak position IS the realized/
# commanded ratio -- measured optically, with no loss model in the loop.
# 0.0 and 0.5 are CONTROLS, not filler:
#   0.0  -> movers commanded nowhere.  Everything else was dropped, so img3 at the FRONT plane
#           must be empty.  This is the false-positive floor AND the proof that the drop worked.
#   0.5  -> movers parked halfway (10 um), out of focus at BOTH planes -> img3 also empty.
#           Distinguishes "the drop killed everything" from "the move does nothing".
# 0.85 / 1.0 / 1.15 bracket the known gap; arrival should peak at the true ratio.
MOVE_SCALES = [0.0, 0.5, 0.85, 1.0, 1.15]

MOVE_NSTEPS = 40                         # 25.06 rad / 40 = 0.63 rad/step -- inside the free regime
MOVE_PERIOD_MS = 0.696
DEPTH_PISTON_CORR = -0.5                 # the ridge measured tonight (jobs 416/418)
WGS3D_Z_MAX = 14.0                       # max |inter_z| is one layer half-gap (12.53) + margin


def _load_base():
    spec = importlib.util.spec_from_file_location("focuswalk", BASE_SCAN)
    m = importlib.util.module_from_spec(spec)
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(PYCTRL, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec.loader.exec_module(m)
    return m


def build(scales=None, nsteps=None, walk_steps=None, load_defocus=None):
    """Stages 1-2 from the focus-walk scan verbatim; stage 3 replaced by the subset move."""
    m = _load_base()
    n_move = MOVE_NSTEPS if nsteps is None else int(nsteps)
    scales = list(MOVE_SCALES if scales is None else scales)

    # Build with the 3-D stage ENABLED (walk_only would make the seq skip the second handoff
    # entirely -- the seq gates on rearrange_kwargs.extras.walk_only), then overwrite
    # rearrange_kwargs2 field by field.  This keeps NumImages=3 and the bookend re-write
    # wiring exactly as the parent scan validated them.
    seq_name, g = m.build(walk_only=False, load_defocus=load_defocus, walk_steps=walk_steps)

    load_z4 = m.LOADING_DEFOCUS if load_defocus is None else float(load_defocus)
    post_walk_z4 = load_z4 + m.WALK_TOTAL_RAD
    # BACK -> FRONT is the negative of the walk: the walk moved the array so the FRONT layer
    # reached the camera plane, so a BACK atom must travel +(FRONT - BACK) in its own depth.
    move_total = -m.WALK_TOTAL_RAD          # the exact gap (scale 1.0)
    per_step = move_total / float(n_move)

    rk2 = g().rearrange_kwargs2
    rk2.protocol = "pingpong"
    rk2.nsteps = n_move
    rk2.step_period_ms = MOVE_PERIOD_MS
    # Same bookend re-write as the parent scan: the cached WGS final_phase must be rebuilt at
    # the POST-WALK carrier, else it snaps the array back by the whole 20 um in one frame.
    rk2.final_phase = m.PHASE_PATH
    rk2.skip_grid_derive = True
    rk2.extras.loading_zernike = [0, 0, 0, 0, post_walk_z4]
    rk2.extras.z4 = post_walk_z4

    # Neutralize the parent's assignment-protocol extras.  They are inert for pingpong (the
    # dispatcher only reads pattern / prob_hungarian / block_max_size on the assignment path)
    # but the extras dict is server-STICKY, so leave nothing ambiguous in the record.
    rk2.extras.pattern = None
    rk2.extras.prob_hungarian = False
    rk2.extras.move_idx = list(MOVE_IDX)      # exactly these sites move; the rest stay put
    # DROP everything that is not moving.  Both layers load, so with the non-movers kept the
    # front layer stays occupied and a moved atom lands on top of a front atom -- "arrival"
    # would then be unreadable at the same camera pixel.  ghost_fraction=0 sends every
    # non-mover to the off-grid sentinel for the whole transit, which kills the front layer
    # and the unselected back sites, leaving ONLY the moved atoms alive for img3.
    # PASSED EXPLICITLY: the extras dict is merge-only sticky.
    rk2.extras.ghost_fraction = 0.0
    rk2.extras.full_n = False
    rk2.extras.step_size = 0.0                # no lateral motion
    rk2.extras.direction = 0.0                # irrelevant at step_size=0, must be numeric
    # dim 1: the commanded step, swept around the known-correct value.
    rk2.extras.step_size_z.scan(1, [(-m.WALK_TOTAL_RAD * sc) / float(n_move)
                                    for sc in scales])
    # skip_final_phase is LOAD-BEARING, not an optimization: the cached WGS bookend encodes the
    # ORIGINAL two-layer geometry, so writing it after the move would snap every moved atom back
    # to its old layer before img3 and the measurement would read null no matter what.  oneway
    # sets it automatically; set it explicitly so the record does not depend on that.
    rk2.extras.skip_final_phase = True
    rk2.extras.depth3d = True
    rk2.extras.random_z = False               # EXPLICIT: sticky from tonight's random_z runs
    rk2.extras.random_z_max = 0.0
    rk2.extras.oneway = True                  # stay at the new depth for img3
    rk2.extras.hold_ms = 0.0
    rk2.extras.piston = 0.0
    rk2.extras.depth_piston_corr = DEPTH_PISTON_CORR
    rk2.extras.wgs3d_warm = True
    rk2.extras.wgs3d_z_max = WGS3D_Z_MAX
    rk2.extras.wgs3d_radius_frac = 0.7
    rk2.extras.precompute = True
    return seq_name, g, dict(move_total=move_total, per_step=per_step, n_move=n_move,
                             post_walk_z4=post_walk_z4, load_z4=load_z4, base=m,
                             scales=scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--scales", default=None,
                    help="comma-separated commanded-move scales (default %s)"
                         % ",".join("%g" % v for v in MOVE_SCALES))
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--walk-steps", type=int, default=None)
    ap.add_argument("--load-defocus", type=float, default=None)
    ap.add_argument("--num-per-group", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    scales = ([float(x) for x in args.scales.split(",") if x.strip()]
              if args.scales else None)
    seq_name, g, info = build(scales=scales, nsteps=args.nsteps,
                              walk_steps=args.walk_steps, load_defocus=args.load_defocus)
    if args.num_per_group:
        g.runp().NumPerGroup = int(args.num_per_group)
    print("seq=%s nseq=%d" % (seq_name, g.nseq()))
    print("  load carrier z4      = %+.4f rad" % info["load_z4"])
    print("  post-walk carrier    = %+.4f rad" % info["post_walk_z4"])
    print("  commanded subset move= %+.4f rad over %d steps = %+.3f rad/step at scale 1.0"
          % (info["move_total"], info["n_move"], info["per_step"]))
    print("  scales swept         = %s  -> %s rad total"
          % (info["scales"], ["%+.2f" % (info["move_total"] * sc) for sc in info["scales"]]))
    print("  movers               = %d of %d back-layer sites (checkerboard)"
          % (len(MOVE_IDX), N_PER_LAYER))
    if args.dry_run:
        e = g.getseq(0)["rearrange_kwargs2"]["extras"]
        print("  rk2: protocol=%s nsteps=%s gf=%s step_size_z=%.4f depth3d=%s oneway=%s "
              "random_z=%s corr=%s warm=%s n_move_idx=%d"
              % (g.getseq(0)["rearrange_kwargs2"].get("protocol"),
                 g.getseq(0)["rearrange_kwargs2"].get("nsteps"),
                 e.get("ghost_fraction"), e.get("step_size_z"), e.get("depth3d"),
                 e.get("oneway"), e.get("random_z"), e.get("depth_piston_corr"),
                 e.get("wgs3d_warm"), len(e.get("move_idx") or [])))
        return
    if not args.force:
        raise SystemExit("refusing to submit without --force (use --dry-run first)")
    from yb_start_scan import ybStartScan
    opts = {"rep": args.reps} if args.reps is not None else {}
    desc = (
        "AXIAL STEP VERIFICATION against a hologram ruler.  Array %s: 2 x 11x11, 5 um pitch, "
        "layers xy-ALIGNED, axial gap 20 um = z4 +-%.4f rad PV (0.798 um/rad).  Per shot, 3 "
        "frames: (1) img1 with the BACK layer at the camera plane (carrier %+.4f); (2) a slow "
        "global phase-only axial WALK of the whole hologram, %+.4f rad, bringing the FRONT "
        "layer onto the SAME camera pixels -> img2; (3) a SELECTED SUBSET of back-layer sites "
        "(checkerboard, %d of %d) moved per-site to the front-layer depth via pingpong + "
        "move_idx + step_size_z = %+.4f rad over %d steps (%+.3f rad/step, oneway so they stay), "
        "ghost_fraction=1 so the non-movers are held as controls -> img3.  Warm 3-D WGS, "
        "depth_piston_corr=%g (the ridge measured 2026-07-31), precompute.  PURPOSE: every "
        "axial number from jobs 412-434 infers displacement from atom LOSS, which cannot "
        "distinguish a gentle step from an under-delivered one.  Here the correct step size is "
        "known independently (the layer gap) and arrival is read OPTICALLY as spots coming into "
        "focus.  Conditioning per site per shot: movers that were LOADED in img1 and EMPTY in "
        "img2 -- an atom there in img3 can only be the moved one.  Controls in the same shots: "
        "movers empty in both (false-positive floor), non-movers loaded-back/empty-front (must "
        "stay empty), non-movers loaded-front (the hold-survival baseline).  DIM 1 sweeps the "
        "commanded step over scales %s of the known gap: arrival should PEAK where commanded "
        "matches true, so the peak position measures realized/commanded axial step optically, "
        "with no loss model in the loop.  skip_final_phase=True is load-bearing -- the cached "
        "WGS bookend encodes the original two-layer geometry and would snap the moved atoms "
        "back before img3."
        % (PHASE_PATH, Z_HALF_RAD, info["load_z4"], info["base"].WALK_TOTAL_RAD,
           len(MOVE_IDX), N_PER_LAYER, info["move_total"], info["n_move"], info["per_step"],
           DEPTH_PISTON_CORR, info["scales"]))
    did = ybStartScan(seq_name, g, url=args.url, label="SLM3DLayerMoveVerifyRearrangeScan",
                      description=desc, **opts)
    print("submitted -> descriptor id %s" % did)


if __name__ == "__main__":
    main()
