"""SLMPingpongGratingComaDirSweep.py -- pingponggrating coma-distortion x step-size sweep,
queued for TWO coma modes x TWO perpendicular motion directions = 4 scans.

Each scan is a 2-D sweep on the 33x33_feedback9 uniform production array:
  * dim 1 -- coma coefficient N in {-3, -2, -1, 0, 1, 2, 3} rad (7 pts). Swept as the SCALAR
             extra ``distortion_z<idx>`` (idx = 7 vertical coma, 8 horizontal coma), folded per
             shot to the list-valued ``distortion_zernike`` the SLM-server pingponggrating
             dispatcher reads (RearrangeCommSeq._fold_distortion_zernike). The coma is added to
             every TRANSIT frame (NOT the disp=0 WGS endpoints); ``reverse_zernike=True`` flips its
             sign on the return leg. N=0 = un-distorted. A SCALAR axis (not a list-valued
             ``distortion_zernike`` axis) keeps the lab-side live scan curve working.
  * dim 2 -- step_size in {0, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8} (12 pts).
             step_size=0 = no motion (pure-distortion control: coma still applied on transit
             frames, but the grating ramp is flat -> traps don't move).
  => 7 x 12 = 84 points/scan; 2 shots/point (rep=2) -> 168 shots/scan; 4 scans queued.
     (2 shots/point is enough: the fine step_size sweep is fit per coma coefficient, so each fit
     uses all 12 step points -- per-point reps just need to anchor the curve, not stand alone.)

Two perpendicular MOTION DIRECTIONS (the grating blaze axis the pingpong oscillates along).
BOTH sweep a SCALAR dim-2 axis -- a list-valued step_size axis ([0,s,0] per point) breaks the
lab-side 2-D scan grid ("concatenation axis doesn't match along axis 0") and errors the scan at
dequeue before any shot runs:
  * "left" (x) -- horizontal blaze; sweeps SCALAR extras.step_size (pingponggrating's default
                  lateral mode, knm-px).
  * "up"   (y) -- vertical blaze; sweeps SCALAR extras.step_y, folded per shot to
                  step_size=[0,s,0] (xyz mode) by RearrangeCommSeq._fold_step_xyz -- the only way
                  to move along y. Scalar axis -> clean N-D grid + live curve, same as "left".

pingponggrating is phase-only (no model frames): each transit frame = WGS_initial + step*disp*blaze,
plus the coma on transit frames (sign flipped on the return leg). 50 steps each way (nsteps=50,
return=True -> triangle 0..50..0 = 101 frames), period 0.696 ms, precompute + precompute_host on.
NOTE: with a distortion active the uint8 precompute cache is bypassed (the runtime builds each
frame on CPU), so the realised per-frame pace may exceed 0.696 ms -- check the per-shot diag
(rearrange.diag.write_lat_ms / compute_ms) for the actual pace, don't assume 0.696 was hit.

Loading / model / phase config = current 33x33_feedback9 settings: the ByPattern[33x33_feedback9]
cooling/imaging/servo overlay auto-applies (initial_pattern/final_pattern); MOT/loading left at
expConfig defaults; loading_defocus = z4 = -5 (matched focal plane).

Run (queues ALL 4 scans):
    cd pyctrl && python YbScans/RearrangeDiagnostics/SLMPingpongGratingComaDirSweep.py
Build-only (no submit; prints point counts + sample swept axes):
    python YbScans/RearrangeDiagnostics/SLMPingpongGratingComaDirSweep.py --dry-run
Subset (e.g. only the x-direction vertical-coma scan):
    python YbScans/RearrangeDiagnostics/SLMPingpongGratingComaDirSweep.py --only x:7

Prereq: the pyctrl backend running at --url + the SLM server reachable. Requires the SLM-server
pingponggrating distortion patch + the pyctrl distortion fold (RearrangeCommSeq.py) -- a backend
restart is needed if those folds are not already loaded.
"""
import argparse
import os
import sys

# ---- 33x33_feedback9 config (current uniform production 33x33 array) ----
PATTERN = "33x33_feedback9"
PHASE_PATH = "phase/33x33_feedback9.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]                 # flattened-uniform array, no baked Zernike
MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"   # 33x33 model

NSTEPS = 50                                     # 50 steps each way (return=True -> triangle)
STEP_PERIOD_MS = 0.696
REPS = 2                                        # 2 shots/point -> 84*2 = 168 shots/scan

# dim-1 coma coefficient axis N (rad), 7 pts. N=0 -> no distortion (baseline).
COMA_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
assert len(COMA_VALUES) == 7

# dim-2 step-size axis, 12 pts. step_size=0 -> no motion (pure-distortion control).
STEP_VALUES = [0.0, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
assert len(STEP_VALUES) == 12

N_PTS = len(COMA_VALUES) * len(STEP_VALUES)     # 84

# ANSI/OSA coma indices: 7 = vertical coma, 8 = horizontal coma.
COMA_MODES = [(7, "vertical-coma"), (8, "horizontal-coma")]
# Motion directions (perpendicular): the grating blaze axis the pingpong oscillates along.
#   "left" -> x (horizontal blaze), scalar step_size (default lateral mode).
#   "up"   -> y (vertical blaze),   step_size = [0, s, 0] (xyz mode; the only y path).
DIRECTIONS = ["left", "up"]

# All 4 (direction, coma-index) scans.
SCANS = [(d, idx, cname) for d in DIRECTIONS for (idx, cname) in COMA_MODES]


def _set_step_axis(g, direction):
    """Set the dim-2 step sweep (12 pts) for a motion direction, ALWAYS as a
    SCALAR swept axis so the lab-side N-D scan grid + live curve build cleanly:
      left -> extras.step_size (horizontal blaze, lateral x mode), scalar px.
      up   -> extras.step_y, folded per shot to step_size=[0,s,0] (vertical
              blaze, xyz y mode) by RearrangeCommSeq._fold_step_xyz.
    A list-valued step_size axis (e.g. [0,s,0] per point) breaks the 2-D grid
    ("concatenation axis doesn't match along axis 0") and errors at dequeue, so
    the up direction sweeps the scalar step_y proxy instead."""
    vals = [float(s) for s in STEP_VALUES]
    if direction == "left":
        g().rearrange_kwargs.extras.step_size.scan(2, vals)
    elif direction == "up":
        g().rearrange_kwargs.extras.step_y.scan(2, vals)
    else:
        raise ValueError("unknown direction %r" % direction)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_group(direction, coma_idx):
    """Build the 84-point (coma x step) ScanGroup for one (direction, coma index)."""
    from scan_group import ScanGroup

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (33x33_feedback9; forwarded once at dequeue with reset_params) ----
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.extras.final_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- pingponggrating: 2-D (coma x step) sweep, one motion direction ----
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = NSTEPS
    g().rearrange_kwargs.step_period_ms = STEP_PERIOD_MS
    # dim 1: SCALAR coma amplitude at ANSI index (folded to distortion_zernike per shot).
    dz_name = "distortion_z%d" % coma_idx
    getattr(g().rearrange_kwargs.extras, dz_name).scan(1, list(COMA_VALUES))
    # dim 2: step sweep (always a SCALAR axis; "up" sweeps step_y -> folded to
    # step_size=[0,s,0] per shot by RearrangeCommSeq._fold_step_xyz).
    _set_step_axis(g, direction)
    g().rearrange_kwargs.extras.reverse_zernike = True   # +coma out, -coma back (motion-relative)
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                  # match loading_defocus (focal plane)
    g().rearrange_kwargs.extras.initial_pattern = PATTERN
    g().rearrange_kwargs.extras.final_pattern = PATTERN

    # ---- run params (current settings; cooling/imaging from ByPattern[33x33_feedback9]) ----
    rp.NumPerGroup = N_PTS * REPS                        # 420
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    return g


def SLMPingpongGratingComaDirSweep(url=None, only=None, dry_run=False):
    """Build + submit one scan per (direction, coma index). Returns descriptor ids (or
    ScanGroups when ``dry_run``)."""
    _bootstrap()

    scans = list(SCANS)
    if only:
        wanted = set(only)
        scans = [(d, idx, cn) for (d, idx, cn) in scans
                 if d in wanted or ("%s:%d" % (d, idx)) in wanted or str(idx) in wanted]
        if not scans:
            raise SystemExit("--only %r matched no scans (have: %s)"
                             % (only, [("%s:%d" % (d, i)) for d, i, _ in SCANS]))

    if dry_run:
        print("DRY RUN -- no submission.")
        print("  coma axis (dim1, scalar distortion_z<idx>, %d): %s" % (len(COMA_VALUES), COMA_VALUES))
        print("  step_size axis (dim2, %d): %s" % (len(STEP_VALUES), STEP_VALUES))
        out = []
        for d, idx, cname in scans:
            g = _build_group(d, idx)
            nseq = g.nseq()
            print("  %-4s %s (z%d): nseq=%d (expect %d); rep=%d -> %d shots; step=%s"
                  % (d, cname, idx, nseq, N_PTS, REPS, nseq * REPS,
                     "step_size(x)" if d == "left" else "step_y->[0,s,0]"))
            out.append(g)
        print("would queue %d scans; ~%d shots total" % (len(scans), N_PTS * REPS * len(scans)))
        return out

    from yb_start_scan import ybStartScan
    out = []
    for d, idx, cname in scans:
        g = _build_group(d, idx)
        step_desc = ("scalar step_size lateral-x" if d == "left"
                     else "scalar step_y -> [0,s,0] xyz-y")
        desc = ("pingponggrating coma-distortion sweep on %s: motion=%s (%s), ANSI z%d (%s); "
                "dim1 coma N %s rad (%d pts, scalar distortion_z%d -> distortion_zernike, "
                "reverse_zernike=True) x dim2 step_size %s (%d pts, %s) = %d points; "
                "nsteps=%d (50/way), period=%g ms, precompute+host, rep=%d (-> %d shots)."
                % (PATTERN, d, step_desc, idx, cname, COMA_VALUES, len(COMA_VALUES), idx,
                   STEP_VALUES, len(STEP_VALUES), step_desc, N_PTS, NSTEPS, STEP_PERIOD_MS,
                   REPS, N_PTS * REPS))
        label = "PPGratingComa_%s_z%d" % (d, idx)
        did = ybStartScan("RearrangeCommSeq", g, url=url, label=label, description=desc, rep=REPS)
        print("submitted %-4s z%d (%s) -> descriptor id %s (%d pts, rep=%d = %d shots)"
              % (d, idx, cname, did, N_PTS, REPS, N_PTS * REPS))
        out.append(did)
    print("queued %d scans; ~%d shots total" % (len(out), N_PTS * REPS * len(out)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Queue pingponggrating coma x step-size sweeps (2 coma modes x 2 directions).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only matching scans: a direction ('left'/'up'), an index ('7'/'8'), "
                         "or 'dir:idx' (e.g. 'up:8'). Default: all 4.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the ScanGroups and print point counts; do NOT submit")
    args = ap.parse_args()
    SLMPingpongGratingComaDirSweep(url=args.url, only=args.only, dry_run=args.dry_run)
