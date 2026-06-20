"""SLMPingpongGratingZernikeDistortionSweep.py -- x-direction pingponggrating with a
purposeful Zernike-distortion sweep, ONE scan per ANSI Zernike index 4..13.

For each ANSI/OSA single-index Zernike j in 4..13 (defocus through vertical secondary
astigmatism) submit ONE scan that 2-D sweeps:
  * dim 1 -- distortion amplitude: [-2, -1, 0, 1, 2] rad. Swept as the SCALAR extra
             ``distortion_z<j>`` (NOT a list-valued ``distortion_zernike`` axis -- a
             list-valued swept axis breaks the lab-side live scan curve, since
             scan_analysis._find_first_numeric ravels an (npts, coeff_len) list-of-lists,
             giving the "size 25 vs 5" concat error). RearrangeCommSeq folds
             ``distortion_z<j>`` -> ``distortion_zernike`` per shot
             (rearrange_runtime.translate_distortion_zernike), so the SLM server still gets the
             list. The dispatcher adds it to every TRANSIT frame (NOT the disp=0 WGS endpoints);
             ``reverse_zernike=True`` flips its sign on the return leg. amp=0 = un-distorted.
  * dim 2 -- step_size: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] knm-px (scalar -> lateral x mode).
=> 5 x 6 = 30 points/scan; 10 reps -> 300 shots/scan; indices 4..13 => 10 scans queued.

pingponggrating is phase-only (no model frames): each frame = WGS_initial + step*disp*x-blaze,
plus the ANSI-j aberration on transit frames (sign flipped on the return leg). precompute=False,
nsteps=50, period=1 ms.

Loading / model / phase config mirrors SLMRearrangementScan.py for the 47x47_feedbackwarm4 array.

Run (queues ALL 10 indices 4..13):
    cd pyctrl && python YbScans/SLMPingpongGratingZernikeDistortionSweep.py
One index only / explicit backend:
    python YbScans/SLMPingpongGratingZernikeDistortionSweep.py --index 7
    python YbScans/SLMPingpongGratingZernikeDistortionSweep.py --url tcp://127.0.0.1:1408
Build-only (no submit; prints point counts + sample swept axis):
    python YbScans/SLMPingpongGratingZernikeDistortionSweep.py --dry-run

Prereq: the pyctrl backend running at --url + the SLM server reachable. NOTE: requires (a) the
SLM-server pingponggrating distortion patch (rearrange_actual.py) and (b) the pyctrl
translate_distortion_zernike fold (rearrange_runtime.py + RearrangeCommSeq.py) -- a backend
restart is needed to pick up (b).
"""
import argparse
import os
import sys

# 47x47_feedbackwarm4 config (mirrors SLMRearrangementScan.py).
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"
PATTERN = "47x47_feedbackwarm4"
PHASE_PATH = "phase/47x47_feedbackwarm4.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

NSTEPS = 50
STEP_PERIOD_MS = 1.0
REPS = 10

# ANSI/OSA single-index Zernike modes to sweep, one scan each (4..13 inclusive).
#   4=defocus  5=vert-astig  6=vert-trefoil  7=vert-coma  8=horiz-coma  9=obliq-trefoil
#   10=obliq-quadrafoil  11=secondary-astig  12=primary-spherical  13=vert-secondary-astig
# NOTE index 4 (defocus, "z4") is included per the explicit 4..13 range, even though depth-mode
# pingponggrating can also scan defocus -- here it is applied as a constant distortion instead.
ZERNIKE_INDICES = list(range(4, 14))                  # [4, 5, ..., 13] -> 10 indices

# dim-1 distortion-amplitude axis: [-2, -1, 0, 1, 2] rad (5 pts). amp=0 -> all-zero coeffs ->
# no aberration that point (the un-distorted pingponggrating baseline).
AMP_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
assert len(AMP_VALUES) == 5

# dim-2 step-size axis: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] px (6 pts), lateral x (knm-1024 px).
STEP_VALUES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
assert len(STEP_VALUES) == 6 and STEP_VALUES[0] == 1.0 and STEP_VALUES[-1] == 2.0

N_PTS = len(AMP_VALUES) * len(STEP_VALUES)            # 30


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_group(index):
    """Build the 30-point (distortion x step) ScanGroup for one ANSI Zernike ``index``."""
    from scan_group import ScanGroup

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (mirrors SLMRearrangementScan for 47x47_feedbackwarm4) ----
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

    # ---- pingponggrating: x-direction lateral, 2-D (distortion x step) sweep ----
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = NSTEPS
    g().rearrange_kwargs.step_period_ms = STEP_PERIOD_MS
    # dim 1: SCALAR distortion amplitude at ANSI index j (folded to distortion_zernike per shot).
    dz_name = "distortion_z%d" % index
    getattr(g().rearrange_kwargs.extras, dz_name).scan(1, list(AMP_VALUES))
    # dim 2: scalar step_size (lateral x).
    g().rearrange_kwargs.extras.step_size.scan(2, list(STEP_VALUES))
    g().rearrange_kwargs.extras.reverse_zernike = True
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                # match loading_defocus (focal plane)
    g().rearrange_kwargs.extras.initial_pattern = PATTERN
    g().rearrange_kwargs.extras.final_pattern = PATTERN

    # ---- run params (mirror SLMRearrangementScan; cooling left at config defaults) ----
    rp.NumPerGroup = N_PTS * REPS                      # 300 -> StackNum=10 (30 pts x 10 reps)
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    return g


def SLMPingpongGratingZernikeDistortionSweep(url=None, indices=None, dry_run=False):
    """Build + submit one scan per ANSI Zernike index. Returns descriptor ids (or ScanGroups
    when ``dry_run``)."""
    _bootstrap()

    indices = list(ZERNIKE_INDICES if indices is None else indices)
    out = []

    if dry_run:
        print("DRY RUN -- no submission.")
        print("  amplitude axis (dim1, scalar distortion_z<j>, %d): %s" % (len(AMP_VALUES), AMP_VALUES))
        print("  step_size axis (dim2, %d): %s" % (len(STEP_VALUES), STEP_VALUES))
        for j in indices:
            g = _build_group(j)
            nseq = g.nseq()
            print("  z%d: nseq=%d (expect %d); reps=%d -> %d shots; swept extra=distortion_z%d"
                  % (j, nseq, N_PTS, REPS, nseq * REPS, j))
            out.append(g)
        print("would queue %d scans (indices %s); ~%d shots total"
              % (len(indices), indices, N_PTS * REPS * len(indices)))
        return out

    from yb_start_scan import ybStartScan
    for j in indices:
        g = _build_group(j)
        desc = ("x-direction pingponggrating purposeful-distortion sweep: ANSI Zernike index %d; "
                "dim1 distortion amplitude %s rad (%d pts, scalar distortion_z%d -> distortion_zernike) "
                "x dim2 step_size %s knm-px (%d pts) = %d points; reverse_zernike=True, "
                "precompute=False, nsteps=%d, period=%g ms, %d reps (-> %d shots). %s."
                % (j, AMP_VALUES, len(AMP_VALUES), j, STEP_VALUES, len(STEP_VALUES), N_PTS,
                   NSTEPS, STEP_PERIOD_MS, REPS, N_PTS * REPS, PATTERN))
        did = ybStartScan("RearrangeCommSeq", g, url=url,
                          label="PPGratingZernDistort_z%d" % j,
                          description=desc, rep=REPS)
        print("submitted z%d -> descriptor id %s (%d pts, %d reps = %d shots)"
              % (j, did, N_PTS, REPS, N_PTS * REPS))
        out.append(did)
    print("queued %d scans (indices %s); ~%d shots total"
          % (len(out), indices, N_PTS * REPS * len(out)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Queue x-direction pingponggrating Zernike-distortion sweeps (ANSI 4..13).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--index", type=int, default=None,
                    help="run ONLY this ANSI Zernike index (default: all 4..13)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the ScanGroups and print point counts; do NOT submit")
    args = ap.parse_args()
    idxs = [args.index] if args.index is not None else None
    SLMPingpongGratingZernikeDistortionSweep(url=args.url, indices=idxs, dry_run=args.dry_run)
