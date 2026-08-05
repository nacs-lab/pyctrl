"""WarmWGSRearrangeStepSweep.py -- 33x33 rearrange2 with WARM-STARTED PHASE-LOCKED WGS transit
frames (server extras ``wgs_warm``, deployed 2026-07-27), sweeping nsteps 10..120.

Survival vs step size: with linear scheduling (``dynamic=False``) every atom moves at
its own ``L_i / nsteps`` knm-px per frame, so the nsteps axis IS the per-frame
step-size axis (model-frame baseline from rearrange2-streaming.md: 20 -> ~0.53,
40 -> ~0.91, 60 -> ~0.98, 80 -> ~0.99, 100-120 -> ~0.98). This run swaps the SLMnet
model for the warm-started WGS producer (``rearrange2/warm_wgs.WarmWGSProducer``):
frame k is solved with ``wgs_iters`` phase-locked WGS iterations at ``wgs_pad`` FFT
pad, seeded from frame k-1's SLM phase -- exact spot positions, exact per-spot phase
contract, faithful per-spot amplitude control. WGS_PAD/WGS_ITERS set the solve
(server defaults 2048 x 3, ~0.55-0.9 ms/frame; 1024 = no padding, ~-7% pattern power
but ~2.5x produce headroom, ~0.25 ms/frame); everything else identical to the
production SLMRearrangementScan defaults.

Single-round (RearrangeCommSeq, 2 images): 33x33_feedback11 -> EVERY_OTHER
checkerboard subset of 33x33_feedback11.

NOTE the scan name MUST contain 'rearrang': the Analysis tab's on-demand slm_diag
sync (run_analysis._maybe_sync_slm_diag) is name-gated, and without the diag the
run is not recognized as rearrangement (no per-shot targets / target-aware
survival). The gate now also sniffs rearrange_kwargs in the JSON sidecar, but keep
the name convention anyway.

Run:
    cd pyctrl && python YbScans/RearrangeDiagnostics/WarmWGSRearrangeStepSweep.py
"""
import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# amplitude-control variant: required for mover_boost with MODEL frames (the plain
# direct_flat model is amplitude-blind; warm-WGS realizes amps natively either way)
MODEL_AMPCTRL = "SLMnet/checkpoints/sinc_3x3_experiment/models/ampctrl_flat/ampctrl_flat_best.pth"

NSTEPS = list(range(10, 121, 10))          # 10, 20, ... 120 (12 points, dim 1)
# DEFAULT REP COUNT -- do not remove. With no rep passed, ybStartScan falls through to
# rp.NumPerGroup (100000), which the runner treats as the actual target rather than the upper
# bound the name suggests: run 627 was submitted without --reps and ran 5416 shots in ~7 h on its
# way to ~69 h before being aborted. 60 reps x 12 nsteps ~ 720 shots (~30 min at 2.5 s/shot) and
# matches what the earlier WarmWGS runs in this family actually took (seq_num 480-504).
DEF_REPS = 60
WGS_PAD = 2048                             # warm-WGS FFT pad (full teacher efficiency)
WGS_ITERS = 3                              # WGS iterations/frame (teacher-grade contract)
DEFOCUS = -4                               # matched loading_defocus / rearrange z4

def _run_desc(wgs_warm, mover_boost=0.0, ampctrl=False):
    if wgs_warm:
        frames = ("warm-start WGS transit frames (wgs_warm, pad %d x %d iters, gaussian beam)"
                  % (WGS_PAD, WGS_ITERS))
    else:
        frames = ("SLMnet %s MODEL transit frames (wgs_warm off)"
                  % ("ampctrl_flat" if ampctrl else "direct_flat"))
    if mover_boost:
        frames += ", mover_boost=%g" % mover_boost
    return (frames + ", 33x33_feedback11 single-round rearrange2 to EVERY_OTHER "
            "checkerboard target, linear scheduling; nsteps sweep 10-120 for survival "
            "vs per-frame step size.")


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build(wgs_warm=True, mover_boost=0.0, ampctrl=False, precompute=False):
    """Build (do NOT submit) the ScanGroup -- offline-exercisable, scan-verification style.
    ``wgs_warm=False`` = A/B control: identical scan but transit frames from the SLMnet
    model (direct_flat, or ampctrl_flat when ``ampctrl``) instead of warm-started WGS.
    ``mover_boost`` = per-spot amplitude boost on the movers (server extras)."""
    _bootstrap()
    from scan_group import ScanGroup

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (identical to SLMRearrangementScan) ----
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_AMPCTRL if ampctrl else MODEL_FILENAME
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

    # ---- rearrange2 + warm-start WGS producer, nsteps sweep ----
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.nsteps.scan(1, list(NSTEPS))      # dim 1: the step-size axis
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.extras.wgs_warm = bool(wgs_warm)  # warm-started phase-locked WGS frames
    g().rearrange_kwargs.extras.wgs_pad = WGS_PAD          # inert when wgs_warm=False
    g().rearrange_kwargs.extras.wgs_iters = WGS_ITERS
    g().rearrange_kwargs.extras.pattern = "every_other"   # checkerboard target ((row+col) parity 0)
    g().rearrange_kwargs.extras.mover_boost = float(mover_boost)   # 0.0 = off (server default)
    g().rearrange_kwargs.extras.prob_hungarian = True
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False            # linear = validated operating point
    # Server extras are STICKY: an unset precompute inherits whatever the previous scan left,
    # so both flags are written explicitly on every build. Default False -- warm-WGS solves each
    # transit frame live from the previous frame's phase, so a precomputed uint8 cache cannot be
    # reused across the nsteps axis anyway, and precompute contention has stalled runs before.
    g().rearrange_kwargs.extras.precompute = bool(precompute)
    g().rearrange_kwargs.extras.precompute_host = bool(precompute)
    g().rearrange_kwargs.extras.ifEnhanced = True
    g().rearrange_kwargs.extras.z4 = DEFOCUS               # MATCH loading_defocus
    g().rearrange_kwargs.extras.initial_pattern = PATTERN
    g().rearrange_kwargs.extras.final_pattern = PATTERN
    g().rearrange_kwargs.extras.description = _run_desc(wgs_warm, mover_boost, ampctrl)

    # ---- run params (runp; loading/cooling stay at expConfig defaults) ----
    rp.NumPerGroup = 100000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = json.dumps([
        {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
        {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
    ])
    return "RearrangeCommSeq", g


def WarmWGSRearrangeStepSweep(url=None, reps=None, wgs_warm=True, pad=None, iters=None,
                              boost=0.0, ampctrl=False, nsteps=None, precompute=False,
                              label=None):
    global WGS_PAD, WGS_ITERS, NSTEPS
    if pad:
        WGS_PAD = int(pad)
    if iters:
        WGS_ITERS = int(iters)
    if nsteps:
        NSTEPS = list(nsteps)
    seq_name, g = build(wgs_warm=wgs_warm, mover_boost=boost, ampctrl=ampctrl,
                        precompute=precompute)
    from yb_start_scan import ybStartScan
    # Label must keep 'rearrang' (the Analysis-tab slm_diag name gate).
    if wgs_warm:
        base = "WarmWGS%dx%d" % (WGS_PAD, WGS_ITERS)
    else:
        base = "Ampctrl" if ampctrl else "ModelFrames"
    label = label or (base + ("Boost%g" % boost if boost else "") + "RearrangeStepSweep")
    opts = {"rep": DEF_REPS if reps is None else int(reps)}
    did = ybStartScan(seq_name, g, url=url, label=label, **opts)
    print("submitted %s -> id %s (nsteps=%s, %s%s)"
          % (label, did, NSTEPS,
             "wgs %dx%d" % (WGS_PAD, WGS_ITERS) if wgs_warm
             else ("ampctrl model" if ampctrl else "direct_flat model"),
             ", boost %g" % boost if boost else ""))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--model", action="store_true",
                    help="A/B control: SLMnet model frames (wgs_warm off)")
    ap.add_argument("--ampctrl", action="store_true",
                    help="with --model: use the ampctrl_flat checkpoint (amp control)")
    ap.add_argument("--pad", type=int, default=None, help="override WGS_PAD")
    ap.add_argument("--iters", type=int, default=None, help="override WGS_ITERS")
    ap.add_argument("--boost", type=float, default=0.0,
                    help="mover_boost (per-spot mover amplitude boost; 0 = off)")
    ap.add_argument("--nsteps", default=None,
                    help="comma-separated nsteps list override, e.g. 5,10,15,20,25,30,35,40")
    ap.add_argument("--precompute", action="store_true",
                    help="turn the uint8 precompute cache ON (default OFF; both extras are "
                         "always written explicitly because server extras are sticky)")
    ap.add_argument("--label", default=None,
                    help="override the submit label (must still contain 'rearrang')")
    args = ap.parse_args()
    ns = [int(x) for x in args.nsteps.split(",")] if args.nsteps else None
    WarmWGSRearrangeStepSweep(url=args.url, reps=args.reps,
                              wgs_warm=not (args.model or args.ampctrl),
                              pad=args.pad, iters=args.iters,
                              boost=args.boost, ampctrl=args.ampctrl, nsteps=ns,
                              precompute=args.precompute, label=args.label)
