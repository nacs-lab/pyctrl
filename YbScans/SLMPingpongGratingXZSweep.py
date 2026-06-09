"""SLMPingpongGratingXZSweep.py -- pingponggrating XZ Cartesian-product sweep.

100 scan points: Cartesian product of 10 x-values × 10 z-values.
step_size = [sx, 0, sz] for all (sx, sz) pairs.

Values are concentrated around |val| in [1.25, 1.75] (where the 50%
survival threshold falls per the sphere scan), with two outer context
points at ±2.0 for reference.

step_period=0.75 ms, precompute=True, precompute_host=True.
"""
import argparse, os, sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

NSTEPS = 25

# 15 values: focus on |val| in [1.25, 1.75] + expanded inner [-1, 1] coverage
#   2 outer context  : -2.0, 2.0
#   3 dense neg      : -1.75, -1.5, -1.25
#   5 inner          : -1.0, -0.5, 0.0, 0.5, 1.0
#   5 dense pos      : 1.25, 1.375, 1.5, 1.625, 1.75
_AXIS_VALS = [
    -2.0,
    -1.75, -1.5, -1.25,
    -1.0, -0.5, 0.0, 0.5, 1.0,
    1.25, 1.375, 1.5, 1.625, 1.75,
    2.0,
]
assert len(_AXIS_VALS) == 15

# Cartesian product: all (sx, sz) pairs — 225 points total
STEP_SIZE_VALUES = [
    [round(sx, 4), 0.0, round(sz, 4)]
    for sx in _AXIS_VALS
    for sz in _AXIS_VALS
]
assert len(STEP_SIZE_VALUES) == 225


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMPingpongGratingXZSweep(url=None, reps=None):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (same as SLMRearrangementScan) ----
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

    # ---- pingponggrating: sweep step_size over 40 (x, z) axis points ----
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = NSTEPS
    g().rearrange_kwargs.step_period_ms = 0.75
    g().rearrange_kwargs.extras.step_size.scan(1, STEP_SIZE_VALUES)
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                # match loading_defocus

    # ---- loading/cooling ----
    g().BlueMOT.LoadingTime = 0.23
    g().BlueMOT.FreqDetuning = -44e6
    g().GreenMOT.BiasCoilCurrent.Y = 0.268
    g().GreenMOT.BiasCoilCurrent.X = 0.040
    g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    g().GreenMOT.CoolDown.Amp = 0.25
    g().GreenMOT.CoolDown.HoldTime = 0.12

    # ---- run params ----
    rp.NumPerGroup = 100000
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMPingpongGratingXZSweep", **opts)
    print("submitted SLMPingpongGratingXZSweep -> id %s (%d pts xz-grid, nsteps=%d)"
          % (did, len(STEP_SIZE_VALUES), NSTEPS))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    SLMPingpongGratingXZSweep(url=args.url, reps=args.reps)
