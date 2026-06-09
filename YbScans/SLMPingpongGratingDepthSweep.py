"""SLMPingpongGratingDepthSweep.py -- pingponggrating unit-sphere sweep (xyz).

Sweeps 50 Fibonacci-sphere directions × 4 radii (1..2) = 200 pts, nsteps=25.
x/y in px, z in rad ANSI Z4 -- survival falloff is ~1-2 in all three, so units
are treated as equivalent. step_size=[sx, sy, sz] xyz mode.

Run:
    cd pyctrl && python YbScans/SLMPingpongGratingDepthSweep.py
"""
import argparse, math as _math, os, sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

NSTEPS = 25
# Fibonacci sphere: 50 evenly-distributed directions on the unit sphere
_PHI = (1.0 + _math.sqrt(5.0)) / 2.0
_DIRS = []
for _i in range(50):
    _theta = _math.acos(1.0 - 2.0 * (_i + 0.5) / 50)
    _phi   = 2.0 * _math.pi * _i / _PHI
    _DIRS.append((_math.sin(_theta) * _math.cos(_phi),
                  _math.sin(_theta) * _math.sin(_phi),
                  _math.cos(_theta)))
# 4 radii × 50 directions = 200 pts; x/y in px, z in rad (units ~equivalent)
_RADII = [1.0 + i / 3.0 for i in range(4)]                        # 1.0, 1.333, 1.667, 2.0
STEP_SIZE_VALUES = [
    [round(r * dx, 5), round(r * dy, 5), round(r * dz, 5)]
    for r in _RADII for dx, dy, dz in _DIRS
]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMPingpongGratingDepthSweep(url=None, reps=None):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (identical to SLMRearrangementScan) ----
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

    # ---- pingponggrating depth: fixed step_size (rad Z4), sweep nsteps ----
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = NSTEPS                     # fixed
    g().rearrange_kwargs.step_period_ms = 1.0
    g().rearrange_kwargs.extras.step_size.scan(1, STEP_SIZE_VALUES)  # 200 pts unit-circle xy, dim 1
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                     # MATCH loading_defocus

    # ---- loading/cooling: SAME as current SLMRearrangementScan.py ----
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
    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMPingpongGratingDepthSweep", **opts)
    print("submitted SLMPingpongGratingDepthSweep -> id %s (nsteps=%d, step_sizes=%s)"
          % (did, NSTEPS, STEP_SIZE_VALUES))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    SLMPingpongGratingDepthSweep(url=args.url, reps=args.reps)
