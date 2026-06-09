"""TwoLayerLoadScan.py -- two-layer SLM loading + depth transport test.

Loads atoms into a SUPERPOSITION of two axially-separated tweezer arrays, then
measures survival under a pingponggrating depth trip.

The loading pattern is computed at submission time by combining two WGS phases:
  33_-5  : 33x33_uniform with ANSI Z4 = -5 rad (current imaging plane)
  33_+10 : 33x33_uniform with ANSI Z4 = -5 + depth_diff rad (second layer)
The complex-field superposition is angle(exp(i*phi_1) + exp(i*phi_2)) -- both
amplitude and 50/50 power split are accepted; we wipe the amplitude and keep
only the phase angle, giving a dual-focal-plane SLM pattern.

After loading, a single pingponggrating depth sweep moves ALL tweezers by
+depth_diff radians of Z4 (i.e. from the first layer position to the second).

Sweep: `return_pong` (True/False)
  True  -- forward trip + return trip (atoms end up back in layer 1).
  False -- forward trip only (atoms end up at the second-layer focal plane).

Expected result: ~100% survival in both configurations when depth_diff is large
enough that the two layers are axially well-separated but small enough for clean
transport.

Run:
    cd pyctrl && python YbScans/TwoLayerLoadScan.py
    cd pyctrl && python YbScans/TwoLayerLoadScan.py --depth_diff 15
    cd pyctrl && python YbScans/TwoLayerLoadScan.py --reps 1
"""

import argparse
import math
import os
import sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
BASE_PHASE_PATH = "phase/33x33_uniform.pt"
BASE_LAYER1_Z4 = -5.0          # z4 of the in-focus (imaging-plane) layer
DEFAULT_DEPTH_DIFF = 15.0      # separation between the two layers in ANSI Z4 rad


# ---------------------------------------------------------------------------
# Superposition phase code -- runs on the SLM server via /eval
# ---------------------------------------------------------------------------

def _superposition_eval_code(depth_diff: float, out_name: str) -> str:
    """Return the Python snippet to compute + save the superposition phase on the SLM server.

    Uses the same ANSI Z4 formula as pingponggrating (depth_mode) so the layer
    positions are byte-consistent with the rearrangement frames.
    """
    z1 = BASE_LAYER1_Z4
    z2 = BASE_LAYER1_Z4 + float(depth_diff)
    return f"""
import torch as _torch, numpy as _np, math as _math
from pathlib import Path as _Path
import slmnet.experimental.slm_server as _ss

_root = _ss._SLM_PROJECT_ROOT

_obj = _torch.load(str(_root / {BASE_PHASE_PATH!r}), weights_only=False, map_location='cpu')
if hasattr(_obj, 'detach'):
    _obj = _obj.detach().cpu().numpy()
_base = _np.asarray(_obj, dtype=_np.float32)

_H, _W = _base.shape
_yy = _np.linspace(-1.0, 1.0, _H, dtype=_np.float32).reshape(_H, 1)
_xx = _np.linspace(-1.0, 1.0, _W, dtype=_np.float32).reshape(1, _W)
# ANSI Z4 (defocus): sqrt(3)*(2*rho^2-1); same formula as rearrange_actual.py depth_mode
_z4_map = (_math.sqrt(3.0) * (2.0 * (_xx**2 + _yy**2) - 1.0)).astype(_np.float32)

_z1, _z2 = {z1!r}, {z2!r}
_c1 = _np.exp(1j * (_base + _z1 * _z4_map))
_c2 = _np.exp(1j * (_base + _z2 * _z4_map))
_superpos = _np.angle(_c1 + _c2).astype(_np.float32)

_out = str(_root / {out_name!r})
_torch.save(_torch.tensor(_superpos), _out)
result = {out_name!r}
"""


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("", "lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d) if d else root
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def TwoLayerLoadScan(url=None, reps=None, depth_diff=DEFAULT_DEPTH_DIFF):
    """Build + submit the two-layer load / depth-transport test scan."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan
    from devices.slm import get_client

    depth_diff = float(depth_diff)
    if depth_diff == 0.0:
        raise ValueError("depth_diff must be non-zero")

    # ---- Compute + cache the superposition loading phase on the SLM server ----
    # The phase is saved to phase/33x33_two_layer_dd<N>.pt relative to the SLM
    # project root.  The file persists until the next call with the same depth_diff
    # (deterministic filename), so re-running the scan script is idempotent.
    dd_key = "%.1f" % depth_diff
    out_name = "phase/33x33_two_layer_dd%s.pt" % dd_key
    code = _superposition_eval_code(depth_diff, out_name)

    print("[TwoLayerLoadScan] computing superposition phase (depth_diff=%s) on SLM server..."
          % dd_key)
    slm_client = get_client()
    slm_client.eval_python(code, timeout_s=60.0)
    print("[TwoLayerLoadScan] superposition saved to %s" % out_name)

    # ---- Build ScanGroup ----
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # -- warmup_kwargs: model + loading pattern (superposition, no extra Zernike) --
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    # The superposition phase has both z4 values baked in; no additional Zernike.
    rp.warmup_kwargs.initial_phase = out_name
    rp.warmup_kwargs.final_phase = out_name
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [0.0, 0.0, 0.0, 0.0, 0.0]
    rp.warmup_kwargs.extras.final_phase_zernike = [0.0, 0.0, 0.0, 0.0, 0.0]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # -- pingponggrating depth transport --
    # step_size carries the sign of depth_diff; nsteps is always positive.
    # The total z4 travel is step_size * nsteps = depth_diff (signed correctly).
    step_size = math.copysign(1.0, depth_diff)   # +1.0 or -1.0 rad/step
    nsteps = int(round(abs(depth_diff)))          # number of 1-rad steps

    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.nsteps = nsteps
    g().rearrange_kwargs.step_period_ms = 1.0
    g().rearrange_kwargs.extras.depth = True
    g().rearrange_kwargs.extras.step_size = step_size
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    # return_trip: True = full pingpong (forward + return); False = forward only.
    # Swept as dim 1 -- two scan points.
    g().rearrange_kwargs.extras.return_trip.scan(1, [True, False])

    # -- MOT / loading (same as SLMRearrangementScan.py) --
    g().BlueMOT.LoadingTime = 0.23
    g().BlueMOT.FreqDetuning = -44e6
    g().BlueMOT.Amp = 0.6
    g().GreenMOT.BiasCoilCurrent.X = 0.040
    g().GreenMOT.BiasCoilCurrent.Y = 0.268
    g().GreenMOT.BiasCoilCurrent.Z = 0.18
    g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    g().GreenMOT.CoolDown.FreqDetuning = 0.35e6
    g().GreenMOT.CoolDown.Amp = 0.25
    g().GreenMOT.CoolDown.HoldTime = 0.12
    g().GreenMOT.CoolDown.RampdownTime = 0.05

    # -- run params --
    rp.NumPerGroup = 100000
    # loading_defocus=0: superposition phase already has both z4 values baked in;
    # adding an extra z4 would shift BOTH layers away from their intended positions.
    rp.loading_defocus = 0
    rp.NumImages = 2     # img1 (loading detection) + img2 (survival detection)
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url,
                      label="TwoLayerLoadScan_dd%s" % dd_key, **opts)
    print("[TwoLayerLoadScan] submitted -> id %s  depth_diff=%s  nsteps=%d  step_size=%+.1f"
          % (did, dd_key, nsteps, step_size))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit TwoLayerLoadScan to the pyctrl backend.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--depth_diff", type=float, default=DEFAULT_DEPTH_DIFF,
                    help="Z4 separation between the two layers in radians (default: %.1f)"
                         % DEFAULT_DEPTH_DIFF)
    args = ap.parse_args()
    TwoLayerLoadScan(url=args.url, reps=args.reps, depth_diff=args.depth_diff)
