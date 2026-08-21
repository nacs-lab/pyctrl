"""TwoLayerLoadScan.py -- two-layer SLM loading + depth-transport test (2-D sweep).

Loads atoms into a SUPERPOSITION of two axially-separated tweezer arrays, then
measures survival under a pingponggrating depth trip.

For a given inter-layer depth `dd` the loading pattern combines two WGS phases:
  front : 33x33_uniform with ANSI Z4 = -5 rad           (the imaging plane)
  back  : 33x33_uniform with ANSI Z4 = -5 + dd rad       (the second layer)
The complex-field superposition is angle(exp(i*phi_front) + exp(i*phi_back)) --
both amplitude and the 50/50 power split are accepted; we wipe the amplitude and
keep only the phase angle, giving a dual-focal-plane SLM pattern.

After loading, a single pingponggrating depth sweep walks ALL tweezers by `dd`
radians of Z4 (i.e. front layer -> back-layer position).

Sweeps (2-D):
  dim 1  `return_trip` (True/False)
    True  -- forward trip + return trip (atoms end up back in the front layer).
    False -- forward trip only (atoms end up at the back-layer focal plane;
             img2 then images the back layer).
  dim 2  inter-layer depth `dd` (default [10, 20, 30, 40] rad of Z4)
    nsteps == dd (step_size = +1 rad/step) AND the loading superposition's
    back-layer Z4 == -5 + dd are swept together, so the transport always moves
    the front layer exactly onto the back layer for every point.

How the per-point loading hologram is applied (no per-shot grid FFT):
  The depth sweep swaps the WGS loading phase per shot via
  `rearrange_kwargs.initial_phase`; the server's `reload_rearrange` writes that
  WGS-initial to the SLM right before each load. The trap LATTICE is the same
  for every shot (the two layers differ only in Z4 -> identical far-field site
  positions, == the clean single-layer `33x33_uniform` grid), so:
    * the grid is derived ONCE at dequeue from `33x33_uniform` (1068 sites) --
      NOT from a superposition, whose defocused + interfering second layer
      derives thousands of spurious peaks (~9098) and fails the server's
      bijection check; and
    * `skip_grid_derive` tells the server to reuse that cached grid instead of
      re-deriving it (an FFT) on every per-shot phase swap. (Requires the
      server-side `skip_grid_derive` flag; needs a server restart.)
  Detection thresholds are still saved/used under the two-layer pattern name
  `33x33_two_layer` (via an explicit imagePatternsJson pairing that name with
  the `33x33_uniform` base grid), NOT base `33x33_uniform` -- so the two-layer
  brightness statistics get their own threshold folder.

Run:
    cd pyctrl && python YbScans/TwoLayerLoadScan.py
    cd pyctrl && python YbScans/TwoLayerLoadScan.py --depths 15 25 35
    cd pyctrl && python YbScans/TwoLayerLoadScan.py --reps 1
"""

import argparse
import json
import os
import sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
BASE_PHASE_PATH = "phase/33x33_uniform.pt"
BASE_LAYER1_Z4 = -5.0               # z4 of the in-focus (imaging-plane) front layer
DEFAULT_DEPTHS = [10.0, 20.0, 30.0, 40.0]   # inter-layer separations to sweep (ANSI Z4 rad)
# Representative two-layer pattern -> detection grid + threshold-registry folder.
TWO_LAYER_PATTERN_NAME = "33x33_two_layer"
# Back-layer x/y offset (xy_half_grid mode): a linear SLM phase ramp of this many cycles across
# the hologram in each axis -> shifts the back array by ~half a grid cell. Measured: the
# 33x33_uniform lattice spacing is 195 px in an 8192-pt FFT of the 1024-px hologram = 24.4 cycles
# across the SLM, so half = ~12.2. (Approximate; exact half-grid not required -- images only.)
HALF_GRID_CYCLES = 195.0 / 16.0


# ---------------------------------------------------------------------------
# Superposition phase code -- runs on the SLM server via /eval
# ---------------------------------------------------------------------------

def _superposition_eval_code(depth_diff: float, out_name: str, xy_cycles: float = 0.0) -> str:
    """Return the Python snippet to compute + save the superposition phase on the SLM server.

    Uses the same ANSI Z4 formula as pingponggrating (depth_mode) so the layer
    positions are byte-consistent with the rearrangement frames. ``xy_cycles`` adds a linear
    phase ramp to the BACK layer (Fourier shift) -> the back array is offset in x and y by
    ``xy_cycles`` cycles-across-hologram (~half a grid cell for HALF_GRID_CYCLES); 0 = z-only.
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
# Back-layer x/y Fourier-shift ramp ({xy_cycles!r} cycles across the hologram: +x right, +y up).
_ncyc = {xy_cycles!r}
_col = _np.arange(_W, dtype=_np.float32).reshape(1, _W)
_row = _np.arange(_H, dtype=_np.float32).reshape(_H, 1)
_rampx = (2.0 * _math.pi * _ncyc / _W) * _col
_rampy = -(2.0 * _math.pi * _ncyc / _H) * _row
_c1 = _np.exp(1j * (_base + _z1 * _z4_map))
_c2 = _np.exp(1j * (_base + _z2 * _z4_map + _rampx + _rampy))
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

def TwoLayerLoadScan(url=None, reps=None, depths=None, xy_half_grid=False):
    """Build + submit the two-layer load / depth-transport test scan (2-D sweep).

    ``xy_half_grid`` (bool): if True, the BACK layer of the loading superposition is additionally
    offset by ~half a grid cell in x AND y (fixed), on top of the swept z-depth -- i.e. the second
    layer is offset in all 3 dims. Detection/grid is NOT meaningful for the offset back layer
    (its atoms sit at interstitial sites); intended for inspecting raw camera images.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan
    from devices.slm import get_client

    depths = [float(d) for d in (depths if depths else DEFAULT_DEPTHS)]
    if any(d <= 0.0 for d in depths):
        raise ValueError("depths must be positive (back z4 = front + dd)")

    xy_cycles = HALF_GRID_CYCLES if xy_half_grid else 0.0
    tag = "xyz_" if xy_half_grid else ""

    slm_client = get_client()

    # ---- Compute + cache the per-depth superposition loading phases on the server ----
    # Deterministic filenames -> re-running the script is idempotent.
    dd_paths = []
    for dd in depths:
        out_name = "phase/33x33_two_layer_%sdd%.1f.pt" % (tag, dd)
        print("[TwoLayerLoadScan] computing %s superposition phase (depth=%.1f, xy_cycles=%.2f)..."
              % (("xyz" if xy_half_grid else "z"), dd, xy_cycles))
        slm_client.eval_python(_superposition_eval_code(dd, out_name, xy_cycles), timeout_s=60.0)
        dd_paths.append(out_name)

    # ---- Build ScanGroup ----
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # -- warmup_kwargs: server grid (from the clean base) + scan-start state (sent ONCE at
    #    dequeue with reset_params). The physical per-depth loading holograms are swapped per
    #    shot below via rearrange_kwargs.initial_phase. --
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    # Grid source = clean single-layer base (1068 sites). Deriving from a two-layer
    # SUPERPOSITION fails: the defocused/interfering back layer yields ~9098 spurious peaks
    # and the server's bijection check rejects it (no grid cached -> every shot fails). The
    # front layer sits at the SAME lateral positions as 33x33_uniform, so this grid is correct.
    rp.warmup_kwargs.initial_phase = BASE_PHASE_PATH
    rp.warmup_kwargs.final_phase = BASE_PHASE_PATH
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

    # -- pingponggrating depth transport (per-shot, sweepable) --
    g().rearrange_kwargs.protocol = "pingponggrating"
    g().rearrange_kwargs.step_period_ms = 1.0
    g().rearrange_kwargs.extras.depth = True
    g().rearrange_kwargs.extras.step_size = 1.0   # +1 rad Z4 per step (front -> back)
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    # Per-shot: swap the WGS loading phase to the swept depth's superposition WITHOUT
    # paying a per-shot grid-derivation FFT (the grid is depth-independent; reuse the cached
    # one from the dequeue setup). reload_rearrange writes the swapped WGS-initial to the SLM
    # before each load. Requires the server-side skip_grid_derive flag.
    g().rearrange_kwargs.skip_grid_derive = True

    # -- sweeps --
    # dim 1: return trip (out-and-back to the front layer) vs one-way (stay at the back layer).
    g().rearrange_kwargs.extras.return_trip.scan(1, [True, False])
    # dim 2: inter-layer depth. nsteps == dd (step_size=1) co-swept with the loading
    # superposition (back layer at z4=-5+dd) so the transport lands the front layer exactly
    # on the back layer for every point. Both on dim 2 -> co-vary by index.
    g().rearrange_kwargs.nsteps.scan(2, [int(round(d)) for d in depths])
    g().rearrange_kwargs.initial_phase.scan(2, dd_paths)

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
    # loading_defocus=0: the superposition phases already have both z4 values baked in;
    # adding an extra z4 would shift BOTH layers away from their intended positions.
    rp.loading_defocus = 0
    rp.NumImages = 2     # img1 (loading detection) + img2 (survival detection)
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    # Detection pattern: thresholds + grid saved/used under the two-layer name (NOT base
    # 33x33_uniform), while the grid itself is derived from the clean 33x33_uniform base. One
    # entry per camera frame (img1 loading, img2 survival).
    _pat = {"name": TWO_LAYER_PATTERN_NAME, "base_phase_path": BASE_PHASE_PATH,
            "order": "col", "legacy_zerniked": False}
    rp.imagePatternsJson = json.dumps([_pat, _pat])

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    label = "TwoLayerLoadScan_%sdd%s" % (tag, "-".join("%g" % d for d in depths))
    did = ybStartScan("RearrangeCommSeq", g, url=url, label=label, **opts)
    print("[TwoLayerLoadScan] submitted -> id %s  depths=%s  xy_half_grid=%s  (2 return modes x %d)"
          % (did, [("%g" % d) for d in depths], xy_half_grid, len(depths)))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit TwoLayerLoadScan to the pyctrl backend.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--depths", type=float, nargs="+", default=None,
                    help="inter-layer separations to sweep, ANSI Z4 rad (default: %s)"
                         % DEFAULT_DEPTHS)
    ap.add_argument("--xy_half_grid", action="store_true",
                    help="offset the back layer by ~half a grid cell in x AND y (3-D offset)")
    args = ap.parse_args()
    TwoLayerLoadScan(url=args.url, reps=args.reps, depths=args.depths,
                     xy_half_grid=args.xy_half_grid)
