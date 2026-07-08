"""TwoLayerIsolationProofScan.py -- ground-truth proof of front-vs-back layer classification.

Loads the FULL 2-layer bifocal array (2x11x11_5um, xy-coincident layers), images it (img1),
then runs a 3-D rearrangement whose TARGET is ONE layer only (``pattern="front_layer"``,
sign picks which): with ``z_weight_px_per_rad`` set prohibitively large, cross-layer
assignments are forbidden, so kept-layer atoms map to their OWN sites (zero motion) and the
other layer's traps are dropped from the hologram -> its atoms are released. img2 then shows
ONLY the kept layer.

Proof logic (per user 2026-07-05): classify img1 per-layer with the spot-shape template
classifier (offline); img2 is physical ground truth for which img1 atoms were in the kept
layer. Agreement/confusion between img1's kept-channel classification and img2 occupancy =
direct front-vs-back classification fidelity. Run once with --sign +1 (keep max-z layer =
baked +2.7778 = the sharp/narrow channel at write -8.25) and once with --sign -1.

Notes:
  * dedup_xy_knm = 0 -- the bifocal layers share xy; dedup would collapse the 242 3-D sites.
  * detection declared under name "2x11x11_5um_3d" so the 121-site production registry
    record (order=col, dedup on) is NOT clobbered by this 242-site no-dedup derive.
  * loading defocus -8.25 = the established per-layer readout operating point.
  * bits->assignment degeneracy is harmless here: with the identity-mapping target the kept
    hologram is front-layer traps at detected-atom xy regardless of per-plane bit labels.
  * This is the first live on-rig 3-D rearrange (offline-validated 2026-06-13);
    gpu_target_graph=True is REQUIRED (dispatcher raises otherwise).

Run:
    cd pyctrl
    python YbScans/RearrangeDiagnostics/TwoLayerIsolationProofScan.py --sign +1 --reps 150
    python YbScans/RearrangeDiagnostics/TwoLayerIsolationProofScan.py --sign -1 --reps 150
"""

import argparse
import json
import os
import sys


PATTERN = "2x11x11_5um"
PHASE_PATH = "phase/2x11x11_5um.pt"
DETECT_NAME = "2x11x11_5um_3d"            # separate registry record (242 sites, no dedup)
PLANES_Z_RAD = [-2.7778, 2.7778]
LOADING_DEFOCUS = -8.25                    # per-layer readout operating point
MODEL_FILENAME = "SLMnet/checkpoints/experiment_3d/models/base5x5_fp16/best_model.pth"
Z_WEIGHT_FORBID = 1000.0                   # cross-layer move cost (1000*5.56)^2 >> any xy move


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def TwoLayerIsolationProofScan(url=None, sign=1, reps=150, nsteps=20, direct=False):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # direct mode: protocol "direct_load" writes the WGS final_phase ONCE per shot (no CNN
    # model frames, no assignment) -> the drop window is a single SLM write. final_phase =
    # the single-layer WGS hologram (same 121 xy, baked z4 = sign*2.7778, 150-iter WGS,
    # sim CV 5.2%, exactly 121 peaks). The 2026-07-06 model-frame runs lost ~80% of ALL
    # atoms in the 20-frame window (model carrier z4 -8.25 likely outside training range),
    # drowning the discrimination -- direct mode removes that loss channel.
    final_phase = ("phase/11x11_5um_%s.pt" % ("frontonly" if sign > 0 else "backonly")
                   if direct else PHASE_PATH)

    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = final_phase
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [0, 0, 0, 0, 0]
    rp.warmup_kwargs.extras.final_phase_zernike = [0, 0, 0, 0, 0]
    rp.warmup_kwargs.extras.planes_z_rad = PLANES_Z_RAD
    rp.warmup_kwargs.extras.dedup_xy_knm = 0          # bifocal same-xy: keep all 242 sites
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    g().rearrange_kwargs.nsteps = int(nsteps)
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "direct_load" if direct else "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.pattern = "front_layer"
    g().rearrange_kwargs.extras.front_layer_sign = int(sign)
    g().rearrange_kwargs.extras.z_weight_px_per_rad = Z_WEIGHT_FORBID
    g().rearrange_kwargs.extras.gpu_target_graph = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = LOADING_DEFOCUS
    # DETECT_NAME (not PATTERN): extras.initial_pattern keys BOTH the rearrange detector
    # calibration (must be the 242-site all-pass record so bits match the server's no-dedup
    # init_grid) AND the per-bseq ByPattern overlay (the _3d alias in expConfig maps to the
    # same config dict, so servo/exposure/cooling stay correct).
    g().rearrange_kwargs.extras.initial_pattern = DETECT_NAME
    g().rearrange_kwargs.extras.final_pattern = DETECT_NAME

    rp.isRearrange = 1
    rp.NumPerGroup = int(reps)
    # loading_phase = the _3d BYTE-COPY of the same hologram: _first_loading_pattern derives the
    # detector-calibration name from the PHASE BASENAME (immune to imagePatternsJson/extras), and
    # the 242-site all-pass calibration + ByPattern alias live under "2x11x11_5um_3d".
    rp.loading_phase = "phase/2x11x11_5um_3d.pt"
    rp.loading_defocus = LOADING_DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 0
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    pat = {"name": DETECT_NAME, "base_phase_path": PHASE_PATH,
           "order": "col_up", "legacy_zerniked": False,
           "planes_z_rad": PLANES_Z_RAD}
    rp.imagePatternsJson = json.dumps([pat, pat])

    which = "front(max-z, baked +2.7778 = narrow@-8.25)" if sign > 0 else "back(min-z, baked -2.7778)"
    did = ybStartScan(
        "RearrangeCommSeq", g, url=url, rep=int(reps),
        label="LayerIsolation_%s%s" % ("front" if sign > 0 else "back",
                                       "_direct" if direct else ""),
        description=("PROOF (layer isolation, user design 2026-07-05): full 2-layer load at "
                     "z4 -8.25 -> img1 -> 3-D rearrange with target = %s only "
                     "(pattern=front_layer sign=%+d, z_weight=%g forbids cross-layer moves -> "
                     "identity mapping, other layer's traps dropped) -> img2 = ground truth. "
                     "img1 spot-shape classification vs img2 occupancy = front-vs-back "
                     "classification fidelity. First live on-rig 3-D rearrange." )
                    % (which, sign, Z_WEIGHT_FORBID))
    print("submitted LayerIsolationProof sign=%+d -> id %s (%d shots)" % (sign, did, reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--sign", type=int, choices=[1, -1], default=1)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--nsteps", type=int, default=20)
    ap.add_argument("--direct", action="store_true",
                    help="direct_load: write the single-layer WGS hologram once per shot "
                         "(no model frames) -- minimal-loss drop window")
    args = ap.parse_args()
    TwoLayerIsolationProofScan(url=args.url, sign=args.sign, reps=args.reps, nsteps=args.nsteps,
                               direct=args.direct)
