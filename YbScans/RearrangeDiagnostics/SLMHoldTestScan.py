"""SLMHoldTestScan.py -- INDEX-MISMATCH diagnostic (the 05/27 "held-tweezer" test), pyctrl port.

WHY THIS EXISTS
---------------
The last production rearrangement scan peaked at TP ~66%, which is *below* the loading rate.
That is the classic signature of a scrambled bit<->site index map: loaded atoms get assigned /
moved to the WRONG targets, so even good loading can't yield good TP. We hit exactly this on
2026-05-27 -- a 90-degree rotation between the SLM/model "software" frame and the atom-imaging
"bit" frame (counting "from bottom-left, going up"), fixed back then by sending
``sweep_order="col_up"`` from MATLAB to the server.

This scan reproduces the 05/27 DIAGNOSTIC, not a physics run: it HOLDS a fixed (random) set of
~500 tweezers in place and images them. If the index map is consistent, the per-site survival
map lights up at EXACTLY the commanded sites; a residual rotation/permutation shows the survivors
at the rotated positions instead. The 556 cooling light is left on a little during the hold to
kick atoms out of "ghost" traps so they don't survive at non-commanded sites (reduces FP).

HOW IT WORKS (no seq/runtime/client code changes -- parameter-only)
------------------------------------------------------------------
Same warmup (model + phases + grid_rotation) as the problematic SLMRearrangementScan, so the
index frame under test is identical. Only the per-shot rearrange_kwargs change:
  * protocol            = "hold"  (server _protocol_hold_in_place: hold a fixed set, no transit)
  * extras.loaded_idx_override = [fixed random 500 of 1068]  (the held set; pinned every shot)
  * extras.ghost_fraction      = 0      (only commanded sites get traps; everything else dark)
  * extras.skip_final_phase    = True   (keep the sparse held frame on the SLM through img2; do
                                         NOT let the WGS bookend repopulate ghosts -- the explicit
                                         05/27 lesson that "washed out the contrast measurement")
  * extras.model_bookend_pre/post = False  (clean isolation, no extra transitions)
  * extras.RearrCoolAmp = 0.12, RearrCoolDet = 0.2e6  (the settled 05/27 cooling values; the
                                         seq already wires these onto 556MOTX + 556RydbergMOTh)

READOUT
-------
loaded_idx_override pins the SAME held set every shot, so the dashboard / run_analysis per-site
survival map should show high survival at the commanded sites and ~0 elsewhere. Overlay the
commanded set's expected camera positions against the bright (surviving) sites: a 90-degree
offset => the index frame is rotated (re-introduce sweep_order="col_up" / fix grid_rotation).

Run it (submits to the RUNNING pyctrl backend over ZMQ):
    cd pyctrl
    python YbScans/SLMHoldTestScan.py
    python YbScans/SLMHoldTestScan.py --reps 1000        # bounded shot count
Prereq: the pyctrl backend (run_monitor) must be up, AND the SLM server reachable.
"""

import argparse
import os
import random
import sys


# --------------------------- PATTERN SELECTION (match the 66%-TP scan) -------------------- #
# Identical to SLMRearrangementScan so the index frame under test is the same one that gave
# TP ~66%. For 33x33_uniform the derived init_grid is 1068 sites.
INIT_PATTERN = "33x33_uniform"
TARGET_PATTERN = "33x33_uniform"
MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"

# --------------------------- HELD SET (fixed, random, reproducible) ----------------------- #
# loaded_idx_override indices are 0-based INTO init_grid. N_INIT must equal the derived site
# count for INIT_PATTERN (33x33_uniform -> 1068, verified 2026-06-05). If the server logs an
# index error, lower N_INIT to the count it reports.
N_INIT = 1068
K_HELD = 500            # ~half the array; the model generates this many traps cleanly (it fails
                        # at ~100 -- see the 05/27 notebook). Random => spatially asymmetric, so
                        # any rotation/permutation is unmistakable in the survival map.
HELD_SEED = 20260605    # fixed seed -> identical held set every shot AND across sessions.
HELD_IDX = sorted(random.Random(HELD_SEED).sample(range(N_INIT), K_HELD))


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        "33x33_1068_zernike-4": ("phase/33x33_1068_zernike-4.pt", [0, 0, 0, 0, -4]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def SLMHoldTestScan(url=None, reps=None):
    """Build + submit the held-tweezer index-mismatch diagnostic. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (dequeue, reset_params) -- IDENTICAL to SLMRearrangementScan -----
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = init_cfg["baked_zernike"]
    rp.warmup_kwargs.extras.final_phase_zernike = target_cfg["baked_zernike"]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs (per-shot setup) -- the DIAGNOSTIC changes --------------------
    g().rearrange_kwargs.protocol = "hold"
    g().rearrange_kwargs.nsteps = 10               # hold barely uses transit; keep small + fixed
    g().rearrange_kwargs.step_period_ms = 2.0      # well above the ~0.7 ms SLM refresh floor
    # The held set: fixed (random) indices into init_grid, pinned every shot.
    g().rearrange_kwargs.extras.loaded_idx_override = HELD_IDX
    g().rearrange_kwargs.extras.ghost_fraction = 0          # only commanded sites get traps
    g().rearrange_kwargs.extras.skip_final_phase = True     # keep the sparse held frame through img2
    g().rearrange_kwargs.extras.model_bookend_pre = False   # clean isolation
    g().rearrange_kwargs.extras.model_bookend_post = False
    g().rearrange_kwargs.extras.hold_ms = 100               # SLM dwell on the held frame (KEY tunable)
    g().rearrange_kwargs.extras.precompute = True           # uniform per-frame timing
    g().rearrange_kwargs.extras.ifEnhanced = True           # match the production loading (BlueLAC)
    g().rearrange_kwargs.extras.z4 = -5                     # model frame at the loading focal plane
    # --- cooling light ON a little during the hold (kick atoms out of ghost traps; reduces FP) ---
    g().rearrange_kwargs.extras.RearrCoolAmp = 0.12         # 556 X + h beams (settled 05/27 value)
    g().rearrange_kwargs.extras.RearrCoolDet = 0.2e6        # +0.2 MHz from Resonance556mj0Freq

    # ---- non-rearrangement scan settings -- IDENTICAL to SLMRearrangementScan ----------
    g().BlueMOT.LoadingTime = 0.5
    g().GreenMOT.CoolDown.HoldTime = 0.2
    g().GreenMOT.BiasCoilCurrent.X = 0.039
    g().GreenMOT.BiasCoilCurrent.Y = 0.27
    g().GreenMOT.BiasCoilCurrent.Z = 0.18
    g().LAC.BlueLAC.FreqDetuning = -3.8e6
    g().LAC.BlueLAC.Amp = 0.17

    # ---- run params (runp) ------------------------------------------------------------
    # No sweep -> one scan point; NumPerGroup is the shot count. ~1000 shots gives solid per-site
    # survival stats (the pattern is usually obvious within a few hundred). Stop early once clear.
    rp.NumPerGroup = 1000
    rp.loading_defocus = -5                        # MATCH extras.z4 (same focal plane)
    rp.NumImages = n_rounds + 1                    # img1 (loading) + img2 (after hold)
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1                      # scan-long slm lock (mandatory for rearrange)

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMHoldTestScan", **opts)
    print("submitted SLMHoldTestScan -> descriptor id %s (url=%s)" % (did, url or "default"))
    print("held set: %d of %d sites, seed=%d, checksum(sum)=%d"
          % (len(HELD_IDX), N_INIT, HELD_SEED, sum(HELD_IDX)))
    print("  first 10: %s" % HELD_IDX[:10])
    print("  last  10: %s" % HELD_IDX[-10:])
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the held-tweezer index-mismatch diagnostic.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="pass count (0 = forever); omit -> derived from NumPerGroup")
    args = ap.parse_args()
    SLMHoldTestScan(url=args.url, reps=args.reps)
