"""SLMRearrangementScan.py -- pyctrl port of ``matlab_new/YbScans/SLMRearrangementScan.m``.

Builds the single-round SLM-rearrangement ScanGroup (seq = ``RearrangeCommSeq``) and submits it to
the RUNNING pyctrl backend over ZMQ. Like the other YbScans ports, this only BUILDS the ScanGroup
+ sends the descriptor JSON; the backend (run loop) does the per-shot rearrangement.

What the backend does with this scan (see runner.py + RearrangeCommSeq.py):
  * AT DEQUEUE -- grab the scan-long ``slm`` lock, write the loading (WGS) phase
    (``warmup_kwargs.initial_phase`` + ``runp().loading_defocus``), and push the initial
    ``setup_rearrangement`` (model + phases + ``reset_params=True``).
  * PER SHOT   -- grab the ``compute`` lock, push setup_rearrangement (the swept params, NO
    reset_params -> sticky), reload_rearrange, detect bits from img1, rearrange(bits), store
    img1; then detect bits from img2, update_rearrange, store img2; release compute; keepalive slm.

Two-round (RearrangeCommSeq2) is a follow-up; this file is the single-round scan.

Run it:
    cd pyctrl
    python YbScans/SLMRearrangementScan.py
    python YbScans/SLMRearrangementScan.py --reps 1
    python YbScans/SLMRearrangementScan.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url, AND the SLM server must be reachable (the
scan-long slm lock is mandatory -- the run errors if it can't be acquired).
"""

import argparse
import os
import sys


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# Initial (loading) and final (target) SLM patterns, resolved to a server-side phase + baked
# Zernike by _pattern_cfg below (port of ybLoadingPatternCfg.m). For a plain rearrangement leave
# them equal. The rearrangement MODEL (warmup_kwargs.model_filename) must match the pattern family.
INIT_PATTERN = "47x47_uniform"
TARGET_PATTERN = "33x33_uniform"
# ------------------------------------------------------------------------------------ #

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"
# MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        # CONFIRMED
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        # NAME-IMPLIED (confirm the baked Zernike before trusting)
        "33x33_1068_zernike-4": ("phase/33x33_1068_zernike-4.pt", [0, 0, 0, 0, -4]),
        "11x11withzernike-4":   ("phase/11x11withzernike-4.pt",   [0, 0, 0, 0, -4]),
        "10x10_z4eq8":          ("phase/10x10_z4eq8.pt",          [0, 0, 0, 0, -8]),
        "15x15_z4eq8":          ("phase/15x15_z4eq8.pt",          [0, 0, 0, 0, -8]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMRearrangementScan(url=None, reps=None):
    """Build + submit the single-round SLM rearrangement scan. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
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

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) -----------------------------
    g().rearrange_kwargs.nsteps = 100#.scan(1, list(range(100, 200, 40)))   # sweep (timing-vs-nsteps)
    g().rearrange_kwargs.step_period_ms = 1.0   # pinned (period = 1 ms)
    g().rearrange_kwargs.protocol = "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    #g().rearrange_kwargs.extras.pattern = "every-other"
    # g().rearrange_kwargs.extras.kagome_crop = 0.88
    # g().rearrange_kwargs.extras.model_bookend_pre = False   # default: no full-grid model bookend
    # g().rearrange_kwargs.extras.model_bookend_post = False
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute.scan(1, [True, False])
    g().rearrange_kwargs.extras.precompute_host = True   # host-resident precompute / pre-pin
    g().rearrange_kwargs.extras.hw_sequence = False
    # g().rearrange_kwargs.extras.flip_immediate = False   # pinned False -- True wedges SLM DMA (bug-rearr-slm-write-dma-stall)
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern): RearrangeCommSeq builds bseq2 (the
    # post-rearrangement image) against this FINAL/target pattern's params; bseq1 uses the initial
    # pattern (the scan-default = warmup_kwargs.initial_phase). No effect until ByPattern is
    # populated for these names (distinct from extras.pattern, the rearrangement protocol shape).
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- non-rearrangement scan settings ----------------------------------------------
    # MOT/loading: 2026-06-05 loading-rate optimization (copied from YbScans/LACScan.py
    # Phase-8 g() block; expConfig.py deliberately left untouched). ~1.9x faster cycle at
    # the same ~0.58 single-atom peak loading rate.
    # g().BlueMOT.LoadingTime = 0.23                    # was 0.5
    # g().BlueMOT.FreqDetuning = -44e6                  # was -40e6 (saturation knee moved left)
    # g().BlueMOT.Amp = 0.6
    # g().GreenMOT.BiasCoilCurrent.X = 0.040            # was 0.039
    # g().GreenMOT.BiasCoilCurrent.Y = 0.268            # was 0.27
    # g().GreenMOT.BiasCoilCurrent.Z = 0.18
    # g().GreenMOT.PowerBroaden.HandoverTime = 0.015    # was 0.030
    # g().GreenMOT.CoolDown.FreqDetuning = 0.35e6
    # g().GreenMOT.CoolDown.Amp = 0.25                  # was 0.20
    # g().GreenMOT.CoolDown.HoldTime = 0.12             # was 0.2
    # g().GreenMOT.CoolDown.RampdownTime = 0.05
    # g().LAC.BlueLAC.FreqDetuning = -3.8e6             # LAC kept at config default
    # g().LAC.BlueLAC.Amp = 0.17                        # LAC kept at config default

    # ---- run params (runp) ------------------------------------------------------------
    rp.NumPerGroup = 100000
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan
    # start (SlmScanSession). MATCHED to rearrange_kwargs.extras.z4 (the rearrange MODEL z4) so the
    # rearrangement model frames sit at the SAME focal plane as the loaded atoms (no transit defocus
    # mismatch). 33x33_uniform has no baked Zernike, so -5 is absolute.
    rp.loading_defocus = -5
    rp.NumImages = n_rounds + 1                  # img1 + one frame per round
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    # Hold the slm lock for the WHOLE scan + write the loading phase once at scan start.
    rp.useScanLongSlmLock = 1

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMRearrangementScan", **opts)
    print("submitted SLMRearrangementScan -> descriptor id %s (url=%s)" % (did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit SLMRearrangementScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    args = ap.parse_args()
    SLMRearrangementScan(url=args.url, reps=args.reps)
