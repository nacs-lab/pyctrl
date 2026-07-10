"""SLMRearrangementScan.py -- pyctrl port of ``matlab_new/YbScans/SLMRearrangementScan.m``.

Builds the SLM-rearrangement ScanGroup and submits it to the RUNNING pyctrl backend over ZMQ.
Like the other YbScans ports, this only BUILDS the ScanGroup + sends the descriptor JSON; the
backend (run loop) does the per-shot rearrangement.

Two variants, dispatched on ``N_ROUNDS`` (the single source of truth, mirroring the MATLAB scan):
  * N_ROUNDS == 1 -> RearrangeCommSeq   (img1 -> rearrange -> img2; NumImages = 2). Two patterns:
                     LOADING (initial) + FINAL (target).
  * N_ROUNDS >= 2 -> RearrangeCommSeq2  (img1 -> rearrange -> img2 -> rearrange -> img3;
                     NumImages = 3). THREE patterns: LOADING + MIDDLE + FINAL, imaged in each,
                     with a rearrangement right after the first two images.

Pattern-write policy (both variants): the LOADING (initial) phase is written once at scan start by
the scan-long SlmScanSession (and re-written if the ``slm`` lock is lost). The MIDDLE / FINAL
patterns are ASSUMED already on the SLM -- produced by the rearrange() calls -- so the middle and
final images just cool + image as fast as possible (no phase re-write). Each image is DETECTED with
its OWN per-pattern registry grid + thresholds (via imagePatternsJson + rearrange_kwargs.extras.
{initial,middle,final}_pattern), so the three patterns may be genuinely different arrays / spot
counts -- provided the lab detection agrees with what the SLM server scores that round.

What the backend does (see runner.py + RearrangeCommSeq / RearrangeCommSeq2):
  * AT DEQUEUE -- grab the scan-long ``slm`` lock, write the loading phase, push the initial
    ``setup_rearrangement`` (model + phases + ``reset_params=True``).
  * PER SHOT   -- grab the ``compute`` lock, push setup_rearrangement (swept params, sticky),
    reload_rearrange, then per round: detect bits/probs from that round's image (its own pattern),
    rearrange(), store the image; the final image runs update_rearrange; release compute; keepalive.

Run it:
    cd pyctrl
    python YbScans/SLMRearrangementScan.py
    python YbScans/SLMRearrangementScan.py --reps 1
    python YbScans/SLMRearrangementScan.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url, AND the SLM server must be reachable (the
scan-long slm lock is mandatory -- the run errors if it can't be acquired).
"""

import argparse
import json
import os
import sys
import numpy as np


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# LOADING (initial) / MIDDLE / FINAL SLM patterns, resolved to a server-side phase + baked Zernike
# by _pattern_cfg below (port of ybLoadingPatternCfg.m). MIDDLE is only used when N_ROUNDS >= 2. For
# a plain rearrangement leave them equal; set them apart to rearrange FROM one pattern THROUGH a
# middle INTO another. The rearrangement MODEL (warmup_kwargs.model_filename) must match the family.
INIT_PATTERN = "33x33_feedback11"
MIDDLE_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

# Rounds of rearrangement. 1 -> single-round (RearrangeCommSeq, 2 images). 2 -> two-round
# (RearrangeCommSeq2, 3 images: LOADING/MIDDLE/FINAL). This is the single source of truth; NumImages
# and the seq are both derived from it.
N_ROUNDS = 1
# ------------------------------------------------------------------------------------ #

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        # CONFIRMED
        "47x47_feedbackwarm4": ("phase/47x47_feedbackwarm4.pt", [0, 0, 0, 0, 0]),
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, -0.75]),
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        # NAME-IMPLIED (confirm the baked Zernike before trusting)
        "33x33_feedback9": ("phase/33x33_feedback9.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),  # 2026-07-10 fb9 depth-reflattened (post optics move); production successor
        "11x11withzernike-4":   ("phase/11x11withzernike-4.pt",   [0, 0, 0, 0, -4]),
        "10x10_z4eq8":          ("phase/10x10_z4eq8.pt",          [0, 0, 0, 0, -8]),
        "15x15_z4eq8":          ("phase/15x15_z4eq8.pt",          [0, 0, 0, 0, -8]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    """One imagePatternsJson entry (per camera frame): name + base phase + baked Zernike to strip.
    ``order='col'`` matches the runner's synthesized default + the server sweep_order the detection
    grid is derived in, so the per-pattern registry grid lines up with what the server scores."""
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(n_rounds, init_cfg, middle_cfg, target_cfg):
    """Per-frame detection declaration. Single-round: [LOADING, FINAL]. Two-round:
    [LOADING, MIDDLE, FINAL]. Each frame is detected against its own per-pattern registry grid, so
    the three arrays may differ in spot count / order (as long as the lab agrees with the server for
    the round it feeds). Explicit imagePatternsJson wins over the runner's 2-frame auto-synthesis
    (which does not know about a MIDDLE pattern)."""
    items = [_pattern_item(INIT_PATTERN, init_cfg)]
    if n_rounds >= 2:
        items.append(_pattern_item(MIDDLE_PATTERN, middle_cfg))
    items.append(_pattern_item(TARGET_PATTERN, target_cfg))
    return json.dumps(items)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build():
    """Build (do NOT submit) the SLM rearrangement ScanGroup. Returns ``(seq_name, g)`` -- the seq
    to run (RearrangeCommSeq / RearrangeCommSeq2, per N_ROUNDS) and the configured ScanGroup. Kept
    separate from :func:`SLMRearrangementScan` so it can be exercised offline WITHOUT touching the
    live backend (the scan-verification convention)."""
    _bootstrap()
    from scan_group import ScanGroup

    n_rounds = max(int(N_ROUNDS), 1)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    middle_cfg = _pattern_cfg(MIDDLE_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    seq_name = "RearrangeCommSeq2" if n_rounds >= 2 else "RearrangeCommSeq"

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
    # g().rearrange_kwargs.step_size.scan(1, np.linspace(0, 3, 15))   # sweep (timing-vs-step_size)])
    # g().rearrange_kwargs.extras.phase_alpha.scan(2, [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
    g().rearrange_kwargs.nsteps.scan(1, [20, 40, 60, 80, 100, 120])
    g().rearrange_kwargs.step_period_ms = 0.696#.scan(1, [1, 1.5, 2, 2.3, 2.5, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.5, 3.7, 4])   # pinned (period = 1 ms)
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False#.scan(2, [False, True])
    g().rearrange_kwargs.extras.max_step_size = 0.75
    # g().rearrange_kwargs.extras.target_clamp = 0#.scan(1, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.pattern = "every-other"

    # g().rearrange_kwargs.extras.kagome_crop = 0.88
    # g().rearrange_kwargs.extras.model_bookend_pre = False   # default: no full-grid model bookend
    # g().rearrange_kwargs.extras.model_bookend_post = False
    # g().rearrange_kwargs.extras.hold_ms = 50
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False #.scan(1, [True, False])
    g().rearrange_kwargs.extras.precompute_host = False   # host-resident precompute / pre-pin
    # g().rearrange_kwargs.extras.hw_sequence = False
    # g().rearrange_kwargs.extras.flip_immediate = False   # pinned False -- True wedges SLM DMA (bug-rearr-slm-write-dma-stall)
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern:
    # RearrangeCommSeq(2) tags each bseq's image with the pattern below, so each resolves
    # cooling/imaging/VSLMServo from ByPattern[that pattern] AND detects with that pattern's registry
    # grid+thresholds. Single-round uses initial_pattern (img1) + final_pattern (img2); two-round
    # adds middle_pattern (img2), with final_pattern on img3.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    if n_rounds >= 2:
        g().rearrange_kwargs.extras.middle_pattern = MIDDLE_PATTERN
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
    # Per-frame detection declaration: [LOADING, (MIDDLE,) FINAL]. Explicit -> wins over the
    # runner's 2-frame auto-synthesis (which has no MIDDLE), so a two-round scan detects img2 with
    # the middle pattern's grid.
    rp.imagePatternsJson = _image_patterns_json(n_rounds, init_cfg, middle_cfg, target_cfg)

    return seq_name, g


def SLMRearrangementScan(url=None, reps=None):
    """Build + SUBMIT the SLM rearrangement scan (single- or two-round per N_ROUNDS) to the running
    pyctrl backend over ZMQ. Returns the descriptor id."""
    seq_name, g = build()
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="SLMRearrangementScan", **opts)
    print("submitted SLMRearrangementScan (%d round(s) -> %s) -> descriptor id %s (url=%s)"
          % (max(int(N_ROUNDS), 1), seq_name, did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit SLMRearrangementScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    args = ap.parse_args()
    SLMRearrangementScan(url=args.url, reps=args.reps)
