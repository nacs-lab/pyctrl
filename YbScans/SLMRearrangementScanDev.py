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

What the backend does (see engine_run.py / slm_runtime.py + RearrangeCommSeq / RearrangeCommSeq2):
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
# INIT_PATTERN = "33x33_feedback11"
# MIDDLE_PATTERN = "33x33_feedback11"
# TARGET_PATTERN = "33x33_feedback11"
INIT_PATTERN = os.environ.get("YB_INIT_PATTERN", "3013_tri")
MIDDLE_PATTERN = os.environ.get("YB_MIDDLE_PATTERN", "2198_kagome_res")
TARGET_PATTERN = os.environ.get("YB_TARGET_PATTERN", "2078_kagome")

# Rounds of rearrangement. 1 -> single-round (RearrangeCommSeq, 2 images). 2 -> two-round
# (RearrangeCommSeq2, 3 images: LOADING/MIDDLE/FINAL). This is the single source of truth; NumImages
# and the seq are both derived from it.
N_ROUNDS = int(os.environ.get("YB_N_ROUNDS", "2"))   # override for a 1-round 3013->2078 test (YB_N_ROUNDS=1)
# ------------------------------------------------------------------------------------ #

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/ampctrl_flat/ampctrl_flat_best.pth"

def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        # CONFIRMED
        "3013_tri": ("phase/tri_3013_camfb.pt", [0, 0, 0, 0, 0]),
        "2078_kagome": ("phase/kagome_2078_camfb.pt", [0, 0, 0, 0, 0]),
        "2198_kagome_res": ("phase/kagome_res_2198.pt", [0, 0, 0, 0, 0]),
        
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


def _registry_name(cfg):
    """The pattern name as the DETECTION REGISTRY + expConfig ByPattern know it = the phase-file
    basename (e.g. 'phase/tri_3013_camfb.pt' -> 'tri_3013_camfb'). The registry
    (yb_dashboard_state/patterns/<name>/) and ByPattern are keyed by THIS name, and the runner
    derives the same name for the frame-0 loading pattern (_first_loading_pattern) -- so
    imagePatternsJson + extras.*_pattern MUST use it, NOT the _pattern_cfg table alias. With the
    alias (e.g. '3013_tri') detector_for() misses the frame-0 cache AND the registry record and
    silently falls back to the DAY-FOLDER grid (wrong site count -> server 400 'bits length X !=
    expected Y'), and the ByPattern cooling/imaging overlay never applies (job #2548/#2549,
    2026-07-16)."""
    return os.path.splitext(os.path.basename(cfg["phase_path"].replace("\\", "/")))[0]


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
    items = [_pattern_item(_registry_name(init_cfg), init_cfg)]
    if n_rounds >= 2:
        items.append(_pattern_item(_registry_name(middle_cfg), middle_cfg))
    items.append(_pattern_item(_registry_name(target_cfg), target_cfg))
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
    # Dev sandbox: the 2-round path runs the Dev seq copy (adds the opt-in MinLoadAtoms /
    # MinMidAtoms shot gates; physics identical -- it delegates the build to RearrangeCommSeq2).
    seq_name = "RearrangeCommSeq2Dev" if n_rounds >= 2 else "RearrangeCommSeq"

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    if n_rounds >= 2:
        # Two-round: the server builds its per-shot stage list (initial -> middle -> final,
        # 2 transitions; "[2c] multi-round ENABLED" in the setup log) ONLY when a middle_phase
        # is supplied at the reset_params dequeue setup. Without it every rearrange call scores
        # against the full INITIAL grid, so round 2's middle-grid bits are rejected
        # ("bits length 2198 != expected 3013", 2026-07-16). str -> server-side filepath; the
        # server derives the middle grid from it and applies the same loading_zernike to the
        # middle WRITE phase (the round-1 bookend img2 is taken at).
        rp.warmup_kwargs.middle_phase = middle_cfg["phase_path"]
        rp.warmup_kwargs.extras.middle_phase_zernike = middle_cfg["baked_zernike"]
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
    
    # ---- OFF-PLANE SPHERICAL (Z12) NULL SWEEP (pingponggrating, depth mode) -------------
    # Measure the corrective primary-spherical coefficient (ANSI Z12, PV rad) model-free via
    # a depth-mode z-asymmetry null. Sim nailed the SIGN + MECHANISM (the objective's off-plane
    # spherical makes +z survive better than -z) but NOT the magnitude -- "amp 8" was an
    # illustrative unnormalized term, not Z12-PV-rad -- so measure it the same way as fill/center:
    # a null sweep. In depth mode the Z4 defocus is made pistonless (no_depth_piston, fill frac
    # PINNED at the measured f=0.65 centred on the beam) so the ONLY residual left to null is the
    # spherical. Adding a corrective Z12 on the TRANSIT frames cancels the system spherical; the
    # asymmetry A(C12)=surv(+z)-surv(-z) crosses ZERO at the correction (there +z and -z coincide
    # and both sit higher = deeper z-reach). We do it at a FIXED |step| in the regime where the
    # asymmetry is largest (best contrast), sweeping C12 in both z-directions.
    #   dim 1: step_size = {+|s|, -|s|} for |s| in {3.5, 4.0} rad PV Z4  => 4 SIGNED values (the
    #          two z-directions x two |step|). Scalar => pure depth. Split by |step| in analysis:
    #          A(C12; |s|) = surv(+|s|) - surv(-|s|), one zero-crossing per |step|. BONUS: if the
    #          null C12 DRIFTS between |s|=3.5 and 4.0 the spherical is defocus-dependent (-> move
    #          to a d-scaled Z12); if it holds, a single C12 correction suffices.
    #   dim 2: distortion_z12 = linspace(-8, 4, 25) PV rad -- scalar fold (rearrange_callbacks
    #          _fold_distortion_zernike) -> distortion_zernike = [0]*12 + [C12]; a LIST-valued
    #          swept axis would break the lab-side live curve. reverse_zernike LEFT DEFAULT (True):
    #          +C12 outward, -C12 on the return leg (matches the +z/-z transit the asymmetry probes).
    #   => 4 x 25 = 100 points.
    # RUN 1 (id 2483, C12 in [-4,4]x17): asymmetry unambiguous but the |step|=4.0 +z branch peaked
    # AT the C12=-4 edge (still climbing) -> its optimum sits BELOW -4, unbracketed. Zero-crossings
    # were |s|=3.5 -> C12~+0.37, |s|=4.0 -> C12~-1.66 (large drift => defocus-dependent spherical).
    # RUN 2 extends C12 to -8 to bracket the |step|=4.0 negative tail (0.5 spacing kept).
    # STEP_ABS = [3.5, 4.0]                                                       # |step| (PV rad Z4)
    # step_signed = [s for a in STEP_ABS for s in (a, -a)]                        # [+3.5,-3.5,+4.0,-4.0]
    # g().rearrange_kwargs.step_size.scan(1, step_signed)                        # +z / -z x |step|
    # g().rearrange_kwargs.extras.depth = True
    # g().rearrange_kwargs.extras.no_depth_piston = True
    # g().rearrange_kwargs.extras.piston = 0                                     # piston stays 0 throughout
    # # Fill frac / center PINNED at the measured beam (from the earlier fill/center null sweeps),
    # # so the pistonless Z4 subtracts the right beam-weighted mean and the ONLY residual is Z12.
    # g().rearrange_kwargs.extras.depth_fill_frac = 0.65                         # measured (pinned)
    # g().rearrange_kwargs.extras.depth_fill_center = [-0.0689, 0.0118]          # normalized [cx, cy]
    # # dim 2: corrective primary spherical, scalar C12 -> distortion_zernike[12] per shot.
    # g().rearrange_kwargs.extras.distortion_z12.scan(2, np.linspace(-8.0, 4.0, 25))
    # g().rearrange_kwargs.nsteps = 50
    # g().rearrange_kwargs.step_period_ms = 0.696  # pinned (period = 1 ms)
    # g().rearrange_kwargs.protocol = "pingponggrating"

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) -----------------------------
    # ss = [1, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5]
    # g().rearrange_kwargs.step_size.scan(1, [0] + ss + list(-1 * np.array(ss)))   # sweep (timing-vs-step_size)])
    # g().rearrange_kwargs.extras.depth = True
    g().rearrange_kwargs.extras.prob_hungarian = True
    if os.environ.get("YB_NSTEPS_SWEEP"):     # e.g. "10,20,...,100" -> sweep axis 1
        _ns = [int(x) for x in os.environ["YB_NSTEPS_SWEEP"].split(",")]
        g().rearrange_kwargs.nsteps.scan(1, _ns)
    else:
        # 2026-07-19 evening: full-chain optimum 70-80 (final fill 0.9675 med @70; 20/60 worse,
        # >=100 flat) -- was 50.
        g().rearrange_kwargs.nsteps = int(os.environ.get("YB_NSTEPS", "70"))
    g().rearrange_kwargs.step_period_ms = 0.696#.scan(2, 0.696 * np.array([1, 2, 3, 4, 5, 6, 7, 8]))  # sweep (timing-vs-step_period_ms)
    g().rearrange_kwargs.protocol = "rearrange2"
    
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False#.scan(2, [False, True])
    # g().rearrange_kwargs.extras.max_step_size = 0.75
    # g().rearrange_kwargs.extras.target_clamp.scan(2, [0.0, 0.15, 0.25, 1.0])
    # g().rearrange_kwargs.extras.mover_boost = 2.0
    # g().rearrange_kwargs.extras.block_max_size = 256
    # g().rearrange_kwargs.extras.pattern = "every-other"

    # g().rearrange_kwargs.extras.kagome_crop = 0.88
    # g().rearrange_kwargs.extras.model_bookend_pre = False   # default: no full-grid model bookend
    # g().rearrange_kwargs.extras.model_bookend_post = False
    # g().rearrange_kwargs.extras.hold_ms = 50
    g().rearrange_kwargs.extras.ifEnhanced = True
    # g().rearrange_kwargs.extras.precompute = True #.scan(1, [True, False])
    # g().rearrange_kwargs.extras.precompute_host = True   # host-resident precompute / pre-pin
    # g().rearrange_kwargs.extras.hw_sequence = False
    # g().rearrange_kwargs.extras.flip_immediate = False   # pinned False -- True wedges SLM DMA (bug-rearr-slm-write-dma-stall)
    # 2026-07-19 DEFOCUS override (env YB_DEFOCUS): sweep the whole focal plane -- loading_defocus,
    # rearrange model z4, AND the middle write (server applies loading_zernike to it) ALL move together
    # so the loaded atoms + rearrange transit frames + middle bookend stay co-planar (no transit mismatch).
    # 2026-07-19: matched-defocus sweep (z4+loading_defocus+middle together) on the live 2-round
    # rearrange -> -4 WINS: loading 71.8% (vs -5's 70.0), CV 11% (vs 16-19%, much more stable),
    # rearrange eff 0.964, 2-round final fill 0.940 (vs -5's 0.899). Was -5.
    _DEFOCUS = float(os.environ.get("YB_DEFOCUS", "-4"))
    g().rearrange_kwargs.extras.z4 = _DEFOCUS       # MATCH rp.loading_defocus (same focal plane)

    # ===================== DEV EXPERIMENT KNOBS (opt-in via env; defaults = production) =====================
    # 2026-07-19 (revised): RearrCoolAmp defaults to 0 -> NO cooling light for the whole rearrange
    # window (detect + GPU compute + SLM playback, ~1 s, twice per shot in 3013-way-split shallow
    # traps). Per-command accounting on run 20260719_040452 shows a ~18%/round arrival loss that is
    # nearly FLAT vs move distance and vs nsteps>=40 -> uniform dark-hold/morph heating, not
    # transport speed. (Earlier "mid d' 2.3" note here was a wrong-grid analysis artifact; true mid
    # d'=3.98 vs load 4.45.) Sweep the transit-cooling amp to null it (SLMHoldTestScan: 0.12/+0.2MHz).
    if os.environ.get("YB_REARR_COOL_AMP_SWEEP"):   # e.g. "0,0.04,0.08,0.12,0.18,0.25" -> axis 1
        _ca = [float(x) for x in os.environ["YB_REARR_COOL_AMP_SWEEP"].split(",")]
        g().rearrange_kwargs.extras.RearrCoolAmp.scan(1, _ca)
    elif os.environ.get("YB_REARR_COOL_AMP"):
        g().rearrange_kwargs.extras.RearrCoolAmp = float(os.environ["YB_REARR_COOL_AMP"])
    if os.environ.get("YB_REARR_COOL_DET"):     # MHz -> Hz offset from Resonance556mj0Freq
        g().rearrange_kwargs.extras.RearrCoolDet = float(os.environ["YB_REARR_COOL_DET"]) * 1e6
    # ======================================================================================================
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern:
    # RearrangeCommSeq(2) tags each bseq's image with the pattern below, so each resolves
    # cooling/imaging/VSLMServo from ByPattern[that pattern] AND detects with that pattern's registry
    # grid+thresholds. Single-round uses initial_pattern (img1) + final_pattern (img2); two-round
    # adds middle_pattern (img2), with final_pattern on img3.
    # REGISTRY names (phase basenames -- see _registry_name), not the table aliases: ByPattern +
    # per-frame detection both key off these.
    g().rearrange_kwargs.extras.initial_pattern = _registry_name(init_cfg)
    if n_rounds >= 2:
        g().rearrange_kwargs.extras.middle_pattern = _registry_name(middle_cfg)
    g().rearrange_kwargs.extras.final_pattern = _registry_name(target_cfg)

    # 2026-07-19 in-sequence 3013 tuning overrides (env), applied to the FIRST array's load+image so
    # the optimization runs on the REAL thermalized 2-round sequence:
    #   YB_IMG1PID / YB_IMG2PID -> BlueMOT.Img1/Img2PIDSet (399 imaging power for the initial image)
    #   YB_BLUELAC_DET (MHz) / YB_BLUELAC_AMP -> LAC.BlueLAC.{FreqDetuning,Amp} (enhanced loading)
    if os.environ.get("YB_IMG1PID"):
        g().BlueMOT.Img1PIDSet = float(os.environ["YB_IMG1PID"])
    if os.environ.get("YB_IMG2PID"):
        g().BlueMOT.Img2PIDSet = float(os.environ["YB_IMG2PID"])
    if os.environ.get("YB_BLUELAC_DET_SWEEP"):    # MHz list -> axis 1
        _bd = [float(x) * 1e6 for x in os.environ["YB_BLUELAC_DET_SWEEP"].split(",")]
        g().LAC.BlueLAC.FreqDetuning.scan(1, _bd)
    elif os.environ.get("YB_BLUELAC_DET"):
        g().LAC.BlueLAC.FreqDetuning = float(os.environ["YB_BLUELAC_DET"]) * 1e6
    if os.environ.get("YB_BLUELAC_AMP_SWEEP"):    # list -> axis 2
        _ba = [float(x) for x in os.environ["YB_BLUELAC_AMP_SWEEP"].split(",")]
        g().LAC.BlueLAC.Amp.scan(2, _ba)
    elif os.environ.get("YB_BLUELAC_AMP"):
        g().LAC.BlueLAC.Amp = float(os.environ["YB_BLUELAC_AMP"])
    if os.environ.get("YB_BLUELAC_TIME"):         # seconds
        g().LAC.BlueLAC.Time = float(os.environ["YB_BLUELAC_TIME"])

    # Min-load shot gates (RearrangeCommSeq2Dev only; 0/unset = off). Atoms, not rate.
    if os.environ.get("YB_MIN_LOAD"):
        g().rearrange_kwargs.extras.MinLoadAtoms = int(os.environ["YB_MIN_LOAD"])
    if os.environ.get("YB_MIN_MID"):
        g().rearrange_kwargs.extras.MinMidAtoms = int(os.environ["YB_MIN_MID"])
    # Mid/final image DOSE (RearrangeCommSeq2Dev only): DDS attenuation on top of the held PID
    # power (user-proposed alternative to a PID relock; AOM nonlinear -> sweep, don't compute).
    if os.environ.get("YB_MID_PAIRS"):
        # paired candidates "a1:a2,a1:a2,..." -> both beams co-vary on axis 1
        _prs = [tuple(float(y) for y in x.split(":"))
                for x in os.environ["YB_MID_PAIRS"].split(",")]
        g().rearrange_kwargs.extras.MidImgAmp1.scan(1, [p[0] for p in _prs])
        g().rearrange_kwargs.extras.MidImgAmp2.scan(1, [p[1] for p in _prs])
    elif os.environ.get("YB_MID_AMP1_SWEEP") and os.environ.get("YB_MID_AMP2_SWEEP"):
        # 2D per-beam mid-image dose: beam1 -> axis 1, beam2 -> axis 2
        _m1 = [float(x) for x in os.environ["YB_MID_AMP1_SWEEP"].split(",")]
        _m2 = [float(x) for x in os.environ["YB_MID_AMP2_SWEEP"].split(",")]
        g().rearrange_kwargs.extras.MidImgAmp1.scan(1, _m1)
        g().rearrange_kwargs.extras.MidImgAmp2.scan(2, _m2)
    elif os.environ.get("YB_MID_AMP_SWEEP"):      # list -> axis 1 (applies to BOTH mid beams)
        _ma = [float(x) for x in os.environ["YB_MID_AMP_SWEEP"].split(",")]
        g().rearrange_kwargs.extras.MidImgAmp1.scan(1, _ma)
        g().rearrange_kwargs.extras.MidImgAmp2.scan(1, _ma)
    else:
        if os.environ.get("YB_MID_AMP1"):
            g().rearrange_kwargs.extras.MidImgAmp1 = float(os.environ["YB_MID_AMP1"])
        if os.environ.get("YB_MID_AMP2"):
            g().rearrange_kwargs.extras.MidImgAmp2 = float(os.environ["YB_MID_AMP2"])
    if os.environ.get("YB_FIN_AMP_SWEEP"):        # list -> axis 1 (both final-image beams)
        _fa = [float(x) for x in os.environ["YB_FIN_AMP_SWEEP"].split(",")]
        g().rearrange_kwargs.extras.FinImgAmp1.scan(1, _fa)
        g().rearrange_kwargs.extras.FinImgAmp2.scan(1, _fa)
    else:
        if os.environ.get("YB_FIN_AMP1"):
            g().rearrange_kwargs.extras.FinImgAmp1 = float(os.environ["YB_FIN_AMP1"])
        if os.environ.get("YB_FIN_AMP2"):
            g().rearrange_kwargs.extras.FinImgAmp2 = float(os.environ["YB_FIN_AMP2"])
    # PID relock variant (fallback; RelockPIDs=1 re-engages the 399 imaging PID per bseq)
    if os.environ.get("YB_RELOCK"):
        g().rearrange_kwargs.extras.RelockPIDs = 1
        if os.environ.get("YB_RELOCK_MS"):
            g().rearrange_kwargs.extras.RelockTime = float(os.environ["YB_RELOCK_MS"]) * 1e-3
    # Trap depth (532 servo setpoint, whole shot -- root bseq sets it; per-pattern ByPattern
    # values don't reach the mid/final bseqs since no step there re-reads Init.VSLMServo).
    if os.environ.get("YB_VSLM_SWEEP"):           # volts list -> axis 1
        _vs = [float(x) for x in os.environ["YB_VSLM_SWEEP"].split(",")]
        g().Init.VSLMServo.scan(1, _vs)
    elif os.environ.get("YB_VSLM"):
        g().Init.VSLMServo = float(os.environ["YB_VSLM"])
    # Inter-image 556 X+h recool time (Cool556hXStep g.Time, default 5 ms). The all-stays
    # controls point at WARM atoms entering the round windows; sweep to rethermalize.
    if os.environ.get("YB_COOL556_TIME_SWEEP"):   # ms list -> axis 1
        _ct = [float(x) * 1e-3 for x in os.environ["YB_COOL556_TIME_SWEEP"].split(",")]
        g().Cool556.Time.scan(1, _ct)
    elif os.environ.get("YB_COOL556_TIME"):       # ms
        g().Cool556.Time = float(os.environ["YB_COOL556_TIME"]) * 1e-3

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
    rp.loading_defocus = _DEFOCUS                 # matched to rearrange z4 (YB_DEFOCUS)
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
    desc = os.environ.get("YB_SCAN_DESC",
                          "Dev 2-round rearrange sandbox (SLMRearrangementScanDev)")
    did = ybStartScan(seq_name, g, url=url, label="SLMRearrangementScanDev",
                      description=desc, **opts)
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
