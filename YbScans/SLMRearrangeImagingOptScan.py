"""SLMRearrangeImagingOptScan.py -- optimize the MIDDLE + FINAL image brightness of the REAL
two-round SLM rearrangement (RearrangeCommSeq2, tri_3013_camfb -> kagome_res_2198 ->
kagome_2078_camfb) by sweeping each frame's 399 imaging DDS amps.

Sibling of ``SLMRearrangementScan.py`` (which is left UNTOUCHED): same patterns, same
rearrangement parameters, same seq -- the ONLY difference is the two swept axes below. Commit the
winner back into ``expConfig.ByPattern[<pattern>].Imag399.Amp1/Amp2``, after which this scan is no
longer needed.

=============================== WHY AMPS AND NOT PIDSet ================================
IMAGING-POWER POLICY (2026-07-27; see the same block in SLMRearrangementScan.py and
yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md). The 399 imaging-power PID is
engaged ONCE, in the ROOT ``BlueMOTStep``, at the LOADING pattern's ``BlueMOT.Img1/Img2PIDSet``
(currently 0.5 / 0.5) and then HELD (PIDMode TTL -> 0) for the whole shot. There is no time to
re-PID mid-shot and a relock rails the integrator against a dark PD. So all three frames are taken
at ONE held optical power, and the ONLY per-frame brightness knob is each frame's DDS amps:

    Imag399.Amp1 -> AmpAbsImag     (beam 1, 369-fiber output)
    Imag399.Amp2 -> Amp399Imag2    (beam 2; on 3013 this beam carries most of the d-prime)

AOM knee (measured 2026-07-16, R212 _214049): amps **0.5-1.0 are optically FLAT**, only **<= 0.5
actually attenuates**. DDS amp -> optical power is NONLINEAR: sweep it, never compute a ratio.

Why not a plain ``g().Imag399.Amp1`` override? Precedence is base < ByPattern < scan ``g()``
(lib/expConfig_helper.py:79-94), so a g() override wins in EVERY bseq and would move all three
frames together. The per-frame knob is instead the pair of optional extras that
``YbSeqs/RearrangeCommSeq2.py`` reads and applies to one Imag399 step each:

    rearrange_kwargs.extras.MidImgAmp1 / MidImgAmp2   -> img2 (MIDDLE, kagome_res_2198)
    rearrange_kwargs.extras.FinImgAmp1 / FinImgAmp2   -> img3 (FINAL,  kagome_2078_camfb)

Absent -> the frame's own ByPattern value is used and the build is byte-identical to production.
img1 (LOADING) has NO such knob on purpose: it is the frame the PID is locked for. To move img1
you must move the shared held setpoint ``BlueMOT.Img1/Img2PIDSet``, which shifts ALL THREE frames
together -- available here as a deliberately SEPARATE, OFF-BY-DEFAULT coarse axis
(``YB_PIDSET_SWEEP``), because it is confounded with loading/img1 quality (already optimized
in-sequence to 0.5/0.5, scans id 2804-2807).

============================ WHAT DETECTION DOES WITH DIM FRAMES =======================
READ THIS BEFORE INTERPRETING THE RESULT. Detection thresholds are ABSOLUTE per-site intensity
cuts; NOTHING in the live path rescales them when we dim a frame. Per frame:

* img1 (LOADING, tri_3013_camfb) -- stays BRIGHT (never swept here). It is the only frame feeding
  the live img1 threshold accumulator (yb_analysis/acquisition/data_manager.py:2240-2264), so the
  loading thresholds are safe.

* img2 (MIDDLE, kagome_res_2198) -- thresholds are LOADED ONCE at scan start from
  ``yb_dashboard_state/patterns/kagome_res_2198/threshold.mat`` and are NEVER refit and NEVER
  accumulated (the mid branch, data_manager.py:2274-2279, appends to no accumulator). So the
  stored mid thresholds CANNOT be contaminated -- but mid detection at a dimmed amp IS BIASED
  LOW (counts scale down, the cut does not). That is not only an analysis artifact: pyctrl's
  rearrangement detector uses the SAME absolute per-site thresholds / Gaussian fits
  (YbExptCtrl/rearrange_runtime.py:502-538, :404-430) and the MIDDLE frame's bits/probs are the
  INPUT to rearrangement round 2. Dimming img2 therefore genuinely degrades round-2 targeting.
  ==> the mid axis is deliberately kept SHALLOW (>= 0.3) and is interpreted as the OPERATIONAL
  trade-off at today's thresholds; a committed dim mid amp would need its thresholds re-anchored
  (bright >= 60-shot run) before it is fairly valued.

* img3 (FINAL, kagome_2078_camfb) -- pure readout (its bits only feed ``update_rearrange`` +
  analysis), so it can be swept deep. BUT it is the ``is_last`` frame and its pattern name
  DIFFERS from frame 0's, so ``_img2_refit_active()`` is TRUE
  (data_manager.py:1997-2005): dim shots DO accumulate (``_pattern_accum['kagome_2078_camfb\\0img2']``)
  and after 200 accumulated shots a full refit fires (data_manager.py:1857-1918) and OVERWRITES
  ``yb_dashboard_state/patterns/kagome_2078_camfb/threshold.mat``. This run is ~600 shots, so it
  WILL fire. That is exactly yb_skills/memory/bug-threshold-dim-scan-contamination.md, and via the
  mtime-keyed detector cache (rearrange_runtime.py:601-607) a contaminated refit propagates into
  live bit-scoring and into every LATER scan.

  ==> MANDATORY BEFORE RUNNING (there is no freeze-detection flag; none exists):
        back up  "<PATH_PREFIX>\\yb_dashboard_state\\patterns\\kagome_2078_camfb\\threshold.mat"
        (PATH_PREFIX = $YB_PATH_PREFIX, default "D:\\OneDrive - Harvard University\\Documents - Yb")
      and restore it afterwards, then re-anchor with a bright (amps 1/1) >= 60-shot run.
      Back up kagome_res_2198's too, cheap insurance.

HOW TO JUDGE THE RESULT: **not** by raw ADU and **not** by the live dashboard survival curve.
  * Fidelity -> per-frame d-prime from the INTENSITY histograms, computed OFFLINE per scan point
    (``intensities_mid`` / ``intensities_img2`` are saved raw and are threshold-free). See
    yb_skills/memory/gotcha-detection-health-dprime-not-abs-adu.md.
  * Survival -> mid->final, NOT the img1-conditioned dashboard number
    (yb_skills/memory/gotcha-stirap-dashboard-img1-vs-mid-frame.md). Because the mid (2198) and
    final (2078) arrays have DIFFERENT site counts there is no per-site pairing, and
    ``analyze_scan(survival_ref='mid')`` SILENTLY FALLS BACK to img1 in exactly this case
    (yb_analysis/analysis/run_analysis.py:521-531). Compute it yourself as an array-level ratio:
        surv_round2(shot) = logicals_img2[shot].sum() / logicals_mid[shot].sum()
        final_fill(shot)  = logicals_img2[shot].sum() / 2078
  * Report BOTH an AS-RUN number (today's fixed thresholds -- the honest operational value, it
    includes the mis-detection penalty round 2 actually paid) and a RE-THRESHOLDED number
    (re-fit the two-Gaussian mixture from that point's OWN pooled intensity histogram, re-derive
    the logicals). The gap between them is the fixed-threshold bias. Re-thresholding repairs the
    READOUT only; it cannot undo a round 2 that was driven by mis-detected mid bits.
  * Normalize per shot before any of this (common-mode brightness wobble ~24% is the dominant
    fidelity limiter -- yb_skills/memory/open-imaging-common-mode-shot-wobble.md).

OFFLINE ANALYSIS RECIPE (everything needed is in the .h5; the dashboard is only a live sanity
check -- its survival number is img1-conditioned and will look wrong here):

    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals
    sc = load_scan_from_path(scan_dir)
    #   sc['intensities_img1'] / ['intensities_mid'] / ['intensities_img2']   (n_shots, n_sites)
    #   sc['logicals_img1']    / ['logicals_mid']    / ['logicals_img2']      as-run bits
    #   sc['seq_ids']  -> the scan point of each shot (use unpack_scan_logicals for the mapping)

    per scan point p (20 of them):
      as-run   : surv2 = mean_over_shots( logicals_img2.sum(1) / logicals_mid.sum(1) )
                 final_fill = mean( logicals_img2.sum(1) ) / 2078
                 mid_fill   = mean( logicals_mid.sum(1) )  / 2198
      d-prime  : per shot divide intensities by that shot's median occupied-site value
                 (common-mode norm), pool all sites x shots at point p, fit a 2-Gaussian
                 mixture -> d' = (mu_a - mu_e) / sqrt((s_a^2 + s_e^2) / 2). Report the pooled d'
                 (primary) and the median of per-site d' (secondary).
      re-thresh: re-derive the logicals from that point's OWN fitted crossing, then recompute
                 final_fill. Compare to as-run to expose the fixed-threshold bias.

    Reshape a 2-D scan COLUMN-MAJOR (axis 1 = mid, fastest): mid_i = p % 4, fin_j = p // 4 --
    row-major silently transposes the map (yb_skills/memory/gotcha-2d-scan-reshape-column-major).
    pyctrl Params are 1-INDEXED (gotcha-pyctrl-params-1-indexed-2d-map): offset by P.min().

DECIDE ON: max final_fill at fidelity (pooled d') no worse than the (1.0, 1.0) baseline within
its own 4-cell null spread. Commit the winner to expConfig
ByPattern[kagome_res_2198].Imag399.Amp1/Amp2 and ByPattern[kagome_2078_camfb].Imag399.Amp1/Amp2,
then RE-ANCHOR the thresholds with a bright >= 60-shot run at the new amps before trusting any
later survival number.

============================== SWEEP DESIGN (edit the constants) =======================
Axis 1 = MIDDLE frame (Amp1, Amp2) PAIRS; axis 2 = FINAL frame (Amp1, Amp2) PAIRS. Pairs (both
beams co-varying on one axis) rather than 4 independent beam axes, because 4 amps x 2 frames is
not affordable and the current config is 1:1 on both beams -- so a pair ladder is "scale today's
config", the one-dimensional question worth asking first. Ratio (Amp1 vs Amp2) is the STAGE-2
refinement, available env-gated below once the dose optimum is known.

Ladders are geometric (~sqrt(2) steps): AOM output and the resulting d-prime both respond
logarithmically near the optimum, so equal ratios carry equal information.

  MID  [1.0, 0.5, 0.4, 0.3]            -- shallow, see the detection caveat above
  FIN  [1.0, 0.5, 0.35, 0.25, 0.175]   -- deep; 0.175 is the 2026-07-16 pre-ND 2198 optimum
  => 4 x 5 = 20 points

Including BOTH 1.0 and 0.5 is deliberate and is NOT waste: per the AOM knee they are optically
IDENTICAL, so (a) the 2x2 block of {1.0,0.5} x {1.0,0.5} cells measures ONE physical condition four
times and hands us an in-scan null distribution for cell-to-cell differences -- exactly the
yardstick needed to call a ~1-2% survival difference real; and (b) if those four cells do NOT
agree, the knee assumption is wrong and we have falsified it for free. (1.0, 1.0) is today's
production baseline every other cell is measured against.

Scrambled (``Scramble = 1``) so slow drift is decorrelated from the grid.

Run it (backend must be up, SLM server reachable; back up threshold.mat first):
    cd pyctrl
    python YbScans/SLMRearrangeImagingOptScan.py            # 20 points x 30 shots = 600 shots
    python YbScans/SLMRearrangeImagingOptScan.py --reps 40  # more stats
    YB_IMGOPT_MODE=fin_ratio python YbScans/SLMRearrangeImagingOptScan.py   # stage-2 beam ratio
"""

import argparse
import json
import os
import sys


# =============================== SWEEP GRID (edit me) ================================= #
# (Amp1, Amp2) pairs. 1.0 = full held power = today's production value. Only <= 0.5 attenuates.
MID_AMP_PAIRS = [(1.0, 1.0), (0.5, 0.5), (0.4, 0.4), (0.3, 0.3)]
FIN_AMP_PAIRS = [(1.0, 1.0), (0.5, 0.5), (0.35, 0.35), (0.25, 0.25), (0.175, 0.175)]

# Passes over the whole grid = SHOTS PER POINT (Scramble reorders within a pass).
# 20 points x 30 = 600 shots; at ~3 s/shot ~= 30 min, at ~4.5 s/shot ~= 45 min. Budget allows
# 40-50 if the cells look too close to call (800-1000 shots, ~40-75 min).
SHOTS_PER_POINT = int(os.environ.get("YB_SHOTS", "30"))

# Pins used by the stage-2 ratio modes (the frame that is NOT being swept).
MID_PIN = (1.0, 1.0)
FIN_PIN = (1.0, 1.0)

# Stage-2 ratio ladders (per-beam, independent axes) for YB_IMGOPT_MODE=mid_ratio / fin_ratio.
RATIO_AMP1 = [1.0, 0.5, 0.35, 0.25]
RATIO_AMP2 = [1.0, 0.5, 0.35, 0.25]

# "grid" (default) | "mid_ratio" | "fin_ratio"
IMGOPT_MODE = os.environ.get("YB_IMGOPT_MODE", "grid")
# ====================================================================================== #

# ---- pattern selection: PINNED to the production 2-round chain ----------------------- #
INIT_PATTERN = os.environ.get("YB_INIT_PATTERN", "3013_tri")
MIDDLE_PATTERN = os.environ.get("YB_MIDDLE_PATTERN", "2198_kagome_res")
TARGET_PATTERN = os.environ.get("YB_TARGET_PATTERN", "2078_kagome")
N_ROUNDS = 2                      # this scan only makes sense for the 2-round (3-frame) chain

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"


def _pattern_cfg(name):
    """Pattern name -> {phase_path, baked_zernike, legacy}. Subset of SLMRearrangementScan's
    table -- only the three patterns this scan is about (plus the 33x33 fallbacks) so a typo in
    a pattern name cannot silently pick an unrelated array."""
    table = {
        "3013_tri": ("phase/tri_3013_camfb.pt", [0, 0, 0, 0, 0]),
        "2078_kagome": ("phase/kagome_2078_camfb.pt", [0, 0, 0, 0, 0]),
        "2198_kagome_res": ("phase/kagome_res_2198.pt", [0, 0, 0, 0, 0]),
        "2198_kagome_res_closer": ("phase/kagome_res_2198_closer.pt", [0, 0, 0, 0, 0]),
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _registry_name(cfg):
    """The DETECTION REGISTRY + expConfig ByPattern key = the phase-file BASENAME
    ('phase/tri_3013_camfb.pt' -> 'tri_3013_camfb'), NOT the table alias. Using the alias makes
    detector_for() miss the registry and silently fall back to the day-folder grid, and the
    ByPattern overlay never applies (yb_skills/memory/gotcha-pattern-names-are-phase-basenames)."""
    return os.path.splitext(os.path.basename(cfg["phase_path"].replace("\\", "/")))[0]


def _pattern_item(name, cfg):
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(init_cfg, middle_cfg, target_cfg):
    """Per-frame detection declaration [LOADING, MIDDLE, FINAL] -- each frame detected against its
    own registry grid + thresholds. Explicit -> beats the runner's 2-frame auto-synthesis."""
    return json.dumps([_pattern_item(_registry_name(init_cfg), init_cfg),
                       _pattern_item(_registry_name(middle_cfg), middle_cfg),
                       _pattern_item(_registry_name(target_cfg), target_cfg)])


def _pairs_env(var):
    """'a1:a2,a1:a2,...' -> [(a1, a2), ...]; None when unset."""
    raw = os.environ.get(var)
    if not raw:
        return None
    out = []
    for tok in raw.split(","):
        parts = tok.split(":")
        if len(parts) != 2:
            raise ValueError("%s: expected 'amp1:amp2' pairs, got %r" % (var, tok))
        out.append((float(parts[0]), float(parts[1])))
    return out


def _floats_env(var):
    raw = os.environ.get(var)
    return [float(x) for x in raw.split(",")] if raw else None


def _one_pair_env(var, default):
    p = _pairs_env(var)
    if p is None:
        return default
    if len(p) != 1:
        raise ValueError("%s: expected exactly one 'amp1:amp2' pair" % var)
    return p[0]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build():
    """Build (do NOT submit) the imaging-optimization ScanGroup. Returns ``(seq_name, g)``.
    Separate from the submit path so it can be exercised offline with no backend (the
    scan-verification convention); ``tools/verify_imaging_opt_scan.py`` does exactly that."""
    _bootstrap()
    from scan_group import ScanGroup

    init_cfg = _pattern_cfg(INIT_PATTERN)
    middle_cfg = _pattern_cfg(MIDDLE_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    seq_name = "RearrangeCommSeq2"        # the PRODUCTION seq (per-frame amps are opt-in there)

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds ----------------------- #
    g().rearrange_kwargs.extras.n_rounds = N_ROUNDS

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) ------------- #
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = init_cfg["phase_path"]
    rp.warmup_kwargs.final_phase = target_cfg["phase_path"]
    # middle_phase is MANDATORY for 2 rounds: without it the server scores every round against
    # the full INITIAL grid ("bits length 2198 != expected 3013", 2026-07-16).
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

    # ---- rearrangement params: PINNED to SLMRearrangementScan.py's production values ---- #
    # Nothing here is swept -- the whole point is to vary ONLY the imaging amps on an otherwise
    # production-identical 2-round shot. (Dev's nsteps=70 finding is deliberately NOT adopted:
    # match what production actually runs, so the winning amps transfer directly.)
    g().rearrange_kwargs.extras.prob_hungarian = True
    g().rearrange_kwargs.nsteps = int(os.environ.get("YB_NSTEPS", "50"))
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.ifEnhanced = True
    _DEFOCUS = float(os.environ.get("YB_DEFOCUS", "-4"))
    g().rearrange_kwargs.extras.z4 = _DEFOCUS      # MATCH rp.loading_defocus (same focal plane)

    # ---- per-bseq pattern tags (ByPattern overlay + per-frame detection grid) ---------- #
    g().rearrange_kwargs.extras.initial_pattern = _registry_name(init_cfg)
    g().rearrange_kwargs.extras.middle_pattern = _registry_name(middle_cfg)
    g().rearrange_kwargs.extras.final_pattern = _registry_name(target_cfg)

    # ================== THE SWEPT AXES: per-frame 399 imaging amps ===================== #
    # These extras are read by RearrangeCommSeq2 and applied to ONE Imag399 step each, so they
    # do NOT leak across frames the way a g().Imag399.Amp1 override would. Setting an amp to 1.0
    # reproduces the ByPattern default exactly (byte-identical), which is why the baseline cell
    # is a real member of the grid rather than a separate control run.
    mode = IMGOPT_MODE
    if mode == "grid":
        mid_pairs = _pairs_env("YB_MID_PAIRS") or MID_AMP_PAIRS
        fin_pairs = _pairs_env("YB_FIN_PAIRS") or FIN_AMP_PAIRS
        # axis 1 = MIDDLE dose ladder (both beams co-vary -> one axis, len(mid_pairs) points)
        g().rearrange_kwargs.extras.MidImgAmp1.scan(1, [p[0] for p in mid_pairs])
        g().rearrange_kwargs.extras.MidImgAmp2.scan(1, [p[1] for p in mid_pairs])
        # axis 2 = FINAL dose ladder -> outer product with axis 1
        g().rearrange_kwargs.extras.FinImgAmp1.scan(2, [p[0] for p in fin_pairs])
        g().rearrange_kwargs.extras.FinImgAmp2.scan(2, [p[1] for p in fin_pairs])
    elif mode == "mid_ratio":
        # STAGE 2: beam-1 vs beam-2 RATIO on the middle frame; final frame pinned.
        a1 = _floats_env("YB_RATIO_AMP1") or RATIO_AMP1
        a2 = _floats_env("YB_RATIO_AMP2") or RATIO_AMP2
        g().rearrange_kwargs.extras.MidImgAmp1.scan(1, a1)
        g().rearrange_kwargs.extras.MidImgAmp2.scan(2, a2)
        fp = _one_pair_env("YB_FIN_PIN", FIN_PIN)
        g().rearrange_kwargs.extras.FinImgAmp1 = fp[0]
        g().rearrange_kwargs.extras.FinImgAmp2 = fp[1]
    elif mode == "fin_ratio":
        # STAGE 2: beam-1 vs beam-2 RATIO on the final frame; middle frame pinned.
        a1 = _floats_env("YB_RATIO_AMP1") or RATIO_AMP1
        a2 = _floats_env("YB_RATIO_AMP2") or RATIO_AMP2
        g().rearrange_kwargs.extras.FinImgAmp1.scan(1, a1)
        g().rearrange_kwargs.extras.FinImgAmp2.scan(2, a2)
        mp = _one_pair_env("YB_MID_PIN", MID_PIN)
        g().rearrange_kwargs.extras.MidImgAmp1 = mp[0]
        g().rearrange_kwargs.extras.MidImgAmp2 = mp[1]
    else:
        raise ValueError("YB_IMGOPT_MODE must be grid | mid_ratio | fin_ratio, got %r" % mode)

    # ---- OPTIONAL coarse axis 3: the SHARED held PID setpoint -- OFF by default -------- #
    # CONFOUNDED ON PURPOSE-NOTICE: Img1/Img2PIDSet is read ONLY by the root BlueMOTStep, so it
    # sets the single held power of ALL THREE frames -- including img1 (loading), whose quality
    # drives round 1 and the whole shot. It was already optimized IN-SEQUENCE to 0.5/0.5 (scans
    # id 2804-2807, 2026-07-19: 0.3 underpowered d' 3.4; 0.5 -> d' 4.67 med; 0.7/0.9 plateau +
    # more img1 heating). Moving it here changes the baseline every amp cell is measured against,
    # so treat it as a SEPARATE coarse pass, not as a third factor to co-fit.
    #   YB_PIDSET_SWEEP="0.4,0.5,0.6"  -> axis 3, Img1 and Img2 co-swept (equal, as in production)
    #   YB_IMG1PID / YB_IMG2PID        -> scalar shift of the whole run (preferred: one value/run)
    _pid_sweep = _floats_env("YB_PIDSET_SWEEP")
    if _pid_sweep:
        g().BlueMOT.Img1PIDSet.scan(3, _pid_sweep)
        g().BlueMOT.Img2PIDSet.scan(3, _pid_sweep)
    else:
        if os.environ.get("YB_IMG1PID"):
            g().BlueMOT.Img1PIDSet = float(os.environ["YB_IMG1PID"])
        if os.environ.get("YB_IMG2PID"):
            g().BlueMOT.Img2PIDSet = float(os.environ["YB_IMG2PID"])

    # ---- run params (runp) ------------------------------------------------------------ #
    rp.NumPerGroup = 100000
    rp.loading_defocus = _DEFOCUS                # matched to rearrange z4
    rp.NumImages = 3                             # img1 + one frame per round
    rp.Scramble = 1                              # decorrelate drift from the grid -- keep at 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json(init_cfg, middle_cfg, target_cfg)

    return seq_name, g


def describe():
    """One-line-per-fact summary of what build() produced (points, axes, shot budget). Used by
    the CLI and by tools/verify_imaging_opt_scan.py."""
    seq_name, g = build()
    npts = g.nseq()
    shots = npts * SHOTS_PER_POINT
    lines = ["seq            : %s" % seq_name,
             "mode           : %s" % IMGOPT_MODE,
             "points         : %d  (dims %d)" % (npts, g.scandim(1)),
             "shots/point    : %d" % SHOTS_PER_POINT,
             "total shots    : %d" % shots,
             "est. duration  : %.0f-%.0f min (3.0-4.5 s/shot)"
             % (shots * 3.0 / 60.0, shots * 4.5 / 60.0)]
    for dim in range(1, g.scandim(1) + 1):
        try:
            params, size = g.get_vars(1, dim)
        except Exception:  # noqa: BLE001
            continue
        if not size:
            continue
        leaves = []

        def _walk(node, path):
            if isinstance(node, dict):
                for k in sorted(node):
                    _walk(node[k], path + [k])
            else:
                leaves.append((".".join(path), list(node)))

        _walk(params, [])
        lines.append("axis %d (%d pts) : %s"
                     % (dim, size, "; ".join("%s = %s" % (n, v) for n, v in leaves)))
    return "\n".join(lines)


def SLMRearrangeImagingOptScan(url=None, reps=None):
    """Build + SUBMIT to the running pyctrl backend over ZMQ. Returns the descriptor id."""
    seq_name, g = build()
    from yb_start_scan import ybStartScan
    npts = g.nseq()
    n_reps = SHOTS_PER_POINT if reps is None else reps
    desc = os.environ.get(
        "YB_SCAN_DESC",
        "2-round rearrange imaging opt: mid/final Imag399 DDS amps (PID held at root 399 lock)")
    did = ybStartScan(seq_name, g, url=url, label="SLMRearrangeImagingOptScan",
                      description=desc, rep=n_reps)
    print("submitted SLMRearrangeImagingOptScan (%s, %d points x %d shots = %d shots) "
          "-> descriptor id %s (url=%s)"
          % (IMGOPT_MODE, npts, n_reps, npts * n_reps, did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 2-round rearrangement imaging-amp optimization scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes = SHOTS PER POINT (default %d; 0 = forever)" % SHOTS_PER_POINT)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + print the scan shape; submit nothing")
    args = ap.parse_args()
    if args.dry_run:
        print(describe())
    else:
        SLMRearrangeImagingOptScan(url=args.url, reps=args.reps)
