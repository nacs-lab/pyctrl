"""WarmWGSKagomeStepSweep.py -- SINGLE-ROUND tri_3013_camfb -> kagome_2078_camfb rearrange2 with
WARM-STARTED PHASE-LOCKED WGS transit frames (server extras ``wgs_warm``, deployed 2026-07-27),
sweeping nsteps 10..70.

CAMPAIGN: median >= 99% fill of the 2078-site kagome array in ONE round from the 3013-site
triangular loading array. The single-round 3013 -> 2078 baseline (SLMnet MODEL transit frames) sits
at gated median fill 0.9812 (39 median holes) with a best shot of 0.9913; the campaign needs
<= 21 holes. Transit loss is the prime suspect, so step 1 of the campaign is this scan: replace the
model frames with warm-started WGS frames and find the best ``nsteps``.

WARM-WGS vs MODEL frames: the production path generates each transit frame with the SLMnet CNN
(``direct_flat``, amplitude-blind) -- fast but only approximately on-target, and its per-spot phase
contract is learned rather than exact. The warm-start WGS producer
(``rearrange2/warm_wgs.WarmWGSProducer``) instead SOLVES frame k with ``wgs_iters`` phase-locked WGS
iterations at ``wgs_pad`` FFT pad, SEEDED from frame k-1's SLM phase (hence "warm"): exact spot
positions, an exact per-spot phase contract, and faithful per-spot amplitude control. Cost is
~0.55-0.9 ms/frame at pad 2048 x 3 iters (1024 = no padding, ~0.25 ms/frame but ~-7% pattern power).
The user's requested operating point here is warm WGS 2048 x 3.

WHY nsteps IS THE STEP-SIZE AXIS: with linear scheduling (``dynamic=False``) every atom traverses
its own path length ``L_i`` in exactly ``nsteps`` equal increments, so each frame advances an atom by
``L_i / nsteps`` knm-px. Sweeping nsteps at fixed ``step_period_ms`` therefore sweeps the PER-FRAME
STEP SIZE (and the total transit time, ``nsteps * 0.696 ms``): small nsteps = big jumps = transit
loss; large nsteps = gentle jumps but a longer transit (heating / vacuum / SLM-frame budget). The
model-frame baseline on 33x33 plateaus around nsteps 60-80; warm-WGS frames should shift that knee.

ANALYSIS METRIC: per-nsteps-point GATED MEDIAN FILL of the 2078 kagome target (and the equivalent
median hole count = 2078 * (1 - fill)), where the gate is "initial load > 72% of 3013" = 2169 atoms,
i.e. >= 91 spare atoms over the 2078 targets. Ungated medians mix in shots that could not possibly
fill the target and understate the transit performance. Also worth reading off per point: best-shot
fill, the fill distribution width, and the mid/final survival.

Single-round (RearrangeCommSeq, 2 images): tri_3013_camfb -> kagome_2078_camfb. The FULL 2078-site
kagome IS the target, so ``extras.pattern`` is deliberately NOT set (that key is the
checkerboard/subset selector used by the 33x33 sweeps).

NOTE the scan name MUST contain 'rearrang': the Analysis tab's on-demand slm_diag sync
(run_analysis._maybe_sync_slm_diag) is name-gated, and without the diag the run is not recognized as
rearrangement (no per-shot targets / target-aware survival). The gate now also sniffs
rearrange_kwargs in the JSON sidecar, but keep the name convention anyway.

Run:
    cd pyctrl && python YbScans/RearrangeDiagnostics/WarmWGSKagomeStepSweep.py

Optional env:
    YB_NSTEPS="10,20,30"          override the nsteps sweep values (dim 1)
    YB_PH_BETA_SWEEP="1e3,5e3"    ALSO sweep prob_hungarian_beta on dim 2 (campaign step 2)
    YB_PH_BETA="5625"             pin prob_hungarian_beta to one scalar instead
    YB_INITAMP_SWEEP="0.5,0.35"   ALSO sweep the FRAME-0 (3013) 399 imaging DDS amps as a PAIR
                                  on dim 3 (campaign step 3 -- imaging survival, see below)
    YB_INITAMP="0.35"             pin the frame-0 amps to one scalar pair instead
    YB_FINAMP_SWEEP="1.0,0.5"     same for the FINAL (2078) frame, on dim 4 (for later use)
    YB_FINAMP="0.5"               pin the final-frame amps to one scalar pair instead
    YB_FINEXP_SWEEP="0.1,0.15,0.2,0.3"   ALSO sweep the FINAL (2078) frame's 399 PULSE length
                                  (seconds) on dim 5 (campaign step 4 -- the only brightness lever
                                  left on the metric-defining image; see below)
    YB_FINEXP="0.2"               pin the final-frame 399 pulse to one scalar instead
    YB_INITEXP_SWEEP="0.1,0.15"   same for the FRAME-0 (3013) 399 pulse, on dim 6
    YB_INITEXP="0.1"              pin the frame-0 399 pulse to one scalar instead
"""
import argparse
import json
import os
import sys

INIT_PATTERN = "tri_3013_camfb"            # REGISTRY name = phase-file basename
FINAL_PATTERN = "kagome_2078_camfb"        # REGISTRY name = phase-file basename
INIT_PHASE_PATH = "phase/tri_3013_camfb.pt"
FINAL_PHASE_PATH = "phase/kagome_2078_camfb.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# amplitude-control variant: required for mover_boost with MODEL frames (the plain
# direct_flat model is amplitude-blind; warm-WGS realizes amps natively either way)
MODEL_AMPCTRL = "SLMnet/checkpoints/sinc_3x3_experiment/models/ampctrl_flat/ampctrl_flat_best.pth"

# nsteps sweep (dim 1). Env YB_NSTEPS="10,20,..." overrides.
NSTEPS = list(range(10, 71, 10))           # 10, 20, ... 70 (7 points, dim 1)
if os.environ.get("YB_NSTEPS"):
    NSTEPS = [int(x) for x in os.environ["YB_NSTEPS"].split(",") if x.strip()]

WGS_PAD = 2048                             # warm-WGS FFT pad (full teacher efficiency)
WGS_ITERS = 3                              # WGS iterations/frame (teacher-grade contract)
DEFOCUS = -4                               # matched loading_defocus / rearrange z4
N_TARGETS = 2078                           # kagome_2078_camfb site count
N_LOADING = 3013                           # tri_3013_camfb site count
LOAD_GATE_FRAC = 0.72                      # analysis gate: initial load > 72% of 3013 = 2169 atoms

# ---- OPTIONAL prob-Hungarian weight axis (campaign step 2) --------------------------------
# SERVER CONVENTION CHANGE (2026-07-29, user): the probability term is now ``beta * nsteps *
# log(p_i)`` per LOADED row (Bayes-consistent: per-frame loss scales the log-likelihood tradeoff
# by nsteps), with SERVER DEFAULT beta = 2 (i.e. a 2*n*log(p) term). Under the OLD convention the
# term was ``beta_old * log(p)`` with default (nsteps*1.5)^2; the 07-28 campaign pinned
# beta_old = 12000 at nsteps 40, which maps to NEW beta = 12000/40 = **300**.
# CAUTION: the NEW default (beta=2 -> multiplier 2n = 80 at nsteps 40) sits BELOW the measured
# saturation knee (total multiplier ~1e3, scans 20260728_195710/200808) -- an unset beta now runs
# SUB-PLATEAU. Pass YB_PH_BETA=300 (or anything >= ~25 at nsteps 40) for production.
# The term only biases WHICH loaded atoms get used -> effect ONLY under atom surplus
# (loaded > targets), this scan's regime (~2170-2470 loaded vs 2078 targets).
PH_BETA_DEFAULT = "300"   # new-convention production value (== old 12000 at nsteps 40)
PH_BETA_SWEEP = [float(x) for x in os.environ.get("YB_PH_BETA_SWEEP", "").split(",") if x.strip()]
PH_BETA = os.environ.get("YB_PH_BETA", PH_BETA_DEFAULT).strip()   # default 300 (new convention)

# ---- OPTIONAL per-frame 399 imaging-amp axes (campaign step 3) -----------------------------
# nsteps and prob_hungarian_beta are both SATURATED (fill ceiling 0.983 +- 0.003), so the remaining
# lever is the IMAGING SURVIVAL of the FIRST (3013) image: the 399 imaging pulse heats atoms, so a
# DIMMER frame 0 should hand rearrangement more surviving atoms. The cost is frame-0 detection
# confidence -- which a high prob_hungarian_beta absorbs by routing around low-p sites. So we sweep
# frame 0's brightness DOWNWARD.
#
# IMAGING-POWER POLICY (do NOT work around it; see the block in SLMRearrangementScan.py and
# yb_skills/memory/gotcha-imaging-pid-held-multiround-rearrange.md): the 399 imaging-power PID locks
# ONCE, in the ROOT BlueMOTStep, at the INITIAL pattern's BlueMOT.Img1/Img2PIDSet (PINNED at
# 0.5/0.5) and then HOLDS for the whole shot. Both frames are therefore taken at that ONE held
# optical power, and the setpoints stay pinned here on purpose -- moving one would also move the
# FINAL image's power and force a re-optimization of the 2078 array. Per-frame brightness comes
# ONLY from the DDS amps.
#
# A plain ``g().Imag399.Amp1`` would apply to BOTH bseqs (precedence base < ByPattern < scan g(),
# lib/expConfig_helper.py:79-94), so the frame-0-only knob is the pair of optional extras
# RearrangeCommSeq reads and applies to one Imag399 step each:
#     extras.InitImgAmp1 / InitImgAmp2  -> img1 (tri_3013_camfb) only
#     extras.FinImgAmp1  / FinImgAmp2   -> img2 (kagome_2078_camfb) only  (same names the
#                                          2-round RearrangeCommSeq2 uses for ITS final image)
# Unset -> NO key emitted, the frame's own ByPattern Amp1/Amp2 applies and the build is
# BYTE-IDENTICAL to the pre-knob scan.
#
# AOM knee (measured 2026-07-16, R212): amps 0.5-1.0 are optically FLAT, only <= 0.5 actually
# attenuates -- so a useful ladder lives at <= 0.5 (e.g. 0.5,0.35,0.25,0.175,0.12, ~sqrt(2) steps;
# 0.5 doubles as the optically-full-power baseline cell). DDS amp -> optical power is NONLINEAR:
# sweep it, never compute a ratio.
#
# Both beams co-vary as a PAIR on one dim (Amp1 and Amp2 declared on the SAME dim = one axis, the
# "scale today's 1:1 config" question). Frame 0 -> dim 3, final frame -> dim 4, so they compose
# with nsteps (dim 1) and beta (dim 2) independently; an unused dim stays a size-0 DUMMY and
# multiplies nothing into nseq() (ScanGroup.scansize skips size-0 vars), so dim 3 is safe even
# when dim 2 is unset -- no renumbering needed.
INITAMP_SWEEP = [float(x) for x in os.environ.get("YB_INITAMP_SWEEP", "").split(",") if x.strip()]
INITAMP = os.environ.get("YB_INITAMP", "").strip()
FINAMP_SWEEP = [float(x) for x in os.environ.get("YB_FINAMP_SWEEP", "").split(",") if x.strip()]
FINAMP = os.environ.get("YB_FINAMP", "").strip()

# ---- OPTIONAL per-frame 399 PULSE-LENGTH axes (campaign step 4) -----------------------------
# The metric is the FINAL (2078) image's occupancy, and that image is already at MAXIMUM available
# 399 power: its DDS amps are 1/1 and the AOM knee makes 0.5-1.0 optically FLAT, while the two 399
# beam-power PID setpoints (BlueMOT.Img1/Img2PIDSet) lock ONCE in the root BlueMOTStep and are HELD
# -- and Imag399Step fires BOTH beams in EVERY image, so raising a setpoint would also brighten
# frame 0 and heat it. Pinned by policy. That leaves TIME as the only lever on the final image:
#
#     extras.FinImgExposure   -> img2 (kagome_2078_camfb) 399 pulse length, SECONDS  (dim 5)
#     extras.InitImgExposure  -> img1 (tri_3013_camfb) 399 pulse length, SECONDS     (dim 6)
#
# A longer 399 pulse deposits more photons per site -> larger histogram separation / per-site d' ->
# fewer false-EMPTY mis-reads, which is exactly the failure mode the ~23% common-mode shot-to-shot
# imaging brightness wobble (yb_skills/memory/open-imaging-common-mode-shot-wobble.md) produces on a
# fill metric pinned at ~0.983 while the best single shots reach 0.993. The extra 399 HEATING on the
# final image is HARMLESS: no atoms are needed after it (frame 0 is the opposite -- its atoms must
# survive into rearrangement, so YB_INITEXP* exists for completeness, not for the campaign).
#
# PRODUCTION DEFAULT (expConfig.py): Imag399.ExposureTime is None in the base config and is
# CROSS-REFERENCED to Orca.ExposureTime (lib/expConfig_helper.apply_cross_refs, re-resolved after
# each per-pattern overlay). Base Orca.ExposureTime = 0.050004 s, but BOTH production patterns
# override it: ByPattern["tri_3013_camfb"].Orca.ExposureTime = 0.1 and
# ByPattern["kagome_2078_camfb"].Orca.ExposureTime = 0.1 -- so the effective default 399 pulse is
# 100 ms on BOTH frames. Neither pattern sets Imag399.ExposureTime directly.
#
# CAMERA vs 399 PULSE -- read this before picking values. The camera exposure is a SINGLE GLOBAL
# hardware setting that the runner syncs ONCE per scan from the resolved Orca.ExposureTime
# (YbExptCtrl/camera_runtime.sync_camera_exposure); it is NOT per-frame and these extras do NOT
# move it. So this sweep changes the 399 PULSE length inside a FIXED ~100 ms collection window:
# values <= 0.1 trade pulse length against the window, and values > 0.1 s put the pulse tail
# OUTSIDE the frame -- extra heating and shot time for zero extra photons. Expect the fill metric
# to be flat above ~0.1 s; to go beyond 100 ms of collected light you must raise
# ByPattern[<pattern>].Orca.ExposureTime (a config change, out of scope for this env hook).
FINEXP_SWEEP = [float(x) for x in os.environ.get("YB_FINEXP_SWEEP", "").split(",") if x.strip()]
FINEXP = os.environ.get("YB_FINEXP", "").strip()
INITEXP_SWEEP = [float(x) for x in os.environ.get("YB_INITEXP_SWEEP", "").split(",") if x.strip()]
INITEXP = os.environ.get("YB_INITEXP", "").strip()


def _run_desc(wgs_warm, mover_boost=0.0, ampctrl=False):
    if wgs_warm:
        frames = ("warm-start WGS transit frames (wgs_warm, pad %d x %d iters, gaussian beam)"
                  % (WGS_PAD, WGS_ITERS))
    else:
        frames = ("SLMnet %s MODEL transit frames (wgs_warm off)"
                  % ("ampctrl_flat" if ampctrl else "direct_flat"))
    if mover_boost:
        frames += ", mover_boost=%g" % mover_boost
    desc = (frames + ", SINGLE-ROUND rearrange2 %s (%d sites) -> %s (%d sites), linear "
            "scheduling; nsteps sweep %s for fill vs per-frame step size. "
            % (INIT_PATTERN, N_LOADING, FINAL_PATTERN, N_TARGETS,
               ",".join(str(n) for n in NSTEPS)))
    desc += ("CAMPAIGN GOAL: median >= 99%% fill of the %d-site kagome array. The single-round "
             "%d -> %d baseline was gated (initial load > %d%% of %d) median fill 0.9812 "
             "(%d median holes), best shot 0.9913; the goal needs <= %d holes. This sweep tests "
             "whether warm-WGS %dx%d transit frames cut the transit loss, and at which nsteps. "
             % (N_TARGETS, N_LOADING, N_TARGETS, int(LOAD_GATE_FRAC * 100), N_LOADING,
                39, 21, WGS_PAD, WGS_ITERS))
    desc += ("ANALYSIS GATE: keep only shots with initial load > %d%% of %d = %d atoms "
             "(>= %d spare atoms over the %d targets); report the gated MEDIAN fill + median hole "
             "count per nsteps point."
             % (int(LOAD_GATE_FRAC * 100), N_LOADING, int(LOAD_GATE_FRAC * N_LOADING),
                int(LOAD_GATE_FRAC * N_LOADING) - N_TARGETS, N_TARGETS))
    if PH_BETA_SWEEP:
        desc += (" ALSO sweeping prob_hungarian_beta on dim 2: %s (NEW convention 2026-07-29: "
                 "term = beta*nsteps*log(p), server default beta=2 which is SUB-plateau)."
                 % ",".join("%g" % b for b in PH_BETA_SWEEP))
    elif PH_BETA:
        desc += (" prob_hungarian_beta pinned to %s (NEW convention: beta*nsteps*log(p); 300 == "
                 "old-convention 12000 at nsteps 40)." % PH_BETA)
    if INITAMP_SWEEP:
        desc += (" ALSO sweeping the FRAME-0 (%s) 399 imaging DDS amps Amp1=Amp2 on dim 3: %s "
                 "(imaging-survival axis; PID setpoints PINNED at 0.5/0.5 -- the root 399 lock is "
                 "held for the whole shot; AOM knee: only amps <= 0.5 attenuate)."
                 % (INIT_PATTERN, ",".join("%g" % a for a in INITAMP_SWEEP)))
    elif INITAMP:
        desc += (" Frame-0 (%s) 399 imaging DDS amps pinned to %s/%s (PID setpoints untouched)."
                 % (INIT_PATTERN, INITAMP, INITAMP))
    if FINAMP_SWEEP:
        desc += (" ALSO sweeping the FINAL (%s) 399 imaging DDS amps Amp1=Amp2 on dim 4: %s."
                 % (FINAL_PATTERN, ",".join("%g" % a for a in FINAMP_SWEEP)))
    elif FINAMP:
        desc += (" Final (%s) 399 imaging DDS amps pinned to %s/%s."
                 % (FINAL_PATTERN, FINAMP, FINAMP))
    if FINEXP_SWEEP:
        desc += (" ALSO sweeping the FINAL (%s) 399 PULSE LENGTH on dim 5: %s s (default 0.1 s "
                 "= ByPattern Orca.ExposureTime; the final image is already at MAX 399 power -- "
                 "amps 1/1, PID setpoints pinned -- so TIME is the last brightness lever, and its "
                 "extra heating is free because no atoms are needed after this image. The CAMERA "
                 "window stays at the globally-synced 0.1 s, so > 0.1 s buys no extra photons)."
                 % (FINAL_PATTERN, ",".join("%g" % t for t in FINEXP_SWEEP)))
    elif FINEXP:
        desc += (" Final (%s) 399 pulse length pinned to %s s (camera window unchanged at the "
                 "globally-synced Orca.ExposureTime)." % (FINAL_PATTERN, FINEXP))
    if INITEXP_SWEEP:
        desc += (" ALSO sweeping the FRAME-0 (%s) 399 PULSE LENGTH on dim 6: %s s (default 0.1 s; "
                 "NOTE longer here = more heating of atoms that must still survive into "
                 "rearrangement)." % (INIT_PATTERN, ",".join("%g" % t for t in INITEXP_SWEEP)))
    elif INITEXP:
        desc += (" Frame-0 (%s) 399 pulse length pinned to %s s."
                 % (INIT_PATTERN, INITEXP))
    return desc


def _axis(parent, name, dim, values):
    """Declare ``parent.<name>`` as a sweep over ``values`` on ``dim`` -- but assign a SCALAR when
    there is only ONE value.

    Why the special case: ``.scan(dim, [x])`` decays to a size-0 (dummy) dimension whose parameter
    keeps its 1-ELEMENT LIST wrapper, so the seq / server sees ``[x]`` instead of ``x``. For a
    server kwarg (nsteps) that ships a list where a number is expected; for a seq-side extra it is
    WORSE THAN THAT -- ``RearrangeCommSeq._extras_num`` does ``float(...)`` inside a bare
    ``except``, so ``[0.35]`` silently degrades to "no override" and the point would run at FULL
    brightness while the label claims otherwise. One value in, one value out."""
    if len(values) == 1:
        setattr(parent, name, values[0])
    else:
        getattr(parent, name).scan(dim, values)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build(wgs_warm=True, mover_boost=0.0, ampctrl=False):
    """Build (do NOT submit) the ScanGroup -- offline-exercisable, scan-verification style.
    ``wgs_warm=False`` = A/B control: identical scan but transit frames from the SLMnet
    model (direct_flat, or ampctrl_flat when ``ampctrl``) instead of warm-started WGS.
    ``mover_boost`` = per-spot amplitude boost on the movers (server extras)."""
    _bootstrap()
    from scan_group import ScanGroup

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (identical to SLMRearrangementScan) ----
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_AMPCTRL if ampctrl else MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = INIT_PHASE_PATH
    rp.warmup_kwargs.final_phase = FINAL_PHASE_PATH
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

    # ---- rearrange2 + warm-start WGS producer, nsteps sweep ----
    g().rearrange_kwargs.protocol = "rearrange2"
    _axis(g().rearrange_kwargs, "nsteps", 1, NSTEPS)       # dim 1: the step-size axis
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.extras.wgs_warm = bool(wgs_warm)  # warm-started phase-locked WGS frames
    g().rearrange_kwargs.extras.wgs_pad = WGS_PAD          # inert when wgs_warm=False
    g().rearrange_kwargs.extras.wgs_iters = WGS_ITERS
    # NOTE: extras.pattern is deliberately NOT set -- the FULL 2078-site kagome is the target
    # (extras.pattern is the 33x33 checkerboard/subset selector, meaningless here).
    g().rearrange_kwargs.extras.mover_boost = float(mover_boost)   # 0.0 = off (server default)
    g().rearrange_kwargs.extras.prob_hungarian = True
    # Prob-Hungarian weight -- see the PH_BETA block at the top. NEW server convention
    # (2026-07-29): term = beta*nsteps*log(p), default beta=2 (SUB-plateau) -> this scan now
    # ALWAYS emits the key, defaulting to beta=300 (== old 12000 at nsteps 40).
    if PH_BETA_SWEEP:
        _axis(g().rearrange_kwargs.extras, "prob_hungarian_beta", 2, PH_BETA_SWEEP)   # dim 2
    elif PH_BETA:
        g().rearrange_kwargs.extras.prob_hungarian_beta = float(PH_BETA)
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False            # linear = validated operating point
    g().rearrange_kwargs.extras.ifEnhanced = True
    g().rearrange_kwargs.extras.z4 = DEFOCUS               # MATCH loading_defocus
    # REGISTRY names (phase-file basenames -- see SLMRearrangementScan._registry_name), NOT the
    # _pattern_cfg table aliases: ByPattern + per-frame detection both key off these.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = FINAL_PATTERN
    # ---- OPTIONAL per-frame 399 imaging-amp axes -- see the INITAMP block at the top -------
    # Emit the keys ONLY when asked; unset -> no key at all -> byte-identical default build.
    # PID setpoints (BlueMOT.Img1/Img2PIDSet) are deliberately NOT touched here (policy).
    if INITAMP_SWEEP:
        # Amp1 and Amp2 on the SAME dim = ONE axis (the beams co-vary as a PAIR).
        _axis(g().rearrange_kwargs.extras, "InitImgAmp1", 3, INITAMP_SWEEP)   # dim 3
        _axis(g().rearrange_kwargs.extras, "InitImgAmp2", 3, INITAMP_SWEEP)
    elif INITAMP:
        g().rearrange_kwargs.extras.InitImgAmp1 = float(INITAMP)
        g().rearrange_kwargs.extras.InitImgAmp2 = float(INITAMP)
    if FINAMP_SWEEP:
        _axis(g().rearrange_kwargs.extras, "FinImgAmp1", 4, FINAMP_SWEEP)     # dim 4
        _axis(g().rearrange_kwargs.extras, "FinImgAmp2", 4, FINAMP_SWEEP)
    elif FINAMP:
        g().rearrange_kwargs.extras.FinImgAmp1 = float(FINAMP)
        g().rearrange_kwargs.extras.FinImgAmp2 = float(FINAMP)
    # ---- OPTIONAL per-frame 399 PULSE-LENGTH axes -- see the FINEXP block at the top ------
    # Same "emit the key ONLY when asked" rule: unset -> no key -> byte-identical default build.
    # One value per frame, so no pairing here (unlike the amps); dim 5 = final, dim 6 = frame 0
    # (dims 1-4 are nsteps / beta / init amps / fin amps; an unused dim stays a size-0 dummy and
    # multiplies nothing into nseq(), so these are safe even when 2-4 are unset).
    if FINEXP_SWEEP:
        _axis(g().rearrange_kwargs.extras, "FinImgExposure", 5, FINEXP_SWEEP)    # dim 5
    elif FINEXP:
        g().rearrange_kwargs.extras.FinImgExposure = float(FINEXP)
    if INITEXP_SWEEP:
        _axis(g().rearrange_kwargs.extras, "InitImgExposure", 6, INITEXP_SWEEP)  # dim 6
    elif INITEXP:
        g().rearrange_kwargs.extras.InitImgExposure = float(INITEXP)

    # ---- OPTIONAL campaign-context overrides (2026-07-29). Emit ONLY when set, so an unset
    # env keeps the build byte-identical. YB_IMG1PID/YB_IMG2PID pin the ROOT BlueMOTStep 399
    # imaging-power setpoints (the campaign runs 0.95/0.95; WITHOUT these the scan runs at the
    # ByPattern default 0.5/0.5 and every dose/amp number lands on a DIFFERENT optical scale --
    # that mismatch was misread as drift on 07-29, job 312 vs 311). YB_BLUELAC_DET (MHz) pins
    # the enhanced-loading blue-LAC detuning (sharp axis; re-tuned -3.0 -> -2.4 on 07-29).
    if os.environ.get("YB_IMG1PID"):
        g().BlueMOT.Img1PIDSet = float(os.environ["YB_IMG1PID"])
    if os.environ.get("YB_IMG2PID"):
        g().BlueMOT.Img2PIDSet = float(os.environ["YB_IMG2PID"])
    if os.environ.get("YB_BLUELAC_DET"):
        g().LAC.BlueLAC.FreqDetuning = float(os.environ["YB_BLUELAC_DET"]) * 1e6

    g().rearrange_kwargs.extras.description = _run_desc(wgs_warm, mover_boost, ampctrl)

    # ---- run params (runp; loading/cooling stay at expConfig defaults) ----
    rp.NumPerGroup = 100000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = json.dumps([
        {"name": INIT_PATTERN, "base_phase_path": INIT_PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
        {"name": FINAL_PATTERN, "base_phase_path": FINAL_PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
    ])
    return "RearrangeCommSeq", g


def WarmWGSKagomeStepSweep(url=None, reps=None, wgs_warm=True, pad=None, iters=None,
                           boost=0.0, ampctrl=False):
    global WGS_PAD, WGS_ITERS
    if pad:
        WGS_PAD = int(pad)
    if iters:
        WGS_ITERS = int(iters)
    seq_name, g = build(wgs_warm=wgs_warm, mover_boost=boost, ampctrl=ampctrl)
    from yb_start_scan import ybStartScan
    # Label must keep 'rearrang' (the Analysis-tab slm_diag name gate).
    if wgs_warm:
        base = "WarmWGS%dx%d" % (WGS_PAD, WGS_ITERS)
    else:
        base = "Ampctrl" if ampctrl else "ModelFrames"
    label = base + ("Boost%g" % boost if boost else "") + "KagomeRearrangeStepSweep"
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label=label, **opts)
    print("submitted %s -> id %s (%d points, nsteps=%s, %s%s%s%s%s%s%s)"
          % (label, did, g.nseq(), NSTEPS,
             "wgs %dx%d" % (WGS_PAD, WGS_ITERS) if wgs_warm
             else ("ampctrl model" if ampctrl else "direct_flat model"),
             ", boost %g" % boost if boost else "",
             ", ph_beta %s" % (PH_BETA_SWEEP or PH_BETA) if (PH_BETA_SWEEP or PH_BETA) else "",
             ", init_amp %s" % (INITAMP_SWEEP or INITAMP) if (INITAMP_SWEEP or INITAMP) else "",
             ", fin_amp %s" % (FINAMP_SWEEP or FINAMP) if (FINAMP_SWEEP or FINAMP) else "",
             ", fin_exp %s" % (FINEXP_SWEEP or FINEXP) if (FINEXP_SWEEP or FINEXP) else "",
             ", init_exp %s" % (INITEXP_SWEEP or INITEXP) if (INITEXP_SWEEP or INITEXP) else ""))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--model", action="store_true",
                    help="A/B control: SLMnet model frames (wgs_warm off)")
    ap.add_argument("--ampctrl", action="store_true",
                    help="with --model: use the ampctrl_flat checkpoint (amp control)")
    ap.add_argument("--pad", type=int, default=None, help="override WGS_PAD")
    ap.add_argument("--iters", type=int, default=None, help="override WGS_ITERS")
    ap.add_argument("--boost", type=float, default=0.0,
                    help="mover_boost (per-spot mover amplitude boost; 0 = off)")
    args = ap.parse_args()
    WarmWGSKagomeStepSweep(url=args.url, reps=args.reps,
                           wgs_warm=not (args.model or args.ampctrl),
                           pad=args.pad, iters=args.iters,
                           boost=args.boost, ampctrl=args.ampctrl)
