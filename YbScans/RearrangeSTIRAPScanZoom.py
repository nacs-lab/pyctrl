"""RearrangeSTIRAPScan.py -- STIRAP push-out survival on a REARRANGED array.

Builds the hybrid ScanGroup (seq = ``RearrangeSTIRAPSeq``): the SLM-rearrangement prologue of
SLMRearrangementScan (load LOADING pattern -> img1 -> rearrange to TARGET) followed by
STIRAPAWGScan's science block (Siglent-AWG two-photon 556+308 STIRAP push-out -> survival
image). Submits the descriptor to the RUNNING pyctrl backend over ZMQ.

Frame layout -- the VERIFY_IMAGE toggle (single source of truth; NumImages +
imagePatternsJson + the seq's bseq structure all derive from it):
  * VERIFY_IMAGE = True  -> 3 frames: img1 load / img2 verify (post-rearrange, feeds
    update_rearrange) / img3 survival (post-pushout). img1->img2 = rearrangement fidelity,
    img2->img3 = clean science survival against the VERIFIED occupancy.
  * VERIFY_IMAGE = False -> 2 frames: img1 load / img2 survival. Shorter shot; an empty img2
    site conflates "move failed" with "pushed out", and the SLM server gets no
    update_rearrange result frame.

Patterns: LOADING (dense load) + TARGET (science array). Reuse one pattern for both (fill a
subset of the same tweezer grid) or set genuinely different arrays -- the rearrangement MODEL
(warmup_kwargs.model_filename) must match the family. The runner writes the LOADING phase at
scan start (scan-long slm lock); the TARGET pattern is produced by the rearrange() call
(ASSUME-WRITTEN, no phase re-write). Do NOT set runp().loading_phase here -- it would win the
loading-pattern priority over warmup_kwargs and mis-declare detection.

Prereqs: pyctrl backend live at --url; SLM server reachable (scan-long slm lock is
mandatory); both patterns present in the SLM server's pattern registry with per-pattern
thresholds (lab side) for detection.

Run it:
    cd pyctrl
    python YbScans/RearrangeSTIRAPScan.py                # rep=10 like STIRAPAWGScan
    python YbScans/RearrangeSTIRAPScan.py --reps 3
    python YbScans/RearrangeSTIRAPScan.py --url tcp://127.0.0.1:1408
"""

import argparse
import json
import numpy as np


# --------------------------- EDIT ME: layout + patterns ----------------------------- #
# img2 verify frame on/off (see module docstring). Per-scan constant -- NOT sweepable (it
# changes the seq structure + frame count).
VERIFY_IMAGE = True

# LOADING (dense load) / TARGET (science array) SLM patterns.
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

# Passes over the sweep. Single source of truth: build() uses it for NumPerGroup
# (= REPS * n_combos) and it is the default for both the API and the CLI, so the planned shot
# count and the submitted rep count can never disagree.
# 2026-09-15 round 2: 8 (was 10) so the WIDENED 117-combo grid costs the same wall-clock as
# round 1's 91-combo grid. This round is looking for the TREND, not a precise argmin.
REPS = 5   # ZOOM: 5 passes over the 25-point grid = 125 shots (~5 shots/pt)

# ---- ZOOM WINDOW (overridden by the CLI; see __main__) ----------------------------
# Defaults reproduce the 09-15 round-3 run (data_20260915_142454). Pass --carrier-center /
# --eom-center to move the box onto a newly fitted optimum without editing this file.
ZOOM = {
    "carrier_center": 96.95,   # MHz, AWG.AWG556.Ch1.carrier_freq_MHz
    "carrier_half": 0.10,      # MHz half-span
    "carrier_step": 0.05,      # MHz
    "eom_center": 392.60,      # MHz, Init.EOM616.Freq (written as e6)
    "eom_half": 0.15,          # MHz half-span
    "eom_step": 0.075,         # MHz
    "note": "",                # free text appended to the run description
}


def _axis(center, half, step):
    """Symmetric inclusive axis about `center`, rounded to 4 dp (the file's idiom)."""
    return [round(center + d, 4) for d in np.arange(-half, half + step / 2.0, step)]
# ------------------------------------------------------------------------------------ #


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
        "17x17_20um":      ("phase/17x17_20um.pt",      [0, 0, 0, 0, 0]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    """One imagePatternsJson entry (per camera frame): name + base phase + baked Zernike to
    strip. ``order='col'`` matches the runner's synthesized default + the server sweep_order
    the detection grid is derived in."""
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(verify, init_cfg, target_cfg):
    """Per-frame detection declaration. With verify: [LOADING, TARGET, TARGET] (img2 verify +
    img3 survival both detect on the TARGET grid). Without: [LOADING, TARGET]. Explicit ->
    wins over the runner's 2-frame auto-synthesis."""
    items = [_pattern_item(INIT_PATTERN, init_cfg), _pattern_item(TARGET_PATTERN, target_cfg)]
    if verify:
        items.append(_pattern_item(TARGET_PATTERN, target_cfg))
    return json.dumps(items)


import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

# The seq this scan runs. Passing the CALLABLE (not the name string) to ybStartScan gives
# go-to-definition in the editor and catches a typo at load time; the descriptor still
# serializes just its __name__, and the backend re-imports it fresh per job as before.
from RearrangeSTIRAPSeq import RearrangeSTIRAPSeq


def build():
    """Build (do NOT submit) the ScanGroup. Kept separate so it can be exercised offline
    WITHOUT touching the live backend (the scan-verification convention)."""
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    verify = bool(VERIFY_IMAGE)
    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)

    g = ScanGroup()

    # ---- frame layout (read by RearrangeSTIRAPSeq at build; per-scan constant) ----------
    g().rearrange_kwargs.extras.verifyImage = verify
    g().rearrange_kwargs.extras.n_rounds = 1        # one rearrangement round (runner ctx)

    # ---- Siglent AWG config (out-of-band; AWGManager reads g().AWG.<name>.*) ------------
    # 2026-08-27 every_other 60 G locks: forward freq = along-ridge co-vary argmin
    # (data_20260827_164918; ridge carrier = 119.1565 + 0.926*(EOM616 - 230.55) from freq-2D
    # data_20260827_162016), survival 0.0836 +- 0.0046. Reverse 556 = data_20260827_170604.
    # dimer_40um forward freq LOCK: refine argmin (data_20260828_052350), survival
    # 0.188 +- 0.049 -- ANOMALOUSLY high floor (~0.19-0.34) vs 0.048 at quadruple_spacing
    # (same 40 um NN); flagged in Notion, cause unresolved (edge illumination / drift?).
    g().AWG.AWG556.Ch1.shape = "rise_quintic"
    # dimer_20um REDO forward freq LOCK: co-vary argmin (data_20260828_101429),
    # survival 0.0445 +- 0.0074; ridge from freq-2D data_20260828_095538.
    # dimer_40um REDO forward lock: co-vary argmin (data_20260828_124812), 0.0709 +- 0.0139;
    # broad floor (6 pts within 1 SEM), ridge from freq-2D data_20260828_121858.
    CARRIER_LIST_MHZ = None  # locked below
    # 2026-09-10: Rydberg state switched 71 3S1 -> 66 3S1, so EVERY pre-09-10 frequency
    # lock above is void. Both 556 carriers reset to today's MEASURED 60 G 556 resonance
    # 119.0363 (scan 20260910174224, FWHM 127 kHz, R^2 0.994, push amp 0.08). The ridge fit
    # carrier = 119.1565 + 0.926*(EOM616 - 230.55) is a LOCAL 71 3S1 linearization -- do NOT
    # extrapolate it across the state change. Two-photon optimum still needs a fresh freq-2D.
    # was: Ch1 118.1133 / Ch2 118.5326 (71 3S1, 08-28 dimer co-vary argmins).
    # 2026-09-14 mj=+1 AT 60 G -- EVERY PRE-09-14 FREQUENCY LOCK IS VOID. The high-field
    # Rydberg 556 AOM (AOM3) was moved from the -1st to the +1st order, so the intermediate
    # state is now 3P1 mj=+1 and every 556/616 frequency shifts wholesale:
    #   556 single-photon line  119.0363 -> 97.3253 MHz  (delta -21.711; scan 20260914_154800,
    #                            FWHM 121.7 kHz, R^2 0.975, reproduced 97.3225 on 20260914_153831)
    #   616 revival (308 res.)  343.839  -> 392.8326 MHz (scan 20260914205331, FWHM 4.91 MHz,
    #                            R^2 0.991; was 392.975 at 16:46 -- a -142 kHz walk on a 4.9 MHz line)
    # SEEDS for the mj=+1 freq-2D, by applying the measured single-photon shifts to the
    # 09-10 every_other locks (carrier 118.9863 / EOM 344.5390):
    #   carrier = 118.9863 - 21.711 = 97.275 MHz
    #   EOM616  = 392.8326 + 0.700  = 393.533 MHz  (mj=-1's two-photon lock sat +0.70 ABOVE
    #                                               its revival -- do NOT assume lock = revival)
    # Pulse widths + delay are NOT re-derived here: they set the adiabatic pulse area, not the
    # resonance, so the every_other locks (pw 3/3 us, delay 1.35 us) carry across the state change
    # and are pinned so this scan measures frequency ONLY.
    # 2026-09-15: the 09-14 SEEDS ABOVE ARE SUPERSEDED -- they were EXTRAPOLATED (09-10 lock +
    # the measured single-photon shift); today both legs were measured directly at this exact
    # operating point (3P1 mj=+1, 60 G, 66 3S1), AFTER the morning's three freq-2D attempts:
    #   556 bare Rydberg line   97.3237 MHz (scan 20260915_123732, FWHM 158.2 kHz, R^2 0.979)
    #   556 Autler-Townes       dips 96.8961 / 97.7954 -> MIDPOINT 97.3457 MHz, splitting
    #                           0.90 +- 0.02 MHz => Omega_308/2pi = 0.90 MHz at Ryd308.Amp 0.4
    #                           (scan 20260915_125339; wide +-3 MHz confirm 20260915_125039)
    #   with EOM616 PARKED AT   392.948 MHz -- the AT midpoint sits only +22 kHz from the bare
    #                           line, so that park is on two-photon resonance to ~44 kHz.
    # The AT midpoint is a FAR tighter anchor than the 616 revival peak (a 4.9 MHz-wide feature),
    # so this freq-2D centers on (97.3457, 392.948), NOT on the extrapolated (97.275, 393.533).
    # Grid is self-consistent with the ridge: |slope| ~0.93 carrier-MHz per EOM-MHz means a
    # +-0.30 EOM span sweeps the ridge +-0.28 in carrier, so carrier spans +-0.30 to contain it.
    # NOTE the ridge SIGN is unknown at this operating point: AOM3 moved from the -1st to the
    # +1st order on 09-14, which should FLIP the old 71-3S1 slope (+0.926) to negative. The grid
    # is symmetric in carrier, so it catches the ridge either way -- do not assume the sign.
    # 2026-09-15 ROUND 2 -- WIDENED. Round 1 (data_20260915_133655, 606/910 shots = ~6.7 passes
    # over the 91 combos) showed no usable structure in the +-0.30 x +-0.30 box, which is enough
    # coverage that a dip inside that window should have been visible. So the optimum is NOT
    # within +-0.30 of the spectroscopic resonance: either the STIRAP two-photon lock sits well
    # off the AT midpoint (the mj=-1 lock sat +0.70 MHz off ITS revival -- same order as this
    # miss), or the ridge runs out of the old box. Both spans roughly DOUBLED to +-0.60 to see
    # the TREND; steps coarsened (0.05->0.10 carrier, 0.10->0.15 EOM) to hold the shot count flat.
    # Sampling is still fine: the 556 line is 158 kHz FWHM, so 0.10 MHz resolves a dip.
    # Ridge containment preserved: |slope| ~0.93 x +-0.60 EOM = +-0.56 carrier, inside +-0.60.
    # ZOOM (09-15 round 3): round 2's freq-2D (data_20260915_140525, 13x9, 2-3 shots/pt) showed a
    # clean two-photon ridge of slope ~0.875 MHz EOM616 per MHz carrier, with excitation WORST on
    # the bare resonance (survival 0.338 at carrier 97.3457 / EOM 392.948) and improving on BOTH
    # sides -- two local optima at opposite intermediate-state detunings. This run zooms the
    # LOW-detuning optimum. NOTE the round-2 low side was still falling at its grid corner
    # (0.212 at 96.7457/392.348, the extreme point), so this box sits on the ridge shoulder
    # (0.259 at 96.9457/392.648) and may NOT contain the true minimum -- deliberate operator
    # choice to resolve the shoulder finely first.
    CARRIER_SEED_MHZ = ZOOM["carrier_center"]
    _car = _axis(CARRIER_SEED_MHZ, ZOOM["carrier_half"], ZOOM["carrier_step"])
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(1, _car)
    g().AWG.AWG556.Ch1.pulse_width_us = 3  # every_other lock (chain baseline)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 14
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87  # 08-06 amp scan monotonic to ceiling (power-limited)

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    # dimer_40um reverse: sweep data_20260828_054718 was pure noise (~22 events/pt, no
    # peak) -- reverse params ADOPTED from the clean geometries (offset +0.09; delay
    # mid-range), NOT measured on this pattern.
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 119.0363  # was 118.5326 (71 3S1); on resonance pending freq-2D
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0  # widths 2.0/2.0 kept (08-19 top-N verify tied-best)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 14
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8  # amp saturated by 7.5 (07-15)
    g().AWG.AWG308.Ch1.amplitude_scale = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2

    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1


    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; kept for the future, currently unused) ---
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these) -------------------------
    # 2026-08-28 CHIRPED forward STIRAP: the 616 EOM ramps Init -> Final during the 308
    # pulse. Chirp CENTERED on the every_other lock (229.6655): init = lock - ramp/2,
    # final = lock + ramp/2. DETUNING_EOM616_RAMP_MHZ = SIGNED total chirp (0 = no ramp,
    # the incumbent). Both lists co-vary on dim 1 (pairing by construction).
    # CHIRP PAUSED 2026-08-28 (resume later) -- uncomment the block below to re-enable.
    #FWD_EOM616_LOCK_MHZ = 229.6655
    #DETUNING_EOM616_RAMP_MHZ = [round(float(d), 4) for d in np.linspace(-1.0, 1.0, 20)]
    #FREQ_EOM616_INIT_MHZ = [round(FWD_EOM616_LOCK_MHZ - d / 2, 4) for d in DETUNING_EOM616_RAMP_MHZ]
    #FREQ_EOM616_FINAL_MHZ = [round(i + d, 4) for i, d in zip(FREQ_EOM616_INIT_MHZ, DETUNING_EOM616_RAMP_MHZ)]
    #g().Init.EOM616.Freq.scan(1, np.asarray(FREQ_EOM616_INIT_MHZ) * 1e6)
    #g().Pushout.EOM616.Freq.Final.scan(1, np.asarray(FREQ_EOM616_FINAL_MHZ) * 1e6)
    EOM616_LIST_MHZ = None  # locked below
    # 2026-09-10 (66 3S1): EOM616 on today's MEASURED 308 revival 343.839 MHz
    # (scan 20260910213014, FWHM 6.7 MHz; the 17:54 fit gave 343.9459 at FWHM 20.1 MHz --
    # peak stable, line NARROWED as the 308 coupling fell over the evening).
    # was 228.7775e6 (71 3S1 every_other lock; that state's revival sat ~229-230 MHz).
    # 2026-09-14 mj=+1 seed (see the carrier block above): revival 392.8326 + 0.700 offset.
    # 2026-09-15: superseded by today's MEASURED park (see the carrier block above) -- the
    # Autler-Townes scan 20260915_125339 was taken with EOM616 = 392.948e6 and its doublet
    # midpoint landed +22 kHz from the bare 556 line, i.e. that park is on two-photon resonance
    # to ~44 kHz. Centering here instead of on the extrapolated 393.533.
    # *** UNITS: Init.EOM616.Freq is in Hz (Consts default 370e6). The pinned value that stood
    # here read ``= 393.533`` with NO e6 -- i.e. 393.5 Hz, not 393.533 MHz. That bug was LIVE:
    # the 1-combo fixed-point runs 20260915_111055/111759/111844/112203/120606/120639/121654/
    # 121757 all recorded "EOM616": {"Freq": 393.533} in their descriptors, so the 616 EOM was
    # driven to ~0 and those shots cannot have transferred. Always write the e6. ***
    # ZOOM: 392.60 is the ridge partner of carrier 96.95 (round-2 best EOM at that carrier was
    # 392.648, survival 0.259). Step halved to 0.075. Still written as e6 -- see the units note above.
    EOM_SEED_MHZ = ZOOM["eom_center"]
    _eom = _axis(EOM_SEED_MHZ, ZOOM["eom_half"], ZOOM["eom_step"])
    g().Init.EOM616.Freq.scan(2, np.array(_eom) * 1e6)   # ALWAYS e6 -- see the units note above
    #g().Pushout.EOM616.Freq.Final = 229.6655e6  # UNUSED (chirp disabled in step)

    g().Pushout.VRydTrap = 2.0
    g().Pushout.BiasCoilCurrent.Ryd = 60

    g().Pushout.STIRAPDelay = 1.35e-6  # every_other lock (chain baseline); pinned for the freq-2D
    g().Pushout.STIRAPReverseDelay = 1.7500e-6
    g().Pushout.STIRAPPadTime = 2e-6  # 07-21 mj=0 quad ridge-3D optimum
    # lifetime: forward -> hold STIRAPGap -> reverse; RETURN vs gap = decay curve.
    GAP_PTS = np.geomspace(0.1e-6, 50e-6, 20)
    g().Pushout.STIRAPGap = 1e-6 #.scan(1, GAP_PTS)

    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.IfRecoveryIonization = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6   # mj=0 (pumps OFF this config)
    g().Pushout.Pump556Freq = 143.556e6
    g().Pushout.Pump556Amp = 0.5
    g().Pushout.SLMAOMAmpGap = 0.55
    g().Pushout.IfGatePulses = 1  # 1 = Rydberg pulses NOT skipped

    g().Pushout.IonizationViaDAC = 0  # 1: DAC ramp; 0: TTL switch
    g().Pushout.TimeIonization = 0.1e-6  # voltage set in InitStep, switched by TTL
    g().Init.VIonizationSet5to8 = 4
    g().Pushout.TIonizationAlign = 0.5e-6  # wait to align trap with ionization pulse
    #g().ReleaseRecapture.Time = 1e-6  # RnR alternative (scienceStep="rnr"); keep off dim 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) --------------
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
    
    g().rearrange_kwargs.extras.wgs_warm = True  # warm WGS for rearrangement
    g().rearrange_kwargs.extras.wgs_pad = 2048
    g().rearrange_kwargs.extras.wgs_iters = 3

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) ------------------------------
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "every_other"  # 2026-09-14 mj=+1 forward-STIRAP campaign
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN
    g().rearrange_kwargs.extras.scienceStep = "stirap"  # "rnr" = RnR alternative

    # ---- run params (runp) ---------------------------------------------------------------
    # rep passes over the whole grid: NumPerGroup = REPS * n_combos (13 carrier x 9 EOM = 117).
    n_car = len(_axis(ZOOM["carrier_center"], ZOOM["carrier_half"], ZOOM["carrier_step"]))
    n_eom = len(_axis(ZOOM["eom_center"], ZOOM["eom_half"], ZOOM["eom_step"]))
    rp.NumPerGroup = REPS * n_car * n_eom   # REPS passes over the whole grid
    rp.loading_defocus = -5  # ANSI z4 (rad) on the loading phase; MATCH rearrange_kwargs.extras.z4
    rp.NumImages = 3 if verify else 2
    # MUST be 0 whenever EOM616 is swept (random EOM jumps kick the 616 out of lock); 1 otherwise.
    # 2026-09-14: Scramble OFF for the freq-2D. This scan sweeps Init.EOM616.Freq, and a
    # SCRAMBLED EOM sweep kicks the 616 out of lock (gotcha-scramble-unlocks-616-eom-sweep:
    # scan dies at seq ~5-7, or the revival reads flat because 308 never shelves). Both
    # successful 09-10 freq-2D runs (20260910224514 / 20260910230416, 484 shots each) used
    # Scramble = 0. Turn it back ON for pulse-width / delay scans, which do not touch the EOM.
    rp.Scramble = 0
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    # Per-frame detection declaration (explicit -> wins over the runner's 2-frame synthesis).
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def _describe():
    """Run description built from the ACTUAL window, so provenance cannot drift from the code."""
    car = _axis(ZOOM["carrier_center"], ZOOM["carrier_half"], ZOOM["carrier_step"])
    eom = _axis(ZOOM["eom_center"], ZOOM["eom_half"], ZOOM["eom_step"])
    return (
        "STIRAP OPTIMIZATION: fine forward freq-2D at the operating point 3P1 mj=+1 / 60 G / "
        "66 3S1, every_other on 33x33_feedback11. Carrier %.4f-%.4f MHz (%d pts x %.4f) x "
        "EOM616 %.4f-%.4f MHz (%d pts x %.4f), %d combos x %d reps = %d shots. Survival LOW = "
        "more two-photon excitation. WHY THIS WINDOW: the 09-15 coarse freq-2D "
        "(data_20260915_140525, 13x9 over +-0.60, 2-3 shots/pt) found a clean two-photon ridge, "
        "slope ~0.875 MHz EOM616 per MHz carrier, with excitation WORST on the bare 556 "
        "resonance (survival 0.338 at carrier 97.3457 / EOM 392.948) and improving on BOTH "
        "sides -- two local optima at opposite intermediate-state detunings; that is also why "
        "round 1's +-0.30 box (data_20260915_133655) saw no structure, the ridge crosses it "
        "diagonally with the worst point at its centre. Round 3 (data_20260915_142454, this "
        "grid at carrier 96.95 / EOM 392.60) reached survival 0.171 +/- 0.006 at carrier 97.00 "
        "/ EOM 392.675. Pulse widths 3/3 us and delay 1.35 us PINNED so this measures frequency "
        "only. IfReverse=0, Scramble=0 (EOM is swept). Bare 556 line 97.3237 MHz, AT midpoint "
        "97.3457 MHz (scans 20260915_123732 / 20260915_125339).%s"
        % (car[0], car[-1], len(car), ZOOM["carrier_step"],
           eom[0], eom[-1], len(eom), ZOOM["eom_step"],
           len(car) * len(eom), REPS, REPS * len(car) * len(eom),
           (" " + ZOOM["note"]) if ZOOM["note"] else ""))


def RearrangeSTIRAPScan(url=None, reps=REPS):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan",
                      description=_describe(),
                      **opts)
    print("submitted RearrangeSTIRAPScan -> descriptor id %s (url=%s, reps=%s, verify=%s, "
          "NumImages=%d)" % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeSTIRAPScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="passes over the sweep (NEVER 0: rep=0 skips the scan-order build, so "
                         "pyctrl writes no config['Params'] and the 2-D axes cannot be mapped)")
    ap.add_argument("--carrier-center", type=float, default=ZOOM["carrier_center"],
                    help="556 AWG carrier center, MHz (e.g. a newly fitted ridge optimum)")
    ap.add_argument("--carrier-half", type=float, default=ZOOM["carrier_half"])
    ap.add_argument("--carrier-step", type=float, default=ZOOM["carrier_step"])
    ap.add_argument("--eom-center", type=float, default=ZOOM["eom_center"],
                    help="616-EOM center, MHz (written as e6 internally)")
    ap.add_argument("--eom-half", type=float, default=ZOOM["eom_half"])
    ap.add_argument("--eom-step", type=float, default=ZOOM["eom_step"])
    ap.add_argument("--note", default="", help="appended to the run description")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the axes + shot count WITHOUT submitting")
    args = ap.parse_args()
    if args.reps == 0:
        raise SystemExit("--reps 0 is refused: it breaks the Params map (see --reps help)")
    ZOOM.update(carrier_center=args.carrier_center, carrier_half=args.carrier_half,
                carrier_step=args.carrier_step, eom_center=args.eom_center,
                eom_half=args.eom_half, eom_step=args.eom_step, note=args.note)
    REPS = args.reps
    if args.dry_run:
        car = _axis(ZOOM["carrier_center"], ZOOM["carrier_half"], ZOOM["carrier_step"])
        eom = _axis(ZOOM["eom_center"], ZOOM["eom_half"], ZOOM["eom_step"])
        print("carrier (%d pts): %s" % (len(car), car))
        print("EOM616  (%d pts): %s" % (len(eom), eom))
        print("shots = %d reps x %d combos = %d" % (args.reps, len(car) * len(eom),
                                                    args.reps * len(car) * len(eom)))
        print("\ndescription:\n" + _describe())
        raise SystemExit(0)
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
