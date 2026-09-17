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
# 2026-09-15 STAGE B (pw556 x pw308 x delay, 5 x 5 x 3 = 75 combos): 5 passes = 375 shots,
# ~15 min. A coarse 3-D box is spent LOCATING the efficient plane, not resolving a cell -- the
# fine co-vary along that plane is where the statistics go. (Was 20 for the 1-D delay scan,
# where 19 points could afford 20 shots each.)
REPS = 30
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
    # ---- the swept grid: carrier (dim 1) x EOM616 (dim 2) --------------------------------
    # Centred on the CURRENT lock (carrier 96.7036 / EOM616 392.4517, data_20260915_162512).
    # Window is DELIBERATELY IDENTICAL to the 09-15 round 2 (data_20260915_140525): +-0.60
    # carrier x +-0.60 EOM, steps 0.10 / 0.15 -> 13 x 9 = 117 combos. Round 2 declared that
    # window CONVERGED (ridge interior, BOTH off-ridge baselines reached), so re-using it makes
    # a walk show up directly as a SHIFT OF THE ARGMIN inside a known-good box.
    # Ridge from round 2: carrier = 97.392 + 1.25*(EOM616 - 392.948), b = +1.25 +- 0.06.
    # |slope| 1.25 x +-0.60 EOM = +-0.75 carrier > the +-0.60 carrier span, so the two extreme
    # EOM columns truncate exactly as they did in round 2 -- ignore their per-column fits.
    # 2026-09-16 1-D amp556 SCAN (per the user: the amp scale deserves its own 1-D scan).
    # WHY IT IS GENUINELY OPEN -- the two readings so far CONTRADICT each other and BOTH are
    # contaminated, so neither settles it:
    #   * job 118 (pw308 x amp556, aborted): at pw308=3.0 the amp row fell monotonically
    #     0.788/0.583/0.441/0.367/0.303 for amp 0.40..1.00 -> "push to the 1.0 ceiling".
    #     But that was 2 shots/cell.
    #   * job 119 (pw308 x delay) at amp556 1.0, pw 3/3, delay 1.5 read 0.3501 +- 0.0077, while
    #     stage C at amp556 0.87, delay 1.35 read 0.3415 +- 0.0030 -> "1.0 is WORSE".
    #     But those are different runs AND stage C ran through the array warm-up (img1 fill
    #     drifted +0.089 within that run; every run since has been flat at ~0.58-0.60), so its
    #     absolute level sat on a moving baseline and is not safely comparable.
    # A single 1-D sweep under today's now-stable loading settles it with all points sharing one
    # run and one baseline. Span goes WELL BELOW the incumbent 0.87, because the physics is not
    # one-sided: the 556 leg carries ~4x the 308 power, and an over-strong pump against a
    # power-limited Stokes leg can cost adiabaticity and add off-resonant scattering, so the
    # optimum is not obviously at the ceiling.
    AMP556_LIST = [round(float(v), 3) for v in np.linspace(0.30, 1.00, 8)]  # dim 1
    # Geometry PINNED at job 119's best on-ridge cell (pw308 3.0 / delay 1.5 us, 0.3501 +-
    # 0.0077, which beat the next-best pw308 row by 4.1 sigma). NOTE delay 1.5 here, not stage
    # C's 1.35 -- 1.5 is what this grid actually measured as best.
    PW556_US   = 3.0
    PW308_US   = 3.0
    DELAY_US   = 1.5
    CARRIER_MHZ = 96.7036
    EOM616_MHZ  = 392.4517
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
    # 2026-09-15 ROUND 2 -- WIDENED to +-0.60 x +-0.60 (steps coarsened 0.05->0.10 carrier,
    # 0.10->0.15 EOM to hold the shot count flat). Ridge containment: |slope| ~1.25 x +-0.60 EOM
    # = +-0.75 carrier, so the two extreme EOM columns truncate -- see below.
    #
    # *** CORRECTION (do not repeat this mistake). Round 1 (data_20260915_133655, 689 shots =
    # 7.57 passes over 91 combos) was called "no usable structure" at the time. THAT WAS WRONG.
    # Round 1 contains a ~62 pp, ~36-sigma diagonal ridge. The miss is the classic Rule-2 failure
    # the STIRAP runbook warns about: reading analyze_scan/run_analysis summary.survival_mean,
    # which COLLAPSES AND RE-ORDERS a 2-D sweep. Re-read with the Rule-1 metric (Params-grouping
    # + select_subset_stirap target mask) the structure is unmissable. Round 1's real defect was
    # only its WINDOW: +-0.30 sits entirely inside the dip's flank, so it never reaches the
    # off-resonant baseline (window max 93%, fitted asymptote 1.19 = unphysical) and its argmin
    # rails on the carrier low edge. Widening was the right call for the wrong stated reason. ***
    #
    # ROUND 2 RESULT (data_20260915_140525, 339 shots = 2.90 passes; verified two ways):
    #   argmin  28.96% +- 1.42 survival at carrier 96.9457 / EOM616 392.648
    #   baseline 98.62% off-ridge  =>  peak excitation ~71%, argmin at 48.7 sigma
    #   ridge    carrier = 97.392 + 1.25*(EOM616 - 392.948)   [b = +1.25 +- 0.06, R^2 0.942]
    #   Window is CONVERGED: ridge interior, BOTH baselines reached, no further widening needed.
    #   Only the two extreme EOM columns (392.348 / 393.548) truncate -- ignore their fits.
    # The ridge intercept sits +0.05..+0.10 MHz ABOVE today's AT midpoint 97.3457, so the AT
    # midpoint is on the ridge but ~0.10 MHz low in carrier (worth ~4-6 pp excitation, 5.2 sigma).
    # NOTE the slope did NOT flip sign despite the 09-14 AOM3 -1st -> +1st order move (old
    # 71-3S1 value was +0.926); b stayed POSITIVE and grew ~35%. Open question for yb-physics --
    # do not assume the sign from the AOM order.
    # 2026-09-15 STEP 2 -- FREQUENCY PAIR PARKED, and the scan axis moves to the DELAY (see the
    # STIRAPDelay block below). Park taken from the FINE 5x5 zoom data_20260915_142454 (125 shots,
    # carrier 96.85-97.05 x 0.05, EOM616 392.450-392.750 x 0.075), which refines the round-2
    # coarse argmin:
    #   PARK  carrier 97.0000 MHz / EOM616 392.675 MHz -> survival 27.00% +- 0.34 (~73% excitation)
    #   The cell is INTERIOR on both axes (index 3 of 5) and well determined: 1678 mid events,
    #   SEM 0.34 pp, vs a 73.9% worst cell in the same zoom.
    # Supersedes the round-2 coarse argmin (96.9457 / 392.648, 28.96% +- 1.42), which the zoom
    # re-reads as 29.00% +- 1.11 -- consistent, and ~2 pp worse than the refined park.
    # CAVEAT, carried deliberately: the ridge bottom is FLAT. The zoom's next-best cells read
    # 27.6 +- 2.1 (96.90/392.525) and 29.0 +- 1.1 (96.95/392.600), i.e. within ~1 sigma of the
    # park. This is the best MEASURED pair, not a resolved optimum -- the runbook's plateau
    # lesson says such an argmin can be noise-selected. Once the delay is optimized, re-read the
    # frequency along the ridge carrier = 97.392 + 1.25*(EOM616 - 392.948) and/or top-N verify.
    g().AWG.AWG556.Ch1.carrier_freq_MHz = CARRIER_MHZ  # PINNED (incumbent)
    # ---- STAGE C (2026-09-15): TOP-N VERIFY, >=100 shots/point ------------------------
    # Stage B's width/delay plateau is flat: the top cells span 0.3026..0.3089 with SEMs
    # ~0.012-0.017, i.e. all within 1 sigma at 5 shots/cell. The runbook says settle that with a
    # TOP-N verify, not by locking a single noise-selected argmin -- so the best 3 cells (each
    # passing a mid-event guard of >=0.7x the median 1298 events, so small-N cannot win) are
    # re-run together at 100 shots each.
    # CO-VARY PATH: all three params sit on scan dim 1, so point k is the triple
    # (PW556_US[k], PW308_US[k], DELAY_US[k]) -- pairing by construction. Runbook Rule 2: the lab
    # analysis would collapse and re-order this, so the read-out MUST group by the logged Params.
    #   k=0  4.0 / 4.0 / 1.35   (stage-B rank 1, 0.3026 +- 0.0165)
    #   k=1  3.5 / 3.0 / 1.00   (rank 2,        0.3072 +- 0.0168)
    #   k=2  3.0 / 3.0 / 1.35   (rank 3 = the INCUMBENT, 0.3089 +- 0.0144)
    # 2026-09-16 FREQ-2D: pulse area PINNED to the incumbent that today's stage-C top-N
    # verify CONFIRMED (scan 20260916173225, 100 shots/pt): pw556 3.0 / pw308 3.0 /
    # delay 1.35 us, survival 0.3415 +- 0.0030 target-masked verify-conditioned. Pinning these
    # makes this scan measure FREQUENCY ONLY.
    g().AWG.AWG556.Ch1.pulse_width_us = PW556_US
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 14
    g().AWG.AWG556.Ch1.amplitude_scale.scan(1, [float(v) for v in AMP556_LIST])  # SWEPT dim 1

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    # dimer_40um reverse: sweep data_20260828_054718 was pure noise (~22 events/pt, no
    # peak) -- reverse params ADOPTED from the clean geometries (offset +0.09; delay
    # mid-range), NOT measured on this pattern.
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 119.0363
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0  # widths 2.0/2.0 kept (08-19 top-N verify tied-best)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 14
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = PW308_US  # PINNED (job 119 best on-ridge)
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
    # 2026-09-15 STEP 2: PARKED partner of carrier 97.0000 (see above), from the 5x5 zoom
    # data_20260915_142454. NOT 392.948 (today's AT park) -- that column's best cell reads
    # ~7 pp worse. UNITS: Hz (this field is Hz; the missing e6 was a real bug earlier today).
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6  # PINNED (incumbent); UNITS: Hz
    #g().Pushout.EOM616.Freq.Final = 229.6655e6  # UNUSED (chirp disabled in step)

    g().Pushout.VRydTrap = 2.0
    g().Pushout.BiasCoilCurrent.Ryd = 60

    # 2026-09-15 STEP 2 -- THE SWEPT AXIS. 1-D delay scan at the parked frequency pair.
    # This is an OPTIMIZATION scan, not the runbook's step-0 existence test: 1.35 us demonstrably
    # transfers at this operating point (round 2 reached ~71% excitation with it), so the
    # "delay does not transfer at mj=+1" hypothesis is already REJECTED. What we do not know is
    # whether 1.35 us -- inherited from the mj=-1 / 71-3S1 every_other lock -- is anywhere near
    # optimal here, and the delay sets the 556<->308 overlap that carries the adiabatic transfer.
    # Range: the quintic splines have COMPACT SUPPORT (playback total = pw = 3 us each), so the
    # pulses stop overlapping beyond delay ~ 3 us. 0.2..3.0 us in 0.2 steps = 15 pts, spanning
    # the current 1.35 us lock (the grid lands on 1.4 = free near-anchor) out to no-overlap.
    #
    # *** DELAY MUST BE STRICTLY POSITIVE -- learned the hard way, 2026-09-15. ***
    # The first attempt at this scan (job 2073, data_20260915_144102) swept -0.6..3.0 and DIED
    # after 2 shots with "run error: Forward_Delay must be > 0 for the forward STIRAP pulse
    # sequence." Both STIRAPPushoutStep.py (line ~180/201) and STIRAPHighFieldPushoutStep.py
    # (line ~182/203) implement the forward pulse as: fire the 308 gate, then s.wait(Forward_Delay),
    # then fire the 556 gate -- so a delay <= 0 is NOT REPRESENTABLE and raises at sequence-BUILD
    # time, per shot, which kills the whole job. The descriptor was correct; the sweep was not.
    #
    # DOCS-VS-CODE CONFLICT, flagged and NOT resolved here: the STIRAP runbook's knobs section
    # says "Pushout.STIRAPDelay -- SIGNED seconds; negative = 556 fires first", and the 07-15
    # mj=0 campaign recorded transfer ONLY at negative delay (dip ~-0.4 us). The live step cannot
    # play a 556-first forward pulse at all. So either the sign convention changed when the pulse
    # shapes moved to the quintic splines, or those historical negative-delay optima are not
    # reproducible with this step as written. Do NOT add negative points back without first
    # changing STIRAPPushoutStep to emit the 556 gate first for Forward_Delay < 0 -- that is a
    # pyctrl code change, and a deliberate one, not a sweep edit.
    DELAY_PTS_US = np.arange(0.2, 3.001, 0.2)  # 15 pts, all > 0
    g().Pushout.STIRAPDelay = DELAY_US * 1e-6  # PINNED (job 119 best on-ridge)
    g().Pushout.STIRAPReverseDelay = 0.0000e-6
    g().Pushout.STIRAPPadTime = 2e-6  # 07-21 mj=0 quad ridge-3D optimum
    # lifetime: forward -> hold STIRAPGap -> reverse; RETURN vs gap = decay curve.
    GAP_PTS = np.geomspace(0.1e-6, 50e-6, 20)
    g().Pushout.STIRAPGap = 1e-6

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
    g().rearrange_kwargs.extras.pattern = "every_other"  # chain
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN
    g().rearrange_kwargs.extras.scienceStep = "stirap"  # "rnr" = RnR alternative

    # ---- run params (runp) ---------------------------------------------------------------
    # rep passes over the sweep: NumPerGroup = REPS * n_points (15 delay points, all > 0).
    rp.NumPerGroup = REPS * len(AMP556_LIST)  # 1-D: 30 x 8 = 240
    rp.loading_defocus = -5  # ANSI z4 (rad) on the loading phase; MATCH rearrange_kwargs.extras.z4
    rp.NumImages = 3 if verify else 2
    # MUST be 0 whenever EOM616 is swept (random EOM jumps kick the 616 out of lock); 1 otherwise.
    # 2026-09-14: Scramble OFF for the freq-2D. This scan sweeps Init.EOM616.Freq, and a
    # SCRAMBLED EOM sweep kicks the 616 out of lock (gotcha-scramble-unlocks-616-eom-sweep:
    # scan dies at seq ~5-7, or the revival reads flat because 308 never shelves). Both
    # successful 09-10 freq-2D runs (20260910224514 / 20260910230416, 484 shots each) used
    # Scramble = 0. Turn it back ON for pulse-width / delay scans, which do not touch the EOM.
    # 2026-09-15 STEP 2: back ON. This scan sweeps STIRAPDelay ONLY -- Init.EOM616.Freq is a
    # pinned scalar (392.648e6), so the EOM never jumps and the lock is safe. Scrambling matters
    # here: loading drifted monotonically 0.33 -> 0.51 across the afternoon's runs, and with
    # Scramble = 0 that drift is correlated with point index within a pass. It was harmless for
    # the freq-2D (the Rule-1 verify-conditioned metric moved only ~1.5 pp over a whole run), but
    # a 1-D delay curve is exactly the shape a monotone drift can fake, so decorrelate it.
    # *** If you ever re-enable an EOM sweep in this file, set this back to 0. ***
    # This scan does NOT sweep Init.EOM616.Freq (it is a pinned scalar), so the 616 lock is
    # safe and scrambling is both allowed AND wanted: img1 loading drifted +0.115 WITHIN a
    # single run today (0.471 -> 0.585 by quarter), and a monotone drift is exactly the shape
    # that can fake a trend along a swept axis.
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    # Per-frame detection declaration (explicit -> wins over the runner's 2-frame synthesis).
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeSTIRAPScan_fwd_amp556_1d_60G_mjp1(url=None, reps=REPS):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan_fwd_amp556_1d_60G_mjp1",
                      description=(
                          "STIRAP 1-D amp556 09-16: AWG556.Ch1.amplitude_scale 0.30..1.00 in 8 "
                          "steps (dim 1) x 30 passes = 240 shots, at max_amplitude_vpp 14. "
                          "Operating point 3P1 mj=+1 / 60 G / 66 3S1, every_other on "
                          "33x33_feedback11. WHY: the two readings so far CONTRADICT and both are "
                          "contaminated -- job 118's pw308=3.0 amp row fell monotonically to the "
                          "1.0 ceiling but at 2 shots/cell, while job 119 at amp 1.0 read 0.3501 "
                          "+- 0.0077 vs stage C's 0.3415 +- 0.0030 at amp 0.87, a CROSS-RUN "
                          "comparison in which stage C ran through the array warm-up (img1 fill "
                          "drifted +0.089 within that run; every run since is flat at ~0.58-0.60). "
                          "One 1-D sweep under today's stable loading settles it on a single "
                          "baseline. Span reaches WELL BELOW the incumbent 0.87 deliberately: the "
                          "556 leg carries ~4x the 308 power, and an over-strong pump against a "
                          "power-limited Stokes leg can cost adiabaticity, so the optimum is not "
                          "obviously at the ceiling. Geometry PINNED at job 119's best on-ridge "
                          "cell (pw556 3.0 / pw308 3.0 / delay 1.5 us -- note 1.5, not stage C's "
                          "1.35), which beat the next-best pw308 row by 4.1 sigma. Frequencies "
                          "pinned at the incumbent (96.7036 / 392.4517). Analysis: MINIMIZE "
                          "survival; group by the logged Params (Rule 2), target-masked "
                          "verify-conditioned metric (Rule 1). IfReverse=0. Scramble=1 (no EOM "
                          "sweep; decorrelates any residual drift from the swept axis)."),
                      **opts)
    print("submitted amp556_1d_60G_mjp1 -> descriptor id %s (url=%s, reps=%s, verify=%s, "
          "NumImages=%d)" % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeSTIRAPScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan_fwd_amp556_1d_60G_mjp1(url=args.url, reps=args.reps)
