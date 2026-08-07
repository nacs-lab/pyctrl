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

    # 2026-07-16 FIXED-POINT VERIFY (100 shots) at the located optimum. Plane rounds 1-4 (data
    # _20260715_232915 + _120548/_121822/_123244/_124105) walked the efficient plane; the argmin
    # kept railing toward longer pulses + more-negative delay, but best-of-round only climbed
    # 93.0->94.2->94.6->95.0% while the SAME anchor cell re-read ~2% LOWER scan-to-scan -> this is a
    # broad ~92-95% PLATEAU, not a real ridge climb (the per-round gain sits inside the drift floor).
    # Stop extending; verify round-4 best @ 100 shots. All .scan() commented -> 1-point config.
    # 2026-07-16 FORWARD STIRAP re-optimization after the DC-Stark E-field re-null (308 line ~234.3 MHz).
    # FINAL: freq-2D re-scan (data_20260716_181005) pinned carrier 143.300 / EOM616 234.444 (best 96.5%);
    # plane 3D re-walk (data_20260716_182213) gave an INTERIOR optimum (not railed) at pw556 4.159 /
    # pw308 3.515 / delay -0.720us, 95.8+/-0.4% (tight 95.4-95.8 plateau; E-field null lifted it ~1% vs
    # the pre-null 93-95%, optimum moved to shorter delay). All .scan() off -> 1-point verify config.
    # 2026-07-16 STIRAP optimum from a QUADRATIC FIT of the 5^3 Cartesian box (data_20260716_184636,
    # 20/pt). Paraboloid vertex (concave, true max) = pw556 4.531 / pw308 4.094 / delay -0.648 us,
    # predicted 97.0% (vs the argmax cell 96.0% -- the flat plateau makes a single-cell pick unreliable,
    # so we fit). pw308 vertex sits ~0.08us past the box edge (fit extrapolation); this 100-shot verify
    # tests it. Freq pair 143.300 / 234.444 (freq-2D re-pin post E-field null). Forward (reverse OFF).
    #CARRIER_MHZ = [round(float(v), 4) for v in np.linspace(142.0, 145.0, 10)]     # axis 1
    #EOM616_MHZ  = [round(float(v), 4) for v in np.linspace(279.0, 285.0, 10)]     # axis 2 (MHz)
    
    
    # ---- 2026-08-03 forward-STIRAP campaign constants (defined BEFORE first use) --------
    # ROUND 4 OPTIMUM (locked; see the round-by-round record below).
    PW556_US, PW308_US, DELAY_US = 6.0, 4.7, 0.4
    # ROUND 5 frequency path -- along the degenerate line with a perpendicular leg (33-pt co-vary).
    CARRIER_MHZ = [round(142.7 + 0.2 * k, 4) for k in range(11)]     # along the line
    PERP_MHZ = [-0.15, 0.0, 0.15]                                    # perpendicular (added to EOM)
    LINE_OFFSET_MHZ = 90.7                                           # EOM - carrier (round 1: 234.2-143.5)
    FREQ_PATH = [(c, c + LINE_OFFSET_MHZ + p) for p in PERP_MHZ for c in CARRIER_MHZ]
    # 2026-08-03 REVERSE 2-D (user directive): Ch2 pulse width 2-7 us x reverse delay -1.5..+1.5 us.
    # The 556 Ch2 and 308 Ch2 widths are CO-VARIED on dim 1 so they stay EQUAL (the equal-width family
    # the physics argues for, and how the 07-31 rounds co-varied them). 7 us = hardware ceiling.
    PW_CH2_US = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]                       # (equal-width 2-D, done)
    # 2-D RESULT (data_20260803_125540, 78 combos x 6): WIDTH dominates, delay barely matters. Best
    # return per width: 2.0 0.771 | 3.0 0.781 | 4.0 0.782 | 5.0 0.777 | 6.0 0.861 | 7.0 0.869 -- a sharp
    # STEP between 5 and 6 (+0.084, >7 SEM) then a plateau (6 vs 7 tied, 0.8 SEM). At w 6-7 the return is
    # FLAT over the whole +-1.5us delay range (0.805-0.869, scatter-dominated); at w 2 delay matters a lot
    # (0.423 @ -1.5 vs 0.771 @ +0.25). Equal-width 7/7 = 0.8686 +-0.0073 now MATCHES the 07-31 asymmetric
    # 1.5/7 champion (0.876 +-0.009, 0.6 SEM apart) -> "equal-width loses" was a STALE-DELAY artifact
    # (07-31 measured 7/7 at 0.826 using a delay located for 2/2 widths).
    #
    # 2026-08-03 ASYMMETRIC Ch2 widths (user directive): 556 Ch2 and 308 Ch2 swept INDEPENDENTLY.
    # pw556 includes 1.5 = the 07-31 champion, so this is the same-day head-to-head. Delay FIXED at
    # -0.5us (the 2-D best, and justified because delay is flat at the large widths that win); CAVEAT --
    # short-556 cells may want a different delay, so re-check the delay of any short-556 winner.
    PW556_CH2_US = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]               # (asymmetric map, done)
    PW308_CH2_US = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]                    # (asymmetric map, done)
    # ASYMMETRIC RESULT (data_20260803_131420, 42 combos x 8, delay -0.5): ASYMMETRIC WINS.
    # 556 1.5 / 308 7 = 0.9106 +-0.0053 beats equal-width 7/7 = 0.8753 +-0.0060 by 4.4 SEM -- same scan,
    # same delay, same day. 308 wants the CEILING (pw308=7 best in every row: 0.875-0.911; pw308<=5 is
    # 0.52-0.78); 556 wants SHORT (down the pw308=7 column: 1.5 0.911 | 2 0.898 | 3 0.898 | 4 0.887 |
    # 5 0.853 | 6 0.882 | 7 0.875). So the 07-31 champion config is REAL and reads even better today
    # (0.911 vs 0.876). CORRECTION: the equal-width 2-D above led me to call the asymmetric advantage a
    # stale-delay artifact -- that was a CROSS-DAY comparison (today's 7/7 vs 07-31's 1.5/7) and it was
    # wrong; the same-day head-to-head settles it the other way.
    #
    # 2026-08-03 SHORT-556 + DELAY round: both winning axes are at a grid EDGE. pw308 rails at the 7us
    # hardware ceiling (cannot extend). pw556 rails LOW at 1.5 -> extend down to 0.5. And the winner's
    # delay was never optimized (fixed -0.5, the EQUAL-width best, while the 2-D showed short-556 cells
    # are delay-SENSITIVE), so sweep the delay here too.
    PW556_SHORT_US = [0.5, 1.0, 1.5, 2.0, 2.5]                       # scan dim 1 (fastest)
    REV_DELAY2_US = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25]            # scan dim 2

    g().AWG.AWG556.Ch1.shape = "rise_quintic"   # anchor; gap = inner-peak separation
    # 2026-07-31 freq-2D redo (data_20260731_102020) + diagonal extension along the +1:+1 line
    # (data_20260731_102751, EOM616 = carrier + 90.6). On a ROBUST (per-shot median) metric every
    # on-resonance pair from carrier 142.9 to 144.5 reads 0.045-0.093 -- i.e. the line did NOT move,
    # and 143.5/234.1 still has the best median (0.061, n=8). Kept the 07-30 lock.
    # ROUND 5 RESULT (data_20260803_120740, 33-pt co-vary x 5, at the round-4 pulse) -- frequency
    # CONFIRMED, no change. Along the line at the correct offset (perp 0), survival vs carrier:
    #   142.7 0.188 | 142.9 0.093 | 143.1 0.046 | 143.3 0.0391 | 143.5 0.0402 | 143.7 0.0414
    #   143.9 0.065 | 144.1 0.074 | 144.3 0.123 | 144.5 0.194 | 144.7 0.282
    # => broad INTERIOR optimum at carrier 143.1-143.7, a factor ~7 worse by 144.7. Along-line position
    # matters a lot, so a 1-D EOM sweep at one carrier is NOT sufficient (this is why round 1 alone was
    # not enough). Best cell 143.3 (0.0391+-0.0057) is within SEM of 143.5 (0.0402+-0.0071) -> KEEP the
    # long-standing 143.5, do not churn the lock on noise. PERPENDICULAR leg: at 143.5, perp 0 (0.0402)
    # clearly beats +-0.15 MHz (0.067 / 0.073), so the EOM = carrier + 90.7 offset is right, the
    # resonance is narrow (+-0.15 costs ~3 pp), and NO tilt is detectable across 143.1-143.7.
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 143.5   # PAIR w/ EOM616 234.2 (round 5 confirmed)
    # 2026-08-03 ROUND 2 -- ridge 3D, 5^3 Cartesian box CENTRED on the 07-21 lock (6.0 / 5.7 / +0.6us),
    # so the centre cell doubles as the runbook's repeated ANCHOR. Quadratic-fit the box (the 07-16
    # precedent, data_20260716_184636) rather than trusting the argmin cell -- the plateau is flat.
    # pulse_width_us must NOT exceed 7 (hardware ceiling).
    # ROUND 2 RESULT (data_20260803_111754, 125 combos x 4, 284 target sites). ALL THREE axes railed at
    # a grid edge: pw556 marginal fell monotonically 0.594->0.052 over 5->7 (rails at the 7us CEILING),
    # pw308 rose 0.079->0.505 over 4.7->6.7 (rails LOW at 4.7), delay fell 0.390->0.179 over 0.2->1.0
    # (rails HIGH). Quadratic fit = SADDLE (eig -0.039/-0.034/+0.214) -> vertex is NOT an optimum, do not
    # lock it. Best cells 0.027-0.036 (96.5-97.3% exc) vs ANCHOR 6/5.7/0.6 re-read 0.0590 +-0.0137
    # (94.10%): top = 7/4.7/0.4 0.0266+-0.0058, 6/5.7/0.8 0.0274+-0.0121, 6/5.2/0.6 0.0286+-0.0049.
    # The monotone marginals are driven by the OFF-plateau cells, not in-plateau structure (broad
    # plateau; good cells share pw556-pw308 ~1-2 with delay rising as pw556 falls = the ridge plane).
    #
    # ROUND 3 RESULT (data_20260803_113657, 32 combos x 5): BOTH extensions are dead ends -- the round-2
    # "rails" were artifacts of off-plateau cells dragging the marginals. pw308 4.7 beat every lower
    # value at every delay (3.2 is far worse: 0.093-0.752), and delay 0.4 beat 1.6 everywhere, so both
    # axes are INTERIOR, not railed. pw556 6 vs 7 ~tied at the good cells (6/4.7/0.4 0.0412+-0.0033 vs
    # 7/4.7/0.4 0.0460+-0.0058). DRIFT FLOOR CONFIRMED: 7/4.7/0.4 re-read 0.0460+-0.0058 here vs
    # 0.0266+-0.0058 in round 2 -- same config, ~2.4 SEM apart -> round 2's best-of-125 was a max-of-N
    # fluctuation. Optimum REGION = pw556 6-7 / pw308 4.7-5.2 / delay 0.4-0.8; do NOT lock an argmin.
    #
    # ROUND 4 -- TOP-N VERIFY (runbook step 4, the flat-plateau form): 6 candidate configs from rounds
    # 2+3 spanning the good region, CO-VARIED on scan dim 1 (Rule 2: re-pair by the logged Params), at
    # 100 shots each. Candidate 6 is the 07-21 ANCHOR/current lock, included so the comparison is a
    # drift-free head-to-head inside ONE scrambled scan rather than across rounds.
    # ROUND 4 RESULT (data_20260803_114419, 6 candidates x 100 shots, ~27k mid-events each) -- the
    # campaign optimum. Target-only verify-conditioned survival:
    #   6.0/4.7/0.4  0.0420 +-0.0022  (95.80% exc)   <== ADOPTED
    #   6.0/5.2/0.6  0.0452 +-0.0035  (95.48%)
    #   6.0/4.7/1.2  0.0465 +-0.0030  (95.35%)
    #   6.0/5.7/0.8  0.0469 +-0.0038  (95.31%)
    #   7.0/4.7/0.4  0.0520 +-0.0029  (94.80%)  <- round 2's 0.0266 was a max-of-N fluctuation
    #   6.0/5.7/0.6  0.0633 +-0.0038  (93.67%)  <- ANCHOR, the 07-21 lock
    # Winner beats the anchor by 4.9 SEM (+2.13 pp excitation) in ONE scrambled scan = drift-free.
    # Plateau top is flat (winner only 0.8 SEM ahead of 6.0/5.2/0.6). pw556 6 > 7 by 2.7 SEM, so the
    # 7us width ceiling is not binding. NOTE the day's imaging was degraded (per-site d' 4.5 vs the
    # 07-29 6.8, per-frame infidelity ~0.5% vs ~0.003%; 399 running 13.5 A without PS mode), which
    # floors every number here ~1.8 pp -- the same anchor config read 95.44% on 07-21. Re-verify at
    # restored 399 power; the RANKING should hold, the absolute values will rise.
    #
    # ROUND 5 -- re-check the two-photon frequency AT the new pulse (the runbook alternates freq <-> ridge
    # until the optimum stops moving; the pulse moved delay 0.6->0.4 and pw308 5.7->4.7). Scanned ALONG
    # the degenerate line (EOM = carrier + 90.7, today's measured offset) with a PERPENDICULAR offset, as
    # a 33-point CO-VARY path -- a 10x10 Cartesian would spend most cells off-line. The perpendicular
    # leg is what tells us whether the +90.7 offset tilts as the carrier moves (a 1-D EOM sweep at one
    # carrier cannot see that).
    # (CARRIER_MHZ / PERP_MHZ / LINE_OFFSET_MHZ / FREQ_PATH and the locked PW556_US / PW308_US /
    #  DELAY_US are defined at the top of build() -- they are used by the AWG block above.)
    # 2026-08-06 POWER RE-WALK ROUND 1. The USER RAISED THE 556 OPTICAL POWER on the hardware side,
    # which invalidates the locked ridge: every prior amplitude scan railed at the ceiling
    # ("monotonic to ceiling, best=1.0, still power-limited"), i.e. the old optimum was a POWER
    # constraint, not a physics optimum. With more light the 556 pulse AREA at a given
    # amplitude_scale x pulse_width is larger, so (a) amplitude_scale may now have a real INTERIOR
    # optimum below 1.0, and (b) the optimal pw556 should come DOWN (same area, more power).
    # amplitude_scale and pulse_width both set pulse area => strongly coupled => scan them as a 2-D,
    # not one at a time. pw308/delay stay at the round-4 lock this round (re-checked next round once
    # the 556 axis has moved). Metric = target-only verify-conditioned survival (runbook Rule 1),
    # re-paired by the logged Params (Rule 2). LOWER survival = better excitation.
    # BASELINE to beat: 6.0 / 4.7 / 0.4 = 0.0420 +-0.0022 (95.80% exc, data_20260803_114419).
    # amplitude_scale 1.0 is included so the baseline is re-read INSIDE this scan (drift-free
    # head-to-head), and the grid extends DOWN because that is the direction more power opens up.
    # 2026-08-06 ROUND 1 (user directive): pw556 FIXED at 5.0 us; sweep AMPLITUDE x DELAY instead.
    # Rationale for this pairing: with more 556 light the pulse AREA at a given amplitude is larger,
    # and amplitude and delay are the two knobs that set whether the adiabatic passage stays on the
    # ridge -- the 556/308 overlap (delay) is what the changed 556 Rabi frequency most directly
    # detunes. Width is held so the 2-D is a clean plane through the ridge, not an area-degenerate
    # slice (amp x width would trade off along constant area and mostly re-measure one axis).
    AMP556_PWR     = [0.4, 0.55, 0.7, 0.85, 1.0]                # scan dim 1 (fastest)
    DELAY_PWR_US   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]             # scan dim 2 (0.4 = the round-4 lock)
    g().AWG.AWG556.Ch1.pulse_width_us = 4 #5.0   # 2026-08-06 FIXED per user (was 6.0, the 08-03 lock)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15   # 2026-07-16 (now HONORED by AWGManager channel-mode; was silently forced to consts default 15)
    g().AWG.AWG556.Ch1.amplitude_scale = 0.85    # 2026-08-06 round 1: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    # g().AWG.AWG556.Ch1.amplitude_scale = 1   # pre-power-raise: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    
    g().AWG.AWG556.Ch2.shape = "fall_quintic"   # anchor; gap = inner-peak separation
    # 2026-08-06 QUADRUPLE_SPACING REVERSE ROUND 2 -- 556 Ch2 WIDTH x CARRIER 2-D (user directive).
    # Scanned together because they are coupled: the carrier sets the detuning from the 556 transition
    # and the width sets both the pulse AREA and the pulse BANDWIDTH (~1/pw; at 2us that is ~0.5 MHz,
    # comparable to the scan step here), so a narrow pulse tolerates a different detuning than a wide
    # one. Doing them one-at-a-time would pin each at the other's wrong value.
    # CARRIER RANGE: 143.5 is inherited from Ch1 (the locked FORWARD two-photon pair 143.5/EOM616
    # 234.2) on the argument that reverse drives the SAME 556 transition. That is physically sound but
    # has never been verified independently for the reverse pulse on any array, so this scan tests it
    # +-0.5 MHz around the inherited value rather than assuming it.
    # WIDTH RANGE: 2.0 was located on the DOUBLE_SPACING array (marginal peak, both axes interior) and
    # is unvalidated here; span 1.0-4.0 to re-locate it on quadruple_spacing. All values < 6 (the
    # user's hard width ceiling).
    PW556_CH2_Q2   = [1.0, 1.5, 2.0, 3.0, 4.0]                    # (width x carrier 2-D, done)
    CARRIER_CH2_Q2 = [143.0, 143.25, 143.5, 143.75, 144.0]        # (width x carrier 2-D, done)
    # Q2 RESULT (data_20260806_213036, 25 cells x 625 shots). BOTH axes INTERIOR.
    #   pw556 marginal:  1.0 0.7555 | 1.5 0.8381 | 2.0 0.8458 | 3.0 0.7114 | 4.0 0.4573  (+-0.006)
    #   car556 marginal: 143.00 0.6672 | 143.25 0.7330 | 143.50 0.7506 | 143.75 0.7350 |
    #                    144.00 0.7195  (+-0.007)
    #   best cell: pw556 1.5 / car 143.50 -> 0.8953 +-0.0114; runners-up 2.0/143.75 0.8811,
    #              2.0/143.50 0.8708, 1.5/143.75 0.8705 (flat top, all within ~2 SEM).
    # CARRIER LOCKED 143.5 -- a real resonance peaked exactly at the value inherited from the forward
    # Ch1 pair (143.5 / EOM616 234.2), with symmetric ~0.03-0.08 falloff at +-0.25-0.5 MHz. The
    # inheritance was never independently tested for the reverse pulse before; it is now.
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 143.5   # 2026-08-06 Q2 optimum (= the Ch1 line, verified)
    # 2026-07-31 REVERSE round 4 -- EQUAL-WIDTH test (per user: the 556 and 308 Ch2 pulses should be
    # at similar width and mostly overlap). pw556_Ch2 and pw308_Ch2 are CO-VARIED on scan dim 1 so
    # they stay equal; delay is dim 2. All values <= 7 (hardware ceiling).
    # Data so far DISAGREES with equal-width: 1.5/7 gave 0.876 +-0.009 (round 3, data_20260731_111148)
    # vs 7/7 = 0.819 +-0.017 and 2/7 = 0.834 +-0.005 (round 1, data_20260731_105620). This round gives
    # the equal-width region a fair shot at 6/pt instead of round 1's 4.
    # 2026-07-31 REVERSE campaign result. Best measured = 556 Ch2 1.5 / 308 Ch2 7 / delay -0.25us,
    # return 0.876 +-0.009 (round 3, data_20260731_111148). It BEATS the equal-width configuration the
    # physics argues for: 7/7 = 0.826 +-0.010, 6/6 = 0.704, 5/5 = 0.721, 4/4 = 0.729 (round 4,
    # data_20260731_111719, 6/pt) -- ~5 points / 4 SEM. NOT a shared-Siglent-playback artifact: the
    # reverse-OFF control (data_20260731_112412, 20/pt) reads the forward floor for BOTH Ch2 configs
    # (1.5/7 -> 0.101 +-0.031, 7/7 -> 0.082 +-0.009). OPEN QUESTION: for the spline shapes
    # total = pulse_width_us and each is ONE monotone ramp, so 556 falls to 0 by t=1.5us while 308 is
    # only ~20% risen -- the two barely overlap, yet it transfers better. Worth understanding.
    # 2026-08-06 REVERSE ROUND 1 under the <6us WIDTH CONSTRAINT (user directive: BOTH the 556 Ch2 and
    # the 308 Ch2 pulse width must be < 6 us). This retires the 07-31 champion (556 1.5 / 308 7.0) --
    # its 308 width is outside the allowed set -- so the reverse optimum must be re-located INSIDE the
    # box, not inherited.
    # WHY a 2-D over the two WIDTHS (not width x delay again): today's data already maps delay, and
    # says it is the WEAK axis inside the live window, while the widths are where the constraint
    # bites. Pooled marginals from data_20260806_190541 (pw556 x reverse delay = 5 x 11; pw308_Ch2
    # was FIXED at 4.0 there, NOT co-varied), target-only verify-conditioned; REVERSE wants HIGH:
    #   delay: -1.50 0.585 | -1.20 0.628 | -0.90 0.662 | -0.60 0.724 | -0.30 0.769 | 0.00 0.766
    #          +0.30 0.759 | +0.60 0.774 | +0.90 0.763 | +1.20 0.750 | +1.50 0.715 (+-0.012 each)
    #   => broad plateau -0.30..+1.20 (all within ~2 SEM), falling off hard below -0.6us.
    #   pw556 (pooled over all delays, at pw308=4.0): 2.0 0.7376 | 3.0 0.7361 | 4.0 0.7170 |
    #          5.0 0.6693 | 6.0 0.7328  (+-0.008 each) => best at the SHORT end (2-3us), 5.0 is a
    #          real dip; so the <6us constraint is NOT binding -- the allowed box already contains
    #          the better 556 widths, and the 07-31 "556 wants SHORT" result survives.
    # PAIRING CORRECTION (this is why the numbers above supersede an earlier reading): the per-shot
    # Params id is 1-indexed AND ScanGroup emits the DEFAULT cell as pid 1 before walking the rest
    # column-major, so a (pid-1)%n0 / (pid-1)//n0 formula mis-attributes EVERY cell (25/25 wrong on
    # this grid). Use tmp_rev_stirap_analyze.py, which rebuilds the pid -> cell map from the scan's
    # own recorded ScanGroup (validated 0/25 mismatches against a live-built group).
    # CAVEAT that motivates this round: today's reverse scans ran at ~1 shot/CELL (58-62 shots over
    # 55-70 combos), so each cell carries ~+-0.025 and the "best cell" (0.818) is a max-of-55
    # fluctuation, NOT a located optimum -- only the pooled marginals above are trustworthy. So:
    # hold delay on the plateau, sweep the two widths INDEPENDENTLY (the 07-31 asymmetric-vs-equal
    # question re-asked inside the legal box), and give each cell real shots.
    # 6 x 5 = 30 combos x --reps 20 = 600 shots = 20/cell (~5.6k mid-events/cell) -> ~+-0.006/cell,
    # enough to resolve the ~0.04-0.07 spread the marginals show instead of re-reading noise.
    # Metric = target-only verify-conditioned RETURN (Rule 1), paired via the scan's own descriptor
    # (Rule 2 + the pairing correction above), i.e. tmp_rev_stirap_analyze.py.
    # 556 axis reaches DOWN to 1.0 (the marginal favours the short end, and 07-31's champion was 1.5)
    # instead of centring on 4-5; 308 axis spans the legal box since its width was never mapped
    # independently below 7 -- every prior 308 conclusion came from the now-illegal 7us cell.
    # WIDTH-ROUND RESULT (data_20260806_191709, 30 combos x ~17 shots/cell, ~4.7k mid-events/cell).
    # BOTH width axes came back INTERIOR (nothing railed) => the <6us constraint is NOT binding, and
    # the reverse pulse genuinely wants SHORT pulses. Target-only verify-conditioned RETURN:
    #   pw556 marginal: 1.0 0.7158 | 1.5 0.7437 | 2.0 0.7709 | 3.0 0.7289 | 4.0 0.7091 | 5.75 0.7194
    #   pw308 marginal: 2.0 0.7255 | 3.0 0.7456 | 4.0 0.7466 | 5.0 0.7309 | 5.75 0.7081  (+-0.003)
    #   top cells: 2.0/2.0 0.8110 +-0.0055 | 1.5/2.0 0.8032 +-0.0058 | 2.0/3.0 0.8008 +-0.0059
    #              default corner 5.75/5.75 0.6959 +-0.0070
    # => ADOPTED 2.0/2.0: beats the 5.75/5.75 corner by +0.115 (~14 SEM). Plateau top is FLAT
    # (2.0/2.0 vs 1.5/2.0 = 1.0 SEM, vs 2.0/3.0 = 1.3 SEM) so this is a plateau-center pick, not an
    # argmax pick; stable across the 500-shot and 2542-shot reads (0.8121 -> 0.8110).
    # ANOMALY (open): 3.0/2.0 reads 0.6126 +-0.0071, ~10 SEM BELOW both neighbours (2.0/2.0 0.811,
    # 4.0/2.0 0.684) -- a single non-monotone dip in an otherwise smooth map. Not explained; re-read
    # it before ever locking a config near pw556 3.0.
    PW556_CH2_REV = [1.0, 1.5, 2.0, 3.0, 4.0, 5.75]   # (width 2-D, done)
    PW308_CH2_REV = [2.0, 3.0, 4.0, 5.0, 5.75]        # (width 2-D, done)
    # 2026-08-06 REVERSE ROUND 2 -- re-locate the reverse DELAY at the new 2.0/2.0 widths. Mandatory
    # because the pinned +0.3us came from a map taken at pw556 4-5us (and, before the pairing fix, was
    # read off a mis-attributed marginal): the 556<->308 overlap that sets adiabatic return scales with
    # the pulse widths, so a delay located at 4-5us cannot be assumed at 2us. Both widths PINNED at the
    # round-1 optimum; delay is the only swept axis (dim 1), spanning the corrected live window
    # (-0.3..+1.2 plateau, dead below -0.6) with a couple of points past each edge to prove interiority.
    # ROUND 2 RESULT (data_20260806_193652, 10 pts x ~15 shots, ~4.3k mid-events/pt). The delay DID
    # move, which is why this round was mandatory: best is now -0.50us (0.8298 +-0.0058), NOT the
    # +0.30 round-1 pin (which re-reads ~0.79 here, interpolating between +0.25 and 0.00).
    #   -0.75 0.6638 | -0.50 0.8298 | -0.25 0.8256 | 0.00 0.8114 | +0.25 0.7743 | +0.50 0.6752
    #   +0.75 0.6652 | +1.00 0.6567 | +1.25 0.6673 | +1.50 0.6299   (+-0.006..0.008 each)
    # Shape: a single ~0.75us-wide window (-0.5..0.0 all >= 0.81) sitting on a ~0.66 floor. The old
    # broad "-0.3..+1.2 plateau" was a WIDTH-DEPENDENT artifact -- at 2us pulses the window is much
    # narrower and shifted negative (556 firing first), consistent with overlap scaling with width.
    # NOT YET INTERIOR on the negative side: -0.50 (0.8298) -> -0.75 (0.6638) is a 0.166 drop in ONE
    # 0.25us step, i.e. the grid is too coarse to say where the cliff is or whether the true peak sits
    # between -0.75 and -0.50. ROUND 3 refines 0.05-0.10us steps across -0.70..-0.15 to place the peak
    # and prove it is interior (runbook: inspect the argmin POSITION, do not lock an edge).
    REV_DELAY_R2_US = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]   # (round 2, done)
    REV_DELAY_R3_US = [-0.70, -0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15]
    # 2026-08-06 QUADRUPLE_SPACING ROUND 2: 556 Ch2 width on dim 1 (paired with the Ch2 carrier on
    # dim 2 -- see the carrier block above for why they are scanned together). The 308 Ch2 width stays
    # pinned at 2.0 this round so the 2-D is a clean plane in the 556 knobs; re-check it after.
    # LOCKED 2.0 (2026-08-06 Q2, data_20260806_213036). Chose the MARGINAL peak (2.0 = 0.8458
    # +-0.0061, pooled over all 5 carriers, ~3.5k mid-events) over the single best CELL
    # (pw556 1.5 / car 143.50 = 0.8953 +-0.0114, ~0.7k events): the two widths are 0.9 SEM apart
    # in the marginal (1.5 = 0.8381), i.e. statistically tied, and the marginal carries ~5x the
    # statistics -- the runbook's flat-plateau lesson says do not lock a max-of-25 cell. 2.0 also
    # matches the independently-located double_spacing optimum, so it is the same physics on both
    # arrays. If a fixed-point verify is ever run, 1.5 vs 2.0 at 143.5 is the pair to settle.
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0   # 2026-08-06 Q2 width optimum (marginal peak)
    #g().AWG.AWG556.Ch2.pulse_width_us.scan(1, PW556_CH2_Q2)   # Q2 width x carrier 2-D (done)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15   # 2026-07-14 raised 11->15 (more STIRAP power)
    # 2026-08-06 REVERSE ROUND 4 -- 556 Ch2 AMPLITUDE scan (user directive). Why it is worth redoing:
    # the standing 0.9 comes from a PRE-power-raise scan whose note says "monotonic to ceiling,
    # best=1.0, still power-limited", i.e. it measured a POWER CEILING, not an optimum -- and since
    # then (a) the user raised the 556 optical power, and (b) the reverse pulse moved to 2.0us widths
    # at -0.5us delay. Peak drive = max_amplitude_vpp * amplitude_scale (pulse_waveform.py), so at
    # 2us the pulse AREA per unit amplitude is ~3x smaller than at the old 6us -> the amplitude that
    # completes the adiabatic return has almost certainly moved UP, and an interior optimum can now
    # exist where before everything railed.
    # Scanned DOWN as well as up: with more light, over-driving is a real failure mode for adiabatic
    # transfer (too-fast Rabi = non-adiabatic), so the grid must be able to show a turnover, not just
    # a ceiling. 1.0 is the hardware ceiling at vpp15 (amplitude_scale is clamped 0-1).
    # SAFE for the forward pulse: AWGManager keys waveforms per CHANNEL (C1/C2) and bakes
    # amplitude_scale into the Ch2 waveform while the device vpp stays fixed, so sweeping Ch2 does
    # NOT perturb the locked forward Ch1 pulse (verified in awg_manager.py).
    AMP556_CH2_REV = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9 #.scan(1, AMP556_CH2_REV)   # 2026-08-06 reverse amp round
    # g().AWG.AWG556.Ch2.amplitude_scale = 0.9   # pre-power-raise pin (railed to ceiling, stale)
    g().AWG.AWG556.Ch2.pad_time_us = 0.0
    
    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 4 #4.7   # 2026-08-03 round 4 optimum (locked; was 5.7, the 07-21 value)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8   # 2026-07-15 raised 5.5->7.5 (amp saturated by 7.5)
    g().AWG.AWG308.Ch1.amplitude_scale = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2
    
    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    # 2026-08-06 reverse width 2-D, dim 2 (INDEPENDENT of the 556 Ch2 width -- the asymmetric-vs-equal
    # question re-asked inside the <6us box). NOTE the prior rounds' "pw308 rails at 7" result is now
    # OUT OF SCOPE by the user's <6us constraint, so it is not inherited: 7 (and the >7 cells in
    # _110200, which were always invalid) are simply not in the allowed set.
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0   # 2026-08-06 reverse width-2D optimum (was .scan dim 2)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1


    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; kept for the future, currently unused) ---
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these; from STIRAPAWGScan) -----
    # 2026-07-21 mj=0 QUADRUPLE_SPACING forward-STIRAP OPTIMUM (campaign: delay pre-scan -> freq-2D ->
    # ridge-3D -> 100-shot verify). Metric = TARGET-ONLY + mid-conditioned survival (see stirap-optimization
    # runbook Rule 1). Verify (data 20260721_190200, 100 shots, 80 target sites): 4.56 +/- 0.56% survival =
    # 95.44% excitation. Pair 143.567/282.067 (freq-2D 181013); pw556 6.0/pw308 5.7/delay +1.0us (ridge 182536).
    # 2026-08-03 ROUND 1 -- re-pin the two-photon resonance. Today's daily 616-revival fit put the 308
    # line at 233.5555 MHz (data_20260803_103159, FWHM 17.0 MHz, R^2 0.988) while this config is parked
    # at 234.1 -> 0.545 MHz off. Sweep EOM616 across the line at the locked carrier 143.5 and the locked
    # forward pulse (pw556 6.0 / pw308 5.7 / delay +0.6us). 1-D across the DEGENERATE line (moving EOM
    # alone crosses it) is far cheaper than a Cartesian 10x10, most of which sits off-line.
    # ROUND 1 RESULT (data_20260803_111040, 23 pts x 6, target-only mid-conditioned, 285 target sites
    # @ 0.949 fill): clean resonance, best EOM616 = 234.2 (surv 0.0673 +-0.0057 = 93.27% exc), with
    # 234.1 TIED at 0.0685 +-0.0054 and 234.3 at 0.0879. Transfer window ~0.9 MHz (surv<0.5 from
    # ~233.75 to ~234.65). So the locked 234.1 was ALREADY on resonance -- the revival peak
    # (233.5555 MHz, 17 MHz wide) is NOT a proxy for the STIRAP two-photon pairing. Re-pinned to 234.2.
    # EOM616_MHZ = [round(233.0 + 0.1 * k, 4) for k in range(23)]     # 233.0 .. 235.2 MHz
    # g().Init.EOM616.Freq.scan(1, [f * 1e6 for f in EOM616_MHZ])
    g().Init.EOM616.Freq = 234.2e6   # PAIR w/ Ch1 143.5 (offset +90.7, round 5 confirmed; degenerate line runs +1:+1)
    g().Pushout.VRydTrap = 1.9 #.scan(1, np.linspace(0.2, 2.5, 10)) #= 1.9
    g().Pushout.BiasCoilCurrent.Ryd = 30

    # ---- post-rearrangement recool (RearrangeCool556hXStep, runs immediately before the pushout) ----
    # Own config block (RearrangeCool556) so it is decoupled from the RNR-tuned Cool556. Defaults come
    # from the ByPattern overlay (33x33_feedback11: Time 5 ms, X/h det 0.12 MHz, amp 0.14) -- leave these
    # commented to run the overlay values; uncomment a pin to override, or a .scan() to sweep.
    #
    # WHY this is the interesting axis (2026-08-06): the recool was the dominant survival limiter.
    # Cool556Step (both beams @ top-level 0.14 MHz / 0.08) -> Cool556hXStep (per-beam 0.12 MHz / 0.14)
    # lifted mid->final 0.341 -> 0.514, uniformly across all gaps (+6..+11 sigma each), i.e. a better
    # initial TEMPERATURE rather than a changed loss rate. A residual remains: at matched total trap-off
    # (~13 us) STIRAP sits at 0.783 vs RNR's 0.943, and the decay is still ~3x faster (tau 13.6 vs 42 us).
    # The STIRAP atoms arrive hotter than the RNR ones -- SLM transport with RearrCoolAmp = 0, plus two
    # 399 exposures (img1 + mid) before the pushout -- so the open hypothesis is that 5 ms at the
    # RNR-tuned amplitude does not fully re-thermalize them. Time is the first knob to try (monotonic
    # expectation: survival climbs then plateaus when the recool saturates); amp/det second.
    #
    # g().RearrangeCool556.Time = 5e-3
    # g().RearrangeCool556.Time.scan(1, np.array([1, 2, 3, 5, 8, 12, 20, 30]) * 1e-3)   # recool-duration sweep
    # g().RearrangeCool556.X.FreqDetuning = 0.12e6
    # g().RearrangeCool556.X.Amp = 0.14
    # g().RearrangeCool556.h.FreqDetuning = 0.12e6
    # g().RearrangeCool556.h.Amp = 0.14
    
    # 2-D per-beam grid (det x amp) on ONE beam, the other pinned -- mirrors the RNR/imaging campaigns:
    # g().RearrangeCool556.X.FreqDetuning.scan(1, np.linspace(0.08e6, 0.24e6, 9))
    # g().RearrangeCool556.X.Amp.scan(2, np.linspace(0.08, 0.24, 9))
    # 2026-08-06 POWER RE-WALK ROUND 1: delay on scan dim 2 (paired with amplitude on dim 1; pw556
    # fixed at 5.0). The grid spans the round-4 lock (0.4) and extends both ways -- the 08-03 delay
    # scan showed transfer only for POSITIVE delay with a broad 0.4-1.6 window, so 0.0-1.0 covers
    # the live region without spending shots on the dead negative side.
    #g().Pushout.STIRAPDelay.scan(2, [d * 1e-6 for d in DELAY_PWR_US])
    # g().Pushout.STIRAPDelay = DELAY_US * 1e-6   # 2026-08-03 round 4 optimum (locked; was 0.6e-6)
    g().Pushout.STIRAPDelay = 1.8e-6 #.scan(2, np.linspace(0.5e-6, 2e-6, 10))   # 2026-08-03 round 4 optimum (locked; was 0.6e-6)
    # 2026-07-30 step-0 delay pre-scan (data_20260730_162309): transfer only for POSITIVE delay, best +0.6us (surv 0.151), broad 0.4-1.6us; <=0 dead (~0.97)
    # 2026-08-03 REVERSE R1 -- re-locate the reverse delay. Two reasons it must move: (a) the 07-31
    # note's own TODO (the delay optimum was found with Ch2 widths 2/2 and pw308_Ch2 is now 7us), and
    # (b) the FORWARD pulse changed this morning (pw308 5.7->4.7, delay 0.6->0.4, excitation 93.7->95.8%),
    # so the Rydberg population the reverse acts on is different. Metric flips: REVERSE wants HIGH
    # survival (atoms returned). Reverse-OFF floor at today's forward = 0.0420 +-0.0022 (r4 verify).
    # R1 RESULT (data_20260803_123528, 15 pts x 10, Ch2 5.0/5.0): broad plateau -0.5..+0.5, best
    # +0.25us = 0.7677 +-0.0125 (return; -0.25/-0.50/+0.50 tied within ~1 SEM), falls off outside
    # +-0.75. Ch2-carrier scan (data_20260803_124313) came back FLAT over 143.0-144.0 -> keep 143.5.
    # 2026-08-03 REVERSE 2-D (user directive): Ch2 width (dim 1, co-varied 556+308) x reverse delay
    # (dim 2), 6 x 13 = 78 combos. Delay range narrowed to +-1.5 (R1 showed <-1.5 is dead).
    REV_DELAY_US = [round(-1.5 + 0.25 * k, 4) for k in range(13)]     # -1.50 .. +1.50 us (equal-width 2-D)
    #g().Pushout.STIRAPReverseDelay.scan(1, np.linspace(-1.5e-6, 1.5e-6, 13))   # 2026-08-03 REVERSE 2-D (user directive)
    # 2026-08-06 REVERSE ROUND 1: delay PINNED (both scan dims are now the two Ch2 widths). +0.3us is
    # the top of today's pooled delay marginal (0.781 +-0.010, data_20260806_190541) and sits mid-
    # plateau (-0.3..+1.2 all within ~2 SEM), so it is the robust hold-point rather than a noise pick.
    # Re-check the delay AT the winning width pair next round -- the optimal 556/308 overlap moves with
    # the widths, which is exactly why the 07-31 delay went stale (its own TODO).
    # 2026-08-06 ROUND 2: delay is now the ONLY swept axis, at the 2.0/2.0 width optimum (see the
    # REV_DELAY_R2_US rationale in the AWG556.Ch2 block). Round 1 ran pinned at +0.3e-6.
    # 2026-08-06 ROUND 3: fine delay across the negative cliff at the 2.0/2.0 widths (see the
    # REV_DELAY_R3_US rationale above -- round 2's 0.25us step left a 0.166 jump unresolved).
    # 2026-08-06 ROUND 4 (amp scan): delay PINNED at the round-2 best -0.5us. Round 3's fine delay
    # walk (-0.70..-0.15, id 286) may nudge this by <=0.1us; that is inside the flat top of the
    # window (-0.5 and -0.25 read 0.8298 vs 0.8256, 0.5 SEM apart), so it does not bias the amp
    # comparison. RE-CHECK the delay at the winning amplitude afterwards -- amp changes the Rabi
    # frequency and therefore the adiabatic overlap, same coupling that moved the delay when the
    # widths changed.
    # 2026-08-06 QUADRUPLE_SPACING REVERSE ROUND 1 -- re-locate the reverse DELAY on the new array.
    # The 2.0/2.0 widths and the delay were all located on the DOUBLE_SPACING array; the array just
    # changed to quadruple_spacing (rearrange_kwargs.extras.pattern), which changes the site pitch
    # and therefore how the 556/308 beams illuminate the target footprint, so none of it is assumed
    # to carry over. Delay is re-located FIRST because it was by far the strongest axis on the old
    # array (0.59 -> 0.83 across the scan, vs 0.06 for amplitude).
    # RANGE: the corrected double-spacing curve (data_20260806_195643, manifest-paired) rose
    # monotonically to the -0.15 edge and was STILL CLIMBING there:
    #   -0.70 0.6254 | -0.60 0.5853 | -0.50 0.6578 | -0.40 0.7588 | -0.30 0.8101 | -0.25 0.8309
    #   -0.20 0.8281 | -0.15 0.8328  (+-0.003 each)
    # i.e. the old scan never bracketed its own peak. So this grid starts at -0.30 and runs PAST
    # zero to +0.45, which brackets the peak from both sides on the new array (the coarse round-2
    # scan had 0.00 = 0.8193 and +0.25 = 0.7662, so the turnover should sit inside this span).
    # Q1 RESULT (data_20260806_211302, 8 pts x 40 shots, manifest-paired -- the manifest reproduced
    # the corrected reconstruction exactly). Target-only verify-conditioned RETURN (HIGH = good):
    #   -0.30 0.8580 | -0.20 0.8727 | -0.10 0.8892 | 0.00 0.8810 | +0.10 0.8826 | +0.20 0.8774
    #   +0.30 0.8694 | +0.45 0.8444   (+-0.007 each)
    # PEAK PROPERLY BRACKETED (rises from -0.30, tops at -0.10, falls to +0.45) -- an INTERIOR
    # optimum, which the double-spacing campaign never achieved. Top is flat: -0.10 .. +0.20 span
    # ~0.5 SEM, so -0.10 is a plateau-centre pick. Reverse-OFF floor on this array = 0.0381 survival
    # (96.19% forward excitation, data_20260806_205429), so the reverse returns ~88.9% of atoms.
    # ADOPTED -0.10us; re-check it once the width/carrier below move (overlap scales with width).
    REV_DELAY_Q1_US = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.45]   # (Q1 delay, done)
    g().Pushout.STIRAPReverseDelay = -0.10e-6   # 2026-08-06 quadruple_spacing Q1 delay optimum
    #g().Pushout.STIRAPReverseDelay.scan(1, [d * 1e-6 for d in REV_DELAY_Q1_US])   # Q1 (done)
    #g().Pushout.STIRAPReverseDelay = -0.5e-6   # double-spacing round-4 pin
    #g().Pushout.STIRAPReverseDelay.scan(1, [d * 1e-6 for d in REV_DELAY_R3_US])   # round 3 (id 286)
    #g().Pushout.STIRAPReverseDelay.scan(1, [d * 1e-6 for d in REV_DELAY_R2_US])   # round 2 (done)
    #g().Pushout.STIRAPReverseDelay = 0.3e-6   # 2026-08-06 round-1 pin (width 2-D)
    #g().Pushout.STIRAPReverseDelay = -0.25e-6   # 2026-07-31 round 3 best (data_20260731_111148: 1.5us Ch2 @ -0.25us -> 0.876 +-0.009); irrelevant for the reverse-OFF control: re-check the delay AT the new widths -- the 0 optimum (data_20260731_105045) was located with pw556/pw308 Ch2 = 2/2, and pw308 is now 7us, so the optimal overlap has likely moved
    g().Pushout.STIRAPPadTime = 2e-6   # 2026-07-21 mj=0 quad ridge-3D optimum (308-first; window +0.6..+1.4us)
    g().Pushout.STIRAPGap = 1e-6 #.scan(2, np.linspace(0e-6, 20e-6, 10))  # short fixed hold (forward optimum). For a Rydberg-lifetime sweep: .scan(1, gap_pts)
    # 2026-08-06 REVERSE ON for the quadruple_spacing reverse campaign. Forward baseline measured on
    # THIS array immediately before, reverse OFF, 206 shots (data_20260806_205429): survival
    # 0.0381 +-0.0021 = 96.19% +-0.21% excitation, 80 target sites, mid-fill 0.989. That survival is
    # the reverse-OFF FLOOR the reverse return has to beat -- with the reverse pulse on, atoms that
    # were excited and brought back are RETAINED, so the metric flips to HIGH = good.
    g().Pushout.IfReverse = 0
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6   # mj=0 pump616 (pumps OFF this config)
    g().Pushout.Pump556Freq = 143.556e6   # mj=0 pump556 (pumps OFF this config)
    g().Pushout.Pump556Amp = 0.5 # 2026-07-16 pump556 amp sweep (data_20260716_210005): best ~0.8-1.0 (monotonic to ceiling)
    g().Pushout.SLMAOMAmpGap = 0.55
    g().Pushout.IfGatePulses = 1  # 1 means we are not skipping Rydberg pulses

    # g().Pushout.Amp369 = 1
    g().Pushout.IonizationViaDAC = 0 #1: Ramp DAC to ionize. 0: Use TTL switch to ionize
    g().Pushout.TimeIonization = 0.1e-6  # The ionization voltage is set during the InitStep, and switched by the TTL
    g().Init.VIonizationSet5to8 = 4  # The ionization voltage is set during the InitStep, and switched by the TTL

    # Test with RNR Step replacing the STIRAP step -- DISABLED 2026-08-06 for the STIRAP power
    # re-walk: with scienceStep="stirap" the RnR step never runs, and leaving this .scan() on dim 1
    # would collide with the pw556 sweep (two params on the same dim = an unintended co-vary).
    #g().ReleaseRecapture.Time = 1e-6
    #g().ReleaseRecapture.Time.scan(1, np.linspace(0.5e-6, 20e-6, 20))   # 2026-07-16 RNR lifetime sweep (data_20260716_210005): best ~0.8-1.0 (monotonic to ceiling)
    
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
    
    g().rearrange_kwargs.extras.wgs_warm = True # We are using the warm WGS for rearrangement
    g().rearrange_kwargs.extras.wgs_pad = 2048
    g().rearrange_kwargs.extras.wgs_iters = 3

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) ------------------------------
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "octuple_spacing" #"quadruple_no_topright"  # 2026-07-10 quadruple_spacing (post optics move) -- the new default for the 33x33 array
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN
    # 2026-08-06: back to STIRAP for the post-power-raise forward re-walk (was defaulting to "rnr"
    # for the RnR A/B, which is what data_20260806_115520 / _121753 ran).
    g().rearrange_kwargs.extras.scienceStep = "stirap"

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = 2000
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan
    # start. MATCHED to rearrange_kwargs.extras.z4 (the rearrange MODEL z4).
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1   # ON per runbook (decorrelates drift). Set to 0 for ANY scan that sweeps EOM616 (scramble + 616-EOM sweep risks a 616 unlock). -- scramble + 616-EOM sweep risks a 616 unlock. (decorrelates drift). Set to 0 for ANY scan that sweeps EOM616 -- scramble + 616-EOM sweep risks a 616 unlock. (decorrelate drift). NOTE: scrambling the EOM616 sweep risked a 616 unlock in the Stark scans; watch the lock -- revert to 0 if it drops.
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    # Per-frame detection declaration (explicit -> wins over the runner's 2-frame synthesis).
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeSTIRAPScan(url=None, reps=10):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan", **opts)
    print("submitted RearrangeSTIRAPScan -> descriptor id %s (url=%s, reps=%s, verify=%s, "
          "NumImages=%d)" % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeSTIRAPScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
