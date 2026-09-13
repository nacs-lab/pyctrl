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
    g().AWG.AWG556.Ch1.shape = "rise_quintic" 
    #CARRIER_LIST_MHZ = [round(float(v), 4) for v in np.linspace(130.45, 133.45, 10)]  # scan dim 1 for low field
    # 2026-08-18 round 2, RECENTRED ON THE MEASURED SINGLE-PHOTON RESONANCE. The STIRAP optimum
    # sits near the individual resonances measured at shallow trap depth, so anchor the carrier to
    # the 60 G 556 Rydberg spectrum rather than extrapolating the two-photon ridge: RydbergSpectrum556Scan_60G
    # data_20260818_174135 fits centre = 118.9242 +- 0.0005 MHz (FWHM 0.505, R^2 0.964), stable all
    # day (118.907 / 118.911 / 118.909 at 08:13 / 13:59 / 14:19). Round 1 (118.1-118.9,
    # data_20260818_175154) put that resonance ON THE TOP EDGE, so its map climbed monotonically to
    # the corner and the "best" cell 118.90/230.50 was a boundary value, not a maximum.
    # EOM616 centre from the ridge at the anchor: 230.60 (round-1 interior rows) / 230.59 (11:43
    # map) -- the two agree well inside a linewidth, so centre the 616 span on 230.55.
    # 2026-08-26 FREQ-2D ROUND 6 -- on the RE-RUN 60 G daily Rydberg calibration (jobs 1365/1366/
    # 1367, 22:40-22:49).  Target double_spacing (20 um, ~284 targets).
    #
    # BRANCH VERIFIED: RearrangeSTIRAPSeq dispatches on Pushout.BiasCoilCurrent.Ryd -- <31 G ->
    # STIRAPPushoutStep, 50-80 G -> STIRAPHighFieldPushoutStep, else raise.  Ryd = 60 here, so we
    # are on the high-field step (556 through the second single-pass AOM, +120 MHz static switch,
    # low-field shutter closed, both AOMs switched DDS -> AWG).  UNITS: the DDS scans subtract
    # HF_AOM_OFFSET_MHZ = 60.0 internally, so their logged frequencies are already in DOUBLE-PASS
    # AOM units -- the same AOM the AWG drives under STIRAP, and the frame these values are in.
    #
    # WHY THE EARLIER ROUNDS WERE EMPTY.  The 16:30-16:55 calibration was internally inconsistent:
    # its AT midpoint (118.508) sat 464 kHz BELOW the bare 556 line (118.972), i.e. the 308 was
    # DETUNED because the 616 revival had been fit at 229.6 right after a 616 relock, on a broad
    # (22.7 MHz FWHM) window-sensitive peak.  The re-run set:
    #   556 dip     118.9865 MHz  (FWHM 127.0 kHz, R^2 0.990, 165 sh)   job 1365 / 20260826224042
    #   616 revival 230.6907 MHz  (FWHM 15.9 MHz,  R^2 0.985, 161 sh)   job 1366 / 20260826224445
    #   AT dips     118.2279 / 119.9111, splitting 1.6832 +- 0.0081 MHz, midpoint 119.0695
    #               (hand-seeded double, R^2 0.814, 305 sh)             job 1367 / 20260826224855
    # The revival moved +1.09 MHz (onto the 08-18 series value 230.6) and the AT midpoint now sits
    # only +83 kHz from the bare line -- symmetric, i.e. the 308 IS on two-photon resonance.
    # CAVEAT: round 1's box nominally contained (119.07, 230.69) and still read flat, so the 616
    # was probably drifting through the evening; treat the re-run cal as the anchor, not round 1.
    #
    # Round 6 window -- brackets the whole re-fit AT structure, biased UPWARD (user: "freq is
    # upper"; the full-power STIRAP pulse splits harder than the 0.08-amp probe, so the working
    # point is expected at or above the upper dressed dip):
    #   carrier 118.20-121.00 -- lower dip 118.2279, bare line 118.9865, midpoint 119.0695, upper
    #     dip 119.9111, plus 1.1 MHz above the upper dip.  0.2 MHz step vs the ~0.33 MHz AT dip
    #     FWHM (broader still under the STIRAP pulse).
    #   EOM616 229.20-232.20 -- +-1.5 MHz about the re-fit revival 230.6907, inside the mj = -1 pi
    #     band.  At the 08-19 ridge slope ~0.814 that 3 MHz EOM span drags the ridge ~2.4 MHz in
    #     carrier, so the 2.8 MHz carrier span lets the ridge cross the box diagonally.
    # 2026-09-07 ROUND 8 -- SAME DESIGN as round 7, re-anchored on TODAY's 60 G daily cal.
    # The whole 60 G structure moved DOWN vs 08-27: revival 230.6907 -> 230.0066 (-684 kHz),
    # bare 556 line 118.9865 -> 119.0253, AT dips 118.2279/119.9111 -> 118.1110/119.6687
    # (splitting 1.6832 -> 1.5577), AT midpoint 119.0695 -> 118.8899 (i.e. -135 kHz BELOW the
    # bare line today, vs +83 kHz above on 08-27 -- the same red-offset sign the 20/30 G runs
    # show, worth watching but not blocking). Windows shifted by exactly that:
    #   carrier 118.00-120.80 -- brackets today's lower dip 118.1110, bare line, midpoint and
    #     upper dip 119.6687, plus ~1.1 MHz above the upper dip (the round-7 "freq is upper"
    #     bias: the full-power pulse splits harder than the 0.08-amp probe).
    #   EOM616 228.50-231.50 -- +-1.5 MHz about today's revival 230.0066.
    # Also NOTE the imaging was re-optimized earlier today (per-site d' 4.4 -> 7.8, survival
    # 0.976 -> 0.995): the verify-conditioned excitation is imaging-fidelity-floored, so this
    # round reads against a materially better floor than the 08-27 rounds did.
    # ROUND 8b FINE 7x7 (campaign protocol: coarse -> fine 7x7 x 5 reps = 245 shots, optimum
    # must be interior). Centred on the coarse best cell (119.20, 230.50), survival 0.0493 +-
    # 0.0088, interior on both axes (data_20260907_181944). The coarse map showed a clean
    # diagonal ridge over the five transferring rows 229.5-231.5; SEM-weighted fit through their
    # parabolic-refined argmins gives carrier = 119.1777 + 0.8447*(EOM616 - 230.5000), in line
    # with the 08-19/08-27 slopes (0.814 / 0.926). The dead rows 228.5/229.0 are EXCLUDED from
    # that fit -- their argmins are noise/edge-railed and flip the fitted slope to -0.08.
    # Fine steps are half the coarse: carrier 0.1 (was 0.2), EOM616 0.125 (was 0.5); the ridge
    # crosses this box diagonally (0.75 MHz of EOM drags the carrier 0.63 MHz).
    #CARRIER_LIST_MHZ = [round(float(v), 4) for v in np.linspace(118.00, 120.80, 15)]  # R8 coarse, 0.2 step
    #EOM616_LIST_MHZ = [round(float(v), 4) for v in np.linspace(228.50, 231.50, 7)]    # R8 coarse, 0.5 step
    # ===== 2026-09-07 ROUND 8c: CO-VARY ALONG THE RIDGE (campaign step 2) =====
    # Both frequencies ride scan dim 1 as a matched PATH -- Rule 2 of the STIRAP runbook: the
    # lab analysis collapses/reorders a co-vary sweep, so the analysis MUST re-pair by the
    # logged 1-indexed ``Params`` combo id, never by run_analysis's summary ordering.
    # Why a path and not another box: R8b's fine 7x7 (data_20260907_184239) railed its optimum
    # to the EOM616 TOP EDGE (0.0271 +- 0.0060 at 119.40/230.875) -- the minimum runs ALONG the
    # ridge, so boxing it just re-rails. This walks the ridge itself over 1.5 MHz of EOM616.
    # Ridge slope: R8 coarse fit 0.8447, R8b fine fit 0.7273 -> 0.79 used here (their mean);
    # anchored on R8b's best cell (119.40, 230.875) rather than either fit's intercept, so the
    # path passes exactly through the deepest measured point.
    #   carrier = 119.40 + 0.79*(EOM616 - 230.875)
    # ===== 2026-09-07 every_other: RE-LOCK after the 08-27 locks failed verification =====
    # The user asked to VERIFY the existing locks first and only optimize if they no longer
    # held. They do not: at the 08-27 forward lock (118.3375, 229.6655, delay 1.35 us) today's
    # 200-shot forward+reverse verify (data_20260907_203539, IfReverse swept [0,1] in one scan)
    # read forward survival 0.1306 +- 0.0024 = excitation 86.94% (recorded 0.0836 +- 0.0046 =
    # 91.64%) and round-trip return 0.5467 +- 0.0050 (recorded ~0.92). Both target-restricted
    # via the 08-27 sibling's mask (536 sites, data_20260827_164918) since this run's own
    # slm_diag had not synced -- leftovers are negligible, so the loss is real, not metric bias.
    # The reverse degrades much harder than the forward, which is what a drifted two-photon
    # resonance does: the round trip must work TWICE and compounds the per-leg error.
    # Today's other geometries both moved (quadruple_spacing locked at EOM616 230.6808,
    # quad_10x10 at 228.7775), so this walks the ridge through the OLD lock to find the new one.
    # Slope 0.85 = today's fitted range (0.845 quadruple_spacing coarse, 0.786 / 0.955
    # quad_10x10 fine / coarse), anchored on the old lock so the path passes through it.
    #   carrier = 118.3375 + 0.85*(EOM616 - 229.6655)
    N_COVARY = 30
    EOM616_LIST_MHZ = [round(float(v), 4) for v in np.linspace(229.00, 230.80, N_COVARY)]
    CARRIER_LIST_MHZ = [round(118.3375 + 0.85 * (e - 229.6655), 4) for e in EOM616_LIST_MHZ]

    # 2026-08-19 TOP-N CO-VARY VERIFY (runbook step 4, top-N form): the freq-2D
    # (data_20260819_084955) mapped a clean diagonal ridge; its best 5 cells (5 shots each,
    # overlapping SEMs) are re-verified together at 100 reps as a co-vary path -- carrier AND
    # EOM616 both on dim 1 (Rule 2: analysis must re-pair by Params). Path ordered by EOM616
    # so the 616 EOM steps stay gentle (Scramble 0). Freq-2D cell survivals: 0.017 / 0.021 /
    # 0.022 / 0.021 / 0.004.
    # 2026-08-19 round 2: EXACT FITTED-LINE pairs. The top-5-cells verify (data_20260819_091118,
    # aborted at 117 shots per user -- enough) came back TIED, 0.028-0.042 across all five (the
    # freq-2D's 0.004 cell was 5-shot noise); best (118.974, 230.50) 0.0276 +- 0.0039. Ridge line
    # fit to the freq-2D per-row argmins over the deep rows 230.30-230.80:
    # carrier = 118.982 + 0.814*(EOM616 - 230.55); it predicts 118.97 at 230.50 = the measured best.
    # 2026-08-19 FORWARD LOCKED at the along-ridge floor (data_20260819_092526): the freq-2D
    # (084955) -> top-5 verify (091118) -> fitted-line verify (091704) -> 20-pt along-ridge chain
    # put the whole EOM 230.35-230.55 floor at survival 0.027-0.035 (tied); locked cell =
    # (118.8856, 230.4316), 0.0267 +- 0.0042 at 20 reps (tightest SEM interior point).
    # Ridge line for re-derivation: carrier = 118.982 + 0.814*(EOM616 - 230.55).
    # re-run 60 G anchors (for a later fixed-point verify): AT midpoint 119.0695 /
    # upper dressed dip 119.9111, both @ EOM616 230.6907
    g().AWG.AWG556.Ch1.carrier_freq_MHz.scan(1, [float(v) for v in CARRIER_LIST_MHZ])
    g().AWG.AWG556.Ch1.pulse_width_us = 3 #5.0   # 2026-08-06 FIXED per user (was 6.0, the 08-03 lock)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15   # 2026-07-16 (now HONORED by AWGManager channel-mode; was silently forced to consts default 15)
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87    # 2026-08-06 round 1: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    
    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 118.8856  # 2026-08-19 track the locked forward carrier (was 119.1)
    # 2026-08-19 REVERSE LOCKED. Delay pre-scan (data_20260819_095938): return peak 0.9348 at
    # -0.20 us (plateau -0.2..+0.3) -> delay kept. Width-2D 5x5 (data_20260819_100925): diagonal
    # plateau 0.918-0.936, 556-wide/308-short corner collapses (0.53). 100-rep top-N verify
    # (data_20260819_101841): (1.75,2.5) 0.9138 / (2.5,2.5) 0.9146 / (3.25,3.25) 0.9188 /
    # incumbent (2.0,2.0) 0.9232 +- 0.0038 -- ALL TIED, incumbent best -> widths KEPT 2.0/2.0.
    # Return floor ~0.92 at the locked forward pair.
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15   
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9 
    g().AWG.AWG556.Ch2.pad_time_us = 0.0
    
    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3 #4.7   # 2026-08-03 round 4 optimum (locked; was 5.7, the 07-21 value)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8   # 2026-07-15 raised 5.5->7.5 (amp saturated by 7.5)
    g().AWG.AWG308.Ch1.amplitude_scale = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2
    
    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0  # kept -- 08-19 verify: incumbent tied-best (see Ch2 556 block)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1


    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; kept for the future, currently unused) ---
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these; from STIRAPAWGScan) -----
    # forward lock 2026-08-19 was 230.4316e6 (paired with carrier 118.8856); swept here on dim 2
    # CO-VARY: EOM616 on dim 1 alongside the carrier (was dim 2 in the freq-2D parent), so the
    # two ride the ridge together as one path. The list is ordered by ASCENDING EOM616 so the
    # 616 EOM steps stay gentle even though Scramble is off.
    g().Init.EOM616.Freq.scan(1, [float(v) * 1e6 for v in EOM616_LIST_MHZ])
    g().Pushout.VRydTrap = 2.0 #.scan(1, np.linspace(0.2, 2.5, 10)) #= 1.9
    g().Pushout.BiasCoilCurrent.Ryd = 60 #30

    # ---- post-rearrangement recool (RearrangeCool556hXStep, runs immediately before the pushout) ----
    # 2026-08-18 17:45 on-ridge delay sweep (data_20260818_174505, post 556 power raise): peak
    # 0.904 +- 0.011 @ 1.222 us, plateau >= 0.885 all the way to 2.0 us -- the adiabatic window is
    # wide, so this is pinned mid-plateau while dim 1 carries the carrier sweep.
    g().Pushout.STIRAPDelay = 1.35e-6   # every_other lock (08-27); re-checked after the re-lock
    # 2026-08-19 REVERSE optimization round 1: delay pre-scan (runbook step-0 analog for the
    # reverse pulse) at the locked forward pair -- sweep the 556<->308 Ch2 order/overlap, maximize
    # verify-conditioned RETURN survival (forward up -> 1 us gap -> reverse down).
    g().Pushout.STIRAPReverseDelay = -0.2e-6  # confirmed optimal 2026-08-19 (r1 pre-scan, peak of the -0.2..+0.3 plateau)
    g().Pushout.STIRAPPadTime = 2e-6   # 2026-07-21 mj=0 quad ridge-3D optimum (308-first; window +0.6..+1.4us)
    # 2026-08-18 Rydberg-lifetime sweep: hold in the Rydberg state for STIRAPGap, then reverse-STIRAP
    # back down -> RETURN vs gap is the decay curve. Dense 1-100 us (5.2 us step) to resolve the fast
    # decay + a coarse 120-200 us tail to pin the asymptote.

    g().Pushout.STIRAPGap = 1e-6   # short fixed hold -- forward-only freq-2D (no lifetime axis)

    g().Pushout.IfReverse = 0  # FORWARD optimization: no de-excitation, minimize survival
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
    g().Pushout.TIonizationAlign = 0.5e-6  # wait before ionization to align the trap with the ionization pulse
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
    # 2026-08-26 per user, ROUND 7: QUADRUPLE_SPACING (40 um, ~82 targets).  Walking the array
    # progressively sparser to get OUT of the interaction-limited regime while hunting the ridge:
    # every_other 14.5 um (536t, R1-R3) -> double_spacing 20 um (284t, R4-R6) -> quadruple_spacing
    # 40 um (82t, here).  40 um is the least-vdW condition on record with a real forward number
    # (30 G quad_ntr: 95.67 +- 0.32, tau_Ryd 72.2 +- 4.8 us) and at 60 G the 40 um quad reverse
    # tops 88.47 +- 1.55 -- so if a ridge exists at these frequencies it should be most visible
    # here.  Series naming: double_spacing (284t) / quadruple_spacing (82t) / eightfold_spacing
    # (24t); all are slm/patterns/*.txt bitmaps, NOT _GENERATED_PATTERNS.  NOTE the statistics
    # trade: 82 targets x 5 reps ~ 410 mid events per cell (vs ~1400 at double_spacing) -- still
    # ample for FINDING a deep dip, but per-cell SEM roughly doubles.
    g().rearrange_kwargs.extras.pattern = "every_other"   # 2026-09-07 (this file's own geometry)
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
    rp.NumPerGroup = 390  # = rep x n_points (13 x 30, R8c co-vary) so the dashboard total matches the real cap
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan
    # start. MATCHED to rearrange_kwargs.extras.z4 (the rearrange MODEL z4).
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    # 0 because dim 2 sweeps Init.EOM616.Freq: scrambling makes consecutive shots jump the 616 EOM
    # by large random amounts, which kicks the 616 laser out of lock (aborts at seq ~5-7, flat
    # revival). A monotonic EOM sweep steps it gently and the lock holds. Set back to 1 for any
    # scan that does NOT sweep EOM616 (it decorrelates slow drift from the swept axis).
    rp.Scramble = 0  # dim 2 sweeps Init.EOM616.Freq -- scramble + 616-EOM sweep risks a 616 unlock
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
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan_fwd_covary_60G_quad",
                      description=(
                          "FORWARD STIRAP freq-2D ROUND 8 at 60 G on quadruple_spacing (40 um, ~82 "
                          "targets; sparsest array, least vdW) -- a RE-OPTIMIZATION of the 08-27 "
                          "round-7 lock, prompted by today's cal showing the whole 60 G structure "
                          "has moved down: 616 revival 230.6907 -> 230.0066 (-684 kHz, scan "
                          "20260907164817, FWHM 17.0 MHz, R^2 0.986), 556 60 G dip 118.9865 -> "
                          "119.0253 (scan 20260907152259, FWHM 116.7 kHz, R^2 0.995), AT dips "
                          "118.2279/119.9111 -> 118.1110/119.6687, splitting 1.6832 -> 1.5577 "
                          "+- 0.0165, midpoint 119.0695 -> 118.8899 (scan 20260907170225, "
                          "hand-seeded 2-peak R^2 0.907). Window shifted by exactly that: carrier "
                          "118.00-120.80 (dim 1, 15 pts, 0.2 step) bracketing today's whole AT "
                          "structure plus ~1.1 MHz above the upper dip; EOM616 228.50-231.50 "
                          "(dim 2, 7 pts, 0.5 step) about today's revival. 105 cells x 5 reps = 525 "
                          "shots. Delay 1.333 us (adopted from the every_other 60 G lock; will be "
                          "re-scanned in the delay step), widths 3/3, amps 0.87/1, IfReverse=0. "
                          "ALSO note the imaging was re-optimized earlier today (per-site d' 4.4 -> "
                          "7.8, survival 0.976 -> 0.995, PIDSet 1.6/1.0, Imag399 det -5, Cool556 "
                          "X(0.18,0.25)/h(0.15,0.21)), and the verify-conditioned excitation is "
                          "imaging-fidelity-floored, so this reads against a better floor than "
                          "round 7 did. Metric = verify-conditioned TARGET-only survival "
                          "(minimize); group by Params. Scramble 0 (EOM616 swept)."),
                      **opts)
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
