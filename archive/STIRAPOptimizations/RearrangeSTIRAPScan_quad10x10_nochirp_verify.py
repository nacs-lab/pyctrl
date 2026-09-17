"""RearrangeSTIRAPScan_quad10x10_nochirp_verify.py -- RE-RUN of the 2026-08-30 no-chirp
BASELINE VERIFY on quad_10x10 (job 1605, data_20260830_183152, S = 0.4351 +- 0.0027).

A byte-for-byte replay of that fixed point: chirp_freq_MHz = 0 scalar, Init.EOM616 = 228.830 MHz
scalar, AWG556 Ch1 carrier = 118.150 MHz, pattern quad_10x10, no swept params, --reps 100.
A separate file so the in-place ladder_spacing config in RearrangeSTIRAPScan.py is not clobbered.
The built params are asserted identical to 183152's logged ScanGroup base before submitting.

STIRAP push-out survival on a REARRANGED array.

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
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 118.1500   # quad_10x10 lock (data_20260830_183152)
    g().AWG.AWG556.Ch1.pulse_width_us = 3
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch1.amplitude_scale = 0.87  # 08-06 amp scan monotonic to ceiling (power-limited)

    g().AWG.AWG556.Ch2.shape = "fall_quintic"
    # dimer_40um reverse: sweep data_20260828_054718 was pure noise (~22 events/pt, no
    # peak) -- reverse params ADOPTED from the clean geometries (offset +0.09; delay
    # mid-range), NOT measured on this pattern.
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 118.5428
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0  # widths 2.0/2.0 kept (08-19 top-N verify tied-best)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15
    g().AWG.AWG556.Ch2.amplitude_scale = 0.9
    g().AWG.AWG556.Ch2.pad_time_us = 0.0

    g().AWG.AWG308.Ch1.shape = "chirped_fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.chirp_freq_MHz = 0.0   # NO chirp (the incumbent); byte-identical to fall_quintic
    g().AWG.AWG308.Ch1.chirp_profile = "quintic"
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
    g().Init.EOM616.Freq = 228.83e6   # quad_10x10 stored 616 lock
    g().Pushout.EOM616.Freq.Final = 229.6655e6  # UNUSED (chirp disabled in step)

    g().Pushout.VRydTrap = 2.0
    g().Pushout.BiasCoilCurrent.Ryd = 60

    g().Pushout.STIRAPDelay = 1.3500e-6
    g().Pushout.STIRAPReverseDelay = 0.0000e-6
    g().Pushout.STIRAPPadTime = 2e-6  # 07-21 mj=0 quad ridge-3D optimum
    # lifetime: forward -> hold STIRAPGap -> reverse; RETURN vs gap = decay curve.
    GAP_PTS = np.geomspace(0.1e-6, 500e-6, 30)
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
    g().rearrange_kwargs.extras.pattern = "quad_10x10"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN
    g().rearrange_kwargs.extras.scienceStep = "stirap"  # "rnr" = RnR alternative

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = 30  # chain
    rp.loading_defocus = -5  # ANSI z4 (rad) on the loading phase; MATCH rearrange_kwargs.extras.z4
    rp.NumImages = 3 if verify else 2
    # MUST be 0 whenever EOM616 is swept (random EOM jumps kick the 616 out of lock); 1 otherwise.
    rp.Scramble = 1  # chain
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    # Per-frame detection declaration (explicit -> wins over the runner's 2-frame synthesis).
    rp.imagePatternsJson = _image_patterns_json(verify, init_cfg, target_cfg)

    return g


def RearrangeSTIRAPScan(url=None, reps=100):
    """Build + SUBMIT the scan to the running pyctrl backend. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="RearrangeSTIRAPScan_quad10x10_nochirp_verify",
                      description=(
                          "quad_10x10 no-chirp baseline RE-RUN of data_20260830_183152: "
                          "chirp_freq_MHz = 0, no swept params, 556 = 118.1500 MHz, "
                          "EOM616 = 228.8300 MHz, %s reps." % reps),
                      **opts)
    print("submitted RearrangeSTIRAPScan_quad10x10_nochirp_verify -> descriptor id %s (url=%s, reps=%s, verify=%s, "
          "NumImages=%d)" % (did, url or "default", reps, VERIFY_IMAGE, 3 if VERIFY_IMAGE else 2))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit RearrangeSTIRAPScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=100,
                    help="passes over the sweep (0 = forever)")
    args = ap.parse_args()
    RearrangeSTIRAPScan(url=args.url, reps=args.reps)
