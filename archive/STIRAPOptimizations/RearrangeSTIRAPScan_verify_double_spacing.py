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
    CARRIER_LIST_MHZ = [round(float(v), 4) for v in np.linspace(130.45, 133.45, 10)]  # scan dim 1
    EOM616_LIST_MHZ = [round(float(v), 4) for v in np.linspace(234.5, 237.5, 10)]     # scan dim 2

    g().AWG.AWG556.Ch1.carrier_freq_MHz = 131.78 #.scan(1, [float(v) for v in CARRIER_LIST_MHZ]) #143.5  
    g().AWG.AWG556.Ch1.pulse_width_us = 3 #5.0   # 2026-08-06 FIXED per user (was 6.0, the 08-03 lock)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15   # 2026-07-16 (now HONORED by AWGManager channel-mode; was silently forced to consts default 15)
    g().AWG.AWG556.Ch1.amplitude_scale = 0.5    # 2026-08-06 round 1: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    
    g().AWG.AWG556.Ch2.shape = "fall_quintic"   
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 131.78   
    # 2026-08-12 REVERSE WIDTH 2D: sweep the two Ch2 pulse widths (dim 1 x dim 2).
    PW556_REV = [round(float(v), 4) for v in np.linspace(1.0, 4.0, 5)]
    PW308_REV = [round(float(v), 4) for v in np.linspace(1.0, 4.0, 5)]
    g().AWG.AWG556.Ch2.pulse_width_us = 2.0   # matched-diagonal, flat 1.0-3.25 (job 926)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15   
    AMP556_REV = [round(float(v), 4) for v in np.linspace(0.2, 1.0, 5)]
    AMP308_REV = [round(float(v), 4) for v in np.linspace(0.2, 1.0, 5)]
    g().AWG.AWG556.Ch2.amplitude_scale = 0.51   # incumbent; tied with best (job 927)
    g().AWG.AWG556.Ch2.pad_time_us = 0.0
    
    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us = 3 #4.7   # 2026-08-03 round 4 optimum (locked; was 5.7, the 07-21 value)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8   # 2026-07-15 raised 5.5->7.5 (amp saturated by 7.5)
    g().AWG.AWG308.Ch1.amplitude_scale = 1
    g().AWG.AWG308.Ch1.pad_time_us = 2
    
    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 2.0   # MUST stay matched to Ch2 556 (job 926)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 1      # incumbent; tied with best (job 927)


    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; kept for the future, currently unused) ---
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 0
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these; from STIRAPAWGScan) -----
    g().Init.EOM616.Freq = 236.5e6 #.scan(2,[float(v) * 1e6 for v in EOM616_LIST_MHZ]) #= 234.2e6   # PAIR w/ Ch1 143.5 (offset +90.7, round 5 confirmed; degenerate line runs +1:+1)
    g().Pushout.VRydTrap = 2.0 #.scan(1, np.linspace(0.2, 2.5, 10)) #= 1.9
    g().Pushout.BiasCoilCurrent.Ryd = 20 #30

    # ---- post-rearrangement recool (RearrangeCool556hXStep, runs immediately before the pushout) ----
    g().Pushout.STIRAPDelay = 1.556e-6 #1.667e-6 
    g().Pushout.STIRAPReverseDelay = -0.0556e-6
    g().Pushout.STIRAPPadTime = 2e-6   # 2026-07-21 mj=0 quad ridge-3D optimum (308-first; window +0.6..+1.4us)
    g().Pushout.STIRAPGap = 1e-6   # 2026-08-12 PINNED short: measure reverse transfer, not lifetime
    #g().Pushout.STIRAPGap.scan(1, np.linspace(1e-6, 200e-6, 30)) #= 1e-6  # short fixed hold (forward optimum). For a Rydberg-lifetime sweep: .scan(1, gap_pts)
    
    # 2026-08-12 EFFICIENCY VERIFY AT 8x SPACING. Sweep IfReverse over [0, 1] in ONE scan so the
    # forward excitation and the reverse round trip are measured drift-free against each other:
    #   IfReverse=0 -> no de-excitation, survival LOW  = 1 - forward excitation
    #   IfReverse=1 -> round trip,      survival HIGH  = round-trip efficiency
    g().Pushout.IfReverse.scan(1, [0, 1])
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
    g().rearrange_kwargs.extras.pattern = "double_spacing"
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
