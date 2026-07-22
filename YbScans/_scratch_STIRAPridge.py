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
    g().AWG.AWG556.Ch1.shape = "rise_quintic"   # anchor; gap = inner-peak separation
    g().AWG.AWG556.Ch1.carrier_freq_MHz = 143.567   # 2026-07-21 mj=0 quad freq-2D lock (data_20260721_181013, 96.5%); PAIR w/ EOM616 282.067
    # 2026-07-21 mj=0 QUADRUPLE_SPACING coarse ridge-3D at the locked resonance.
    g().AWG.AWG556.Ch1.pulse_width_us.scan(1, list(np.linspace(4.5, 6.5, 5)))   # pw556 (center 5.5)
    g().AWG.AWG556.Ch1.max_amplitude_vpp = 15   # 2026-07-16 (now HONORED by AWGManager channel-mode; was silently forced to consts default 15)
    g().AWG.AWG556.Ch1.amplitude_scale = 1   # opt: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    
    g().AWG.AWG556.Ch2.shape = "fall_quintic"   # anchor; gap = inner-peak separation
    g().AWG.AWG556.Ch2.carrier_freq_MHz = 143.656  # mj=0 Ch2 (fall); reverse OFF, unused
    g().AWG.AWG556.Ch2.pulse_width_us = 4.531  # lobe 1/e half-width (us); 2026-07-14 STIRAP opt (was 1.467). opt: 0.8-2.0 coarse -> 1.2-1.73 zoom (x delay)
    g().AWG.AWG556.Ch2.max_amplitude_vpp = 15   # 2026-07-14 raised 11->15 (more STIRAP power)
    g().AWG.AWG556.Ch2.amplitude_scale = 1   # opt: scan 0.4-1.0 @ vpp15 -> monotonic to ceiling, best=1.0 (still power-limited)
    g().AWG.AWG556.Ch2.pad_time_us = 0.0
    
    g().AWG.AWG308.Ch1.shape = "fall_quintic"
    g().AWG.AWG308.Ch1.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch1.pulse_width_us.scan(2, list(np.linspace(4.2, 6.2, 5)))   # pw308 (center 5.2)
    g().AWG.AWG308.Ch1.max_amplitude_vpp = 8   # 2026-07-15 raised 5.5->7.5 (amp saturated by 7.5)
    g().AWG.AWG308.Ch1.amplitude_scale = 0.9
    g().AWG.AWG308.Ch1.pad_time_us = 2
    
    g().AWG.AWG308.Ch2.shape = "rise_quintic"
    g().AWG.AWG308.Ch2.carrier_freq_MHz = 200
    g().AWG.AWG308.Ch2.pulse_width_us = 4.094 #.scan(1, np.linspace(1, 2.5, 10))  # 2026-07-14 STIRAP opt (was 1.5). opt: 1.0-2.5 coarse -> 1.17-1.83 zoom (x delay)
    g().AWG.AWG308.Ch2.max_amplitude_vpp = 8
    g().AWG.AWG308.Ch2.amplitude_scale = 0.95


    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (DEFERRED port; kept for the future, currently unused) ---
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these; from STIRAPAWGScan) -----
    g().Init.EOM616.Freq = 282.067e6   # 2026-07-21 mj=0 quad freq-2D lock (data_20260721_181013, 96.5%): PAIR w/ carrier 143.567 (interior line, off-rail)
    g().Pushout.VRydTrap = 2
    g().Pushout.BiasCoilCurrent.Ryd = 30
    g().Pushout.STIRAPDelay.scan(3, list(np.linspace(0.6e-6, 1.4e-6, 5)))   # delay: 2026-07-21 mj=0 transfer window (data_20260721_174943 plateau +0.6..+1.4us)
    g().Pushout.STIRAPReverseDelay = 0.132e-6   # 2026-07-16 reverse-delay sweep best (data_20260716_203004, broad plateau ~0.90)
    g().Pushout.STIRAPGap = 0.1e-6 #.scan(1, np.linspace(0.1e-6, 200e-6, 20))
    g().Pushout.IfReverse = 0   # 2026-07-17 reverse drift-check verify (reverse ON; round-trip survival @ ReverseDelay 0.132)
    g().Pushout.IfPump = 0
    g().Pushout.PumpTime = 1e-6
    g().Pushout.Pump616Freq = 282.355e6   # mj=0 pump616; pumps OFF this campaign
    g().Pushout.Pump556Freq = 143.556e6   # mj=0 pump556; pumps OFF this campaign
    g().Pushout.Pump556Amp = 0.5 # 2026-07-16 pump556 amp sweep (data_20260716_210005): best ~0.8-1.0 (monotonic to ceiling)

    # g().Pushout.Amp369 = 1
    g().Pushout.Time369 = 2e-6 #
    g().Pushout.Vy = 4 #

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

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) ------------------------------
    g().rearrange_kwargs.nsteps = 50
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "quadruple_spacing"   # 2026-07-21 double -> quadruple_spacing
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = False
    g().rearrange_kwargs.extras.precompute_host = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = 500   # 2026-07-21 ridge-3D: 125 pts (5x5x5) x 4 reps (pass rep=4 to match)
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan
    # start. MATCHED to rearrange_kwargs.extras.z4 (the rearrange MODEL z4).
    rp.loading_defocus = -5
    rp.NumImages = 3 if verify else 2
    rp.Scramble = 1   # 2026-07-16 per user: scramble ON (decorrelate drift). NOTE: scrambling the EOM616 sweep risked a 616 unlock in the Stark scans; watch the lock -- revert to 0 if it drops.
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
    opts["description"] = (
        "mj=0 QUADRUPLE_SPACING forward-STIRAP ridge-3D coarse 2026-07-21 (STEP 2a). Cartesian 5^3 box "
        "on (pw556 4.5-6.5, pw308 4.2-6.2, delay 0.6-1.4us) at the locked mj=0 resonance (carrier "
        "143.567 / EOM616 282.067, freq-2D data_20260721_181013 = 96.5%). Trap-off block on (VRydTrap 2 "
        "= pre-ramp only), IfReverse 0. Coarse box to fit the efficient plane pw556(delay)/pw308(delay); "
        "perpendicular tube to follow. Metric: verify-conditioned survival (LOW=best); watch for edge-rail.")
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="STIRAPmj0quad_ridge3D", **opts)
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
