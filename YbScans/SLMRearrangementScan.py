"""SLMRearrangementScan.py -- THE PRODUCTION SLM-rearrangement scan (settled 2026-07-28 campaign).

SETTLED PROTOCOL: SINGLE-ROUND ``tri_3013_camfb`` -> ``kagome_2078_camfb`` with ``rearrange2``,
nsteps 40 @ 0.696 ms/frame (27.8 ms transit), linear scheduling, WARM-STARTED phase-locked WGS
transit frames (pad 2048 x 3 iters), prob-Hungarian assignment with beta 300, matched defocus -4,
and the recalibrated 399 imaging chain (PID setpoints 0.95/0.95, frame-0 DDS amps 0.23).

ACHIEVED (gated median fill of the 2078-site kagome, gate = initial load > 72% of 3013 = 2169
atoms): **0.9832 - 0.9844** (~35 median holes), best shots ~0.993. The campaign ceiling is
imaging-fidelity-limited, not transit-limited: nsteps and prob_hungarian_beta are both saturated
(see the sweeps below), and the residual holes track the ~24% common-mode shot-to-shot 399
brightness wobble (yb_skills/memory/open-imaging-common-mode-shot-wobble.md).

Provenance: Notion lab notebook, 2026 > July > "07/28" campaign page. Sweeps that fixed each value
live in ``YbScans/RearrangeDiagnostics/WarmWGSKagomeStepSweep.py`` -- run THAT (not this file) to
re-sweep nsteps / beta / per-frame 399 amps / per-frame 399 pulse length; this file is the pinned
operating point.

Two variants, dispatched on ``N_ROUNDS`` (the single source of truth):
  * N_ROUNDS == 1 (DEFAULT, the settled protocol) -> RearrangeCommSeq (img1 -> rearrange -> img2;
                   NumImages = 2). Two patterns: LOADING (initial) + FINAL (target).
  * N_ROUNDS >= 2 -> RearrangeCommSeq2 (img1 -> rearrange -> img2 -> rearrange -> img3;
                   NumImages = 3). THREE patterns: LOADING + MIDDLE + FINAL, imaged in each, with a
                   rearrangement right after the first two images. Kept working (env YB_N_ROUNDS=2)
                   but superseded -- single-round beats the 2-round 3013->2198->2078 route.

Pattern-write policy (both variants): the LOADING (initial) phase is written once at scan start by
the scan-long SlmScanSession (and re-written if the ``slm`` lock is lost). The MIDDLE / FINAL
patterns are ASSUMED already on the SLM -- produced by the rearrange() calls -- so the middle and
final images just cool + image as fast as possible (no phase re-write). Each image is DETECTED with
its OWN per-pattern registry grid + thresholds (via imagePatternsJson + rearrange_kwargs.extras.
{initial,middle,final}_pattern), so the patterns may be genuinely different arrays / spot counts --
provided the lab detection agrees with what the SLM server scores that round.

What the backend does (see engine_run.py / slm_runtime.py + RearrangeCommSeq / RearrangeCommSeq2):
  * AT DEQUEUE -- grab the scan-long ``slm`` lock, write the loading phase, push the initial
    ``setup_rearrangement`` (model + phases + ``reset_params=True``).
  * PER SHOT   -- grab the ``compute`` lock, push setup_rearrangement (swept params, sticky),
    reload_rearrange, then per round: detect bits/probs from that round's image (its own pattern),
    rearrange(), store the image; the final image runs update_rearrange; release compute; keepalive.

ANALYSIS: gate on "initial load > 72% of 3013" (= 2169 atoms, >= 91 spare over the 2078 targets)
and report the GATED MEDIAN fill + median hole count. Ungated medians mix in shots that could not
possibly fill the target and understate performance.

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


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# LOADING (initial) / MIDDLE / FINAL SLM patterns, resolved to a server-side phase + baked Zernike
# by _pattern_cfg below (port of ybLoadingPatternCfg.m). MIDDLE is only used when N_ROUNDS >= 2.
INIT_PATTERN = "3013_tri"
MIDDLE_PATTERN = "2198_kagome_res"
TARGET_PATTERN = "2078_kagome"

# Rounds of rearrangement. 1 (DEFAULT, settled) -> single-round RearrangeCommSeq, 2 images,
# 3013 -> 2078. 2 -> two-round RearrangeCommSeq2, 3 images, 3013 -> 2198 -> 2078 (kept working,
# superseded). Single source of truth: NumImages and the seq are both derived from it.
N_ROUNDS = int(os.environ.get("YB_N_ROUNDS", "1"))
# ------------------------------------------------------------------------------------ #

# Harmless under warm WGS (the WarmWGSProducer solves each transit frame instead of running the
# CNN), but still forwarded: the server needs a model_filename to complete its warmup handshake.
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"


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
        "33x33_feedback11": ("phase/33x33_feedback11.pt", [0, 0, 0, 0, 0]),
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
    the arrays may differ in spot count / order (as long as the lab agrees with the server for the
    round it feeds). Explicit imagePatternsJson wins over the runner's 2-frame auto-synthesis
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
    seq_name = "RearrangeCommSeq2" if n_rounds >= 2 else "RearrangeCommSeq"

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

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) -----------------------------
    g().rearrange_kwargs.protocol = "rearrange2"
    # nsteps 40 @ step_period_ms 0.696 = 27.8 ms transit. SATURATED: the warm-WGS nsteps sweep
    # plateaus from ~30 upward (WarmWGSKagomeStepSweep.py); 40 sits on the plateau with margin.
    # With dynamic=False (linear scheduling) each atom advances L_i / nsteps per frame, so nsteps
    # IS the per-frame step-size axis -- re-sweep it there, not here.
    if os.environ.get("YB_NSTEPS_SWEEP"):     # e.g. "10,20,...,100" -> sweep axis 1
        _ns = [int(x) for x in os.environ["YB_NSTEPS_SWEEP"].split(",") if x.strip()]
        if len(_ns) == 1:
            g().rearrange_kwargs.nsteps = _ns[0]   # .scan(dim, [x]) would ship a 1-elem LIST
        else:
            g().rearrange_kwargs.nsteps.scan(1, _ns)
    else:
        g().rearrange_kwargs.nsteps = 40
    g().rearrange_kwargs.step_period_ms = 0.696
    g().rearrange_kwargs.extras.dynamic = False    # linear = validated operating point
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.ifEnhanced = True

    # WARM-STARTED PHASE-LOCKED WGS transit frames (server extras, deployed 2026-07-27). Instead
    # of the SLMnet CNN (amplitude-blind, learned per-spot phase contract), frame k is SOLVED with
    # ``wgs_iters`` phase-locked WGS iterations at ``wgs_pad`` FFT pad, SEEDED from frame k-1's SLM
    # phase: exact spot positions, exact per-spot phase contract, faithful amplitude control.
    # Cost ~0.55-0.9 ms/frame at 2048 x 3 (pad 1024 is ~0.25 ms/frame but loses ~7% pattern power).
    # Measured EQUIVALENT to the model frames in final fill -- adopted for the exact contract.
    g().rearrange_kwargs.extras.wgs_warm = True
    g().rearrange_kwargs.extras.wgs_pad = 2048
    g().rearrange_kwargs.extras.wgs_iters = 3

    # PROB-HUNGARIAN assignment: bias the assignment toward high-confidence loaded sites (only
    # matters under atom surplus, i.e. ~2170-2470 loaded vs 2078 targets -- this scan's regime).
    # NEW SERVER CONVENTION (2026-07-29): the probability term is ``beta * nsteps * log(p_i)`` per
    # loaded row (Bayes-consistent -- per-frame loss scales the log-likelihood tradeoff by nsteps).
    # The SERVER DEFAULT beta = 2 gives a multiplier 2*nsteps = 80 at nsteps 40, which is BELOW the
    # measured saturation knee (total multiplier ~1e3, scans 20260728_195710 / 20260728_200808) --
    # an UNSET beta therefore runs SUB-PLATEAU. So the key is ALWAYS emitted. beta = 300 maps to
    # the old-convention 12000 at nsteps 40 (12000 / 40), i.e. the saturated plateau the 07-28
    # campaign pinned.
    g().rearrange_kwargs.extras.prob_hungarian = True
    g().rearrange_kwargs.extras.prob_hungarian_beta = 300

    # 2026-07-19 matched-defocus sweep (rearrange model z4 + loading_defocus + the 2-round middle
    # write ALL move together, so loaded atoms + transit frames + bookends stay co-planar): -4 WINS
    # -- loading 71.8% (vs -5's 70.0), CV 11% (vs 16-19%), rearrange eff 0.964. Env YB_DEFOCUS.
    _DEFOCUS = float(os.environ.get("YB_DEFOCUS", "-4"))
    g().rearrange_kwargs.extras.z4 = _DEFOCUS       # MATCH rp.loading_defocus (same focal plane)

    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern:
    # RearrangeCommSeq(2) tags each bseq's image with the pattern below, so each resolves
    # cooling/imaging/VSLMServo from ByPattern[that pattern] AND detects with that pattern's
    # registry grid + thresholds. Single-round uses initial_pattern (img1) + final_pattern (img2);
    # two-round adds middle_pattern (img2), with final_pattern on img3.
    # REGISTRY names (phase basenames -- see _registry_name), not the table aliases.
    g().rearrange_kwargs.extras.initial_pattern = _registry_name(init_cfg)
    if n_rounds >= 2:
        g().rearrange_kwargs.extras.middle_pattern = _registry_name(middle_cfg)
    g().rearrange_kwargs.extras.final_pattern = _registry_name(target_cfg)

    # ---- IMAGING-POWER POLICY (settled 2026-07-28) -------------------------------------
    # ONE PID setpoint for ALL images. The 399 imaging-power PID is engaged ONCE in the ROOT
    # BlueMOTStep -- at BlueMOT.Img1/Img2PIDSet -- and then HELD for the rest of the shot (PIDMode
    # TTL back to 0), so img2 (and img3 when 2-round) are taken at that SAME held voltage. The
    # middle/final patterns' own ByPattern Img1/Img2PIDSet are INERT here and are kept equal to the
    # initial pattern's in expConfig on purpose. Do NOT try to relock mid-shot (integrator reset /
    # dark-PD rail collapses the counts -- gotcha-imaging-pid-held-multiround-rearrange; the opt-in
    # relock lives only in RearrangeCommSeq2Dev).
    #
    # SETPOINTS 0.95 / 0.95 -- the PID-RECALIBRATION route, chosen by an in-scan A/B (20260728_223454)
    # that beat the old pinned 0.5/0.5 by +16 filled sites. Raising the shared setpoint raises the
    # optical flux on EVERY image; the FINAL (2078) image keeps its ByPattern DDS amps at 1/1 and so
    # POCKETS the full 2.58x photon gain -> larger per-site histogram separation / higher d' -> fewer
    # false-EMPTY mis-reads on the metric-defining frame. Do NOT re-pin 0.5.
    #
    # FRAME-0 AMPS 0.23 -- with the setpoint up 2.58x, frame 0 (tri_3013_camfb) must be brought BACK
    # DOWN to production flux or its 399 pulse over-heats the atoms that still have to survive
    # rearrangement. extras.InitImgAmp1/2 are the frame-0-ONLY DDS-amp knobs RearrangeCommSeq applies
    # to img1's Imag399 step (a plain g().Imag399.Amp1 would hit BOTH bseqs). 0.23 re-matches the
    # 0.5-at-0.5-setpoint flux, from the amp ladder 20260728_222904. AOM knee (2026-07-16, R212):
    # amps 0.5-1.0 are optically FLAT, only <= 0.5 attenuates; DDS amp -> optical power is NONLINEAR,
    # so this value is MEASURED, never computed as a ratio.
    #
    # Env overrides (in-sequence tuning on the REAL thermalized sequence) take precedence. NOTE
    # YB_IMG1PID/YB_IMG2PID are read only by the root BlueMOTStep, so they set the SHARED held
    # imaging power of ALL images (a g() override beats ByPattern in every bseq, but only the root
    # bseq engages the lock).
    g().BlueMOT.Img1PIDSet = float(os.environ.get("YB_IMG1PID", "0.95"))
    g().BlueMOT.Img2PIDSet = float(os.environ.get("YB_IMG2PID", "0.95"))
    g().rearrange_kwargs.extras.InitImgAmp1 = 0.23
    g().rearrange_kwargs.extras.InitImgAmp2 = 0.23

    # Enhanced-loading (blue LAC) in-sequence tuning hooks; unset -> expConfig defaults.
    if os.environ.get("YB_BLUELAC_DET"):
        g().LAC.BlueLAC.FreqDetuning = float(os.environ["YB_BLUELAC_DET"]) * 1e6
    if os.environ.get("YB_BLUELAC_AMP"):
        g().LAC.BlueLAC.Amp = float(os.environ["YB_BLUELAC_AMP"])

    # ---- run params (runp; loading/cooling stay at expConfig defaults) -----------------
    rp.NumPerGroup = 100000
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan start
    # (SlmScanSession). MATCHED to rearrange_kwargs.extras.z4 (the rearrange MODEL z4) so the
    # rearrangement model/WGS frames sit at the SAME focal plane as the loaded atoms (no transit
    # defocus mismatch). tri_3013_camfb has no baked Zernike, so -4 is absolute.
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
    did = ybStartScan(seq_name, g, url=url, label="SLMRearrangementScan", **opts)
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
