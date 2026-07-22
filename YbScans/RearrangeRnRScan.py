"""RearrangeRnRScan.py -- rearrange -> release-and-recapture -> image survival scan (pyctrl).

Marries the FULL single-round SLM-rearrangement path (a stripped-to-N_ROUNDS==1 copy of
YbScans/SLMRearrangementScan.py's build: same warmup_kwargs / rearrange_kwargs / pattern config /
scan-long SLM lock / per-pattern detection) with a release-and-recapture (R&R) test on the
REARRANGED array as the main measurement. Runs ``RearrangeRnRCommSeq``:

    img1 (Imag399 #1) -> rearrange(probs) -> Cool556 -> RELEASE-AND-RECAPTURE -> img2 (Imag399 #2)

``NumImages == 2`` (img1 = initial load, img2 = post-R&R). The observable is survival vs release
time, measured on the rearranged (target-pattern) array -- the recapture probability decays with
release time at a rate set by the atom temperature.

Two-image design + the zero-release baseline (the point of this scan)
---------------------------------------------------------------------
Only TWO images are needed because ``ReleaseRecapture.Time == 0`` is used as the proxy for "did
NOT release and recapture": :func:`RearrangeRnRStep` SKIPS itself entirely at t=0 (adds NO bytes
-- no trap drop, no scope trigger, no 3 us AOM settle), so the t=0 shot is a byte-clean "held in
traps" baseline (rearrange, then image, trap never perturbed). VERIFIED: at serialize() the swept
``ReleaseRecapture.Time`` resolves to a concrete float per point (0.0 at t=0), the t=0 branch is
numerically taken, and it produces a DISTINCT, ~23 B SMALLER blob than any t>0 point (R&R absent).
So the survival at t=0 is the "no R&R" reference; finite times are the actual R&R survival.

Sweep (editable): ``ReleaseRecapture.Time = 0 .. tmax`` in ``tstep`` steps (default 0 .. 50 us @
5 us -> 11 points, INCLUDING the t=0 baseline). Override on the CLI with ``--tstep`` / ``--tmax``.
The 0-anchored integer colon ``(0:1:N)*tstep`` is MATLAB-exact (matlab_colon); the swept value
feeds ``RearrangeRnRStep`` as ``t_release -> s.wait(t_release)``.

PATTERN / MODEL config is edit-me below, mirroring SLMRearrangementScan.py -- keep the loading and
target patterns EQUAL for a plain "rearrange then R&R" measurement (default: 33x33_feedback11).
The rearrangement MODEL (``MODEL_FILENAME``) must match the pattern family.

Run it (pyctrl backend must be live at --url, AND the SLM server reachable -- the scan-long slm
lock is mandatory):
    cd pyctrl
    python YbScans/RearrangeRnRScan.py                        # default 0..50 us @ 5 us, 11 pts
    python YbScans/RearrangeRnRScan.py --tstep 2e-6 --tmax 40e-6
    python YbScans/RearrangeRnRScan.py --reps 4
    python YbScans/RearrangeRnRScan.py --url tcp://127.0.0.1:1408
"""

import argparse
import json
import os
import sys


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# LOADING (initial) / FINAL (target) SLM patterns, resolved to a server-side phase + baked
# Zernike by _pattern_cfg below (port of ybLoadingPatternCfg.m). For a plain rearrangement leave
# them EQUAL; set them apart to rearrange FROM one pattern INTO another. The rearrangement MODEL
# (MODEL_FILENAME) must match the family. This scan is single-round only (no MIDDLE pattern).
INIT_PATTERN = "33x33_feedback11"
TARGET_PATTERN = "33x33_feedback11"
# ------------------------------------------------------------------------------------ #

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
# MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"

# Default release-time sweep colon: (0:1:N)*tstep, 0-anchored so the t=0 baseline is included.
DEF_TSTEP = 5e-6       # release-time step (s)
DEF_TMAX = 50e-6       # release-time upper bound (s) -> default 11 pts (0,5,...,50 us)


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
        "11x11withzernike-4":   ("phase/11x11withzernike-4.pt",   [0, 0, 0, 0, -4]),
        "10x10_z4eq8":          ("phase/10x10_z4eq8.pt",          [0, 0, 0, 0, -8]),
        "15x15_z4eq8":          ("phase/15x15_z4eq8.pt",          [0, 0, 0, 0, -8]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _pattern_item(name, cfg):
    """One imagePatternsJson entry (per camera frame): name + base phase + baked Zernike to strip.
    ``order='col'`` matches the runner's synthesized default + the server sweep_order the detection
    grid is derived in, so the per-pattern registry grid lines up with what the server scores."""
    it = {"name": name, "base_phase_path": cfg["phase_path"], "order": "col",
          "legacy_zerniked": bool(cfg["legacy"])}
    if cfg["legacy"]:
        it["baked_zernike"] = cfg["baked_zernike"]
    return it


def _image_patterns_json(init_cfg, target_cfg):
    """Per-frame detection declaration for the single-round R&R seq: [LOADING, FINAL]. Each frame
    is detected against its own per-pattern registry grid. Explicit imagePatternsJson wins over
    the runner's 2-frame auto-synthesis."""
    items = [_pattern_item(INIT_PATTERN, init_cfg), _pattern_item(TARGET_PATTERN, target_cfg)]
    return json.dumps(items)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build(tstep=DEF_TSTEP, tmax=DEF_TMAX):
    """Build (do NOT submit) the rearrange -> R&R ScanGroup. Returns ``(seq_name, g)`` -- the seq
    (always ``RearrangeRnRCommSeq``, single-round) and the configured ScanGroup. Kept separate from
    :func:`RearrangeRnRScan` so it can be exercised offline WITHOUT touching the live backend (the
    scan-verification convention).

    Everything except the swept ``ReleaseRecapture.Time`` + NumImages mirrors
    SLMRearrangementScan.build() at N_ROUNDS==1.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    seq_name = "RearrangeRnRCommSeq"

    g = ScanGroup()

    # ---- single source of truth: single-round rearrangement ---------------------------
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
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

    # ---- rearrange_kwargs (g(); per-shot setup, sweepable) -----------------------------
    # NOTE: the rearrangement-parameter sweeps from SLMRearrangementScan (nsteps / overdrive /
    # step_size / phase_alpha ...) are DELIBERATELY pinned here to single values -- this scan's
    # sweep axis is ReleaseRecapture.Time (below), and the rearrange should be a fixed, known-good
    # configuration so survival-vs-release-time is not confounded by a changing rearrange. Pin the
    # rearrange to your production values, then uncomment a rearrange sweep only for a 2-D study.
    g().rearrange_kwargs.nsteps = 100
    g().rearrange_kwargs.step_period_ms = 0.696  # pinned (period = 1 ms)
    g().rearrange_kwargs.protocol = "rearrange2"
    g().rearrange_kwargs.extras.overdrive = True
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.max_step_size = 0.75
    g().rearrange_kwargs.extras.pattern = "every-other"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True   # host-resident precompute / pre-pin
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    # Per-bseq cooling/imaging overlay (expConfig ByPattern) + per-frame detection pattern:
    # RearrangeRnRCommSeq tags img1 with initial_pattern, img2 with final_pattern, so each resolves
    # cooling/imaging/VSLMServo from ByPattern[that pattern] AND detects with that pattern's grid.
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- swept param: the release-and-recapture window --------------------------------
    # ReleaseRecapture.Time = (0:1:N)*tstep -- 0-anchored so the t=0 "held in traps" baseline is
    # in-scan (RearrangeRnRStep skips itself at t=0). Integer colon => matlab_colon is bit-exact
    # and the *tstep multiply matches MATLAB per element.
    n = int(round(tmax / tstep))                          # number of intervals
    times = [v * tstep for v in matlab_colon(0, 1, n)]    # n+1 pts incl. 0, MATLAB-exact
    g().ReleaseRecapture.Time.scan(1, times)
    g().ReleaseRecapture.Hold = 0   # set for faithfulness; UNREAD by RearrangeRnRStep (no byte effect)

    # ---- run params (runp) ------------------------------------------------------------
    rp.NumPerGroup = 100000
    # Loading defocus (ANSI z4, rad) added to the base loading phase on the SLM write at scan
    # start (SlmScanSession). MATCHED to rearrange_kwargs.extras.z4 so the rearrangement model
    # frames sit at the SAME focal plane as the loaded atoms.
    rp.loading_defocus = -5
    rp.NumImages = 2                # img1 (load) + img2 (post-R&R); single round
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    # Hold the slm lock for the WHOLE scan + write the loading phase once at scan start.
    rp.useScanLongSlmLock = 1
    # Per-frame detection declaration: [LOADING, FINAL]. Explicit -> wins over auto-synthesis.
    rp.imagePatternsJson = _image_patterns_json(init_cfg, target_cfg)

    return seq_name, g


def RearrangeRnRScan(url=None, reps=None, tstep=DEF_TSTEP, tmax=DEF_TMAX):
    """Build + SUBMIT the rearrange -> release-and-recapture scan to the running pyctrl backend
    over ZMQ. Returns the descriptor id."""
    seq_name, g = build(tstep=tstep, tmax=tmax)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="RearrangeRnRScan", **opts)
    print("submitted RearrangeRnRScan (single-round -> %s, %d release-time pts) -> descriptor id %s (url=%s)"
          % (seq_name, g.nseq(), did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit RearrangeRnRScan (rearrange -> release-and-recapture -> image) to pyctrl.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    ap.add_argument("--tstep", type=float, default=DEF_TSTEP,
                    help="release-time step in s (default 5e-6)")
    ap.add_argument("--tmax", type=float, default=DEF_TMAX,
                    help="release-time upper bound in s (default 50e-6); sweep is 0..tmax incl. baseline")
    args = ap.parse_args()
    RearrangeRnRScan(url=args.url, reps=args.reps, tstep=args.tstep, tmax=args.tmax)
