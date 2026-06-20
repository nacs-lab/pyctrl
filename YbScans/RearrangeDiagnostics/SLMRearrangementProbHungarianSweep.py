"""SLMRearrangementProbHungarianSweep.py -- A/B sweep of the imaging-weighted Hungarian.

Identical to ``SLMRearrangementScan.py`` (single-round 47x47 rearrangement, seq = ``RearrangeCommSeq``)
EXCEPT it sweeps the new ``prob_hungarian`` protocol kwarg over ``[False, True]`` so we can directly
compare the probability-weighted assignment against the plain (distance-only) Hungarian on the SAME
array, SAME shots (``Scramble=1`` randomly interleaves the two arms -> no time-correlated bias).

What prob_hungarian does (server commit "imaging weighted hungarian", dee4fdb):
  The per-site presence PROBABILITIES the client already sends every shot (``ctx.detect_probs`` ->
  ``c.rearrange(probs)``) are injected into ``_protocol_rearrange`` as ``site_probs``. With
  ``prob_hungarian=True`` the Hungarian cost gains a per-loaded-atom term ``-beta * log(p)`` (an
  additive per-ROW penalty), so when atoms are in SURPLUS (n_loaded > n_active_targets) the
  assignment preferentially DROPS the low-confidence (dim/marginal) atoms. ``prob_hungarian=False``
  is byte-identical to the current production behaviour (cost = d^2 only).

  beta: LEFT UNSET on purpose -> the server default ``prob_hungarian_beta = 1.0`` applies (per the
  request). NOTE on what to expect: the penalty only matters under surplus, and at beta=1 a p=0.5
  atom costs only ``-log(0.5) ~= 0.69`` px^2 extra (~0.83 px equivalent) vs transit d^2 of order
  hundreds of px^2 -- so the effect is subtle and is largest when many marginal atoms are near-
  equidistant from a target. If the A/B comes out flat, the levers are (a) a more compact TARGET
  (fewer active targets than loaded -> guaranteed surplus every shot) and (b) a larger beta.

Run it:
    cd pyctrl
    python YbScans/SLMRearrangementProbHungarianSweep.py
    python YbScans/SLMRearrangementProbHungarianSweep.py --reps 1
    python YbScans/SLMRearrangementProbHungarianSweep.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url, AND the SLM server must be reachable AND running
the dee4fdb build (verify ``prob_hungarian`` appears in GET /slm/protocols/rearrange -- otherwise the
kwarg is silently dropped by the protocol signature filter and BOTH arms are identical).
"""

import argparse
import os
import sys


# --------------------------- PATTERN SELECTION (edit me) ---------------------------- #
# Initial (loading) and final (target) SLM patterns. Equal -> plain rearrangement on one array.
INIT_PATTERN = "47x47_feedbackwarm3"
TARGET_PATTERN = "47x47_feedbackwarm3"
# ------------------------------------------------------------------------------------ #

MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct/direct_best.pth"


def _pattern_cfg(name):
    """Port of ybLoadingPatternCfg.m: pattern name -> {phase_path, baked_zernike, legacy}."""
    table = {
        # CONFIRMED
        "47x47_feedbackwarm3": ("phase/47x47_feedbackwarm3.pt", [0, 0, 0, 0, 0]),
        "2x15x15_xyoffset_5um": ("phase/2x15x15_xyoffset_5um.pt", [0, 0, 0, 0, -0.75]),
        "47x47_uniform": ("phase/47x47_uniform.pt", [0, 0, 0, 0, 0]),
        "33x33_uniform": ("phase/33x33_uniform.pt", [0, 0, 0, 0, 0]),
        "3270_z4eq4":    ("phase/3270_z4eq4.pt",    [0, 0, 0, 0, -4]),
        # NAME-IMPLIED (confirm the baked Zernike before trusting)
        "33x33_1068_zernike-4": ("phase/33x33_1068_zernike-4.pt", [0, 0, 0, 0, -4]),
        "11x11withzernike-4":   ("phase/11x11withzernike-4.pt",   [0, 0, 0, 0, -4]),
        "10x10_z4eq8":          ("phase/10x10_z4eq8.pt",          [0, 0, 0, 0, -8]),
        "15x15_z4eq8":          ("phase/15x15_z4eq8.pt",          [0, 0, 0, 0, -8]),
    }
    if name not in table:
        raise ValueError("Unknown loading pattern %r (add it to _pattern_cfg)" % name)
    path, baked = table[name]
    return {"phase_path": path, "baked_zernike": [float(z) for z in baked],
            "legacy": any(z != 0 for z in baked)}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMRearrangementProbHungarianSweep(url=None, reps=None):
    """Build + submit the prob_hungarian A/B rearrangement sweep. Returns the descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    init_cfg = _pattern_cfg(INIT_PATTERN)
    target_cfg = _pattern_cfg(TARGET_PATTERN)
    n_rounds = 1

    g = ScanGroup()

    # ---- single source of truth: number of rearrangement rounds -----------------------
    g().rearrange_kwargs.extras.n_rounds = n_rounds

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
    g().rearrange_kwargs.nsteps = 200
    g().rearrange_kwargs.step_period_ms = 1
    g().rearrange_kwargs.protocol = "rearrange"
    g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.pattern = "every-other"
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5            # MATCH rp.loading_defocus (same focal plane)
    g().rearrange_kwargs.extras.initial_pattern = INIT_PATTERN
    g().rearrange_kwargs.extras.final_pattern = TARGET_PATTERN

    # ---- THE SWEEP: imaging-weighted Hungarian OFF vs ON -------------------------------
    # prob_hungarian is a real declared _protocol_rearrange signature kwarg (kind=protocol on the
    # live /slm/protocols/rearrange), passed via the extras escape hatch. Left UNSET to a scalar so
    # ScanGroup will accept the .scan().
    g().rearrange_kwargs.extras.prob_hungarian.scan(1, [False, True])
    # beta = None EXPLICITLY (per request). A None inside extras is DROPPED client-side by
    # slm_client._build_setup_body (`if ev is not None`), so it is never sent -> the server's
    # _protocol_rearrange default prob_hungarian_beta = 1.0 applies. (`float(None)` would crash the
    # protocol, but None never reaches it -- verified in the client body builder.)
    g().rearrange_kwargs.extras.prob_hungarian_beta = None

    # ---- run params (runp) ------------------------------------------------------------
    rp.NumPerGroup = 100000
    rp.loading_defocus = -5
    rp.NumImages = n_rounds + 1                  # img1 + one frame per round
    rp.Scramble = 1                              # interleave the two arms -> clean A/B
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    opts = {}
    if reps is not None:
        opts["rep"] = reps

    did = ybStartScan("RearrangeCommSeq", g, url=url,
                      label="SLMRearrangementProbHungarianSweep", **opts)
    print("submitted SLMRearrangementProbHungarianSweep -> descriptor id %s (url=%s)"
          % (did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the prob_hungarian A/B rearrangement sweep to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    args = ap.parse_args()
    SLMRearrangementProbHungarianSweep(url=args.url, reps=args.reps)
