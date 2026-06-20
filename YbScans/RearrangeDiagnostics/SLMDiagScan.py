"""SLMDiagScan.py -- parametrized rearrangement DIAGNOSTIC scans (pyctrl).

Built to answer "why does the every-other rearrange scan top out at ~66% TP?". The 66% scan
(sid=20260605061706) showed TP_return ~37% mean / ~63% best, Loss ~63%, FP ~0.8%, with survival
falling off steeply with transit distance (47%@0px -> ~0%@175px) -- i.e. TRANSIT LOSS, not an
index/assignment bug (the held-tweezer test already cleared indexing: 94% survival, 109x contrast).

This file builds three single-point diagnostics that isolate the pieces, all on the SAME warmup as
the 66% scan (33x33_uniform, experiment_sinc_ampmap_v3, grid_rotation=90), cooling OFF (matching the
66% scan), <500 sequences each:

  * "pingpong"   -- transit-fidelity baseline. Move atoms out-and-back at a SAFE per-frame step;
                    survival = how well atoms survive being moved at all (no assignment).
  * "shift_left" -- directional uniform 1-column move + Hungarian assignment. Asymmetric, so it also
                    reveals any TARGET-side frame issue the symmetric every-other would hide.
  * "every_other"-- the real scan, pushed to nsteps=250 (beyond the 150 the 66% sweep maxed at) to
                    test the "not enough transit steps for the long moves" hypothesis.

Run standalone (submit only):
    python YbScans/SLMDiagScan.py --protocol pingpong
Or drive submit+monitor+abort via tmp/_run_diag.py (caps shots, auto-aborts).
"""

import argparse
import os
import sys

INIT_PATTERN = "33x33_uniform"
MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

# 2D sweep axes (dim 1 = nsteps, dim 2 = step_period_ms), shared by all protocols so each gets a
# TP-vs-(nsteps, step_period) surface like the original 66% scan -- but coarse (3x3=9 points) to fit
# 50 shots/point under the <500-sequence cap. nsteps spans past the 150 the 66% sweep maxed at (300
# tests "more steps"); step_period spans fast (near the ~0.7 ms SLM floor) to slow (where the 66%
# surface cratered).
NSTEPS_SWEEP = [50, 100, 150]
STEP_PERIOD_SWEEP = [0.5, 1.0, 2.0]

# Per-protocol fixed params. Cooling OFF (RearrCoolAmp=0) to match the 66% scan.
PROTOCOLS = {
    "pingpong": dict(
        seq="RearrangeCommSeq", protocol="pingpong",
        extras=dict(oneway=False, direction="random", step_size=0.5),
        note="transit-fidelity baseline (out-and-back, safe step_size=0.5); nsteps sets excursion"),
    "shift_left": dict(
        seq="RearrangeCommSeq", protocol="shift_left",
        extras=dict(full_n=True),
        note="directional 1-column move + Hungarian assignment (asymmetric)"),
    "every_other": dict(
        seq="RearrangeCommSeq", protocol="rearrange",
        extras=dict(pattern="every-other", kagome_crop=0.88,
                    model_bookend_pre=True, model_bookend_post=True),
        note="the real scan; nsteps sweep reaches 300 (vs the 150 the 66% sweep maxed at)"),
}


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build_diag_scangroup(protocol, num_per_group=450):
    """Return (ScanGroup, seq_name) for a named diagnostic protocol. num_per_group<500."""
    _bootstrap()
    from scan_group import ScanGroup
    if protocol not in PROTOCOLS:
        raise ValueError("unknown protocol %r; choose from %s" % (protocol, list(PROTOCOLS)))
    cfg = PROTOCOLS[protocol]
    if num_per_group >= 500:
        raise ValueError("num_per_group must be < 500 (got %d)" % num_per_group)

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # warmup (dequeue, reset_params) -- identical to the 66% scan
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.extras.final_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # per-shot rearrange_kwargs -- 2D sweep over nsteps (dim 1) x step_period_ms (dim 2)
    g().rearrange_kwargs.protocol = cfg["protocol"]
    g().rearrange_kwargs.nsteps.scan(1, list(NSTEPS_SWEEP))
    g().rearrange_kwargs.step_period_ms.scan(2, list(STEP_PERIOD_SWEEP))
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = True
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.block_max_size = 256
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.RearrCoolAmp = 0       # cooling OFF (match the 66% scan)
    for k, v in cfg["extras"].items():
        setattr(g().rearrange_kwargs.extras, k, v)

    # non-rearrangement settings -- identical to the 66% scan
    g().BlueMOT.LoadingTime = 0.5
    g().GreenMOT.CoolDown.HoldTime = 0.2
    g().GreenMOT.BiasCoilCurrent.X = 0.039
    g().GreenMOT.BiasCoilCurrent.Y = 0.27
    g().GreenMOT.BiasCoilCurrent.Z = 0.18
    g().LAC.BlueLAC.FreqDetuning = -3.8e6
    g().LAC.BlueLAC.Amp = 0.17

    rp.NumPerGroup = int(num_per_group)                # <500 sequences
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    return g, cfg["seq"]


def main():
    ap = argparse.ArgumentParser(description="Submit a rearrangement diagnostic scan.")
    ap.add_argument("--protocol", required=True, choices=list(PROTOCOLS))
    ap.add_argument("--url", default=None)
    ap.add_argument("--num", type=int, default=450, help="NumPerGroup (<500)")
    args = ap.parse_args()
    _bootstrap()
    from yb_start_scan import ybStartScan
    g, seq = build_diag_scangroup(args.protocol, num_per_group=args.num)
    did = ybStartScan(seq, g, url=args.url, label="SLMDiag_" + args.protocol)
    print("submitted SLMDiag_%s -> id %s : %s" % (args.protocol, did, PROTOCOLS[args.protocol]["note"]))


if __name__ == "__main__":
    main()
