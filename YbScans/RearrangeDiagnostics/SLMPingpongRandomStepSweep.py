"""SLMPingpongRandomStepSweep.py -- DEBUG: random-direction pingpong, sweep step_size 0..2.0 px.

Same as SLMPingpongStepSweep.py but direction="random": each LOADED tweezer moves in its own
random direction (out-and-back), instead of all-leftward together. A/B-tests whether survival
depends on coherent collective motion vs individualized per-atom motion.

With full_n=False this is SPARSE live-only (after the 2026-06-08 protocol fix): ONLY the loaded
atoms are emitted and moved (random dirs); the unloaded ("dead") sites are DROPPED entirely (no
stationary ghosts) -- identical isolation to the fixed-direction sparse run, just with random
directions. So it cleanly A/B-tests coherent vs individualized motion with no ghost confound.
(Requires the SLMnet pingpongrandom update pushed/pulled to the SLM server.)

NOTE: table being DEFLOATED for other reasons -- recorded in run description.

Run:  cd pyctrl && python YbScans/SLMPingpongRandomStepSweep.py
"""
import argparse, os, sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
STEP_SIZES = [round(0.2 * i, 1) for i in range(0, 11)]   # 0.0..2.0 inclusive
RUN_DESC = ("DEBUG pingpong RANDOM-direction step_size sweep 0-2px, nsteps=20 two-way (reduced "
            "from 50 to avoid collisions); SPARSE live-only (full_n=False, dead sites dropped, no "
            "ghosts); A/B vs fixed-direction sparse run. TABLE BEING DEFLOATED.")


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMPingpongRandomStepSweep(url=None, reps=None):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1
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

    g().rearrange_kwargs.protocol = "pingpong"
    g().rearrange_kwargs.nsteps = 20                       # 20 out + 20 back (reduced from 50 to avoid collisions)
    g().rearrange_kwargs.step_period_ms = 1.0
    g().rearrange_kwargs.extras.step_size.scan(1, list(STEP_SIZES))   # 0..2.0 px
    g().rearrange_kwargs.extras.oneway = False            # random requires two-way
    g().rearrange_kwargs.extras.direction = "random"      # <-- per-tweezer random direction
    g().rearrange_kwargs.extras.full_n = False            # SPARSE live-only (dead sites dropped, no ghosts)
    g().rearrange_kwargs.extras.seed = 0
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5
    g().rearrange_kwargs.extras.description = RUN_DESC

    g().BlueMOT.LoadingTime = 0.23
    g().BlueMOT.FreqDetuning = -44e6
    g().GreenMOT.BiasCoilCurrent.Y = 0.268
    g().GreenMOT.BiasCoilCurrent.X = 0.040
    g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    g().GreenMOT.CoolDown.Amp = 0.25
    g().GreenMOT.CoolDown.HoldTime = 0.12

    rp.NumPerGroup = 100000
    rp.loading_defocus = -5
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMPingpongRandomStepSweep", **opts)
    print("submitted SLMPingpongRandomStepSweep -> id %s (step_sizes=%s)" % (did, STEP_SIZES))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    SLMPingpongRandomStepSweep(url=args.url, reps=args.reps)
