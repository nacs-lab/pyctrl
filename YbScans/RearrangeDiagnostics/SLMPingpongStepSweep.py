"""SLMPingpongStepSweep.py -- DEBUG: normal two-way pingpong, sweep step_size 0..2.0 px.

Goal: debug the ~92% stay-still (near-zero transit) survival seen on the every-other
rearrange runs. Pingpong moves the LOADED (live) tweezers out-and-back to their OWN source
(two-way), so survival is measured at the source sites -- it ISOLATES live-tweezer survival
with NO target-assignment / extraction-frame confound. step_size=0 = pure hold over the
2*nsteps+1 frames (the true stay-still baseline); the sweep crosses the per-frame transit-step
cliff (~1.25 px). Same warmup/loading/cooling/z4 as SLMRearrangementScan.py so it's comparable.

NOTE: the optical table is being DEFLOATED (vibration isolation off) right now for other
reasons -- expect possible vibration-induced loss; recorded in the run description.

Run:
    cd pyctrl && python YbScans/SLMPingpongStepSweep.py
"""
import argparse, os, sys

INIT_PATTERN = "33x33_uniform"
MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

STEP_SIZES = [round(0.2 * i, 1) for i in range(0, 11)]   # 0.0, 0.2, ... 2.0 inclusive
RUN_DESC = ("DEBUG pingpong step_size sweep 0-2px, nsteps=50 two-way; isolating live-tweezer "
            "survival to debug ~92% stay-still. TABLE BEING DEFLOATED (vibration isolation off) "
            "for other reasons -- expect possible vibration-induced loss.")


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMPingpongStepSweep(url=None, reps=None):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (identical to SLMRearrangementScan) ----
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

    # ---- pingpong: nsteps=50 each way (two-way), sweep step_size; period fixed ----
    g().rearrange_kwargs.protocol = "pingpong"
    g().rearrange_kwargs.nsteps = 50                       # 50 out + 50 back (fixed -> no recompile)
    g().rearrange_kwargs.step_period_ms = 1.0             # fixed
    g().rearrange_kwargs.extras.step_size.scan(1, list(STEP_SIZES))   # 0..2.0 px, dim 1
    g().rearrange_kwargs.extras.oneway = False            # two-way -> returns to source
    g().rearrange_kwargs.extras.direction = 180.0         # normal (fixed) direction, -x
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                   # MATCH loading_defocus
    g().rearrange_kwargs.extras.description = RUN_DESC

    # ---- loading/cooling: SAME as current SLMRearrangementScan.py ----
    g().BlueMOT.LoadingTime = 0.23
    g().BlueMOT.FreqDetuning = -44e6
    g().GreenMOT.BiasCoilCurrent.Y = 0.268
    g().GreenMOT.BiasCoilCurrent.X = 0.040
    g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    g().GreenMOT.CoolDown.Amp = 0.25
    g().GreenMOT.CoolDown.HoldTime = 0.12

    # ---- run params ----
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
    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMPingpongStepSweep", **opts)
    print("submitted SLMPingpongStepSweep -> id %s (step_sizes=%s)" % (did, STEP_SIZES))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    SLMPingpongStepSweep(url=args.url, reps=args.reps)
