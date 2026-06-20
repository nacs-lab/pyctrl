"""SLMPingpongBlurStepSweep.py -- two-way pingpong, 2D sweep: blur x step_size.

DEBUG/diagnostic pingpong (same warmup / loading / cooling / z4 as
SLMPingpongStepSweep.py so it is directly comparable). Two scanned axes:

  * dim 1 (MAIN, varies fastest): step_size 0.0 .. 2.0 px every 0.1 px (21 values).
  * dim 2 (extra axis):           blur in {0, 5} (2 values).

ScanGroup enumerates column-major with dim 1 fastest, so each blur value gets a
full step_size sweep -> 21 x 2 = 42 scan points.

period (step_period_ms) = 1.0 ms; precompute = precompute_host = True.

!! DEPLOY CHECK -- ``blur`` IS implemented in the server-side ``pingpong``
   protocol, but the LIVE SLM server must be running that version for it to take
   effect: hot-reload / pull the implementing SLMnet on the rearrange computer
   before this scan (the SLMnet checkout on the exp-control machine is stale and
   does not yet show the ``blur`` kwarg). If the deployed server lacks it, the
   dispatcher silently drops the kwarg and blur=0 vs blur=5 are IDENTICAL.

Run:
    cd pyctrl && python YbScans/SLMPingpongBlurStepSweep.py
"""
import argparse, os, sys

MODEL_FILENAME = "slmnet/checkpoints/experiment_sinc_ampmap_v3/best_model.pth"
PHASE_PATH = "phase/33x33_uniform.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]

NSTEPS = 50                                              # 50 out + 50 back (two-way), fixed
STEP_SIZES = [round(0.1 * i, 1) for i in range(0, 21)]  # 0.0, 0.1, ... 2.0 inclusive (21 vals)
BLUR_VALUES = [0.0, 5.0]                                 # extra axis (2 vals)
RUN_DESC = ("pingpong 2D sweep: step_size 0-2px @0.1 (dim1) x blur {0,5} (dim2), nsteps=50 "
            "two-way, period=1.0ms, precompute+precompute_host. blur honoured by the pingpong "
            "protocol server-side (ensure the live SLM server is on the implementing version).")


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def SLMPingpongBlurStepSweep(url=None, reps=None):
    _bootstrap()
    from scan_group import ScanGroup
    from yb_start_scan import ybStartScan

    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup (identical to SLMPingpongStepSweep / SLMRearrangementScan) ----
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

    # ---- pingpong: nsteps fixed (no recompile), 2D sweep step_size x blur ----
    g().rearrange_kwargs.protocol = "pingpong"
    g().rearrange_kwargs.nsteps = NSTEPS                  # 50 out + 50 back (fixed -> no recompile)
    g().rearrange_kwargs.step_period_ms = 1.0             # period = 1.0 ms (fixed)
    g().rearrange_kwargs.extras.step_size.scan(1, list(STEP_SIZES))  # MAIN axis (dim 1, fastest)
    g().rearrange_kwargs.extras.blur.scan(2, list(BLUR_VALUES))      # extra axis (dim 2)
    g().rearrange_kwargs.extras.oneway = False            # two-way -> returns to source
    g().rearrange_kwargs.extras.direction = 180.0         # normal (fixed) direction, -x
    g().rearrange_kwargs.extras.precompute = True
    g().rearrange_kwargs.extras.precompute_host = True
    g().rearrange_kwargs.extras.ifEnhanced = False
    g().rearrange_kwargs.extras.hw_sequence = False
    g().rearrange_kwargs.extras.z4 = -5                   # MATCH loading_defocus
    g().rearrange_kwargs.extras.description = RUN_DESC

    # ---- loading/cooling: SAME as current SLMPingpongStepSweep.py ----
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
    did = ybStartScan("RearrangeCommSeq", g, url=url, label="SLMPingpongBlurStepSweep", **opts)
    print("submitted SLMPingpongBlurStepSweep -> id %s (%d step_sizes x %d blur = %d pts, nsteps=%d)"
          % (did, len(STEP_SIZES), len(BLUR_VALUES), len(STEP_SIZES) * len(BLUR_VALUES), NSTEPS))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    SLMPingpongBlurStepSweep(url=args.url, reps=args.reps)
