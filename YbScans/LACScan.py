"""LACScan.py -- pyctrl port of ``matlab_new/YbScans/LACScan.m`` (run directly to submit).

Builds the LAC loading ScanGroup (seq = ``TweezerLoadingSeq``) and submits it to the RUNNING
pyctrl backend over ZMQ (``submit_scan_descriptor``). **Configured here as a green-MOT X-bias
(MOT position) sweep**: it sweeps ``GreenMOT.BiasCoilCurrent.X`` to optimize tweezer loading
RATE + UNIFORMITY, leaving every other parameter at the ``expConfig.py`` defaults (the old
single-point Phase-8 overrides are commented out in :func:`LACScan`). This only BUILDS the
ScanGroup + sends the descriptor JSON -- it does NOT load the engine, so any interpreter with
pyctrl importable + zmq works (e.g. the yb_analysis env, base, or .venv-engine-py312).

Run it:
    cd pyctrl
    python YbScans/LACScan.py                 # StackNum=max(ceil(NumPerGroup/nseqs),2) passes
    python YbScans/LACScan.py --reps 1        # one pass over the sweep
    python YbScans/LACScan.py --reps 0        # run forever (continuous loading monitor)
    python YbScans/LACScan.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url (default tcp://127.0.0.1:1408 -- the
monitor's URL). Submit, then watch the dashboard's Tweezer Array / loading.
"""

import argparse
import os
import sys
import numpy as np


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def LACScan(url=None, reps=None):
    """Build + submit the LAC loading scan. Returns the queued descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from yb_start_scan import ybStartScan

    g = ScanGroup()

    # ===== SCAN: GreenMOT X bias coil (green-MOT position along x) ========
    # The X bias current shifts the green-MOT cloud along x relative to the
    # tweezer array, so it sets both how much the MOT overlaps the array (peak
    # loading RATE) and whether that overlap is centered (loading UNIFORMITY).
    # It is razor-sharp: +/-0.01 A off the optimum kills loading
    # (scan-methodology "fast+flat": GreenMOT bias Y x Z x X). Bracket the
    # expConfig default (0.040 A) by +/-0.01 A so both shoulders fall off,
    # 11 pts @ 2 mA. matlab_colon -> MATLAB-exact float64 (X feeds VBiasCoilX,
    # a byte-affecting analog value). Recenter / expand next run if the peak
    # pins to an edge or sits off-center.
    xvals = matlab_colon(0.030, 0.002, 0.050)                # 11 pts, centered on 0.040 A
    #g().GreenMOT.BiasCoilCurrent.X.scan(1, xvals)

    # ---- everything else at expConfig defaults (unscanned -> commented out) --
    # The 2026-06-05 "fast+flat" loading optimum is already the apparatus default
    # in expConfig.py, so we leave it there and sweep ONLY X. Two of the old
    # Phase-8 overrides differ slightly from expConfig and now fall back to it:
    # BlueMOT.LoadingTime 0.23 -> 0.30 s and GreenMOT.CoolDown.HoldTime 0.12 ->
    # 0.20 s (a touch more saturated/longer); the rest are identical to expConfig.
    g().Init.VSLMservo = 3.5
    ampscan = np.linspace(0.4, 0.6, 10)
    g().Imag399.Amp1.scan(1, ampscan)  
    g().Imag399.Amp2.scan(1, ampscan)  
    # g().BlueMOT.LoadingTime = 0.23
    # g().BlueMOT.FreqDetuning = -44e6
    # g().BlueMOT.Amp = 0.6
    # g().GreenMOT.BiasCoilCurrent.Y = 0.268
    # g().GreenMOT.BiasCoilCurrent.Z = 0.18
    # g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    # g().GreenMOT.CoolDown.FreqDetuning = 0.35e6
    # g().GreenMOT.CoolDown.Amp = 0.25
    # g().GreenMOT.CoolDown.HoldTime = 0.12
    # g().GreenMOT.CoolDown.RampdownTime = 0.05
    # LAC at default (single-atom verified).

    # ---- run params (runp) ------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 200          # image-batch cadence; rep (below) sets shots/point
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    g.runp().loading_phase = "phase/47x47_feedbackwarm4.pt"   # server-side WGS phase path
    g.runp().loading_defocus = -5                         # ANSI z4 loading defocus (rad)

    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps

    did = ybStartScan("TweezerLoadingSeq", g, url=url, label="LACScan_Xbias", **opts)
    print("submitted LACScan X-bias sweep (%d pts %.3f..%.3f A) -> descriptor id %s (url=%s)"
          % (len(xvals), xvals[0], xvals[-1], did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit LACScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=15,
                    help="passes = shots/point (0 = forever); default 15 -- more shots/point "
                         "than a survival scan because per-site CV/gradient (uniformity) needs "
                         "the statistics. Lower (~8) if you only care about the rate trend.")
    args = ap.parse_args()
    LACScan(url=args.url, reps=args.reps)
