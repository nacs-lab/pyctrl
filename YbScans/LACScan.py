"""LACScan.py -- pyctrl port of ``matlab_new/YbScans/LACScan.m`` (run directly to submit).

Builds the LAC loading ScanGroup (seq = ``TweezerLoadingSeq``) and submits it to the RUNNING
pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring LACScan.m's
``ybStartScan(LACParamScan(), @TweezerLoadingSeq)``. This only BUILDS the ScanGroup + sends the
descriptor JSON -- it does NOT load the engine, so any interpreter with pyctrl importable + zmq
works (e.g. the yb_analysis env, base, or .venv-engine).

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
    from yb_start_scan import ybStartScan

    g = ScanGroup()

    # ===== PHASE 8: FINAL verify (single point, 50 shots) =================
    # Phase 7 (id63, 50 shots/pt): Y bias controls the vertical gradient
    # (corr_y -0.85@0.25 -> +0.61@0.275, zero-crossing ~0.268); mean peaks 0.265-0.27
    # (~0.576), CV floor ~21% INDEPENDENT of the gradient -> residual non-uniformity
    # is random per-site (SLM trap depth), not MOT. Adopt Y=0.268 (flat gradient,
    # peak rate). THIS IS THE FINAL FAST + FLATTENED CONFIG.
    g().BlueMOT.LoadingTime = 0.23                            # was 0.5
    g().BlueMOT.FreqDetuning = -44e6                          # was -40e6
    g().BlueMOT.Amp = 0.6
    g().GreenMOT.BiasCoilCurrent.Y = 0.268                    # was 0.27 (flat vertical gradient)
    g().GreenMOT.BiasCoilCurrent.Z = 0.18
    g().GreenMOT.BiasCoilCurrent.X = 0.040                    # was 0.039
    g().GreenMOT.PowerBroaden.HandoverTime = 0.015            # was 0.030
    g().GreenMOT.CoolDown.FreqDetuning = 0.35e6
    g().GreenMOT.CoolDown.Amp = 0.25                          # was 0.20
    g().GreenMOT.CoolDown.HoldTime = 0.12                     # was 0.2
    g().GreenMOT.CoolDown.RampdownTime = 0.05
    # LAC at default (single-atom verified). No .scan() -> single point.

    # ---- run params (runp) ------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 200          # image-batch cadence; rep (below) sets shots/point
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 1
    rp.isHC = 0
    rp.isGrid2 = 0

    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps

    did = ybStartScan("TweezerLoadingSeq", g, url=url, label="LACScan", **opts)
    print("submitted LACScan -> descriptor id %s (url=%s)" % (did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit LACScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=8,
                    help="passes = shots/point (0 = forever); default 8 for ~8 shots/point")
    args = ap.parse_args()
    LACScan(url=args.url, reps=args.reps)
