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
    import numpy as np   # np.linspace etc. are accepted directly by .scan() (see examples below)

    g = ScanGroup()

    # ---- active fixed params (from LACScan.m) -----------------------------
    g().BlueMOT.LoadingTime = 0.4
    g().GreenMOT.CoolDown.HoldTime = 0.2

    # ---- sweeps (commented in LACScan.m; uncomment + edit to scan) --------
    # A sweep is .scan(dim, vals); same dim co-varies, different dims = grid.
    #   g(1).LAC.FreqDetuning.scan(np.linspace(0.05e6, 1.0e6, 20))   # 1-D, dim 1
    #   g(1).LAC.Time.scan(np.linspace(1e-3, 51e-3, 11))
    g(1).GreenMOT.BiasCoilCurrent.Y.scan(1, np.linspace(0.24, 0.32, 17))  # dim 1, 17-pt sweep

    # ---- run params (runp) ------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 500
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 0
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
    ap.add_argument("--reps", type=int, default=None,
                    help="passes (0 = forever); omit -> StackNum derived from NumPerGroup")
    args = ap.parse_args()
    LACScan(url=args.url, reps=args.reps)
