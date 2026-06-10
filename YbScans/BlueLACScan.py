"""BlueLACScan.py -- pyctrl port of ``matlab_new/YbScans/BlueLACScan.m``.

Builds the blue-LAC loading ScanGroup (seq = ``BlueTweezerLoadingSeq``) and submits it to the
RUNNING pyctrl backend over ZMQ, mirroring BlueLACScan.m's
``ybStartScan(LACParamScan(), @BlueTweezerLoadingSeq)``.

SWEEP DECISION (the .m left it ambiguous -- ``scannedFreq = -(2:0.6:9)*1e6`` is computed but
every ``.scan(...)`` line is commented, so the captured .m runs a SINGLE point). For the A/B
comparison we wire that documented ``scannedFreq`` in as the active sweep:
    g().LAC.BlueLAC.FreqDetuning.scan(1) = -(2:0.6:9)*1e6      # 12 points
This gives BlueLAC a genuine MULTI-point byte-equality check and a real loading-rate-vs-blue-LAC-
detuning curve for the live A/B (instead of a trivial single point). ``BlueTweezerLoadingSeq``
reads ``LAC.BlueLAC.FreqDetuning`` via ``BlueLACStep`` (BlueLACStep.m:26 ->
Freq556RydbergMOTh, :37), so sweeping it changes the serialized DDS frequency per point.
One ``Imag399Step`` => NumImages=1 (loading image).

``-(2:0.6:9)*1e6`` has a non-integer (0.6-MHz) step, so it uses :func:`scan_export.matlab_colon`
-- a naive ``a+k*step`` drifts 1 ULP (at index 7) and the swept value goes straight into the bytes.

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/BlueLACScan.py                 # short A/B run: rep=3 passes over 12 pts
    python YbScans/BlueLACScan.py --reps 5
    python YbScans/BlueLACScan.py --reps 0        # run forever (continuous loading monitor)
    python YbScans/BlueLACScan.py --url tcp://127.0.0.1:1408
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


def build():
    """The BlueLACScan ScanGroup (single group, 1-D LAC.BlueLAC.FreqDetuning sweep).

    Mirrors BlueLACScan.m's active params (none are uncommented in the .m besides the runp), and
    activates the documented ``scannedFreq`` as the sweep. ``runp`` drives the live run but never
    the per-seq bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- swept param: LAC.BlueLAC.FreqDetuning = -(2:0.6:9)*1e6 ------------
    detunings = [-(v * 1e6) for v in matlab_colon(2.0, 0.6, 9.0)]   # 12 pts, MATLAB-exact
    g().LAC.BlueLAC.FreqDetuning.scan(1, detunings)

    # ---- run params (runp) ------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 500
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 1
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    # g.runp().loading_phase = "phase/33x33_uniform.pt"   # server-side WGS phase path
    # g.runp().loading_defocus = -5                         # ANSI z4 loading defocus (rad)
    return g


def BlueLACScan(url=None, reps=3):
    """Build + submit the blue-LAC loading scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan("BlueTweezerLoadingSeq", g, url=url, label="BlueLACScan", **opts)
    print("submitted BlueLACScan -> descriptor id %s (url=%s, reps=%s, 12 detuning pts)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit BlueLACScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the 12-pt sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    BlueLACScan(url=args.url, reps=args.reps)
