"""Spectrum556Scan.py -- pyctrl port of ``matlab_new/YbScans/Spectrum556Scan.m``.

Builds the 556 push-out survival spectrum ScanGroup (seq = ``PushoutSurvivalSeq``) and submits
it to the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
Spectrum556Scan.m's ``ybStartScan(FreqPushOut556Scan(), @PushoutSurvivalSeq)``.

Active scan (from Spectrum556Scan.m, the ``|mj|=1, check trap depth`` block):
    g().Pushout.Green.Amp = 0.18
    g().Pushout.Time      = 20e-3
    g().Pushout.Green.Freq.scan(1) = (103.5:0.1:106.5)*1e6     # 31 points
``PushoutStep`` reads Pushout.Green.Freq/Amp + Pushout.Time (PushoutStep.m:5,9,10); two
``Imag399Step`` calls => NumImages=2 (image before + after push-out => survival vs freq).

Byte-equality note: the 0.1-MHz colon step is NOT integer-valued in float64, so the swept
frequency is generated with :func:`scan_export.matlab_colon` -- a bit-identical reproduction of
MATLAB's colon operator (a naive ``a+k*step`` differs by 1 ULP and would break THE ONE RULE,
since the swept value serializes as a raw float64). The exact 31 frequencies this build emits
are byte-verified against MATLAB per point by ``tools/check_ab_byte_equality.py``.

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/Spectrum556Scan.py                 # short A/B run: rep=3 passes over 31 pts
    python YbScans/Spectrum556Scan.py --reps 5
    python YbScans/Spectrum556Scan.py --reps 0        # run forever
    python YbScans/Spectrum556Scan.py --url tcp://127.0.0.1:1408
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
    """The Spectrum556Scan ScanGroup (single group, 1-D Pushout.Green.Freq sweep).

    Mirrors Spectrum556Scan.m's active block; the byte-affecting params only (the dbstack
    scanname/scanfilename + debug=0 metadata are dropped -- neither enters the serialized
    bytes). ``runp`` drives the live run (NumImages=2) but never the per-seq bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed params (PushoutStep reads these) ---------------------------
    g().Pushout.Green.Amp = 0.18
    g().Pushout.Time = 20e-3

    # ---- swept param: Pushout.Green.Freq = (103.5:0.1:106.5)*1e6 ----------
    freqs = [v * 1e6 for v in matlab_colon(103.5, 0.1, 106.5)]   # 31 pts, MATLAB-exact
    g().Pushout.Green.Freq.scan(1, freqs)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    return g


def Spectrum556Scan(url=None, reps=3):
    """Build + submit the 556 spectrum scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan("PushoutSurvivalSeq", g, url=url, label="Spectrum556Scan", **opts)
    print("submitted Spectrum556Scan -> descriptor id %s (url=%s, reps=%s, 31 freq pts)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit Spectrum556Scan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the 31-pt sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    Spectrum556Scan(url=args.url, reps=args.reps)
