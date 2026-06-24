"""FreqAbsImgScan.py -- pyctrl port of ``matlab_new/YbScans/FreqAbsImgScan.m``.

Builds the absorption-imaging frequency-scan ScanGroup (seq = ``GreenMOTSeq``) and submits
it to the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
FreqAbsImgScan.m's ``ybStartScan(FreqAbsImgScanTest(), @GreenMOTSeq)``.

This is a free-space absorption-imaging MOT diagnostic (NOT a tweezer survival scan): it loads
a blue->green MOT, then images the 556 MOT cloud in free space with the 399 absorption beam over
time-of-flight, sweeping the 399 imaging frequency to find the absorption resonance.
``GreenMOTSeq`` (Init -> BlueMOT -> GreenMOT -> AbsImag -> wait -> Init) and all four steps it
composes (``InitStep``/``BlueMOTStep``/``GreenMOTStep``/``AbsImagStep``) already exist in pyctrl
and are reused verbatim -- this port adds only the scan descriptor.

Active scan (from FreqAbsImgScan.m's ``FreqAbsImgScanTest`` block):
    g().AbsImag.Freq.scan(1) = (300:2:330)*1e6           # 16 points, 2 MHz step
    g().AbsImag.TOF          = 5e-3                       # time-of-flight before imaging
    g().BlueMOT.LoadingTime  = 1                          # 1 s blue-MOT load
    g().GreenMOT.CoolDown.FreqDetuning = 0.25e6
``AbsImagStep`` reads AbsImag.Freq/TOF (AbsImagStep.py:25,31); ``BlueMOTStep`` reads
BlueMOT.LoadingTime (BlueMOTStep.py:48); ``GreenMOTStep`` reads GreenMOT.CoolDown.FreqDetuning.
``GreenMOTSeq`` makes a single absorption exposure path => NumImages=1.

"Port + modernize": only params that actually DIFFER from the current expConfig defaults are set
here. As of expConfig.py (2026-06) those defaults are AbsImag.TOF=0, BlueMOT.LoadingTime=0.6,
GreenMOT.CoolDown.FreqDetuning=0.35e6, AbsImag.Freq=315e6 -- all four MATLAB overrides differ, so
all four are kept; the commented-out alternative sweeps in the .m are dropped (they never enter
the bytes). MATLAB's ``g().scanname/scanfilename`` dbstack metadata is dropped (not serialized).

The colon ``300:2:330`` is integer-valued, so ``*1e6`` is exact -- no 1-ULP trap. It still uses
:func:`scan_export.matlab_colon` for parity with the Spectrum399/556 siblings (for an integer
sweep that is equivalent to a plain list).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine-py312).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/FreqAbsImgScan.py                 # short A/B run: 3 passes over 16 pts
    python YbScans/FreqAbsImgScan.py --reps 5
    python YbScans/FreqAbsImgScan.py --reps 0        # run forever
    python YbScans/FreqAbsImgScan.py --url tcp://127.0.0.1:1408
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
    """The FreqAbsImgScan ScanGroup (single group, 1-D AbsImag.Freq sweep).

    Mirrors FreqAbsImgScan.m's active ``FreqAbsImgScanTest`` block; the byte-affecting params
    only (the dbstack scanname/scanfilename metadata is dropped -- it never enters the
    serialized bytes). ``runp`` drives the live run (NumImages=1) but never the per-seq bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed params (differ from expConfig defaults) --------------------
    g().AbsImag.TOF = 5e-3                          # default 0; image after 5 ms TOF
    g().BlueMOT.LoadingTime = 5                     # default 0.6 s; longer load for the diagnostic
    #g().GreenMOT.CoolDown.FreqDetuning = 0.25e6     # default 0.35e6

    # ---- swept param: AbsImag.Freq ---------------------------------------
    # (300:2:330)*1e6 -- 16 pts @ 2 MHz, brackets the 399 absorption resonance
    # (AbsImag.Freq default 315e6 sits mid-window).
    freqs = [v * 1e6 for v in matlab_colon(300, 2, 330)]   # 16 pts, integer-valued => exact
    g().AbsImag.Freq.scan(1, freqs)

    # ---- run params (runp); no byte effect, drive the live run -----------
    rp = g.runp()
    rp.NumPerGroup = 3000
    rp.NumImages = 1
    rp.isGrid2 = 0
    rp.isInit = 1
    rp.isHC = 1
    rp.Scramble = 0
    return g


def FreqAbsImgScan(url=None, reps=3):
    """Build + submit the absorption-imaging frequency scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan("GreenMOTSeq", g, url=url, label="FreqAbsImgScan", **opts)
    print("submitted FreqAbsImgScan -> descriptor id %s (url=%s, reps=%s, 16 freq pts)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit FreqAbsImgScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    FreqAbsImgScan(url=args.url, reps=args.reps)
