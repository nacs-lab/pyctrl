"""Spectrum399Scan.py -- pyctrl port of ``matlab_new/YbScans/Spectrum399Scan.m``.

Builds the 399 push-out survival spectrum ScanGroup (seq = ``PushoutSurvival399Seq``) and
submits it to the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
Spectrum399Scan.m's ``ybStartScan(FreqPushOut399Scan(), @PushoutSurvival399Seq)``.

Active scan (the ``AbsImg beam probing mj=0`` block from Spectrum399Scan.m):
    g().Pushout.Blue.Amp1 = 0.25
    g().Pushout.Time     = 10e-3
    g().Pushout.Blue.Freq.scan(1) = (220:3:360)*1e6   # 47 points, 3 MHz step
``Pushout399Step`` reads Pushout.Blue.Freq/Amp + Pushout.Time (Pushout399Step.m:5,7,8);
the two ``Imag399Step`` calls in PushoutSurvival399Seq => NumImages=2 (image before + after
push-out => survival vs the 399 absorption-imaging frequency).

The colon ``220:3:360`` is integer-valued, so ``*1e6`` is exact -- no 1-ULP trap. It still uses
:func:`scan_export.matlab_colon` for parity with the 556 sibling (for an integer sweep that is
equivalent to a plain list).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine-py312).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/Spectrum399Scan.py                 # short A/B run: 3 passes over 47 pts
    python YbScans/Spectrum399Scan.py --reps 5
    python YbScans/Spectrum399Scan.py --reps 0        # run forever
    python YbScans/Spectrum399Scan.py --url tcp://127.0.0.1:1408
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
    """The Spectrum399Scan ScanGroup (single group, 1-D Pushout.Blue.Freq sweep).

    Mirrors Spectrum399Scan.m's active ``AbsImg beam probing mj=0`` block; the byte-affecting
    params only (the dbstack scanname/scanfilename metadata is dropped -- it never enters the
    serialized bytes). ``runp`` drives the live run (NumImages=2) but never the per-seq bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed push-out params (Pushout399Step reads these) ----------------
    g().Pushout.Blue.Amp1 = 0.25   # 0.2 for AbsImg; 0.015 for MOT-beam probing mj=1
    g().Pushout.Time = 10e-3

    # ---- swept param: Pushout.Blue.Freq ------------------------------------
    # (220:3:360)*1e6 -- 47 pts @ 3 MHz, the AbsImg-beam mj=0 probing window.
    freqs = [v * 1e6 for v in matlab_colon(220, 3, 360)]   # 47 pts, integer-valued => exact
    g().Pushout.Blue.Freq.scan(1, freqs)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 10000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    # g.runp().loading_phase = "phase/33x33_uniform.pt"   # server-side WGS phase path
    # g.runp().loading_defocus = -5                         # ANSI z4 loading defocus (rad)
    return g


def Spectrum399Scan(url=None, reps=3):
    """Build + submit the 399 spectrum scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan("PushoutSurvival399Seq", g, url=url, label="Spectrum399Scan", **opts)
    print("submitted Spectrum399Scan -> descriptor id %s (url=%s, reps=%s, 47 freq pts)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit Spectrum399Scan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    Spectrum399Scan(url=args.url, reps=args.reps)
