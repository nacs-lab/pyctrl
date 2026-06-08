"""Spectrum556Scan.py -- pyctrl port of ``matlab_new/YbScans/Spectrum556Scan.m``.

Builds the 556 push-out survival spectrum ScanGroup (seq = ``PushoutSurvivalSeq``) and submits
it to the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
Spectrum556Scan.m's ``ybStartScan(FreqPushOut556Scan(), @PushoutSurvivalSeq)``.

Active scan (from Spectrum556Scan.m, the ``mj=0, check ULE shift`` / 0-field block --
the validated mj=0 resonance-calibration recipe; this block produced the 107.735 MHz
fit recorded in expConfig.m:122, Lorentzian dip R^2=0.97, FWHM 58 kHz, 2026-06-05):
    g().Pushout.Green.Amp = 0.10
    g().Pushout.Time      = 5e-3
    g().Pushout.Green.Freq.scan(1) = (107.5:0.01:107.9)*1e6    # 41 points, 10 kHz step
The window brackets the current mj=0 resonance (107.735 MHz) with margin in both
directions. ``PushoutStep`` reads Pushout.Green.Freq/Amp + Pushout.Time
(PushoutStep.m:5,9,10); two ``Imag399Step`` calls => NumImages=2 (image before + after
push-out => survival vs freq).

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


def build(mj=0):
    """The Spectrum556Scan ScanGroup (single group, 1-D Pushout.Green.Freq sweep).

    Mirrors Spectrum556Scan.m's active blocks; the byte-affecting params only (the dbstack
    scanname/scanfilename + debug=0 metadata are dropped -- neither enters the serialized
    bytes). ``runp`` drives the live run (NumImages=2) but never the per-seq bytes.

    ``mj`` selects which Spectrum556Scan.m block to reproduce (both at 0 field):
      * ``mj=0`` -- "check ULE shift": weak/short push-out (Amp 0.10, 5 ms), 41 pts @ 10 kHz
        over (107.5:0.01:107.9) MHz, bracketing the 107.735 MHz mj=0 resonance (FWHM ~58 kHz).
        This path is BYTE-IDENTICAL to the original mj=0 build (THE ONE RULE / A-B oracle).
      * ``mj=1`` -- "check trap depth": stronger/longer push-out (Amp 0.18, 20 ms), 31 pts @
        100 kHz over (103.5:0.1:106.5) MHz, the broader |mj|=1 trap-shifted feature.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed push-out params (PushoutStep reads these); per-mj recipe ----
    if mj == 0:
        # mj=0 calibration push-out: weak/short so the dip width tracks the line,
        # not power/time broadening (the recipe behind 107.735 MHz, expConfig.m:122).
        g().Pushout.Green.Amp = 0.10
        g().Pushout.Time = 5e-3
    elif mj == 1:
        # |mj|=1 "check trap depth": stronger + longer to drive the weaker,
        # trap-shifted |mj|=1 feature (Spectrum556Scan.m active block).
        g().Pushout.Green.Amp = 0.18
        g().Pushout.Time = 20e-3
    else:
        raise ValueError("mj must be 0 or 1, got %r" % (mj,))

    # ---- fast-loading MOT config now lives in expConfig defaults ----------
    # The 2026-06-05 fast-loading optimum (BlueMOT.LoadingTime 0.23 / FreqDetuning
    # -44e6, GreenMOT bias X 0.040 / Y 0.268, HandoverTime 0.015, CoolDown Amp 0.25 /
    # HoldTime 0.12) is the apparatus default in expConfig.py/.m, so the scan no
    # longer overrides it here.

    # ---- swept param: Pushout.Green.Freq ----------------------------------
    if mj == 0:
        # mj=0 calibration window: 41 pts @ 10 kHz, centered ~107.7, brackets the
        # current 107.735 MHz resonance (FWHM ~58 kHz -> ~6 pts across the dip).
        freqs = [v * 1e6 for v in matlab_colon(107.5, 0.01, 107.9)]   # 41 pts, MATLAB-exact
    else:
        # |mj|=1 window: 31 pts @ 100 kHz over (103.5:0.1:106.5) MHz.
        freqs = [v * 1e6 for v in matlab_colon(103.5, 0.1, 106.5)]    # 31 pts, MATLAB-exact
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


def Spectrum556Scan(url=None, reps=3, mj=0):
    """Build + submit the 556 spectrum scan (mj=0 or mj=1). Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(mj=mj)
    npts = 41 if mj == 0 else 31
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    label = "Spectrum556Scan_mj%d" % mj
    did = ybStartScan("PushoutSurvivalSeq", g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, %d freq pts)"
          % (label, did, url or "default", reps, npts))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit Spectrum556Scan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    ap.add_argument("--mj", type=int, default=0, choices=(0, 1),
                    help="which 556 block: 0 = ULE-shift mj=0 (default), 1 = |mj|=1 trap-depth")
    args = ap.parse_args()
    Spectrum556Scan(url=args.url, reps=args.reps, mj=args.mj)
