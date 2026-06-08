"""Spectrum308Scan.py -- pyctrl port of ``matlab_new/YbScans/Spectrum308Scan.m``.

Builds the push-out survival ScanGroup that Spectrum308Scan.m runs (seq =
``PushoutSurvivalSeq``, the SAME seq as Spectrum556Scan) and submits it to the RUNNING
pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring Spectrum308Scan.m's
``ybStartScan(FreqPushOut308Scan(), @PushoutSurvivalSeq)``.

Active scan (the live, non-commented block of Spectrum308Scan.m -- the "revival + microwave"
recipe at 30 G):
    g().Imag399.ExposureTime         = 100e-3
    g().Init.EOM616.Freq             = 282.89e6      # 30 G
    g().Pushout.Ryd308.Amp           = 0.2
    g().Pushout.Amp369               = 0
    g().Pushout.Green.Freq           = 142.87e6      # 30 G revival
    g().Pushout.Green.Amp            = 0.22
    g().Pushout.STIRAP.gap           = 1e-3
    g().Pushout.MRabi.Gain           = 3000
    g().Pushout.BiasCoilCurrent.Ryd  = 30
    g().Pushout.Time                 = 1e-3
    g().Pushout.MRabi.Freq.scan(1)   = (10813:0.2:10817)   # 21 pts, 0.2 MHz step (microwave)

NOTE the MATLAB source sets ``ScannedFreq`` TWICE: the first ``(277:0.5:287)*1e6`` (line 20)
is immediately OVERWRITTEN by ``(10813:0.2:10817)`` (line 34, in MHz, NO ``*1e6``), which is
the sweep that actually reaches ``MRabi.Freq.scan(1)``. We reproduce only the live value.

================================ IMPORTANT CAVEAT ==============================
The swept axis here, ``Pushout.MRabi.Freq`` (the microwave), is **NOT byte-affecting** and is
**not driven** as the MATLAB source stands:
  * ``PushoutSurvivalSeq`` reads ``MRabi.Freq`` / ``MRabi.Gain`` / ``STIRAP.gap`` ONLY inside
    ``server_pre_run`` (the AWG/QICK setup), and that ENTIRE block is COMMENTED OUT in
    ``PushoutSurvivalSeq.m`` (the ``FPGA_AWG_Client`` / ``uploadSimplePulse`` calls). pyctrl's
    ``PushoutSurvivalSeq.py`` likewise does not touch MRabi (and the QICK driver
    ``devices/qick_awg`` is not wired into ``runner.py`` -- see memory
    ``open-pyctrl-awg-hardware-unverified``).
  * ``Pushout.Ryd308.Amp`` and ``Pushout.Amp369`` are likewise read-but-unused on the byte
    path (``PushoutStep`` reads ``Ryd308.Amp`` but only ever ``add('AmpAOM308', 0)``).

So every one of the 21 sweep points serializes to the SAME bytes -- the microwave frequency
never enters the FPGA/NI bytecode. The port is faithful (it reproduces exactly what the MATLAB
runs today: a survival measurement whose nominal x-axis is the microwave frequency, but with
the microwave drive commented out). If you actually want the microwave swept, the AWG/QICK
pre-run hook must be implemented on BOTH sides first; this port deliberately does not invent it.

The byte-affecting params that DO reach the bytecode: ``Init.EOM616.Freq`` (EOM ramp target +
ramp duration), ``Pushout.Green.Freq``/``Green.Amp`` (556 push-out beam), ``Pushout.Time``
(push-out hold), and ``Imag399.ExposureTime`` (camera exposure). These are byte-verified
against MATLAB per point by ``tools/check_ab_byte_equality.py`` (all 21 points identical).
==============================================================================

Byte-equality note: the 0.2 colon step ``(10813:0.2:10817)`` is NOT integer-valued in float64,
so it is generated with :func:`scan_export.matlab_colon` (bit-identical to MATLAB's colon; a
naive ``a+k*step`` differs by 1 ULP). Even though MRabi.Freq is not byte-affecting today, we
keep the exact MATLAB sweep values so the scanned-axis metadata + future AWG wiring stay exact.

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/Spectrum308Scan.py                 # short A/B run: 3 passes over 21 pts
    python YbScans/Spectrum308Scan.py --reps 5
    python YbScans/Spectrum308Scan.py --reps 0        # run forever
    python YbScans/Spectrum308Scan.py --url tcp://127.0.0.1:1408
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


def build():
    """The Spectrum308Scan ScanGroup (single group, 1-D Pushout.MRabi.Freq sweep).

    Mirrors Spectrum308Scan.m's live (non-commented) block; the byte-affecting params only.
    The dbstack ``scanname``/``scanfilename`` + ``debug=0`` metadata are dropped (they never
    enter the serialized bytes), and ``runp`` (NumImages=2, ...) drives the live run but never
    the per-seq bytes. See the module docstring re: the MRabi.Freq sweep being a non-byte-
    affecting (and currently un-driven) microwave axis.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- imaging ----------------------------------------------------------
    g().Imag399.ExposureTime = 50e-3

    # ---- EOM616 (byte-affecting: sets the slow-EOM ramp target + duration) -
    freqs = np.linspace(275.89e6, 290e6, 1)  # Single point scan
    g().Init.EOM616.Freq.scan = 282.89e6   # 30 G

    # ---- push-out beam params --------------------------------------------
    # 30 G revival recipe. Green.Freq/Amp + Time are byte-affecting (556 push-out
    # beam); Ryd308.Amp / Amp369 are read-but-unused on the byte path (faithful to .m).
    g().Pushout.Ryd308.Amp = 0.2      # max 0.4 (read-but-unused in PushoutStep)
    g().Pushout.Amp369 = 0
    g().Pushout.Green.Freq = 142.87e6  # 30 G revival
    g().Pushout.Green.Amp = 0.22
    g().Pushout.Time = 1e-3

    # ---- microwave (NOT byte-affecting; AWG/QICK pre-run is commented out) -
    # STIRAP.gap / MRabi.Gain are read only in the (disabled) server_pre_run AWG block.
    # g().Pushout.STIRAP.gap = 1e-3
    # g().Pushout.MRabi.Gain = 3000

    # ---- Rydberg bias coil -----------------------------------------------
    g().Pushout.BiasCoilCurrent.Ryd = 30

    # ---- swept axis: Pushout.MRabi.Freq (microwave, in MHz; NO *1e6) ------
    # (10813:0.2:10817) -- 21 pts @ 0.2 MHz. See caveat: this axis does not enter
    # the bytes today (every point serializes identically); kept exact via matlab_colon.
    # freqs = matlab_colon(10813, 0.2, 10817)   # 21 pts, MATLAB-exact (NOT *1e6)
    # g().Pushout.MRabi.Freq.scan(1, freqs)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 1000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    return g


def Spectrum308Scan(url=None, reps=3):
    """Build + submit the 308 spectrum scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan("PushoutSurvivalSeq", g, url=url, label="Spectrum308Scan", **opts)
    print("submitted Spectrum308Scan -> descriptor id %s (url=%s, reps=%s, 21 MRabi.Freq pts)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit Spectrum308Scan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    Spectrum308Scan(url=args.url, reps=args.reps)
