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

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from PushoutSurvival399Seq import PushoutSurvival399Seq


def build(amp2=None, time_s=None):
    """The Spectrum399Scan ScanGroup (single group, 1-D Pushout.Blue.Freq sweep).

    Mirrors Spectrum399Scan.m's active ``AbsImg beam probing mj=0`` block; the byte-affecting
    params only (the dbstack scanname/scanfilename metadata is dropped -- it never enters the
    serialized bytes). ``runp`` drives the live run (NumImages=2) but never the per-seq bytes.

    ``amp2`` / ``time_s`` override the push-out strength (defaults 0.5 / 10 ms) -- for when the
    dip goes flat and you need to test whether the 399 beam-2 push is simply too weak. 2026-08-03:
    the standard 0.5 / 10 ms returned a DEAD FLAT spectrum (survival 0.96-0.99 over 275-359 MHz,
    R^2 -0.185, scan 20260803102204) two days after the same settings gave a clean 0.99 -> 0.877
    dip at 313.79 MHz (20260801093505) -- i.e. a push-power regression, not a statistics problem.
    """
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed push-out params (Pushout399Step reads these) ----------------
    # 0.2 for AbsImg; 0.015 for MOT-beam probing mj=1. NOTE the imaging 399 is PID-servoed with
    # the DDS amps at base 1, so a 0.5 push amp is a WEAKER 399 dose than a normal image.
    g().Pushout.Blue.Amp2 = 0.5 if amp2 is None else float(amp2)
    g().Pushout.Time = 10e-3 if time_s is None else float(time_s)

    # ---- swept param: Pushout.Blue.Freq ------------------------------------
    # (275:3:360)*1e6 -- 29 pts @ 3 MHz, the AbsImg-beam mj=0 probing window.
    freqs = [v * 1e6 for v in matlab_colon(275, 3, 360)]   # 29 pts, integer-valued => exact
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
    g.runp().loading_phase = "phase/33x33_feedback11.pt"   # match 556 scans + the array on the SLM
    # 2026-08-10: loading plane now comes from the per-array config
    # (ByPattern[<pattern>].SLM.Loading.Defocus -> slm_runtime._pattern_defocus);
    # setting g.runp().loading_defocus here would override it, so it is left unset.
    return g


def Spectrum399Scan(url=None, reps=3, amp2=None, time_s=None):
    """Build + submit the 399 spectrum scan. Returns the queued descriptor id."""
    from yb_start_scan import ybStartScan

    g = build(amp2=amp2, time_s=time_s)
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan(PushoutSurvival399Seq, g, url=url, label="Spectrum399Scan", **opts)
    print("submitted Spectrum399Scan -> descriptor id %s (url=%s, reps=%s, %d freq pts, "
          "Blue.Amp2 %.3f / %.1f ms)"
          % (did, url or "default", reps, g.nseq(),
             g().Pushout.Blue.Amp2(), g().Pushout.Time() * 1e3))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit Spectrum399Scan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    ap.add_argument("--amp2", type=float, default=None,
                    help="override Pushout.Blue.Amp2, the 399 beam-2 push amp (default 0.5). "
                         "Raise it when the dip goes flat -- the push is weaker than a normal image")
    ap.add_argument("--time", dest="time_s", type=float, default=None,
                    help="override Pushout.Time in seconds (default 0.010)")
    args = ap.parse_args()
    Spectrum399Scan(url=args.url, reps=args.reps, amp2=args.amp2, time_s=args.time_s)
