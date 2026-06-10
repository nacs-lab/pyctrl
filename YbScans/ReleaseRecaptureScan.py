"""ReleaseRecaptureScan.py -- pyctrl port of ``matlab_new/YbScans/ReleaseRecaptureScan.m``.

The .m's ``ybStartScan(ReleaseTimeScan(), @ReleaseRecaptureSeq)``: a release-and-recapture
*release-time* scan (atom-temperature measurement). ``ReleaseRecaptureSeq`` images (Imag399)
-> applies 556 cooling (``Cool556hXStep``) -> drops the SLM trap for a swept release time
(``ReleaseRecaptureStep``) -> re-images. ``NumImages=2`` => survival vs release time; the
recapture probability decays with release time at a rate set by the atom temperature
(longer free flight => hotter atoms escape => lower survival).

Active scan (ReleaseTimeScan, the un-commented block):
    g().Imag399.ExposureTime           = 100e-3
    g().SLM.VServo                      = 5
    g().ReleaseRecapture.Time.scan(1)  = (0:1:50)*1e-6      # 51 release times (s)
    g().ReleaseRecapture.Hold          = 0                  # set but UNREAD by the step
    g().Cool556.Time                   = 5e-3
=> a 1-D 51-point sweep of the free-flight release time, 0 .. 50 us in 1 us steps.

Param wiring: the swept ``ReleaseRecapture.Time`` reaches ``ReleaseRecaptureStep`` as
``t_release`` -> ``s.wait(t_release)`` (the per-point free-flight gap). The fixed scalars:
``Imag399.ExposureTime`` -> ``Imag399Step`` ``s.wait``; ``SLM.VServo`` -> ``SLMStep``
``s.add('VSLMservo', V)``; ``Cool556.Time`` -> ``Cool556hXStep`` ``s.wait``.
``ReleaseRecapture.Hold`` is set for faithfulness to the .m but unread by the step (which reads
only ``Time`` + ``SLMAOMAmp``); ``SLMAOMAmp`` stays at its ``Consts().SLM.AOM.Amp`` default.

The sweep ``(0:1:50)*1e-6`` is integer-valued, so it needs no special colon handling (no 1-ULP
trap, unlike the non-integer steps in BlueLAC/Spectrum556).

``runp`` (NumPerGroup/NumImages/Scramble/...) drives the live run only; it never enters per-seq
bytes. The .m sets ``NumPerGroup = numel(ScannedTime)*rep`` with ``rep=2`` => 102 (=> StackNum
= ceil(102/51) = 2 passes); the CLI ``--reps`` overrides the pass count for A/B runs.

Run (pyctrl backend must be live at --url; reps drive passes, 0 = forever):
    cd pyctrl
    python YbScans/ReleaseRecaptureScan.py                 # .m default: 51 pts, 2 passes
    python YbScans/ReleaseRecaptureScan.py --reps 8        # more statistics per point
    python YbScans/ReleaseRecaptureScan.py --tmax 40e-6    # sweep 0..40 us instead
    python YbScans/ReleaseRecaptureScan.py --url tcp://127.0.0.1:1408
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


# Default release-time sweep colon (the .m's ReleaseTimeScan block): (0:1:50)*1e-6 -> 51 pts.
DEF_TSTEP = 1e-6      # release-time step (s)
DEF_TMAX = 50e-6      # release-time upper bound (s)


def build(tstep=DEF_TSTEP, tmax=DEF_TMAX):
    """The ReleaseRecaptureScan ScanGroup (single group, 1-D ReleaseRecapture.Time sweep).

    The no-arg default reproduces the .m's ``ReleaseTimeScan`` EXACTLY -> the A/B byte oracle
    (build_releasetime) stays valid. ``tstep``/``tmax`` let the live run resize the sweep
    (e.g. a coarser/finer free-flight grid); the param->byte path is identical, only the .m
    default (0..50 us @ 1 us) is oracle-pinned.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- fixed params (the steps read these) ------------------------------
    g().Imag399.ExposureTime = 100e-3          # -> Imag399Step t_Imag399 (s.wait)
    g().SLM.VServo = 5                          # -> SLMStep V_SLMServo (s.add 'VSLMservo')
    g().Cool556.Time = 5e-3                     # -> Cool556hXStep t_Cool556 (s.wait)
    g().ReleaseRecapture.Hold = 0              # set but UNREAD by ReleaseRecaptureStep (no byte effect)

    # ---- swept param: ReleaseRecapture.Time = (0:1:tmax/tstep)*tstep ------
    # 0:1:50 is integer-valued, so matlab_colon is exact and the *tstep scalar multiply is a
    # bit-identical IEEE-754 double (MATLAB evaluates (0:1:N)*step the same way per element).
    n = int(round(tmax / tstep))               # number of intervals (50 for the .m default)
    times = [v * tstep for v in matlab_colon(0, 1, n)]   # n+1 pts, MATLAB-exact
    g().ReleaseRecapture.Time.scan(1, times)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = len(times) * 2            # .m: numel(ScannedTime)*rep, rep=2 -> StackNum=2
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


def ReleaseRecaptureScan(url=None, reps=2, tstep=DEF_TSTEP, tmax=DEF_TMAX):
    """Build + submit the release-recapture release-time scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(tstep=tstep, tmax=tmax)
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan("ReleaseRecaptureSeq", g, url=url, label="ReleaseRecaptureScan", **opts)
    print("submitted ReleaseRecaptureScan -> descriptor id %s (url=%s, reps=%s, nseq=%d)"
          % (did, url or "default", reps, g.nseq()))
    return did


def main():
    ap = argparse.ArgumentParser(
        description="Submit ReleaseRecaptureScan (release-recapture release-time scan) to pyctrl.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=2,
                    help="passes over the sweep (0 = forever); default 2 (the .m's rep)")
    ap.add_argument("--tstep", type=float, default=DEF_TSTEP,
                    help="release-time step in s (default 1e-6, the .m value)")
    ap.add_argument("--tmax", type=float, default=DEF_TMAX,
                    help="release-time upper bound in s (default 50e-6, the .m value)")
    args = ap.parse_args()
    ReleaseRecaptureScan(url=args.url, reps=args.reps, tstep=args.tstep, tmax=args.tmax)


if __name__ == "__main__":
    main()
