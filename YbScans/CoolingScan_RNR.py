"""CoolingScan_RNR.py -- pyctrl port of ``matlab_new/YbScans/CoolingScan_RNR.m``.

The .m's ``ybStartScan(FreqCooling556Scan(), @ReleaseRecaptureSeq)``: a release-and-recapture
survival measurement that scans the 556 cooling *h*-beam. ``ReleaseRecaptureSeq`` images
(Imag399) -> applies 556 cooling (``Cool556hXStep``) -> drops the SLM trap for a fixed release
time (``ReleaseRecaptureStep``) -> re-images. ``NumImages=2`` => survival = how well the cooling
recaptures the atom; better h-beam cooling => higher survival.

Active scan (FreqCooling556Scan, the un-commented block):
    g().Cool556.Time            = 5e-3
    g().Cool556.X.FreqDetuning  = 0.11e6       # X beam pinned
    g().Cool556.X.Amp           = 0.16
    g().Cool556.h.FreqDetuning.scan(1) = (0.08:0.01:0.16)*1e6   # 9 detunings (Hz)
    g().Cool556.h.Amp.scan(2)          = 0:0.02:0.2             # 11 amplitudes
    g().ReleaseRecapture.Time   = 25e-6
=> a 2-D 9x11 = 99-point sweep of the h-beam (detuning x amplitude).

``Cool556hXStep`` reads ``Cool556.{Time, X.FreqDetuning, X.Amp, h.FreqDetuning, h.Amp}`` and the
swept h detuning enters as ``Freq556RydbergMOTh = Resonance556mj0Freq + detuning`` (the step adds
the resonance, so the SWEPT value here is the bare detuning in Hz -- *1e6 is applied to the colon,
not the resonance offset). ``ReleaseRecaptureStep`` reads ``ReleaseRecapture.Time`` (the 25 us
release window); ``ReleaseRecapture.SLMAOMAmp`` is left at its ``Consts().SLM.AOM.Amp`` default.

Byte-equality (THE ONE RULE): the swept detuning/amp colons have non-integer steps (0.01, 0.02),
so they are built with ``scan_export.matlab_colon`` (bit-identical to MATLAB's colon operator -- a
naive ``a+k*step`` differs by 1 ULP and a swept value serializes as a raw float64). The detuning
``*1e6`` is applied per element in the SAME order MATLAB evaluates ``(a:d:b)*1e6``. The fixed
scalars are plain float literals (no ``Consts()`` SubProps proxy to coerce). Verified per point by
the A/B oracle (``tools/check_ab_byte_equality.py`` <-> ``tools/scan_point_list_ab.m`` build_coolingrnr).

``runp`` (NumPerGroup/NumImages/Scramble/...) drives the live run only; it never enters per-seq bytes.

Run (pyctrl backend must be live at --url; reps drive passes, 0 = forever):
    cd pyctrl
    python YbScans/CoolingScan_RNR.py                       # .m default 9x11 grid
    python YbScans/CoolingScan_RNR.py --reps 4
    python YbScans/CoolingScan_RNR.py --fdet 0.08 0.01 0.16 --famp 0 0.02 0.2 --reps 0
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


# Default sweep colons (the .m's active FreqCooling556Scan block).
DEF_FDET = (0.08, 0.01, 0.16)   # detuning colon in MHz: (0.08:0.01:0.16)*1e6 -> 9 pts (Hz)
DEF_AMP = (0.0, 0.02, 0.2)      # amp colon: 0:0.02:0.2                        -> 11 pts
DEF_HAMP = DEF_AMP              # back-compat alias (old name)
# Pinned (non-swept) beam defaults: the .m's fixed X = (0.11 MHz, 0.16); config h = (0.11 MHz, 0.14).
DEF_X_PIN = (0.11e6, 0.16)      # (FreqDetuning Hz, Amp)
DEF_H_PIN = (0.11e6, 0.14)


def _runp(g):
    rp = g.runp()
    rp.NumPerGroup = 10000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    # g.runp().loading_phase = "phase/33x33_uniform.pt"   # server-side WGS phase path
    # g.runp().loading_defocus = -5                         # ANSI z4 loading defocus (rad)


def build(beam="h", fdet=DEF_FDET, famp=DEF_AMP, x_pin=DEF_X_PIN, h_pin=DEF_H_PIN,
          release_time=25e-6):
    """2-D release-recapture cooling scan: sweep one 556 beam's {FreqDetuning (dim1), Amp (dim2)},
    pin the OTHER beam at a fixed (FreqDetuning Hz, Amp).

    The no-arg default (``beam='h'``, ``x_pin=(0.11e6, 0.16)``, the .m grids, 25 us release)
    reproduces the .m's ``FreqCooling556Scan`` EXACTLY -> the A/B byte oracle (build_coolingrnr)
    stays valid. The interleaved X<->h optimization sweeps ``beam='X'``/``'h'`` alternately,
    pinning the OTHER beam at its running optimum -- X and h are COUPLED (total 556 cooling power
    is what matters; see today's Notion cooling-opt page), so the optimum is the joint X<->h fixed
    point, found by iterating to convergence. The swept ``FreqDetuning`` is the BARE detuning in
    Hz; ``Cool556hXStep`` adds ``Resonance556mj0Freq``. Pinning to an arbitrary float is byte-safe
    (the param->byte path is verified; only the .m-default grid is oracle-pinned).
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()
    g().Cool556.Time = 5e-3
    g().ReleaseRecapture.Time = float(release_time)

    # pin the NON-swept beam (a swept beam must be left unset -- ScanGroup refuses to .scan() a
    # param already assigned a fixed value)
    if beam == "h":
        g().Cool556.X.FreqDetuning = float(x_pin[0])
        g().Cool556.X.Amp = float(x_pin[1])
        node = g().Cool556.h
    elif beam == "X":
        g().Cool556.h.FreqDetuning = float(h_pin[0])
        g().Cool556.h.Amp = float(h_pin[1])
        node = g().Cool556.X
    else:
        raise ValueError("beam must be 'X' or 'h', got %r" % (beam,))

    # swept beam: detuning (dim 1, Hz) x amp (dim 2)
    freqs = [v * 1e6 for v in matlab_colon(*fdet)]   # (a:d:b)*1e6, MATLAB-exact
    amps = matlab_colon(*famp)
    node.FreqDetuning.scan(1, freqs)
    node.Amp.scan(2, amps)

    _runp(g)
    return g


def CoolingScan_RNR(url=None, reps=2, beam="h", fdet=DEF_FDET, famp=DEF_AMP,
                    x_pin=DEF_X_PIN, h_pin=DEF_H_PIN, release_time=25e-6):
    """Build + submit the release-recapture cooling scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(beam=beam, fdet=fdet, famp=famp, x_pin=x_pin, h_pin=h_pin, release_time=release_time)
    opts = {"rep": reps} if reps is not None else {}
    label = "CoolingScan_RNR_%s" % beam
    did = ybStartScan("ReleaseRecaptureSeq", g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, nseq=%d)"
          % (label, did, url or "default", reps, g.nseq()))
    return did


def main():
    ap = argparse.ArgumentParser(
        description="Submit CoolingScan_RNR (release-recapture 556 X/h cooling scan) to pyctrl.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=2, help="passes over the sweep (0 = forever)")
    ap.add_argument("--beam", choices=["X", "h"], default="h", help="which 556 beam to sweep")
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=DEF_FDET,
                    help="swept-beam detuning colon in MHz (FreqDetuning = colon*1e6)")
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=DEF_AMP,
                    help="swept-beam amp colon")
    ap.add_argument("--xpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="pin X = (det MHz, amp) when sweeping h")
    ap.add_argument("--hpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="pin h = (det MHz, amp) when sweeping X")
    ap.add_argument("--rnr-time", type=float, default=25e-6,
                    help="ReleaseRecapture.Time (s); default 25e-6 (.m value); live runs may use 50e-6")
    args = ap.parse_args()
    x_pin = (args.xpin[0] * 1e6, args.xpin[1]) if args.xpin else DEF_X_PIN
    h_pin = (args.hpin[0] * 1e6, args.hpin[1]) if args.hpin else DEF_H_PIN
    CoolingScan_RNR(url=args.url, reps=args.reps, beam=args.beam, fdet=tuple(args.fdet),
                    famp=tuple(args.famp), x_pin=x_pin, h_pin=h_pin, release_time=args.rnr_time)


if __name__ == "__main__":
    main()
