"""CoolingScan.py -- pyctrl port of ``matlab_new/YbScans/CoolingScan.m``.

Optimises the 556 cooling beams (the X + h push-out beams run AT imaging amplitudes during a
100 ms imaging hold -- ``ImagingPushoutSurvivalSeq``, ``NumImages=2`` => survival = how well the
556 light keeps the atom alive while it is being imaged). Better cooling => higher survival.

The .m leaves every sweep commented (it documents ``ScannedFreq = (0.10:0.02:0.26)*1e6`` detuning
and ``ScannedAmp = 0.1:0.02:0.28``, with ``Blue.Amp = 0.3``). This port exposes the three scans the
optimisation campaign needs, each on the SAME byte-verified seq:

  * ``blue_amp``  -- 1-D sweep of ``Pushout.Blue.Amp`` (the 399 imaging intensity). Used FIRST to pick
    an imaging amplitude that puts survival in a sensitive mid-range (good distinction between the
    cooling-parameter points that follow) with decent atom discrimination.
  * ``x2d``       -- 2-D ``Pushout.Green.X.{Freq,Amp}`` (Freq = Resonance556mj0 + detuning on dim 1,
    Amp on dim 2). Find the X-beam cooling optimum.
  * ``h2d``       -- 2-D ``Pushout.Green.h.{Freq,Amp}`` (same structure). Find the h-beam optimum.

Byte-equality (THE ONE RULE): every fixed freq/amp is read from ``Consts()`` exactly as the .m;
the swept detunings/amps are built with ``scan_export.matlab_colon`` (MATLAB-exact colon) and the
X/h frequency is ``Resonance556mj0Freq + detuning*1e6`` in the SAME order MATLAB evaluates it.
``float(...)`` wraps the bare-Consts-leaf amps (a bare leaf would store a SubProps proxy -- see the
pyctrl skill findings.md). Verify a representative config per point with the A/B oracle.

Run (pyctrl backend must be live at --url; reps drive passes, 0 = forever):
    cd pyctrl
    python YbScans/CoolingScan.py blue_amp --amp-lo 0.1 --amp-hi 0.5 --amp-step 0.05 --reps 4
    python YbScans/CoolingScan.py x2d --blue-amp 0.3 --reps 2
    python YbScans/CoolingScan.py h2d --blue-amp 0.3 --reps 2
"""

import argparse
import os
import sys


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _consts():
    from seq_config import SeqConfig
    from consts import Consts
    if not SeqConfig.get().consts:
        SeqConfig.load_real()
    return Consts()


# Default sweep grids (mirror the .m's documented ScannedFreq / ScannedAmp).
DEF_FREQ = (0.10, 0.02, 0.26)   # detuning colon (MHz units below): (0.10:0.02:0.26)*1e6  -> 9 pts
DEF_AMP = (0.10, 0.02, 0.28)    # amp colon: 0.1:0.02:0.28                                -> 10 pts


def _set_fixed(g, c, blue_amp, *, time_s=100e-3, fix_blue=True, fix_x=True, fix_h=True,
               x_freq=None, x_amp=None, h_freq=None, h_amp=None):
    """Set the always-fixed imaging/cooling params (Time, Blue, and whichever 556 beam is NOT swept).

    A param that will be SWEPT must be left unset here -- ScanGroup refuses to ``.scan()`` a param
    already assigned a fixed value ("Cannot scan a fixed parameter").

    The fixed 556 beam defaults to the config (`Consts()`) cooling value, but ``x_freq/x_amp/
    h_freq/h_amp`` (Hz / amplitude) override it -- this is how the optimisation campaign pins one
    beam at its running optimum while sweeping the other. Overriding to an arbitrary value does NOT
    threaten THE ONE RULE: the param->byte path is already verified, so any float64 serialises
    identically (only the MATLAB-equivalence of the *default* config grid is what the oracle pins).
    """
    g().Pushout.Time = float(time_s)
    g().Pushout.Blue.Freq = c.Resonance399Freq + c.Imag399.FreqDetuning
    if fix_blue:
        g().Pushout.Blue.Amp = float(blue_amp)
    if fix_x:
        g().Pushout.Green.X.Freq = (float(x_freq) if x_freq is not None
                                    else c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning)
        g().Pushout.Green.X.Amp = float(x_amp) if x_amp is not None else float(c.Imag399.Cool556.X.Amp)
    if fix_h:
        g().Pushout.Green.h.Freq = (float(h_freq) if h_freq is not None
                                    else c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning)
        g().Pushout.Green.h.Amp = float(h_amp) if h_amp is not None else float(c.Imag399.Cool556.h.Amp)


def _runp(g):
    rp = g.runp()
    rp.NumPerGroup = 4000
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


def build_blue_amp(amps):
    """1-D Pushout.Blue.Amp sweep (X + h fixed at config cooling defaults)."""
    _bootstrap()
    from scan_group import ScanGroup
    c = _consts()
    g = ScanGroup()
    _set_fixed(g, c, blue_amp=0.3, fix_blue=False, fix_x=True, fix_h=True)
    g().Pushout.Blue.Amp.scan(1, [float(a) for a in amps])
    _runp(g)
    return g


def build_2d(beam, blue_amp, freq_det, amps, fixed_freq=None, fixed_amp=None, time_s=100e-3):
    """2-D sweep of one 556 beam: Freq = Resonance556mj0 + det*1e6 (dim 1), Amp (dim 2).

    beam: 'X' or 'h'. freq_det: list of detunings in MHz. amps: list of amplitudes.
    ``fixed_freq`` (Hz) / ``fixed_amp`` pin the OTHER (non-swept) 556 beam at its running optimum
    (default = config cooling value). The grids (freq_det/amps) are free to change every round --
    recenter/expand around the optimum or refine the step as the campaign proceeds.
    """
    _bootstrap()
    from scan_group import ScanGroup
    c = _consts()
    g = ScanGroup()
    # the non-swept beam gets the optimum override (if given)
    fx = dict(x_freq=fixed_freq, x_amp=fixed_amp) if beam == "h" else dict(h_freq=fixed_freq, h_amp=fixed_amp)
    _set_fixed(g, c, blue_amp=blue_amp, time_s=time_s, fix_x=(beam != "X"), fix_h=(beam != "h"), **fx)
    reson = float(c.Resonance556mj0Freq)
    freqs = [reson + d * 1e6 for d in freq_det]
    node = g().Pushout.Green.X if beam == "X" else g().Pushout.Green.h
    node.Freq.scan(1, freqs)
    node.Amp.scan(2, [float(a) for a in amps])
    _runp(g)
    return g


def build():
    """Default config for the A/B byte oracle: the X 2-D scan at Blue.Amp=0.3, .m default grids."""
    _bootstrap()
    from scan_export import matlab_colon
    det = matlab_colon(*DEF_FREQ)
    amp = matlab_colon(*DEF_AMP)
    return build_2d("X", 0.3, det, amp)


def _submit(seqname, g, url, label, reps):
    _bootstrap()
    from yb_start_scan import ybStartScan
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan(seqname, g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, nseq=%d)"
          % (label, did, url or "default", reps, g.nseq()))
    return did


def main():
    _bootstrap()
    from scan_export import matlab_colon
    ap = argparse.ArgumentParser(description="Submit a CoolingScan variant to the pyctrl backend.")
    ap.add_argument("mode", choices=["blue_amp", "x2d", "h2d"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blue-amp", type=float, default=0.3, help="fixed Blue.Amp for x2d/h2d")
    # blue_amp sweep range (colon lo:step:hi)
    ap.add_argument("--amp-lo", type=float, default=0.1)
    ap.add_argument("--amp-hi", type=float, default=0.5)
    ap.add_argument("--amp-step", type=float, default=0.05)
    # 2-D grids (override the .m defaults)
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=DEF_FREQ,
                    help="556 detuning colon in MHz (Freq = Resonance556mj0 + det*1e6)")
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=DEF_AMP,
                    help="556 amp colon")
    ap.add_argument("--fix-freq", type=float, default=None,
                    help="pin the NON-swept 556 beam's Freq (Hz) at its optimum (default: config)")
    ap.add_argument("--fix-amp", type=float, default=None,
                    help="pin the NON-swept 556 beam's Amp at its optimum (default: config)")
    ap.add_argument("--time", type=float, default=100e-3, help="Pushout.Time (s); default 0.1")
    args = ap.parse_args()

    if args.mode == "blue_amp":
        amps = matlab_colon(args.amp_lo, args.amp_step, args.amp_hi)
        g = build_blue_amp(amps)
        _submit("ImagingPushoutSurvivalSeq", g, args.url, "CoolingScan_blueamp", args.reps)
    else:
        beam = "X" if args.mode == "x2d" else "h"
        det = matlab_colon(*args.fdet)
        amp = matlab_colon(*args.famp)
        g = build_2d(beam, args.blue_amp, det, amp, fixed_freq=args.fix_freq, fixed_amp=args.fix_amp,
                     time_s=args.time)
        _submit("ImagingPushoutSurvivalSeq", g, args.url, "CoolingScan_%s2d" % beam, args.reps)


if __name__ == "__main__":
    main()
