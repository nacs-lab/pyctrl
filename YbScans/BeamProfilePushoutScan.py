"""BeamProfilePushoutScan.py -- map ONE 399 imaging beam's spatial profile via per-site push-out.

Derived from ``StrobeImageScan.py``. Same seq ``ImagingPushoutSurvivalSeq`` (image -> cool ->
PUSHOUT slot -> cool -> image; ``NumImages=2`` => survival). The middle push-out step
(``PushouthXStep``) is repurposed: instead of replaying the full imaging illumination (both 399 beams
+ 556 cooling, as StrobeImageScan does), this scan drives ONLY ONE 399 beam and turns the 556 cooling
OFF, then sweeps that 399 beam's FREQUENCY across the imaging resonance.

Idea: with the 556 cooling off, a single 399 beam heats atoms out of the (assumed uniform-depth) traps
by photon scattering. The fraction pushed out is maximal on resonance and falls off as a Lorentzian in
detuning. The on-resonance push-out amount per site therefore tracks that site's LOCAL 399 intensity --
i.e. fit ``1 - survival`` vs frequency to a Lorentzian PEAK at each site and the peak value is a map of
the imaging beam profile (analysis: ``pyctrl/tmp/beam_profile_fit.py``). Uniform trap depth is the
assumption that makes per-site push-out a clean intensity proxy (deeper traps need more scattering to
eject), so flatten depth first if needed (trap-depth-feedback runbook).

Beam mapping (PushouthXStep reads ``Pushout.Blue.{Amp1,Amp2,Freq}``; both 399 beams share Blue.Freq):
  * beam 1 -> ``AmpAbsImag``   <- ``Pushout.Blue.Amp1``
  * beam 2 -> ``Amp399Imag2``  <- ``Pushout.Blue.Amp2``
``--beam 1`` drives Amp1 at ``--amp`` and pins Amp2=0 (beam 2 off); ``--beam 2`` does the reverse.

The two SURVIVAL imaging pulses (Imag399Step) are untouched -- they use the array's normal imaging
config for clean detection. Only the push-out slot is modified.

Choosing amp + push-out time (the key knobs -- tune so the dip is PARTIAL, not saturated): the readout
is the on-resonance survival dip depth. If on-resonance survival barely drops, raise ``--amp`` or
``--pushout``; if it pegs to ~0 / flat-tops across several points (saturated -> washes out per-site
differences and the Lorentzian can't be located), lower them. Aim for an on-resonance survival of
roughly 0.3-0.6 at typical sites. Defaults are the dose CALIBRATED 2026-06-22 on 33x33_feedback9, beam 1
(amp-bracket ``beam_profile_tune.py``): at 10 ms the on-resonance survival vs amp is 0.06->0.98,
0.094->0.61, 0.13->0.20, >=0.16->~0, so amp 0.1 / 10 ms = ~45% on-resonance push-out (the steep
dose-response amplifies per-site intensity differences into the dip-depth map). The amp->power curve is
steep, so re-bracket for a different array / beam 2 / cooling-on.

Run (pyctrl backend live at --url; --reps = passes per grid point, total shots = reps x n_points):
    cd pyctrl
    python YbScans/BeamProfilePushoutScan.py
    python YbScans/BeamProfilePushoutScan.py --beam 1 --amp 0.1 --pushout 10e-3 --reps 8
    python YbScans/BeamProfilePushoutScan.py --det -15 1 15            # +-15 MHz, 1 MHz step (31 pts)
    python YbScans/BeamProfilePushoutScan.py --cool                   # leave 556 cooling on (NOT a profile)
"""

import argparse
import os
import sys

# ---- scan settings (override on the command line) -------------------------
LOADING_PHASE = "phase/33x33_feedback9.pt"   # SLM loading hologram (+ its ByPattern config)
LOADING_DEFOCUS = -5                          # ANSI z4 loading defocus (rad)
PUSHOUT_TIME = 10e-3                          # push-out (399 strobe) hold (s) -- calibrated 2026-06-22
BEAM = 1                                      # which 399 beam: 1 (AmpAbsImag) or 2 (Amp399Imag2)
AMP = 0.1                                     # the ON beam's 399 amplitude -- ~45% on-res push-out @ 10 ms
COOL = False                                  # 556 cooling during push-out: OFF = clean intensity probe

# Frequency detuning grid as a MATLAB colon (lo, step, hi) in MHz, RELATIVE TO Resonance399Freq.
DET_GRID_MHZ = (-25.0, 2.0, 25.0)             # +-15 MHz, 1 MHz step -> 31 points


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build(det_grid_mhz=DET_GRID_MHZ, amp=AMP, beam=BEAM, pushout_time=PUSHOUT_TIME,
          cool=COOL, loading_phase=LOADING_PHASE, loading_defocus=LOADING_DEFOCUS,
          center_freq=None):
    """ScanGroup: 1-D sweep of Pushout.Blue.Freq over (center +- det grid) MHz.

    ONE 399 beam is on (``--beam`` at ``amp``), the other pinned to 0; the 556 cooling beams during
    the push-out are off (``cool=False``) so the loss is a clean function of the local 399 intensity.
    ``center_freq`` (Hz) sets the sweep center; default = Consts().Resonance399Freq. (Measured 2026-06-22
    on 33x33_feedback9 the true line is ~313.8 MHz, ~3.8 MHz above the config 310 -- center there so the
    +-15 MHz window straddles the dip.)
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from seq_config import SeqConfig
    from consts import Consts

    # Consts() reads SeqConfig.get().consts; a freshly-launched client process has the empty default
    # config. Load the real expConfig if it is not already active (the backend / A/B byte oracle load
    # it before build(); a direct ``python YbScans/BeamProfilePushoutScan.py`` does not).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    g = ScanGroup()
    c = Consts()

    center = float(center_freq) if center_freq else float(c.Resonance399Freq)
    # swept push-out 399 frequency: sweep center + the detuning grid (MATLAB-exact colon -- the
    # value goes straight into the seq bytes; an integer-MHz step is exact in float64, a fractional one
    # needs matlab_colon to match MATLAB's colon to the ULP).
    dets_hz = [float(d) * 1e6 for d in matlab_colon(*det_grid_mhz)]
    freqs = [center + d for d in dets_hz]
    g().Pushout.Blue.Freq.scan(1, freqs)                                   # dim 1 = 399 detuning

    # ONE 399 beam on, the other off (both beams share Blue.Freq -- the swept axis).
    if int(beam) == 1:
        g().Pushout.Blue.Amp1 = float(amp)     # beam 1 (AmpAbsImag) ON
        g().Pushout.Blue.Amp2 = 0.0            # beam 2 (Amp399Imag2) OFF
    elif int(beam) == 2:
        g().Pushout.Blue.Amp1 = 0.0            # beam 1 OFF
        g().Pushout.Blue.Amp2 = float(amp)     # beam 2 ON
    else:
        raise ValueError("beam must be 1 or 2, got %r" % (beam,))

    g().Pushout.Time = float(pushout_time)

    # 556 cooling during the push-out. OFF (amp 0) by default -> the push-out is pure 399 heating, so
    # the per-site loss maps the 399 intensity. ``--cool`` re-enables it at the imaging-cooling config
    # (then this is NO LONGER a clean beam-profile probe -- cooling competes with the push-out).
    if cool:
        g().Pushout.Green.X.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning
        g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
        g().Pushout.Green.h.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning
        g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)
    else:
        g().Pushout.Green.X.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning
        g().Pushout.Green.X.Amp = 0.0
        g().Pushout.Green.h.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning
        g().Pushout.Green.h.Amp = 0.0

    rp = g.runp()
    rp.NumImages = 2          # 2 frames/shot -> survival
    rp.Scramble = 1           # randomise point order within each pass
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = loading_phase        # load the array hologram + apply its ByPattern config
    rp.loading_defocus = loading_defocus
    return g


def main():
    _bootstrap()
    from yb_start_scan import ybStartScan

    ap = argparse.ArgumentParser(
        description="Submit the 1-D 399 push-out FREQUENCY scan (single beam) for a beam-profile map.")
    ap.add_argument("--url", default=None, help="ExptServer URL (default tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=8,
                    help="passes per grid point (total shots = reps x n_points); 0 = run forever")
    ap.add_argument("--det", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=DET_GRID_MHZ,
                    help="Pushout.Blue.Freq detuning colon in MHz, relative to Resonance399Freq "
                         "(default -15 1 15 -> 31 pts)")
    ap.add_argument("--amp", type=float, default=AMP, help="the ON 399 beam's amplitude (default 0.1)")
    ap.add_argument("--beam", type=int, default=BEAM, choices=(1, 2),
                    help="which 399 beam is ON: 1=AmpAbsImag (Blue.Amp1), 2=Amp399Imag2 (Blue.Amp2)")
    ap.add_argument("--pushout", type=float, default=PUSHOUT_TIME,
                    help="Pushout.Time (s); default 10e-3 -- tune for a partial (~0.3-0.6) dip")
    ap.add_argument("--cool", action="store_true",
                    help="leave 556 X/h cooling ON during push-out (default OFF; ON is not a clean profile)")
    ap.add_argument("--loading-phase", default=LOADING_PHASE,
                    help="SLM loading hologram (+ its ByPattern overlay); default %s" % LOADING_PHASE)
    ap.add_argument("--defocus", type=float, default=LOADING_DEFOCUS, help="ANSI z4 loading defocus (rad)")
    ap.add_argument("--center", type=float, default=None,
                    help="sweep center freq in MHz (default Resonance399Freq=310; measured line ~313.8)")
    args = ap.parse_args()

    g = build(det_grid_mhz=tuple(args.det), amp=args.amp, beam=args.beam, pushout_time=args.pushout,
              cool=args.cool, loading_phase=args.loading_phase, loading_defocus=args.defocus,
              center_freq=(args.center * 1e6 if args.center else None))
    n_points = g.nseq()
    # Keep the dashboard's "shots scheduled" total honest: reps x n_points (the NumPerGroup footgun --
    # see the experiment-running skill). reps=0 -> run forever.
    if args.reps and args.reps > 0:
        g.runp().NumPerGroup = args.reps * n_points

    did = ybStartScan(
        "ImagingPushoutSurvivalSeq", g, url=args.url, label="BeamProfilePushout_399",
        rep=args.reps,
        description=(
            "Single-399-beam push-out FREQUENCY scan to map the imaging beam profile. Beam %d on "
            "(amp %.3f), other beam off, 556 cooling %s; push-out %.1f ms; Pushout.Blue.Freq swept "
            "%.1f..%.1f MHz (step %.2f) about Resonance399Freq (=+-%g MHz detuning). Seq "
            "ImagingPushoutSurvivalSeq (NumImages=2 survival), array %s. Fit 1-survival per site to a "
            "Lorentzian peak; the peak value per site = local 399 intensity (assumes uniform trap "
            "depth). Analysis: pyctrl/tmp/beam_profile_fit.py."
            % (args.beam, args.amp, "ON" if args.cool else "OFF", args.pushout * 1e3,
               args.det[0], args.det[2], args.det[1], max(abs(args.det[0]), abs(args.det[2])),
               args.loading_phase)),
    )
    print("submitted BeamProfilePushout_399 -> id %s (url=%s, beam=%d, amp=%.3f, cool=%s, reps=%s, "
          "n_points=%d, shots=%s)"
          % (did, args.url or "default", args.beam, args.amp, args.cool, args.reps, n_points,
             args.reps * n_points if args.reps else "forever"))
    return did


if __name__ == "__main__":
    main()
