"""StrobeImageScan.py -- 2-D sweep of the two 399 strobe-imaging beam amplitudes.

Runs ``ImagingPushoutSurvivalSeq`` (image -> cool -> PUSHOUT slot -> cool -> image; ``NumImages=2``
=> survival). The middle push-out step is ``PushouthXStep``: it drives the two 399 beams + the 556
X/h beams for the push-out time. This scan repurposes that slot as a strobe-imaging pulse by pinning
every ``Pushout.*`` parameter to its imaging/cooling config value (PushouthXStep's own defaults are
the push-out config, so the scan sets them explicitly below).

This scan sweeps the two 399 strobe-beam amplitudes independently:

  * beam 1 -> ``AmpAbsImag``   <- ``Pushout.Blue.Amp1`` (dim 1)
  * beam 2 -> ``Amp399Imag2``  <- ``Pushout.Blue.Amp2`` (dim 2)

Everything else is pinned to its imaging value: the 399 strobe FREQUENCY
(``Resonance399Freq + Imag399.FreqDetuning``) and the 556 X/h cooling (``Imag399.Cool556.{X,h}``)
are set explicitly below, and the two survival imaging pulses use the config imaging params.
Push-out/strobe hold is 200 ms; array loaded is ``47x47_feedbackwarm4`` (declaring the loading
pattern also applies that array's per-pattern config overlay).

Run (pyctrl backend live at --url; --reps = passes per grid point, total shots = reps x n_points):
    cd pyctrl
    python YbScans/StrobeImageScan.py
    python YbScans/StrobeImageScan.py --reps 8
    python YbScans/StrobeImageScan.py --amp1 0.2 0.03 0.5 --amp2 0.01 0.03 0.25 --pushout 0.2
"""

import argparse
import os
import sys

# ---- scan settings (override on the command line) -------------------------
LOADING_PHASE = "phase/33x33_uniform.pt"   # SLM loading hologram (+ its ByPattern config)
LOADING_DEFOCUS = -5                             # ANSI z4 loading defocus (rad)
PUSHOUT_TIME = 200e-3                            # strobe (push-out slot) hold (s)

# 399 strobe amplitude grids as MATLAB colons (lo, step, hi).
AMP1_GRID = (0.05, 0.05, 0.5)                     # beam 1 (AmpAbsImag)
AMP2_GRID = (0.05, 0.03, 0.4)                   # beam 2 (Amp399Imag2)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build(amp1_grid=AMP1_GRID, amp2_grid=AMP2_GRID, pushout_time=PUSHOUT_TIME,
          loading_phase=LOADING_PHASE, loading_defocus=LOADING_DEFOCUS):
    """ScanGroup: Pushout.Blue.Amp1 (dim 1) x Pushout.Blue.Amp2 (dim 2).

    Only the two 399 strobe amplitudes are swept; the 399 frequency + 556 X/h cooling are pinned to
    their imaging values below (PushouthXStep reads Pushout.Blue.{Amp,Amp2,Freq} and
    Pushout.Green.{X,h}.{Freq,Amp} -- sweep any of those if needed).
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from seq_config import SeqConfig
    from consts import Consts

    # Consts() reads SeqConfig.get().consts; a freshly-launched client process has the empty
    # default config. Load the real expConfig if it is not already active (the backend / A/B byte
    # oracle load it before build(); a direct ``python YbScans/StrobeImageScan.py`` does not).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    g = ScanGroup()
    c = Consts()
    
    g().Pushout.Blue.Amp1.scan(1, [float(a) for a in matlab_colon(*amp1_grid)])   # beam 1 (PushouthXStep reads Blue.Amp1)
    g().Pushout.Blue.Amp2.scan(2, [float(a) for a in matlab_colon(*amp2_grid)])   # beam 2
    g().Pushout.Blue.Freq = c.Resonance399Freq + c.Imag399.FreqDetuning   # strobe freq default = imaging line
    g().Pushout.Time = float(pushout_time)                                        # 200 ms strobe
    
    # Setting the Green X/h during pushout to the cooling config values when we are not scanning
    g().Pushout.Green.X.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning
    g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
    g().Pushout.Green.h.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning
    g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)

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

    ap = argparse.ArgumentParser(description="Submit the 2-D 399 strobe-amplitude StrobeImageScan.")
    ap.add_argument("--url", default=None, help="ExptServer URL (default tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=5,
                    help="passes per grid point (total shots = reps x n_points); 0 = run forever")
    ap.add_argument("--amp1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=AMP1_GRID,
                    help="Pushout.Blue.Amp1 colon -- beam 1 (AmpAbsImag)")
    ap.add_argument("--amp2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=AMP2_GRID,
                    help="Pushout.Blue.Amp2 colon -- beam 2 (Amp399Imag2)")
    ap.add_argument("--pushout", type=float, default=PUSHOUT_TIME, help="Pushout.Time (s); default 0.2")
    ap.add_argument("--loading-phase", default=LOADING_PHASE,
                    help="SLM loading hologram (+ its ByPattern overlay); default 47x47_feedbackwarm4")
    ap.add_argument("--defocus", type=float, default=LOADING_DEFOCUS, help="ANSI z4 loading defocus (rad)")
    args = ap.parse_args()

    g = build(args.amp1, args.amp2, args.pushout,
              loading_phase=args.loading_phase, loading_defocus=args.defocus)
    n_points = g.nseq()
    # Keep the dashboard's "shots scheduled" total honest: reps x n_points (see the
    # NumPerGroup footgun in the experiment-running skill). reps=0 -> run forever.
    if args.reps and args.reps > 0:
        g.runp().NumPerGroup = args.reps * n_points

    did = ybStartScan(
        "ImagingPushoutSurvivalSeq", g, url=args.url, label="StrobeImageScan_pushout2d",
        rep=args.reps,
        description=(
            "2-D sweep of the two 399 strobe-imaging beam amplitudes: Pushout.Blue.Amp1 "
            "(beam 1, AmpAbsImag) x Pushout.Blue.Amp2 (beam 2, Amp399Imag2), at %.0f ms strobe, "
            "array 47x47_feedbackwarm4. Seq ImagingPushoutSurvivalSeq; the push-out step "
            "(PushouthXStep) replays imaging illumination -- 399 freq + 556 X/h cooling pinned "
            "to the imaging/cooling config and only the two 399 amplitudes vary -- to find "
            "the (Amp1, Amp2) pair that maximises survival."
            % (args.pushout * 1e3)),
    )
    print("submitted StrobeImageScan_pushout2d -> id %s (url=%s, reps=%s, n_points=%d, shots=%s)"
          % (did, args.url or "default", args.reps, n_points,
             args.reps * n_points if args.reps else "forever"))
    return did


if __name__ == "__main__":
    main()
