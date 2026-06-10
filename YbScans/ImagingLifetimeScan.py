"""ImagingLifetimeScan.py -- pyctrl port of ``matlab_new/YbScans/ImagingLifetimeScan.m``.

Builds the imaging-lifetime ScanGroup (seq = ``ImagingPushoutSurvivalSeq``) and submits it to
the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
ImagingLifetimeScan.m's ``ybStartScan(TimeImagingScan(), @ImagingPushoutSurvivalSeq)``.

Active scan (mirrors the modified ImagingLifetimeScan.m / TimeImagingScan -- IMAGING during the
push-out: the push-out beams run AT the imaging amplitudes, so the atoms are imaged throughout
the hold and this measures the IMAGING lifetime):
    g().Pushout.Time.scan(1)  = [0.005, 0.1, 1, 2, 4, 8]                         # 6 pts
    g().Pushout.Blue.Freq     = Consts().Resonance399Freq + Consts().Imag399.FreqDetuning
    g().Pushout.Blue.Amp      = Consts().Imag399.Amp
    g().Pushout.Green.X.Freq  = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.X.FreqDetuning
    g().Pushout.Green.X.Amp   = Consts().Imag399.Cool556.X.Amp
    g().Pushout.Green.h.Freq  = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.h.FreqDetuning
    g().Pushout.Green.h.Amp   = Consts().Imag399.Cool556.h.Amp
``PushouthXStep`` reads Pushout.Time + Pushout.Blue.{Freq,Amp} + Pushout.Green.{X,h}.{Freq,Amp}
(PushouthXStep.m:5-13); two ``Imag399Step`` calls => NumImages=2 (image before + after the
push-out => survival vs hold Time, with imaging light ON during the hold).

Byte-equality notes (THE ONE RULE):
  * The push-out freqs AND amps are computed FROM ``Consts()`` exactly as the .m does -- so they
    track the live expConfig (e.g. Resonance556mj0Freq, refit 2026-06-05; the Imag399/Cool556
    amplitudes) instead of being frozen here. This requires the real config active, so
    ``build()`` loads it (the empty default test config would give wrong/zero consts).
  * The swept ``Pushout.Time = [0.005, 0.1, 1, 2, 4, 8]`` is an explicit list (NOT a
    colon/logspace expression): every value is the same float64 in Python and MATLAB (``0.1`` is
    the identical inexact double on both sides; 1/2/4/8 are exact), so it is wired directly and
    byte-verified per point by the A/B oracle (``tools/check_ab_byte_equality.py`` against the
    MATLAB twin in ``tools/scan_point_list_ab.m``).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/ImagingLifetimeScan.py                 # short A/B run: rep=3 passes over 6 pts
    python YbScans/ImagingLifetimeScan.py --reps 5
    python YbScans/ImagingLifetimeScan.py --reps 0        # run forever
    python YbScans/ImagingLifetimeScan.py --url tcp://127.0.0.1:1408
"""

import argparse
import os
import sys


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    # root itself carries expConfig.py (build() loads the real config via Consts()).
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build():
    """The ImagingLifetimeScan ScanGroup (single group, 1-D Pushout.Time sweep).

    Mirrors ImagingLifetimeScan.m's TimeImagingScan byte-affecting params only (the dbstack
    scanname/scanfilename metadata are dropped -- they never enter the serialized bytes). The
    push-out Freq fixed params are derived from ``Consts()`` exactly as the .m does, so the real
    config is ensured active. ``runp`` drives the live run (NumImages=2) but never the per-seq
    bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from seq_config import SeqConfig
    from consts import Consts

    # Consts() reads SeqConfig.get().consts; the empty default test config has none. Load the
    # real expConfig if it is not already active (the A/B oracle loads it before calling build()).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    c = Consts()
    g = ScanGroup()

    # ---- fixed params (PushouthXStep reads these) -------------------------
    # "Imaging during the push-out": the push-out beams are driven AT the imaging amplitudes
    # (not off) so the atoms are imaged throughout the hold -> this measures the imaging
    # lifetime. Freqs + amps are the Imag399 / Cool556 imaging settings, all from Consts().
    # NB: a bare Consts() leaf is a SubProps PROXY -- assigning it directly would store the
    # proxy, not its value (the freqs below resolve only because ``+`` forces _value()). So the
    # amps are wrapped in float() to resolve+coerce to the FLOAT64 the .m's double yields.
    g().Pushout.Blue.Freq = c.Resonance399Freq + c.Imag399.FreqDetuning
    g().Pushout.Blue.Amp = float(c.Imag399.Amp)

    g().Pushout.Green.X.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning
    g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
    g().Pushout.Green.h.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning
    g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)

    # ---- swept param: Pushout.Time = [0.005, 0.1, 1, 2, 4, 8] s -----------
    # Explicit list mirroring the modified ImagingLifetimeScan.m (NOT a colon/logspace expr).
    # Every value is the same float64 in Python and MATLAB (0.1 is the identical inexact
    # double both sides; 1/2/4/8 exact), so it is wired directly and byte-verified per point.
    g().Pushout.Time.scan(1, [0.005, 0.1, 1.0, 2.0, 4.0, 8.0])

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
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
    return g


def ImagingLifetimeScan(url=None, reps=3):
    """Build + submit the imaging-lifetime scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    did = ybStartScan("ImagingPushoutSurvivalSeq", g, url=url, label="ImagingLifetimeScan", **opts)
    print("submitted ImagingLifetimeScan -> descriptor id %s (url=%s, reps=%s, 6 time pts 5ms..8s, imaging ON)"
          % (did, url or "default", reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit ImagingLifetimeScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the 6-pt sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    ImagingLifetimeScan(url=args.url, reps=args.reps)
