"""ImagingLifetimeScan.py -- pyctrl port of ``matlab_new/YbScans/ImagingLifetimeScan.m``.

Builds the imaging-lifetime ScanGroup (seq = ``ImagingPushoutSurvivalSeq``) and submits it to
the RUNNING pyctrl backend over ZMQ (``submit_scan_descriptor``), mirroring
ImagingLifetimeScan.m's ``ybStartScan(TimeImagingScan(), @ImagingPushoutSurvivalSeq)``.

Active scan (mirrors the modified ImagingLifetimeScan.m / TimeImagingScan -- IMAGING during the
push-out: the push-out beams run AT the imaging amplitudes, so the atoms are imaged throughout
the hold and this measures the IMAGING lifetime):
    g().Pushout.Time.scan(1)  = [0.005, 0.1, 1, 2, 4, 8]                         # 6 pts
    g().Pushout.Blue.Freq     = Consts().Resonance399Freq + Consts().Imag399.FreqDetuning
    g().Pushout.Blue.Amp1     = Consts().Imag399.Amp1
    g().Pushout.Blue.Amp2     = Consts().Imag399.Amp2
    g().Pushout.Green.X.Freq  = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.X.FreqDetuning
    g().Pushout.Green.X.Amp   = Consts().Imag399.Cool556.X.Amp
    g().Pushout.Green.h.Freq  = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.h.FreqDetuning
    g().Pushout.Green.h.Amp   = Consts().Imag399.Cool556.h.Amp
``PushouthXStep`` reads Pushout.Time + Pushout.Blue.{Freq,Amp,Amp2} + Pushout.Green.{X,h}.{Freq,Amp}
(PushouthXStep.m:5-13); two ``Imag399Step`` calls => NumImages=2 (image before + after the
push-out => survival vs hold Time, with imaging light ON during the hold).

Notes:
  * The push-out HOLD's freqs + amps are resolved against the LOADING PATTERN this scan images
    (expConfig base (+) ByPattern overlay), so the hold tracks the array rather than bare base.
    Set ``LOADING_PHASE`` (top of file) to image + track a specific array; None uses expConfig's
    SLM.Loading default. ``build()`` loads the real config (the empty test config gives zero
    consts). NOTE: this DIVERGES from the MATLAB twin (no ByPattern in MATLAB) -- the
    tools/check_ab_byte_equality.py "imaginglifetime" case will differ by design.
  * The swept ``Pushout.Time = [0.005, 0.1, 1, 2, 4, 8]`` is an explicit list, wired directly
    (no colon needed; the values are the same float64 in both stacks).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works (yb_analysis env, base, or .venv-engine-py312).

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


# --- SLM loading pattern for THIS scan (the single override point) -------------------------
# The push-out "hold" replays the imaging illumination, so its imaging + cooling values are
# resolved against this pattern's expConfig ByPattern overlay (base < ByPattern) -- the hold
# TRACKS the array you image, not bare base. Set to a server-side phase path to image + track a
# specific array; None -> use expConfig's SLM.Loading default (what a no-pattern scan loads
# anyway). To override one hold value, edit its g().Pushout.* line in build() with a literal.
LOADING_PHASE = "phase/47x47_feedbackwarm4.pt"
LOADING_DEFOCUS = -5            # ANSI z4 loading defocus (rad); applied only when LOADING_PHASE is set


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    # root itself carries expConfig.py (build() loads the real config via Consts()).
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build():
    """The ImagingLifetimeScan ScanGroup (single group, 1-D Pushout.Time sweep).

    Like ImagingLifetimeScan.m's TimeImagingScan, but the push-out hold's imaging/cooling are
    resolved against the loading PATTERN's ByPattern overlay (a pyctrl-only enhancement; MATLAB
    has no per-pattern overlay) so the hold tracks the imaged array. ``runp`` drives the live run
    (NumImages=2) but never the per-seq bytes.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from seq_config import SeqConfig
    from dyn_props import DynProps
    import expConfig_helper

    # Consts() reads SeqConfig.get().consts; the empty default test config has none. Load the
    # real expConfig if it is not already active (the A/B oracle loads it before calling build()).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    # Resolve the hold's imaging/cooling against the loading PATTERN this scan images (base (+)
    # ByPattern overlay) so the hold TRACKS the array, not bare base. Effective pattern mirrors
    # the runner's non-rearrange logic (_first_loading_pattern): LOADING_PHASE wins, else
    # expConfig's every-scan default (SLM.Loading.DefaultPhase when AllScansLoadPattern). The
    # name<-path map matches runner.py (basename, no extension). apply_pattern() returns base
    # UNCHANGED for an unknown/empty pattern, so None + no default = bare base (byte-identical).
    base_store = SeqConfig.get().consts
    loading_cfg = (base_store.get("SLM", {}) or {}).get("Loading", {}) or {}
    eff_phase = LOADING_PHASE
    if not eff_phase and loading_cfg.get("AllScansLoadPattern"):
        eff_phase = str(loading_cfg.get("DefaultPhase") or "")
    pattern = os.path.splitext(os.path.basename((eff_phase or "").replace("\\", "/")))[0]
    c = DynProps(expConfig_helper.apply_pattern(base_store, pattern))
    g = ScanGroup()

    # ---- fixed params (PushouthXStep reads these) -------------------------
    # "Imaging during the push-out": the push-out beams are driven AT the imaging amplitudes
    # (not off) so the atoms are imaged throughout the hold -> this measures the imaging
    # lifetime. Freqs + amps are the Imag399 / Cool556 imaging settings, resolved against the
    # loading PATTERN above (c = base (+) ByPattern[pattern]) so the hold TRACKS the array. To
    # override one value, replace its RHS here with a literal (a scan g() param wins at runtime).
    # NB: a bare consts leaf is a SubProps PROXY -- assigning it directly would store the proxy,
    # not its value (the freqs below resolve only because ``+`` forces _value()). So the amps are
    # wrapped in float() to resolve+coerce to the FLOAT64 the .m's double yields.
    # Beam 1 <- Pushout.Blue.Amp1, beam 2 <- Pushout.Blue.Amp2: PushouthXStep now reads each
    # independently, so the hold images at the pattern's Amp1/Amp2 (e.g. warm4 -> 0.3/0.2).
    g().Pushout.Blue.Freq = c.Resonance399Freq + c.Imag399.FreqDetuning
    g().Pushout.Blue.Amp1 = float(c.Imag399.Amp1)       # beam 1 -> AmpAbsImag
    g().Pushout.Blue.Amp2 = float(c.Imag399.Amp2)       # beam 2 -> Amp399Imag2 (both imaged during hold)

    g().Pushout.Green.X.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.X.FreqDetuning
    g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
    g().Pushout.Green.h.Freq = c.Resonance556mj0Freq + c.Imag399.Cool556.h.FreqDetuning
    g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)

    # ---- swept param: Pushout.Time = [0.005, 0.1, 1, 2, 4, 8] s -----------
    # Explicit list mirroring the modified ImagingLifetimeScan.m (NOT a colon/logspace expr).
    # Every value is the same float64 in Python and MATLAB (0.1 is the identical inexact
    # double both sides; 1/2/4/8 exact), so it is wired directly and byte-verified per point.
    g().Pushout.Time.scan(1, [0.001, 0.005, 0.05, 0.2])

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    # SLM loading pattern (set via LOADING_PHASE at the top of the file): write it + hold the SLM
    # lock + detect with its per-pattern thresholds. This is the SAME pattern the hold's imaging/
    # cooling were resolved against above, so the hold and the readout images stay on one array.
    if LOADING_PHASE:
        rp.loading_phase = LOADING_PHASE
        rp.loading_defocus = LOADING_DEFOCUS
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
