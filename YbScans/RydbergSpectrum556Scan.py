"""RydbergSpectrum556Scan.py -- high-field 556 push-out spectrum (Rydberg push-out step).

The high-field sibling of ``Spectrum556Scan``. It submits ``RydbergPushoutSurvivalSeq`` (the
``RydbergPushoutStep`` variant that actually APPLIES the Ryd bias field), so the validated
0-field calibration path (``Spectrum556Scan`` -> ``PushoutSurvivalSeq`` -> ``PushoutStep``) is
left untouched.

High-field model (from the retired Spectrum556Scan.m comments):
    0 field   -> ~107.7503 MHz (the current mj=0 resonance, expConfig Resonance556mj0Freq)
    20 G      -> ~131 MHz
    30 G      -> ~143 MHz
i.e. the feature shifts roughly LINEARLY with field. This scan reproduces that as a *rigid shift*
of the 0-field window: the swept frequencies are the 0-field window (107.5:0.01:107.9) MHz with a
constant ``ZEEMAN_SLOPE_MHZ_PER_G * field_G`` added (the "linear addition to the 0-field one").

  * ``field_G`` (Gauss) maps DIRECTLY to ``Pushout.BiasCoilCurrent.Ryd`` (30 -> 1.5 V on VRydCoil
    -> 30 G), which ``RydbergPushoutStep`` ramps on for the push-out.
  * Push-out is stronger/shorter than the 0-field calibration: the amp scales LINEARLY with
    field (0.2 @ 0 G -> 0.4 @ 30 G), time 1 ms (edit AMP_AT_0G/AMP_AT_30G in build() to retune).
  * ``field_G=0`` degenerates to the unshifted window at 0 G (Ryd=0), but still uses the Rydberg
    push-out step + strong push -- for the *validated* 0-field line use ``Spectrum556Scan`` instead.

NOTE on the narrow window: the rigid shift keeps the 0.4 MHz / 10 kHz 0-field window, so at 30 G a
~0.05 MHz/G slope error already walks the 0.4 MHz window off the line. Tune ``ZEEMAN_SLOPE_MHZ_PER_G``
in build() if the dip isn't bracketed.

The 0.01-MHz colon step is not integer-valued, so the swept window uses
:func:`scan_export.matlab_colon` (a naive ``a+k*step`` drifts 1 ULP); the field shift is added
in MHz before the ``*1e6``.

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/RydbergSpectrum556Scan.py --field 30
    python YbScans/RydbergSpectrum556Scan.py --field 0          # 0 G, unshifted, Rydberg step
    python YbScans/RydbergSpectrum556Scan.py --field 30 --reps 0   # run forever
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


def build(field_G=0):
    """ScanGroup for the high-field 556 push-out spectrum (seq = ``RydbergPushoutSurvivalSeq``).

    The swept ``Pushout.Green.Freq`` is centred at ``RES0 + ZEEMAN_SLOPE * field_G`` (the mj=0
    resonance shifted linearly by the field). ``field_G`` drives the Ryd bias coil
    (``BiasCoilCurrent.Ryd``); the push-out amp scales linearly with field (0.2 @ 0 G ->
    0.4 @ 30 G), time 1 ms.

    This is the SHORT-SCAN / coarse-locate configuration: at field the window is wide (+/-0.5 MHz)
    and coarse (50 kHz) so the dip is bracketed despite slope/drift uncertainty; at 0 G it is the
    narrow (+/-0.2 MHz, 10 kHz) known window. Once a field's dip is located, narrow ``COARSE_*``
    here and re-run for the fine scan.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    # 0-field push-out center (MHz) + its Zeeman shift per Gauss, BOTH fit from the 2026-06-10
    # finer push-out spectra (Lorentzian centers at 0/5/10/20/30 G:
    # 107.803/113.714/119.605/131.355/143.185 MHz, linear R^2=1.0000).
    # RES0 is the fitted 0-field intercept -- this strong-push Rydberg line sits ~55 kHz ABOVE the
    # expConfig mj=0 calibration (107.7503, the prior RES0_MHZ); slope was 1.18 (retired
    # Spectrum556Scan.m windows). center_mhz = RES0 + slope*field now reproduces the fit line.
    RES0_MHZ = 107.77
    ZEEMAN_SLOPE_MHZ_PER_G = 1.178

    # Window half-width + step (MHz): coarse + wide at field (locate), fine + narrow at 0 G (known).
    # Widened 0.5 -> 1.5 MHz half-width 2026-06-12: the +/-0.5 MHz window showed NO dip at 30 G
    # (scan 20260612102118, flat ~0.92-0.96), so the line drifted >0.5 MHz out of the old window;
    # the wider window + stronger push (run with --amp 0.6) re-locates it. Narrow back once found.
    COARSE_HALF_MHZ, COARSE_STEP_MHZ = 1.5, 0.05
    FINE_HALF_MHZ, FINE_STEP_MHZ = 0.50, 0.03

    center_mhz = RES0_MHZ + ZEEMAN_SLOPE_MHZ_PER_G * field_G
    half_mhz, step_mhz = (FINE_HALF_MHZ, FINE_STEP_MHZ) if field_G == 0 \
        else (COARSE_HALF_MHZ, COARSE_STEP_MHZ)

    g = ScanGroup()

    # ---- high-field push-out params (RydbergPushoutStep reads these) -------
    # Push-out amp scales LINEARLY with field: weaker push at low field, stronger at high
    # field -- 0.2 @ 0 G -> 0.4 @ 30 G, i.e. amp = 0.2 + (0.4 - 0.2) * field_G / 30.
    # (Pure linear: extrapolates for field > 30 G; the `amp=` arg still overrides this.)
    AMP_AT_0G, AMP_AT_30G = 0.2, 0.5
    g().Pushout.Green.Amp = AMP_AT_0G + (AMP_AT_30G - AMP_AT_0G) * field_G / 30.0
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G      # Gauss -> Ryd coil current (30 -> 30 G)

    # ---- swept param: Pushout.Green.Freq, centred on the field-shifted resonance ----
    freqs = [v  * 1e6 for v in matlab_colon(center_mhz - half_mhz, step_mhz, center_mhz + half_mhz)]
    g().Pushout.Green.Freq.scan(1, freqs)

    # ---- turn on the 308 UV light on resonance to see Autler-Townes splitting ---- 
    # g().Pushout.Ryd308.Amp = 0.4;
    # g().Init.EOM616.Freq = 284.04e6; %252.48e6;
    
    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
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


def RydbergSpectrum556Scan(url=None, reps=3, field_G=0, amp=None):
    """Build + submit the high-field 556 spectrum scan. Returns the queued descriptor id.

    ``amp`` (if given) overrides the field-default ``Pushout.Green.Amp`` -- for live iteration on
    the push strength (the flat-high / no-push case wants a stronger push).
    """
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(field_G=field_G)
    if amp is not None:
        g().Pushout.Green.Amp = amp
    npts = g().Pushout.Green.Freq.size(1)
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    label = "RydbergSpectrum556Scan_%dG" % round(field_G)
    did = ybStartScan("RydbergPushoutSurvivalSeq", g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, field=%sG, %d freq pts)"
          % (label, did, url or "default", reps, field_G, npts))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the high-field RydbergSpectrum556Scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    ap.add_argument("--field", type=float, default=0,
                    help="bias field in Gauss -> Pushout.BiasCoilCurrent.Ryd (default 0)")
    ap.add_argument("--amp", type=float, default=None,
                    help="override Pushout.Green.Amp (else field-default 0.15@20G / 0.22 else)")
    args = ap.parse_args()
    RydbergSpectrum556Scan(url=args.url, reps=args.reps, field_G=args.field, amp=args.amp)
