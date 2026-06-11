"""556AutlerTownesScan.py -- Autler-Townes splitting of the 556 push-out line at 30 G.

Copied from ``RydbergSpectrum556Scan`` (same seq ``RydbergPushoutSurvivalSeq`` ->
``RydbergPushoutStep``, which applies the Ryd bias field + pushes with the 556 Rydberg beam).
The difference: the 308 coupling laser is turned ON, on resonance and at maximum AOM amp, so the
556 push-out line splits into the Autler-Townes doublet (two dips) -- exactly the recipe the
commented block of ``RydbergSpectrum556Scan`` describes ("turn on the 308 UV light on resonance
to see Autler-Townes splitting").

  * **308 on resonance via the 616 EOM:** 308 nm is frequency-doubled 616 nm, so ``Init.EOM616.Freq``
    sets the 308 frequency. We park it on the measured 30 G resonance -- the revival peak from
    ``Revival616Scan`` (scan 20260610173416): 282.77 MHz. (``--eom616`` to retune.)
  * **308 at maximum AOM amp:** ``Pushout.Ryd308.Amp = 0.4`` (max) -> ``AmpAOM308`` (DDS) -- the
    strong coupling field whose Rabi frequency sets the AT splitting. (``--ryd308-amp`` to retune.)
  * **556 = the probe:** swept over a WIDE window (+/-3 MHz, 0.1 MHz) centred on the 30 G single-photon
    resonance (RES0 + slope*30 = 143.184 MHz), wide enough to bracket the two dressed-state dips.

The ``AmpAOM616`` (DDS8) direct-616 beam is a SEPARATE channel from the 616 EOM / 308; it is left
as ``RydbergPushoutStep`` has it (0 during push-out, the Revival616Scan edit), which is how the
282.77 MHz resonance was measured -- it does not gate the 308.

Workflow (the user's two-stage plan):
  1. SHORT run -> inspect the ARRAY-AVERAGE survival vs 556 freq for TWO dips (the AT doublet).
     Widen ``--half`` / retune ``--eom616`` / raise ``--ryd308-amp`` until two dips are clear.
  2. Once two dips are clear, run for >100 reps to get the SITE-RESOLVED splitting (per-site
     two-dip fit; needs ~100+ shots/point/site).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine.

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/556AutlerTownesScan.py                 # 30 G, inspect run (default reps)
    python YbScans/556AutlerTownesScan.py --reps 120      # site-resolved run (>100 reps)
    python YbScans/556AutlerTownesScan.py --half 4 --ryd308-amp 0.4 --eom616 282.77e6
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


def build(field_G=30, eom616_freq=282.77e6, ryd308_amp=0.4, green_amp=None,
          half_mhz=3.0, step_mhz=0.1):
    """ScanGroup for the 30 G 556 Autler-Townes scan (seq = ``RydbergPushoutSurvivalSeq``).

    Sweeps ``Pushout.Green.Freq`` (the 556 probe) over a wide window centred on the field-shifted
    resonance, with the 308 coupling laser on resonance (``Init.EOM616.Freq``) at max AOM amp
    (``Pushout.Ryd308.Amp``). ``green_amp=None`` uses the field-scaled push amp (0.2 @ 0 G ->
    0.5 @ 30 G), matching ``RydbergSpectrum556Scan``.
    """
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    # 0-field push-out center + Zeeman slope (mirrors RydbergSpectrum556Scan's 2026-06-10 fit).
    RES0_MHZ = 107.8049
    ZEEMAN_SLOPE_MHZ_PER_G = 1.1793
    center_mhz = RES0_MHZ + ZEEMAN_SLOPE_MHZ_PER_G * field_G     # 30 G -> 143.184 MHz

    g = ScanGroup()

    # ---- high-field push-out params (RydbergPushoutStep reads these) -------
    # 556 probe push amp: field-scaled (0.2 @ 0 G -> 0.5 @ 30 G) unless overridden. A weaker probe
    # resolves the AT doublet better; raise/lower with --amp if the two dips smear or don't push.
    AMP_AT_0G, AMP_AT_30G = 0.2, 0.5
    if green_amp is None:
        green_amp = AMP_AT_0G + (AMP_AT_30G - AMP_AT_0G) * field_G / 30.0
    g().Pushout.Green.Amp = green_amp
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G      # Gauss -> Ryd coil current (30 -> 30 G)

    # ---- 308 coupling laser ON: on resonance (via 616 EOM) + max AOM amp -> AT splitting ----
    g().Pushout.Ryd308.Amp = ryd308_amp            # 0.4 = max; AmpAOM308 (the coupling Rabi)
    g().Init.EOM616.Freq = eom616_freq             # 308 frequency (doubled 616) on resonance

    # ---- swept probe: Pushout.Green.Freq (556), WIDE window for the AT doublet ----
    freqs = [v * 1e6 for v in matlab_colon(center_mhz - half_mhz, step_mhz, center_mhz + half_mhz)]
    g().Pushout.Green.Freq.scan(1, freqs)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (see RydbergSpectrum556Scan). ---
    # g.runp().loading_phase = "phase/33x33_uniform.pt"
    # g.runp().loading_defocus = -5
    return g


def AutlerTownes556Scan(url=None, reps=4, field_G=30, eom616_freq=282.77e6,
                        ryd308_amp=0.4, green_amp=None, half_mhz=3.0, step_mhz=0.1):
    """Build + submit the 556 Autler-Townes scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(field_G=field_G, eom616_freq=eom616_freq, ryd308_amp=ryd308_amp,
              green_amp=green_amp, half_mhz=half_mhz, step_mhz=step_mhz)
    npts = g().Pushout.Green.Freq.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "556AutlerTownesScan_%dG" % round(field_G)
    did = ybStartScan("RydbergPushoutSurvivalSeq", g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, field=%sG, %d 556 pts, "
          "556 amp %.2f, 308 amp %.2f, EOM616 %.3f MHz, window +-%.1f MHz @ %.0f kHz)"
          % (label, did, url or "default", reps, field_G, npts,
             g().Pushout.Green.Amp(), ryd308_amp, eom616_freq / 1e6, half_mhz, step_mhz * 1e3))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 30 G 556 Autler-Townes scan.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=4,
                    help="passes over the sweep (0 = forever); default 4 for the inspect run; "
                         "use >100 for the site-resolved splitting run")
    ap.add_argument("--field", type=float, default=30,
                    help="bias field in Gauss -> Pushout.BiasCoilCurrent.Ryd (default 30)")
    ap.add_argument("--eom616", type=float, default=282.77e6,
                    help="616-EOM freq (Hz) = 308 resonance (default 282.77e6, the measured 30 G revival)")
    ap.add_argument("--ryd308-amp", type=float, default=0.4,
                    help="308 coupling AOM amp, max 0.4 (default 0.4)")
    ap.add_argument("--amp", type=float, default=None,
                    help="override the 556 probe push amp (else field-scaled 0.5 @ 30 G)")
    ap.add_argument("--half", type=float, default=3.0,
                    help="556 window half-width in MHz (default 3.0; widen if the doublet is clipped)")
    ap.add_argument("--step", type=float, default=0.1,
                    help="556 window step in MHz (default 0.1)")
    args = ap.parse_args()
    AutlerTownes556Scan(url=args.url, reps=args.reps, field_G=args.field, eom616_freq=args.eom616,
                        ryd308_amp=args.ryd308_amp, green_amp=args.amp,
                        half_mhz=args.half, step_mhz=args.step)
