"""556AutlerTownesScan.py -- Autler-Townes splitting of the 556 push-out line (30 G or 50-80 G).

Copied from ``RydbergSpectrum556Scan`` (same seq ``RydbergPushoutSurvivalSeq`` ->
``RydbergPushoutStep``, which applies the Ryd bias field + pushes with the 556 Rydberg beam).
The difference: the 308 coupling laser is turned ON, on resonance and at maximum AOM amp, so the
556 push-out line splits into the Autler-Townes doublet (two dips) -- exactly the recipe the
commented block of ``RydbergSpectrum556Scan`` describes ("turn on the 308 UV light on resonance
to see Autler-Townes splitting").

  * **308 on resonance via the 616 EOM:** 308 nm is frequency-doubled 616 nm, so ``Init.EOM616.Freq``
    sets the 308 frequency. We park it on the measured 30 G resonance -- the revival peak from
    ``Revival616Scan``: 236.5 MHz (scan 20260702134131; 235.38 MHz on 06-29 scan 20260629174417,
    FWHM ~8-10 MHz). Switched 2026-07-02 from the old ~283 MHz peak (282.52 MHz, scan
    20260611122308). (``--eom616`` to retune.)
  * **308 at maximum AOM amp:** ``Pushout.Ryd308.Amp = 0.4`` (max) -> ``AmpAOM308`` (DDS) -- the
    strong coupling field whose Rabi frequency sets the AT splitting. (``--ryd308-amp`` to retune.)
  * **556 = the probe:** swept over a WIDE window (+/-3 MHz, 0.1 MHz) centred on the field-shifted
    single-photon resonance (Resonance556mj0Freq + 1.178 MHz/G * field), wide enough to bracket the
    two dressed-state dips.

FIELD BANDS (mirrors ``RydbergPushoutSurvivalSeq``): ``field_G < 31`` -> ``RydbergPushoutStep``;
``50 <= field_G <= 80`` -> ``RydbergHighFieldPushoutStep``, which switches a SECOND single-pass AOM
(``Freq556RydbergHF`` = 120e6) into a separate high-field 556 path. That +120 MHz optical is reached
on the DOUBLE-pass first AOM 60 MHz LOWER, so the swept window is shifted -60 MHz inside ``build()``
and ``--center`` stays in low-field units at every field. 31-49 G is INVALID (the seq raises).

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
    python YbScans/556AutlerTownesScan.py --field 70      # high-field step, 556 auto -60 MHz,
                                                          #   308 parked at 230.6 MHz
    python YbScans/556AutlerTownesScan.py --reps 120      # site-resolved run (>100 reps)
    python YbScans/556AutlerTownesScan.py --half 4 --ryd308-amp 0.4 --eom616 236.5e6
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=30, eom616_freq=None, ryd308_amp=0.4, green_amp=None,
          half_mhz=3.0, step_mhz=0.1, center_mhz=None):
    """ScanGroup for the 30 G 556 Autler-Townes scan (seq = ``RydbergPushoutSurvivalSeq``).

    Sweeps ``Pushout.Green.Freq`` (the 556 probe) over a wide window centred on the field-shifted
    resonance, with the 308 coupling laser on resonance (``Init.EOM616.Freq``) at max AOM amp
    (``Pushout.Ryd308.Amp``). ``green_amp=None`` uses the field-scaled push amp (0.1 @ 0 G ->
    0.4 @ 30 G), matching ``RydbergSpectrum556Scan`` and ``Revival616Scan``.
    """
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from seq_config import SeqConfig
    from consts import Consts

    # Consts() reads SeqConfig.get().consts; a freshly-launched client process has the empty default
    # config. Load the real expConfig if it is not already active (the backend / byte oracle load it
    # before build(); a direct ``python YbScans/556AutlerTownesScan.py`` does not).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    # 2026-08-18: centre from the Zeeman MODEL (RES0 + slope * field_G), tracking the daily mj=0
    # calibration, exactly as RydbergSpectrum556Scan / Revival616Scan now do -- the three scans MUST
    # stay in lockstep because this one's window is centred on the dip RydbergSpectrum556Scan
    # locates. Was a hard-coded 143.4 (30 G only) with stale RES0 107.8037 / slope 1.1793.
    # ALTERNATIVE worth knowing: the 2026-08-18 20-70 G fit of the RYDBERG push-out line itself gives
    # intercept 108.3184 MHz + slope 1.17652 MHz/G (R^2 0.9999999) -- 240.6 kHz above the mj=0 line,
    # i.e. the mj=0 calibration is NOT the true intercept for this line. Swap the two lines below to
    # use it; left on the mj=0 convention so all three scans agree. Irrelevant at +/-3 MHz anyway.
    RES0_MHZ = float(Consts().Resonance556mj0Freq) / 1e6
    ZEEMAN_SLOPE_MHZ_PER_G = 1.178
    # High-field path offset, in Pushout.Green.Freq (first, double-pass AOM) units -- see the
    # swept-probe block below for the derivation (120 MHz single-pass / 2).
    HF_AOM_OFFSET_MHZ = 60.0
    if center_mhz is None:
        center_mhz = RES0_MHZ + ZEEMAN_SLOPE_MHZ_PER_G * field_G
    center_mhz = float(center_mhz)

    # 308 park frequency (616-EOM). Defaults per FIELD BAND, from measured revival peaks:
    #   < 31 G  -> 236.5 MHz (the 30 G revival, scan 20260702134131)
    #   50-80 G -> 230.6 MHz -- the 2026-08-18 revival series measured 231.016 / 230.825 / 230.622 MHz
    #              at 50 / 60 / 70 G (Lorentzian + linear baseline), spread only 0.394 MHz and slope
    #              -0.0197 MHz/G, i.e. the revival is FIELD-INDEPENDENT across the whole HF band, so
    #              one default covers 50-80 G. ``--eom616`` still overrides.
    if eom616_freq is None:
        eom616_freq = 236.5e6 if field_G < 31 else 230.6e6
    eom616_freq = float(eom616_freq)

    g = ScanGroup()

    # ---- high-field push-out params (RydbergPushoutStep reads these) -------
    # 556 probe push amp: field-scaled (0.1 @ 0 G -> 0.4 @ 30 G) unless overridden. The 30 G value
    # MUST match RydbergSpectrum556Scan / Revival616Scan -- the whole 30 G set runs on ONE amp so
    # the fed-forward centers (30 G dip -> revival -> this scan) are comparable (2026-08-01 user
    # directive; was 0.15 until then, which put the AT doublet center -335 kHz off the 0.4-push dip).
    # A weaker probe resolves the doublet better -- use --amp for a deliberate one-off, not silently.
    # 2026-08-03 (user directive): 0.4 @ 30 G is TOO HIGH -> 0.15 (also what every run on/before
    # 08-01 actually pushed). Kept in lockstep with RydbergSpectrum556Scan / Revival616Scan.
    # 2026-08-18: flat 0.15 at every field, matching RydbergSpectrum556Scan / Revival616Scan
    # (both dropped the field-scaling on the same date). Was
    # ``0.1 + (0.15-0.1)*field_G/30``, which extrapolated to 0.217 at 70 G and so silently broke
    # the "whole set on ONE push amp" rule the comment above depends on.
    if green_amp is None:
        green_amp = 0.15
    g().Pushout.Green.Amp = green_amp
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G      # Gauss -> Ryd coil current (30 -> 30 G)

    # ---- 308 coupling laser ON: on resonance (via 616 EOM) + max AOM amp -> AT splitting ----
    g().Pushout.Ryd308.Amp = ryd308_amp            # 0.4 = max; AmpAOM308 (the coupling Rabi)
    g().Init.EOM616.Freq = eom616_freq             # 308 frequency (doubled 616) on resonance

    # ---- swept probe: Pushout.Green.Freq (556), WIDE window for the AT doublet ----
    freqs = [v * 1e6 for v in matlab_colon(center_mhz - half_mhz, step_mhz, center_mhz + half_mhz)]
    # 2026-08-18: from ~50 G up the seq takes RydbergHighFieldPushoutStep, which switches on a
    # SECOND, single-pass AOM (``Freq556RydbergHF`` = 120e6) in a separate high-field 556 Rydberg
    # path. That adds +120 MHz optical; the swept first AOM is DOUBLE-pass, so the same optical
    # frequency is reached at a Pushout.Green.Freq 60 MHz LOWER. Keep ``center_mhz`` in the low-field
    # convention (143.5-like numbers) at every field and shift here. Branch thresholds mirror
    # RydbergPushoutSurvivalSeq (low field < 31 G, high field 50-80 G; 31-49 G raises).
    if 50 <= field_G <= 80:
        freqs = [f - HF_AOM_OFFSET_MHZ * 1e6 for f in freqs]
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
    g.runp().loading_phase = "phase/33x33_feedback11.pt"
    # 2026-08-10: loading plane now comes from the per-array config
    # (ByPattern[<pattern>].SLM.Loading.Defocus -> slm_runtime._pattern_defocus);
    # setting rp.loading_defocus here would override it, so it is left unset.
    return g


def AutlerTownes556Scan(url=None, reps=4, field_G=30, eom616_freq=None,
                        ryd308_amp=0.4, green_amp=None, half_mhz=3.0, step_mhz=0.1,
                        center_mhz=None):
    """Build + submit the 556 Autler-Townes scan. Returns the queued descriptor id.

    ``eom616_freq=None`` takes the per-field-band default in build() (236.5 MHz below 31 G,
    230.6 MHz for 50-80 G). Was a hard-coded 233.78e6 here that matched NEITHER the build()
    default (236.5e6) nor the CLI default (282e6) -- three different values for one knob.
    """
    from yb_start_scan import ybStartScan

    # Normalise the field once: the ``round(field_G)`` label + the band tests would otherwise see a
    # None/str from a programmatic call. 30 G matches the --field default.
    field_G = 30.0 if field_G is None else float(field_G)

    g = build(field_G=field_G, eom616_freq=eom616_freq, ryd308_amp=ryd308_amp,
              green_amp=green_amp, half_mhz=half_mhz, step_mhz=step_mhz,
              center_mhz=center_mhz)
    npts = g().Pushout.Green.Freq.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "556AutlerTownesScan_%dG" % round(field_G)
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, field=%sG, %d 556 pts, "
          "556 amp %.2f, 308 amp %.2f, EOM616 %.3f MHz, window +-%.1f MHz @ %.0f kHz)"
          % (label, did, url or "default", reps, field_G, npts,
             g().Pushout.Green.Amp(), ryd308_amp, g().Init.EOM616.Freq() / 1e6,
             half_mhz, step_mhz * 1e3))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 556 Autler-Townes scan (30 G or the 50-80 G high-field band).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=5,
                    help="passes over the sweep (0 = forever); default 5 for the inspect run; "
                         "use >100 for the site-resolved splitting run")
    ap.add_argument("--field", type=float, default=30,
                    help="bias field in Gauss -> Pushout.BiasCoilCurrent.Ryd (default 30); "
                         "valid bands: < 31 G (low-field step) or 50-80 G (high-field step, 556 "
                         "probe window auto-shifted -60 MHz); 31-49 G raises in the seq")
    ap.add_argument("--eom616", type=float, default=None,
                    help="616-EOM freq (Hz) = the 308 park frequency (default: per field band -- "
                         "236.5e6 below 31 G, the 30 G revival from scan 20260702134131; 230.6e6 "
                         "for 50-80 G, the 2026-08-18 revival series which is field-independent "
                         "across that band). Was 282e6, the pre-2026-07-02 value its own help text "
                         "already described as superseded")
    ap.add_argument("--ryd308-amp", type=float, default=0.4,
                    help="308 coupling AOM amp, max 0.4 (default 0.4)")
    ap.add_argument("--amp", type=float, default=None,
                    help="override the 556 probe push amp (else 0.15 at every field, in lockstep "
                         "with RydbergSpectrum556Scan / Revival616Scan)")
    ap.add_argument("--half", type=float, default=3.0,
                    help="556 window half-width in MHz (default 3.0; widen if the doublet is "
                         "clipped). Was 1.5 here while build() and this help both said 3.0")
    ap.add_argument("--step", type=float, default=0.1,
                    help="556 window step in MHz (default 0.1)")
    ap.add_argument("--center", type=float, default=None,
                    help="556 window centre in MHz, in LOW-FIELD (first double-pass AOM) units "
                         "(default: the Zeeman model, Resonance556mj0Freq + 1.178 MHz/G * field, so "
                         "no longer 30 G-only); pass the day's measured bare dip to re-target. The "
                         "50-80 G -60 MHz shift is applied on top")
    args = ap.parse_args()
    AutlerTownes556Scan(url=args.url, reps=args.reps, field_G=args.field, eom616_freq=args.eom616,
                        ryd308_amp=args.ryd308_amp, green_amp=args.amp,
                        half_mhz=args.half, step_mhz=args.step, center_mhz=args.center)
