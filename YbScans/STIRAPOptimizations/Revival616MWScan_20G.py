"""Revival616Scan.py -- 616-EOM "revival" sweep at 30 G with the 556 fixed on resonance.

Sits the 556 push-out on the measured 30 G resonance and sweeps the 616-EOM frequency
(``Init.EOM616.Freq``) to look for a survival revival. It is the field-on counterpart of the
retired MATLAB ``Spectrum308Scan.m`` "revival" recipe (same 30 G config: ``Ryd308.Amp=0.2``,
``Amp369=0``, the ``(277:0.5:287) MHz`` 616 window centred on the 30 G EOM value ~282.89 MHz),
but it runs ``RydbergPushoutSurvivalSeq`` instead of ``PushoutSurvivalSeq`` -- because:

  * ``Spectrum308Scan`` -> ``PushoutSurvivalSeq`` -> ``PushoutStep`` applies **NO bias field** and
    never fires 308 (``PushoutStep`` only ever ``add('VRydCoil', 0)`` / ``add('AmpAOM308', 0)``),
    so it can't actually be at 30 G despite setting ``BiasCoilCurrent.Ryd``.
  * ``RydbergPushoutSurvivalSeq`` -> ``RydbergPushoutStep`` applies the Ryd field
    (``BiasCoilCurrent.Ryd`` -> ``VRydCoil``; 30 -> 1.5 V -> 30 G), fires 308
    (``Pushout.Ryd308.Amp`` -> ``AmpAOM308``), and pushes with the 556 Rydberg beam at
    ``Pushout.Green.Freq``. This is the SAME seq ``RydbergSpectrum556Scan`` used to measure the
    30 G resonance (143.185 MHz, 2026-06-10 finer fit).

Byte-affecting swept axis: ``Init.EOM616.Freq`` -- it sets the slow-EOM ramp target (and the ramp
duration), so every sweep point genuinely re-ramps the 616 EOM and enters the FPGA/NI bytecode
(unlike ``Spectrum308Scan``'s ``MRabi.Freq`` microwave axis, which is inert / un-driven).

The 556 push-out is held fixed at the 30 G resonance ``RES0 + ZEEMAN_SLOPE * 30`` (= 143.184 MHz;
fitted center 143.185 MHz), with the 30 G push amp (0.4) -- the SAME push conditions under which
that resonance was measured, so the 556 actually sits on the line. Override the push amp with
``--green-amp`` / the 308 amp with ``--ryd308-amp`` for live iteration.

The 0.5-MHz colon window ``(277:0.5:287)`` is integer-on-0.5 (exactly representable) but uses
:func:`scan_export.matlab_colon` for consistency with the other scans; the field shift is in MHz
before the ``*1e6``.

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/Revival616Scan.py                       # 30 G, 616 = (277:0.5:287) MHz, 21 pts
    python YbScans/Revival616Scan.py --reps 5
    python YbScans/Revival616Scan.py --green-amp 0.22       # weaker push (Spectrum308 value)
    python YbScans/Revival616Scan.py --reps 0              # run forever
"""

import argparse
import numpy as np

# MW knobs (overridden by the CLI in __main__)
MW_GAIN = 8000
MW_LO = 11300.0
MW_HI = 11365.0
MW_PTS = 27

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=30, green_amp=0.15, ryd308_amp=0.4, green_freq_mhz=None):
    """ScanGroup for the 30 G 616-revival sweep (seq = ``RydbergPushoutSurvivalSeq``).

    Fixes ``Pushout.Green.Freq`` on the field-shifted 556 resonance and sweeps
    ``Init.EOM616.Freq`` over the 30 G revival window. ``field_G`` drives the Ryd bias coil
    (``BiasCoilCurrent.Ryd``); ``green_amp`` is the 556 Rydberg push amp (30 G default 0.4);
    ``ryd308_amp`` is the 308 pulse amp (max 0.4).

    ``green_freq_mhz`` (if given) OVERRIDES the model-predicted 556 resonance ``RES0+slope*field``
    with an explicit MHz value -- use the freshly LOCATED 30 G dip when it has drifted out of the
    model window (e.g. 2026-06-12: measured 142.281 MHz vs model 143.184 MHz). This keeps the
    revival's 556 actually on the line WITHOUT retuning the calibration constants on one day's drift.
    """
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    # 556 push-out resonance (MHz): mirrors RydbergSpectrum556Scan's calibration (2026-06-10 fit).
    # RES0_MHZ = 107.8049
    # ZEEMAN_SLOPE_MHZ_PER_G = 1.1793
    res556_mhz = 143.5368 #RES0_MHZ + ZEEMAN_SLOPE_MHZ_PER_G * field_G   # 30 G -> 143.184 MHz (model)
    if green_freq_mhz is not None:
        res556_mhz = float(green_freq_mhz)   # explicit override: the located dip after drift

    # 616-EOM sweep window (MHz): the Spectrum308Scan.m "revival" window, centred near the 30 G
    # EOM value (~282.89 MHz). 21 pts @ 0.5 MHz. Edit these to re-centre / refine.
    # 2026-07-22: revival back near ~234 MHz (mj-1 pi regime); bracket 220-250 @ 1 MHz (31 pts).
    # (2026-07-21 sigma-pol test used 260-300 @ 1 MHz for the ~282 MHz sigma regime; the historical
    # mj-1 pi revival is ~234, comment note "210,1,260 = mj-1 pi revival ~234".)
    # 2026-07-23: back to the mj-1 pi revival window (556 on the 30 G line; revival peak ~234 MHz).
    # 210-260 @ 1 MHz = 51 pts (daily-scan runbook). The 275-290 sigma-regime window (~282 MHz) is the
    # 07-21 sigma-pol test; DO NOT use it for mj-1 operation.
    EOM_LO_MHZ, EOM_STEP_MHZ, EOM_HI_MHZ = 210, 1, 260

    g = ScanGroup()


    # ---- swept axis: Init.EOM616.Freq (byte-affecting; slow-EOM ramp target) ----
    eom_freqs = [v * 1e6 for v in matlab_colon(EOM_LO_MHZ, EOM_STEP_MHZ, EOM_HI_MHZ)]
    # 2026-08-12 REVIVAL-DESTRUCTION MW CHECK AT 20 G.
    # 616-EOM PINNED at today's measured 20 G revival peak (236.0263 MHz, scan 20260812105321).
    # The swept axis is now the QICK microwave frequency: driving the 3S1<->3P2 transition
    # empties the shelved Rydberg S state and DESTROYS the revival -> survival DIPS on resonance.
    # Independent of the STIRAP round trip (this is the incoherent RydbergPushout path).
    g().Init.EOM616.Freq = 236.0263e6
    MW_FREQ_MHZ = [round(float(v), 6) for v in np.linspace(MW_LO, MW_HI, MW_PTS)]
    g().QICK.template = "Sine"
    g().QICK.freq.scan(1, MW_FREQ_MHZ)                # SWEPT dim 1
    g().QICK.gain = MW_GAIN
    # 2026-08-12 ROUND 2: duration MUST cover the whole push-out. Round 1 used 20 us inside a
    # 1 ms Pushout.Time = 2% duty cycle -> max possible revival destruction ~2% = the noise floor,
    # and the scan came back flat (0.42-0.48 over 11300-11365, max-min 1.9 sigma). The revival
    # itself was healthy (~0.45), so that null was duty cycle, not physics.
    # build_sine/_split_duration chunk a long tone into loop(N,[Sine]), so 1 ms is supported.
    g().QICK.duration = 1e-3                          # == Pushout.Time: MW on for the whole window
    g().QICK.rabi_freq = 4.825e6
    g().QICK.wait_time = 1e-6
    g().QICK.phase = 0.0
    g.runp().QICK = True                              # arm the QICK; RydbergPushoutStep pulses TTLQickTrig

    # ---- 556 push-out fixed ON the 30 G resonance (RydbergPushoutStep: 556 Rydberg beam) ----
    g().Pushout.Green.Freq = res556_mhz * 1e6   # 143.184 MHz @ 30 G
    # 2026-08-03 (user directive): 0.4 @ 30 G is TOO HIGH -> 0.15 (also what every run on/before
    # 08-01 actually pushed). Kept in lockstep with RydbergSpectrum556Scan's AMP_AT_30G.
    # 30 G push amp: MUST match RydbergSpectrum556Scan (0.15 @ 30 G) -- this scan's 556 sits on the
    # dip THAT scan located, so a different push amp saturates/broadens differently and the
    # fed-forward center is off (2026-08-01 user directive; was 0.15 until then).
    g().Pushout.Green.Amp = green_amp

    # ---- 308 + ionization (RydbergPushoutStep fires AmpAOM308 = Pushout.Ryd308.Amp) ----
    g().Pushout.Ryd308.Amp = ryd308_amp         # max 0.4
    g().Pushout.Amp369 = 0                       # read-but-unused on the byte path (faithful)

    # ---- push-out timing + Rydberg bias field ----------------------------
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G    # Gauss -> VRydCoil (30 -> 1.5 V -> 30 G)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    g.runp().loading_phase = "phase/33x33_feedback11.pt"   # server-side WGS phase path
    # 2026-08-10: loading plane now comes from the per-array config
    # (ByPattern[<pattern>].SLM.Loading.Defocus -> slm_runtime._pattern_defocus);
    # setting g.runp().loading_defocus here would override it, so it is left unset.
    return g


def Revival616Scan(url=None, reps=3, field_G=30, green_amp=0.15, ryd308_amp=0.4,
                   green_freq_mhz=None):
    """Build + submit the 30 G 616-revival scan. Returns the queued descriptor id."""
    from yb_start_scan import ybStartScan

    g = build(field_G=field_G, green_amp=green_amp, ryd308_amp=ryd308_amp,
              green_freq_mhz=green_freq_mhz)
    npts = g().Init.EOM616.Freq.size(1)
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    label = "Revival616Scan_%dG" % round(field_G)
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, field=%sG, %d EOM616 pts, "
          "556@%.3f MHz amp %.2f, 308 amp %.2f)"
          % (label, did, url or "default", reps, field_G, npts,
             g().Pushout.Green.Freq() / 1e6, green_amp, ryd308_amp))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 30 G 616-EOM revival scan (556 on resonance).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    ap.add_argument("--field", dest="field_G", type=float, default=None,
                    help="bias field in Gauss -> Pushout.BiasCoilCurrent.Ryd (default 30)")
    ap.add_argument("--556-amp", dest="green_amp", type=float, default=0.15,
                    help="556 Rydberg push-out amp (default 0.15 = the 30 G value used by "
                         "RydbergSpectrum556Scan and 556AutlerTownesScan; keep the whole 30 G "
                         "set on ONE amp so the fed-forward centers are comparable)")
    ap.add_argument("--308-amp", dest="ryd308_amp", type=float, default=None,
                    help="308 pulse amp, max 0.4 (default 0.4)")
    ap.add_argument("--green-freq-mhz", dest="green_freq_mhz", type=float, default=None,
                    help="override the fixed 556 push freq in MHz (else RES0+slope*field); "
                         "pass the freshly located 30 G dip when it has drifted out of the model")
    ap.add_argument("--mw-gain", type=int, default=8000, help="QICK DAC gain (power knob)")
    ap.add_argument("--mw-lo", type=float, default=11300.0)
    ap.add_argument("--mw-hi", type=float, default=11365.0)
    ap.add_argument("--mw-pts", type=int, default=27)
    args = ap.parse_args()
    MW_GAIN = args.mw_gain; MW_LO = args.mw_lo; MW_HI = args.mw_hi; MW_PTS = args.mw_pts
    for _k in ("mw_gain", "mw_lo", "mw_hi", "mw_pts"):
        delattr(args, _k)
    print("MW gain %d | window %.1f-%.1f MHz, %d pts" % (MW_GAIN, MW_LO, MW_HI, MW_PTS))
    # Flag names match Revival616Scan's params; None = not passed -> use the signature default.
    Revival616Scan(**{k: v for k, v in vars(args).items() if v is not None})
