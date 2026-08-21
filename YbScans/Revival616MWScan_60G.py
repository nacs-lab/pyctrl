"""Revival616MWScan_60G.py -- 60 G MW spectrum via REVIVAL DESTRUCTION (incoherent pushout path).

The 616-EOM is PINNED at the freshly measured 60 G revival peak and the QICK microwave carrier is
SWEPT: driving the 3S1<->3P2 transition empties the shelved Rydberg S state and DESTROYS the
revival -> survival DIPS on resonance. Needs NO f0 prior and NO working pi pulse / reverse STIRAP
(the whole point vs the Rabi-spectrum scan) -- the MW is on for the entire 1 ms push-out window.

Method precedent = the 2026-08-12 20 G campaign (STIRAPOptimizations/Revival616MWScan_20G.py):
  * QICK.duration MUST equal Pushout.Time (1 ms). The first 20 G round used a 20 us tone inside
    the 1 ms push = 2% duty cycle -> max possible destruction ~2% = noise floor, flat scan.
    build_sine/_split_duration chunk the long tone, so 1 ms is supported.
  * The dip is power-broadened AND power-shifted at high gain (20 G: FWHM ~ gain^1.59; the
    gain-8000 centre read 0.6 MHz LOW). Coarse-locate at gain 8000, then LADDER DOWN
    (8000/4000/2000/1000/500 via --gain) and take f0 from the low-power limit.

This file is the 60 G port built on the CURRENT Revival616Scan (2026-08-18 base): high-field band
(50-80 G -> RydbergHighFieldPushoutStep, 556 push auto -60 MHz on the double-pass AOM) + the
Zeeman-model 556 centre (Resonance556mj0Freq + 1.178 MHz/G * field) tracking the daily calibration.

--eom-pin is REQUIRED (MHz): today's fitted 60 G revival peak from a fresh
``Revival616Scan.py --field 60`` EOM sweep -- the revival must be healthy at the pin or there is
nothing to destroy.

Run it (pyctrl backend live; QICK server up on 192.168.0.72):
    cd pyctrl
    python YbScans/Revival616MWScan_60G.py --eom-pin 230.43 --reps 3
    python YbScans/Revival616MWScan_60G.py --eom-pin 230.43 --mw-lo 11250 --mw-hi 11300 --mw-pts 51
    python YbScans/Revival616MWScan_60G.py --eom-pin 230.43 --gain 1000 --mw-pts 101   # ladder step
"""

import argparse
import numpy as np

# MW knobs (CLI-overridable in __main__). Window 11250-11300 MHz = 2026-08-19 user directive
# (the 60 G line is expected BELOW the 20 G f0 11334.93; 30->20 G moved f0 +16 MHz).
MW_GAIN = 8000                # coarse-locate gain; ladder down for f0 (power-shifts the centre)
MW_LO = 11250.0
MW_HI = 11300.0
MW_PTS = 51                   # 1 MHz step (gain-8000 FWHM was ~16 MHz at 20 G -- plenty)

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=60, green_amp=0.15, ryd308_amp=0.4, green_freq_mhz=None, eom_pin_mhz=None):
    """ScanGroup for the 60 G revival-destruction MW sweep (seq = RydbergPushoutSurvivalSeq).

    Same fixed pushout config as Revival616Scan (556 on the Zeeman-model resonance, 308 firing,
    1 ms push, bias field on), EXCEPT the 616 EOM is pinned at ``eom_pin_mhz`` (the measured
    revival peak) and the swept axis is ``QICK.freq``.
    """
    from scan_group import ScanGroup
    from seq_config import SeqConfig
    from consts import Consts

    if eom_pin_mhz is None:
        raise ValueError("--eom-pin is required: today's fitted revival peak in MHz "
                         "(from Revival616Scan --field %g)" % field_G)

    # Consts() reads SeqConfig.get().consts; load the real expConfig off the backend.
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    RES0_MHZ = float(Consts().Resonance556mj0Freq) / 1e6
    ZEEMAN_SLOPE_MHZ_PER_G = 1.178
    HF_AOM_OFFSET_MHZ = 60.0   # high-field path: +120 MHz single-pass / double-pass first AOM

    res556_mhz = RES0_MHZ + ZEEMAN_SLOPE_MHZ_PER_G * field_G
    if green_freq_mhz is not None:
        res556_mhz = float(green_freq_mhz)   # explicit override (low-field units at every field)

    g = ScanGroup()

    # ---- 616 EOM PINNED at the measured revival peak (was the swept axis in Revival616Scan) ----
    g().Init.EOM616.Freq = float(eom_pin_mhz) * 1e6

    # ---- swept axis: QICK microwave carrier (revival destruction) --------------------------
    MW_FREQ_MHZ = [round(float(v), 6) for v in np.linspace(MW_LO, MW_HI, MW_PTS)]
    g().QICK.template = "Sine"
    g().QICK.freq.scan(1, MW_FREQ_MHZ)                # SWEPT dim 1
    g().QICK.gain = MW_GAIN
    g().QICK.duration = 1e-3                          # == Pushout.Time: MW on the WHOLE window (duty-cycle lesson)
    g().QICK.rabi_freq = 4.825e6                      # unused by Sine template
    g().QICK.wait_time = 1e-6                         # unused by Sine template
    g().QICK.phase = 0.0
    g.runp().QICK = True                              # arm the QICK; RydbergPushoutStep pulses TTLQickTrig

    # ---- 556 push-out fixed on the Zeeman-model resonance (HF-shifted for 50-80 G) ---------
    if 50 <= field_G <= 80:
        res556_mhz -= HF_AOM_OFFSET_MHZ
    g().Pushout.Green.Freq = res556_mhz * 1e6
    g().Pushout.Green.Amp = green_amp                 # 0.15, lockstep with RydbergSpectrum556Scan

    # ---- 308 + ionization (RydbergPushoutStep fires AmpAOM308 = Pushout.Ryd308.Amp) --------
    g().Pushout.Ryd308.Amp = ryd308_amp               # max 0.4
    g().Pushout.Amp369 = 0                            # read-but-unused on the byte path (faithful)

    # ---- push-out timing + Rydberg bias field ----------------------------------------------
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G         # 50-80 G -> RydbergHighFieldPushoutStep

    # ---- run params (runp) -------------------------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    g.runp().loading_phase = "phase/33x33_feedback11.pt"   # server-side WGS phase path
    return g


def Revival616MWScan_60G(url=None, reps=3, field_G=60, green_amp=0.15, ryd308_amp=0.4,
                         green_freq_mhz=None, eom_pin_mhz=None):
    """Build + submit the 60 G revival-destruction MW sweep. Returns the descriptor id."""
    from yb_start_scan import ybStartScan

    field_G = 60.0 if field_G is None else float(field_G)
    g = build(field_G=field_G, green_amp=green_amp, ryd308_amp=ryd308_amp,
              green_freq_mhz=green_freq_mhz, eom_pin_mhz=eom_pin_mhz)
    npts = g().QICK.freq.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "Revival616MW_%dG" % round(field_G)
    desc = ("Revival-destruction MW spectrum at %g G: 616 EOM PINNED at the measured revival "
            "peak %.4f MHz, 556 push on the Zeeman model (%.3f MHz at the AOM), 308 amp %.2f, "
            "1 ms push. SWEEP QICK.freq %.2f..%.2f MHz (%d pts) at gain %d, MW on for the whole "
            "window (duration == Pushout.Time). Survival DIPS on the 3S1<->3P2 resonance "
            "(shelved-S destruction). Centre is power-shifted at high gain -- ladder the gain "
            "down and take f0 from the low-power limit (20 G precedent: FWHM ~ gain^1.59)."
            % (field_G, float(eom_pin_mhz), g().Pushout.Green.Freq() / 1e6, ryd308_amp,
               MW_LO, MW_HI, npts, MW_GAIN))
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label=label,
                      description=desc, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s; EOM pin %.4f MHz, MW %d pts "
          "%.2f..%.2f MHz @ gain %d, 556@%.3f MHz amp %.2f)"
          % (label, did, url or "default", reps, float(eom_pin_mhz), npts, MW_LO, MW_HI,
             MW_GAIN, g().Pushout.Green.Freq() / 1e6, green_amp))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the 60 G revival-destruction MW spectrum "
                                             "(616 pinned at the revival peak, sweep QICK.freq).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3, help="passes over the sweep (0 = forever)")
    ap.add_argument("--eom-pin", dest="eom_pin_mhz", type=float, required=True,
                    help="616-EOM pin in MHz = today's fitted 60 G revival peak "
                         "(from Revival616Scan --field 60)")
    ap.add_argument("--field", dest="field_G", type=float, default=60.0,
                    help="bias field in Gauss (default 60; 50-80 G = high-field step)")
    ap.add_argument("--556-amp", dest="green_amp", type=float, default=0.15,
                    help="556 Rydberg push-out amp (default 0.15, lockstep with RydbergSpectrum556Scan)")
    ap.add_argument("--308-amp", dest="ryd308_amp", type=float, default=0.4,
                    help="308 pulse amp, max 0.4 (default 0.4)")
    ap.add_argument("--green-freq-mhz", dest="green_freq_mhz", type=float, default=None,
                    help="override the fixed 556 push freq in MHz, LOW-FIELD units "
                         "(else Zeeman model; the 50-80 G -60 MHz shift is applied on top)")
    ap.add_argument("--mw-gain", dest="mw_gain", type=int, default=MW_GAIN,
                    help="QICK DAC gain (default %d; ladder down for the unshifted f0)" % MW_GAIN)
    ap.add_argument("--mw-lo", dest="mw_lo", type=float, default=MW_LO)
    ap.add_argument("--mw-hi", dest="mw_hi", type=float, default=MW_HI)
    ap.add_argument("--mw-pts", dest="mw_pts", type=int, default=MW_PTS)
    args = ap.parse_args()
    MW_GAIN = args.mw_gain; MW_LO = args.mw_lo; MW_HI = args.mw_hi; MW_PTS = args.mw_pts
    for _k in ("mw_gain", "mw_lo", "mw_hi", "mw_pts"):
        delattr(args, _k)
    print("MW gain %d | window %.2f-%.2f MHz, %d pts" % (MW_GAIN, MW_LO, MW_HI, MW_PTS))
    Revival616MWScan_60G(**vars(args))
