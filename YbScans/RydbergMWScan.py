"""RydbergMWScan.py -- QICK microwave ON during the Rydberg push-out, scanning the MW carrier
FREQUENCY. NO rearrange.

Structured like ``Revival616Scan`` (seq = ``RydbergPushoutSurvivalSeq`` -> ``RydbergPushoutStep``:
Poisson-loaded array, NO rearrangement), but instead of sweeping the 616 EOM it ARMS a QICK
microwave program and sweeps the MW carrier frequency. The 556 push, 308, and Rydberg bias field
are held fixed on the 30 G config (the same fixed conditions Revival616Scan sits on).

Why the microwave actually PLAYS here but not in Revival616Scan / STIRAPAWGScan:

  * ``RydbergPushoutStep`` already pulses ``TTLQickTrig`` (FPGA1/TTL14) during the push-out --
    UNCONDITIONALLY (unlike ``STIRAPPushoutStep``, which gates it on ``g.IfMW``). So the trigger
    edge is always emitted.
  * BUT the trigger only produces microwave if a QICK program has been ARMED. That happens only
    when the scan opts in: ``g().runp().QICK = True`` + a program declared via ``g().QICK.*``. The
    run loop (``engine_run`` -> ``awg_runtime.qick_setup``) then batch-uploads one program per
    unique swept point ONCE and arms the active one per shot (the board is one-shot). Revival616Scan
    never sets ``runp().QICK`` -> the trigger fires into a silent board.

So this scan = Revival616Scan's no-rearrange Rydberg push-out + the QICK microwave block from
``RearrangeMWScan`` (Sine template, carrier-frequency sweep). Sweeping the carrier maps the
microwave resonance as a survival lineshape.

QICK program (out-of-band; NOT in the serialized byte blob, so THE ONE RULE does not apply):
``template = "Sine"`` -- a single tone of ``QICK.duration`` at ``QICK.freq`` (MHz), ``QICK.gain``
(DAC units; must be nonzero to emit). See ``devices/qick_awg/templates.py``. The MW program must
finish inside the push-out window: keep ``QICK.duration`` <= ``Pushout.Time``.

Swept axis = ``QICK.freq`` (MHz). Out-of-band, so it does NOT enter the FPGA/NI byte blob; the run
loop keys one uploaded program per swept freq. Two ``Imag399Step`` calls -> NumImages = 2 (survival
vs MW freq).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so any
interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend live at --url; QICK server up on 192.168.0.72):
    cd pyctrl
    python YbScans/RydbergMWScan.py                       # 30 G, Sine MW, carrier sweep
    python YbScans/RydbergMWScan.py --reps 8
    python YbScans/RydbergMWScan.py --556-amp 0.22        # weaker push
    python YbScans/RydbergMWScan.py --reps 0              # run forever
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=30, green_amp=0.2, ryd308_amp=0.4, green_freq_mhz=None,
          mw_gain=10000, mw_npts=50):
    """ScanGroup for the 30 G Rydberg push-out with a QICK microwave, sweeping the MW carrier.

    Fixes the 556 push on the field-shifted resonance + the 30 G field + 308, arms a Sine QICK
    program, and sweeps ``QICK.freq``. ``field_G`` drives the Ryd bias coil
    (``BiasCoilCurrent.Ryd``); ``green_amp`` is the 556 Rydberg push amp; ``ryd308_amp`` the 308
    pulse amp (max 0.4). ``green_freq_mhz`` overrides the model 556 resonance with a located dip.
    ``mw_gain`` is the QICK DAC gain (lower = less power broadening -> narrower dip); ``mw_npts``
    the number of carrier points across the window.
    """
    from scan_group import ScanGroup
    import numpy as np

    # 556 push-out resonance (MHz): mirrors Revival616Scan's fixed value (2026-06-10 fit / located dip).
    res556_mhz = 143.524
    if green_freq_mhz is not None:
        res556_mhz = float(green_freq_mhz)   # explicit override: the located dip after drift

    # ---- QICK carrier-frequency sweep window (MHz). Edit to re-centre / refine. ----
    # 2026-07-23 ZOOM on dip 3 (71 3S1 mj=-1 -> 71 3P2 mj=-2, sigma-): coarse scan
    # data_20260723_145123 fit center 11318.5 MHz, FWHM 18.8 MHz (broadest of the 4). Bracket
    # +-40 MHz -- wide enough to see baseline both sides. NPTS from mw_npts (50 @ full power,
    # 25 for the gain series).
    MW_LO_MHZ, MW_HI_MHZ, MW_NPTS = 11315, 11325, int(mw_npts)

    g = ScanGroup()

    # ---- swept axis: QICK.freq (out-of-band; the run loop keys one program per unique value) ----
    g().QICK.template = "Sine"
    g().QICK.freq.scan(1, [round(float(v), 4) for v in
                           np.linspace(MW_LO_MHZ, MW_HI_MHZ, MW_NPTS)])
    g().QICK.gain = int(mw_gain)  # DAC gain (nonzero to emit; 0 = silent); lower = less power broadening
    g().QICK.duration = 0.5e-3   # single-tone length (s); must be <= Pushout.Time (1 ms below)
    g().QICK.phase = 0.0         # tone phase (deg)
    g.runp().QICK = True         # opt in -> engine_run wires qick_setup / arm / cleanup

    # ---- 616 EOM fixed on the revival peak (Revival616Scan SWEEPS this; here it is a fixed point) ----
    # Revival616Scan swept Init.EOM616.Freq over the 30 G revival window (210-260 MHz, mj-1 pi
    # revival peak ~234 MHz). Here the MW carrier is the swept axis, so the 616 EOM sits fixed on
    # that revival peak. Byte-affecting (slow-EOM ramp target); single value -> single ramp.
    EOM616_MHZ = 234.172
    g().Init.EOM616.Freq = EOM616_MHZ * 1e6

    # ---- 556 push-out fixed ON the 30 G resonance (RydbergPushoutStep: 556 Rydberg beam) ----
    g().Pushout.Green.Freq = res556_mhz * 1e6
    g().Pushout.Green.Amp = green_amp

    # ---- 308 + ionization (RydbergPushoutStep fires AmpAOM308 = Pushout.Ryd308.Amp) ----
    g().Pushout.Ryd308.Amp = ryd308_amp         # max 0.4
    g().Pushout.Amp369 = 0                       # read-but-unused on the byte path (faithful)

    # ---- push-out timing + Rydberg bias field ----------------------------
    # Pushout.Time is the TTLQickTrig HIGH window (RydbergPushoutStep: trig on -> wait(t) -> trig off).
    # Keep it >= QICK.duration so the whole MW program plays inside the gate.
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G    # Gauss -> VRydCoil (30 -> 1.5 V -> 30 G)

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # Optional per-scan SLM loading-pattern override (mirrors Revival616Scan). Comment out to use
    # the expConfig SLM.Loading default.
    g.runp().loading_phase = "phase/33x33_feedback11.pt"   # server-side WGS phase path
    g.runp().loading_defocus = -5                          # ANSI z4 loading defocus (rad)
    return g


def RydbergMWScan(url=None, reps=3, field_G=30, green_amp=0.2, ryd308_amp=0.4,
                  green_freq_mhz=None, mw_gain=10000, mw_npts=50):
    """Build + submit the 30 G Rydberg-pushout MW scan. Returns the queued descriptor id."""
    from yb_start_scan import ybStartScan

    g = build(field_G=field_G, green_amp=green_amp, ryd308_amp=ryd308_amp,
              green_freq_mhz=green_freq_mhz, mw_gain=mw_gain, mw_npts=mw_npts)
    freqs, _ = g.get_scanaxis(1, 1, 1)     # actual swept QICK carrier list (MHz); field 1 = sole axis
    npts = len(freqs)
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
    label = "RydbergMWScan_%dG_g%d" % (round(field_G), int(mw_gain))
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, field=%sG, gain=%d, %d QICK freq pts "
          "%.4f..%.4f MHz, 556@%.3f MHz amp %.2f, 308 amp %.2f)"
          % (label, did, url or "default", reps, field_G, int(mw_gain), npts, freqs[0], freqs[-1],
             g().Pushout.Green.Freq() / 1e6, green_amp, ryd308_amp))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 30 G Rydberg-pushout QICK-microwave carrier-frequency scan (no rearrange).")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    ap.add_argument("--field", dest="field_G", type=float, default=None,
                    help="bias field in Gauss -> Pushout.BiasCoilCurrent.Ryd (default 30)")
    ap.add_argument("--556-amp", dest="green_amp", type=float, default=0.15,
                    help="556 Rydberg push-out amp (default 0.15, the 30 G value)")
    ap.add_argument("--308-amp", dest="ryd308_amp", type=float, default=None,
                    help="308 pulse amp, max 0.4 (default 0.4)")
    ap.add_argument("--green-freq-mhz", dest="green_freq_mhz", type=float, default=None,
                    help="override the fixed 556 push freq in MHz (else the fitted resonance); "
                         "pass the freshly located 30 G dip when it has drifted")
    ap.add_argument("--gain", dest="mw_gain", type=int, default=None,
                    help="QICK DAC gain (default 10000; lower = less power broadening)")
    ap.add_argument("--npts", dest="mw_npts", type=int, default=None,
                    help="number of carrier points across the window (default 50)")
    args = ap.parse_args()
    # Flag names match RydbergMWScan's params; None = not passed -> use the signature default.
    RydbergMWScan(**{k: v for k, v in vars(args).items() if v is not None})
