"""2D VIonizationSet5to8 x 616-EOM revival at 30 G -- how far the ionization DC bias Stark-shifts 308.

Created 2026-08-05. Init.VIonizationSet5to8 is a DC analog level on Dev1/2 that InitStep applies to
electrodes 5-8. TTLIonizationSwitch5to8 (FPGA1/TTL48) DOES gate it -- InitStep sets the gate to 0 and
only STIRAPPushoutStep pulses it to 1 -- but the gate's off-isolation is imperfect and the Rydberg
line is extremely sensitive to the residue: the 308 DC-Stark curvature is ~1000-1165 MHz/V^2 in
electrode control volts (2026-06-28 Vy/Vz Stark fits), so even tens of mV of leak moves the 308 line
by MHz. Measured 2026-08-05: VIonizationSet5to8 = 0.5 V shifted the revival peak from ~234 MHz to
~294-300 MHz (>=60 MHz, still rising at the window edge), and >=1.0 V pushed it clean out of the
210-300 MHz window -- which is why a RearrangeSTIRAPScan-style VIonizationSet5to8 = 4 flattened the
616 revival completely (survival 0.07-0.14 across the whole 210-260 MHz window, scan 20260805190803).
To hold the 308 shift under ~1 MHz the residue must stay below ~0.03 control-V equivalent, i.e. the
gate needs ~-42 dB off-isolation at a 4 V setting.

This scan walks the ionization voltage up in small steps and re-locates the revival at each step, so
the shift can be tracked (and the usable operating point chosen) instead of jumping straight to 4 and
losing the line. Per ionization voltage: a full 616-EOM revival sweep (a Lorentzian PEAK whose center
IS the 308 resonance).

  dim 1: Init.EOM616.Freq  -- the 308/revival probe (keep the window WIDE; the line moves)
  dim 2: Init.VIonizationSet5to8 -- 0 .. 4 V (must stay < 5 V per the expConfig note)

Fit with pyctrl/tools/stark_vx_fit.py-style per-slice peak fits (the shift here is NOT expected to be
a clean parabola about 0 -- it is one arm of the Stark parabola in a different actuator -- so read the
per-voltage centers, don't force a vertex).

Same physics config as Revival616Scan / StarkVxRevival616Scan: 30 G, 556 push fixed on the measured
30 G resonance, 308 amp 0.4, Pushout.Time 1 ms, Scramble OFF (a random EOM616 jump unlocks 616).
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=30, green_amp=0.10, ryd308_amp=0.4, green_freq_mhz=143.5332,
          vion=(0.0, 0.5, 4.0 + 1e-9), eom=(210.0, 2.0, 300.0)):
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- dim 1: 616-EOM revival probe (308 resonance) ----
    eom_freqs = [v * 1e6 for v in matlab_colon(*eom)]
    g().Init.EOM616.Freq.scan(1, eom_freqs)

    # ---- dim 2: the ionization DC bias on electrodes 5-8 ----
    vion_vals = [float(v) for v in matlab_colon(*vion)]
    if any(v >= 5.0 for v in vion_vals):
        raise ValueError("VIonizationSet5to8 must stay < 5 V, got %r" % (vion_vals,))
    g().Init.VIonizationSet5to8.scan(2, vion_vals)

    # ---- 556 push fixed ON the measured 30 G resonance ----
    # 2026-08-05: amp 0.10 (NOT the 0.15 default) -- the 556 Rydberg push-out beam power was raised
    # ~7x, so 0.10 now reproduces the historical 0.15 line shape (dip 0.20-0.26, FWHM ~152 kHz).
    g().Pushout.Green.Freq = float(green_freq_mhz) * 1e6
    g().Pushout.Green.Amp = green_amp

    # ---- 308 ON + ionization beam off (this scan probes the DC bias, not the 369 pulse) ----
    g().Pushout.Ryd308.Amp = ryd308_amp
    g().Pushout.Amp369 = 0

    # ---- timing + Ryd bias field ----
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G

    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 0          # a large random EOM616 jump UNLOCKS 616 -- keep 0 for any 616 sweep
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = "phase/33x33_feedback11.pt"
    rp.loading_defocus = -5
    return g, eom_freqs, vion_vals


def submit(url=None, reps=3, **kw):
    from yb_start_scan import ybStartScan
    g, eom_freqs, vion_vals = build(**kw)
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label="IonizationRampRevival616Scan",
                      description=("VIonizationSet5to8 DC-Stark survey: 2D ionization bias x 616-EOM "
                                   "revival at 30 G. Per VIonizationSet5to8 (%d pts %.2f..%.2f V) run a "
                                   "308 revival (616-EOM %.0f..%.0f MHz, %d pts) and fit the peak; the "
                                   "center vs voltage IS the Stark shift from the un-gated DC bias on "
                                   "electrodes 5-8. 556 fixed %.4f MHz amp %.2f, 308 amp %.2f, 1 ms."
                                   % (len(vion_vals), vion_vals[0], vion_vals[-1],
                                      eom_freqs[0] / 1e6, eom_freqs[-1] / 1e6, len(eom_freqs),
                                      kw.get("green_freq_mhz", 143.5332), kw.get("green_amp", 0.10),
                                      kw.get("ryd308_amp", 0.4))),
                      **opts)
    print("submitted IonizationRampRevival616Scan -> id %s (reps=%s) | EOM %d pts %.0f-%.0f MHz x "
          "VIon %d pts %s" % (did, reps, len(eom_freqs), eom_freqs[0] / 1e6, eom_freqs[-1] / 1e6,
                              len(vion_vals), [round(v, 2) for v in vion_vals]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="2D VIonizationSet5to8 x 616-EOM revival (ionization-bias DC-Stark survey), 30 G.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--field", dest="field_G", type=float, default=30)
    ap.add_argument("--556-amp", dest="green_amp", type=float, default=0.10,
                    help="556 push amp (default 0.10 -- post-2026-08-05 7x power increase)")
    ap.add_argument("--308-amp", dest="ryd308_amp", type=float, default=0.4)
    ap.add_argument("--green-freq-mhz", dest="green_freq_mhz", type=float, default=143.5332,
                    help="556 push freq = the measured 30 G dip (default = the 2026-08-05 fit)")
    ap.add_argument("--vion", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.0, 0.5, 4.0 + 1e-9),
                    help="VIonizationSet5to8 colon (V, must stay < 5); default 0..4 by 0.5 = 9 pts")
    ap.add_argument("--eom", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(210.0, 2.0, 300.0),
                    help="616-EOM window colon (MHz); default 210..300 @ 2 MHz = 46 pts (WIDE: the "
                         "line moves with the bias)")
    a = ap.parse_args()
    submit(url=a.url, reps=a.reps, field_G=a.field_G, green_amp=a.green_amp,
           ryd308_amp=a.ryd308_amp, green_freq_mhz=a.green_freq_mhz,
           vion=tuple(a.vion), eom=tuple(a.eom))
