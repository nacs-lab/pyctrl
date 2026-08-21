"""StarkVxRevival616Scan.py -- 2D Vx x 616-EOM revival scan for DC-Stark / E-field nulling.

For each electrode control voltage Vx (Init.Electrodes.Vx), run a 616-EOM revival (the 308
resonance via the doubled 616 EOM) at 30 G with the 556 fixed on the 30 G resonance. The 308
resonance center shifts QUADRATICALLY with Vx (DC Stark shift); fitting center-vs-Vx to a parabola
gives the vertex = the Vx that nulls the residual DC E-field along x.

Same seq/config as Revival616Scan (RydbergPushoutSurvivalSeq -> RydbergPushoutStep): 556 push fixed
on the 30 G resonance (143.35 MHz), 308 ON at max AOM amp, Ryd bias 30 G. The ONLY additions are
the second swept axis Init.Electrodes.Vx (the InitStep electrode control voltage) and the narrower
616 window.

Axes:
  dim 1: Init.EOM616.Freq -- 260..300 MHz @ 1 MHz (41 pts), the 308 revival probe.
  dim 2: Init.Electrodes.Vx -- -0.2..0.2 V, 7 pts. InitStep maps Vx/Vy/Vz onto the 8 electrodes:
         VElectrode1 = +Vx+Vy-Vz, ...4 = -Vx+Vy+Vz, ...6 = -Vx-Vy-Vz, ...7 = +Vx-Vy+Vz
         (Vy, Vz held at the expConfig Init.Electrodes defaults).

Grid 41 x 7 = 287 pts; with rep=3 that is ~861 shots. NumImages=2 survival, Scramble on.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/StarkVxRevival616Scan.py --reps 3
    python YbScans/StarkVxRevival616Scan.py --vx -0.2 0.0667 0.2 --eom 260 1 300
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RydbergPushoutSurvivalSeq import RydbergPushoutSurvivalSeq


def build(field_G=30, green_amp=0.12, ryd308_amp=0.4, green_freq_mhz=143.35,
          axis="Vx", vx=(-2.0, 4.0 / 6, 2.0 + 1e-9), eom=(260.0, 1.0, 300.0)):
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- dim 1: 616-EOM revival probe (308 resonance) ----
    eom_freqs = [v * 1e6 for v in matlab_colon(*eom)]
    g().Init.EOM616.Freq.scan(1, eom_freqs)

    # ---- dim 2: electrode control voltage (DC Stark axis) -- Vx | Vy | Vz ----
    if axis not in ("Vx", "Vy", "Vz"):
        raise ValueError("axis must be Vx|Vy|Vz, got %r" % axis)
    vx_vals = [float(v) for v in matlab_colon(*vx)]
    getattr(g().Init.Electrodes, axis).scan(2, vx_vals)

    # ---- 556 push fixed ON the 30 G resonance ----
    g().Pushout.Green.Freq = float(green_freq_mhz) * 1e6
    g().Pushout.Green.Amp = green_amp

    # ---- 308 ON + ionization off ----
    g().Pushout.Ryd308.Amp = ryd308_amp
    g().Pushout.Amp369 = 0

    # ---- timing + Ryd bias field ----
    # 2026-07-16: 2e-3 -> 1e-3 to match the reference Revival616Scan (data_20260716_114348); the
    # longer 2 ms Rydberg hold let the atom decay before readout and SHRANK the revival peak
    # (~0.2-0.3 vs the reference 0.665 at the same 308 amp 0.4). Bright revival needs 1 ms.
    g().Pushout.Time = 1e-3
    g().Pushout.BiasCoilCurrent.Ryd = field_G

    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    # 2026-07-16: Scramble OFF. Scrambling jumps Init.EOM616.Freq randomly shot-to-shot; large
    # random EOM616 jumps UNLOCK the 616 laser (scan aborts at seq ~5-7). A monotonic (unscrambled)
    # EOM sweep steps the EOM gently and the lock survives -- confirmed vs the plain Revival616Scan
    # (Scramble 0) which ran clean under identical config. Keep 0 for any 616-freq sweep.
    rp.Scramble = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = "phase/33x33_feedback11.pt"
    # 2026-08-10: loading plane now comes from the per-array config
    # (ByPattern[<pattern>].SLM.Loading.Defocus -> slm_runtime._pattern_defocus);
    # setting rp.loading_defocus here would override it, so it is left unset.
    return g, eom_freqs, vx_vals


def submit(url=None, reps=3, **kw):
    from yb_start_scan import ybStartScan
    g, eom_freqs, vx_vals = build(**kw)
    opts = {"rep": reps} if reps is not None else {}
    ax = kw.get("axis", "Vx")
    did = ybStartScan(RydbergPushoutSurvivalSeq, g, url=url, label="StarkV%sRevival616Scan" % ax[-1].lower(),
                      description=("DC-Stark E-field nulling: 2D %s x 616-EOM revival at 30 G. "
                                   "Per %s (Init.Electrodes.%s, %d pts %.3f..%.3f V) run a 308 "
                                   "revival (616-EOM %.0f..%.0f MHz, %d pts); fit center-vs-%s "
                                   "parabola -> vertex = E-field-null %s. 556 fixed %.3f MHz, "
                                   "308 amp %.2f." % (ax, ax, ax, len(vx_vals), vx_vals[0], vx_vals[-1],
                                   eom_freqs[0] / 1e6, eom_freqs[-1] / 1e6, len(eom_freqs), ax, ax,
                                   kw.get("green_freq_mhz", 143.35), kw.get("ryd308_amp", 0.4))),
                      **opts)
    print("submitted StarkV%sRevival616Scan -> id %s (reps=%s) | EOM %d pts %.0f-%.0f MHz x %s %d pts %s"
          % (ax[-1].lower(), did, reps, len(eom_freqs), eom_freqs[0] / 1e6, eom_freqs[-1] / 1e6,
             ax, len(vx_vals), [round(v, 3) for v in vx_vals]))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2D Vx x 616-EOM revival (DC-Stark E-field nulling), 30 G.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--field", dest="field_G", type=float, default=30)
    ap.add_argument("--556-amp", dest="green_amp", type=float, default=0.12)
    ap.add_argument("--308-amp", dest="ryd308_amp", type=float, default=0.4)
    ap.add_argument("--green-freq-mhz", dest="green_freq_mhz", type=float, default=143.35)
    ap.add_argument("--axis", choices=("Vx", "Vy", "Vz"), default="Vx",
                    help="electrode control axis to sweep (Init.Electrodes.<axis>)")
    ap.add_argument("--vx", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(-2.0, 4.0 / 6, 2.0 + 1e-9),
                    help="swept-axis colon (V); default Vx -2..2 = 7 pts. For Vy/Vz use e.g. -0.1 0.0333 0.1")
    ap.add_argument("--eom", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(260.0, 1.0, 300.0),
                    help="616-EOM freq colon (MHz); default 260 1 300 = 41 pts")
    a = ap.parse_args()
    submit(url=a.url, reps=a.reps, field_G=a.field_G, green_amp=a.green_amp, ryd308_amp=a.ryd308_amp,
           green_freq_mhz=a.green_freq_mhz, axis=a.axis, vx=tuple(a.vx), eom=tuple(a.eom))
