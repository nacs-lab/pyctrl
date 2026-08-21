"""fit_rabi_spectrum.py -- fit the finite-pulse Rabi spectroscopy lineshape to a 1-D MW scan.

Model (square pulse of KNOWN length t, swept carrier):

    P(f)  = Omega^2/(Omega^2 + delta^2) * sin^2( sqrt(Omega^2 + delta^2) * t / 2 )
    S(f)  = S0 - A * P(f)                       (survival DIP on resonance)

with delta = 2*pi*(f - f0) and Omega = 2*pi*f_Rabi. Free params: f0, f_Rabi, A, S0. The pulse
length t is passed in (--t-pulse-ns) and held FIXED; f_Rabi is fit (it measures the true pulse
area -- for an exact pi pulse f_Rabi = 1/(2t)). This captures the sinc-like sidelobes a
Lorentzian cannot (fit_spectrum.py underfits Rabi lineshapes).

The swept axis is assumed to be in MHz (QICK.freq convention). Data pipeline identical to
fit_spectrum.py: run_analysis.analyze_scan -> summary.survival_mean vs sweep.values[0].

Run with the yb_analysis env python:
  <yb_analysis-python> fit_rabi_spectrum.py <scan_id> --t-pulse-ns 125 [--recache]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--t-pulse-ns", type=float, required=True,
                    help="pulse length in ns (held fixed; e.g. 125 = the intended pi pulse)")
    ap.add_argument("--xlabel", default="QICK MW freq [MHz]")
    ap.add_argument("--recache", action="store_true",
                    help="rebuild analysis_payload.json (needed on/after mid-run analyses)")
    ap.add_argument("--ref", choices=("img1", "mid"), default="mid",
                    help="survival conditioning frame (default 'mid' = verify-conditioned "
                         "P(final|mid), the honest metric on 3-image rearrange-STIRAP scans; "
                         "'img1' is floored/diluted by rearrangement fill)")
    ap.add_argument("--all-sites", action="store_true",
                    help="use the whole array instead of target-only (default target-only)")
    args = ap.parse_args()

    import numpy as np
    from scipy.optimize import curve_fit
    from yb_analysis.analysis.run_analysis import analyze_scan

    d = analyze_scan(args.scan, include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False,
                     survival_ref=args.ref, target_only=not args.all_sites,
                     force_recache=args.recache)
    print("survival_ref=%s target_only=%s%s"
          % (d.get("survival_ref"), d.get("target_only_active"),
             (" (target_only_error: %s)" % d["target_only_error"]) if d.get("target_only_error") else ""))
    scan_dir = d.get("scan_dir")
    x = np.asarray(d["sweep"]["values"][0], float)          # MHz
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    if y.size == 0 or not np.isfinite(y).any():
        raise SystemExit("survival_mean is empty (run-forever scan? re-run with finite --reps)")
    m = np.isfinite(x) & np.isfinite(y)
    x, y, ye = x[m], y[m], ye[m]
    ye = np.where(np.isfinite(ye) & (ye > 0), ye, np.nanmedian(ye[ye > 0]) if (ye > 0).any() else 1.0)

    t_s = args.t_pulse_ns * 1e-9

    def model(f_mhz, f0_mhz, fr_mhz, A, S0):
        delta = 2 * np.pi * (f_mhz - f0_mhz) * 1e6          # rad/s
        Om = 2 * np.pi * abs(fr_mhz) * 1e6                  # rad/s
        W2 = Om**2 + delta**2
        W = np.sqrt(W2)
        P = np.where(W2 > 0, (Om**2 / W2) * np.sin(W * t_s / 2.0)**2, np.sin(0)**2)
        return S0 - A * P

    # Inits: f0 at the argmin, f_Rabi at the exact-pi value 1/(2t), contrast from the data range.
    f0_0 = float(x[np.argmin(y)])
    fr_0 = 1.0 / (2 * t_s) / 1e6                            # MHz
    S0_0 = float(np.nanpercentile(y, 90))
    A_0 = float(max(S0_0 - np.nanmin(y), 0.05))
    p0 = [f0_0, fr_0, A_0, S0_0]
    bounds = ([x.min(), 0.05 * fr_0, 0.0, 0.0], [x.max(), 20 * fr_0, 1.5, 1.2])
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=ye, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    f0, fr, A, S0 = popt
    res = y - model(x, *popt)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - float(np.sum(res**2)) / ss_tot if ss_tot > 0 else float("nan")
    area_pi = 2 * abs(fr) * 1e6 * t_s                       # pulse area in units of pi
    fwhm_mhz = 1.6 * abs(fr)                                # pi-pulse approx (half-max at ~0.8*Omega)

    out = {"scan_id": args.scan, "scan_dir": scan_dir, "n_shots": d.get("n_shots"),
           "n_params": d.get("n_params"), "model": "rabi_finite_pulse",
           "t_pulse_ns": args.t_pulse_ns,
           "f0_MHz": float(f0), "f0_err_MHz": float(perr[0]),
           "f_rabi_MHz": float(abs(fr)), "f_rabi_err_MHz": float(perr[1]),
           "pulse_area_pi": float(area_pi), "A": float(A), "S0": float(S0),
           "fwhm_est_MHz": float(fwhm_mhz), "r_squared": r2}

    print("scan %s | %s shots, %s pts" % (args.scan, d.get("n_shots"), d.get("n_params")))
    print("  [Rabi lineshape, t=%.4g ns fixed] f0 = %.4f +- %.4f MHz   f_Rabi = %.3f +- %.3f MHz"
          % (args.t_pulse_ns, f0, perr[0], abs(fr), perr[1]))
    print("  pulse area = %.2f pi (1.00 = exact pi pulse)   A = %.3f  S0 = %.3f   R^2 = %.3f"
          % (area_pi, A, S0, r2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        ax.errorbar(x, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2,
                    label="array-avg survival", zorder=5)
        xf = np.linspace(x.min(), x.max(), 800)
        ax.plot(xf, model(xf, *popt), "-", color="C3", lw=1.5,
                label="Rabi lineshape  R²=%.3f" % r2)
        ax.axvline(f0, color="C3", ls="--", lw=0.9)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("survival (P11)")
        ax.set_title("%s  f0 %.4f(%.0f) MHz  f_Rabi %.3f MHz  area %.2f pi  R2 %.3f"
                     % (args.scan, f0, perr[0] * 1e4, abs(fr), area_pi, r2), fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(scan_dir, "fit_rabi_%s.png" % args.scan)
        fig.savefig(png, dpi=120, bbox_inches="tight")
        out["png"] = png
        print("  saved %s" % png)
    except Exception as e:  # noqa: BLE001
        print("  (plot skipped: %s)" % e)

    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
