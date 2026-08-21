"""fit_rabi_time.py -- damped Rabi-oscillation fit of a 1-D MW duration sweep.

Model: S(t) = S0 - A * sin^2(pi * f_Rabi * t) * exp(-t / tau)   (on-resonance drive).
Data pipeline identical to fit_decay.py: run_analysis.analyze_scan -> summary.survival_mean vs
sweep.values[0] (axis in SECONDS). Defaults to the verify-conditioned metric (survival_ref='mid',
target-only when slm_diag allows).

Run with the yb_analysis env python:
  <yb_analysis-python> fit_rabi_time.py <scan_id> [--f0-mhz 4.8] [--recache] [--ref mid|img1]
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
    ap.add_argument("--f0-mhz", type=float, default=4.8,
                    help="initial guess for f_Rabi in MHz (default 4.8, today's gain-2000 value)")
    ap.add_argument("--xlabel", default="MW duration [us]")
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--ref", choices=("img1", "mid"), default="mid")
    ap.add_argument("--all-sites", action="store_true")
    ap.add_argument("--envelope", choices=("exp", "gauss"), default="exp",
                    help="decay envelope: 'exp' = exp(-t/tau) (homogeneous), "
                         "'gauss' = exp(-(t/tau)^2) (inhomogeneous dephasing)")
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
    x = np.asarray(d["sweep"]["values"][0], float)          # seconds
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, ye = x[m], y[m], ye[m]
    ye = np.where(np.isfinite(ye) & (ye > 0), ye, np.nanmedian(ye[ye > 0]) if (ye > 0).any() else 0.05)
    t_us = x * 1e6

    if args.envelope == "gauss":
        def env(t, tau_us):
            return np.exp(-(t / np.maximum(tau_us, 1e-3)) ** 2)
    else:
        def env(t, tau_us):
            return np.exp(-t / np.maximum(tau_us, 1e-3))

    # Damped Rabi: the OSCILLATION damps toward its MIDPOINT (S0 - A/2), the midpoint stays.
    # (The earlier form S0 - A sin^2 * env wrongly relaxed to the oscillation TOP S0 at long t
    # -- dephased data sits at the midpoint, so that fit was systematically off; 2026-08-19.)
    # S0 = t=0 survival (top), A = full peak-to-peak contrast.
    def model(t, f_mhz, A, S0, tau_us):
        return (S0 - A / 2.0) + (A / 2.0) * np.cos(2 * np.pi * f_mhz * t) * env(t, tau_us)

    # sin^2 fits have dense local minima in f -- coarse-grid the frequency and keep the best
    # chi^2 refinement (a single p0 routinely locks onto A~0). Grid spans 0.3-12 MHz regardless
    # of --f0-mhz; the flag now only orders the grid (nearest-first, cosmetic).
    A0 = float(np.nanmax(y) - np.nanmin(y))
    best = None
    for f0 in np.concatenate([np.arange(0.3, 12.0, 0.05), np.arange(12.0, 40.0, 0.1)]):
        try:
            p, c = curve_fit(model, t_us, y, p0=[f0, A0, float(np.nanmax(y)), 20.0],
                             sigma=ye, absolute_sigma=True, maxfev=5000,
                             bounds=([0.2, 0.0, 0.0, 0.1], [45.0, 1.5, 1.2, 5000.0]))
            chi2 = float((((y - model(t_us, *p)) / ye) ** 2).sum())
            if best is None or chi2 < best[0]:
                best = (chi2, p, c)
        except Exception:
            continue
    if best is None:
        raise SystemExit("Rabi fit failed at every grid frequency")
    _, popt, pcov = best
    perr = np.sqrt(np.diag(pcov))
    f, A, S0, tau = popt
    r = y - model(t_us, *popt)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(r ** 2)) / sst if sst > 0 else float("nan")
    t_pi_ns = 1e3 / (2 * f)

    out = {"scan_id": args.scan, "scan_dir": scan_dir, "n_shots": d.get("n_shots"),
           "n_params": d.get("n_params"), "model": "rabi_time_damped_" + args.envelope,
           "f_rabi_MHz": float(f), "f_rabi_err_MHz": float(perr[0]),
           "t_pi_ns": float(t_pi_ns), "A": float(A), "A_err": float(perr[1]),
           "S0": float(S0), "tau_us": float(tau), "tau_err_us": float(perr[3]),
           "r_squared": r2}
    print("scan %s | %s shots, %s pts" % (args.scan, d.get("n_shots"), d.get("n_params")))
    print("  [damped Rabi] f_Rabi = %.4f +- %.4f MHz (T_pi = %.1f ns)   A = %.3f   S0 = %.3f   "
          "tau = %.2f +- %.2f us   R^2 = %.3f" % (f, perr[0], t_pi_ns, A, S0, tau, perr[3], r2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.7))
        ax.errorbar(t_us, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2, label="survival")
        tf = np.linspace(t_us.min(), t_us.max(), 1200)
        _env_lbl = "e^(-(t/τ)²)" if args.envelope == "gauss" else "e^(-t/τ)"
        ax.plot(tf, model(tf, *popt), "-", color="C3", lw=1.2,
                label="S_mid + (A/2)cos(2πft) %s  R²=%.3f" % (_env_lbl, r2))
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("survival")
        ax.set_title("%s  f_Rabi %.4f(%.0f) MHz  T_pi %.1f ns  tau %.1f us  R2 %.3f"
                     % (args.scan, f, perr[0] * 1e4, t_pi_ns, tau, r2), fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        suffix = "_gauss" if args.envelope == "gauss" else ""
        png = os.path.join(scan_dir, "fit_rabi_time_%s%s.png" % (args.scan, suffix))
        fig.text(0.005, 0.005, png, fontsize=5, color="0.4")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        out["png"] = png
        print("  saved %s" % png)
    except Exception as e:  # noqa: BLE001
        print("  (plot skipped: %s)" % e)

    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
