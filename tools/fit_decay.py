"""fit_decay.py -- exponential-decay fit of a 1-D time sweep (e.g. Rydberg lifetime vs STIRAPGap).

Model: S(t) = S_inf + A * exp(-t / tau). Data pipeline identical to fit_rabi_spectrum.py:
run_analysis.analyze_scan -> summary.survival_mean vs sweep.values[0] (axis in SECONDS unless
--x-unit us). Defaults to the verify-conditioned metric (survival_ref='mid', target-only when
slm_diag allows) -- the honest frame pair on 3-image rearrange-STIRAP scans.

Run with the yb_analysis env python:
  <yb_analysis-python> fit_decay.py <scan_id> [--recache] [--ref mid|img1] [--all-sites]
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
    ap.add_argument("--xlabel", default="gap [us]")
    ap.add_argument("--x-unit", choices=("s", "us"), default="s",
                    help="unit the swept axis is stored in (default s; plotted in us)")
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--ref", choices=("img1", "mid"), default="mid")
    ap.add_argument("--all-sites", action="store_true")
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
    x = np.asarray(d["sweep"]["values"][0], float)
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, ye = x[m], y[m], ye[m]
    ye = np.where(np.isfinite(ye) & (ye > 0), ye, np.nanmedian(ye[ye > 0]) if (ye > 0).any() else 0.05)
    t_us = x * (1e6 if args.x_unit == "s" else 1.0)

    def model(t, S_inf, A, tau_us):
        return S_inf + A * np.exp(-t / np.maximum(tau_us, 1e-6))

    S_inf0 = float(np.nanmean(y[t_us > np.nanpercentile(t_us, 75)]))
    A0 = float(np.nanmean(y[t_us < np.nanpercentile(t_us, 15)]) - S_inf0)
    tau0 = float(np.nanmax(t_us) / 3.0)
    popt, pcov = curve_fit(model, t_us, y, p0=[S_inf0, A0, tau0], sigma=ye,
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-0.2, -1.5, 0.01], [1.2, 1.5, 100 * np.nanmax(t_us)]))
    perr = np.sqrt(np.diag(pcov))
    S_inf, A, tau = popt
    r = y - model(t_us, *popt)
    sst = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - float(np.sum(r**2)) / sst if sst > 0 else float("nan")

    out = {"scan_id": args.scan, "scan_dir": scan_dir, "n_shots": d.get("n_shots"),
           "n_params": d.get("n_params"), "model": "exp_decay",
           "tau_us": float(tau), "tau_err_us": float(perr[2]),
           "A": float(A), "A_err": float(perr[1]),
           "S_inf": float(S_inf), "S_inf_err": float(perr[0]), "r_squared": r2}
    print("scan %s | %s shots, %s pts" % (args.scan, d.get("n_shots"), d.get("n_params")))
    print("  [exp decay] tau = %.3f +- %.3f us   A = %.3f +- %.3f   S_inf = %.3f +- %.3f   R^2 = %.3f"
          % (tau, perr[2], A, perr[1], S_inf, perr[0], r2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        ax.errorbar(t_us, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2, label="survival")
        tf = np.linspace(t_us.min(), t_us.max(), 400)
        ax.plot(tf, model(tf, *popt), "-", color="C3", lw=1.5,
                label="S_inf + A exp(-t/tau)  R²=%.3f" % r2)
        ax.axhline(S_inf, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("survival")
        ax.set_title("%s  tau %.2f +- %.2f us  A %.3f  S_inf %.3f  R2 %.3f"
                     % (args.scan, tau, perr[2], A, S_inf, r2), fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(scan_dir, "fit_decay_%s.png" % args.scan)
        fig.text(0.005, 0.005, png, fontsize=5, color="0.4")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        out["png"] = png
        print("  saved %s" % png)
    except Exception as e:  # noqa: BLE001
        print("  (plot skipped: %s)" % e)

    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
