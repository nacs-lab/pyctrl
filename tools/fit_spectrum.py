"""fit_spectrum.py -- array-averaged Lorentzian fit of a 1-D push-out spectrum.

Uses the lab's canonical pipeline: ``run_analysis.analyze_scan`` buckets shots onto
the swept axis via ``config['Params']`` (the shot->scan-point map pyctrl writes at job
start for FINITE-rep runs) and returns ``summary.survival_mean`` (site-averaged P11) vs
``sweep.values[0]``. We fit a Lorentzian DIP to that and report center / FWHM / R^2.

NOTE: a run submitted with ``--reps 0`` (run-forever) has NO ``config['Params']`` (the
forever path can't pre-stack an infinite order), so survival_mean comes back empty and
the axis is unknown -- run calibrations with a finite ``--reps``.

``--peaks 2`` additionally fits a DOUBLE Lorentzian dip (for two-component / mj-split
lines such as the 399 ``1S0->1P1`` line) and saves a single-vs-double comparison plot
(``fit_spectrum_<sid>_2lor.png``); the JSON gains a ``double`` block with both centers,
both FWHMs, the splitting, and the 2-peak R^2. Default (``--peaks 1``) is unchanged.

Run with the yb_analysis env python:
  <yb_analysis-python> fit_spectrum.py <scan_id|latest> [--ref 107.7503e6] [--peaks 1|2]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _latest_scan_id():
    import urllib.request
    for h in ("127.0.0.1:8050", "100.86.15.43:8050"):
        try:
            with urllib.request.urlopen("http://%s/api/runs/list?max=1" % h, timeout=20) as r:
                return json.load(r)["runs"][0]["scan_id"]
        except Exception:
            continue
    raise SystemExit("could not reach dashboard to resolve 'latest'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--ref", type=float, default=None, help="reference freq (Hz) for a delta report")
    ap.add_argument("--peaks", type=int, choices=(1, 2), default=1,
                    help="1 = single Lorentzian dip (default); 2 = also fit a double "
                         "Lorentzian dip (mj-split / two-component lines, e.g. 399) and "
                         "save a single-vs-double comparison")
    args = ap.parse_args()

    import numpy as np
    from yb_analysis.analysis.run_analysis import analyze_scan
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian, fit_double_lorentzian

    sid = _latest_scan_id() if args.scan == "latest" else args.scan
    d = analyze_scan(sid, include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False)
    scan_dir = d.get("scan_dir")
    x = np.asarray(d["sweep"]["values"][0], float)
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    ld = np.asarray(d["summary"].get("loading_rate") or [], float)

    if y.size == 0 or not np.isfinite(y).any():
        raise SystemExit(
            "survival_mean is empty -- this scan has no config['Params'] map.\n"
            "Was it submitted with --reps 0 (run-forever)? Re-run with a finite --reps.")

    fit = fit_lorentzian(x, y, ye, mode="dip")
    if fit is None:
        raise SystemExit("Lorentzian fit failed (too few finite points?)")
    center, fwhm, r2 = fit["center"], abs(fit["width"]), fit["r_squared"]

    span = x.max() - x.min()
    edge = (center <= x.min() + 0.02 * span or center >= x.max() - 0.02 * span)
    out = {"scan_id": sid, "scan_dir": scan_dir, "n_shots": d.get("n_shots"),
           "n_params": d.get("n_params"), "n_peaks": args.peaks,
           "center_Hz": center, "fwhm_Hz": fwhm,
           "r_squared": r2, "x_min_Hz": float(x.min()), "x_max_Hz": float(x.max()),
           "loading_mean": (float(np.nanmean(ld)) if ld.size else None), "edge_pinned": bool(edge)}
    if args.ref is not None:
        out["ref_Hz"] = args.ref
        out["delta_Hz"] = center - args.ref

    print("scan %s | %s shots, %s pts%s"
          % (sid, d.get("n_shots"), d.get("n_params"),
             ("  loading ~%.2f" % np.nanmean(ld)) if ld.size else ""))
    print("  [1 Lorentzian] center = %.4f MHz   FWHM = %.1f kHz   R^2 = %.3f%s"
          % (center / 1e6, fwhm / 1e3, r2, "   *** EDGE-PINNED ***" if edge else ""))
    print("  window %.3f-%.3f MHz | survival %.2f-%.2f"
          % (x.min() / 1e6, x.max() / 1e6, np.nanmin(y), np.nanmax(y)))
    if args.ref is not None:
        print("  delta from ref %.4f MHz = %+.1f kHz" % (args.ref / 1e6, (center - args.ref) / 1e3))

    # Optional second model: a double Lorentzian dip (two-component / mj-split lines).
    dfit = None
    if args.peaks == 2:
        dfit = fit_double_lorentzian(x, y, ye, mode="dip")
        if dfit is None:
            print("  [2 Lorentzian] failed or degenerate (components merged) -> "
                  "single peak is the better description")
        else:
            c1, c2 = dfit["centers"]
            w1, w2 = dfit["widths"]
            out["double"] = {"center1_Hz": float(c1), "fwhm1_Hz": float(w1),
                             "center2_Hz": float(c2), "fwhm2_Hz": float(w2),
                             "splitting_Hz": dfit["splitting"], "r_squared": dfit["r_squared"]}
            print("  [2 Lorentzian] peak1 = %.4f MHz (FWHM %.1f kHz) | peak2 = %.4f MHz (FWHM %.1f kHz)"
                  % (c1 / 1e6, w1 / 1e3, c2 / 1e6, w2 / 1e3))
            print("                 splitting = %.3f MHz   R^2 = %.3f  (vs %.3f single)"
                  % (dfit["splitting"] / 1e6, dfit["r_squared"], r2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        ax.errorbar(x / 1e6, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2,
                    label="array-avg survival", zorder=5)
        ax.plot(fit["x_fit"] / 1e6, fit["y_fit"], "-", color="C3",
                lw=(1.2 if dfit else 1.5), alpha=(0.8 if dfit else 1.0),
                label="1 Lorentzian  R²=%.3f" % r2)
        if dfit is not None:
            ax.plot(dfit["x_fit"] / 1e6, dfit["y_fit"], "-", color="C1", lw=2.0,
                    label="2 Lorentzian  R²=%.3f" % dfit["r_squared"])
            ax.plot(dfit["x_fit"] / 1e6, dfit["comp1_fit"], "--", color="C1", lw=0.8, alpha=0.6)
            ax.plot(dfit["x_fit"] / 1e6, dfit["comp2_fit"], "--", color="C1", lw=0.8, alpha=0.6)
            for c in dfit["centers"]:
                ax.axvline(c / 1e6, color="C1", ls=":", lw=0.7, alpha=0.5)
        else:
            ax.axvline(center / 1e6, color="C3", ls="--", lw=0.9)
        if args.ref is not None:
            ax.axvline(args.ref / 1e6, color="k", ls=":", lw=0.9, label="prev ref")
        ax.set_xlabel("push-out freq [MHz]")
        ax.set_ylabel("survival (P11)")
        if dfit is not None:
            ttl = ("%s  2-peak %.4f / %.4f MHz  split %.2f MHz  R²=%.3f (1pk %.3f)"
                   % (sid, dfit["centers"][0] / 1e6, dfit["centers"][1] / 1e6,
                      dfit["splitting"] / 1e6, dfit["r_squared"], r2))
        else:
            ttl = ("%s  center %.4f MHz  FWHM %.0f kHz  R2 %.3f"
                   % (sid, center / 1e6, fwhm / 1e3, r2))
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        suffix = "_2lor" if args.peaks == 2 else ""
        png = os.path.join(scan_dir, "fit_spectrum_%s%s.png" % (sid, suffix))
        fig.savefig(png, dpi=120, bbox_inches="tight")
        out["png"] = png
        print("  saved %s" % png)
    except Exception as e:  # noqa: BLE001
        print("  (plot skipped: %s)" % e)

    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
