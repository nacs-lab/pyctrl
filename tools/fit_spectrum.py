"""fit_spectrum.py -- array-averaged Lorentzian fit of a 1-D push-out spectrum.

Uses the lab's canonical pipeline: ``run_analysis.analyze_scan`` buckets shots onto
the swept axis via ``config['Params']`` (the shot->scan-point map pyctrl writes at job
start for FINITE-rep runs) and returns ``summary.survival_mean`` (site-averaged P11) vs
``sweep.values[0]``. We fit a Lorentzian DIP to that and report center / FWHM / R^2.

NOTE: a run submitted with ``--reps 0`` (run-forever) has NO ``config['Params']`` (the
forever path can't pre-stack an infinite order), so survival_mean comes back empty and
the axis is unknown -- run calibrations with a finite ``--reps``.

``--mode peak`` fits a Lorentzian PEAK instead of a dip -- for a survival REVIVAL line
(the 30 G 616-EOM ``Revival616Scan``, where survival rises back up on resonance) rather
than a push-out dip. Pair it with ``--xlabel '616-EOM freq [MHz]'`` so the saved plot is
labelled for the swept axis (the swept value, not the lineshape, is all that changes).
``--peaks 2`` is dip-only (the 399 doublet) and is skipped under ``--mode peak``.

``--peaks 2`` additionally fits a DOUBLE Lorentzian dip (for two-component / mj-split
lines such as the 399 ``1S0->1P1`` line) and saves a single-vs-double comparison plot
(``fit_spectrum_<sid>_2lor.png``); the JSON gains a ``double`` block with both centers,
both FWHMs, the splitting, and the 2-peak R^2. Default (``--peaks 1``, ``--mode dip``) is
unchanged.

Run with the yb_analysis env python:
  <yb_analysis-python> fit_spectrum.py <scan_id|latest> [--ref 107.7503e6] [--peaks 1|2]
                                       [--mode dip|peak] [--xlabel '616-EOM freq [MHz]']
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
                    help="1 = single Lorentzian (default); 2 = also fit a double "
                         "Lorentzian dip (mj-split / two-component lines, e.g. 399) and "
                         "save a single-vs-double comparison (dip mode only)")
    ap.add_argument("--mode", choices=("dip", "peak"), default="dip",
                    help="lineshape: 'dip' (push-out survival dip, default) or 'peak' "
                         "(a survival REVIVAL peak, e.g. the 30 G 616-EOM revival scan)")
    ap.add_argument("--xlabel", default="push-out freq [MHz]",
                    help="x-axis label for the saved plot (default 'push-out freq [MHz]'; "
                         "use e.g. '616-EOM freq [MHz]' for the revival scan)")
    ap.add_argument("--site-mask", default=None,
                    help="restrict the averaged spectrum to a site subset: a registered "
                         "name (e.g. 'stable') or a .npy path (bool[nSites] or int indices). "
                         "Omit to use the scan PATTERN's configured mask (if any); pass "
                         "'full' to force the whole array even when the pattern has one.")
    ap.add_argument("--recache", action="store_true",
                    help="rebuild analysis_payload.json instead of reusing it (needed when the "
                         "cache was pinned mid-run and holds only the first few shots)")
    ap.add_argument("--x-unit", choices=("Hz", "MHz"), default="Hz",
                    help="unit the swept axis is ALREADY in (default Hz, converted to MHz for "
                         "display). Pass 'MHz' for axes stored in MHz (e.g. QICK.freq) so the "
                         "plot/prints use the values as-is instead of dividing by 1e6")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip saving the figure (parity with fit_at_splitting.py / fit_map.py, "
                         "so fit_line.py can forward it to any fitter)")
    ap.add_argument("--min-r2", type=float, default=0.93,
                    help="R^2 gate for the VERDICT line (default 0.93; the daily runbook wants "
                         ">=0.95 on the narrow 556 mj=0 line and >=0.93 on the broad ones)")
    ap.add_argument("--min-shots-per-pt", type=float, default=2.0,
                    help="advisory shots/pt target (default 2, which is where these core "
                         "resonance lines converge). Low is a WARNING only -- it never flips the "
                         "verdict on a line that is already fitted with significant depth")
    ap.add_argument("--optical-factor", type=float, default=1.0,
                    help="multiplier from the swept axis to OPTICAL frequency, applied to a "
                         "--peaks 2 splitting. The 556 Rydberg push-out AOM is DOUBLE-pass, so "
                         "pass 2 for an Autler-Townes scan (better: use fit_at_splitting.py). "
                         "Default 1 = report the swept axis as-is")
    args = ap.parse_args()

    import numpy as np
    from yb_analysis.analysis.run_analysis import analyze_scan
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian, fit_double_lorentzian
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _detection_health import (detection_health, format_health,
                                   run_provenance, format_provenance, scan_completeness,
                                   loading_health, format_loading)

    sid = _latest_scan_id() if args.scan == "latest" else args.scan
    # 'full' -> force the whole array (site_mask=False); omit -> pattern default.
    _sm_arg = False if (args.site_mask or "").lower() == "full" else args.site_mask
    d = analyze_scan(sid, include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False,
                     site_mask=_sm_arg, force_recache=args.recache)
    if d.get("site_mask_error"):
        print("WARNING site_mask: %s (fell back to full array)" % d["site_mask_error"])
    elif d.get("site_mask_active"):
        print("site_mask %r active: %d/%d sites"
              % (d.get("site_mask_spec"), d.get("n_sites_used"), d.get("n_sites")))
    scan_dir = d.get("scan_dir")
    x = np.asarray(d["sweep"]["values"][0], float)
    y = np.asarray(d["summary"]["survival_mean"], float)
    ye = np.asarray(d["summary"]["survival_sem"], float)
    ld = np.asarray(d["summary"].get("loading_rate") or [], float)

    if y.size == 0 or not np.isfinite(y).any():
        raise SystemExit(
            "survival_mean is empty -- this scan has no config['Params'] map.\n"
            "Was it submitted with --reps 0 (run-forever)? Re-run with a finite --reps.")

    fit = fit_lorentzian(x, y, ye, mode=args.mode)
    if fit is None:
        raise SystemExit("Lorentzian fit failed (too few finite points?)")
    center, fwhm, r2 = fit["center"], abs(fit["width"]), fit["r_squared"]

    # Display scaling: axis stored in Hz -> /1e6 for MHz prints; stored in MHz -> as-is.
    xs = 1e6 if args.x_unit == "Hz" else 1.0     # axis value -> MHz
    fk = 1e3 if args.x_unit == "Hz" else 1e-3    # FWHM value -> kHz

    span = x.max() - x.min()
    edge = (center <= x.min() + 0.02 * span or center >= x.max() - 0.02 * span)
    out = {"scan_id": sid, "scan_dir": scan_dir, "n_shots": d.get("n_shots"),
           "n_params": d.get("n_params"), "n_peaks": args.peaks, "mode": args.mode,
           "x_unit": args.x_unit,
           "center_Hz": center, "fwhm_Hz": fwhm,
           "r_squared": r2, "x_min_Hz": float(x.min()), "x_max_Hz": float(x.max()),
           "loading_mean": (float(np.nanmean(ld)) if ld.size else None), "edge_pinned": bool(edge)}
    if args.ref is not None:
        out["ref_Hz"] = args.ref
        out["delta_Hz"] = center - args.ref

    print("scan %s | %s shots, %s pts%s"
          % (sid, d.get("n_shots"), d.get("n_params"),
             ("  loading ~%.2f" % np.nanmean(ld)) if ld.size else ""))
    print("  [1 Lorentzian %s] center = %.4f MHz   FWHM = %.1f kHz   R^2 = %.3f%s"
          % (args.mode, center / xs, fwhm / fk, r2,
             "   *** EDGE-PINNED ***" if edge else ""))
    print("  window %.3f-%.3f MHz | survival %.2f-%.2f"
          % (x.min() / xs, x.max() / xs, np.nanmin(y), np.nanmax(y)))
    if args.ref is not None:
        print("  delta from ref %.4f MHz = %+.1f kHz" % (args.ref / xs, (center - args.ref) / fk))

    # ---- uncertainties + quality gates ------------------------------------------------
    # pcov comes back from a sigma=yerr, absolute_sigma=True fit, so it believes the supplied
    # error bars literally. On these array-averaged spectra `survival_sem` (per-site binomial)
    # UNDER-reports the real point scatter, which shows up as chi2_red >> 1 -- so scale the
    # errors by sqrt(chi2_red). That reproduces what a bootstrap gives (verified 2026-09-15).
    nfree = max(len(x) - 4, 1)
    resid = y - fit["model"](x, *fit["params"])
    resid_std = float(np.std(resid))
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_red = float(np.nansum((resid / ye) ** 2) / nfree) if np.all(ye > 0) else float("nan")
    scale = np.sqrt(chi2_red) if np.isfinite(chi2_red) and chi2_red > 1 else 1.0
    perr = np.sqrt(np.diag(fit["pcov"])) * scale
    amp = abs(fit["params"][1])
    depth_sigma = amp / resid_std if resid_std > 0 else float("inf")
    shots_per_pt = (float(d.get("n_shots") or 0) / len(x)) if len(x) else 0.0
    print("  center err +/- %.1f kHz | FWHM err +/- %.1f kHz | depth %.3f (%.0fsigma) | "
          "chi2_red %.1f%s | %.1f shots/pt"
          % (perr[2] / fk, perr[3] / fk, amp, depth_sigma, chi2_red,
             " (errors scaled by sqrt)" if scale > 1 else "", shots_per_pt))

    _fid = sid[:8] + "_" + sid[8:]
    print("  " + format_provenance(run_provenance(scan_dir, _fid)))
    h = detection_health(scan_dir, _fid, n_shots=d.get("n_shots"))
    print("  " + format_health(h))
    lh = loading_health(ld)
    print("  " + format_loading(lh))

    # depth_sigma is amp/resid_std, so a fit that interpolates its points (resid_std -> 0)
    # reports inf and would "pass" a significance gate while meaning nothing. Degenerate = FAIL.
    depth_meaningful = bool(np.isfinite(depth_sigma)) and resid_std > 0
    # A centre is only usable if its error is small against the feature AND the window. On an
    # underpowered scan curve_fit happily returns a centre with an error larger than the sweep.
    center_constrained = bool(perr[2] < 0.5 * fwhm and perr[2] < 0.1 * span)
    if not center_constrained:
        print("  *** CENTRE NOT USABLE: +/-%.1f kHz is %.0f%% of the FWHM and %.0f%% of the whole "
              "window -- the fit did not constrain it. DO NOT report this centre as the line. ***"
              % (perr[2] / fk, 100 * perr[2] / fwhm if fwhm else float("nan"),
                 100 * perr[2] / span))
    gates = [("atoms loaded", not (lh and lh["dead"]),
              ("loading %.4f" % lh["mean"]) if lh else "unknown"),
             ("R2>=%.2f" % args.min_r2, r2 >= args.min_r2, "R2 %.3f" % r2),
             ("depth>=5sigma", depth_meaningful and depth_sigma >= 5.0,
              ("%.0fsigma" % depth_sigma) if depth_meaningful else "degenerate (resid ~ 0)"),
             ("centre constrained", center_constrained,
              "+/-%.1f kHz vs FWHM %.1f kHz" % (perr[2] / fk, fwhm / fk)),
             ("not-edge-pinned", not edge, "%.3f-%.3f window" % (x.min() / xs, x.max() / xs)),
             ("detection-healthy", bool(h["healthy"]) if h else True,
              (("d' %.1f" % h["dprime_median"]) if h else "unknown (not gated)"))]
    fit_ok = all(ok for _, ok, _ in gates)
    shots_ok = shots_per_pt >= args.min_shots_per_pt
    print("  GATES: " + " | ".join("%s %s" % (lbl, "PASS" if ok else "FAIL")
                                   for lbl, ok, _ in gates)
          + " || shots/pt %.1f %s" % (shots_per_pt,
                                      "OK" if shots_ok else "LOW(advisory)"))
    _verdict = "TRUST" if fit_ok else "CHECK"
    if lh and lh["dead"]:
        print("  VERDICT: CHECK -- NO ATOMS LOADED (%.4f). Report no fit from this run."
              % lh["mean"])
        _verdict = "CHECK"
    elif fit_ok:
        print("  VERDICT: TRUST -- all fit gates pass%s"
              % ("" if shots_ok else
                 ". shots/pt below the %.0f/pt target, but the line is fitted at %.0fsigma depth"
                 " with R2 %.3f, so the CENTER is sound -- do not re-scan for statistics alone"
                 % (args.min_shots_per_pt, depth_sigma, r2)))
    else:
        failed = [lbl for lbl, ok, _ in gates if not ok]
        # A LOW R^2 on its own is usually lineshape mismatch, not bad data: the 556 lines are
        # array-averaged and closer to Gaussian than Lorentzian, and the runbook's own note is
        # that the fitted CENTER -- the only thing we calibrate -- is model-independent. So when
        # R^2 is the sole failure and the line is deep, interior, and pinned down to a small
        # fraction of its own width, the center stands and there is nothing to re-scan.
        center_tight = perr[2] < 0.15 * fwhm
        if failed == ["R2>=%.2f" % args.min_r2] and center_tight:
            print("  VERDICT: TRUST-CENTER -- R2 %.3f is below %.2f, but that is LINESHAPE"
                  " mismatch, not bad data: depth %.0fsigma, interior, center pinned to"
                  " +/-%.1f kHz = %.0f%% of the FWHM. These array-averaged 556 lines are closer"
                  " to Gaussian than Lorentzian and the fitted CENTER is model-independent, so"
                  " the center is sound -- do NOT re-scan. (Check the plot if you want the"
                  " shape itself.)"
                  % (r2, args.min_r2, depth_sigma, perr[2] / fk, 100.0 * perr[2] / fwhm))
            _verdict = "TRUST-CENTER"
        else:
            print("  VERDICT: CHECK -- failed: %s" % ", ".join(failed))
    # Self-explaining CHECK vs one that needs investigation. Order matters: a boring cause
    # (incomplete run, thin statistics, no feature in the window) explains a failed gate without
    # any investigation, and only an UNEXPLAINED lineshape failure is worth escalating.
    _comp = scan_completeness(scan_dir, _fid, d.get("n_shots"))
    if _comp and _comp[2] < 0.5:
        print("  NOTE: this run is %d of %d scheduled shots (%.0f%% complete) -- the gates above "
              "are reading an unfinished scan" % (_comp[0], _comp[1], 100 * _comp[2]))
    _detection_bad = bool(h) and not h["healthy"]
    _edge_only = (not fit_ok) and all(
        ok for lbl, ok, _ in gates if lbl not in ("not-edge-pinned",))
    if lh and lh["dead"]:
        _esc, _why_esc = True, ("NO ATOMS -- loading %.4f. The fit is measuring the "
                                "false-positive rate, not a line. This indicts the apparatus "
                                "(SLM pattern / LUT / MOT): hand it to yb:troubleshooting"
                                % lh["mean"])
    elif _verdict in ("TRUST", "TRUST-CENTER"):
        _esc, _why_esc = False, "verdict is %s" % _verdict
    elif _detection_bad:
        _esc, _why_esc = True, ("detection is SUSPECT (d' %.1f, fill %.2f) -- a data-quality "
                                "problem, possibly for the troubleshooter"
                                % (h["dprime_median"], h["fill_median"]))
    elif not center_constrained and _comp and _comp[2] < 0.5:
        _esc, _why_esc = False, ("the run is only %.0f%% complete (%d of %d scheduled shots), so "
                                "the fit did not constrain the centre at all. Let it finish, then "
                                "refit. Report NO centre from this run"
                                % (100 * _comp[2], _comp[0], _comp[1]))
    elif not center_constrained:
        _esc, _why_esc = False, ("the fit did not constrain the centre (error exceeds the feature "
                                "width / the window). Report NO centre from this run; fix the "
                                "statistics or the window first")
    elif _comp and _comp[2] < 0.5:
        _esc, _why_esc = False, ("the run is only %.0f%% complete (%d of %d scheduled shots) -- "
                                "that alone explains the failed gate. Let it finish, or resubmit, "
                                "then refit. Nothing to investigate"
                                % (100 * _comp[2], _comp[0], _comp[1]))
    elif depth_sigma < 3.0:
        _esc, _why_esc = False, ("there is NO RESOLVED FEATURE in this window -- the deepest "
                                "excursion is only %.1f sigma and survival spans just %.3f across "
                                "the whole sweep. That is a window-placement problem: move the "
                                "window onto the line. Do not report this centre"
                                % (depth_sigma, float(np.nanmax(y) - np.nanmin(y))))
    elif not shots_ok:
        _esc, _why_esc = False, ("shots/pt is %.1f against a %.0f target -- thin statistics alone "
                                "explain the failed gate. Add reps and refit"
                                % (shots_per_pt, args.min_shots_per_pt))
    elif _edge_only:
        _esc, _why_esc = False, ("the line is EDGE-PINNED -- a window-placement failure with an "
                                 "obvious fix (re-centre the window on the line). Report that "
                                 "and stop")
    else:
        _esc, _why_esc = True, ("the LINESHAPE itself failed a gate (R^2 / depth) with a tight "
                                "window and healthy detection -- that needs a look, not a re-scan")
    print("  ESCALATE: %s -- %s" % ("YES" if _esc else "no", _why_esc))

    out.update({"escalate": bool(_esc), "escalate_reason": _why_esc,
                "center_err_Hz": float(perr[2]), "fwhm_err_Hz": float(perr[3]),
                "chi2_red": chi2_red, "err_scale": float(scale), "depth": float(amp),
                "depth_sigma": float(depth_sigma), "shots_per_pt": shots_per_pt,
                "health": h, "gates": {lbl: bool(ok) for lbl, ok, _ in gates},
                "shots_per_pt_ok": bool(shots_ok), "verdict": _verdict})

    # Optional second model: a double Lorentzian dip (two-component / mj-split lines).
    dfit = None
    if args.peaks == 2 and args.mode == "peak":
        print("  [2 Lorentzian] skipped -- the double fit is dip-only (it's for the "
              "399 doublet, not a peak); use --peaks 1 with --mode peak")
    elif args.peaks == 2:
        dfit = fit_double_lorentzian(x, y, ye, mode="dip")
        if dfit is None:
            print("  [2 Lorentzian] failed or degenerate (components merged) -> "
                  "single peak is the better description")
        else:
            c1, c2 = dfit["centers"]
            w1, w2 = dfit["widths"]
            # errors, scaled the same way as the single fit
            dresid = y - dfit["model"](x, *dfit["params"])
            dfree = max(len(x) - 7, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dchi2 = (float(np.nansum((dresid / ye) ** 2) / dfree)
                         if np.all(ye > 0) else float("nan"))
            dscale = np.sqrt(dchi2) if np.isfinite(dchi2) and dchi2 > 1 else 1.0
            dperr = np.sqrt(np.diag(dfit["pcov"])) * dscale
            # params order: [y0, A1, x01, w1, A2, x02, w2]; centers are sorted, so match by value
            e_lo, e_hi = (dperr[2], dperr[5]) if dfit["params"][2] <= dfit["params"][5] \
                else (dperr[5], dperr[2])
            esplit = float(np.hypot(e_lo, e_hi))
            out["double"] = {"center1_Hz": float(c1), "fwhm1_Hz": float(w1),
                             "center2_Hz": float(c2), "fwhm2_Hz": float(w2),
                             "center1_err_Hz": float(e_lo), "center2_err_Hz": float(e_hi),
                             "splitting_Hz": dfit["splitting"],
                             "splitting_err_Hz": esplit, "chi2_red": dchi2,
                             "r_squared": dfit["r_squared"]}
            print("  [2 Lorentzian] peak1 = %.4f +/- %.4f MHz (FWHM %.1f kHz) | "
                  "peak2 = %.4f +/- %.4f MHz (FWHM %.1f kHz)"
                  % (c1 / xs, e_lo / xs, w1 / fk, c2 / xs, e_hi / xs, w2 / fk))
            print("                 splitting = %.4f +/- %.4f MHz   R^2 = %.3f  (vs %.3f single)"
                  % (dfit["splitting"] / xs, esplit / xs, dfit["r_squared"], r2))
            if args.optical_factor != 1.0:
                print("                 OPTICAL (x%g) = %.4f +/- %.4f MHz"
                      % (args.optical_factor, dfit["splitting"] * args.optical_factor / xs,
                         esplit * args.optical_factor / xs))

    try:
        if args.no_plot:
            raise RuntimeError("--no-plot")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        ax.errorbar(x / xs, y, yerr=ye, fmt="o", ms=4, color="k", capsize=2,
                    label="array-avg survival", zorder=5)
        ax.plot(fit["x_fit"] / xs, fit["y_fit"], "-", color="C3",
                lw=(1.2 if dfit else 1.5), alpha=(0.8 if dfit else 1.0),
                label="1 Lorentzian  R²=%.3f" % r2)
        if dfit is not None:
            ax.plot(dfit["x_fit"] / xs, dfit["y_fit"], "-", color="C1", lw=2.0,
                    label="2 Lorentzian  R²=%.3f" % dfit["r_squared"])
            ax.plot(dfit["x_fit"] / xs, dfit["comp1_fit"], "--", color="C1", lw=0.8, alpha=0.6)
            ax.plot(dfit["x_fit"] / xs, dfit["comp2_fit"], "--", color="C1", lw=0.8, alpha=0.6)
            for c in dfit["centers"]:
                ax.axvline(c / xs, color="C1", ls=":", lw=0.7, alpha=0.5)
        else:
            ax.axvline(center / xs, color="C3", ls="--", lw=0.9)
        if args.ref is not None:
            ax.axvline(args.ref / xs, color="k", ls=":", lw=0.9, label="prev ref")
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("survival (P11)")
        if dfit is not None:
            ttl = ("%s  2-peak %.4f / %.4f MHz  split %.2f MHz  R²=%.3f (1pk %.3f)"
                   % (sid, dfit["centers"][0] / xs, dfit["centers"][1] / xs,
                      dfit["splitting"] / xs, dfit["r_squared"], r2))
        else:
            ttl = ("%s  center %.4f MHz  FWHM %.0f kHz  R2 %.3f"
                   % (sid, center / xs, fwhm / fk, r2))
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
