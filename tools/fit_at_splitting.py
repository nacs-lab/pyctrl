"""fit_at_splitting.py -- array-averaged Autler-Townes splitting from a 556AutlerTownesScan.

Fits a DOUBLE Lorentzian dip to the ARRAY-AVERAGE survival(P11) vs 556 freq and reports the
two dip centers + the AT splitting (= the 308 coupling Rabi -> a beam-alignment / polarization
optimization metric: maximize the splitting). Saves a fit plot next to the scan.

Why not fit_spectrum.py --peaks 2: its auto-seed straddles the GLOBAL minimum, which fails when
the AT doublet is wide + the two dips are unequal depth (both Lorentzians collapse onto one dip ->
negative R^2). This seeds from the two deepest WELL-SEPARATED minima instead.

Reads a finished OR still-growing scan (live h5 fallback, locking off).

Usage (PROJECT ROOT, yb_analysis python):
  python pyctrl/tools/fit_at_splitting.py <scan_id|scan_dir> [--min-sep 0.6] [--no-plot]
e.g.  python pyctrl/tools/fit_at_splitting.py 20260630144109
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _detection_health import (detection_health, format_health, run_provenance,
                               format_provenance, loading_health, format_loading)

from yb_analysis.analysis.unpack import unpack_scan_logicals
from yb_analysis.analysis.probabilities import prob11_site_resolved


def _resolve_dir(arg):
    if os.path.isdir(arg):
        return arg
    sid = str(arg)
    fid = sid[:8] + "_" + sid[8:]
    base = r"D:\OneDrive - Harvard University\Documents - Yb\Data"
    cand = os.path.join(base, sid[:8], "data_" + fid)
    if os.path.isdir(cand):
        return cand
    raise SystemExit("not a dir and could not resolve scan_id %s -> %s" % (arg, cand))


def _snapshot(scan_dir):
    sid = os.path.basename(scan_dir).replace("data_", "")
    scan = json.load(open(os.path.join(scan_dir, "data_%s.json" % sid)))
    import h5py
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    h5 = os.path.join(scan_dir, "data_%s.h5" % sid)
    last = None
    for _ in range(40):
        try:
            with h5py.File(h5, "r", swmr=True) as f:
                n = min(f["seq_ids"].shape[0], f["logicals_img1"].shape[0], f["logicals_img2"].shape[0])
                return scan, sid, f["seq_ids"][:n], f["logicals_img1"][:n], f["logicals_img2"][:n], n
        except Exception as e:
            last = e; time.sleep(0.25)
    raise RuntimeError("could not read h5: %r" % last)


def _lor(x, A, x0, w):
    return A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def _double_dip(x, y0, A1, x1, w1, A2, x2, w2):
    return y0 - _lor(x, A1, x1, w1) - _lor(x, A2, x2, w2)


def _smooth(y, k=3):
    m = np.isfinite(y)
    ys = np.where(m, y, 0.0); w = m.astype(float)
    num = np.convolve(ys, np.ones(k), "same"); den = np.convolve(w, np.ones(k), "same")
    return np.where(den > 0, num / den, np.nan)


def _seed_two_dips(x, ys, min_sep):
    """Two deepest minima at least `min_sep` MHz apart (-> robust double-dip seed)."""
    finite = np.where(np.isfinite(ys))[0]
    o = finite[np.argsort(ys[finite])]          # ascending survival = deepest first
    d1 = o[0]
    d2 = next((k for k in o[1:] if abs(x[k] - x[d1]) > min_sep), o[1] if len(o) > 1 else o[0])
    return sorted([float(x[d1]), float(x[d2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="scan_id (e.g. 20260630144109) or a scan_dir path")
    ap.add_argument("--min-sep", type=float, default=0.6, help="min dip separation (MHz) for seeding")
    ap.add_argument("--max-shots", type=int, default=None,
                    help="fit only the FIRST N shots of the scan (e.g. 200). The scan sweeps its "
                         "points in order, so a truncated fit is a partial-rep fit; the plot is "
                         "saved with a _firstN suffix so the full-scan plot survives")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--bare", type=float, default=None,
                    help="bare (un-dressed) line center in MHz, swept-axis -- prints the doublet "
                         "midpoint offset, the symmetry check that the doublet really is AT dressing")
    ap.add_argument("--min-shots-per-pt", type=float, default=12.0,
                    help="advisory shots/pt target (default 12; low is only a warning when the fit "
                         "is otherwise resolved -- see GATES)")
    ap.add_argument("--optical-factor", type=float, default=2.0,
                    help="swept-axis -> optical multiplier. The swept Pushout.Green.Freq AOM is "
                         "DOUBLE-pass, so optical = 2x swept, and it is the OPTICAL splitting that "
                         "equals Omega_308/2pi. Use 1.0 only for a single-pass axis.")
    args = ap.parse_args()

    scan_dir = _resolve_dir(args.scan)
    scan, sid, sid_arr, l1, l2, nshot = _snapshot(scan_dir)
    if args.max_shots is not None:
        if args.max_shots < 1:
            raise SystemExit("--max-shots must be >= 1")
        if args.max_shots > nshot:
            print("  NOTE: --max-shots %d > %d shots in the scan; using all of them"
                  % (args.max_shots, nshot))
        nshot = min(nshot, args.max_shots)
        sid_arr, l1, l2 = sid_arr[:nshot], l1[:nshot], l2[:nshot]
    sp, logic1, logic2, reps = unpack_scan_logicals(scan, seq_ids=sid_arr, logicals_img1=l1, logicals_img2=l2)
    sp = np.asarray(sp, float)
    x = sp.reshape(sp.shape[0], -1)[:, 0] / 1e6      # MHz
    order = np.argsort(x); x = x[order]
    M, _ = prob11_site_resolved(logic1, logic2)
    y = np.nanmean(M, axis=0)[order]                 # array-averaged survival

    m = np.isfinite(y)
    xf, yf = x[m], y[m]
    if xf.size < 7:
        raise SystemExit("too few finite points (%d)" % xf.size)
    ys = _smooth(y, 3)
    span = x.max() - x.min()
    y0g = float(np.nanpercentile(yf, 90))
    s1, s2 = _seed_two_dips(x, ys, args.min_sep)
    Ag = max(y0g - float(np.nanmin(yf)), 0.05)

    p0 = [y0g, Ag, s1, 0.5, Ag, s2, 0.5]
    lo = [0, 0, x.min(), 0.05, 0, x.min(), 0.05]
    hi = [1.2, 1.5, x.max(), span, 1.5, x.max(), span]
    try:
        p, pcov = curve_fit(_double_dip, xf, yf, p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception as e:
        raise SystemExit("double-dip fit failed: %s" % e)
    r2 = 1 - np.sum((yf - _double_dip(xf, *p)) ** 2) / np.sum((yf - yf.mean()) ** 2)

    # Parameter errors. The fit is UNWEIGHTED, so curve_fit already scales pcov by the residual
    # variance (ssr/(n-p)); sqrt(diag(pcov)) is therefore the residual-scaled standard error -- the
    # same number a bootstrap gives (verified 2026-09-15: pcov 0.0024 vs an 800-sample bootstrap
    # 0.0025 MHz on this splitting). So DO NOT hand-roll a bootstrap; these are the errors to quote.
    perr = np.sqrt(np.diag(pcov))

    # order the two components by center, carrying their errors
    comps = sorted([(p[2], abs(p[3]), p[1], perr[2], perr[3]),
                    (p[5], abs(p[6]), p[4], perr[5], perr[6])], key=lambda c: c[0])
    c1, w1, A1, ec1, ew1 = comps[0]
    c2, w2, A2, ec2, ew2 = comps[1]
    split = abs(c2 - c1)
    esplit = float(np.hypot(ec1, ec2))

    resid = yf - _double_dip(xf, *p)
    resid_std = float(np.std(resid))
    shots_per_pt = float(nshot) / xf.size
    loading = float(np.nanmean(logic1))
    avg_surv = float(np.nanmean(M))
    mean_fwhm = 0.5 * (w1 + w2)
    resolution = split / mean_fwhm if mean_fwhm > 0 else float("inf")
    depth_sigma = (min(A1, A2) / resid_std) if resid_std > 0 else float("inf")
    edge_margin = min(c1 - x.min(), x.max() - c2)
    W_LO = 0.05  # the width lower bound used in `lo` above

    # --- detection health (never fatal: the fit stands even if this cannot be read)
    health = detection_health(scan_dir, sid, scan_json=scan, n_shots=nshot)
    lh = loading_health(loading)

    # --- gates: the five that decide whether the FIT itself is sound
    gates = [("atoms loaded", not (lh and lh["dead"]),
              ("loading %.4f" % lh["mean"]) if lh else "unknown"),
             ("R2>=0.95", r2 >= 0.95, "R2 %.3f" % r2),
             ("resolved>=2xFWHM", resolution >= 2.0, "%.1fx FWHM" % resolution),
             ("depth>=5sigma", depth_sigma >= 5.0, "%.0fsigma" % depth_sigma),
             ("not-edge-pinned", edge_margin >= 0.05 * span, "%.2f MHz margin" % edge_margin),
             ("widths-off-bound", min(w1, w2) > 1.01 * W_LO, "min FWHM %.0f kHz" % (min(w1, w2) * 1e3))]
    fit_ok = all(ok for _, ok, _ in gates)
    # shots/pt is ADVISORY: the guideline exists to get the doublet resolved, so a low count does not
    # condemn a fit that is already resolved with significant depth. Never escalate on this alone.
    shots_ok = shots_per_pt >= args.min_shots_per_pt

    k = args.optical_factor
    print("AT scan %s | %d shots, %d pts (%.1f shots/pt), window %.2f-%.2f MHz"
          % (sid, nshot, xf.size, shots_per_pt, x.min(), x.max()))
    print("  " + format_provenance(run_provenance(scan_dir, sid, scan_json=scan)))
    print("  loading %.2f | avg survival %.2f | off-resonant baseline %.3f"
          % (loading, avg_surv, p[0]))
    print("  dip1 = %.4f +/- %.4f MHz (FWHM %.0f +/- %.0f kHz, depth %.3f)"
          % (c1, ec1, w1 * 1e3, ew1 * 1e3, A1))
    print("  dip2 = %.4f +/- %.4f MHz (FWHM %.0f +/- %.0f kHz, depth %.3f)"
          % (c2, ec2, w2 * 1e3, ew2 * 1e3, A2))
    print("  AT SPLITTING (swept axis) = %.4f +/- %.4f MHz   R^2 = %.3f%s"
          % (split, esplit, r2, "   *** LOW R^2 -- check the plot / --min-sep ***" if r2 < 0.7 else ""))
    if k != 1.0:
        print("  OPTICAL (x%g, double-pass AOM) = %.4f +/- %.4f MHz  == Omega_308/2pi"
              % (k, split * k, esplit * k))
        print("     ^ the SWEPT number is the splitting trend; the OPTICAL number is the Rabi"
              " frequency -- never mix them. COMPARE ONLY LIKE-FOR-LIKE: a splitting is only"
              " comparable across runs with the SAME window/center, the same 308 park"
              " (Init.EOM616.Freq) and the same Ryd308.Amp -- a different park can be a"
              " different Rydberg state entirely, and Omega is state-dependent.")
    mid = 0.5 * (c1 + c2)
    if args.bare is not None:
        print("  midpoint %.4f MHz = %+.0f kHz vs the bare line %.4f (%+.0f kHz optical)"
              " -- AT symmetry OK if |offset| << FWHM"
              % (mid, (mid - args.bare) * 1e3, args.bare, (mid - args.bare) * k * 1e3))
    else:
        print("  midpoint %.4f MHz  (pass --bare <bare-dip-MHz> for the AT symmetry check)" % mid)
    print("  " + format_health(health))
    print("  " + format_loading(lh))
    print("  GATES: " + " | ".join("%s %s" % (lbl, "PASS" if ok else "FAIL") for lbl, ok, _ in gates)
          + " || shots/pt %.1f %s" % (shots_per_pt, "OK" if shots_ok else "LOW(advisory)"))
    if lh and lh["dead"]:
        print("  VERDICT: CHECK -- NO ATOMS LOADED (%.4f); this measures the false-positive "
              "rate, not a doublet. Hand to yb:troubleshooting." % lh["mean"])
    elif fit_ok:
        print("  VERDICT: TRUST -- all fit gates pass%s" % (
            "" if shots_ok else
            ". shots/pt is below the %.0f/pt guideline, but the doublet is resolved at %.1fx FWHM"
            " with %.0fsigma depths, so the fit is sound -- do NOT re-scan for statistics alone"
            % (args.min_shots_per_pt, resolution, depth_sigma)))
    else:
        print("  VERDICT: CHECK -- failed: %s. Investigate before trusting this."
              % ", ".join(lbl for lbl, ok, _ in gates if not ok))
    out = {"scan_id": sid, "n_shots": nshot, "shots_per_pt": shots_per_pt,
           "dip1_MHz": c1, "dip1_err_MHz": ec1, "dip2_MHz": c2, "dip2_err_MHz": ec2,
           "fwhm1_kHz": w1 * 1e3, "fwhm1_err_kHz": ew1 * 1e3,
           "fwhm2_kHz": w2 * 1e3, "fwhm2_err_kHz": ew2 * 1e3,
           "splitting_MHz": split, "splitting_err_MHz": esplit,
           "optical_factor": k, "splitting_optical_MHz": split * k,
           "splitting_optical_err_MHz": esplit * k, "midpoint_MHz": mid,
           "midpoint_offset_kHz": (None if args.bare is None else (mid - args.bare) * 1e3),
           "r_squared": r2, "resolution_x_fwhm": resolution, "depth_sigma": depth_sigma,
           "baseline": float(p[0]), "loading": loading, "avg_survival": avg_surv,
           "edge_margin_MHz": edge_margin, "health": health,
           "gates": {lbl: bool(ok) for lbl, ok, _ in gates},
           "shots_per_pt_ok": bool(shots_ok), "verdict": ("TRUST" if fit_ok else "CHECK")}
    print("JSON " + json.dumps(out))

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xg = np.linspace(x.min(), x.max(), 400)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        ax.plot(x, y, "ko", ms=4, label="array-avg survival")
        ax.plot(xg, _double_dip(xg, *p), "-", color="C1", lw=2, label="double Lorentzian  R$^2$=%.3f" % r2)
        ax.plot(xg, p[0] - _lor(xg, p[1], p[2], p[3]), "--", color="C1", lw=0.8, alpha=0.6)
        ax.plot(xg, p[0] - _lor(xg, p[4], p[5], p[6]), "--", color="C1", lw=0.8, alpha=0.6)
        for c in (c1, c2):
            ax.axvline(c, color="C3", ls=":", lw=0.8)
        ax.annotate("", xy=(c1, 0.55), xytext=(c2, 0.55), arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
        ax.text((c1 + c2) / 2, 0.57, "AT splitting = %.3f MHz" % split, ha="center", fontsize=11, color="0.2")
        ax.set_xlabel("556 freq [MHz]"); ax.set_ylabel("survival (P11)")
        ax.set_title("556 Autler-Townes (array-avg)  %s\nd1 %.4f / d2 %.4f MHz, split %.3f MHz, %d shots"
                     % (sid, c1, c2, split, nshot), fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        suffix = "" if args.max_shots is None else "_first%d" % nshot
        png = os.path.join(scan_dir, "at_avg_doublefit_%s%s.png" % (sid, suffix))
        fig.savefig(png, bbox_inches="tight")
        print("  saved %s" % png)


if __name__ == "__main__":
    main()
