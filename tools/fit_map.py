"""fit_map.py -- summarise and JUDGE a swept optimization scan (1-D or 2-D).

This is the model-free counterpart of the shape fitters (fit_spectrum / fit_at_splitting /
fit_decay / fit_rabi_*). Use it when the question is "where is the optimum and do I believe it",
not "what is this line's centre". Two cases it exists for:

  * **2-D scans** -- the one shape the 1-D fitters silently get WRONG: hand a freq-2D to
    fit_spectrum.py and it fits a Lorentzian to a grid that is not a line at all.
  * **1-D optimization sweeps with no lineshape** (an amplitude, a gap, a coil current): fitting
    a Lorentzian to those is equally meaningless, but an optimum + an error bar is exactly right.

Same contract as the other fit tools: print the numbers, the detection health, the as-run
operating point, explicit GATES and a VERDICT, and save a figure. What it answers:

  * where is the optimum, with an error bar;
  * is it INTERIOR (bracketed) or on a window EDGE -- i.e. is the real optimum outside the grid;
  * is it a POINT, or a RIDGE / PLATEAU of statistically indistinguishable points? On a
    two-photon freq-2D it is nearly always a ridge, and quoting one "best" pixel off a noisy
    ridge is the classic way to fool yourself;
  * are there enough shots per point to tell those apart at all.

    <yb_analysis-python> pyctrl/tools/fit_map.py <scan_id> [--maximize] [--no-plot]

Default is MINIMIZE (survival low = more excitation/loss, the push-out convention). Pass
--maximize when high survival is the goal (a cooling or imaging-survival optimisation).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _detection_health import (detection_health, format_health,      # noqa: E402
                               run_provenance, format_provenance)

DATA = r"D:\OneDrive - Harvard University\Documents - Yb\Data"


def _scan_dir(sid):
    sid = str(sid).replace("_", "")
    fid = sid[:8] + "_" + sid[8:]
    d = os.path.join(DATA, sid[:8], "data_" + fid)
    if not os.path.isdir(d):
        raise SystemExit("no such scan dir: %s" % d)
    return d, fid


def swept_axes(scan_dir, fid):
    """The swept axes in dim order, straight from the run's own descriptor."""
    with open(os.path.join(scan_dir, "data_%s.json" % fid)) as fh:
        js = json.load(fh)
    desc = js.get("descriptor")
    if isinstance(desc, str):
        desc = json.loads(desc)
    axes = {}
    for k, v in ((desc or {}).get("params") or {}).items():
        if isinstance(v, dict) and "scan" in v:
            axes[int(v["scan"])] = (k, [float(x) for x in v["values"]])
    return js, [axes[d] for d in sorted(axes)]


def _scale(name, vals):
    """Display in MHz when an axis is plainly stored in Hz; us for sub-ms times."""
    v = np.asarray(vals, float)
    if np.nanmax(np.abs(v)) > 1e6:
        return v / 1e6, name + " [MHz]"
    if np.nanmax(np.abs(v)) < 1e-3 and np.nanmax(np.abs(v)) > 0:
        return v * 1e6, name + " [us]"
    return v, name


def main():
    ap = argparse.ArgumentParser(description="Summarise + judge a 1-D or 2-D optimization scan.")
    ap.add_argument("scan")
    ap.add_argument("--maximize", action="store_true",
                    help="optimum = HIGHEST survival (default: lowest, the push-out convention)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--min-shots-per-pt", type=float, default=5.0)
    ap.add_argument("--recache", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    from yb_analysis.analysis.run_analysis import analyze_scan

    scan_dir, fid = _scan_dir(args.scan)
    js, axes = swept_axes(scan_dir, fid)
    if len(axes) not in (1, 2):
        raise SystemExit("fit_map handles 1-D and 2-D sweeps; this scan has %d swept axes."
                         % len(axes))

    sid = str(args.scan).replace("_", "")
    d = analyze_scan(sid, include_per_site=False, include_diag_aggregate=False,
                     force_recache=args.recache)
    s = d["summary"]
    y = np.asarray(s["survival_mean"], float)
    e = np.asarray(s.get("survival_sem_pershot") or [], float)
    nsh = np.asarray(s.get("survival_n_shots") or [], float)
    ld = np.asarray(s.get("loading_rate") or [], float)

    shape = tuple(len(v) for _, v in axes)
    if y.size != int(np.prod(shape)):
        raise SystemExit("grid mismatch: %d points but axes are %s -- the run may have been "
                         "aborted mid-pass. Try --recache, or treat it as incomplete."
                         % (y.size, "x".join(str(n) for n in shape)))

    # yb-basic: multi-axis scans are column-major, dim-0 fastest.
    Y = y.reshape(shape, order="F")
    E = e.reshape(shape, order="F") if e.size == y.size else np.full(shape, np.nan)
    N = nsh.reshape(shape, order="F") if nsh.size == y.size else np.full(shape, np.nan)
    disp = [_scale(n, v) for n, v in axes]

    sign = -1.0 if args.maximize else 1.0
    idx = np.unravel_index(np.nanargmin(sign * Y), shape)
    best, best_err = Y[idx], E[idx]
    shots_per_pt = float(np.nanmean(N))

    err_med = np.nanmedian(E[np.isfinite(E)]) if np.isfinite(E).any() else np.nan
    err = np.where(np.isfinite(E) & (E > 0), E, err_med)
    be = best_err if np.isfinite(best_err) and best_err > 0 else err_med
    with np.errstate(invalid="ignore"):
        z = (sign * (Y - best)) / np.hypot(err, be)
    tied = np.argwhere(np.isfinite(z) & (z < 2.0))
    n_tied = len(tied)
    interior = all(0 < i < shape[k] - 1 for k, i in enumerate(idx))
    health = detection_health(scan_dir, fid, scan_json=js, n_shots=d.get("n_shots"))

    # ---- ridge (2-D only): do the tied points span BOTH axes, i.e. a line not a blob?
    ridge = None
    if len(shape) == 2 and n_tied >= 3:
        ii, jj = tied[:, 0], tied[:, 1]
        if len(set(ii.tolist())) >= 2 and len(set(jj.tolist())) >= 2:
            x0, x1 = disp[0][0], disp[1][0]
            slope, icept = np.polyfit(x0[ii], x1[jj], 1)
            ridge = {"slope": float(slope), "intercept": float(icept),
                     "scatter": float(np.std(x1[jj] - (slope * x0[ii] + icept))),
                     "n": int(n_tied)}

    # ---------------------------------------------------------------- report
    dims = "x".join(str(n) for n in shape)
    print("%s map %s | %s shots, %s = %d pts (%.1f shots/pt)"
          % ("2-D" if len(shape) == 2 else "1-D", fid, d.get("n_shots"), dims, y.size,
             shots_per_pt))
    print("  " + format_provenance(run_provenance(scan_dir, fid, scan_json=js)))
    for k, (name, vals) in enumerate(axes):
        xk, labk = disp[k]
        print("  axis dim%d %s  %.4f..%.4f (%d pts)" % (k + 1, labk, xk[0], xk[-1], len(vals)))
    if ld.size:
        print("  loading %.2f-%.2f | survival %.3f-%.3f"
              % (np.nanmin(ld), np.nanmax(ld), np.nanmin(Y), np.nanmax(Y)))
    print("  " + format_health(health))
    print("")
    at = ",  ".join("%s %.4f" % (disp[k][1], disp[k][0][i]) for k, i in enumerate(idx))
    print("  %s = %.3f +/- %.3f  at  %s"
          % ("MAX survival" if args.maximize else "MIN survival", best, best_err, at))
    print("  within 2 sigma of it: %d of %d points%s"
          % (n_tied, y.size, "" if n_tied <= 1 else
             "  <- the optimum is NOT a single point"))
    if ridge:
        print("  RIDGE: %s = %.4f * %s + %.4f  (scatter %.4f over %d pts) -- quote the ridge and a"
              " point ON it, not the best pixel"
              % (disp[1][1], ridge["slope"], disp[0][1], ridge["intercept"],
                 ridge["scatter"], ridge["n"]))
    elif n_tied > 1 and len(shape) == 1:
        lo = disp[0][0][tied[:, 0].min()]
        hi = disp[0][0][tied[:, 0].max()]
        print("  PLATEAU: %s %.4f..%.4f is indistinguishable from the optimum -- quote a range"
              % (disp[0][1], lo, hi))

    resolved = n_tied <= max(3, y.size // 10)
    gates = [("interior (bracketed)", interior, "" if interior else "optimum on a window edge"),
             ("optimum resolved", resolved, "%d pts tied" % n_tied),
             ("detection healthy", bool(health["healthy"]) if health else True,
              ("d' %.1f" % health["dprime_median"]) if health else "unknown")]
    shots_ok = shots_per_pt >= args.min_shots_per_pt
    ok = all(g[1] for g in gates)
    print("  GATES: " + " | ".join("%s %s" % (g[0], "PASS" if g[1] else "FAIL") for g in gates)
          + " || shots/pt %.1f %s" % (shots_per_pt, "OK" if shots_ok else "LOW"))
    if ok and shots_ok:
        print("  VERDICT: TRUST -- the optimum is interior and resolved.")
    elif not interior:
        print("  VERDICT: CHECK -- the optimum sits on a WINDOW EDGE, so the true optimum is"
              " probably OUTSIDE this grid. Extend or shift the window along that axis before"
              " treating this as an optimum.")
    elif not resolved:
        print("  VERDICT: CHECK -- %d points are within 2 sigma of the best, so this grid"
              " localizes a %s, not a point.%s"
              % (n_tied, "ridge" if ridge else ("plateau" if len(shape) == 1 else "region"),
                 "" if shots_ok else " Raise shots/pt before reading any best point off it."))
    else:
        print("  VERDICT: CHECK -- failed: %s"
              % ", ".join(g[0] for g in gates if not g[1]))

    # Is this CHECK self-explaining? A window-design failure (edge, ridge/plateau, thin
    # statistics) comes with its own prescribed fix and needs no interpretation. A DATA failure
    # (detection unhealthy) does. Only the latter is worth escalating.
    detection_bad = bool(health) and not health["healthy"]
    escalate = detection_bad
    if ok and shots_ok:
        why_esc = "verdict is TRUST"
    elif detection_bad:
        why_esc = ("detection is SUSPECT (d' %.1f, fill %.2f) -- that is a data-quality problem, "
                   "not a window problem; it may belong to the troubleshooter"
                   % (health["dprime_median"], health["fill_median"]))
    else:
        fix = ("extend or shift the window along the failing axis" if not interior
               else ("raise shots/pt" if not shots_ok
                     else "quote the ridge/plateau instead of a single point"))
        why_esc = ("the failure is WINDOW DESIGN, already diagnosed above -- the fix is to %s. "
                   "Report that and stop; investigating adds nothing" % fix)
    print("  ESCALATE: %s -- %s" % ("YES" if escalate else "no", why_esc))

    out = {"scan_id": sid, "ndim": len(shape), "shape": list(shape),
           "escalate": bool(escalate), "escalate_reason": why_esc,
           "n_shots": d.get("n_shots"), "shots_per_pt": shots_per_pt,
           "axes": [n for n, _ in axes],
           "best": float(best),
           "best_err": (None if not np.isfinite(best_err) else float(best_err)),
           "best_at": [float(disp[k][0][i]) for k, i in enumerate(idx)],
           "interior": bool(interior), "n_within_2sigma": int(n_tied), "ridge": ridge,
           "health": health, "gates": {g[0]: bool(g[1]) for g in gates},
           "shots_per_pt_ok": bool(shots_ok),
           "verdict": ("TRUST" if (ok and shots_ok) else "CHECK")}
    print("JSON " + json.dumps(out))

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if len(shape) == 2:
                x0, lab0 = disp[0]
                x1, lab1 = disp[1]
                fig, ax = plt.subplots(figsize=(7.5, 6), dpi=130)
                im = ax.pcolormesh(x1, x0, Y, shading="nearest", cmap="viridis")
                fig.colorbar(im, ax=ax, label="survival (P11)")
                ax.plot(x1[idx[1]], x0[idx[0]], "r*", ms=16, mec="w", label="best %.3f" % best)
                if n_tied > 1:
                    ax.plot(x1[tied[:, 1]], x0[tied[:, 0]], "w.", ms=4, alpha=0.8,
                            label="within 2$\\sigma$ (%d)" % n_tied)
                if ridge:
                    ax.plot(ridge["slope"] * x0 + ridge["intercept"], x0, "w--", lw=1,
                            label="ridge, slope %.3f" % ridge["slope"])
                ax.set_xlabel(lab1)
                ax.set_ylabel(lab0)
                ax.legend(fontsize=7, loc="best")
            else:
                x0, lab0 = disp[0]
                fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
                ax.errorbar(x0, Y, yerr=np.where(np.isfinite(E), E, 0), fmt="ko-", ms=4, lw=1,
                            capsize=2, label="survival")
                ax.plot(x0[idx[0]], best, "r*", ms=16, label="best %.3f" % best)
                if n_tied > 1:
                    ax.axvspan(x0[tied[:, 0].min()], x0[tied[:, 0].max()], color="C1", alpha=0.15,
                               label="within 2$\\sigma$ (%d pts)" % n_tied)
                ax.set_xlabel(lab0)
                ax.set_ylabel("survival (P11)")
                ax.legend(fontsize=8)
            ax.set_title("%s  %s, %.1f shots/pt  ->  %s"
                         % (fid, dims, shots_per_pt, out["verdict"]), fontsize=9)
            fig.tight_layout()
            png = os.path.join(scan_dir, "fit_map_%s.png" % fid)
            fig.savefig(png, bbox_inches="tight")
            print("  saved %s" % png)
        except Exception as exc:
            print("  (plot skipped: %s)" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
