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
                               run_provenance, format_provenance, scan_completeness,
                               loading_health, format_loading)

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
            # Several params can share a dim (a CO-VARYING axis). Keep them all -- keying by dim
            # alone silently drops every name but the last.
            axes.setdefault(int(v["scan"]), []).append((k, [float(x) for x in v["values"]]))
    out = []
    for _, group in sorted(axes.items()):
        out.append((group[0][0], group[0][1], [n for n, _ in group[1:]]))
    return js, out


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
    ap.add_argument("--observable", choices=("auto", "survival", "loading"), default="auto",
                    help="what to optimise. 'auto' (default) uses survival when the run has it "
                         "and falls back to LOADING for single-image runs (a loading or MOT "
                         "optimisation has no survival at all)")
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
    surv = np.asarray(s["survival_mean"], float)
    ld = np.asarray(s.get("loading_rate") or [], float)

    # Pick the observable. A loading / MOT optimisation is a single-image run: it has no
    # survival at all, and survival_mean comes back all-NaN. Fitting that used to crash in
    # nanargmin; the right observable there is the LOADING rate.
    obs = args.observable
    if obs == "auto":
        obs = "survival" if np.isfinite(surv).sum() >= 3 else "loading"
    if obs == "survival":
        y = surv
        e = np.asarray(s.get("survival_sem_pershot") or [], float)
        nsh = np.asarray(s.get("survival_n_shots") or [], float)
        obs_label = "survival (P11)"
    else:
        y = ld
        e = np.asarray(s.get("loading_sem_pershot") or s.get("loading_rate_sem") or [], float)
        nsh = np.asarray(s.get("loading_n_shots") or s.get("survival_n_shots") or [], float)
        obs_label = "loading rate"
    if np.isfinite(y).sum() < 3:
        raise SystemExit(
            "no usable observable: survival has %d finite points and loading %d. This run may be "
            "empty, aborted, or of a kind that carries neither -- nothing to optimise here."
            % (int(np.isfinite(surv).sum()), int(np.isfinite(ld).sum())))
    if obs == "loading" and not args.maximize:
        print("  NOTE: optimising LOADING and minimising it -- for a loading/MOT optimisation you "
              "almost certainly want --maximize.")

    shape = tuple(len(a[1]) for a in axes)
    if y.size != int(np.prod(shape)):
        raise SystemExit("grid mismatch: %d points but axes are %s -- the run may have been "
                         "aborted mid-pass. Try --recache, or treat it as incomplete."
                         % (y.size, "x".join(str(n) for n in shape)))

    # yb-basic: multi-axis scans are column-major, dim-0 fastest.
    Y = y.reshape(shape, order="F")
    E = e.reshape(shape, order="F") if e.size == y.size else np.full(shape, np.nan)
    N = nsh.reshape(shape, order="F") if nsh.size == y.size else np.full(shape, np.nan)
    disp = [_scale(a[0], a[1]) for a in axes]

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
    print("  observable: %s%s" % (obs_label,
          "" if args.observable != "auto" else
          ("  (auto: survival is all-NaN here, so this is a single-image run)"
           if obs == "loading" else "")))
    print("%s map %s | %s shots, %s = %d pts (%.1f shots/pt)"
          % ("2-D" if len(shape) == 2 else "1-D", fid, d.get("n_shots"), dims, y.size,
             shots_per_pt))
    print("  " + format_provenance(run_provenance(scan_dir, fid, scan_json=js)))
    for k, a in enumerate(axes):
        xk, labk = disp[k]
        print("  axis dim%d %s  %.4f..%.4f (%d pts)" % (k + 1, labk, xk[0], xk[-1], len(a[1])))
        if a[2]:
            print("       ^ CO-VARYING on this axis: %s -- the optimum below is a point on that "
                  "path, NOT an independent optimum in each parameter" % ", ".join(a[2]))
    if ld.size and np.isfinite(ld).any():
        print("  loading %.2f-%.2f | %s %.3f-%.3f"
              % (np.nanmin(ld), np.nanmax(ld), obs_label, np.nanmin(Y), np.nanmax(Y)))
    print("  " + format_health(health))
    lh = loading_health(ld if ld.size else Y)
    print("  " + format_loading(lh))
    print("")
    at = ",  ".join("%s %.4f" % (disp[k][1], disp[k][0][i]) for k, i in enumerate(idx))
    print("  %s = %.3f +/- %.3f  at  %s"
          % (("MAX " if args.maximize else "MIN ") + obs_label, best, best_err, at))
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

    # (1) An optimum with no error bar is not an optimum. best_err comes back nan when a point
    #     has too few shots to estimate one, and every 2-sigma comparison below is then vacuous.
    opt_constrained = bool(np.isfinite(best_err) and best_err > 0)
    if not opt_constrained:
        print("  *** OPTIMUM NOT USABLE: the best point has no finite error bar (too few shots at "
              "that point), so it cannot be distinguished from any other. DO NOT report it as the "
              "optimum. ***")
    # (2) No modulation across the whole map = nothing was resolved, whatever the best pixel says.
    span_y = float(np.nanmax(Y) - np.nanmin(Y))
    noise = float(err_med) if np.isfinite(err_med) else float("nan")
    modulated = bool(np.isfinite(noise) and noise > 0 and span_y > 5.0 * noise)
    if not modulated:
        print("  *** NO STRUCTURE: survival spans only %.3f across the whole map against a typical "
              "point error of %.3f -- that is noise, not a feature. DO NOT report an optimum. ***"
              % (span_y, noise))
    # An optimisation is only meaningful if the observable reached a usable value SOMEWHERE.
    # Healthy tweezer loading is ~0.3-0.6 (yb-basic); a grid whose best point is ~0.002 did not
    # load anywhere, and its "optimum" is the best of nothing.
    # Judge "did anything load" by the RIGHT statistic for the scan's purpose:
    #   * a LOADING optimisation is SEARCHING for a setting that loads, so most of the grid is
    #     legitimately near zero. Use the BEST point -- mean loading would call a successful
    #     optimisation dead just because most of its grid failed.
    #   * a SURVIVAL scan should have loaded at EVERY point, so use the mean.
    loading_floor = 0.05
    if obs == "loading":
        dead_loading = bool(np.nanmax(Y) < loading_floor)
    else:
        dead_loading = bool(lh and lh["dead"])
    if dead_loading:
        print("  *** NOTHING LOADED: the best point on this grid reaches loading %.4f, far below "
              "the ~0.3-0.6 healthy band. No setting in this window loads atoms, so the 'optimum' "
              "is the best of nothing. Fix loading first -- this is not an optimisation result. ***"
              % np.nanmax(Y))
    comp = scan_completeness(scan_dir, fid, d.get("n_shots"), scan_json=js)
    if comp and comp[2] < 0.5:
        print("  NOTE: this run is %d of %d scheduled shots (%.0f%% complete) -- the gates are "
              "reading an unfinished scan" % (comp[0], comp[1], 100 * comp[2]))

    resolved = n_tied <= max(3, y.size // 10)
    gates = [("observable usable", not dead_loading,
              ("max loading %.4f" % np.nanmax(Y)) if obs == "loading" else "n/a"),
             ("interior (bracketed)", interior, "" if interior else "optimum on a window edge"),
             ("optimum constrained", opt_constrained,
              ("+/-%.3f" % best_err) if opt_constrained else "no error bar"),
             ("structure present", modulated, "span %.3f vs noise %.3f" % (span_y, noise)),
             ("optimum resolved", resolved, "%d pts tied" % n_tied),
             ("detection healthy", bool(health["healthy"]) if health else True,
              ("d' %.1f" % health["dprime_median"]) if health else "unknown")]
    shots_ok = shots_per_pt >= args.min_shots_per_pt
    ok = all(g[1] for g in gates)
    print("  GATES: " + " | ".join("%s %s" % (g[0], "PASS" if g[1] else "FAIL") for g in gates)
          + " || shots/pt %.1f %s" % (shots_per_pt, "OK" if shots_ok else "LOW"))
    if ok and shots_ok:
        print("  VERDICT: TRUST -- the optimum is interior and resolved.")
    elif dead_loading:
        print("  VERDICT: CHECK -- NOTHING LOADED anywhere on this grid (best %.4f). There is no "
              "optimum to report; fix loading before reading anything off this scan."
              % np.nanmax(Y))
    elif not modulated:
        print("  VERDICT: CHECK -- NO STRUCTURE: the surface is flat within noise, so no optimum "
              "is established here. Move the window onto the feature, or add shots.")
    elif not opt_constrained:
        print("  VERDICT: CHECK -- the best point has NO ERROR BAR, so no optimum is established. "
              "Add shots before reading anything off this map.")
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
    elif dead_loading:
        escalate = False
        why_esc = ("nothing loaded anywhere on this grid (best %.4f vs a ~0.3-0.6 healthy band). "
                   "That is a loading/apparatus problem upstream of this scan, not something to "
                   "read an optimum from -- if loading was healthy when this ran, it belongs to "
                   "the troubleshooter" % np.nanmax(Y))
    elif comp and comp[2] < 0.5:
        escalate = False
        why_esc = ("the run is only %.0f%% complete (%d of %d scheduled shots) -- that alone "
                   "explains the failed gate. Let it finish, then refit. Report NO optimum"
                   % (100 * comp[2], comp[0], comp[1]))
    elif not modulated:
        escalate = False
        why_esc = ("there is NO STRUCTURE in this map -- the whole surface is flat within noise. "
                   "That is a window-placement or statistics problem, not something to "
                   "investigate: move the window onto the feature, or add shots")
    elif not opt_constrained:
        escalate = False
        why_esc = ("the best point has no error bar, so no optimum is established. Add shots "
                   "before reading anything off this map")
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
           "observable": obs, "n_shots": d.get("n_shots"), "shots_per_pt": shots_per_pt,
           "axes": [a[0] for a in axes],
           "covarying": {a[0]: a[2] for a in axes if a[2]},
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
                fig.colorbar(im, ax=ax, label=obs_label)
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
                ax.set_ylabel(obs_label)
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
