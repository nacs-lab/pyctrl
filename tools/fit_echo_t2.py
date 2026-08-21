"""fit_echo_t2.py -- Hahn-echo T2 from a 2D (closing-pi/2 phase x free-evolution T) scan.

The 2026-07-24 method (Notion "07/24 Hahn-echo, Dipolar spin-exchange"):
  1. Group shots per (phase, T) via config['Params'] (1-indexed, flat COLUMN-MAJOR: dim1 fastest).
  2. Per T: fit  P(phi) = y0 + (C/2) * cos(phi + phi0)  -> C(T) = echo fringe contrast.
     Drop T points whose cosine fit has R^2 < 0.4 (no coherence left to read).
  3. Fit C(T) = C0 * exp(-(T/T2)^n)  (stretched exponential; also report n=1 and n=2 fixed).
Sanity checks printed: phi0(T) walk (residual detuning), monotonic C(T).

Axis roles are AUTODETECTED from the sweep axis names (works with phase on dim1 or dim2 --
the 08-19 re-run has T on dim1, phase on dim2; 07-24 had the transpose).

Metric: mid-conditioned survival (survival_ref='mid'), target-only when slm_diag allows,
via analyze_scan's per-2D bucketing... implemented directly from the h5 logicals + Params
(the payload's survival_mean is 1-D-flattened; we need the 2-D grid, so we re-pair here).

Run with the yb_analysis env python:
  <yb_analysis-python> fit_echo_t2.py <scan_dir> [--all-sites]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--all-sites", action="store_true",
                    help="use all mid-occupied sites (default: slm_diag target union when present)")
    ap.add_argument("--rsq-min", type=float, default=0.4,
                    help="min cosine-fit R^2 for a T point to enter the C(T) fit (default 0.4)")
    args = ap.parse_args()

    import h5py
    import numpy as np
    from scipy.optimize import curve_fit

    sd = args.scan_dir.rstrip("\\/")
    sid = os.path.basename(sd).replace("data_", "")
    cfg = json.load(open(os.path.join(sd, "data_%s.json" % sid)))
    P = np.asarray(cfg["Params"], int)

    with h5py.File(os.path.join(sd, "data_%s.h5" % sid), "r", swmr=True) as f:
        n = min(f["logicals_mid"].shape[0], f["logicals_img2"].shape[0], f["seq_ids"].shape[0])
        seq = np.asarray(f["seq_ids"][:n], np.int64)
        mid = f["logicals_mid"][:n].astype(bool)
        fin = f["logicals_img2"][:n].astype(bool)
    nS = mid.shape[1]

    # target mask (union of slm_diag target_paired)
    sites = None
    diag = os.path.join(sd, "slm_diag.h5")
    if not args.all_sites and os.path.exists(diag):
        with h5py.File(diag, "r", swmr=True) as f:
            g = f["/diag"]
            dmap = {int(s): np.asarray(g["target_paired"][i]).ravel()
                    for i, s in enumerate(np.asarray(g["seq_id"][:], np.int64))}
        tm = np.zeros(nS, bool)
        for s in seq:
            t = dmap.get(int(s))
            if t is not None:
                t = t[(t >= 0) & (t < nS)]
                tm[t] = True
        if tm.any():
            sites = np.where(tm)[0]
            print("target mask: %d sites (slm_diag union)" % sites.size)
    if sites is None:
        occ = mid.mean(0)
        sites = np.where(occ >= 0.25)[0]
        print("occupancy fallback: %d sites" % sites.size)

    # ---- 2D axes from the analysis payload's sweep (cols/dims/values, dim order = 1,2) ---
    pj = os.path.join(sd, "analysis_payload.json")
    if not os.path.exists(pj):
        # never analyzed -> generate the payload (also snapshots sweep values)
        from yb_analysis.analysis.run_analysis import analyze_scan
        analyze_scan(sid.replace("_", ""), include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False)
    pl = json.load(open(pj))["payload"]
    sw = pl["sweep"]
    cols = [str(c) for c in sw["cols"]]
    dvals = [np.asarray(v, float) for v in sw["values"]]
    assert len(cols) == 2, "need a 2D scan, got axes %s" % cols
    n1, n2 = len(dvals[0]), len(dvals[1])
    roles = ["phase" if "phase" in c.lower() else
             ("T" if ("wait" in c.lower() or "gap" in c.lower()) else "?") for c in cols]
    assert set(roles) == {"phase", "T"}, "cannot identify axes: %s" % list(zip(cols, roles))
    i_ph = roles.index("phase")
    ph_vals = dvals[i_ph]                              # deg
    T_vals = dvals[1 - i_ph] * 1e6                     # us (stored in s)
    d_phase = i_ph + 1
    print("axes: dim%d=phase (%d pts, %g-%g deg), dim%d=T (%d pts, %g-%g us)"
          % (d_phase, len(ph_vals), ph_vals[0], ph_vals[-1], 2 - i_ph, len(T_vals),
             T_vals[0], T_vals[-1]))

    # ---- per-(flat param) mid-conditioned survival over target sites --------------------
    n_flat = n1 * n2
    li = seq - 1
    ok = (li >= 0) & (li < P.size)
    p_of = P[li[ok]] - 1
    m = mid[ok][:, sites]; fq = fin[ok][:, sites]
    num = np.zeros(n_flat); den = np.zeros(n_flat)
    for p in range(n_flat):
        sh = p_of == p
        if sh.any():
            num[p] = (m[sh] & fq[sh]).sum(); den[p] = m[sh].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = np.where(den > 0, num / den, np.nan)
        serr = np.where(den > 0, np.sqrt(np.maximum(surv * (1 - surv), 0) / np.maximum(den, 1)),
                        np.nan)
    # column-major: flat p -> i(dim1) = p % n1, j(dim2) = p // n1
    S = surv.reshape(n2, n1).T          # S[i_dim1, j_dim2]
    E = serr.reshape(n2, n1).T
    if d_phase == 1:
        Sg, Eg = S, E                   # [phase, T]
    else:
        Sg, Eg = S.T, E.T
    print("shots used %d | grid %d phase x %d T | shots/pt ~%.1f"
          % (int(ok.sum()), len(ph_vals), len(T_vals), ok.sum() / n_flat))

    # ---- per-T cosine fit -> C(T) --------------------------------------------------------
    def cosf(phi_deg, y0, C, phi0):
        return y0 + (C / 2.0) * np.cos(np.deg2rad(phi_deg) + phi0)

    rows = []
    for j, T in enumerate(T_vals):
        y = Sg[:, j]; e = Eg[:, j]
        mm = np.isfinite(y)
        if mm.sum() < 4:
            rows.append((T, np.nan, np.nan, np.nan, -1)); continue
        try:
            p0 = [float(np.nanmean(y)), float(2 * (np.nanmax(y) - np.nanmin(y)) / 2), 0.0]
            popt, pcov = curve_fit(cosf, ph_vals[mm], y[mm], p0=p0,
                                   sigma=np.where(np.isfinite(e[mm]) & (e[mm] > 0), e[mm], 0.08),
                                   bounds=([0, 0, -np.pi], [1.2, 1.5, np.pi]), maxfev=20000)
            r = y[mm] - cosf(ph_vals[mm], *popt)
            sst = float(np.sum((y[mm] - y[mm].mean()) ** 2))
            r2 = 1 - float(np.sum(r ** 2)) / sst if sst > 0 else np.nan
            rows.append((T, popt[1], float(np.sqrt(pcov[1, 1])), popt[2], r2))
        except Exception:
            rows.append((T, np.nan, np.nan, np.nan, -1))
    print("  T[us]   C       +-      phi0[rad]  R2")
    for T, C, Ce, ph0, r2 in rows:
        print("  %-7.3g %-7.3f %-7.3f %-9.2f  %.2f" % (T, C, Ce, ph0, r2))

    good = [(T, C, Ce) for T, C, Ce, _p, r2 in rows if np.isfinite(C) and r2 >= args.rsq_min]
    out = {"scan_id": sid, "scan_dir": sd, "n_shots": int(ok.sum()),
           "phase_pts": len(ph_vals), "T_pts": len(T_vals),
           "per_T": [{"T_us": float(T), "C": float(C), "C_err": float(Ce),
                      "phi0_rad": float(p), "r2": float(r2)} for T, C, Ce, p, r2 in rows],
           "rsq_min": args.rsq_min}
    if len(good) >= 3:
        Tg = np.array([g[0] for g in good]); Cg = np.array([g[1] for g in good])
        Ceg = np.array([max(g[2], 1e-3) for g in good])

        def stretch(T, C0, T2, nexp):
            return C0 * np.exp(-(T / np.maximum(T2, 1e-3)) ** nexp)

        fits = {}
        for lbl, nfix in (("free", None), ("n1", 1.0), ("n2", 2.0)):
            try:
                if nfix is None:
                    popt, pcov = curve_fit(stretch, Tg, Cg, p0=[Cg[0], 10.0, 1.0], sigma=Ceg,
                                           bounds=([0, 0.1, 0.2], [1.5, 500, 4]), maxfev=20000)
                    C0, T2, nn = popt; T2e = float(np.sqrt(pcov[1, 1])); ne = float(np.sqrt(pcov[2, 2]))
                else:
                    fpart = lambda T, C0, T2: stretch(T, C0, T2, nfix)
                    popt, pcov = curve_fit(fpart, Tg, Cg, p0=[Cg[0], 10.0], sigma=Ceg,
                                           bounds=([0, 0.1], [1.5, 500]), maxfev=20000)
                    C0, T2 = popt; nn, ne = nfix, 0.0; T2e = float(np.sqrt(pcov[1, 1]))
                pred = stretch(Tg, C0, T2, nn)
                sst = float(np.sum((Cg - Cg.mean()) ** 2))
                r2 = 1 - float(np.sum((Cg - pred) ** 2)) / sst if sst > 0 else np.nan
                fits[lbl] = {"C0": float(C0), "T2_us": float(T2), "T2_err_us": T2e,
                             "n": float(nn), "n_err": ne, "r2": r2}
                print("  [C(T) %s] C0=%.3f T2=%.2f +- %.2f us n=%.2f +- %.2f R2=%.3f"
                      % (lbl, C0, T2, T2e, nn, ne, r2))
            except Exception as ex:
                print("  [C(T) %s] fit failed: %s" % (lbl, ex))
        out["decay_fits"] = fits
    else:
        print("  only %d usable T points (R2 >= %.2f) -- no decay fit yet" % (len(good), args.rsq_min))

    # ---- plot: per-T phase scans + C(T) --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nT = len(T_vals)
        fig, axs = plt.subplots(1, nT + 1, figsize=(3.3 * (nT + 1), 3.4))
        phf = np.linspace(ph_vals.min(), ph_vals.max(), 300)
        for j, T in enumerate(T_vals):
            axp = axs[j]
            axp.errorbar(ph_vals, Sg[:, j], yerr=Eg[:, j], fmt="o", ms=4, color="C0", capsize=2)
            T_, C, Ce, ph0, r2 = rows[j]
            if np.isfinite(C):
                y0fit = float(np.nanmean(Sg[:, j]))
                axp.plot(phf, cosf(phf, y0fit, C, ph0), "-", color="C3", lw=1.3)
            axp.set_title("T=%.3g us  C=%.2f R2=%.2f" % (T, C, max(r2, 0)), fontsize=8)
            axp.set_xlabel("closing pi/2 phase [deg]", fontsize=7)
            axp.grid(alpha=0.25); axp.tick_params(labelsize=7)
        axc = axs[-1]
        Ts = [r[0] for r in rows]; Cs = [r[1] for r in rows]; Ces = [r[2] for r in rows]
        gm = [r[4] >= args.rsq_min for r in rows]
        axc.errorbar([t for t, g in zip(Ts, gm) if g], [c for c, g in zip(Cs, gm) if g],
                     yerr=[e for e, g in zip(Ces, gm) if g], fmt="o", color="C0", capsize=3)
        axc.errorbar([t for t, g in zip(Ts, gm) if not g], [c for c, g in zip(Cs, gm) if not g],
                     yerr=[e for e, g in zip(Ces, gm) if not g], fmt="o", color="0.7", capsize=3)
        if len(good) >= 3 and "free" in out.get("decay_fits", {}):
            ft = out["decay_fits"]["free"]
            tf = np.linspace(min(Ts), max(Ts), 300)
            axc.plot(tf, ft["C0"] * np.exp(-(tf / ft["T2_us"]) ** ft["n"]), "-", color="C3",
                     label="T2=%.1f+-%.1f us n=%.2f" % (ft["T2_us"], ft["T2_err_us"], ft["n"]))
            axc.legend(fontsize=7)
        axc.set_xlabel("T [us]"); axc.set_ylabel("echo contrast C(T)")
        axc.set_title("C(T) decay", fontsize=9); axc.grid(alpha=0.25)
        fig.suptitle("Hahn-echo T2  %s  (phase-scan per T -> C(T); mid-conditioned)" % sid,
                     fontsize=10)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        png = os.path.join(sd, "fit_echo_t2_%s.png" % sid)
        fig.text(0.005, 0.005, png, fontsize=5, color="0.4")
        fig.savefig(png, dpi=130, bbox_inches="tight")
        out["png"] = png
        print("  saved %s" % png)
    except Exception as e:  # noqa: BLE001
        print("  (plot skipped: %s)" % e)

    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
