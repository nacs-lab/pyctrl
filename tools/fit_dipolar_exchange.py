"""fit_dipolar_exchange.py -- pairwise dipolar spin-exchange analysis with SPAM/loss correction.

The 2026-07-24 pipeline (Notion "07/24 Hahn-echo, Dipolar spin-exchange", jobs 135/136):

  1. PAIRS: rearranged pair pattern (e.g. eight_pairs / dimer_20um). Target sites from the
     slm_diag target_paired union; pairs = mutual nearest neighbours in grid coords.
  2. Per (pair, T), conditioned on BOTH atoms loaded at mid: classify the final frame into
     dd / d0 / 0d / 00 (detected = ended in S; P and lifetime loss are undetected).
  3. SPAM/LOSS calibration eta(T) = A0*exp(-T/T1_eff): an S atom is detected with prob eta(T);
     the coherent exchange conserves S+P with <S> ~ 1/2, so the per-atom mean detection's
     slow envelope is eta(T)/2. Fit dbar(T) = 0.5*A0*exp(-T/T1) -> eta.
  4. Invert the 2-atom detection statistics per T (both atoms share eta):
        SS = dd/eta^2
        SP = (d0 - SS*eta*(1-eta))/eta ;  PS = (0d - ...)/eta ;  PP = 1 - SS - SP - PS
  5. Errors: multinomial Cov(f) = (diag(f)-f f^T)/N propagated through the inversion Jacobian
     (numerical), plus the eta-fit uncertainty term.
  6. Weighted damped-cosine fit to corrected SS(T):  SS = c + a*cos(2*pi*J*T + phi)*exp(-T/tau)
     -> J-tilde. SS and PP must oscillate ANTI-PHASE about a SHARED center (the loss-calib check);
     SP/PS stay flat (leakage).
  Plots: ensemble 4-state + fit; per-pair SS/PP panels with per-pair J.

Run with the yb_analysis env python:
  <yb_analysis-python> fit_dipolar_exchange.py <scan_dir>
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
    ap.add_argument("--j-init-mhz", type=float, default=0.48,
                    help="initial J guess in MHz (default 0.48 = the 20 um 07-24 value)")
    ap.add_argument("--exclude-pairs", default="",
                    help="comma-separated pair labels/indices to DROP before the ensemble and per-pair fits, e.g. 'P6' or '6'. Labels are the P<n> numbering printed above (assigned before exclusion, so they stay stable). Outputs get a _noP<n> suffix so the full-set figures are not overwritten.")
    args = ap.parse_args()

    import h5py
    import numpy as np
    from scipy.optimize import curve_fit

    sd = args.scan_dir.rstrip("\\/")
    sid = os.path.basename(sd).replace("data_", "")
    cfg = json.load(open(os.path.join(sd, "data_%s.json" % sid)))
    P = np.asarray(cfg["Params"], int)
    gx = np.asarray(cfg["initGridLocationsX"], float).ravel()
    gy = np.asarray(cfg["initGridLocationsY"], float).ravel()

    with h5py.File(os.path.join(sd, "data_%s.h5" % sid), "r", swmr=True) as f:
        n = min(f["logicals_mid"].shape[0], f["logicals_img2"].shape[0], f["seq_ids"].shape[0])
        seq = np.asarray(f["seq_ids"][:n], np.int64)
        mid = f["logicals_mid"][:n].astype(bool)
        fin = f["logicals_img2"][:n].astype(bool)
    nS = mid.shape[1]

    # ---- target sites + mutual-NN pairing -------------------------------------------------
    diag = os.path.join(sd, "slm_diag.h5")
    tm = np.zeros(nS, bool)
    if os.path.exists(diag):
        try:
            with h5py.File(diag, "r", swmr=True) as f:
                g = f["/diag"]
                dmap = {int(s): np.asarray(g["target_paired"][i]).ravel()
                        for i, s in enumerate(np.asarray(g["seq_id"][:], np.int64))}
            for s in seq:
                t = dmap.get(int(s))
                if t is not None:
                    t = t[(t >= 0) & (t < nS)]
                    tm[t] = True
        except OSError as ex:  # e.g. OneDrive cloud-only placeholder
            print("slm_diag unreadable (%s) -> occupancy fallback" % ex)
    if not tm.any():
        tm = mid.mean(0) >= 0.25
        print("no slm_diag -> occupancy fallback for target sites")
    tgt = np.where(tm)[0]
    pts = np.column_stack([gx[tgt], gy[tgt]])
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    pairs = []
    used = set()
    for i in range(tgt.size):
        j = nn[i]
        if nn[j] == i and i < j and i not in used and j not in used:
            pairs.append((tgt[i], tgt[j]))
            used.update((i, j))
    dists = [D[np.where(tgt == a)[0][0], np.where(tgt == b)[0][0]] for a, b in pairs]
    med = float(np.median(dists))
    keep = [abs(d - med) <= 0.2 * med for d in dists]
    if not all(keep):
        print("dropping %d pair(s) off the median intra-pair distance (%.0f px):"
              % (keep.count(False), med))
        for (a, b), d, k in zip(pairs, dists, keep):
            if not k:
                print("  dropped %d-%d (%.0f px)" % (a, b, d))
        pairs = [p for p, k in zip(pairs, keep) if k]
        dists = [d for d, k in zip(dists, keep) if k]
    print("pairs: %d (of %d target sites); intra-pair dist %.1f +- %.1f px"
          % (len(pairs), tgt.size, np.mean(dists), np.std(dists)))
    for k, (a, b) in enumerate(pairs):
        print("  P%d: %d-%d" % (k + 1, a, b))

    # ---- optional pair exclusion (--exclude-pairs) ------------------------------------
    labels = ["P%d" % (k + 1) for k in range(len(pairs))]
    tag = ""
    if args.exclude_pairs.strip():
        want = set()
        for tok in args.exclude_pairs.split(","):
            tok = tok.strip().upper().lstrip("P")
            if tok:
                want.add("P" + tok)
        drop = [i for i, L in enumerate(labels) if L in want]
        if not drop:
            raise SystemExit("--exclude-pairs %r matched none of %s" % (args.exclude_pairs, labels))
        for i in drop:
            print("  EXCLUDING %s (%d-%d) from ensemble + per-pair fits"
                  % (labels[i], pairs[i][0], pairs[i][1]))
        keep_i = [i for i in range(len(pairs)) if i not in drop]
        pairs = [pairs[i] for i in keep_i]
        dists = [dists[i] for i in keep_i]
        labels = [labels[i] for i in keep_i]
        tag = "_no" + "".join(sorted(want, key=lambda x: int(x[1:])))
        print("  -> %d pairs remain: %s" % (len(pairs), ", ".join(labels)))

    # ---- T axis (payload sweep; generate if absent) ---------------------------------------
    pj = os.path.join(sd, "analysis_payload.json")
    pl = None
    if os.path.exists(pj):
        try:
            pl = json.load(open(pj))["payload"]
        except OSError:
            pl = None   # cloud-only placeholder etc. -> regenerate
    if pl is None:
        from yb_analysis.analysis.run_analysis import analyze_scan
        analyze_scan(sid.replace("_", ""), include_per_site=False, include_diag_aggregate=False,
                     include_per_iteration=False, sync_slm_diag=False, force_recache=True)
        pl = json.load(open(pj))["payload"]
    sw = pl["sweep"]
    icol = [i for i, c in enumerate(sw["cols"]) if "wait" in str(c).lower()]
    icol = icol[0] if icol else 0
    T_us = np.asarray(sw["values"][icol], float) * 1e6
    n_T = T_us.size

    li = seq - 1
    ok = (li >= 0) & (li < P.size)
    p_of = P[li[ok]] - 1
    midk, fink = mid[ok], fin[ok]

    # ---- per-T pair-outcome counts (pooled over pairs) + per-pair --------------------------
    def counts(pair_list):
        cnt = np.zeros((n_T, 4))                 # dd, d0, 0d, 00
        for a, b in pair_list:
            both = midk[:, a] & midk[:, b]
            for t in range(n_T):
                sh = both & (p_of == t)
                if not sh.any():
                    continue
                fa, fb = fink[sh, a], fink[sh, b]
                cnt[t, 0] += (fa & fb).sum()
                cnt[t, 1] += (fa & ~fb).sum()
                cnt[t, 2] += (~fa & fb).sum()
                cnt[t, 3] += (~fa & ~fb).sum()
        return cnt

    # ---- eta(T): per-atom mean detection envelope = eta/2 ---------------------------------
    def eta_fit(cnt):
        N = cnt.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dbar = (2 * cnt[:, 0] + cnt[:, 1] + cnt[:, 2]) / (2 * np.maximum(N, 1))
        mfin = N > 0
        def env(T, A0, T1):
            return 0.5 * A0 * np.exp(-T / np.maximum(T1, 1e-3))
        popt, pcov = curve_fit(env, T_us[mfin], dbar[mfin], p0=[0.9, 80.0],
                               bounds=([0.2, 1.0], [1.2, 5000.0]), maxfev=20000)
        return popt, pcov, dbar

    def invert(cnt, A0, T1):
        N = np.maximum(cnt.sum(1), 1)
        f = cnt / N[:, None]
        eta = A0 * np.exp(-T_us / T1)
        SS = f[:, 0] / eta**2
        SP = (f[:, 1] - SS * eta * (1 - eta)) / eta
        PS = (f[:, 2] - SS * eta * (1 - eta)) / eta
        PP = 1 - SS - SP - PS
        return np.stack([SS, SP, PS, PP], 1), f, N, eta

    def errors(cnt, popt, pcov):
        """Multinomial + eta-fit errors on corrected populations (numerical Jacobians)."""
        N = np.maximum(cnt.sum(1), 1)
        f = cnt / N[:, None]
        A0, T1 = popt
        s0, _f, _N, eta = invert(cnt, A0, T1)
        err = np.zeros_like(s0)
        for t in range(n_T):
            ft = f[t]
            covf = (np.diag(ft) - np.outer(ft, ft)) / N[t]
            # Jacobian ds/df at fixed eta (numeric)
            J = np.zeros((4, 4))
            for k in range(4):
                d = np.zeros(4); d[k] = 1e-5
                c1 = (ft + d) * N[t]
                J[:, k] = (invert_row(c1, eta[t]) - invert_row(ft * N[t], eta[t])) / 1e-5
            var_f = J @ covf @ J.T
            # eta term
            dA, dT1 = 1e-4, 1e-2
            sA = invert_row(ft * N[t], (A0 + dA) * np.exp(-T_us[t] / T1))
            sT = invert_row(ft * N[t], A0 * np.exp(-T_us[t] / (T1 + dT1)))
            s_base = invert_row(ft * N[t], eta[t])
            gA = (sA - s_base) / dA
            gT = (sT - s_base) / dT1
            var_eta = (np.outer(gA, gA) * pcov[0, 0] + np.outer(gT, gT) * pcov[1, 1]
                       + (np.outer(gA, gT) + np.outer(gT, gA)) * pcov[0, 1])
            err[t] = np.sqrt(np.maximum(np.diag(var_f + var_eta), 0))
        return err

    def invert_row(c, eta):
        import numpy as np
        Nn = max(c.sum(), 1)
        f = c / Nn
        SS = f[0] / eta**2
        SP = (f[1] - SS * eta * (1 - eta)) / eta
        PS = (f[2] - SS * eta * (1 - eta)) / eta
        return np.array([SS, SP, PS, 1 - SS - SP - PS])

    def fit_ss(Tv, SS, Werr, j0):
        # Damped cosine + SLOW TRANSIENT term. The transient (initial-state settling, e.g. the
        # T->0 SS excess) otherwise steals the fit: a free J slides to the slow envelope
        # (2026-08-19 job 1278), and the FFT's low-frequency weight is that transient + the
        # damping-broadened shoulder of the ONE exchange peak (07/24 FFT note) -- NOT a tone
        # to fit. J grid restricted to the physical exchange band around j0.
        def mod(T, c, b, taus, a, J, phi, tau):
            return (c + b * np.exp(-T / np.maximum(taus, 0.05))
                    + a * np.cos(2 * np.pi * J * T + phi) * np.exp(-T / np.maximum(tau, 0.05)))
        W = np.maximum(Werr, 1e-3)
        # Model-select: plain damped cosine (b pinned ~0) vs cosine + transient. On clean data
        # the oscillation STARTS at its extremum and a free transient steals the whole signal
        # (2026-08-20 job 1283); on transient-y data the plain cosine slides to the envelope.
        best = None
        for jg in np.arange(0.6 * j0, 1.6 * j0, 0.004):
            for bmax in (1e-6, 0.8):
                try:
                    modJ = lambda T, c, b, taus, a, phi, tau, _J=jg: mod(T, c, b, taus, a, _J, phi, tau)
                    p6, _ = curve_fit(modJ, Tv, SS,
                                      p0=[SS.mean(), min(0.2, bmax / 2), 1.0, 0.2, 0.0, 6.0],
                                      sigma=W, absolute_sigma=True,
                                      bounds=([0, -bmax, 0.1, 0, -np.pi, 0.1],
                                              [1, bmax, 20.0, 1, np.pi, 500]), maxfev=6000)
                    chi2 = float((((SS - modJ(Tv, *p6)) / W) ** 2).sum())
                    # transient variant must be MEANINGFULLY better to win (2 extra dof)
                    penalty = 0.0 if bmax < 1e-3 else 4.0
                    if best is None or chi2 + penalty < best[0]:
                        best = (chi2 + penalty, jg, p6)
                except Exception:
                    continue
        if best is None:
            return None
        _c2, jbest, p6 = best
        p0 = [p6[0], p6[1], p6[2], p6[3], jbest, p6[4], p6[5]]
        p, cvar = curve_fit(mod, Tv, SS, p0=p0, sigma=W, absolute_sigma=True,
                            bounds=([0, -0.8, 0.1, 0, 0.95 * jbest, -np.pi, 0.1],
                                    [1, 0.8, 20.0, 1, 1.05 * jbest, np.pi, 500]), maxfev=20000)
        pred = mod(Tv, *p)
        sst = float(((SS - SS.mean()) ** 2).sum())
        r2 = 1 - float(((SS - pred) ** 2).sum()) / sst if sst > 0 else np.nan
        return dict(c=p[0], a=p[3], J_MHz=p[4], J_err_MHz=float(np.sqrt(cvar[4, 4])),
                    phi=p[5], tau_us=p[6], tau_err_us=float(np.sqrt(cvar[6, 6])), r2=r2,
                    model=mod, popt=p)

    # ======== ensemble ========
    cnt = counts(pairs)
    popt, pcov, dbar = eta_fit(cnt)
    A0, T1 = popt
    print("eta(T) = %.3f +- %.3f * exp(-T / %.1f +- %.1f us)"
          % (A0, np.sqrt(pcov[0, 0]), T1, np.sqrt(pcov[1, 1])))
    s, f, N, eta = invert(cnt, A0, T1)
    serr = errors(cnt, popt, pcov)
    SS, SP, PS, PP = s.T
    print("corrected centers: SS %.3f | PP %.3f (loss-calib check: should match)"
          % (np.nanmean(SS), np.nanmean(PP)))
    ft = fit_ss(T_us, SS, serr[:, 0], args.j_init_mhz)
    if ft:
        print("ENSEMBLE: J = %.1f +- %.1f kHz (period %.3f us), tau = %.2f +- %.2f us, R2 = %.3f"
              % (ft["J_MHz"] * 1e3, ft["J_err_MHz"] * 1e3, 1 / ft["J_MHz"], ft["tau_us"],
                 ft["tau_err_us"], ft["r2"]))

    # ======== per-pair ========
    pair_fits = []
    for k, pr in enumerate(pairs):
        ck = counts([pr])
        sk, _fk, Nk, _ek = invert(ck, A0, T1)          # global eta
        ek = errors(ck, popt, pcov)
        fk = fit_ss(T_us, sk[:, 0], ek[:, 0], args.j_init_mhz)
        pair_fits.append((pr, sk, ek, fk))
        if fk:
            print("  %s (%d-%d): J = %.1f +- %.1f kHz, tau = %.1f us, R2 = %.2f"
                  % (labels[k], pr[0], pr[1], fk["J_MHz"] * 1e3, fk["J_err_MHz"] * 1e3,
                     fk["tau_us"], fk["r2"]))
    Js = [fk["J_MHz"] * 1e3 for _p, _s, _e, fk in pair_fits if fk]
    if Js:
        print("per-pair mean J = %.1f +- %.1f kHz (n=%d)" % (np.mean(Js), np.std(Js), len(Js)))

    # ======== plots ========
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(T_us, SS, yerr=serr[:, 0], fmt="o-", ms=4, lw=0.7, color="C0", capsize=2, label="SS")
    ax.errorbar(T_us, PP, yerr=serr[:, 3], fmt="o-", ms=4, lw=0.7, color="C3", capsize=2, label="PP")
    ax.errorbar(T_us, SP, yerr=serr[:, 1], fmt="s-", ms=3, lw=0.5, color="0.6", capsize=2, label="SP")
    ax.errorbar(T_us, PS, yerr=serr[:, 2], fmt="^-", ms=3, lw=0.5, color="0.75", capsize=2, label="PS")
    if ft:
        tfine = np.linspace(T_us.min(), T_us.max(), 1500)
        ax.plot(tfine, ft["model"](tfine, *ft["popt"]), "-", color="k", lw=1.2, alpha=0.7,
                label="SS fit: J=%.0f±%.0f kHz τ=%.1f µs R²=%.2f"
                      % (ft["J_MHz"] * 1e3, ft["J_err_MHz"] * 1e3, ft["tau_us"], ft["r2"]))
    ax.set_xlabel("free-evolution T [us]"); ax.set_ylabel("corrected pair population")
    ax.set_title("Dipolar spin exchange, loss-calibrated 4-state  %s  (%d pairs, eta=%.3f·exp(-T/%.0fus); "
                 "SS/PP centers %.2f/%.2f)" % (sid, len(pairs), A0, T1, np.nanmean(SS), np.nanmean(PP)),
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    png1 = os.path.join(sd, "fit_dipolar_4state_%s%s.png" % (sid, tag))
    fig.text(0.005, 0.005, png1, fontsize=5, color="0.4")
    fig.tight_layout(); fig.savefig(png1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", png1)

    # Per-pair panels AT ARRAY POSITIONS (pair centers binned to grid rows/cols; cells with no
    # pair stay empty) -- feedback-persite-grid-figure-style.
    npair = len(pairs)
    pcx = np.array([(gx[a] + gx[b]) / 2 for a, b in pairs])
    pcy = np.array([(gy[a] + gy[b]) / 2 for a, b in pairs])

    def _bins(coords):
        u = np.sort(np.unique(coords))
        d = np.diff(u)
        tol = 0.4 * (np.median(d[d > 20]) if (d > 20).any() else (u.max() - u.min() or 1.0))
        centers = [u[0]]
        for c in u[1:]:
            if c - centers[-1] > tol:
                centers.append(c)
        centers = np.asarray(centers)
        return np.array([int(np.argmin(np.abs(centers - c))) for c in coords]), centers.size

    col_of, ncol = _bins(pcx)
    row_of, nrow = _bins(pcy)
    fig2, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 2.8 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    usedq = np.zeros((nrow, ncol), bool)
    for k, (pr, sk, ek, fk) in enumerate(pair_fits):
        axp = axs[row_of[k], col_of[k]]
        usedq[row_of[k], col_of[k]] = True
        axp.errorbar(T_us, sk[:, 0], yerr=ek[:, 0], fmt="o", ms=3, color="C0", capsize=0, label="SS")
        axp.errorbar(T_us, sk[:, 3], yerr=ek[:, 3], fmt="o", ms=3, color="C3", capsize=0, label="PP")
        if fk:
            tfine = np.linspace(T_us.min(), T_us.max(), 1200)
            axp.plot(tfine, fk["model"](tfine, *fk["popt"]), "-", color="k", lw=1.0, alpha=0.7)
            axp.set_title("P%d (%d-%d): J=%.0f±%.0f kHz τ=%.1f R²=%.2f"
                          % (k + 1, pr[0], pr[1], fk["J_MHz"] * 1e3, fk["J_err_MHz"] * 1e3,
                             fk["tau_us"], fk["r2"]), fontsize=8)
        else:
            axp.set_title("P%d (%d-%d): no fit" % (k + 1, pr[0], pr[1]), fontsize=8)
        axp.grid(alpha=0.25); axp.tick_params(labelsize=7)
    for r in range(nrow):
        for c in range(ncol):
            if not usedq[r, c]:
                axs[r, c].axis("off")
    axs[0, 0].legend(fontsize=7)
    fig2.suptitle("Per-pair SS/PP AT ARRAY POSITIONS, loss-calibrated (global eta)  %s%s" %
                  (sid, ("  mean J %.1f±%.1f kHz" % (np.mean(Js), np.std(Js))) if Js else ""),
                  fontsize=11)
    fig2.supxlabel("T [us]", fontsize=9); fig2.supylabel("corrected population", fontsize=9)
    fig2.tight_layout(rect=[0.01, 0.02, 1, 0.95])
    png2 = os.path.join(sd, "fit_dipolar_perpair_%s%s.png" % (sid, tag))
    fig2.text(0.005, 0.005, png2, fontsize=5, color="0.4")
    fig2.savefig(png2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print("saved", png2)

    out = {"scan_id": sid, "n_pairs": len(pairs), "pairs": [list(map(int, p)) for p in pairs],
           "eta_A0": float(A0), "eta_A0_err": float(np.sqrt(pcov[0, 0])),
           "eta_T1_us": float(T1), "eta_T1_err_us": float(np.sqrt(pcov[1, 1])),
           "SS_center": float(np.nanmean(SS)), "PP_center": float(np.nanmean(PP)),
           "ensemble": (None if not ft else {"J_kHz": ft["J_MHz"] * 1e3,
                                             "J_err_kHz": ft["J_err_MHz"] * 1e3,
                                             "tau_us": ft["tau_us"], "tau_err_us": ft["tau_err_us"],
                                             "r2": ft["r2"]}),
           "per_pair_J_kHz": Js, "png": [png1, png2]}
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
