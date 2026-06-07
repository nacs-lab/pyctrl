"""trap_depth_sr.py -- site-resolved trap depth from a 556 push-out spectrum, vs per-site loading.

Verifies whether the per-site loading spread tracks per-site trap depth. From a 556 mj=1
push-out survival spectrum (PushoutSurvivalSeq), per site:
  * loading  = img1 occupancy (loading_rate_site_resolved)
  * survival vs push-out freq (prob11_site_resolved) -> a Lorentzian DIP at the push-out
    resonance; the fitted center is AC-Stark-shifted by the local tweezer depth.
delta_nu = 2*(mj0_ref - center) is the light shift (~ trap depth). Correlate loading vs depth.

Run with the yb_analysis env python:
  <yb_analysis-python> trap_depth_sr.py <scan_id|latest> [--ref 107.735e6]
"""

import argparse
import json
import os
import sys
import urllib.request

HOSTS = ["127.0.0.1:8050", "100.86.15.43:8050"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _get(path, hosts):
    last = None
    for h in hosts:
        try:
            with urllib.request.urlopen("http://%s%s" % (h, path), timeout=40) as r:
                return json.load(r), h
        except Exception as e:                       # noqa: BLE001
            last = e
    raise SystemExit("dashboard unreachable on %s: %s" % (hosts, last))


def main():
    ap = argparse.ArgumentParser(description="Site-resolved trap depth from a 556 spectrum.")
    ap.add_argument("scan", help="'latest' or scan_id")
    ap.add_argument("--host", default=None)
    ap.add_argument("--ref", type=float, default=107.735e6, help="mj=0 reference freq (Hz)")
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    scan_dir = d.get("scan_dir")
    ps = d.get("per_site") or {}
    X = ps.get("x") or []
    Y = ps.get("y") or []

    import numpy as np
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals
    from yb_analysis.analysis import probabilities as prob
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian_site_resolved

    sd = load_scan_from_path(scan_dir)
    scan_params, logic1, logic2, reps = unpack_scan_logicals(
        sd["Scan"], logicals=sd.get("logicals"), seq_ids=sd.get("seq_ids"),
        mat_path=sd.get("mat_path"),
        logicals_img1=sd.get("logicals_img1"), logicals_img2=sd.get("logicals_img2"))
    sp = np.asarray(scan_params, float).ravel()
    if logic2 is None:
        raise SystemExit("no img2 (need NumImages=2 survival spectrum)")

    load_sr, _ = prob.loading_rate_site_resolved(logic1, reps)       # (nSites, nParams)
    load = np.nanmean(load_sr, axis=1)                               # per-site loading
    p11_sr, p11_sem_sr = prob.prob11_site_resolved(logic1, logic2)   # per-site survival vs freq
    centers, widths, params, fits = fit_lorentzian_site_resolved(sp, p11_sr, p11_sem_sr, mode="dip")
    delta_nu = 2.0 * (args.ref - centers)                            # light shift ~ depth

    nS = len(centers)
    span = sp.max() - sp.min()
    widths = np.asarray(widths, float)
    # reject diverged fits: center must lie within the scan range, width sane
    in_range = (centers >= sp.min()) & (centers <= sp.max())
    sane_w = np.isfinite(widths) & (widths > 0) & (widths < span)
    ok = (np.isfinite(centers) & np.isfinite(load) & np.isfinite(delta_nu)
          & in_range & sane_w)
    nfit = int(ok.sum())
    print("  (rejected %d diverged/out-of-range fits)" % int((np.isfinite(centers) & ~ok).sum()))
    print("scan %s  nsites=%d  good_fits=%d  freq %.2f-%.2f MHz (%d pts)"
          % (sid, nS, nfit, sp.min() / 1e6, sp.max() / 1e6, len(sp)))
    if nfit < 10:
        print("too few good fits to correlate"); return

    L = load[ok]; C = centers[ok]; D = delta_nu[ok]

    def corr(u, v):
        u = u - u.mean(); v = v - v.mean()
        den = (((u * u).sum()) * ((v * v).sum())) ** 0.5
        return float((u * v).sum() / den) if den else float("nan")

    print("center: %.3f-%.3f MHz (mean %.3f)" % (C.min() / 1e6, C.max() / 1e6, C.mean() / 1e6))
    print("light-shift (depth proxy) delta_nu: %.2f-%.2f MHz (mean %.2f)"
          % (D.min() / 1e6, D.max() / 1e6, D.mean() / 1e6))
    print("CORR(loading, light-shift/depth) = %+.3f" % corr(L, D))
    print("CORR(loading, center)            = %+.3f" % corr(L, C))
    order = np.argsort(D); n = len(D)
    for name, sl in [("shallow", slice(0, n // 3)), ("mid", slice(n // 3, 2 * n // 3)),
                     ("deep", slice(2 * n // 3, n))]:
        idx = order[sl]
        print("  %-8s depth %.2f-%.2f MHz : mean loading %.3f"
              % (name, D[idx].min() / 1e6, D[idx].max() / 1e6, L[idx].mean()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (1) scatter loading vs depth
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(D / 1e6, L, s=10, alpha=0.5)
    ax.set_xlabel("light shift 2*(ref-center) [MHz]  (~ trap depth)")
    ax.set_ylabel("per-site loading")
    ax.set_title("%s: loading vs trap depth  (r=%+.2f)" % (sid, corr(L, D)), fontsize=10)
    ax.grid(alpha=0.3)
    out1 = os.path.join(scan_dir, "depth_vs_loading_%s.png" % sid)
    fig.savefig(out1, dpi=120, bbox_inches="tight"); plt.close(fig)
    # (2) per-site depth map
    Xa = np.asarray(X[:nS], float); Ya = np.asarray(Y[:nS], float)
    m = ok
    if Xa.shape[0] == nS and m.sum() > 10:
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        scc = ax.scatter(Xa[m], Ya[m], c=delta_nu[m] / 1e6, cmap="viridis", s=16)
        fig.colorbar(scc, ax=ax, label="light shift [MHz] (~depth)")
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_title("%s: site-resolved trap depth (light shift)" % sid, fontsize=10)
        out2 = os.path.join(scan_dir, "depthmap_%s.png" % sid)
        fig.savefig(out2, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(out2)
    print(out1)


if __name__ == "__main__":
    main()
