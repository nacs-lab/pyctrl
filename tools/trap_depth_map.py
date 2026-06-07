"""trap_depth_map.py -- CLEAN site-resolved TRAP DEPTH (uK) map from a 556 mj=1 push-out spectrum.

Fits a per-site Lorentzian dip, keeps trustworthy fits (interior center, R^2 cut, sane width),
converts the light shift to ground-state trap depth via the GeneralAnalysis recipe
(trap_depth_from_lightshift), and maps it over the array. mj=0 reference defaults to the
expConfig value (Resonance556mj0Freq = 107.735e6). Rejected/artifact fits are greyed out.

Run with the yb_analysis env python:
  <yb_analysis-python> trap_depth_map.py <scan_id|latest> [--ref 107.735e6] [--r2min 0.7]
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


def trap_depth_from_lightshift_K(delta_nu_hz):
    """GeneralAnalysis recipe: excited-state light-shift difference (Hz) -> ground-state
    trap depth (K). 532 nm params, mj1=1, mj0=0, theta=0. (Verified vs notebook: 3.47 MHz->277 uK.)"""
    import numpy as np
    h = 6.62607015e-34
    kB = 1.380649e-23
    mj1, mj0 = 1, 0
    theta = 0.0
    alpha_s, alpha_t, alpha_g = 22.4, -7.6, 37.9   # Hz/(W/cm^2)
    T = (1 - 3 * np.cos(theta) ** 2) / 2
    alpha_e1 = alpha_s - alpha_t * T * (3 * mj1 ** 2 - 2)
    alpha_e0 = alpha_s - alpha_t * T * (3 * mj0 ** 2 - 2)
    I = -4 * np.asarray(delta_nu_hz, float) / (alpha_e1 - alpha_e0)   # W/cm^2
    depth_Hz = 0.25 * abs(alpha_g) * I
    return h * depth_Hz / kB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--host", default=None)
    ap.add_argument("--ref", type=float, default=107.735e6, help="mj=0 reference (expConfig Resonance556mj0Freq)")
    ap.add_argument("--r2min", type=float, default=0.7)
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    scan_dir = d.get("scan_dir")
    ps = d.get("per_site") or {}

    import numpy as np
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals
    from yb_analysis.analysis import probabilities as prob
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian_site_resolved

    X = np.asarray(ps.get("x") or [], float)
    Y = np.asarray(ps.get("y") or [], float)
    sd = load_scan_from_path(scan_dir)
    sp_, l1, l2, reps = unpack_scan_logicals(
        sd["Scan"], logicals=sd.get("logicals"), seq_ids=sd.get("seq_ids"),
        mat_path=sd.get("mat_path"),
        logicals_img1=sd.get("logicals_img1"), logicals_img2=sd.get("logicals_img2"))
    sp = np.asarray(sp_, float).ravel()
    p11, p11e = prob.prob11_site_resolved(l1, l2)
    centers, widths, params, fits = fit_lorentzian_site_resolved(sp, p11, p11e, mode="dip")
    centers = np.asarray(centers, float)
    widths = np.asarray(widths, float)
    r2 = np.array([(f["r_squared"] if f else np.nan) for f in fits])

    lo, hi = sp.min(), sp.max()
    eps = 0.2e6
    clean = (np.isfinite(centers) & (centers > lo + eps) & (centers < hi - eps)
             & np.isfinite(widths) & (widths > 0) & (widths < (hi - lo)) & (r2 > args.r2min))

    delta_nu = 2.0 * (args.ref - centers)                       # Hz light-shift difference
    depth_uK = trap_depth_from_lightshift_K(delta_nu) * 1e6      # trap depth in uK

    nS = len(centers)
    if X.shape[0] != nS:
        raise SystemExit("site count mismatch (logic=%d, api x=%d)" % (nS, X.shape[0]))

    Dc = depth_uK[clean]
    mu, sdv = Dc.mean(), Dc.std()
    vmin, vmax = np.percentile(Dc, 2), np.percentile(Dc, 98)

    def corr(u, v):
        u = u - u.mean(); v = v - v.mean()
        den = ((u * u).sum() * (v * v).sum()) ** 0.5
        return float((u * v).sum() / den) if den else float("nan")
    cx = corr(Dc, X[clean]); cy = corr(Dc, Y[clean])
    print("scan %s: clean=%d/%d  trap depth mean %.0f uK  std %.0f uK (±%.0f%%)  range(p2-p98) %.0f-%.0f uK"
          % (sid, int(clean.sum()), nS, mu, sdv, 100 * sdv / mu, vmin, vmax))
    print("  (light shift mean %.2f MHz)  corr(depth,x)=%+.2f corr(depth,y)=%+.2f"
          % (2 * (args.ref - centers[clean]).mean() / 1e6, cx, cy))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    bad = ~clean & np.isfinite(X) & np.isfinite(Y)
    ax.scatter(X[bad], Y[bad], c="0.75", s=10, marker="x", label="rejected fit")
    sc = ax.scatter(X[clean], Y[clean], c=Dc, cmap="coolwarm", s=18, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="trap depth [uK]  (556 mj=1 light shift)")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("%s: site-resolved trap depth (clean fits R2>%.1f, n=%d)\nmean %.0f uK, std %.0f (±%.0f%%), corr(x)=%+.2f"
                 % (sid, args.r2min, int(clean.sum()), mu, sdv, 100 * sdv / mu, cx), fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    out = os.path.join(scan_dir, "trap_depth_uK_map_%s.png" % sid)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
