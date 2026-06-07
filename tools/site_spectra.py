"""site_spectra.py -- inspect individual per-site 556 push-out spectra + Lorentzian fits.

Picks sites spanning the fitted-center range and plots survival-vs-freq + fit, to judge whether
extreme 'trap depths' are real dips or edge-pinned/bad fits. Reports the light-shift spread for
ALL in-range vs INTERIOR centers vs interior&good-R^2 (the trustworthy spread).

Run with the yb_analysis env python:
  <yb_analysis-python> site_spectra.py <scan_id|latest> [--ref 107.735e6]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--host", default=None)
    ap.add_argument("--ref", type=float, default=107.735e6)
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    scan_dir = d.get("scan_dir")

    import numpy as np
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals
    from yb_analysis.analysis import probabilities as prob
    from yb_analysis.analysis.fittings.lorentzian import fit_lorentzian_site_resolved

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
    eps = 0.15e6
    inrange = np.isfinite(centers) & (centers >= lo) & (centers <= hi)
    interior = inrange & (centers > lo + eps) & (centers < hi - eps) & np.isfinite(widths) & (widths > 0) & (widths < (hi - lo))
    goodr = interior & (r2 > 0.3)

    def dn(c):
        return 2.0 * (args.ref - c)

    print("scan %s: %d sites | in-range=%d  interior=%d  edge-pinned=%d  interior&R2>0.3=%d"
          % (sid, len(centers), int(inrange.sum()), int(interior.sum()),
             int((inrange & ~interior).sum()), int(goodr.sum())))
    for label, mask in [("ALL in-range", inrange), ("interior", interior), ("interior&R2>0.3", goodr)]:
        c = centers[mask]
        if len(c):
            D = dn(c)
            print("  %-16s center %.3f-%.3f MHz | light-shift %.2f-%.2f MHz (mean %.2f, std %.2f) n=%d"
                  % (label, c.min() / 1e6, c.max() / 1e6, D.min() / 1e6, D.max() / 1e6,
                     D.mean() / 1e6, D.std() / 1e6, len(c)))

    # pick 6 sites spanning the in-range fitted centers (min, 25, 50, 75, max + one extra)
    cand = np.where(inrange)[0]
    order = cand[np.argsort(centers[cand])]
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    picks = [order[int(q * (len(order) - 1))] for q in qs]
    picks = list(dict.fromkeys(picks))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    axs = axs.ravel()
    for k, s in enumerate(picks[:6]):
        ax = axs[k]
        ax.errorbar(sp / 1e6, p11[s], yerr=p11e[s], fmt="o", ms=3, capsize=2)
        f = fits[s]
        if f:
            ax.plot(f["x_fit"] / 1e6, f["y_fit"], "r-", lw=1.3)
        ax.axvline(centers[s] / 1e6, color="r", ls="--", lw=0.8)
        ax.set_title("site %d  center %.2f MHz  depth %.2f  R2 %.2f"
                     % (s, centers[s] / 1e6, dn(centers[s]) / 1e6,
                        r2[s] if np.isfinite(r2[s]) else -1), fontsize=8)
        ax.set_xlabel("556 push-out freq [MHz]")
        ax.set_ylabel("survival")
        ax.set_ylim(-0.05, 1.05)
    for k in range(len(picks), 6):
        axs[k].axis("off")
    fig.suptitle("%s: per-site push-out spectra spanning fitted centers" % sid, fontsize=10)
    fig.tight_layout()
    out = os.path.join(scan_dir, "site_spectra_%s.png" % sid)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
