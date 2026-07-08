"""plot_survival_map.py -- per-site SURVIVAL map for a scan (auto-scaled colorbar).

Scatter of tweezer sites (x,y) coloured by survival_mean, colorbar auto-scaled to the data
range (vmin/vmax from min/max survival). Sibling of plot_sitemap.py (which maps loading).

    python plot_survival_map.py latest
    python plot_survival_map.py 20260626135907 --title "Per-site survival"
"""

import argparse
import json
import os
import urllib.request

HOSTS = ["127.0.0.1:8050", "100.86.15.43:8050"]


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
    ap = argparse.ArgumentParser(description="Per-site survival map (auto-scaled colorbar).")
    ap.add_argument("scan", help="'latest' or scan_id")
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--clip", type=float, default=2.0,
                    help="percentile clip for colorbar (default 2 -> 2nd..98th pct; 0 = full min/max)")
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    ps = d.get("per_site") or {}
    S = ps.get("survival_mean") or []
    X = ps.get("x") or []
    Y = ps.get("y") or []
    pts = [(s, x, y) for s, x, y in zip(S, X, Y) if s is not None and s == s]
    if not pts:
        raise SystemExit("no per_site survival for %s" % sid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    n = len(ss)
    mu = sum(ss) / n
    sd = (sum((v - mu) ** 2 for v in ss) / n) ** 0.5

    if args.clip > 0:                                 # percentile-clip so outliers don't flatten the scale
        import numpy as np
        vlo, vhi = np.percentile(ss, [args.clip, 100 - args.clip])
    else:
        vlo, vhi = min(ss), max(ss)                   # full data range
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sc = ax.scatter(xs, ys, c=ss, cmap="RdYlGn", vmin=vlo, vmax=vhi, s=16)
    fig.colorbar(sc, ax=ax, label="survival fraction")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("%s  (mean=%.2f, CV=%.0f%%)"
                 % (args.title or ("%s per-site survival" % sid), mu, 100 * sd / mu if mu else 0),
                 fontsize=10)
    out = args.out or os.path.join(d.get("scan_dir") or ".", "survivalmap_%s.png" % sid)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
