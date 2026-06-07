"""plot_sitemap.py -- per-site loading MAP (uniformity check) for a scan.

Scatter of tweezer sites (x,y) coloured by loading_rate, with mean/CV + gradient in the title.
Use to watch the spatial uniformity of loading (it is non-uniform -- a diagonal gradient).

    python plot_sitemap.py latest
    python plot_sitemap.py 20260605220905 --title "Phase 3 per-site loading"
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
    ap = argparse.ArgumentParser(description="Per-site loading map (uniformity).")
    ap.add_argument("scan", help="'latest' or scan_id")
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    ps = d.get("per_site") or {}
    L = ps.get("loading_rate") or []
    X = ps.get("x") or []
    Y = ps.get("y") or []
    pts = [(l, x, y) for l, x, y in zip(L, X, Y) if l is not None and l == l]
    if not pts:
        raise SystemExit("no per_site loading for %s" % sid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ls = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    n = len(ls)
    mu = sum(ls) / n
    sd = (sum((v - mu) ** 2 for v in ls) / n) ** 0.5

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sc = ax.scatter(xs, ys, c=ls, cmap="RdYlGn", vmin=0.0, vmax=max(0.7, max(ls)), s=16)
    fig.colorbar(sc, ax=ax, label="loading fraction")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("%s  (mean=%.2f, CV=%.0f%%)"
                 % (args.title or ("%s per-site loading" % sid), mu, 100 * sd / mu if mu else 0),
                 fontsize=10)
    out = args.out or os.path.join(d.get("scan_dir") or ".", "sitemap_%s.png" % sid)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
