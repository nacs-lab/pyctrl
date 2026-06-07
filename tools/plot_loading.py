"""plot_loading.py -- render a loading-rate plot (PNG) for a scan from the dashboard analysis API.

1-D sweep  -> errorbar line (loading vs swept param).
2-D grid   -> heatmap (loading over the two axes), best cell starred + per-cell value labels.
Saves to <scan_dir>/loading_<scan_id>.png (the run's data folder) and prints the path, so it can
be piped straight into tools/notion_upload.py.

    python plot_loading.py latest
    python plot_loading.py 20260605212232 --title "Phase 0a -- LoadingTime curve"
    python plot_loading.py latest --out C:/tmp/fig.png
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
            with urllib.request.urlopen("http://%s%s" % (h, path), timeout=30) as r:
                return json.load(r), h
        except Exception as e:                       # noqa: BLE001
            last = e
    raise SystemExit("dashboard unreachable on %s: %s" % (hosts, last))


def main():
    ap = argparse.ArgumentParser(description="Plot loading-rate curve/heatmap for a scan.")
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

    sw = d.get("sweep") or {}
    sm = d.get("summary") or {}
    cols = sw.get("cols") or []
    dims = sw.get("dims") or []
    vals = sw.get("values") or []
    lr = sm.get("loading_rate") or []
    sem = sm.get("loading_sem_pershot") or [0.0] * len(lr)
    name = d.get("scan_name") or d.get("name")
    title = args.title or "%s  %s  (n_shots=%s)" % (name, sid, d.get("n_shots"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if len(dims) == 1:
        x = vals[0]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.errorbar(x, lr, yerr=sem, marker="o", capsize=3, lw=1.5)
        ax.set_xlabel(cols[0])
        ax.set_ylabel("array loading fraction")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        bp = int(np.nanargmax(lr))
        ax.annotate("best %.3f @ %.4g" % (lr[bp], x[bp]), (x[bp], lr[bp]),
                    textcoords="offset points", xytext=(0, 8), color="red", fontsize=8)
    elif len(dims) == 2:
        s0, s1 = int(dims[0]), int(dims[1])
        Z = np.full((s1, s0), np.nan)
        for p in range(len(lr)):
            Z[(p // s0) % s1, p % s0] = lr[p]
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="array loading fraction")
        ax.set_xticks(range(s0))
        ax.set_xticklabels(["%.4g" % v for v in vals[0]], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(s1))
        ax.set_yticklabels(["%.4g" % v for v in vals[1]], fontsize=8)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title(title, fontsize=10)
        for i1 in range(s1):
            for i0 in range(s0):
                if np.isfinite(Z[i1, i0]):
                    ax.text(i0, i1, "%.2f" % Z[i1, i0], ha="center", va="center",
                            color="w", fontsize=7)
        bp = int(np.nanargmax(lr))
        ax.plot(bp % s0, (bp // s0) % s1, "r*", ms=18, mec="k")
    else:
        raise SystemExit("no 1-D/2-D sweep to plot (dims=%s)" % dims)

    out = args.out or os.path.join(d.get("scan_dir") or ".", "loading_%s.png" % sid)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
