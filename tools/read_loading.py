"""read_loading.py -- read per-sweep-point loading fraction from the Dash analysis API.

Pulls ``/api/runs/<id>/analysis`` from the running yb_analysis dashboard and prints the
loading-rate-vs-swept-parameter curve (``summary.loading_rate`` grouped/unscrambled per point,
with ``sweep.values`` for the axes). For a single-point run (no sweep) it prints the
time-ordered per-shot ``loaded_frac`` with the first K (warmup) shots dropped -> mean/std.

Stdlib only (urllib + json) -- runs under ANY python. The dashboard binds to the tailnet IP on
this machine, so we try a list of hosts (loopback first, then the tailnet IP).

    python read_loading.py latest                 # most-recent run
    python read_loading.py 20260605201322         # explicit scan_id
    python read_loading.py latest --drop 5        # warmup shots to drop (single-point path)
    python read_loading.py latest --host 100.86.15.43:8050
"""

import argparse
import json
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
    raise SystemExit("could not reach dashboard on %s: %s" % (hosts, last))


def _decode_axes(p, dims):
    """Flat scan-point index -> per-axis indices (MATLAB column-major, axis0 fastest)."""
    idx = []
    rem = p
    for s in dims:
        s = int(s) or 1
        idx.append(rem % s)
        rem //= s
    return idx


def main():
    ap = argparse.ArgumentParser(description="Read loading-rate curve from the dashboard analysis API.")
    ap.add_argument("scan", help="'latest' or an explicit scan_id (e.g. 20260605201322)")
    ap.add_argument("--host", default=None, help="host:port (default: try loopback then tailnet)")
    ap.add_argument("--drop", type=int, default=5, help="warmup shots to drop for single-point runs")
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]

    d, host = _get("/api/runs/%s/analysis" % sid, hosts)
    name = d.get("scan_name") or d.get("name")
    print("scan_id=%s  name=%s  n_shots=%s  (via %s)" % (sid, name, d.get("n_shots"), host))

    sw = d.get("sweep") or {}
    sm = d.get("summary") or {}
    cols = sw.get("cols") or []
    dims = sw.get("dims") or []
    vals = sw.get("values") or []
    lr = sm.get("loading_rate")
    sem = sm.get("loading_sem_pershot") or []
    ns = sm.get("loading_n_shots") or []

    if isinstance(lr, list) and cols and dims:
        print("sweep: %s  dims=%s" % (" x ".join(cols), dims))
        rows = []
        for p in range(len(lr)):
            ai = _decode_axes(p, dims)
            axv = [vals[a][ai[a]] for a in range(len(dims))]
            rate = lr[p] if lr[p] is not None else float("nan")
            e = sem[p] if (p < len(sem) and sem[p] is not None) else float("nan")
            nv = ns[p] if (p < len(ns) and ns[p] is not None) else 0
            rows.append((axv, rate, e, nv))
        # in-axis-order table
        print("\n-- curve (axis order) --")
        for axv, rate, e, n in rows:
            axs = ", ".join("%s=%.6g" % (cols[a], axv[a]) for a in range(len(axv)))
            print("  %-48s load=%.4f +/- %.4f  (n=%d)" % (axs, rate, e, n))
        # ranked
        best = sorted(rows, key=lambda r: r[1], reverse=True)
        print("\n-- top 5 by loading --")
        for axv, rate, e, n in best[:5]:
            axs = ", ".join("%s=%.6g" % (cols[a], axv[a]) for a in range(len(axv)))
            print("  %-48s load=%.4f +/- %.4f  (n=%d)" % (axs, rate, e, n))
        bv = best[0]
        print("\nBEST: %s  load=%.4f +/- %.4f" % (
            ", ".join("%s=%.6g" % (cols[a], bv[0][a]) for a in range(len(bv[0]))), bv[1], bv[2]))
    else:
        pi = d.get("per_iteration") or {}
        lf = pi.get("loaded_frac") or []
        kept = lf[args.drop:] if len(lf) > args.drop else lf
        if kept:
            m = sum(kept) / len(kept)
            var = sum((x - m) ** 2 for x in kept) / max(1, len(kept) - 1)
            sd = var ** 0.5
            print("single-point: %d shots (dropped first %d warmup) -> load=%.4f +/- %.4f (std), per-shot SEM=%.4f"
                  % (len(kept), args.drop, m, sd, sd / max(1, len(kept)) ** 0.5))
            print("  time series:", [round(x, 3) for x in lf])
        else:
            print("no per-shot loaded_frac available; summary.loading_rate=", lr)

    # ---- per-site UNIFORMITY (always; loading is non-uniform -- watch it) ----
    ps = d.get("per_site") or {}
    sl = ps.get("loading_rate") or []
    sx = ps.get("x") or []
    sy = ps.get("y") or []
    site = [(l, x, y) for l, x, y in zip(sl, sx, sy) if l is not None and l == l]
    if site:
        ls = [v[0] for v in site]
        nn = len(ls)
        mu = sum(ls) / nn
        sd = (sum((v - mu) ** 2 for v in ls) / nn) ** 0.5
        ss = sorted(ls)

        def _corr(a, b):
            ma = sum(a) / len(a); mb = sum(b) / len(b)
            num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
            da = (sum((ai - ma) ** 2 for ai in a)) ** 0.5
            db = (sum((bi - mb) ** 2 for bi in b)) ** 0.5
            return num / (da * db) if da * db else 0.0

        cx = _corr(ls, [v[1] for v in site])
        cy = _corr(ls, [v[2] for v in site])
        print("\n-- per-site uniformity (%d sites) --" % nn)
        print("  mean=%.3f  CV=%.0f%%  min=%.2f  p10=%.2f  p50=%.2f  p90=%.2f  max=%.2f"
              % (mu, 100 * sd / mu if mu else 0, min(ls), ss[int(.1 * (nn - 1))],
                 ss[int(.5 * (nn - 1))], ss[int(.9 * (nn - 1))], max(ls)))
        print("  gradient corr(load,x)=%+.2f  corr(load,y)=%+.2f" % (cx, cy))


if __name__ == "__main__":
    main()
