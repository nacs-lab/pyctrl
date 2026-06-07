"""uniformity_by_param.py -- per-sweep-point loading UNIFORMITY (CV + x/y gradient).

The dashboard API's per_site is pooled over the WHOLE run; this computes, FOR EACH swept
parameter value, the per-site loading -> mean, CV, and gradient corr(load,x)/corr(load,y).
Lets one grid scan reveal which knob flattens the spatial loading gradient.

Reads the scan .h5 via the yb_analysis loader + unpack_scan_logicals, and site (x,y) from the
dashboard analysis API. RUN WITH THE yb_analysis ENV PYTHON (has h5py/numpy + the package):

  <yb_analysis-python> uniformity_by_param.py <scan_id|latest> [--host h:p]
"""

import argparse
import json
import os
import sys
import urllib.request

HOSTS = ["127.0.0.1:8050", "100.86.15.43:8050"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiment-control
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
    ap = argparse.ArgumentParser(description="Per-sweep-point loading uniformity (CV + gradient).")
    ap.add_argument("scan", help="'latest' or scan_id")
    ap.add_argument("--host", default=None)
    args = ap.parse_args()
    hosts = [args.host] if args.host else HOSTS

    sid = args.scan
    if sid == "latest":
        lst, _ = _get("/api/runs/list?max=1", hosts)
        sid = lst["runs"][0]["scan_id"]
    d, _ = _get("/api/runs/%s/analysis" % sid, hosts)
    scan_dir = d.get("scan_dir")
    ps = d.get("per_site") or {}
    Xl = ps.get("x") or []
    Yl = ps.get("y") or []

    import numpy as np
    from yb_analysis.analysis.load_data import load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals

    sd = load_scan_from_path(scan_dir)
    sp, logic1, _logic2, reps = unpack_scan_logicals(
        sd["Scan"], logicals=sd.get("logicals"), seq_ids=sd.get("seq_ids"),
        mat_path=sd.get("mat_path"),
        logicals_img1=sd.get("logicals_img1"), logicals_img2=sd.get("logicals_img2"))
    nS = logic1.shape[0]
    X = np.asarray(Xl[:nS], float)
    Y = np.asarray(Yl[:nS], float)
    if len(X) != nS:
        print("WARN: site count mismatch (logic1=%d, api x=%d) -- gradient skipped" % (nS, len(X)))
        X = Y = None
    sp = np.asarray(sp)

    def corr(a, b):
        if b is None:
            return float("nan")
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        aa = a[m] - a[m].mean()
        bb = b[m] - b[m].mean()
        den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
        return float((aa * bb).sum() / den) if den else float("nan")

    print("scan %s  nsites=%d  nparams=%d" % (sid, nS, logic1.shape[1]))
    print("%-30s %6s %5s %7s %7s %5s" % ("param", "mean", "CV%", "corrX", "corrY", "nrep"))
    rows = []
    for p in range(logic1.shape[1]):
        r = int(reps[p])
        if r <= 0:
            continue
        load = logic1[:, p, :r].sum(axis=1) / r          # per-site loading at this param
        mu = float(load.mean())
        sdv = float(load.std())
        cv = 100 * sdv / mu if mu else 0.0
        cx = corr(load, X)
        cy = corr(load, Y)
        val = sp[p] if sp.ndim == 1 else tuple(round(float(v), 5) for v in sp[p])
        rows.append((val, mu, cv, cx, cy, r))
        print("%-30s %6.3f %5.0f %+7.2f %+7.2f %5d" % (str(val), mu, cv, cx, cy, r))
    # Flattest (min |corrY|) among points with decent loading
    good = [x for x in rows if x[1] > 0.3]
    if good:
        flat = min(good, key=lambda x: abs(x[4]))
        print("\nFlattest-Y (load>0.3): param=%s  mean=%.3f CV=%.0f%% corrY=%+.2f"
              % (flat[0], flat[1], flat[2], flat[4]))


if __name__ == "__main__":
    main()
