"""loading_report.py -- the analyst's ONE command for a loading scan (TweezerLoadingSeq, one image).

    <yb_analysis-python> pyctrl/tools/loading_report.py <scan_id|latest> [--drop N] [--json]

Sibling of fit_line.py / fit_spectrum.py for the loading vehicle, where there is no line to fit.
Everything it prints is derived from the run itself; the operator's handoff only has to say WHICH
numbers decide (rate / uniformity / stability / canary). It reports, in order:

  per-shot   the first shots in SHOT order + first-half vs second-half -> warm-up and drift
  per-cell   rate / SEM / CV / grad_x / grad_y per swept cell (if swept); 3-cell 1-D = canary -> R, S
  per-site   pooled over the kept shots: shot-noise-corrected CV, sites below thresholds, 3x3 block
             means, 5x5 corner means vs centre, worst sites
  2-atom     pooled per-site-normalized intensity: fraction in the 1.5-2.5 window (LAC judgement)
  d'         per-site discrimination health
  VERDICT    TRUST / CHECK / RE-RUN with the gate that decided it

Metric is SELF-THRESHOLDED per-site occupancy (per-site 2-Gaussian EM over the kept shots), the same
as loading_round.py, so stale dashboard thresholds are never trusted. Warm-up shots are dropped
BEFORE the EM. On a possibly-empty array the EM splits pedestal noise 50/50 and reports ~0.49 for
every cell -- the d' line is printed first for exactly that reason.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "pyctrl", "tools")
DATA = r"D:\OneDrive - Harvard University\Documents - Yb\Data"
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _scan_dir(sid):
    if str(sid) == "latest":
        days = sorted(d for d in glob.glob(os.path.join(DATA, "2*")) if os.path.isdir(d))
        runs = sorted(glob.glob(os.path.join(days[-1], "data_*")), key=os.path.getmtime)
        d = runs[-1]
        fid = os.path.basename(d)[5:]
        return d, fid
    sid = str(sid).replace("_", "")
    fid = sid[:8] + "_" + sid[8:]
    d = os.path.join(DATA, sid[:8], "data_" + fid)
    if not os.path.isdir(d):
        raise SystemExit("no such scan dir: %s" % d)
    return d, fid


def _load(d, fid):
    import numpy as np
    import h5py
    cfg = json.load(open(os.path.join(d, "data_%s.json" % fid)))
    with h5py.File(os.path.join(d, "data_%s.h5" % fid), "r") as f:
        seq = f["seq_ids"][:]
        ik = "intensities_img1" if "intensities_img1" in f else "intensities"
        I = f[ik][:].astype(float)
    n = min(len(seq), I.shape[0])
    return cfg, seq[:n], I[:n]


def _em(I):
    import numpy as np
    from cell_pooled_imaging import em2
    ns = I.shape[1]
    thr = np.empty(ns); dp = np.empty(ns); me_ = np.empty(ns); ma_ = np.empty(ns)
    for k in range(ns):
        me, se, ma, sa, _ = em2(I[:, k])
        thr[k] = (me * sa + ma * se) / (se + sa)
        dp[k] = (ma - me) / np.sqrt((se ** 2 + sa ** 2) / 2.0)
        me_[k], ma_[k] = me, ma
    return thr, dp, me_, ma_


def _lattice(v, ng):
    import numpy as np
    lo, hi = v.min(), v.max()
    return np.clip(np.round((v - lo) / ((hi - lo) / (ng - 1))).astype(int), 0, ng - 1)


def main():
    import numpy as np
    from yb_analysis.detection.scan_analysis import extract_scan_dims

    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--drop", type=int, default=10, help="warm-up shots to drop (0 disables)")
    ap.add_argument("--json", action="store_true", help="also print a JSON line")
    a = ap.parse_args()

    d, fid = _scan_dir(a.sid)
    sid = fid.replace("_", "")
    cfg, seq, I = _load(d, fid)
    n_all = len(seq)
    out = {"scan_id": sid, "scan_dir": d, "n_shots": int(n_all)}
    gates = {}

    # ---- per-shot series (ALL shots, shot order) -- read before anything is dropped -----------
    thr_all, dp_all, _, _ = _em(I)
    occ_all = I > thr_all[None, :]
    order = np.argsort(seq)
    per = occ_all[order].mean(axis=1)
    print("=" * 88)
    print("LOADING REPORT  scan=%s  shots=%d  sites=%d  dir=%s" % (sid, n_all, I.shape[1], d))
    desc = cfg.get("description") or ""
    if desc:
        print("  desc: %s" % (desc[:160] + ("..." if len(desc) > 160 else "")))
    print("  d' (all shots): median %.2f  frac>3 %.3f  sites<2: %d"
          % (np.median(dp_all), (dp_all > 3).mean(), int((dp_all < 2).sum())))
    gates["detection-healthy"] = bool(np.median(dp_all) >= 3 and (dp_all > 3).mean() >= 0.90)
    k = min(15, len(per))
    print("  per-shot (first %d, shot order): %s" % (k, " ".join("%.3f" % v for v in per[:k])))
    h = len(per) // 2
    steady = per[a.drop:].mean() if len(per) > a.drop + 2 else per.mean()
    warm = per[:3].mean() if len(per) >= 3 else per.mean()
    drift = (per[h:].mean() / max(per[:h].mean(), 1e-9) - 1.0) if h else 0.0
    dims0 = extract_scan_dims(cfg)
    if dims0:
        print("  first 3 shots %.3f  vs steady %.3f  (scrambled sweep: cells mix into the per-shot "
              "series, so judge warm-up from a single-point run, not from this line)" % (warm, steady))
    else:
        print("  first 3 shots %.3f  vs steady %.3f  -> %s" % (
            warm, steady, "WARM-UP PRESENT" if warm < 0.7 * steady else "no warm-up"))
    print("  first half %.4f  second half %.4f  -> %+.1f%%" % (per[:h].mean(), per[h:].mean(), 100 * drift))
    out.update(per_shot_first=[float(v) for v in per[:k]], steady=float(steady),
               warmup_present=bool(warm < 0.7 * steady), drift_frac=float(drift))
    gates["no-drift"] = bool(abs(drift) < 0.05)

    # ---- drop warm-up ------------------------------------------------------------------------
    ndrop = 0
    if a.drop > 0:
        if len(seq) - a.drop >= 3:
            keep = np.sort(order[a.drop:]); seq, I = seq[keep], I[keep]; ndrop = a.drop
        else:
            print("  *** WARNING: only %d shots, cannot drop %d warm-up shots -- numbers below are "
                  "WARM-UP-BIASED (read LOW). Re-run with >= %d shots. ***" % (len(seq), a.drop, a.drop + 20))
    gates["enough-shots"] = bool(len(seq) >= 20)
    thr, dp, me_, ma_ = _em(I)
    occ = I > thr[None, :]
    N = occ.shape[0]
    print("  kept %d shots (dropped %d warm-up)" % (N, ndrop))

    # ---- per-cell (swept) ----------------------------------------------------------------------
    dims = extract_scan_dims(cfg)
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    flat = P[seq - 1] - 1
    ncell = int(np.prod([x["size"] for x in dims])) if dims else 1
    if not dims:
        flat = np.zeros_like(flat)
    X = np.asarray(cfg.get("initGridLocationsX") or [], float)
    Y = np.asarray(cfg.get("initGridLocationsY") or [], float)
    have_xy = len(X) == I.shape[1] and len(Y) == I.shape[1]
    cells = []; best = None
    if ncell > 1:
        print("-" * 88)
        for dd in dims:
            vals = [round(v * (1e-6 if abs(v) > 1e4 else 1), 4) for v in dd["values"]]
            print("  axis size=%d values=%s%s" % (dd["size"], vals, " (MHz)" if abs(dd["values"][0]) > 1e4 else ""))
        print("  %5s %9s %8s %8s %8s %8s %6s" % ("cell", "load", "sem", "CV", "grad_x", "grad_y", "n"))
        for p in range(ncell):
            r = np.where(flat == p)[0]
            if r.size == 0:
                print("  %5d  (no shots)" % p); cells.append(None); continue
            sm = occ[r].mean(axis=0)
            gx = float(np.corrcoef(X, sm)[0, 1]) if have_xy else float("nan")
            gy = float(np.corrcoef(Y, sm)[0, 1]) if have_xy else float("nan")
            c = dict(p=p, load=float(sm.mean()), sem=float(occ[r].mean(axis=1).std() / np.sqrt(r.size)),
                     cv=float(sm.std() / max(sm.mean(), 1e-9)), gx=gx, gy=gy, n=int(r.size))
            cells.append(c)
            print("  %5d %9.4f %8.4f %8.3f %8.3f %8.3f %6d" % (p, c["load"], c["sem"], c["cv"], gx, gy, r.size))
        good = [c for c in cells if c]
        best = max(good, key=lambda c: c["load"])
        edge = False
        if len(dims) == 1:
            edge = best["p"] in (0, ncell - 1)
        else:
            s0 = dims[0]["size"]; i0, j1 = best["p"] % s0, best["p"] // s0
            edge = i0 in (0, s0 - 1) or j1 in (0, dims[1]["size"] - 1)
        print("  BEST cell=%d load=%.4f+-%.4f  -> %s" % (best["p"], best["load"], best["sem"],
                                                       "EDGE-PINNED (expand)" if edge else "interior"))
        gates["optimum-interior"] = not edge
        out["cells"] = good; out["best"] = best
        if ncell == 3 and len(dims) == 1 and all(cells):
            R = cells[2]["load"] / max(cells[0]["load"], 1e-9)
            print("  CANARY: R = L(high)/L(low) = %.3f   S = L(centre) = %.4f   grad_x(centre) = %+.3f"
                  % (R, cells[1]["load"], cells[1]["gx"]))
            out["canary"] = dict(R=float(R), S=cells[1]["load"], gx_centre=cells[1]["gx"])

    # ---- per-site: the whole run for a single point, the BEST CELL for a sweep ---------------
    # Pooling a sweep's cells would count its deliberately off-peak cells (a bracket's shoulders,
    # a grid's dead edge) as non-uniformity. The uniformity gates are only meaningful with enough
    # shots at ONE condition, so on a sweep they are applied to the best cell and only if it has
    # >= 20 shots; otherwise they are reported but not gated.
    print("-" * 88)
    if ncell > 1:
        sel = np.where(flat == best["p"])[0]
        occ_u = occ[sel]; Nu = int(sel.size)
        where = "best cell %d (%d shots)" % (best["p"], Nu)
    else:
        occ_u = occ; Nu = N; where = "%d shots" % N
    gate_uniformity = Nu >= 20
    ps = occ_u.mean(axis=0)
    obs = ps.std() / max(ps.mean(), 1e-9)
    shot = np.sqrt(ps * (1 - ps) / Nu).mean() / max(ps.mean(), 1e-9)
    true_cv = float(np.sqrt(max(obs ** 2 - shot ** 2, 0)))
    print("  per-site over %s%s: mean %.4f | CV obs %.3f, shot-noise floor %.3f -> TRUE CV %.3f"
          % (where, "" if gate_uniformity else " -- NOT GATED, too few shots; run a single-point verify",
             ps.mean(), obs, shot, true_cv))
    lows = {t: int((ps < t).sum()) for t in (0.10, 0.25, 0.35, 0.45)}
    print("  sites below 0.10/0.25/0.35/0.45: %d / %d / %d / %d" % tuple(lows.values()))
    out.update(mean=float(ps.mean()), true_cv=true_cv, sites_below=lows, uniformity_shots=Nu,
               uniformity_gated=bool(gate_uniformity),
               dprime_median=float(np.median(dp)), dprime_frac_gt3=float((dp > 3).mean()))
    if gate_uniformity:
        gates["no-dead-sites"] = bool(lows[0.25] <= 0.01 * len(ps))
    corners = None
    if have_xy:
        ng = int(np.ceil(np.sqrt(len(ps))))
        ix, iy = _lattice(X, ng), _lattice(Y, ng)
        G = np.full((ng, ng), np.nan)
        for r_, c_, v in zip(iy, ix, ps):
            G[r_, c_] = v
        def blk(r0, r1, c0, c1):
            s = G[r0:r1, c0:c1]; s = s[~np.isnan(s)]; return float(s.mean()) if s.size else float("nan")
        e = [0, ng // 3, 2 * ng // 3, ng]
        print("  3x3 blocks (row 0 = low y): " + " | ".join(
            " ".join("%.3f" % blk(e[r], e[r + 1], e[c], e[c + 1]) for c in range(3)) for r in range(3)))
        kk = 5; cen = blk(ng // 2 - 3, ng // 2 + 4, ng // 2 - 3, ng // 2 + 4)
        corners = dict(BL=blk(0, kk, 0, kk), BR=blk(0, kk, ng - kk, ng), TL=blk(ng - kk, ng, 0, kk),
                       TR=blk(ng - kk, ng, ng - kk, ng), centre=cen)
        print("  corners 5x5: BL %.3f  BR %.3f  TL %.3f  TR %.3f  vs centre %.3f  (worst corner %.0f%% of centre)"
              % (corners["BL"], corners["BR"], corners["TL"], corners["TR"], cen,
                 100 * min(corners[q] for q in ("BL", "BR", "TL", "TR")) / max(cen, 1e-9)))
        if gate_uniformity:
            gates["no-dead-corner"] = bool(min(corners[q] for q in ("BL", "BR", "TL", "TR")) >= 0.8 * cen)
        gx = float(np.corrcoef(X, ps)[0, 1]); gy = float(np.corrcoef(Y, ps)[0, 1])
        print("  gradient pooled: grad_x %+.3f  grad_y %+.3f" % (gx, gy))
        out.update(corners=corners, grad_x=gx, grad_y=gy)
    lo = np.argsort(ps)[:8]
    print("  worst sites: " + ", ".join("s%d=%.2f(d'%.1f)" % (i, ps[i], dp[i]) for i in lo))

    # ---- 2-atom fraction -----------------------------------------------------------------------
    Z = (I - me_[None, :]) / np.maximum(ma_ - me_, 1e-9)[None, :]
    z = Z.ravel()
    f2 = float(((z >= 1.5) & (z < 2.5)).mean()); loaded = float((z > 0.5).mean())
    print("  2-atom window (1.5-2.5 of the per-site 1-atom level): %.4f of site-shots = %.1f%% of loaded"
          % (f2, 100 * f2 / max(loaded, 1e-9)))
    out["two_atom_frac"] = f2
    gates["single-atom"] = bool(f2 / max(loaded, 1e-9) < 0.03)

    # ---- verdict -------------------------------------------------------------------------------
    print("-" * 88)
    print("  GATES: " + " | ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in gates.items()))
    hard = ["detection-healthy", "enough-shots"]
    if any(not gates[k] for k in hard):
        verdict = "RE-RUN"
    elif all(gates.values()):
        verdict = "TRUST"
    else:
        verdict = "CHECK"
    failed = [k for k, v in gates.items() if not v]
    print("  VERDICT: %s%s" % (verdict, (" -- " + ", ".join(failed)) if failed else " -- all gates pass"))
    print("=" * 88)
    out.update(gates=gates, verdict=verdict)
    if a.json:
        print("JSON " + json.dumps(out, default=float))


if __name__ == "__main__":
    main()
