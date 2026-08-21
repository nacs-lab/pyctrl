"""loading_round.py -- generic 1-D/2-D LOADING sweep driver (loading-optimization.md runbook).

One tool for every phase of the loading runbook: sweep any dotted config path(s) on the loading
sequence and read the honest loading metrics. Sibling of ``bluelac_round.py`` (blue-LAC only) and
``imaging_round.py`` (imaging); this one owns the MOT/loading knobs.

  vehicle : ``TweezerLoadingSeq`` (default) or ``BlueTweezerLoadingSeq`` (--blue-lac, enhanced
            loading). NumImages=1 -- the loading image only.
  metric  : SELF-THRESHOLDED per-site occupancy (per-site 2-Gaussian EM over the whole run --
            valid because imaging is FIXED across cells in a loading sweep), so the dashboard's
            per-pattern thresholds (stale on a freshly re-loaded array) are never trusted.
            Reported per cell: loading mean, CV, per-site d' median, and the x/y gradient
            (uniformity is co-equal with rate -- loading-optimization.md).
  imaging : the loading readout can be sharpened with --img1/--img2 (BlueMOT.Img1/Img2PIDSet);
            img2 never fires here, and more img1 light only helps detection (no survival at stake).

Paths are dotted config keys, e.g. ``BlueMOT.LoadingTime``, ``GreenMOT.BiasCoilCurrent.X``,
``GreenMOT.CoolDown.Amp``, ``LAC.FreqDetuning``. Values in the config's OWN units (Hz for
frequencies) unless --mhz is given for that axis.

Usage (yb_analysis env, from pyctrl/):
    python tools/loading_round.py submit --round 1 --pattern tri_3013_camfb \
        --loading-phase phase/tri_3013_camfb.pt --defocus -2 --img1 0.8 \
        --p1 BlueMOT.LoadingTime --v1 0.1 0.15 0.7 --reps 4
    python tools/loading_round.py watch --round 1
"""
import argparse
import json
import os
import sys
import time

import bluelac_round as B

STATE_DIR = B.STATE_DIR


def _state_path(rnd):
    return os.path.join(STATE_DIR, "loading_state_r%d.json" % rnd)


def _set_path(g, path, value=None, dim=None, values=None):
    """Apply g().<dotted.path> = value  (or .scan(dim, values))."""
    parts = path.split(".")
    parent = g()
    for part in parts[:-1]:
        parent = getattr(parent, part)
    if values is not None:
        getattr(parent, parts[-1]).scan(dim, values)
    else:
        setattr(parent, parts[-1], value)


def build(args):
    B._pyctrl_path()
    from scan_group import ScanGroup
    g = ScanGroup()

    if args.img1 is not None:
        _set_path(g, "BlueMOT.Img1PIDSet", value=float(args.img1))
    if args.img2 is not None:
        _set_path(g, "BlueMOT.Img2PIDSet", value=float(args.img2))
    for pin in (args.pin or []):
        path, val = pin.split("=", 1)
        _set_path(g, path.strip(), value=float(val))

    desc_bits = []
    axes = []
    for i, (path, vals, mhz) in enumerate(
            [(args.p1, args.v1, args.mhz1), (args.p2, args.v2, args.mhz2)]):
        if not path:
            continue
        vv = [float(v) for v in B._colon(*vals)]
        scaled = [v * 1e6 for v in vv] if mhz else vv
        _set_path(g, path, dim=i + 1, values=scaled)
        axes.append({"path": path, "values": vv, "mhz": bool(mhz)})
        desc_bits.append("%s%s=%s" % (path, "(MHz)" if mhz else "", [round(v, 4) for v in vv]))
    if not axes:
        desc_bits.append("SINGLE POINT")
    if args.img1 is not None or args.img2 is not None:
        desc_bits.append("PIDset %s/%s" % (args.img1, args.img2))
    if args.pin:
        desc_bits.append("pinned " + ",".join(args.pin))
    axis_desc = "  ".join(desc_bits)

    rp = g.runp()
    rp.NumImages = 1
    rp.Scramble = 1                      # randomize point order vs slow drift
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = B.LOADING_PHASE
    rp.loading_defocus = float(args.defocus)
    return g, axis_desc, axes


def do_submit(args):
    B._pyctrl_path()
    from scan_export import scangroup_to_descriptor
    from yb_start_scan import submit_descriptor

    if args.pattern:
        B.PATTERN = args.pattern
    if args.loading_phase:
        B.LOADING_PHASE = args.loading_phase
    root = B._data_root(); pre = B._data_dirs(root)
    start = B._zmq_int("get_seq_num")
    if start is None:
        print("ERROR: backend not answering at", B.URL); sys.exit(2)
    g, axis_desc, axes = build(args)
    nseq = g.nseq()
    seq = "BlueTweezerLoadingSeq" if args.blue_lac else "TweezerLoadingSeq"
    lbl = "LoadingOpt_r%d" % args.round
    desc = scangroup_to_descriptor(
        g, seq, opts={"rep": args.reps}, label=lbl,
        description=(args.desc or
                     ("Loading re-optimization on %s (%s): %s. Metric = self-thresholded per-site "
                      "occupancy + CV + x/y gradient." % (B.PATTERN, seq, axis_desc))))
    did = submit_descriptor(B.URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d: id=%d seq=%s nseq=%d reps=%d -> target=%d start=%d"
          % (args.round, did, seq, nseq, args.reps, target, start))
    print("  " + axis_desc)

    data_dir = None
    t0 = time.time(); last = start
    while time.time() - t0 < 150:
        time.sleep(5)
        cur = B._zmq_int("get_seq_num")
        new = B._data_dirs(root) - pre
        if new and data_dir is None:
            data_dir = sorted(new, key=os.path.getmtime)[-1]
        if cur is not None and cur > last:
            print("  ALIVE: seq_num %d -> %d after %ds; data_dir=%s"
                  % (last, cur, int(time.time() - t0), data_dir))
            if cur >= start + 2 and data_dir:
                break
            last = cur
    st = {"round": args.round, "start": start, "target": target, "nseq": nseq,
          "reps": args.reps, "data_dir": data_dir, "id": did, "axis_desc": axis_desc,
          "pattern": B.PATTERN, "axes": axes, "seq": seq}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


def analyze(data_dir, st):
    sys.path.insert(0, B.REPO)
    import numpy as np
    import h5py
    from yb_analysis.detection.scan_analysis import extract_scan_dims
    from cell_pooled_imaging import em2

    if not data_dir or not os.path.isdir(data_dir):
        print("ANALYZE: no data dir (%r)." % data_dir); return
    sid = os.path.basename(data_dir.rstrip("/\\"))
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    dims = extract_scan_dims(cfg)
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    with h5py.File(os.path.join(data_dir, sid + ".h5"), "r") as f:
        seq_ids = f["seq_ids"][:]
        ik = "intensities_img1" if "intensities_img1" in f else "intensities"
        I1 = f[ik][:].astype(float)
    n = min(len(seq_ids), I1.shape[0])
    seq_ids, I1 = seq_ids[:n], I1[:n]

    # per-site 2-Gaussian EM over the WHOLE run (imaging fixed across cells) -> threshold + d'
    thr = np.empty(I1.shape[1]); dp = np.empty(I1.shape[1])
    for k in range(I1.shape[1]):
        me, se, ma, sa, _ = em2(I1[:, k])
        thr[k] = (me * sa + ma * se) / (se + sa)
        dp[k] = (ma - me) / np.sqrt((se ** 2 + sa ** 2) / 2.0)
    occ = I1 > thr[None, :]

    xy = _site_xy(cfg, I1.shape[1])
    flat = P[seq_ids - 1] - 1
    ncell = int(np.prod([d["size"] for d in dims])) if dims else 1
    if not dims:
        flat = np.zeros_like(flat)

    print("=" * 86)
    print("LOADING round %d  scan=%s  n=%d cells=%d sites=%d pattern=%s"
          % (st["round"], sid.replace("data_", ""), n, ncell, I1.shape[1], st.get("pattern")))
    print("  " + st.get("axis_desc", ""))
    print("  d' (whole run): median %.2f  frac>3 %.2f" % (np.median(dp), (dp > 3).mean()))
    print("  %5s %9s %8s %8s %8s %8s %7s" % ("cell", "load", "sem", "CV", "grad_x", "grad_y", "nshot"))
    rows = []
    for p in range(ncell):
        r = np.where(flat == p)[0]
        if r.size == 0:
            rows.append(None); print("  %5d  (no shots)" % p); continue
        site_means = occ[r].mean(axis=0)
        load = float(site_means.mean())
        sem = float(occ[r].mean(axis=1).std() / max(np.sqrt(r.size), 1))
        cv = float(site_means.std() / max(site_means.mean(), 1e-9))
        gx = gy = float("nan")
        if xy is not None:
            gx = float(np.corrcoef(xy[:, 0], site_means)[0, 1])
            gy = float(np.corrcoef(xy[:, 1], site_means)[0, 1])
        rows.append({"p": p, "load": load, "sem": sem, "cv": cv, "gx": gx, "gy": gy,
                     "n": int(r.size)})
        print("  %5d %9.4f %8.4f %8.3f %8.3f %8.3f %7d" % (p, load, sem, cv, gx, gy, r.size))
    if dims:
        for d in dims:
            print("  axis dim=%s size=%d values=%s" % (
                d.get("dim"), d["size"],
                [round(v * (1e-6 if abs(v) > 1e4 else 1), 4) for v in d["values"]]))
    good = [r for r in rows if r]
    if good:
        best = max(good, key=lambda r: r["load"])
        print("  BEST cell=%d load=%.4f+-%.4f CV=%.3f" % (best["p"], best["load"], best["sem"], best["cv"]))
        print("  PICK_JSON " + json.dumps({"rows": good}))
    print("=" * 86)


def _site_xy(cfg, nsite):
    """Site (x,y) from the scan config's detection grid, if present."""
    import numpy as np
    for key in ("sites", "grid", "detection"):
        v = cfg.get(key)
        if isinstance(v, dict):
            for xk, yk in (("x", "y"), ("image_x", "image_y")):
                if xk in v and yk in v and len(v[xk]) == nsite:
                    return np.column_stack([np.asarray(v[xk], float), np.asarray(v[yk], float)])
        if isinstance(v, list) and len(v) == nsite and isinstance(v[0], dict):
            for xk, yk in (("x", "y"), ("image_x", "image_y")):
                if xk in v[0]:
                    return np.array([[float(s[xk]), float(s[yk])] for s in v])
    return None


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    target, start = st["target"], st["start"]
    t0 = time.time(); last, last_change = start, time.time()
    while True:
        cur = B._zmq_int("get_seq_num"); el = int(time.time() - t0)
        if cur is not None:
            if cur != last:
                last, last_change = cur, time.time()
            print("  [%ds] seq_num=%d  %d/%d" % (el, cur, cur - start, target - start))
            if cur >= target:
                print("  COMPLETE"); break
            if time.time() - last_change > 240:
                print("  STALL 240s; partial."); break
        if time.time() - t0 > args.timeout:
            print("  TIMEOUT; partial."); break
        time.sleep(10)
    time.sleep(8)
    analyze(st["data_dir"], st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "watch", "analyze"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--p1", type=str, default=None, help="dim-1 dotted config path")
    ap.add_argument("--v1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=None)
    ap.add_argument("--mhz1", action="store_true", help="dim-1 values are MHz (x1e6)")
    ap.add_argument("--p2", type=str, default=None, help="dim-2 dotted config path")
    ap.add_argument("--v2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=None)
    ap.add_argument("--mhz2", action="store_true", help="dim-2 values are MHz (x1e6)")
    ap.add_argument("--pin", action="append", default=None, metavar="PATH=VALUE",
                    help="pin any dotted path (repeatable), e.g. BlueMOT.LoadingTime=0.3")
    ap.add_argument("--img1", type=float, default=None, help="BlueMOT.Img1PIDSet (loading readout)")
    ap.add_argument("--img2", type=float, default=None, help="BlueMOT.Img2PIDSet")
    ap.add_argument("--blue-lac", action="store_true", help="use BlueTweezerLoadingSeq (enhanced)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--defocus", type=float, default=-2.0)
    ap.add_argument("--pattern", type=str, default=None)
    ap.add_argument("--loading-phase", type=str, default=None)
    ap.add_argument("--desc", type=str, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    elif a.phase == "watch":
        do_watch(a)
    else:
        with open(_state_path(a.round)) as f:
            st = json.load(f)
        analyze(st["data_dir"], st)
