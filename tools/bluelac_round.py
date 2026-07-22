"""bluelac_round.py -- drive ONE round of ENHANCED-LOADING optimization (blue-detuned LAC).

seq = BlueTweezerLoadingSeq (Init -> BlueMOT -> SLM -> GreenMOT -> BlueLACStep(LAC cfg) ->
Cool556hX -> Imag399, NumImages=1). The BlueLAC pulse (blue-detuned 556 on the h beam) ejects ONE
atom of each colliding pair (vs red LAC ejecting both), so occupancy can exceed the ~50-60%
collisional-blockade ceiling -- the "enhanced loading" mechanism. All knobs swept here live under
``LAC.BlueLAC.*`` / ``LAC.*`` and are applied via g() overrides -- NO sequence changes, and
expConfig is untouched until a config is locked in (per-pattern params only).

Modes:
  det     : 1-D sweep of LAC.BlueLAC.FreqDetuning (MHz) -- the primary information scan.
  amptime : 2-D LAC.BlueLAC.Amp (dim1) x LAC.BlueLAC.Time (dim2) at --bdet.
  redlac  : 2-D LAC.FreqDetuning (dim1, MHz) x LAC.Time (dim2) -- the red clean-up pulse
            interplay at the chosen blue point (--bdet/--bamp/--btime).
  single  : 1x1 baseline/confirm point (all knobs pinned; use --reps high).

METRIC = LOADING (img1 fill fraction). The dashboard logicals are UNRELIABLE during a sweep that
swings loading (threshold-accumulator contamination -- bug-threshold-dim-scan-contamination), so
`watch` reports BOTH the dashboard-logicals loading AND a self-thresholded per-site loading
(per-site 2-Gaussian split, tools/persite_imaging.py math). Trust the self-thresholded one.
AFTER a dim-swinging campaign: re-anchor detection with a bright >=60-shot run (memory rule).

Phases (yb_analysis env): submit | watch, state in pyctrl/tmp/bluelac_state_r*.json.
"""
import argparse
import json
import os
import sys
import time

URL = "tcp://127.0.0.1:1408"
REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PYCTRL = os.path.join(REPO, "pyctrl")
STATE_DIR = os.path.join(PYCTRL, "tmp")
PATTERN = "tri_3013_camfb"                    # enhanced loading is for the BIG array (per user)
LOADING_PHASE = "phase/tri_3013_camfb.pt"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _pyctrl_path():
    for p in (PYCTRL, os.path.join(PYCTRL, "lib"), os.path.join(PYCTRL, "YbExptCtrl"),
              os.path.join(PYCTRL, "YbScans")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _zmq_int(verb, timeout_ms=4000):
    import zmq
    ctx = zmq.Context(); s = ctx.socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(URL); s.send_string(verb)
        if s.poll(timeout_ms) == 0:
            return None
        return int.from_bytes(s.recv(), "little")
    finally:
        s.close(linger=0); ctx.term()


def _consts():
    _pyctrl_path()
    from seq_config import SeqConfig
    import expConfig_helper
    from dyn_props import DynProps
    if not SeqConfig.get().consts:
        SeqConfig.load_real()
    return DynProps(expConfig_helper.apply_pattern(SeqConfig.get().consts, PATTERN))


def _colon(lo, step, hi):
    _pyctrl_path()
    from scan_export import matlab_colon
    return list(matlab_colon(lo, step, hi))


def _data_root():
    sys.path.insert(0, REPO)
    from yb_analysis import config
    return config.DATA_DIR


def _data_dirs(root):
    import glob
    return set(glob.glob(os.path.join(root, "*", "data_*")))


def _state_path(rnd):
    return os.path.join(STATE_DIR, "bluelac_state_r%d.json" % rnd)


def build(args):
    _pyctrl_path()
    from scan_group import ScanGroup
    c = _consts()
    g = ScanGroup()

    # pinned blue-LAC point (overridden per mode below); defaults = the pattern overlay/base
    bdet = float(args.bdet) * 1e6
    bamp = float(args.bamp) if args.bamp is not None else float(c.LAC.BlueLAC.Amp)
    btime = float(args.btime) if args.btime is not None else float(c.LAC.BlueLAC.Time)

    if args.mode == "det":
        dets = _colon(*args.bdets)                              # MHz (negative = blue side per config sign)
        g().LAC.BlueLAC.FreqDetuning.scan(1, [d * 1e6 for d in dets])
        g().LAC.BlueLAC.Amp = bamp
        g().LAC.BlueLAC.Time = btime
        axis_desc = "BlueLAC.FreqDetuning(MHz)=%s amp=%.3f time=%.3fs" % (
            [round(d, 3) for d in dets], bamp, btime)
    elif args.mode == "amptime":
        amps = [float(a) for a in _colon(*args.bamps)]
        times = [float(t) for t in _colon(*args.btimes)]
        g().LAC.BlueLAC.FreqDetuning = bdet
        g().LAC.BlueLAC.Amp.scan(1, amps)
        g().LAC.BlueLAC.Time.scan(2, times)
        axis_desc = "BlueLAC.Amp=%s x Time(s)=%s det=%.2fMHz" % (
            [round(a, 3) for a in amps], [round(t, 3) for t in times], args.bdet)
    elif args.mode == "redlac":
        rdets = _colon(*args.rdets)                             # MHz
        rtimes = [float(t) for t in _colon(*args.rtimes)]
        g().LAC.BlueLAC.FreqDetuning = bdet
        g().LAC.BlueLAC.Amp = bamp
        g().LAC.BlueLAC.Time = btime
        g().LAC.FreqDetuning.scan(1, [d * 1e6 for d in rdets])
        g().LAC.Time.scan(2, rtimes)
        axis_desc = "RED LAC.FreqDetuning(MHz)=%s x Time(s)=%s @ blue(%.2fMHz,%.3f,%.3fs)" % (
            [round(d, 3) for d in rdets], [round(t, 3) for t in rtimes], args.bdet, bamp, btime)
    else:  # single
        g().LAC.BlueLAC.FreqDetuning = bdet
        g().LAC.BlueLAC.Amp = bamp
        g().LAC.BlueLAC.Time = btime
        if args.ramp is not None:
            g().LAC.Amp = float(args.ramp)
        if args.rtime is not None:
            g().LAC.Time = float(args.rtime)
        axis_desc = "SINGLE blue(%.2fMHz,%.3f,%.3fs) red(amp=%s,time=%s)" % (
            args.bdet, bamp, btime, args.ramp, args.rtime)

    rp = g.runp()
    rp.NumImages = 1                     # loading image only
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = LOADING_PHASE
    rp.loading_defocus = float(args.defocus)
    return g, axis_desc


def do_submit(args):
    _pyctrl_path()
    from scan_export import scangroup_to_descriptor
    from yb_start_scan import submit_descriptor

    global PATTERN, LOADING_PHASE
    if args.pattern:
        PATTERN = args.pattern
    if args.loading_phase:
        LOADING_PHASE = args.loading_phase
    root = _data_root(); pre = _data_dirs(root)
    start = _zmq_int("get_seq_num")
    if start is None:
        print("ERROR: backend not answering at", URL); sys.exit(2)
    g, axis_desc = build(args)
    nseq = g.nseq()
    lbl = "BlueLAC_%s_r%d" % (args.mode, args.round)
    desc = scangroup_to_descriptor(
        g, "BlueTweezerLoadingSeq", opts={"rep": args.reps}, label=lbl,
        description=("Enhanced-loading campaign on %s: %s mode -- blue-detuned LAC toward 75-80%% "
                     "loading (2026-07-16, seq unchanged, per-pattern params only). %s"
                     % (PATTERN, args.mode, axis_desc)))
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d (%s): id=%d nseq=%d reps=%d -> target=%d start=%d"
          % (args.round, args.mode, did, nseq, args.reps, target, start))
    print("  " + axis_desc)

    data_dir = None
    t0 = time.time(); last = start
    while time.time() - t0 < 120:
        time.sleep(5)
        cur = _zmq_int("get_seq_num")
        new = _data_dirs(root) - pre
        if new and data_dir is None:
            data_dir = sorted(new, key=os.path.getmtime)[-1]
        if cur is not None and cur > last:
            print("  ALIVE: seq_num %d -> %d after %ds; data_dir=%s"
                  % (last, cur, int(time.time() - t0), data_dir))
            if cur >= start + 2 and data_dir:
                break
            last = cur
    st = {"round": args.round, "mode": args.mode, "start": start, "target": target,
          "nseq": nseq, "reps": args.reps, "data_dir": data_dir, "id": did,
          "axis_desc": axis_desc, "pattern": PATTERN}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


def _self_threshold_occ(I1):
    """Per-site 2-Gaussian threshold occupancy (persite_imaging math, trimmed)."""
    import numpy as np
    nsh, nsit = I1.shape
    occ = np.zeros((nsh, nsit), bool)
    for k in range(nsit):
        x = I1[:, k]
        x = x[np.isfinite(x)]
        if x.size < 8:
            continue
        xs = np.sort(x)
        mu_e, mu_a = xs[: x.size // 2].mean(), xs[x.size // 2:].mean()
        s_e = s_a = max(xs.std(), 1e-6)
        w = 0.5
        for _ in range(40):
            pe = (1 - w) * np.exp(-0.5 * ((I1[:, k] - mu_e) / s_e) ** 2) / (s_e + 1e-12)
            pa = w * np.exp(-0.5 * ((I1[:, k] - mu_a) / s_a) ** 2) / (s_a + 1e-12)
            r = pa / (pe + pa + 1e-30)
            wa = r.sum()
            if wa < 1 or (I1.shape[0] - wa) < 1:
                break
            mu_a2 = (r * I1[:, k]).sum() / wa
            mu_e2 = ((1 - r) * I1[:, k]).sum() / (I1.shape[0] - wa)
            s_a = np.sqrt((r * (I1[:, k] - mu_a2) ** 2).sum() / wa) + 1e-6
            s_e = np.sqrt(((1 - r) * (I1[:, k] - mu_e2) ** 2).sum() / (I1.shape[0] - wa)) + 1e-6
            w = wa / I1.shape[0]
            if abs(mu_a2 - mu_a) + abs(mu_e2 - mu_e) < 1e-4:
                mu_e, mu_a = mu_e2, mu_a2
                break
            mu_e, mu_a = mu_e2, mu_a2
        if mu_a < mu_e:
            mu_e, mu_a = mu_a, mu_e
            s_e, s_a = s_a, s_e
        thr = (mu_e * s_a + mu_a * s_e) / (s_e + s_a)
        occ[:, k] = I1[:, k] >= thr
    return occ


def analyze(data_dir, st):
    sys.path.insert(0, REPO)
    import numpy as np
    import h5py
    from yb_analysis.detection.scan_analysis import extract_scan_dims

    if not data_dir or not os.path.isdir(data_dir):
        print("ANALYZE: no data dir (%r)." % data_dir); return
    sid = os.path.basename(data_dir.rstrip("/\\"))
    cfg = json.load(open(os.path.join(data_dir, sid + ".json")))
    dims = extract_scan_dims(cfg)
    P = np.asarray(cfg["Params"]).ravel().astype(int)
    with h5py.File(os.path.join(data_dir, sid + ".h5"), "r") as f:
        seq_ids = f["seq_ids"][:]
        # NumImages=1 runs store unsuffixed keys ('intensities'/'logicals')
        ik = "intensities_img1" if "intensities_img1" in f else "intensities"
        lk = "logicals_img1" if "logicals_img1" in f else "logicals"
        I1 = f[ik][:].astype(float)
        L1 = f[lk][:].astype(bool)
    n = min(len(seq_ids), I1.shape[0])
    seq_ids, I1, L1 = seq_ids[:n], I1[:n], L1[:n]
    flat = P[seq_ids - 1] - 1
    if not dims:
        flat = np.zeros_like(flat); ncell = 1; labels = ["(single)"]
    else:
        ncell = int(np.prod([d["size"] for d in dims]))
        labels = None

    # SELF-THRESHOLD occupancy: per site over the WHOLE run only when the sweep doesn't change
    # imaging (it doesn't here -- imaging fixed, only loading varies), so the atom/empty peaks are
    # cell-independent and a whole-run per-site threshold is valid + max-statistics.
    occ = _self_threshold_occ(I1)

    print("=" * 74)
    print("BLUELAC round %d (%s)  scan_id=%s  n_shots=%d  cells=%d  pattern=%s"
          % (st["round"], st["mode"], sid.replace("data_", ""), n, ncell, st.get("pattern")))
    print("  " + st.get("axis_desc", ""))
    print("  %6s %10s %10s %10s %8s" % ("cell", "load_SELF", "load_dash", "CV_self", "nshot"))
    best = (None, -1.0)
    rows = []
    for p in range(ncell):
        r = np.where(flat == p)[0]
        if r.size == 0:
            rows.append(None); continue
        ls = occ[r].mean()
        ld = L1[r].mean()
        site_means = occ[r].mean(axis=0)
        cv = site_means.std() / max(site_means.mean(), 1e-9)
        rows.append({"p": p, "load_self": float(ls), "load_dash": float(ld),
                     "cv": float(cv), "n": int(r.size)})
        print("  %6d %10.4f %10.4f %10.3f %8d" % (p, ls, ld, cv, r.size))
        if ls > best[1]:
            best = (p, ls)
    if dims:
        for d in dims:
            print("  axis dim=%s size=%d values=%s" % (d.get("dim"), d["size"],
                  [round(v * (1e-6 if abs(v) > 1e4 else 1), 4) for v in d["values"]]))
    print("  BEST cell=%s load_SELF=%.4f" % best)
    print("  PICK_JSON " + json.dumps({"best_cell": best[0], "best_load": best[1],
          "rows": [r for r in rows if r]}))
    print("=" * 74)


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    target, start = st["target"], st["start"]
    t0 = time.time(); last, last_change = start, time.time()
    while True:
        cur = _zmq_int("get_seq_num"); el = int(time.time() - t0)
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
    ap.add_argument("phase", choices=["submit", "watch"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--mode", choices=["det", "amptime", "redlac", "single"], default="det")
    ap.add_argument("--bdets", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(-9.0, 0.6, -2.0),
                    help="det mode: BlueLAC.FreqDetuning colon (MHz); default = the documented "
                         "BlueLACScan sweep -(2:0.6:9) MHz")
    ap.add_argument("--bdet", type=float, default=-3.8, help="pinned BlueLAC.FreqDetuning (MHz)")
    ap.add_argument("--bamp", type=float, default=None, help="pinned BlueLAC.Amp (default: config)")
    ap.add_argument("--btime", type=float, default=None, help="pinned BlueLAC.Time s (default: config)")
    ap.add_argument("--bamps", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.05, 0.04, 0.29), help="amptime mode: BlueLAC.Amp colon")
    ap.add_argument("--btimes", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.1, 0.2, 0.9), help="amptime mode: BlueLAC.Time colon (s)")
    ap.add_argument("--rdets", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.10, 0.06, 0.34), help="redlac mode: LAC.FreqDetuning colon (MHz)")
    ap.add_argument("--rtimes", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.0, 0.015, 0.045), help="redlac mode: LAC.Time colon (s)")
    ap.add_argument("--ramp", type=float, default=None, help="single mode: pin LAC.Amp")
    ap.add_argument("--rtime", type=float, default=None, help="single mode: pin LAC.Time (s)")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--defocus", type=float, default=-2.0)
    ap.add_argument("--pattern", type=str, default=None)
    ap.add_argument("--loading-phase", type=str, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    else:
        do_watch(a)
