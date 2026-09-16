"""cooling_round.py -- drive ONE round of the CoolingScan_RNR 2-D optimization.

Two phases (run under the yb_analysis env -- has zmq + numpy + h5py + the yb_analysis pkg):

  submit : record start seq_num, submit the grid via the descriptor ZMQ verb (NO engine
           import -- builds the ScanGroup with CoolingScan_RNR.build + submit_descriptor),
           confirm the shot counter actually advances (apparatus is alive), find the new
           data dir, write a state file.
  watch  : read the state file, poll get_seq_num until target (R*nseq) or stall, then
           analyze_scan_dir -> survival(det,amp) map + peak (MATLAB column-major mapping).

Grid is the colon (lo, step, hi) tuple CoolingScan_RNR uses: fdet in MHz (*1e6 -> Hz),
famp in raw amplitude.
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
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _pyctrl_path():
    for p in (PYCTRL, os.path.join(PYCTRL, "lib"), os.path.join(PYCTRL, "YbExptCtrl"),
              os.path.join(PYCTRL, "YbScans")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _zmq_int(verb, timeout_ms=4000):
    import zmq
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(URL)
        s.send_string(verb)
        if s.poll(timeout_ms) == 0:
            return None
        return int.from_bytes(s.recv(), "little")
    finally:
        s.close(linger=0)
        ctx.term()


def _queue_json(timeout_ms=4000):
    """Parsed queue_list reply (dict with queued/running/history), or None on timeout."""
    import zmq
    ctx = zmq.Context(); s = ctx.socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(URL); s.send_string("queue_list")
        if s.poll(timeout_ms) == 0:
            return None
        return json.loads(s.recv().decode())
    finally:
        s.close(linger=0); ctx.term()


def _queue_find(q, did):
    """(where, entry) for job id `did`: 'queued' (with _pos), 'running', 'history', or (None, None)."""
    if not q:
        return None, None
    run = q.get("running")
    if isinstance(run, dict) and run.get("id") == did:
        return "running", run
    for i, ent in enumerate(q.get("queued") or []):
        if isinstance(ent, dict) and ent.get("id") == did:
            ent = dict(ent); ent["_pos"] = i
            return "queued", ent
    for ent in q.get("history") or []:
        if isinstance(ent, dict) and ent.get("id") == did:
            return "history", ent
    return None, None


def _job_data_dir(root, ent):
    fid = ent.get("file_id") if ent else None
    if not fid:
        return None
    return os.path.join(root, fid[:8], "data_" + fid)


def _data_root():
    sys.path.insert(0, REPO)
    from yb_analysis import config
    return config.DATA_DIR


def _data_dirs(root):
    import glob
    return set(glob.glob(os.path.join(root, "*", "data_*")))


def _state_path(rnd):
    return os.path.join(STATE_DIR, "cooling_state_r%d.json" % rnd)


def do_submit(args):
    _pyctrl_path()
    from scan_export import scangroup_to_descriptor
    from yb_start_scan import submit_descriptor
    import CoolingScan_RNR

    root = _data_root()
    pre = _data_dirs(root)

    start = _zmq_int("get_seq_num")
    status = None
    try:
        import zmq  # noqa
        ctx = __import__("zmq").Context(); s = ctx.socket(__import__("zmq").REQ)
        s.setsockopt(__import__("zmq").LINGER, 0); s.connect(URL); s.send_string("get_status")
        status = s.recv().decode() if s.poll(3000) else None
        s.close(linger=0); ctx.term()
    except Exception:
        pass
    if start is None:
        print("ERROR: backend not answering get_seq_num at", URL); sys.exit(2)
    print("backend get_status=%r  start seq_num=%d" % (status, start))

    x_pin = (args.xpin[0] * 1e6, args.xpin[1]) if args.xpin else CoolingScan_RNR.DEF_X_PIN
    h_pin = (args.hpin[0] * 1e6, args.hpin[1]) if args.hpin else CoolingScan_RNR.DEF_H_PIN
    g = CoolingScan_RNR.build(beam=args.beam, fdet=tuple(args.fdet), famp=tuple(args.famp),
                              x_pin=x_pin, h_pin=h_pin, release_time=args.rnr_time)

    # --- optional per-array loading phase + detection pattern (default None = unchanged behavior).
    # Setting loading_phase makes SlmScanSession write that hologram at scan start AND apply the
    # matching expConfig ByPattern overlay (VSLMServo etc). Detection uses the named pattern.
    if args.loading_phase:
        rp = g.runp()
        rp.loading_phase = args.loading_phase
        rp.loading_defocus = float(args.defocus)
        rp.useScanLongSlmLock = 1
        pat = {"name": args.pattern_name, "base_phase_path": args.loading_phase,
               "order": "col", "legacy_zerniked": False}
        if args.planes:
            pat["planes_z_rad"] = [float(z) for z in args.planes]
        rp.imagePatternsJson = json.dumps([pat, pat])   # 2 frames (NumImages=2)
        print("  loading_phase=%s defocus=%g pattern=%s planes=%s"
              % (args.loading_phase, args.defocus, args.pattern_name, args.planes))

    nseq = g.nseq()
    lbl = "CoolingScan_RNR_%s_r%d" % (args.beam, args.round)
    desc = scangroup_to_descriptor(g, "ReleaseRecaptureSeq", opts={"rep": args.reps}, label=lbl,
                                   description=getattr(args, "desc", None) or "")
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    det = [round(v * 1e6) for v in _colon(*args.fdet)]
    amp = [round(v, 4) for v in _colon(*args.famp)]
    pinned = ("h@(%.3fMHz,%.3f)" % (h_pin[0] / 1e6, h_pin[1])) if args.beam == "X" \
             else ("X@(%.3fMHz,%.3f)" % (x_pin[0] / 1e6, x_pin[1]))
    print("submitted round %d: id=%d beam=%s nseq=%d reps=%d rnr=%.1fus pinned=%s -> target=%d"
          % (args.round, did, args.beam, nseq, args.reps, args.rnr_time * 1e6, pinned, target))
    print("  %s det(Hz)=%s" % (args.beam, det))
    print("  %s amp    =%s" % (args.beam, amp))

    # Resolve OUR job by queue id -> file_id (robust to other runs queued/running in between).
    data_dir = None
    t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(5)
        where, ent = _queue_find(_queue_json(), did)
        if where == "queued":
            print("  QUEUED behind %d job(s) after %ds (watch will wait)." % (ent.get("_pos", 0), int(time.time() - t0)))
            break
        if where in ("running", "history"):
            data_dir = _job_data_dir(root, ent)
            print("  %s after %ds; data_dir=%s" % (where.upper(), int(time.time() - t0), data_dir))
            break
    else:
        print("  WARNING: job id %d not visible in queue_list after 120 s." % did)

    st = {"round": args.round, "beam": args.beam, "start": start, "target": target, "nseq": nseq,
          "reps": args.reps, "rnr_time": args.rnr_time, "x_pin": list(x_pin), "h_pin": list(h_pin),
          "data_dir": data_dir, "fdet": list(args.fdet), "famp": list(args.famp), "id": did}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


def _colon(lo, step, hi):
    _pyctrl_path()
    from scan_export import matlab_colon
    return matlab_colon(lo, step, hi)


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    data_dir = st.get("data_dir"); did = st.get("id")
    total = st["target"] - st["start"]
    root = _data_root()
    print("watching round %d: job id=%s total=%d data_dir=%s" % (args.round, did, total, data_dir))
    t0 = time.time(); last_prog, last_change = -1, time.time()
    while True:
        el = int(time.time() - t0)
        where, ent = _queue_find(_queue_json(), did)
        if where is None:
            print("  [%ds] job %s not in queue_list (backend restarted?)" % (el, did))
        elif where == "queued":
            last_change = time.time()
            print("  [%ds] QUEUED (position %d) -- other runs ahead" % (el, ent.get("_pos", 0)))
        else:
            if data_dir is None:
                data_dir = _job_data_dir(root, ent); st["data_dir"] = data_dir
                with open(_state_path(args.round), "w") as f:
                    json.dump(st, f, indent=2)
            if where == "history":
                print("  [%ds] FINISHED state=%s status=%s (%d shots)"
                      % (el, ent.get("state"), ent.get("status"), ent.get("seq_num") or -1))
                break
            prog = ent.get("seq_num")
            if prog is not None:
                if prog != last_prog:
                    last_prog, last_change = prog, time.time()
                print("  [%ds] RUNNING  progress=%d/%d (%.0f%%)" % (el, prog, total, 100.0 * prog / max(total, 1)))
            if time.time() - last_change > 240:
                print("  STALL 240 s; analyzing partial."); break
        if time.time() - t0 > args.timeout:
            print("  TIMEOUT; analyzing partial."); break
        time.sleep(10)
    time.sleep(8)  # let the runner flush the final HDF5 + json sidecar
    analyze(data_dir, st)


def analyze(data_dir, st):
    sys.path.insert(0, REPO)
    import numpy as np
    from yb_analysis.analysis.run_analysis import analyze_scan_dir

    if not data_dir or not os.path.isdir(data_dir):
        print("ANALYZE: no data dir (%r) -- cannot analyze." % data_dir); return

    res = None
    for attempt in range(6):
        try:
            res = analyze_scan_dir(data_dir, include_per_iteration=False, sync_slm_diag=False)
            break
        except Exception as ex:
            print("  analyze attempt %d failed: %s" % (attempt, ex)); time.sleep(5)
    if res is None:
        print("ANALYZE: failed."); return

    sweep = res.get("sweep", {})
    cols = sweep.get("cols", [])
    values = sweep.get("values", [])
    dims = sweep.get("dims", [])
    summ = res.get("summary", {})
    surv = np.asarray(summ.get("survival_mean", []), dtype=float)
    sem = np.asarray(summ.get("survival_sem", []), dtype=float) if summ.get("survival_sem") else None
    load = np.asarray(summ.get("loading_rate", []), dtype=float)
    n_shots = res.get("n_shots")
    print("=" * 70)
    beam = st.get("beam", "h")
    xp, hp = st.get("x_pin", [0.11e6, 0.16]), st.get("h_pin", [0.11e6, 0.14])
    pin = ("h@(%.4fMHz,%.3f)" % (hp[0] / 1e6, hp[1])) if beam == "X" \
          else ("X@(%.4fMHz,%.3f)" % (xp[0] / 1e6, xp[1]))
    print("ANALYSIS round %d  beam=%s pinned=%s  scan_id=%s  n_shots=%s"
          % (st["round"], beam, pin, res.get("scan_id"), n_shots))
    print("  cols=%s  dims=%s" % (cols, dims))
    print("  axis0 values=%s" % (np.asarray(values[0]).tolist() if len(values) > 0 else None))
    if len(values) > 1:
        print("  axis1 values=%s" % np.asarray(values[1]).tolist())

    if len(dims) < 2 or dims[0] * dims[1] != surv.size:
        print("  (non-2D or size mismatch) survival=%s" % surv.tolist())
        print("  loading =%s" % load.tolist()); return

    s0, s1 = int(dims[0]), int(dims[1])          # s0=det (dim0, fastest), s1=amp
    det = np.asarray(values[0], dtype=float)      # Hz
    amp = np.asarray(values[1], dtype=float)
    # column-major: flat p -> det_idx = p % s0, amp_idx = p // s0
    S = np.full((s1, s0), np.nan)                 # rows=amp, cols=det
    L = np.full((s1, s0), np.nan)
    for p in range(surv.size):
        i_det, i_amp = p % s0, p // s0
        S[i_amp, i_det] = surv[p]
        L[i_amp, i_det] = load[p] if p < load.size else np.nan

    print("\n  SURVIVAL (rows=amp, cols=det in MHz):")
    hdr = "    amp\\det " + " ".join("%7.3f" % (d / 1e6) for d in det)
    print(hdr)
    for ia in range(s1):
        row = " ".join(("%7.3f" % S[ia, idx]) if np.isfinite(S[ia, idx]) else "    nan"
                        for idx in range(s0))
        print("    %6.3f  %s" % (amp[ia], row))
    print("\n  LOADING (rows=amp, cols=det):")
    for ia in range(s1):
        row = " ".join(("%7.3f" % L[ia, idx]) if np.isfinite(L[ia, idx]) else "    nan"
                        for idx in range(s0))
        print("    %6.3f  %s" % (amp[ia], row))

    # peak: max survival among points with adequate loading
    mask = np.isfinite(surv) & (load >= 0.10) if load.size == surv.size else np.isfinite(surv)
    if not mask.any():
        print("\n  NO points with loading>=0.10 -- apparatus likely not loading. max loading=%.3f"
              % (np.nanmax(load) if load.size else float("nan"))); return
    cand = np.where(mask, surv, -np.inf)
    p_star = int(np.argmax(cand))
    i_det, i_amp = p_star % s0, p_star // s0
    semv = sem[p_star] if sem is not None and p_star < sem.size else float("nan")
    print("\n  PEAK: survival=%.3f%s  at det=%.4f MHz  amp=%.3f  (loading=%.3f)"
          % (surv[p_star], (" +/- %.3f" % semv) if np.isfinite(semv) else "",
             det[i_det] / 1e6, amp[i_amp], load[p_star]))
    print("  PEAK_JSON " + json.dumps({"round": st["round"], "beam": beam,
          "det_mhz": det[i_det] / 1e6, "amp": float(amp[i_amp]), "survival": float(surv[p_star]),
          "loading": float(load[p_star]), "mean_loading": float(np.nanmean(load)),
          "n_shots": n_shots}))
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "watch"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"))
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--beam", choices=["X", "h"], default="h")
    ap.add_argument("--xpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None)
    ap.add_argument("--hpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None)
    ap.add_argument("--rnr-time", type=float, default=50e-6, help="ReleaseRecapture.Time (s)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--desc", type=str, default=None,
                    help="run description (purpose/context/operating point) stamped into the scan sidecar")
    # optional per-array loading phase + detection pattern (None = default behavior)
    ap.add_argument("--loading-phase", default=None,
                    help="server WGS phase path, e.g. phase/2x15x15_xyoffset_5um.pt")
    ap.add_argument("--pattern-name", default=None, help="detection pattern / ByPattern key")
    ap.add_argument("--defocus", type=float, default=-5.0, help="ANSI z4 loading defocus")
    ap.add_argument("--planes", type=float, nargs="+", default=None,
                    help="planes_z_rad for 3-D per-layer extraction (e.g. -0.768 0.768)")
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    else:
        do_watch(a)
