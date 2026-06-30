"""cooling_img_round.py -- drive ONE round of the CoolingScan (imaging-during-pushout) 2-D
optimization. Imaging-cooling analog of cooling_round.py (which is RNR-only).

seq = ImagingPushoutSurvivalSeq (NumImages=2 survival = how well 556 X+h keep the atom alive
during a Pushout.Time imaging hold). Sweeps Pushout.Green.{X|h}.{Freq,Amp}; Freq = Resonance556mj0
+ detuning. The tuned values map to Imag399.Cool556.

Two phases (run under the yb_analysis env -- zmq + numpy + h5py + yb_analysis pkg):
  submit : record start seq_num, build the grid via CoolingScan.build_2d + submit_descriptor (NO
           engine import), confirm the shot counter advances, find the new data dir, write state.
  watch  : poll get_seq_num to target (R*nseq) or stall, then analyze_scan_dir -> survival(freq,amp)
           map + peak (MATLAB column-major). Freq printed as DETUNING (freq - Resonance556mj0).

Pins are given as (det_MHz, amp); converted to the absolute fix_freq build_2d wants.
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

DEF_X_PIN = (0.16, 0.20)   # (det MHz, amp) -- 47x47_feedbackwarm4 seeded Imag399.Cool556.X
DEF_H_PIN = (0.16, 0.14)   # (det MHz, amp) -- 47x47_feedbackwarm4 seeded Imag399.Cool556.h

# Two 399 imaging beams held at the warm4 pattern defaults during the hold (Imag399.Amp1/Amp2).
# beam 1 -> Pushout.Blue.Amp (AmpAbsImag/DDS18), beam 2 -> Pushout.Blue.Amp2 (Amp399Imag2/DDS17).
DEF_BLUE_AMP = 0.30        # Imag399.Amp1
DEF_BLUE_AMP2 = 0.20       # Imag399.Amp2


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


def _reson():
    _pyctrl_path()
    import CoolingScan
    return float(CoolingScan._consts().Resonance556mj0Freq)


def _data_root():
    sys.path.insert(0, REPO)
    from yb_analysis import config
    return config.DATA_DIR


def _data_dirs(root):
    import glob
    return set(glob.glob(os.path.join(root, "*", "data_*")))


def _state_path(rnd):
    return os.path.join(STATE_DIR, "cooling_img_state_r%d.json" % rnd)


def _colon(lo, step, hi):
    _pyctrl_path()
    from scan_export import matlab_colon
    return matlab_colon(lo, step, hi)


def do_submit(args):
    _pyctrl_path()
    from scan_export import scangroup_to_descriptor
    from yb_start_scan import submit_descriptor
    import CoolingScan

    root = _data_root()
    pre = _data_dirs(root)
    reson = _reson()

    start = _zmq_int("get_seq_num")
    if start is None:
        print("ERROR: backend not answering get_seq_num at", URL); sys.exit(2)
    print("backend start seq_num=%d  Resonance556mj0=%.4f MHz" % (start, reson / 1e6))

    x_pin = tuple(args.xpin) if args.xpin else DEF_X_PIN
    h_pin = tuple(args.hpin) if args.hpin else DEF_H_PIN
    pin = h_pin if args.beam == "X" else x_pin       # the NON-swept beam
    fix_freq = reson + pin[0] * 1e6
    fix_amp = pin[1]

    det = list(_colon(*args.fdet))                   # detunings in MHz
    amp = list(_colon(*args.famp))
    blue_det_hz = None if args.blue_det is None else float(args.blue_det) * 1e6
    g = CoolingScan.build_2d(args.beam, args.blue_amp, det, amp,
                             fixed_freq=fix_freq, fixed_amp=fix_amp, time_s=args.time,
                             blue_amp2=args.blue_amp2, blue_det_hz=blue_det_hz)

    # --- optional per-array loading phase + detection pattern (e.g. the 2x15x15 two-layer array).
    # Setting loading_phase makes SlmScanSession write that hologram once at scan start AND apply
    # the matching expConfig ByPattern overlay (VSLMServo etc). Default (None) = unchanged 47x47
    # behavior. Detection uses the named pattern's cached registry grid.
    if args.loading_phase:
        rp = g.runp()
        rp.loading_phase = args.loading_phase
        rp.loading_defocus = float(args.defocus)
        rp.useScanLongSlmLock = 1
        # 2D arrays: let the runner AUTO-DERIVE the detection pattern from loading_phase (the proven
        # path -- matches imaging_round and warm4 rounds 1-2). Only force imagePatternsJson for 3D
        # (--planes), where the runner can't infer the per-layer z-splitting.
        if args.planes:
            pat = {"name": args.pattern_name, "base_phase_path": args.loading_phase,
                   "order": "col", "legacy_zerniked": False,
                   "planes_z_rad": [float(z) for z in args.planes]}
            rp.imagePatternsJson = json.dumps([pat, pat])   # 2 frames (NumImages=2)
        print("  loading_phase=%s defocus=%g pattern=%s planes=%s"
              % (args.loading_phase, args.defocus, args.pattern_name, args.planes))

    nseq = g.nseq()
    lbl = "CoolingScan_img_%s_r%d" % (args.beam, args.round)
    desc = scangroup_to_descriptor(g, "ImagingPushoutSurvivalSeq", opts={"rep": args.reps}, label=lbl)
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d: id=%d beam=%s nseq=%d reps=%d blue=%.2f/%.2f time=%.2fs pin(%s)=(%.3fMHz,%.3f) -> target=%d"
          % (args.round, did, args.beam, nseq, args.reps, args.blue_amp, args.blue_amp2, args.time,
             "h" if args.beam == "X" else "X", pin[0], pin[1], target))
    print("  %s det(MHz)=%s" % (args.beam, [round(d, 4) for d in det]))
    print("  %s amp     =%s" % (args.beam, [round(a, 4) for a in amp]))

    data_dir = None
    t0 = time.time(); last = start
    while time.time() - t0 < 120:
        time.sleep(5)
        cur = _zmq_int("get_seq_num")
        new = _data_dirs(root) - pre
        if new and data_dir is None:
            data_dir = sorted(new, key=os.path.getmtime)[-1]
        if cur is not None and cur > last:
            print("  ALIVE: seq_num %d -> %d (+%d) after %ds; data_dir=%s"
                  % (last, cur, cur - last, int(time.time() - t0), data_dir))
            if cur >= start + 2 and data_dir:
                break
            last = cur
    else:
        print("  WARNING: seq_num did not advance enough in 120 s (cur=%s)." % _zmq_int("get_seq_num"))

    st = {"round": args.round, "beam": args.beam, "start": start, "target": target, "nseq": nseq,
          "reps": args.reps, "blue_amp": args.blue_amp, "blue_amp2": args.blue_amp2,
          "time": args.time, "reson": reson,
          "x_pin": list(x_pin), "h_pin": list(h_pin), "data_dir": data_dir,
          "fdet": list(args.fdet), "famp": list(args.famp), "id": did}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    target, start, nseq = st["target"], st["start"], st["nseq"]
    data_dir = st["data_dir"]
    print("watching round %d: start=%d target=%d nseq=%d data_dir=%s"
          % (args.round, start, target, nseq, data_dir))

    t0 = time.time(); last, last_change = start, time.time()
    while True:
        cur = _zmq_int("get_seq_num")
        el = int(time.time() - t0)
        if cur is None:
            print("  [%ds] get_seq_num TIMEOUT" % el)
        else:
            if cur != last:
                last, last_change = cur, time.time()
            done = cur - start
            print("  [%ds] seq_num=%d  progress=%d/%d (%.0f%%)"
                  % (el, cur, done, target - start, 100.0 * done / max(target - start, 1)))
            if cur >= target:
                print("  COMPLETE: reached target %d" % target); break
            if time.time() - last_change > 240:
                print("  STALL: seq_num frozen at %d for 240 s. Analyzing partial data." % cur); break
        if time.time() - t0 > args.timeout:
            print("  TIMEOUT after %ds; analyzing whatever exists." % args.timeout); break
        time.sleep(10)

    time.sleep(8)
    analyze(data_dir, st)


def analyze(data_dir, st):
    sys.path.insert(0, REPO)
    import numpy as np
    from yb_analysis.analysis.run_analysis import analyze_scan_dir

    if not data_dir or not os.path.isdir(data_dir):
        print("ANALYZE: no data dir (%r)." % data_dir); return
    res = None
    for attempt in range(6):
        try:
            res = analyze_scan_dir(data_dir, include_per_iteration=False, sync_slm_diag=False)
            break
        except Exception as ex:
            print("  analyze attempt %d failed: %s" % (attempt, ex)); time.sleep(5)
    if res is None:
        print("ANALYZE: failed."); return

    reson = st.get("reson", 107.7673e6)
    sweep = res.get("sweep", {}); cols = sweep.get("cols", [])
    values = sweep.get("values", []); dims = sweep.get("dims", [])
    summ = res.get("summary", {})
    surv = np.asarray(summ.get("survival_mean", []), dtype=float)
    sem = np.asarray(summ.get("survival_sem", []), dtype=float) if summ.get("survival_sem") else None
    load = np.asarray(summ.get("loading_rate", []), dtype=float)
    n_shots = res.get("n_shots")
    beam = st.get("beam", "X")
    xp, hp = st.get("x_pin"), st.get("h_pin")
    pin = ("h@(%.3fMHz,%.3f)" % tuple(hp)) if beam == "X" else ("X@(%.3fMHz,%.3f)" % tuple(xp))
    print("=" * 70)
    print("ANALYSIS round %d  beam=%s pinned=%s  blue=%.2f time=%.2fs  scan_id=%s  n_shots=%s"
          % (st["round"], beam, pin, st.get("blue_amp"), st.get("time"), res.get("scan_id"), n_shots))
    print("  cols=%s  dims=%s" % (cols, dims))

    if len(dims) < 2 or dims[0] * dims[1] != surv.size:
        print("  (non-2D or size mismatch) survival=%s" % surv.tolist())
        print("  loading =%s" % load.tolist()); return

    s0, s1 = int(dims[0]), int(dims[1])              # s0=freq (dim0 fastest), s1=amp
    freq = np.asarray(values[0], dtype=float)        # absolute Hz
    detm = (freq - reson) / 1e6                       # detuning MHz
    amp = np.asarray(values[1], dtype=float)
    S = np.full((s1, s0), np.nan); L = np.full((s1, s0), np.nan)
    for p in range(surv.size):
        i_f, i_a = p % s0, p // s0
        S[i_a, i_f] = surv[p]
        L[i_a, i_f] = load[p] if p < load.size else np.nan

    print("\n  SURVIVAL (rows=amp, cols=556 detuning in MHz):")
    print("    amp\\det " + " ".join("%7.3f" % d for d in detm))
    for ia in range(s1):
        row = " ".join(("%7.3f" % S[ia, j]) if np.isfinite(S[ia, j]) else "    nan" for j in range(s0))
        print("    %6.3f  %s" % (amp[ia], row))
    print("\n  LOADING (rows=amp, cols=det):")
    for ia in range(s1):
        row = " ".join(("%7.3f" % L[ia, j]) if np.isfinite(L[ia, j]) else "    nan" for j in range(s0))
        print("    %6.3f  %s" % (amp[ia], row))

    mask = np.isfinite(surv) & (load >= 0.10) if load.size == surv.size else np.isfinite(surv)
    if not mask.any():
        print("\n  NO points with loading>=0.10. max loading=%.3f"
              % (np.nanmax(load) if load.size else float("nan"))); return
    p_star = int(np.argmax(np.where(mask, surv, -np.inf)))
    i_f, i_a = p_star % s0, p_star // s0
    semv = sem[p_star] if sem is not None and p_star < sem.size else float("nan")
    print("\n  PEAK: survival=%.3f%s  at det=%.4f MHz (%.4f MHz abs)  amp=%.3f  (loading=%.3f)"
          % (surv[p_star], (" +/- %.3f" % semv) if np.isfinite(semv) else "",
             detm[i_f], freq[i_f] / 1e6, amp[i_a], load[p_star]))
    print("  PEAK_JSON " + json.dumps({"round": st["round"], "beam": beam,
          "det_mhz": float(detm[i_f]), "amp": float(amp[i_a]), "survival": float(surv[p_star]),
          "loading": float(load[p_star]), "mean_loading": float(np.nanmean(load)), "n_shots": n_shots}))
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "watch"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--beam", choices=["X", "h"], default="X")
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"))
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"))
    ap.add_argument("--xpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None)
    ap.add_argument("--hpin", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None)
    ap.add_argument("--blue-amp", type=float, default=DEF_BLUE_AMP,
                    help="399 imaging beam 1 amp (Imag399.Amp1 -> Pushout.Blue.Amp); warm4 default 0.30")
    ap.add_argument("--blue-amp2", type=float, default=DEF_BLUE_AMP2,
                    help="399 imaging beam 2 amp (Imag399.Amp2 -> Pushout.Blue.Amp2); warm4 default 0.20")
    ap.add_argument("--blue-det", type=float, default=None,
                    help="399 imaging detuning during the hold in MHz (default: config Imag399.FreqDetuning)")
    ap.add_argument("--time", type=float, default=1.0, help="Pushout.Time (s)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    # optional per-array loading phase + detection pattern (None = default 47x47 behavior)
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
