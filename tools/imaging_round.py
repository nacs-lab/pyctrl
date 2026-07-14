"""imaging_round.py -- drive ONE round of imaging optimization (fidelity + survival).

seq = ImagingPushoutSurvivalSeq (image1 -> cool -> 399 PUSHOUT/strobe (200 ms) -> cool -> image2,
NumImages=2; switched from StrobeImagingSurvivalSeq 2026-06-20 per "always use CoolingScan"). The two
survival images (Imag399Step) AND the pushout (PushouthXStep) are driven at the test imaging
condition, so BOTH metrics respond to the swept knob. (369 shutter stays CLOSED during the pushout --
PushouthXStep -- confirmed irrelevant to the 399 imaging here, 2026-06-20.)

  * imaging FIDELITY -- from img1: per-cell pool the masked-site intensities, split by
    logicals_img1 (atom vs empty); report dist (mu_atom-mu_empty ADU), d' and infidelity
    (optimal-cut, threshold-free) -> fidelity = 1-infid.  (math copied from
    _feedback47x47/imag2d_fidelity.py)
  * SURVIVAL -- from img2: per-site P(img2=1 | img1=1) = prob11, averaged over sites. ALWAYS run at
    0 pushout (hold ~0) so this is the REAL 50 ms two-image survival at the swept frames, given current
    cooling. The long-pushout proxy + per-image-survival (survpi/surv1) models were REMOVED 2026-06-20
    (inaccurate per user) -- measure real survival at 0 pushout, never model an amplified hold.

Two modes:
  detuning : 1-D sweep of Imag399.FreqDetuning (MHz). The strobe Blue.Freq is LINKED to the
             same detuning (PushouthXStep's default reads BASE Consts(), not the warm4
             overlay, so it would otherwise stay at the old line). Amps held at 0.30/0.20.
  amps     : 2-D sweep of Imag399.Amp1 (dim1) x Imag399.Amp2 (dim2). The strobe Blue.Amp1/Amp2
             are LINKED to the same axes. Detuning held at --det.

All other imaging/cooling values are resolved against the 47x47_feedbackwarm4 ByPattern overlay
at BUILD time (the just-optimized 556 cooling) and pinned explicitly so the strobe doesn't fall
back to bare base. Loading pattern + detection = 47x47_feedbackwarm4.

Two phases (run under the yb_analysis env -- zmq + numpy + h5py + yb_analysis pkg):
  submit : record start seq_num, build + submit_descriptor (NO engine), confirm seq_num
           advances, find data dir, write state.
  watch  : poll get_seq_num to target/stall, then analyze img1 fidelity + img2 survival.
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
# Default array overlay + loading hologram. Override per-run with --pattern / --loading-phase
# (always pass them explicitly for a campaign). Update these when the live array changes.
PATTERN = "33x33_feedback11"
LOADING_PHASE = "phase/33x33_feedback11.pt"
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


def _warm4_consts():
    """expConfig consts with the active ByPattern overlay applied (build-time)."""
    _pyctrl_path()
    from seq_config import SeqConfig
    import expConfig_helper
    from dyn_props import DynProps
    if not SeqConfig.get().consts:
        SeqConfig.load_real()
    base = SeqConfig.get().consts
    return DynProps(expConfig_helper.apply_pattern(base, PATTERN))


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
    return os.path.join(STATE_DIR, "imaging_state_r%d.json" % rnd)


def _pin_strobe_and_cooling(g, c, blue_amp, blue_amp2):
    """Pin the strobe amps + the during-image/strobe 556 cooling to the warm4 values."""
    reson556 = float(c.Resonance556mj0Freq)
    g().Pushout.Time = 0.2
    g().Pushout.Blue.Amp1 = float(blue_amp)
    g().Pushout.Blue.Amp2 = float(blue_amp2)
    g().Pushout.Green.X.Freq = reson556 + float(c.Imag399.Cool556.X.FreqDetuning)
    g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
    g().Pushout.Green.h.Freq = reson556 + float(c.Imag399.Cool556.h.FreqDetuning)
    g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)


def build(args):
    _pyctrl_path()
    from scan_group import ScanGroup
    c = _warm4_consts()
    reson399 = float(c.Resonance399Freq)
    g = ScanGroup()

    if args.mode == "detuning":
        dets = _colon(*args.fdet)                       # MHz
        dets_hz = [float(d) * 1e6 for d in dets]
        # imaging FRAMES detuning (Imag399Step reads g.FreqDetuning) ...
        g().Imag399.FreqDetuning.scan(1, dets_hz)
        # ... and the strobe Blue.Freq LINKED on the SAME axis (else it stays at base line).
        g().Pushout.Blue.Freq.scan(1, [reson399 + d for d in dets_hz])
        _pin_strobe_and_cooling(g, c, args.blue_amp, args.blue_amp2)
        axis_desc = "Imag399.FreqDetuning(MHz)=%s amps=%.2f/%.2f" % (
            [round(d, 3) for d in dets], args.blue_amp, args.blue_amp2)
    elif args.mode == "amps":
        a1 = [float(a) for a in _colon(*args.amp1)]
        a2 = [float(a) for a in _colon(*args.amp2)]
        det_hz = float(args.det) * 1e6
        # imaging FRAMES amps (Imag399Step reads g.Amp1/Amp2) ...
        g().Imag399.Amp1.scan(1, a1)
        g().Imag399.Amp2.scan(2, a2)
        # ... strobe amps LINKED on the same axes; freq fixed at the chosen detuning.
        g().Pushout.Blue.Amp1.scan(1, a1)
        g().Pushout.Blue.Amp2.scan(2, a2)
        g().Imag399.FreqDetuning = det_hz
        g().Pushout.Blue.Freq = reson399 + det_hz
        reson556 = float(c.Resonance556mj0Freq)
        # cooling-during-imaging: --xcool/--hcool (det MHz, amp) override Imag399.Cool556 via g() so the
        # 0-pushout scan can test a NEW cooling WITHOUT a config write (e.g. the X det 0.16 from R3).
        # Default = the loading PATTERN's current Imag399.Cool556 (warm4 ByPattern overlay).
        xcd = float(args.xcool[0]) * 1e6 if args.xcool else float(c.Imag399.Cool556.X.FreqDetuning)
        xca = float(args.xcool[1]) if args.xcool else float(c.Imag399.Cool556.X.Amp)
        hcd = float(args.hcool[0]) * 1e6 if args.hcool else float(c.Imag399.Cool556.h.FreqDetuning)
        hca = float(args.hcool[1]) if args.hcool else float(c.Imag399.Cool556.h.Amp)
        # IMAGE-step cooling (what matters at 0 pushout): override Imag399.Cool556 directly.
        g().Imag399.Cool556.X.FreqDetuning = xcd
        g().Imag399.Cool556.X.Amp = xca
        g().Imag399.Cool556.h.FreqDetuning = hcd
        g().Imag399.Cool556.h.Amp = hca
        g().Pushout.Time = float(args.hold)
        # pushout proxy cooling (irrelevant at hold~0; set consistently):
        g().Pushout.Green.X.Freq = reson556 + xcd
        g().Pushout.Green.X.Amp = xca
        g().Pushout.Green.h.Freq = reson556 + hcd
        g().Pushout.Green.h.Amp = hca
        axis_desc = "Imag399.Amp1=%s x Amp2=%s det=%.2fMHz hold=%.3fs cool X(%.2fMHz,%.2f) h(%.2fMHz,%.2f)" % (
            [round(a, 3) for a in a1], [round(a, 3) for a in a2], args.det, args.hold,
            xcd / 1e6, xca, hcd / 1e6, hca)
    elif args.mode == "ratio":  # 1-D magnitude sweep along a FIXED Amp1:Amp2 ratio (default 3:2)
        a1 = [float(a) for a in _colon(*args.amp1)]
        a2 = [round(a * args.amp2_ratio, 6) for a in a1]   # beam2 = beam1 * ratio (keep 3:2)
        det_hz = float(args.det) * 1e6
        # both imaging-frame amps swept TOGETHER on axis 1 (so the ratio is held)...
        g().Imag399.Amp1.scan(1, a1)
        g().Imag399.Amp2.scan(1, a2)
        # ... strobe amps linked on the same axis (matters only if hold>0).
        g().Pushout.Blue.Amp1.scan(1, a1)
        g().Pushout.Blue.Amp2.scan(1, a2)
        g().Imag399.FreqDetuning = det_hz
        g().Pushout.Blue.Freq = reson399 + det_hz
        reson556 = float(c.Resonance556mj0Freq)
        g().Pushout.Time = float(args.hold)   # small (e.g. 0.001) -> CLEAN single-shot survival
        g().Pushout.Green.X.Freq = reson556 + float(c.Imag399.Cool556.X.FreqDetuning)
        g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
        g().Pushout.Green.h.Freq = reson556 + float(c.Imag399.Cool556.h.FreqDetuning)
        g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)
        axis_desc = "ratio Amp1=%s Amp2(x%.4f)=%s det=%.2fMHz hold=%.3fs" % (
            [round(a, 3) for a in a1], args.amp2_ratio, [round(a, 3) for a in a2], args.det, args.hold)
    elif args.mode == "cool":
        # 0-PUSHOUT cooling scan: sweep Imag399.Cool556.{beam}.{FreqDetuning, Amp} at the REAL 50 ms
        # image (hold ~0), amps + the OTHER beam's cooling FIXED. Survival is the real readout (low
        # contrast -> use high --reps). Compares cooling settings DRIFT-FREE in one scan.
        beam = args.beam
        dets = _colon(*args.cdet); dets_hz = [float(d) * 1e6 for d in dets]
        cam = [float(a) for a in _colon(*args.famp)]
        # NEW 399-imaging scheme: DDS Amp1/Amp2 held at 1; the actual imaging power is set
        # by the BlueMOT PID setpoints Img1PIDSet/Img2PIDSet (V). --img1/--img2 carry W here.
        g().Imag399.Amp1 = 1
        g().Imag399.Amp2 = 1
        g().BlueMOT.Img1PIDSet = float(args.img1)
        g().BlueMOT.Img2PIDSet = float(args.img2)
        g().Imag399.FreqDetuning = float(args.det) * 1e6
        g().Pushout.Time = float(args.hold)            # 0 pushout (real 50 ms survival)
        if beam == "X":
            hcd = float(args.hcool[0]) * 1e6 if args.hcool else float(c.Imag399.Cool556.h.FreqDetuning)
            hca = float(args.hcool[1]) if args.hcool else float(c.Imag399.Cool556.h.Amp)
            g().Imag399.Cool556.h.FreqDetuning = hcd
            g().Imag399.Cool556.h.Amp = hca
            g().Imag399.Cool556.X.FreqDetuning.scan(1, dets_hz)
            g().Imag399.Cool556.X.Amp.scan(2, cam)
            fixed = "h(%.2fMHz,%.2f)" % (hcd / 1e6, hca)
        else:
            xcd = float(args.xcool[0]) * 1e6 if args.xcool else float(c.Imag399.Cool556.X.FreqDetuning)
            xca = float(args.xcool[1]) if args.xcool else float(c.Imag399.Cool556.X.Amp)
            g().Imag399.Cool556.X.FreqDetuning = xcd
            g().Imag399.Cool556.X.Amp = xca
            g().Imag399.Cool556.h.FreqDetuning.scan(1, dets_hz)
            g().Imag399.Cool556.h.Amp.scan(2, cam)
            fixed = "X(%.2fMHz,%.2f)" % (xcd / 1e6, xca)
        axis_desc = "COOL %s: det(MHz)=%s x amp=%s  PIDset %.2f/%.2f det %.2f fixed %s hold %.4fs" % (
            beam, [round(d, 3) for d in dets], [round(a, 3) for a in cam],
            args.img1, args.img2, args.det, fixed, args.hold)
    else:  # pushout: img1/img2 detection FIXED (good), sweep the PUSHOUT imaging dose (amp1 x amp2).
        # Decouples survival from detection: prob11 differences across the pushout grid are REAL atom
        # loss (the detection floor is constant). Per-image survival = (prob11/floor)^(t_img/hold).
        a1 = [float(a) for a in _colon(*args.amp1)]
        a2 = [float(a) for a in _colon(*args.amp2)]
        det_hz = float(args.det) * 1e6
        # FIXED detection frames (the readout) at a known-good, high-fidelity setting:
        g().Imag399.Amp1 = float(args.frame_amp1)
        g().Imag399.Amp2 = float(args.frame_amp2)
        g().Imag399.FreqDetuning = det_hz
        # SWEPT pushout imaging dose (the thing under test):
        g().Pushout.Blue.Amp1.scan(1, a1)
        g().Pushout.Blue.Amp2.scan(2, a2)
        g().Pushout.Blue.Freq = reson399 + det_hz
        reson556 = float(c.Resonance556mj0Freq)
        g().Pushout.Time = float(args.hold)   # the imaging-dose duration (e.g. 0.2 s)
        g().Pushout.Green.X.Freq = reson556 + float(c.Imag399.Cool556.X.FreqDetuning)
        g().Pushout.Green.X.Amp = float(c.Imag399.Cool556.X.Amp)
        g().Pushout.Green.h.Freq = reson556 + float(c.Imag399.Cool556.h.FreqDetuning)
        g().Pushout.Green.h.Amp = float(c.Imag399.Cool556.h.Amp)
        axis_desc = "PUSHOUT Amp1=%s x Amp2=%s (frames FIXED %.3f/%.3f) det=%.2fMHz hold=%.3fs" % (
            [round(a, 3) for a in a1], [round(a, 3) for a in a2],
            args.frame_amp1, args.frame_amp2, args.det, args.hold)

    rp = g.runp()
    rp.NumImages = 2
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
    if getattr(args, "loading_phase", None):
        LOADING_PHASE = args.loading_phase
    if getattr(args, "pattern", None):
        PATTERN = args.pattern
    root = _data_root(); pre = _data_dirs(root)
    start = _zmq_int("get_seq_num")
    if start is None:
        print("ERROR: backend not answering get_seq_num at", URL); sys.exit(2)
    g, axis_desc = build(args)
    nseq = g.nseq()
    lbl = "ImagingScan_%s_r%d" % (args.mode, args.round)
    desc = scangroup_to_descriptor(g, "ImagingPushoutSurvivalSeq", opts={"rep": args.reps}, label=lbl)
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d (%s): id=%d nseq=%d reps=%d -> target=%d  start_seq=%d"
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
            print("  ALIVE: seq_num %d -> %d (+%d) after %ds; data_dir=%s"
                  % (last, cur, cur - last, int(time.time() - t0), data_dir))
            if cur >= start + 2 and data_dir:
                break
            last = cur
    else:
        print("  WARNING: seq_num did not advance enough in 120 s (cur=%s)." % _zmq_int("get_seq_num"))

    st = {"round": args.round, "mode": args.mode, "start": start, "target": target, "nseq": nseq,
          "reps": args.reps, "data_dir": data_dir, "id": did, "axis_desc": axis_desc}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


def _optimal_infidelity(i_empty, i_atom):
    import numpy as np
    if len(i_empty) < 5 or len(i_atom) < 5:
        return float("nan")
    lo = min(i_empty.min(), i_atom.min()); hi = max(i_empty.max(), i_atom.max())
    ts = np.linspace(lo, hi, 400)
    e = np.sort(i_empty); a = np.sort(i_atom)
    fp = 1.0 - np.searchsorted(e, ts, side="right") / e.size
    fn = np.searchsorted(a, ts, side="left") / a.size
    return float(np.min(0.5 * (fp + fn)))


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
        I1 = f["intensities_img1"][:]
        L1 = f["logicals_img1"][:].astype(bool)
        L2 = f["logicals_img2"][:].astype(bool) if "logicals_img2" in f else None
    n = min(len(seq_ids), I1.shape[0])
    seq_ids, I1, L1 = seq_ids[:n], I1[:n], L1[:n]
    if L2 is not None:
        L2 = L2[:n]
    flat = P[seq_ids - 1] - 1
    ncell = int(np.prod([d["size"] for d in dims]))

    print("=" * 74)
    print("IMAGING round %d (%s)  scan_id=%s  n_shots=%d  cells=%d"
          % (st["round"], st["mode"], sid.replace("data_", ""), n, ncell))
    print("  " + st.get("axis_desc", ""))

    rows_all = []
    for p in range(ncell):
        rows = np.where(flat == p)[0]
        if rows.size == 0:
            rows_all.append(None); continue
        I = I1[rows].ravel(); L = L1[rows].ravel()
        i0, i1 = I[~L], I[L]
        rec = {"p": p, "nshot": int(rows.size), "load": float(L.mean())}
        if i0.size and i1.size:
            m0, s0_, m1, s1_ = i0.mean(), i0.std(), i1.mean(), i1.std()
            denom = np.sqrt((s0_**2 + s1_**2) / 2)
            rec["dist"] = float(m1 - m0)
            rec["dprime"] = float((m1 - m0) / denom) if denom > 0 else float("nan")
            rec["fidelity"] = 1.0 - _optimal_infidelity(i0, i1)
        else:
            rec["dist"] = rec["dprime"] = rec["fidelity"] = float("nan")
        if L2 is not None:
            l1 = L1[rows]; l2 = L2[rows]
            loaded = l1.sum(axis=0); joint = (l1 & l2).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                p11 = np.where(loaded > 0, joint / np.maximum(loaded, 1), np.nan)
            rec["survival"] = float(np.nanmean(p11))
        else:
            rec["survival"] = float("nan")
        rows_all.append(rec)

    # Always run at 0 pushout (--hold ~0): 'survival' (prob11) is then the real per-shot 50 ms
    # two-image survival at the swept frames. (The detuning double-application + surv1 de-magnification
    # model was removed 2026-06-20 -- measure at 0 pushout instead of modelling an amplified hold.)

    if len(dims) == 1:
        vals = np.asarray(dims[0]["values"], float)
        unit = 1e-6 if st["mode"] == "detuning" else 1.0
        lab = "det(MHz)" if st["mode"] == "detuning" else "amp1"
        print("  NOTE: survival = real per-shot 50 ms two-image survival (run with hold ~0). "
              "Fidelity is from img1 at the swept value.")
        print("\n  %9s %8s %8s %8s %9s %6s"
              % (lab, "fidelity", "dprime", "dist", "survival", "load"))
        for p, rec in enumerate(rows_all):
            if rec is None:
                continue
            print("  %9.3f %8.4f %8.3f %8.1f %9.4f %6.3f"
                  % (vals[p] * unit, rec["fidelity"], rec["dprime"], rec["dist"],
                     rec["survival"], rec["load"]))
        _pick(rows_all, vals, unit, lab)
    else:
        s0, s1 = dims[0]["size"], dims[1]["size"]
        a1v = np.asarray(dims[0]["values"], float); a2v = np.asarray(dims[1]["values"], float)
        # Real survival only. Run the imaging-amp scan at 0 pushout (--hold ~0) so 'survival' (prob11)
        # IS the real 50 ms two-image survival at the swept frames, given current cooling. The
        # long-pushout proxy + per-image-survival (survpi) model was REMOVED 2026-06-20 (inaccurate).
        metrics = ["fidelity", "survival"]
        for metric in metrics:
            lab2 = {"survpi": "PER-IMAGE SURVIVAL (decoupled)"}.get(metric, metric.upper())
            print("\n  %s (rows=Amp2, cols=Amp1):" % lab2)
            print("    A2\\A1 " + " ".join("%7.3f" % a for a in a1v))
            for j in range(s1):
                cells = [rows_all[j * s0 + i] for i in range(s0)]
                row = " ".join(("%7.4f" % cc[metric]) if cc and np.isfinite(cc.get(metric, float("nan")))
                               else "    nan" for cc in cells)
                print("    %5.3f %s" % (a2v[j], row))
        _pick2d(rows_all, a1v, a2v, s0, s1)
    print("=" * 74)


def _joint_pick(recs):
    """Among cells with loading>=0.10, report best fidelity, best survival, and a balanced pick
    (max survival among cells within 0.002 fidelity of the max fidelity)."""
    import numpy as np
    cand = [r for r in recs if r and r.get("load", 0) >= 0.10
            and np.isfinite(r.get("fidelity", np.nan)) and np.isfinite(r.get("survival", np.nan))]
    if not cand:
        return None, None, None
    best_f = max(cand, key=lambda r: r["fidelity"])
    best_s = max(cand, key=lambda r: r["survival"])
    fmax = best_f["fidelity"]
    near = [r for r in cand if r["fidelity"] >= fmax - 0.002]
    balanced = max(near, key=lambda r: r["survival"])
    return best_f, best_s, balanced


def _pick(recs, vals, unit, lab):
    bf, bs, bal = _joint_pick(recs)
    if bf is None:
        print("\n  NO cells with loading>=0.10."); return
    def show(tag, r):
        print("  %-18s %s=%.3f  fidelity=%.4f  dprime=%.2f  survival=%.4f  load=%.3f"
              % (tag, lab, vals[r["p"]] * unit, r["fidelity"], r["dprime"], r["survival"], r["load"]))
    print()
    show("BEST FIDELITY:", bf)
    show("BEST SURVIVAL:", bs)
    show("BALANCED (rec):", bal)
    print("  PICK_JSON " + json.dumps({"best_fid_x": vals[bf["p"]] * unit, "best_fid": bf["fidelity"],
          "best_surv_x": vals[bs["p"]] * unit, "best_surv": bs["survival"],
          "balanced_x": vals[bal["p"]] * unit, "balanced_fid": bal["fidelity"],
          "balanced_surv": bal["survival"]}))


def _pick2d(recs, a1v, a2v, s0, s1):
    bf, bs, bal = _joint_pick(recs)
    if bf is None:
        print("\n  NO cells with loading>=0.10."); return
    def xy(r):
        return a1v[r["p"] % s0], a2v[r["p"] // s0]
    def show(tag, r):
        a1, a2 = xy(r)
        print("  %-18s Amp1=%.3f Amp2=%.3f  fidelity=%.4f  dprime=%.2f  survival=%.4f  load=%.3f"
              % (tag, a1, a2, r["fidelity"], r["dprime"], r["survival"], r["load"]))
    print()
    show("BEST FIDELITY:", bf)
    show("BEST SURVIVAL:", bs)
    show("BALANCED (rec):", bal)
    a1b, a2b = xy(bal)
    print("  PICK_JSON " + json.dumps({"balanced_amp1": a1b, "balanced_amp2": a2b,
          "balanced_fid": bal["fidelity"], "balanced_surv": bal["survival"]}))


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    target, start = st["target"], st["start"]
    data_dir = st["data_dir"]
    print("watching round %d: start=%d target=%d data_dir=%s" % (args.round, start, target, data_dir))
    t0 = time.time(); last, last_change = start, time.time()
    while True:
        cur = _zmq_int("get_seq_num"); el = int(time.time() - t0)
        if cur is None:
            print("  [%ds] get_seq_num TIMEOUT" % el)
        else:
            if cur != last:
                last, last_change = cur, time.time()
            done = cur - start
            print("  [%ds] seq_num=%d  progress=%d/%d (%.0f%%)"
                  % (el, cur, done, target - start, 100.0 * done / max(target - start, 1)))
            if cur >= target:
                print("  COMPLETE"); break
            if time.time() - last_change > 240:
                print("  STALL 240 s; analyzing partial."); break
        if time.time() - t0 > args.timeout:
            print("  TIMEOUT; analyzing partial."); break
        time.sleep(10)
    time.sleep(8)
    analyze(data_dir, st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "watch"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--mode", choices=["detuning", "amps", "ratio", "pushout", "cool"], default="detuning")
    ap.add_argument("--beam", choices=["X", "h"], default="X", help="cool mode: which 556 beam to sweep")
    ap.add_argument("--cdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.12, 0.02, 0.24),
                    help="cool mode: Imag399.Cool556.{beam}.FreqDetuning colon in MHz "
                         "(2x-finer default 2026-06-24: step 0.02; brackets X 0.158 & h 0.14)")
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.18, 0.02, 0.30),
                    help="cool mode: Imag399.Cool556.{beam}.Amp colon "
                         "(2x-finer default 2026-06-24: step 0.02; X 0.267 & h 0.24 sit mid-grid)")
    ap.add_argument("--amp2-ratio", type=float, default=2.0 / 3.0,
                    help="ratio mode: Amp2 = Amp1 * this (default 2/3 = the 0.30:0.20 = 3:2 ratio)")
    ap.add_argument("--hold", type=float, default=0.001,
                    help="Pushout.Time (s); DEFAULT 0.001 = 0 pushout (real 50 ms survival). Always keep ~0.")
    ap.add_argument("--frame-amp1", type=float, default=0.35,
                    help="pushout mode: FIXED Imag399.Amp1 for the detection frames (good readout)")
    ap.add_argument("--frame-amp2", type=float, default=0.2333,
                    help="pushout mode: FIXED Imag399.Amp2 for the detection frames")
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(-9.0, 0.5, -1.0),
                    help="detuning mode: Imag399.FreqDetuning colon in MHz "
                         "(2x-finer default 2026-06-24: step 0.5 MHz)")
    ap.add_argument("--blue-amp", type=float, default=0.30, help="detuning mode: fixed Imag399.Amp1")
    ap.add_argument("--blue-amp2", type=float, default=0.20, help="detuning mode: fixed Imag399.Amp2")
    ap.add_argument("--img1", type=float, default=1.0,
                    help="cool mode: BlueMOT.Img1PIDSet (399 beam-1 imaging-power setpoint, V) = W")
    ap.add_argument("--img2", type=float, default=0.35,
                    help="cool mode: BlueMOT.Img2PIDSet (399 beam-2 imaging-power setpoint, V) = W")
    ap.add_argument("--amp1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.16, 0.015, 0.30),
                    help="amps mode: Imag399.Amp1 colon "
                         "(2x-finer default 2026-06-24: step 0.015, span recentred on the 0.16-0.30 "
                         "plateau->heating-cliff; drop the dead 0.30-0.38 zone)")
    ap.add_argument("--amp2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.015, 0.22),
                    help="amps mode: Imag399.Amp2 colon "
                         "(2x-finer default 2026-06-24: step 0.015; Amp2 weak lever, narrow span around 0.12)")
    ap.add_argument("--det", type=float, default=-5.0, help="amps mode: fixed Imag399.FreqDetuning (MHz)")
    ap.add_argument("--xcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="amps mode: override Imag399.Cool556.X (det MHz, amp) via g() (default: pattern overlay)")
    ap.add_argument("--hcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="amps mode: override Imag399.Cool556.h (det MHz, amp) via g() (default: pattern overlay)")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--defocus", type=float, default=-5.0)
    ap.add_argument("--loading-phase", type=str, default=None,
                    help="override LOADING_PHASE (e.g. phase/33x33_feedback1.pt) to measure another version")
    ap.add_argument("--pattern", type=str, default=None,
                    help="override PATTERN (ByPattern overlay key, e.g. 33x33_feedback1)")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    else:
        do_watch(a)
