"""strobe_imaging_round.py -- drive ONE round of STROBE-imaging optimization (fidelity + survival).

Strobe-imaging twin of tools/imaging_round.py. seq = StrobeImagingPushoutSeq (img1 -> cool -> PushouthX
hold -> cool -> img2, NumImages=2; BOTH survival images are StrobeImag399Step -- interleaved 399-image
<-> 556-recool). Measures, per scan cell:

  * imaging FIDELITY -- from img1: per-cell pool masked-site intensities, split by logicals_img1
    (atom vs empty); dist (mu_atom-mu_empty), d', optimal-cut infidelity -> fidelity = 1-infid.
  * SURVIVAL -- from img2: per-site P(img2=1 | img1=1) = prob11, averaged over sites. Run at 0 hold
    (--hold ~0) so this is the REAL two-image strobe survival at the swept condition.

(The analysis is reused verbatim from imaging_round -- identical .h5 schema.)

Modes (the SWEPT axes; BeamPulseTime/RecoolTime are FIXED per round -> the duration is the OUTER loop,
run more rounds to vary it, it can NEVER be a .scan() axis):
  detuning : 1-D Imag399.FreqDetuning (MHz)        -- the 399 imaging line.
  amps     : 2-D Imag399.Amp1 (dim1) x Amp2 (dim2) -- the two 399 imaging-beam amplitudes.
  cool     : 2-D Imag399.Strobe.Cool556.{beam}.{FreqDetuning(dim1,MHz), Amp(dim2)} -- the strobe RECOOL.

Collect window = Orca.ExposureTime (set base/ByPattern); n_cycles = round(ExposureTime/(BeamPulseTime+
RecoolTime)) auto-fills it. WARN: small RecoolTime -> many cycles -> big per-shot sequence; validate with
a short Orca.ExposureTime first.

Two phases (run under the yb_analysis env):
  submit : record start seq_num, build + submit_descriptor (NO engine), confirm seq_num advances, write state.
  watch  : poll get_seq_num to target/stall, then analyze img1 fidelity + img2 survival.

  python tools/strobe_imaging_round.py submit --round 1 --mode amps --pulse-time 800e-9 --recool-time 50e-6
  python tools/strobe_imaging_round.py watch  --round 1
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
PATTERN = "3270_tri"
LOADING_PHASE = "phase/3270_tri.pt"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

sys.path.insert(0, os.path.join(PYCTRL, "tools"))
# Reuse imaging_round's proven helpers + analysis (identical .h5 schema). _pyctrl_path sets sys.path.
from imaging_round import (_pyctrl_path, _zmq_int, _data_root, _data_dirs, analyze)  # noqa: E402


def _state_path(rnd):
    return os.path.join(STATE_DIR, "strobe_imaging_state_r%d.json" % rnd)


def _pattern_consts():
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


def build(args):
    _pyctrl_path()
    from scan_group import ScanGroup
    c = _pattern_consts()
    reson399 = float(c.Resonance399Freq)
    g = ScanGroup()

    # FIXED strobe timing (loop structure -> never a scan axis).
    g().Imag399.Strobe.BeamPulseTime = float(args.pulse_time)
    g().Imag399.Strobe.RecoolTime = float(args.recool_time)
    g().Imag399.Strobe.PulsesPerBurst = int(args.burst)
    g().Pushout.Time = float(args.hold)        # 0 = real survival; PushouthX beams default off

    # Recool tones: --xcool/--hcool (det MHz, amp) override the config strobe recool via g().
    xcd = float(args.xcool[0]) * 1e6 if args.xcool else float(c.Imag399.Strobe.Cool556.X.FreqDetuning)
    xca = float(args.xcool[1]) if args.xcool else float(c.Imag399.Strobe.Cool556.X.Amp)
    hcd = float(args.hcool[0]) * 1e6 if args.hcool else float(c.Imag399.Strobe.Cool556.h.FreqDetuning)
    hca = float(args.hcool[1]) if args.hcool else float(c.Imag399.Strobe.Cool556.h.Amp)

    _duty = 2.0 * args.burst * args.pulse_time / (2.0 * args.burst * args.pulse_time + args.recool_time)
    tdesc = "burst=%d pulse=%gns recool=%gus duty=%.0f%% hold=%gs" % (
        args.burst, args.pulse_time * 1e9, args.recool_time * 1e6, _duty * 100, args.hold)

    if args.mode == "detuning":
        dets = _colon(*args.fdet); dets_hz = [float(d) * 1e6 for d in dets]
        g().Imag399.FreqDetuning.scan(1, dets_hz)
        g().Imag399.Amp1 = float(args.blue_amp)
        g().Imag399.Amp2 = float(args.blue_amp2)
        g().Imag399.Strobe.Cool556.X.FreqDetuning = xcd; g().Imag399.Strobe.Cool556.X.Amp = xca
        g().Imag399.Strobe.Cool556.h.FreqDetuning = hcd; g().Imag399.Strobe.Cool556.h.Amp = hca
        axis_desc = "%s | Imag399.FreqDetuning(MHz)=%s amps=%.2f/%.2f cool X(%.2f,%.2f) h(%.2f,%.2f)" % (
            tdesc, [round(d, 3) for d in dets], args.blue_amp, args.blue_amp2,
            xcd / 1e6, xca, hcd / 1e6, hca)
    elif args.mode == "amps":
        a1 = [float(a) for a in _colon(*args.amp1)]
        a2 = [float(a) for a in _colon(*args.amp2)]
        g().Imag399.Amp1.scan(1, a1)
        g().Imag399.Amp2.scan(2, a2)
        g().Imag399.FreqDetuning = float(args.det) * 1e6
        g().Imag399.Strobe.Cool556.X.FreqDetuning = xcd; g().Imag399.Strobe.Cool556.X.Amp = xca
        g().Imag399.Strobe.Cool556.h.FreqDetuning = hcd; g().Imag399.Strobe.Cool556.h.Amp = hca
        axis_desc = "%s | Amp1=%s x Amp2=%s det=%.2fMHz cool X(%.2f,%.2f) h(%.2f,%.2f)" % (
            tdesc, [round(a, 3) for a in a1], [round(a, 3) for a in a2], args.det,
            xcd / 1e6, xca, hcd / 1e6, hca)
    else:  # cool: 2-D strobe recool of one beam; pin the OTHER beam + amps + detuning.
        beam = args.beam
        dets = _colon(*args.cdet); dets_hz = [float(d) * 1e6 for d in dets]
        cam = [float(a) for a in _colon(*args.famp)]
        # 399 imaging: default to the loading pattern's tuned Amp1/Amp2 (untouched), but --img-amp1/
        # --img-amp2 OVERRIDE them via g() (NOT a config edit) -- e.g. drive the DDS amp to 1.0 (max)
        # to brighten the dim strobe frame. FreqDetuning stays pinned to the pattern.
        img_a1 = float(args.img_amp1) if args.img_amp1 is not None else float(c.Imag399.Amp1)
        img_a2 = float(args.img_amp2) if args.img_amp2 is not None else float(c.Imag399.Amp2)
        g().Imag399.Amp1 = img_a1
        g().Imag399.Amp2 = img_a2
        g().Imag399.FreqDetuning = float(c.Imag399.FreqDetuning)
        if beam == "X":
            g().Imag399.Strobe.Cool556.h.FreqDetuning = hcd; g().Imag399.Strobe.Cool556.h.Amp = hca
            g().Imag399.Strobe.Cool556.X.FreqDetuning.scan(1, dets_hz)
            g().Imag399.Strobe.Cool556.X.Amp.scan(2, cam)
            fixed = "h(%.2f,%.2f)" % (hcd / 1e6, hca)
        else:
            g().Imag399.Strobe.Cool556.X.FreqDetuning = xcd; g().Imag399.Strobe.Cool556.X.Amp = xca
            g().Imag399.Strobe.Cool556.h.FreqDetuning.scan(1, dets_hz)
            g().Imag399.Strobe.Cool556.h.Amp.scan(2, cam)
            fixed = "X(%.2f,%.2f)" % (xcd / 1e6, xca)
        tag = "g()-OVERRIDE" if (args.img_amp1 is not None or args.img_amp2 is not None) else "pattern"
        axis_desc = "%s | COOL %s det(MHz)=%s x amp=%s | imaging(%s) %.3f/%.3f det %.2fMHz fixed %s" % (
            tdesc, beam, [round(d, 3) for d in dets], [round(a, 3) for a in cam],
            tag, img_a1, img_a2, float(c.Imag399.FreqDetuning) / 1e6, fixed)

    rp = g.runp()
    rp.NumImages = 2; rp.Scramble = 1; rp.isInit = 0; rp.isHC = 0; rp.isGrid2 = 0
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
    lbl = "StrobeImagingScan_%s_r%d" % (args.mode, args.round)
    desc = scangroup_to_descriptor(g, "StrobeImagingPushoutSeq", opts={"rep": args.reps}, label=lbl,
                                   description="Strobe imaging optimization round %d (%s). %s"
                                               % (args.round, args.mode, axis_desc))
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d (%s): id=%d nseq=%d reps=%d -> target=%d start_seq=%d"
          % (args.round, args.mode, did, nseq, args.reps, target, start))
    print("  " + axis_desc)

    data_dir = None; t0 = time.time(); last = start
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


def do_watch(args):
    with open(_state_path(args.round)) as f:
        st = json.load(f)
    target, start, data_dir = st["target"], st["start"], st["data_dir"]
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
            print("  [%ds] seq_num=%d progress=%d/%d (%.0f%%)"
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
    ap.add_argument("--mode", choices=["detuning", "amps", "cool"], default="amps")
    ap.add_argument("--beam", choices=["X", "h"], default="X", help="cool mode: which recool beam")
    # strobe timing round-knobs (FIXED per round)
    ap.add_argument("--pulse-time", type=float, default=1e-6,
                    help="per-beam 399 pulse (s); beam1 then beam2 each get this (period=2*pulse+recool)")
    ap.add_argument("--recool-time", type=float, default=50e-6, help="556 recool window per cycle (s)")
    ap.add_argument("--burst", type=int, default=5,
                    help="PulsesPerBurst N: N beam1/beam2 SHORT-pulse pairs per recool (raises duty)")
    ap.add_argument("--hold", type=float, default=0.001, help="PushouthX hold (s); ~0 = real survival")
    # cool grids
    ap.add_argument("--cdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.26),
                    help="cool: recool FreqDetuning colon (MHz)")
    ap.add_argument("--famp", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.28),
                    help="cool: recool Amp colon")
    # amps grids
    ap.add_argument("--amp1", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.30))
    ap.add_argument("--amp2", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(0.10, 0.02, 0.22))
    # detuning grid
    ap.add_argument("--fdet", type=float, nargs=3, metavar=("LO", "STEP", "HI"), default=(-9.0, 0.5, -1.0))
    # fixed values
    ap.add_argument("--det", type=float, default=-5.0, help="fixed Imag399.FreqDetuning (MHz)")
    ap.add_argument("--blue-amp", type=float, default=0.18, help="fixed Imag399.Amp1 (amps/detuning modes)")
    ap.add_argument("--blue-amp2", type=float, default=0.18, help="fixed Imag399.Amp2 (amps/detuning modes)")
    ap.add_argument("--img-amp1", type=float, default=None,
                    help="cool mode: g() OVERRIDE Imag399.Amp1 (NOT config); e.g. 1.0 = max DDS amp")
    ap.add_argument("--img-amp2", type=float, default=None,
                    help="cool mode: g() OVERRIDE Imag399.Amp2 (NOT config); e.g. 1.0 = max DDS amp")
    ap.add_argument("--xcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="override Imag399.Strobe.Cool556.X (det MHz, amp)")
    ap.add_argument("--hcool", type=float, nargs=2, metavar=("DET_MHZ", "AMP"), default=None,
                    help="override Imag399.Strobe.Cool556.h (det MHz, amp)")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--defocus", type=float, default=-1.25, help="loading z4 defocus (rad); 3270_tri = -1.25")
    ap.add_argument("--loading-phase", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    else:
        do_watch(a)
