"""loading_det_round.py -- 1-D sweep of a LOADING-stage 399 knob, read as loading fraction.

Sibling of ``bluelac_round.py`` (whose submit/self-threshold/analyze machinery it reuses): same
vehicle (``BlueTweezerLoadingSeq``, NumImages=1, self-thresholded per-site occupancy), but the
swept axis is a **399 blue-MOT** knob rather than a blue-LAC one:

  motdet : ``BlueMOT.FreqDetuning`` (MHz) -- the 399 capture detuning (the loading "speed lever",
           loading-optimization.md Phase 4). The one to re-check when the 399 line has moved.
  motamp : ``BlueMOT.Amp`` -- 399 capture intensity (flat 0.4-0.7 historically).

The blue-LAC point is pinned from the pattern overlay (or --bdet/--bamp/--btime) so only the
swept axis moves. Run under the yb_analysis env:

    python tools/loading_det_round.py submit --round 92 --mode motdet \
        --dets -50 2 -38 --reps 4 --pattern tri_3013_camfb \
        --loading-phase phase/tri_3013_camfb.pt --defocus -2 --bdet -3.0
    python tools/loading_det_round.py watch --round 92
"""
import argparse
import json
import os
import sys
import time

import bluelac_round as B      # same dir; reuse URL/state/self-threshold/analyze

STATE_DIR = B.STATE_DIR


def _state_path(rnd):
    return os.path.join(STATE_DIR, "loadingdet_state_r%d.json" % rnd)


def build(args):
    B._pyctrl_path()
    from scan_group import ScanGroup
    c = B._consts()
    g = ScanGroup()

    # pin the blue-LAC point (defaults = the pattern overlay's committed values)
    bdet = float(args.bdet) * 1e6 if args.bdet is not None else float(c.LAC.BlueLAC.FreqDetuning)
    bamp = float(args.bamp) if args.bamp is not None else float(c.LAC.BlueLAC.Amp)
    btime = float(args.btime) if args.btime is not None else float(c.LAC.BlueLAC.Time)
    g().LAC.BlueLAC.FreqDetuning = bdet
    g().LAC.BlueLAC.Amp = bamp
    g().LAC.BlueLAC.Time = btime

    if args.mode == "motdet":
        vals = B._colon(*args.dets)                      # MHz (negative = red of the 399 line)
        g().BlueMOT.FreqDetuning.scan(1, [float(v) * 1e6 for v in vals])
        axis_desc = "BlueMOT.FreqDetuning(MHz)=%s" % [round(v, 3) for v in vals]
    else:  # motamp
        vals = [float(a) for a in B._colon(*args.amps)]
        g().BlueMOT.Amp.scan(1, vals)
        axis_desc = "BlueMOT.Amp=%s" % [round(v, 3) for v in vals]
    axis_desc += "  blueLAC(%.2fMHz,%.3f,%.3fs)" % (bdet / 1e6, bamp, btime)

    rp = g.runp()
    rp.NumImages = 1
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = B.LOADING_PHASE
    rp.loading_defocus = float(args.defocus)
    return g, axis_desc


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
    g, axis_desc = build(args)
    nseq = g.nseq()
    lbl = "LoadingDet_%s_r%d" % (args.mode, args.round)
    desc = scangroup_to_descriptor(
        g, "BlueTweezerLoadingSeq", opts={"rep": args.reps}, label=lbl,
        description=("Loading 399 re-optimization on %s: %s. Loading fraction (self-thresholded) "
                     "vs the swept blue-MOT knob; blue-LAC point pinned. %s"
                     % (B.PATTERN, args.mode, axis_desc)))
    did = submit_descriptor(B.URL, json.dumps(desc, ensure_ascii=False), lbl)
    target = start + args.reps * nseq
    print("submitted round %d (%s): id=%d nseq=%d reps=%d -> target=%d start=%d"
          % (args.round, args.mode, did, nseq, args.reps, target, start))
    print("  " + axis_desc)

    data_dir = None
    t0 = time.time(); last = start
    while time.time() - t0 < 120:
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
    st = {"round": args.round, "mode": args.mode, "start": start, "target": target,
          "nseq": nseq, "reps": args.reps, "data_dir": data_dir, "id": did,
          "axis_desc": axis_desc, "pattern": B.PATTERN}
    with open(_state_path(args.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state ->", _state_path(args.round))


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
    B.analyze(st["data_dir"], st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["submit", "watch"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--mode", choices=["motdet", "motamp"], default="motdet")
    ap.add_argument("--dets", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(-50.0, 2.0, -38.0), help="motdet: BlueMOT.FreqDetuning colon (MHz)")
    ap.add_argument("--amps", type=float, nargs=3, metavar=("LO", "STEP", "HI"),
                    default=(0.4, 0.1, 0.8), help="motamp: BlueMOT.Amp colon")
    ap.add_argument("--bdet", type=float, default=None, help="pin BlueLAC.FreqDetuning (MHz)")
    ap.add_argument("--bamp", type=float, default=None, help="pin BlueLAC.Amp")
    ap.add_argument("--btime", type=float, default=None, help="pin BlueLAC.Time (s)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--defocus", type=float, default=-2.0)
    ap.add_argument("--pattern", type=str, default=None)
    ap.add_argument("--loading-phase", type=str, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.phase == "submit":
        do_submit(a)
    else:
        do_watch(a)
