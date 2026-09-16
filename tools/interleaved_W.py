"""Interleaved, drift-IMMUNE head-to-head of whole imaging working sets W.

Why this exists (2026-09-15): the 399 wavemeter PID lock is disengaged and the laser free-runs
several MHz on ~15-min timescales, which moves the effective imaging detuning. Tonight the SAME
nominal config measured 10 minutes apart differed by ~1.3% in survival -- larger than every
difference this campaign is trying to resolve. Comparing cells ACROSS runs is therefore invalid.

The fix: put every candidate W on ONE scan axis, so `Scramble` interleaves them shot-by-shot and
any drift is common-mode. Several params are scanned on the SAME dim, which pairs them elementwise
(ScanGroup semantics) -- so cell k is the complete config W[k], not a grid corner.

Usage:  python pyctrl/tools/interleaved_W.py --reps 40 [--round N] [--desc "..."]
"""
import argparse, json, os, sys, time

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PYCTRL = os.path.join(REPO, "pyctrl")
URL = "tcp://127.0.0.1:1408"
for _d in ("", "lib", "YbExptCtrl", "YbScans"):
    p = os.path.join(PYCTRL, _d)
    if p not in sys.path:
        sys.path.insert(0, p)

# name, Img1PIDSet, (X det MHz, X amp), (h det MHz, h amp)
CONFIGS = [
    ("A_incumbent_09-14",   1.0, (0.18, 0.30), (0.15, 0.17)),
    ("B_newpower_only",     1.8, (0.18, 0.30), (0.15, 0.17)),
    ("C_newpower_newh",     1.8, (0.18, 0.30), (0.18, 0.24)),
]
DET_MHZ = -10.0          # Imag399.FreqDetuning, committed tonight
IMG2 = 0.4               # beam 2 PARKED (user 2026-09-15); its servo does not regulate, so this
                         # setpoint changes no light -- R1006 (Img2 1.0) and R1021 (Img2 0.4) give
                         # the SAME atom-empty distance at matched Img1 (10.1 vs 10.2, 13.9 vs 13.9)


def build():
    from scan_group import ScanGroup
    from seq_config import SeqConfig
    from consts import Consts
    if not SeqConfig.get().consts:
        SeqConfig.load_real()
    c = Consts()
    reson556 = float(c.Resonance556mj0Freq)
    reson399 = float(c.Resonance399Freq)
    g = ScanGroup()
    g().Imag399.Amp1 = 1
    g().Imag399.Amp2 = 1
    g().BlueMOT.Img2PIDSet = IMG2
    g().Imag399.FreqDetuning = DET_MHZ * 1e6
    g().Pushout.Time = 0.001                     # 0 pushout = the REAL 50 ms two-image survival
    g().Pushout.Blue.Amp1 = 1
    g().Pushout.Blue.Amp2 = 1
    g().Pushout.Blue.Freq = reson399 + DET_MHZ * 1e6
    # every knob that differs between configs goes on dim 1 -> paired elementwise
    g().BlueMOT.Img1PIDSet.scan(1, [float(k[1]) for k in CONFIGS])
    g().Imag399.Cool556.X.FreqDetuning.scan(1, [k[2][0] * 1e6 for k in CONFIGS])
    g().Imag399.Cool556.X.Amp.scan(1, [float(k[2][1]) for k in CONFIGS])
    g().Imag399.Cool556.h.FreqDetuning.scan(1, [k[3][0] * 1e6 for k in CONFIGS])
    g().Imag399.Cool556.h.Amp.scan(1, [float(k[3][1]) for k in CONFIGS])
    g().Pushout.Green.X.Freq.scan(1, [reson556 + k[2][0] * 1e6 for k in CONFIGS])
    g().Pushout.Green.X.Amp.scan(1, [float(k[2][1]) for k in CONFIGS])
    g().Pushout.Green.h.Freq.scan(1, [reson556 + k[3][0] * 1e6 for k in CONFIGS])
    g().Pushout.Green.h.Amp.scan(1, [float(k[3][1]) for k in CONFIGS])
    rp = g.runp()
    rp.NumImages = 2
    rp.Scramble = 1                 # THE point: interleave the configs shot-by-shot
    rp.isInit = 0; rp.isHC = 0; rp.isGrid2 = 0
    rp.loading_phase = "phase/33x33_feedback11.pt"
    rp.loading_defocus = -4.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40, help="shots per config")
    ap.add_argument("--round", type=int, default=1019)
    ap.add_argument("--desc", default="")
    ap.add_argument("--configs", default=None,
                    help='JSON list of [name, Img1PIDSet, [Xdet,Xamp], [hdet,hamp]] overriding CONFIGS')
    ap.add_argument("--img2", type=float, default=None, help="override the parked Img2PIDSet")
    a = ap.parse_args()
    from scan_export import scangroup_to_descriptor
    from yb_start_scan import submit_descriptor
    global CONFIGS, IMG2
    if a.configs:
        CONFIGS = [(k[0], float(k[1]), (float(k[2][0]), float(k[2][1])),
                    (float(k[3][0]), float(k[3][1]))) for k in json.loads(a.configs)]
    if a.img2 is not None:
        IMG2 = float(a.img2)
    g = build()
    lbl = "ImagingW_interleaved_r%d" % a.round
    desc = scangroup_to_descriptor(g, "ImagingPushoutSurvivalSeq", opts={"rep": a.reps},
                                   label=lbl, description=a.desc)
    did = submit_descriptor(URL, json.dumps(desc, ensure_ascii=False), lbl)
    print("submitted r%d id=%d  %d configs x %d shots" % (a.round, did, len(CONFIGS), a.reps))
    for k in CONFIGS:
        print("   %-20s Img1PIDSet %.2f  X(%.2f,%.2f)  h(%.2f,%.2f)" % (k[0], k[1], k[2][0], k[2][1], k[3][0], k[3][1]))
    st = {"round": a.round, "id": did, "configs": [list(k) for k in CONFIGS], "reps": a.reps,
          "mode": "interleavedW", "axis_desc": "interleaved W: " + ", ".join(k[0] for k in CONFIGS)}
    with open(os.path.join(PYCTRL, "tmp", "imaging_state_r%d.json" % a.round), "w") as f:
        json.dump(st, f, indent=2)
    print("state written")


if __name__ == "__main__":
    main()
