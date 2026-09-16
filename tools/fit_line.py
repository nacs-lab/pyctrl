"""fit_line.py -- ONE entry point for fitting ANY scan by scan id.

Why this exists: a fit is a ~1 second command, but a reader (human or agent) used to spend five
tool calls working out WHICH fitter and WHICH flags a given scan needs -- grepping for "Autler",
reading tool docstrings, looking up the mj=0 reference frequency. All of it is derivable from the
run itself, so derive it here and leave nothing to decide.

    <yb_analysis-python> pyctrl/tools/fit_line.py <scan_id>

It reads the run's own descriptor -- how many axes were swept, what they are, and the ScanName --
picks the right fitter and flags, says which and why, then runs it and passes the output through.
Anything extra you pass is forwarded, so an override is still one command:

    fit_line.py 20260915125339 --bare 97.3237     # AT doublet, symmetry check vs the bare line
    fit_line.py 20260915122816 --peaks 1          # force a single Lorentzian on the 399 line
    fit_line.py 20260915140525 --maximize         # a 2-D map where HIGH survival is the goal
    fit_line.py <sid> --show                      # print the chosen command, fit nothing

**Dispatch order** (first match wins), designed so a wrong MODEL is never silently applied:

  1. a scan NAME with a dedicated tool (AT doublet, Stark vertex, Hahn echo, ...);
  2. the AXIS COUNT -- any 2-axis scan goes to fit_map.py, because fitting a 1-D lineshape to a
     grid is meaningless and used to happen silently;
  3. the scan NAME for the known 1-D line shapes (556 / 399 / revival / decay / Rabi);
  4. fallback: fit_map.py, which is model-free -- it reports the optimum, whether it is bracketed,
     and whether it is a point or a plateau. An unrecognised scan gets an honest answer rather
     than a Lorentzian fitted to something that is not a line.

Run from the PROJECT ROOT. Nothing here fits anything itself -- it only dispatches, so each
fitter stays the single source of truth for its own numbers.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "pyctrl", "tools")
DATA = r"D:\OneDrive - Harvard University\Documents - Yb\Data"

SCAN_ID, SCAN_DIR = "scan_id", "scan_dir"      # what the target tool wants as its positional


def _scan_dir(sid):
    sid = str(sid).replace("_", "")
    fid = sid[:8] + "_" + sid[8:]
    d = os.path.join(DATA, sid[:8], "data_" + fid)
    if not os.path.isdir(d):
        raise SystemExit("no such scan dir: %s" % d)
    return d, fid


def _run_json(scan_dir, fid):
    with open(os.path.join(scan_dir, "data_%s.json" % fid)) as fh:
        return json.load(fh)


def _scan_name(js):
    n = js.get("ScanName")
    if isinstance(n, dict) and n.get("scanname"):
        return "".join(chr(c) for c in n["scanname"])
    if isinstance(n, str):
        return n
    d = js.get("descriptor")
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            d = {}
    return (d or {}).get("seq") or ""


def _axes(js):
    """[(param_name, n_points), ...] in dim order, from the run's own descriptor."""
    d = js.get("descriptor")
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            d = {}
    found = {}
    for k, v in ((d or {}).get("params") or {}).items():
        if isinstance(v, dict) and "scan" in v:
            # SEVERAL params can share one dim -- a CO-VARYING axis. Keep them all: keying by dim
            # alone drops every name but the last, which is how a co-varying scan gets
            # mis-attributed (the stirap-optimization runbook warns about exactly this).
            found.setdefault(int(v["scan"]), []).append((k, len(v.get("values") or [])))
    return [(names[0][0], names[0][1], [n for n, _ in names[1:]])
            for _, names in sorted(found.items())]


def _cfg(js, key):
    """A value from the run's OWN expConfig snapshot (not today's file)."""
    c = js.get("expConfig")
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            return None
    if isinstance(c, dict) and isinstance(c.get(key), (int, float)):
        return float(c[key])
    return None


# ---- 1. scan names with a DEDICATED tool. These win over the axis-count rule, because the tool
#         was written for that scan's exact shape (several are themselves 2-D).
BY_NAME_FIRST = [
    ("StarkVxRevival616", "stark_vx_fit.py", [], SCAN_ID,
     "the Stark/Vx 2-D -- its dedicated tool fits a revival PEAK per Vx group, then the "
     "centre-vs-voltage PARABOLA whose vertex is the null voltage"),
    ("Echo", "fit_echo_t2.py", [], SCAN_DIR,
     "a Hahn-echo 2-D (closing-phase x free-evolution) -> T2"),
    ("DipolarExchange", "fit_dipolar_exchange.py", [], SCAN_DIR,
     "pairwise dipolar spin-exchange, with SPAM/loss correction"),
    ("AutlerTownes", "fit_at_splitting.py", [], SCAN_ID,
     "an AT doublet -- the purpose-built double-dip fitter, which also prints the optical x2 "
     "= Omega_308. Pass --bare <bare-dip-MHz> for the symmetry check"),
]

# ---- 3. known 1-D LINE SHAPES, by scan name.
BY_NAME_1D = [
    ("Revival616MW", "fit_spectrum.py", ["--mode", "peak", "--xlabel", "MW freq [MHz]"], SCAN_ID,
     "a survival REVIVAL peak swept against MW frequency"),
    ("Revival616", "fit_spectrum.py", ["--mode", "peak", "--xlabel", "616-EOM freq [MHz]"],
     SCAN_ID, "a survival REVIVAL (a PEAK, not a dip)"),
    ("MWRabiSpectrum", "fit_rabi_spectrum.py", [], SCAN_ID,
     "a finite-pulse Rabi lineshape -- NOTE it needs --t-pulse-ns (the known pulse length)"),
    ("GRRabi", "fit_rabi_time.py", [], SCAN_ID,
     "a damped Rabi oscillation vs drive duration -> f_Rabi and tau"),
    ("ReleaseRecapture", "fit_decay.py", [], SCAN_ID,
     "release-recapture survival vs release time -> an exponential decay (a temperature proxy)"),
    ("ImagingLifetime", "fit_decay.py", [], SCAN_ID,
     "survival vs imaging time -> an exponential lifetime"),
    ("Spectrum399", "fit_spectrum.py", ["--peaks", "2"], SCAN_ID,
     "the 399 line, a two-component doublet -- 2 Lorentzians (the fitter declines to 1 if the "
     "second component is not significant)"),
    ("Spectrum556Scan_mj0", "fit_spectrum.py", ["--min-r2", "0.95"], SCAN_ID,
     "the mj=0 CALIBRATION line -- tighter R^2 gate, --ref from the run's own expConfig"),
    ("Spectrum556Scan_mj1", "fit_spectrum.py", [], SCAN_ID,
     "the |mj|=1 light-shifted line (broad; reference/trend only)"),
    ("RydbergSpectrum556", "fit_spectrum.py", [], SCAN_ID,
     "a 556 Rydberg push-out DIP"),
    ("Spectrum556", "fit_spectrum.py", [], SCAN_ID,
     "a 556 spectroscopy DIP"),
]

# Swept-axis names that mean "a time sweep", i.e. a decay/oscillation rather than a line shape.
TIME_AXIS_HINTS = ("Delay", "Gap", "Time", "Duration", "width_us", "Tau")


def main():
    ap = argparse.ArgumentParser(
        description="Fit ANY scan by id; the fitter and flags are chosen from the run itself.")
    ap.add_argument("scan", help="14-digit scan_id (or file_id with the underscore)")
    ap.add_argument("--show", action="store_true",
                    help="print the chosen command and exit WITHOUT fitting")
    args, passthrough = ap.parse_known_args()

    scan_dir, fid = _scan_dir(args.scan)
    js = _run_json(scan_dir, fid)
    name = _scan_name(js)

    # A run that never wrote a data file produced no shots at all. Say that plainly -- the
    # analysis layer otherwise raises "failed to load scan data" from three frames down.
    if not (os.path.exists(os.path.join(scan_dir, "data_%s.h5" % fid))
            or os.path.exists(os.path.join(scan_dir, "data_%s.mat" % fid))):
        print("fit_line: %s | NO DATA" % (name or "<unknown scan name>"))
        print("  The scan directory holds only the descriptor -- there is no .h5/.mat, so this run "
              "never wrote a shot. It was submitted and then died, or was cancelled before the "
              "first frame. Nothing to fit; check the backend log for why it produced no data.")
        return 2
    axes = _axes(js)
    n_ax = len(axes)

    # A LOADING scan (TweezerLoadingSeq, one image) has no line to fit and is legitimately 0-, 1-
    # or 2-axis -- a single-point verify, a knob sweep, or the 3-point bias canary -- so it is
    # routed BEFORE the axis-count rules below would call a single point "nothing to fit".
    # loading_report.py reports rate / uniformity / stability / 2-atom fraction / d' and a verdict.
    # ScanName carries the submit LABEL (loading_round.py labels its rounds "LoadingOpt_r<N>"); the
    # sequence itself is the descriptor's "seq". Match on either.
    _d = js.get("descriptor")
    if isinstance(_d, str):
        try:
            _d = json.loads(_d)
        except Exception:
            _d = {}
    seq_name = str((_d or {}).get("seq") or "")
    if "tweezerloadingseq" in (name + " " + seq_name).lower() or name.lower().startswith("loadingopt"):
        cmd = [sys.executable, os.path.join(TOOLS, "loading_report.py"), fid.replace("_", "")] + passthrough
        print("fit_line: %s (%s) | %d swept %s -> loading_report.py"
              % (name, seq_name or "?", n_ax, "axis" if n_ax == 1 else "axes"))
        for i, (an, npn, co) in enumerate(axes):
            print("    dim%d %s (%d pts)%s" % (i + 1, an, npn, ("  + co-varying: " + ", ".join(co)) if co else ""))
        print("  why: a loading scan -- no lineshape; judged on rate, uniformity (corners, gradient), "
              "per-shot stability, the 2-atom fraction and d'. A 3-cell 1-D is the bias canary (R, S).")
        print("  cmd: " + " ".join(cmd))
        if args.show:
            return 0
        return subprocess.call(cmd, cwd=ROOT)

    # 0. nothing was swept -- there is no axis to fit against. Say so here rather than
    #    dispatching to a fitter that will error on it.
    if n_ax == 0:
        print("fit_line: %s | 0 swept axes -> NOTHING TO FIT" % (name or "<unknown scan name>"))
        print("  This run has no swept parameter (its ScanGroup vars.params is empty), so it is a "
              "fixed-point repeat, not a scan: there is no line, no map and no optimum to extract.")
        print("  The only number available is the single-point survival with its error, which "
              "analyze_scan gives directly. To calibrate a knob, re-submit as a 1-D or 2-D sweep "
              "over it and fit THAT scan id.")
        return 2

    # 0b. more than 2 swept axes: no tool here summarises a 3-D+ grid, and collapsing it
    #     silently would be worse than saying so.
    if n_ax > 2:
        print("fit_line: %s | %d swept axes -> NOT SUPPORTED" % (name or "<unknown>", n_ax))
        for i, (an, npn, co) in enumerate(axes):
            print("    dim%d %s (%d pts)%s" % (i + 1, an, npn,
                  ("  + co-varying: " + ", ".join(co)) if co else ""))
        print("  fit_map.py summarises 1-D and 2-D sweeps only. A %d-D grid needs either a slice "
              "(re-fit holding the other axes fixed) or a purpose-built analysis -- collapsing it "
              "to a line or a plane would hide exactly the structure it was taken to find."
              % n_ax)
        return 2

    tool = flags = target = why = None

    # 1. dedicated tool by name
    for frag, t, f, tgt, w in BY_NAME_FIRST:
        if frag.lower() in name.lower():
            tool, flags, target, why = t, list(f), tgt, w
            break

    # 2. any OTHER 2-axis scan -> the map summariser, never a 1-D lineshape
    if tool is None and n_ax == 2:
        tool, flags, target = "fit_map.py", [], SCAN_ID
        why = ("a 2-axis scan (%s x %s) -- fitting a 1-D lineshape to a grid is meaningless, so "
               "this goes to the map summariser: optimum + error, interior-or-edge, "
               "point-or-ridge" % (axes[0][0], axes[1][0]))

    # 3. known 1-D line shapes by name
    if tool is None and n_ax == 1:
        for frag, t, f, tgt, w in BY_NAME_1D:
            if frag.lower() in name.lower():
                tool, flags, target, why = t, list(f), tgt, w
                break

    # 4. model-free fallback -- an honest answer beats a wrong model
    if tool is None:
        tool, flags, target = "fit_map.py", [], SCAN_ID
        swept = axes[0][0] if n_ax == 1 else "?"
        is_time = any(h.lower() in swept.lower() for h in TIME_AXIS_HINTS)
        why = ("no dedicated fitter matches ScanName %r. Falling back to the MODEL-FREE map "
               "summariser (optimum, bracketed?, point-or-plateau) rather than assuming a "
               "lineshape.%s" % (name, (" The swept axis %r looks like a TIME sweep: if you want a"
                                        " decay or an oscillation fitted, use fit_decay.py or"
                                        " fit_rabi_time.py." % swept) if is_time else
                                 (" If it IS a line, pass it to fit_spectrum.py yourself"
                                  " (--mode peak / --peaks 2 as needed).")))

    # a --ref the caller did not give, where the run itself defines one
    if "--ref" not in set(passthrough) and tool == "fit_spectrum.py":
        ref = None
        if "Spectrum556Scan_mj0" in name:
            ref = _cfg(js, "Resonance556mj0Freq")
        elif "Spectrum399" in name:
            ref = 310e6
        if ref:
            flags += ["--ref", repr(float(ref))]

    positional = scan_dir if target == SCAN_DIR else args.scan
    cmd = [sys.executable, os.path.join(TOOLS, tool), positional] + flags + passthrough
    print("fit_line: %s | %d swept ax%s -> %s"
          % (name or "<unknown scan name>", n_ax, "is" if n_ax == 1 else "es", tool))
    for i, (an, np_, co) in enumerate(axes):
        print("    dim%d %s (%d pts)%s"
              % (i + 1, an, np_,
                 ("  + CO-VARYING on this axis: " + ", ".join(co)) if co else ""))
    print("  why: %s" % why)
    print("  cmd: %s" % " ".join(('"%s"' % c) if " " in c else c for c in cmd[1:]))
    print("")
    sys.stdout.flush()          # keep our header above the child's output
    if args.show:
        return 0

    env = dict(os.environ)
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
