"""picomotor_move.py -- ONE picomotor step on an NI analog-out channel, with measured timing
and a persistent move log.

A picomotor "move" here is a VOLTAGE HELD FOR A TIME on the mirror's NI AO channel
(VPicoMotor308h = Dev1/10, VPicoMotor308v = Dev1/11, 369h/v = Dev1/22,23): set V, hold,
return to 0. The physically meaningful quantity is the impulse V*s, which is what the
2026-06-30 alignment session logged ("-1.0 V, 0.5 s hold (move = -0.5 V*s)").

WHY THIS EXISTS rather than two dashboard /api/nidaq/set calls: each dashboard write spawns a
fresh engine subprocess and takes ~0.65 s round-trip, so a "0.5 s" hold built from two HTTP
calls is really ~1.1 s -- more than 2x the intended step. This does set -> sleep -> set inside
ONE process (same devices.nidaq.nidaq_io_handler.set_channel the dashboard driver uses) and
REPORTS THE MEASURED HOLD, so the logged V*s is what the hardware actually saw.

SAFETY (mirrors the dashboard route's guards):
  * refuses to move while a scan is running -- a run reserves Dev1's AO subsystem
  * |V| <= 10 V, and this tool additionally caps |V| at --max-volt (default 1.0)
  * ALWAYS returns the channel to 0 V, including on Ctrl-C or a mid-move exception
  * --dry resolves and logs nothing, touching no hardware

Run from pyctrl/ with the ENGINE python (nidaqmx lives only there):
    .venv-engine-py312\\Scripts\\python.exe tools/picomotor_move.py --axis v --volts -1.0 --hold 0.5
    ... --axis h --volts +1.0 --hold 0.5 --note "reversing after turnover"
    ... --dry            # resolve + show what would happen

Log: pyctrl/tmp/picomotor_log.json (one record per move, with cumulative V*s per axis).
Read it back with --show.
"""
import argparse
import json
import os
import sys
import time

BEAM_CHANNELS = {
    ("308", "h"): "VPicoMotor308h",
    ("308", "v"): "VPicoMotor308v",
    ("369", "h"): "VPicoMotor369h",
    ("369", "v"): "VPicoMotor369v",
}
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp",
                        "picomotor_log.json")
URL = "tcp://127.0.0.1:1408"


def _pyctrl_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("", "lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans"):
        p = os.path.join(root, d) if d else root
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def _resolve(alias):
    """alias -> 'Dev1/N' via expConfig, the same source the dashboard driver uses."""
    _pyctrl_root()
    import expConfig
    cfg = expConfig.build_config()
    keys = list(cfg["channel_alias_keys"])
    vals = list(cfg["channel_alias_vals"])
    if alias not in keys:
        raise SystemExit("unknown channel alias %r (not in expConfig)" % alias)
    return vals[keys.index(alias)]


def _scan_running():
    try:
        import zmq
    except ImportError:
        return None                      # cannot check -> caller decides
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(URL)
    s.send_string("queue_list")
    raw = s.recv() if s.poll(8000) else None
    s.close(0)
    if raw is None:
        return None
    q = json.loads(raw)
    return bool(q.get("running") and q["running"].get("id") is not None)


def _load_log():
    path = os.path.normpath(LOG_PATH)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"moves": []}


def _save_log(log):
    path = os.path.normpath(LOG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return path


def _cumulative(log, beam, axis):
    return sum(m["volt_seconds"] for m in log["moves"]
               if m["beam"] == beam and m["axis"] == axis and not m.get("dry"))


def do_move(args):
    alias = BEAM_CHANNELS[(args.beam, args.axis)]
    backend = _resolve(alias)
    log = _load_log()
    cum_before = _cumulative(log, args.beam, args.axis)

    if abs(args.volts) > args.max_volt:
        raise SystemExit("|%g V| exceeds --max-volt %g -- refusing (keep steps small)"
                         % (args.volts, args.max_volt))

    print("%s %s-axis (%s = %s): %+.3f V for %.3f s  [cumulative before: %+.3f V*s]"
          % (args.beam, args.axis, alias, backend, args.volts, args.hold, cum_before))
    if args.dry:
        print("DRY RUN -- no hardware touched, nothing logged")
        return 0

    running = _scan_running()
    if running:
        raise SystemExit("REFUSING: a scan is running -- it owns Dev1's AO subsystem")
    if running is None and not args.force:
        raise SystemExit("REFUSING: could not verify the backend is idle (pass --force to override)")

    from devices.nidaq.nidaq_io_handler import set_channel, read_channel

    t_on = t_off = None
    try:
        t_on = time.perf_counter()
        set_channel(backend, float(args.volts))
        t_set = time.perf_counter()
        rb = None
        try:
            rb = float(read_channel(backend))
        except Exception:                                    # noqa: BLE001
            pass
        remain = args.hold - (time.perf_counter() - t_set)
        if remain > 0:
            time.sleep(remain)
    finally:
        set_channel(backend, 0.0)                            # ALWAYS return to rest
        t_off = time.perf_counter()

    hold = t_off - t_set if t_on is not None else float("nan")
    vs = float(args.volts) * hold
    rec = {
        "n": len([m for m in log["moves"] if not m.get("dry")]) + 1,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "beam": args.beam, "axis": args.axis, "alias": alias, "backend": backend,
        "volts": float(args.volts), "hold_requested_s": float(args.hold),
        "hold_measured_s": round(hold, 4), "volt_seconds": round(vs, 4),
        "cumulative_volt_seconds": round(cum_before + vs, 4),
        "readback_v": rb, "note": args.note or "",
        "scan_after": None, "splitting_after_MHz": None,
    }
    log["moves"].append(rec)
    path = _save_log(log)

    print("MOVED: %+.3f V held %.3f s (requested %.3f) -> impulse %+.4f V*s"
          % (rec["volts"], rec["hold_measured_s"], rec["hold_requested_s"], rec["volt_seconds"]))
    print("       readback while held: %s V | channel returned to 0 V"
          % ("%+.4f" % rb if rb is not None else "n/a"))
    print("       cumulative on %s-%s: %+.4f V*s   (move #%d)"
          % (args.beam, args.axis, rec["cumulative_volt_seconds"], rec["n"]))
    print("       logged -> %s" % path)
    return 0


def do_show(args):
    log = _load_log()
    moves = [m for m in log["moves"] if not m.get("dry")]
    if not moves:
        print("no moves logged yet (%s)" % os.path.normpath(LOG_PATH))
        return 0
    print("%-3s %-19s %-4s %-4s %8s %8s %10s %12s  %s"
          % ("#", "time", "beam", "axis", "volts", "hold_s", "V*s", "cum V*s", "splitting/note"))
    for m in moves:
        tail = ""
        if m.get("splitting_after_MHz") is not None:
            tail = "%.4f MHz" % m["splitting_after_MHz"]
            if m.get("scan_after"):
                tail += " (%s)" % m["scan_after"]
        if m.get("note"):
            tail = (tail + " | " if tail else "") + m["note"]
        print("%-3d %-19s %-4s %-4s %+8.3f %8.3f %+10.4f %+12.4f  %s"
              % (m["n"], m["iso"], m["beam"], m["axis"], m["volts"],
                 m["hold_measured_s"], m["volt_seconds"], m["cumulative_volt_seconds"], tail))
    for beam, axis in sorted({(m["beam"], m["axis"]) for m in moves}):
        print("cumulative %s-%s: %+.4f V*s" % (beam, axis, _cumulative(log, beam, axis)))
    return 0


def do_annotate(args):
    """Attach the post-move measurement to the most recent move (or --move N)."""
    log = _load_log()
    moves = [m for m in log["moves"] if not m.get("dry")]
    if not moves:
        raise SystemExit("no moves logged")
    rec = moves[-1] if args.move is None else next(
        (m for m in moves if m["n"] == args.move), None)
    if rec is None:
        raise SystemExit("no move #%s" % args.move)
    if args.scan:
        rec["scan_after"] = args.scan
    if args.splitting is not None:
        rec["splitting_after_MHz"] = float(args.splitting)
    if args.note:
        rec["note"] = (rec.get("note", "") + " | " if rec.get("note") else "") + args.note
    _save_log(log)
    print("move #%d annotated: scan=%s splitting=%s"
          % (rec["n"], rec["scan_after"], rec["splitting_after_MHz"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description="One picomotor step (NI AO channel) + move log.")
    ap.add_argument("--beam", default="308", choices=["308", "369"])
    ap.add_argument("--axis", choices=["h", "v"], help="h = horizontal, v = vertical")
    ap.add_argument("--volts", type=float, help="step voltage (sign = direction)")
    ap.add_argument("--hold", type=float, default=0.5, help="hold seconds (default 0.5)")
    ap.add_argument("--max-volt", type=float, default=1.0,
                    help="refuse steps larger than this (default 1.0 V -- keep steps small)")
    ap.add_argument("--note", default="", help="free-text note stored with the move")
    ap.add_argument("--dry", action="store_true", help="resolve only; no hardware, no log")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the backend idle state cannot be verified")
    ap.add_argument("--show", action="store_true", help="print the move log and exit")
    ap.add_argument("--annotate", action="store_true",
                    help="attach --scan/--splitting to the last move (or --move N)")
    ap.add_argument("--move", type=int, default=None, help="annotate this move number")
    ap.add_argument("--scan", default=None, help="scan id measured after the move")
    ap.add_argument("--splitting", type=float, default=None, help="AT splitting (MHz) after")
    args = ap.parse_args()

    if args.show:
        return do_show(args)
    if args.annotate:
        return do_annotate(args)
    if not args.axis or args.volts is None:
        ap.error("--axis and --volts are required for a move")
    return do_move(args)


if __name__ == "__main__":
    sys.exit(main())
