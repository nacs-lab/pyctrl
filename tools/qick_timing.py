#!/usr/bin/env python3
"""qick_timing.py -- measure the wall-clock cost of the QICK per-shot arm path (NEEDS THE BOARD).

The per-shot QICK arm recompiles + ``config_all`` ON THE BOARD every shot (the firmware is one-shot;
see devices/qick_awg/fpga_awg_manager.arm_for_seq). Its wall-clock sets the shot-rate ceiling, so we
measure it directly. The client blocks on each command's ack, so client-side ``perf_counter`` around a
call captures the FULL round-trip incl. the server-side compile + config -- the true per-shot cost.

What it times (against a live ``run_server.py`` on the board):
  * connect / disconnect
  * set_trigger_mode
  * prepare: upload one program's waveforms + program (per program); and the whole batch setup(N)
  * arm (stop+start) repeated M times -- SAME program (re-arm) and a DIFFERENT program (switch);
    expected ~equal since arm is always stop+start+recompile
  * cancel: stop_program alone
  * scaling: repeat for programs of growing complexity (more loop chunks -> longer compile)

Usage (from pyctrl/, any interpreter -- the QICK client is pure stdlib socket):
    python -m tools.qick_timing                       # defaults: host/port from expConfig c["QICK"]
    python -m tools.qick_timing --host 192.168.0.72 --port 1234 --repeats 20
    python -m tools.qick_timing --template Echo --waits 1e-6 5e-6 20e-6

No hardware is driven beyond the QICK socket; safe to run while the rest of the rig is idle. Requires
the board server up (``run_server.py``) and no other client holding the single-client socket.
"""
import argparse
import statistics
import sys
import time

from devices.qick_awg import (FPGAAWGClient, FPGAAWGManager, build_program,
                              qick_program_duration)
from devices.qick_awg.fpga_awg_client import DEFAULT_HOST, DEFAULT_PORT


def _default_qick_consts():
    """The c["QICK"] defaults from expConfig (host/port/template/params); {} if unavailable."""
    try:
        import expConfig
        return dict(expConfig.expConfig().get("QICK", {}))
    except Exception:  # noqa: BLE001
        return {}


def _params_for(defaults, template, wait_time=None, drive_time=None):
    """A resolved QICK param dict for one program (defaults overlaid with the swept scalar)."""
    p = dict(defaults)
    p.setdefault("freq", 10863.04)
    p.setdefault("gain", 3000)              # nonzero so the compile path is exercised realistically
    p.setdefault("rabi_freq", 7.187e6)
    p.setdefault("phase", 0.0)
    p["template"] = template
    if wait_time is not None:
        p["wait_time"] = wait_time
    if drive_time is not None:
        p["drive_time"] = drive_time
    return p


def _time_call(fn, *args, **kwargs):
    """Run fn(*args) once, return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1e3


def _stats_ms(fn, repeats):
    """Call fn() `repeats` times; return dict(mean/std/min/max) in ms."""
    samples = []
    for _ in range(repeats):
        _, ms = _time_call(fn)
        samples.append(ms)
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def _row(label, s):
    return "  %-28s n=%-3d mean=%8.2f  std=%7.2f  min=%8.2f  max=%8.2f  (ms)" % (
        label, s["n"], s["mean"], s["std"], s["min"], s["max"])


def run_timing(host, port, template, waits, drives, repeats, log=print):
    """Connect to the board and print the prepare / arm / switch / cancel timing table."""
    defaults = _default_qick_consts()

    # Build the set of programs to exercise (distinct keys -> distinct uploads).
    if template.lower() == "rabi":
        progs = [build_program(_params_for(defaults, template, drive_time=d)) for d in drives]
        sweep_label = "drive_time"
        sweep_vals = drives
    else:
        progs = [build_program(_params_for(defaults, template, wait_time=w)) for w in waits]
        sweep_label = "wait_time"
        sweep_vals = waits

    log("=" * 78)
    log("QICK timing -- %s:%d  template=%s  repeats=%d" % (host, port, template, repeats))
    log("  %d program(s) over %s = %s" % (len(progs), sweep_label,
                                          ", ".join("%g" % v for v in sweep_vals)))
    for pr, v in zip(progs, sweep_vals):
        dur_us = qick_program_duration(_params_for(
            defaults, template,
            wait_time=(None if template.lower() == "rabi" else v),
            drive_time=(v if template.lower() == "rabi" else None))) * 1e6
        n_pulses = len(pr.pulses)
        log("    %s=%g -> key=%r  %d pulse(s)  playtime~%.2f us"
            % (sweep_label, v, pr.key, n_pulses, dur_us))
    log("=" * 78)

    client = FPGAAWGClient()

    # connect
    _, ms = _time_call(client.connect, host, port)
    log(_row("connect", {"n": 1, "mean": ms, "std": 0.0, "min": ms, "max": ms}))

    try:
        # set_trigger_mode
        _, ms = _time_call(client.set_trigger_mode, "external")
        log(_row("set_trigger_mode(external)", {"n": 1, "mean": ms, "std": 0.0, "min": ms, "max": ms}))

        # prepare: clear, then upload each program's waveforms + program; time per program + total.
        for fn in (client.delete_all_envelope_data, client.delete_all_waveform_cfg,
                   client.delete_all_programs):
            _time_call(fn)
        log("-- prepare (upload; once per scan) " + "-" * 43)
        key_to_progname = {}
        t_batch0 = time.perf_counter()
        for i, pr in enumerate(progs):
            prog_name = "prog%03d" % i
            t0 = time.perf_counter()
            for pname, cfg in pr.pulses.items():
                ns = "p%03d_%s" % (i, pname)
                nc = dict(cfg); nc["name"] = ns
                import json as _json
                client.upload_waveform_cfg(_json.dumps(nc).encode("utf-8"), ns)
            # program structure with namespaced pulse names
            from devices.qick_awg.simple_pulse import compile_chn, simple_prog_cfg
            name_map = {pn: "p%03d_%s" % (i, pn) for pn in pr.pulses}
            struct = compile_chn(pr.channels, name_map=name_map)
            import json as _json
            client.upload_program(_json.dumps(simple_prog_cfg(prog_name, struct)).encode("utf-8"),
                                  prog_name)
            key_to_progname[pr.key] = prog_name
            log("  upload %-10s (%d pulse(s)) = %8.2f ms" % (
                prog_name, len(pr.pulses), (time.perf_counter() - t0) * 1e3))
        log("  batch upload total (%d programs) = %8.2f ms" % (
            len(progs), (time.perf_counter() - t_batch0) * 1e3))

        # arm: stop+start, same program, M times.
        first = progs[0].key
        arm_same = _stats_ms(lambda: (client.stop_program(), client.start_program(
            key_to_progname[first])), repeats)
        log("-- arm / cancel (per shot) " + "-" * 51)
        log(_row("arm SAME (stop+start)", arm_same))

        # arm: switch between programs (alternate), M times.
        if len(progs) > 1:
            names = [key_to_progname[p.key] for p in progs]
            state = {"i": 0}

            def _switch():
                state["i"] = (state["i"] + 1) % len(names)
                client.stop_program()
                client.start_program(names[state["i"]])
            arm_switch = _stats_ms(_switch, repeats)
            log(_row("arm SWITCH (alternate)", arm_switch))

        # cancel: stop alone (board is firing after the last start).
        _, ms = _time_call(client.stop_program)
        log(_row("cancel (stop_program)", {"n": 1, "mean": ms, "std": 0.0, "min": ms, "max": ms}))

    finally:
        _, ms = _time_call(client.disconnect)
        log(_row("disconnect", {"n": 1, "mean": ms, "std": 0.0, "min": ms, "max": ms}))
    log("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Time the QICK prepare/arm/switch/cancel path.")
    d = _default_qick_consts()
    ap.add_argument("--host", default=d.get("host", DEFAULT_HOST))
    ap.add_argument("--port", type=int, default=int(d.get("port", DEFAULT_PORT)))
    ap.add_argument("--template", default=d.get("template", "Ramsey"))
    ap.add_argument("--repeats", type=int, default=20, help="arm/switch repeats for stats")
    ap.add_argument("--waits", type=float, nargs="+", default=[1e-6, 5e-6, 20e-6],
                    help="Ramsey/Echo wait_time values (s) -> one program each")
    ap.add_argument("--drives", type=float, nargs="+", default=[100e-9, 500e-9, 2e-6],
                    help="Rabi drive_time values (s) -> one program each")
    args = ap.parse_args(argv)
    try:
        run_timing(args.host, args.port, args.template, args.waits, args.drives, args.repeats)
    except Exception as e:  # noqa: BLE001
        print("qick_timing failed: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
