"""strobe_scope_capture.py -- capture per-beam photodiode response of the 800 ns strobe.

Why: StrobeImag399Step commands 399/556 AOMs at ~800 ns, but the AOM optical rise is ~1 us
(DDS ~500 ns). So the optical pulse is a partial triangle, not a square -- you must MEASURE the
real on-fraction / rise time on a photodiode, not assume it.

There is NO scope-sync TTL wired, so the scope triggers on the FIRST beam PD pulse itself. Two modes:

  read   -- read current scope state (comms check).
  pulse  -- trigger SINGLE on a PD channel, zoom one pulse (--tb ~400ns); report per-channel
            10-90 % rise time + on-fraction within one ImageTime window. (screen-resolution NORM read)
  train  -- trigger SINGLE on a PD channel, deep-memory RAW capture of the WHOLE 100 ms collect on
            ONE channel; report pulse count, mean rise, on-fraction, and early-vs-late high level
            (intensity drift over the train). Saves the full waveform to <out>.npz.

Run THIS first (it arms + waits), THEN fire the strobe scan so the scope triggers on the first pulse.

Usage:
    python strobe_scope_capture.py --read
    python strobe_scope_capture.py --mode pulse  --trig CHANnel1 --level 0.1 --tb 400e-9 --channels 1 2 3
    python strobe_scope_capture.py --mode train  --trig CHANnel1 --level 0.1 --tb 10e-3 \
        --raw-channel 1 --mdepth 1200000 --image-time 800e-9 --out tmp/strobe_train_r1

Uses the in-repo scope_control submodule's raw-SCPI ScpiScope (stdlib socket + numpy; no pyvisa).
Run with any python that has numpy (e.g. the yb_analysis env or base anaconda).
"""
import argparse
import json
import os
import sys
import time

REPO = r"c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
sys.path.insert(0, os.path.join(REPO, "scope_control"))

import numpy as np  # noqa: E402
from scope_control.scpi_tcp import ScpiScope  # noqa: E402


def _edge_metrics(volts, dt, image_time):
    """10-90 % rise time (s) of the first rising edge + on-fraction within one image window."""
    v = np.asarray(volts, float)
    if v.size < 5:
        return float("nan"), float("nan"), 0.0
    lo, hi = float(v.min()), float(v.max())
    vpp = hi - lo
    if vpp <= 0:
        return float("nan"), float("nan"), 0.0
    t10, t90, half = lo + 0.10 * vpp, lo + 0.90 * vpp, lo + 0.5 * vpp
    above10 = np.where(v >= t10)[0]
    if above10.size == 0:
        return float("nan"), float("nan"), vpp
    i10 = int(above10[0])
    above90 = np.where(v[i10:] >= t90)[0]
    rise = float(above90[0] * dt) if above90.size else float("nan")
    nwin = max(1, int(round(image_time / dt))) if dt and dt > 0 else v.size
    win = v[i10:i10 + nwin]
    on_frac = float(np.mean(win >= half)) if win.size else float("nan")
    return rise, on_frac, vpp


def _wait_trigger(sc, timeout):
    """Poll :TRIGger:STATus? until STOP/TD (captured) or timeout. Returns the final status."""
    t0 = time.time(); stat = ""
    while time.time() - t0 < timeout:
        r = sc.command(":TRIGger:STATus?")
        stat = (r.get("response") or "").strip().upper() if r.get("ok") else ""
        if stat in ("STOP", "TD"):
            return stat
        time.sleep(0.3)
    return stat


def do_read(sc, channels):
    st = sc.read_state(channels=channels, max_plot_points=1200)
    if not st.get("ok"):
        print("READ FAILED:", st.get("error")); return
    print("scope:", st["idn"], "| tb=%s s/div  srate=%s" % (
        st["timebase_s_per_div"], st["sample_rate_hw"]))
    print("trigger:", st["trigger"])
    for ch in channels:
        c = st["channels"].get(str(ch), {})
        if not c.get("on"):
            print("  CH%d off" % ch); continue
        print("  CH%d  scale=%s V/div  coupling=%s  vpp=%s vmin=%s vmax=%s n=%s" % (
            ch, c.get("scale"), c.get("coupling"), c.get("vpp"),
            c.get("vmin"), c.get("vmax"), c.get("n")))


def do_pulse(sc, args):
    chans = args.channels
    for ch in chans:
        sc.set_channel(ch, display=True, coupling="DC")
    sc.set_timebase(args.tb)
    if args.offset is not None:
        sc.set_timebase_offset(args.offset)
    sc.set_trigger(sweep="SINGLE", source=args.trig, slope="POSitive", level=args.level)
    if not sc.single().get("ok"):
        print("ARM FAILED"); return
    print("ARMED single @ %s tb=%g s/div trig=%s(rising,%gV). >>> FIRE the strobe scan (%ds)..." % (
        sc.host, args.tb, args.trig, args.level, args.timeout))
    stat = _wait_trigger(sc, args.timeout)
    if stat not in ("STOP", "TD"):
        print("  no trigger (status=%r); reading current frame anyway." % stat)
    time.sleep(0.3)
    st = sc.read_state(channels=chans, max_plot_points=1200)
    if not st.get("ok"):
        print("READ FAILED:", st.get("error")); return
    print("\n  %-5s %12s %10s %9s %6s" % ("CH", "rise(10-90)", "on-frac", "vpp(V)", "n"))
    out = {"host": sc.host, "idn": st["idn"], "mode": "pulse", "tb": st["timebase_s_per_div"],
           "image_time": args.image_time, "channels": {}}
    for ch in chans:
        c = st["channels"].get(str(ch), {})
        if not c.get("on") or c.get("no_data") or "volts" not in c:
            print("  CH%-3d  (off / no data)" % ch); continue
        rise, on_frac, vpp = _edge_metrics(c["volts"], c.get("dt", 0.0), args.image_time)
        rise_us = rise * 1e6 if rise == rise else float("nan")
        print("  CH%-3d %9.3f us %9.1f %% %8.4f %6d" % (ch, rise_us, on_frac * 100, vpp, c.get("n", 0)))
        out["channels"][ch] = {"rise_s": rise, "on_frac": on_frac, "vpp": vpp,
                               "dt": c.get("dt"), "t0": c.get("t0"), "volts": c["volts"]}
    _save(args.out, out)
    print("\nNOTE: rise ~1 us at ImageTime=800 ns => AOM-limited; on-frac < 100 %% => beam never reaches "
          "full intensity in the window.")


def _read_raw(sc, ch, npts, chunk=250000):
    """Deep-memory RAW read of one channel. Returns (volts ndarray, dt). Scope must be STOPped."""
    with sc._lock:
        sc._ensure()
        sc._write(":STOP")
        sc._write(":WAVeform:SOURce CHANnel%d" % ch)
        sc._write(":WAVeform:MODE RAW")
        sc._write(":WAVeform:FORMat BYTE")
        pre = sc._query(":WAVeform:PREamble?").strip().split(",")
        xinc = float(pre[4]); yinc = float(pre[7]); yorig = float(pre[8]); yref = float(pre[9])
        parts = []
        start = 1
        while start <= npts:
            stop = min(start + chunk - 1, npts)
            sc._write(":WAVeform:STARt %d" % start)
            sc._write(":WAVeform:STOP %d" % stop)
            sc._write(":WAVeform:DATA?")
            parts.append(np.frombuffer(sc._read_ieee_block(), dtype=np.uint8))
            start = stop + 1
        codes = np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)
    volts = (codes - yorig - yref) * yinc
    return volts, xinc


def _train_stats(volts, dt, image_time):
    """Pulse-train metrics: count, mean period, mean 10-90% rise, on-fraction, early-vs-late high."""
    v = np.asarray(volts, float)
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    vpp = hi - lo
    if vpp <= 0 or v.size < 10:
        return {"vpp": vpp, "n_pulses": 0}
    half = lo + 0.5 * vpp
    above = v >= half
    rises = np.where(np.diff(above.astype(np.int8)) == 1)[0] + 1  # rising-edge indices
    falls = np.where(np.diff(above.astype(np.int8)) == -1)[0] + 1
    n = rises.size
    period = float(np.mean(np.diff(rises)) * dt) if n > 1 else float("nan")
    # high level per pulse = mean of samples between rise and the next fall above half
    highs = []
    for r in rises:
        f = falls[falls > r]
        seg = v[r:(int(f[0]) if f.size else min(r + max(1, int(image_time / dt)), v.size))]
        if seg.size:
            highs.append(float(seg.mean()))
    highs = np.asarray(highs)
    onfrac = float(np.mean(above))  # fraction of WHOLE train above half (duty incl. recool)
    early = float(np.mean(highs[: max(1, n // 20)])) if highs.size else float("nan")
    late = float(np.mean(highs[-max(1, n // 20):])) if highs.size else float("nan")
    return {"vpp": vpp, "n_pulses": int(n), "period_s": period, "duty_frac": onfrac,
            "high_early": early, "high_late": late,
            "drift_pct": (100.0 * (late - early) / early) if early else float("nan")}


def do_train(sc, args):
    ch = args.raw_channel
    # only the captured channel on -> max mem depth for one channel.
    for c in (1, 2, 3, 4):
        sc.set_channel(c, display=(c == ch), coupling="DC")
    sc.run()
    sc.set_acquire(mdepth=str(int(args.mdepth)))
    sc.set_timebase(args.tb)
    if args.offset is not None:
        sc.set_timebase_offset(args.offset)
    sc.set_trigger(sweep="SINGLE", source=args.trig, slope="POSitive", level=args.level)
    if not sc.single().get("ok"):
        print("ARM FAILED"); return
    print("ARMED single @ %s tb=%g s/div mdepth=%d CH%d trig=%s(rising,%gV)." % (
        sc.host, args.tb, int(args.mdepth), ch, args.trig, args.level))
    print(">>> FIRE the strobe scan now. Waiting for trigger (%ds)..." % args.timeout)
    stat = _wait_trigger(sc, args.timeout)
    if stat not in ("STOP", "TD"):
        print("  no trigger (status=%r); aborting train capture." % stat); return
    print("  triggered -- reading %d-pt deep memory (may take ~10-60 s)..." % int(args.mdepth))
    volts, dt = _read_raw(sc, ch, int(args.mdepth))
    if volts.size == 0:
        print("  RAW read returned no points."); return
    stats = _train_stats(volts, dt, args.image_time)
    span = volts.size * dt
    print("\n  captured %d pts, dt=%.1f ns, span=%.3f ms" % (volts.size, dt * 1e9, span * 1e3))
    print("  vpp=%.4f V  pulses=%d  period=%.3f us  duty=%.1f %%" % (
        stats["vpp"], stats.get("n_pulses", 0),
        stats.get("period_s", float("nan")) * 1e6, stats.get("duty_frac", float("nan")) * 100))
    print("  high level early=%.4f late=%.4f  drift=%.1f %% over the train" % (
        stats.get("high_early", float("nan")), stats.get("high_late", float("nan")),
        stats.get("drift_pct", float("nan"))))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out + ".npz", volts=volts, dt=dt, channel=ch, **stats)
        print("  saved ->", args.out + ".npz")


def _save(out, obj):
    if not out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".json", "w") as f:
        json.dump(obj, f, indent=2)
    print("saved ->", out + ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.0.41")
    ap.add_argument("--read", action="store_true", help="just read current scope state (comms check)")
    ap.add_argument("--mode", choices=["read", "pulse", "train"], default="pulse")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3],
                    help="pulse mode: channels to digitize (beam PDs)")
    ap.add_argument("--raw-channel", type=int, default=1, help="train mode: which PD channel to deep-capture")
    ap.add_argument("--trig", default="CHANnel1", help="trigger source = the PD channel (no TTL wired)")
    ap.add_argument("--level", type=float, default=0.1, help="trigger level (V) -- set above the PD baseline")
    ap.add_argument("--tb", type=float, default=400e-9, help="timebase s/div (pulse: ~400ns; train: ~10ms)")
    ap.add_argument("--offset", type=float, default=None, help="timebase offset (s)")
    ap.add_argument("--mdepth", type=float, default=1200000, help="train mode: RAW memory depth (1ch)")
    ap.add_argument("--image-time", type=float, default=800e-9, help="ImageTime for on-fraction window")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None, help="output path prefix")
    a = ap.parse_args()
    if a.read:
        a.mode = "read"

    sc = ScpiScope(a.host)
    try:
        if a.mode == "read":
            do_read(sc, a.channels)
        elif a.mode == "train":
            do_train(sc, a)
        else:
            do_pulse(sc, a)
    finally:
        sc.close()


if __name__ == "__main__":
    main()
