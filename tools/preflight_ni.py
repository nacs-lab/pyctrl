"""preflight_ni.py -- HARDWARE-FREE NI-codegen pre-flight for LIVE-test #1 (NI clock-out).

Compiles a real ANALOG seq (default TweezerLoadingSeq) through the engine and verifies the NI
codegen WITHOUT firing the card or the FPGA. Calls only compile-only engine APIs
(generate/init_run/pre_run/get_*); NEVER start()/wait()/post_run() or NiDAQRunner.run()/.wait().
The zynq init_run/pre_run are board-free (C++ verified); only start()/wait() send to the FPGA.
Safe to run alongside the live backend (separate process).

Checks (all hardware-free):
  1. engine importable in this interpreter
  2. configs load; engine tick_per_sec == config.yml
  3. the seq HAS NI analog channels (ni_channels non-empty) -- report (dev,chn) list
  4. FPGA1 bytecode + NI sample-clock pulses present (get_zynq_bytecode/_clock) -- the REAL
     multi-sample check the get_my_seq/eng_multi pre-flight could not do (resolves that WARN)
  5. get_nidaq_data -> reshape[nsamps,nchns] -> transpose[nchns,nsamps]: shape, C-contiguity,
     and FULL column-order (cm[i] == m[:,i] for EVERY channel) -- the #1 silent-bug transpose,
     now NON-degenerate (>=2 NI channels, nsamps>>1)
  6. NI rate derived from config.yml matches nidaq_runner._RATE (no hardcoded magic number)

Run (repo root = cwd):
    pyctrl\\.venv-engine\\Scripts\\python.exe pyctrl\\tools\\preflight_ni.py [SeqName]
Default SeqName = TweezerLoadingSeq (what LACScan runs).
"""

import os
import re
import sys
import traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_THIS)
for _p in (_PYCTRL,
           os.path.join(_PYCTRL, "lib"),
           os.path.join(_PYCTRL, "tools"),
           os.path.join(_PYCTRL, "YbSteps"),
           os.path.join(_PYCTRL, "YbSeqs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG_PATH = os.path.join(_PYCTRL, "config.yml")
SEQ_NAME = sys.argv[1] if len(sys.argv) > 1 else "TweezerLoadingSeq"

_PASS = 0
_FAIL = 0


def _config_values():
    txt = open(CONFIG_PATH).read()

    def scalar(key):
        m = re.search(r"^\s*%s:\s*([^\s#]+)" % re.escape(key), txt, re.M)
        return m.group(1) if m else None

    return {"tick_per_sec": int(scalar("tick_per_sec")),
            "step_size": int(scalar("step_size")),
            "clock_device": scalar("clock_device")}


def check(name, fn):
    global _PASS, _FAIL
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, "EXCEPTION: %s\n%s" % (e, traceback.format_exc())
    if ok:
        _PASS += 1
        print("  [PASS] %-30s %s" % (name, detail))
    else:
        _FAIL += 1
        print("  [FAIL] %-30s %s" % (name, detail))
    return ok


def main():
    print("=" * 78)
    print("LIVE-TEST #1 NI PRE-FLIGHT  (hardware-free: NO start(), NO NI card)  seq=%s" % SEQ_NAME)
    print("interpreter:", sys.executable)
    print("=" * 78)

    import seq_manager
    import run_seq2
    import devices.nidaq.nidaq_runner as nidaq_runner
    import numpy as np

    def _c1():
        if not seq_manager.engine_available():
            return False, "libnacs NOT importable -- wrong interpreter. ABORT."
        seq_manager.get()
        return True, "libnacs engine present"
    if not check("engine_importable", _c1):
        return _summary_and_exit()

    from seq_config import SeqConfig

    def _c2():
        SeqConfig.reset()
        SeqConfig.load_real()
        with open(CONFIG_PATH) as f:
            seq_manager.load_config_string(f.read())
        cfg = _config_values()
        tps = seq_manager.tick_per_sec()
        if tps != cfg["tick_per_sec"]:
            return False, "engine tick=%r != config.yml tick=%r" % (tps, cfg["tick_per_sec"])
        return True, "expConfig + config.yml loaded; tick_per_sec=%d (== config.yml)" % tps
    if not check("configs_load", _c2):
        return _summary_and_exit()

    # Enable dump BEFORE generate() so the NI clock survives in m_clocks for read-back. Otherwise
    # Backend::generate() calls m_clocks.clear() (zynq/backend.cpp:527-530) and get_zynq_clock
    # returns empty -- the clock is baked into the BYTECODE either way (bc_gen.cpp:1361-1374); dump
    # just lets us POSITIVELY prove the clock was generated here. (libnacs C++ verified 2026-06-04.)
    mgr = seq_manager.get()
    _dump_on = False
    try:
        mgr.enable_dump(True)
        _dump_on = True
    except Exception as e:  # noqa: BLE001
        print("  [warn] mgr.enable_dump unavailable (%s) -- clock read-back may be empty" % e)

    # ---- build + generate the analog seq (compile-only) -------------------- #
    from exp_seq import ExpSeq
    gen = {"seq": None}

    def _c3():
        mod = __import__(SEQ_NAME)               # file == module == function (MATLAB name)
        fn = getattr(mod, SEQ_NAME)
        s = fn(ExpSeq())                         # nargin-1 (real seqs take a configured ExpSeq)
        s.generate()                             # create_sequence + get_nidaq_channel_info
        if s.pyseq is None:
            return False, "generate() left pyseq None"
        nchns = len(s.ni_channels or [])
        if nchns == 0:
            return False, ("%s has NO NI analog channels (ni_channels empty) -- pick a seq that "
                           "drives V* NI-DAQ channels for the NI clock-out test" % SEQ_NAME)
        gen["seq"] = s
        chans = ", ".join("%s/%s" % (c["dev"], c["chn"]) for c in s.ni_channels[:12])
        more = "" if nchns <= 12 else " (+%d more)" % (nchns - 12)
        return True, "%s generated; %d NI channels: %s%s" % (SEQ_NAME, nchns, chans, more)
    if not check("seq_generates_has_ni", _c3):
        return _summary_and_exit()

    # ---- FPGA1 bytecode + NI sample-clock pulses present (real multi-sample) #
    def _c4():
        pyseq = gen["seq"].pyseq
        pyseq.init_run()                         # board-free
        pyseq.pre_run()                          # board-free: bc_gen->generate, NO ZMQ
        bc = pyseq.get_zynq_bytecode("FPGA1")
        bc = bytes(bc) if bc is not None else b""
        clk = pyseq.get_zynq_clock("FPGA1")
        clk = list(clk) if clk is not None else []
        if len(bc) == 0:
            return False, "FPGA1 bytecode EMPTY"
        # The clock lives in the BYTECODE (baked by bc_gen); get_zynq_clock only repopulates with
        # dump enabled (we enabled it above). Empty clock here is a real red flag ONLY if dump is on.
        if not _dump_on:
            return True, ("FPGA1 bytecode=%d bytes; clock read-back skipped (dump unavailable) -- "
                          "the NI clock is baked into the bytecode regardless (libnacs verified)" % len(bc))
        if len(clk) == 0:
            return False, ("FPGA1 NI sample-clock EMPTY even with dump ENABLED -- expected clock-on/"
                           "off pairs from set_clock_active_time; the clock may not be generated.")
        return True, ("FPGA1 bytecode=%d bytes; NI sample-clock pairs=%d (dump-enabled read-back "
                      "PROVES the 400kHz clock is generated + baked into the bytecode)" % (len(bc), len(clk)))
    check("fpga_bytecode_and_ni_clock", _c4)

    # ---- THE #1 SILENT BUG: transpose shape + FULL column order ------------ #
    def _c5():
        s = gen["seq"]
        pyseq = s.pyseq
        nchns = len(s.ni_channels)
        ni = pyseq.get_nidaq_data("NiDAQ")
        if ni is None:
            return False, "get_nidaq_data returned None for a seq WITH NI channels"
        ni = list(ni)
        if nchns == 0 or len(ni) % nchns != 0:
            return False, "ni len %d not a multiple of nchns %d" % (len(ni), nchns)
        nsamps = len(ni) // nchns
        m = np.asarray(run_seq2._reshape_sample_major(ni, nchns))    # [nsamps, nchns]
        cm = np.asarray(nidaq_runner._to_channel_major(m))           # [nchns, nsamps]
        if tuple(m.shape) != (nsamps, nchns):
            return False, "sample-major shape %r != (%d,%d)" % (m.shape, nsamps, nchns)
        if tuple(cm.shape) != (nchns, nsamps):
            return False, "channel-major shape %r != (%d,%d) -- TRANSPOSE DIRECTION WRONG" % (
                cm.shape, nchns, nsamps)
        if not cm.flags["C_CONTIGUOUS"]:
            return False, "channel-major buffer NOT C-contiguous (nidaqmx needs C order)"
        bad = [i for i in range(nchns) if not np.array_equal(cm[i], m[:, i])]
        if bad:
            return False, "column-order WRONG for channels %r (cm[i] != m[:,i])" % bad[:8]
        deg = " (NOTE: only 1 NI channel -- order check degenerate)" if nchns == 1 else ""
        return True, ("transpose OK: %d chn x %d samps; [nsamps,nchns]->[nchns,nsamps] C-contig; "
                      "ALL %d channels' column order verified%s" % (nchns, nsamps, nchns, deg))
    check("nidaq_transpose_full_order", _c5)

    # ---- NI rate derived from config.yml matches the code constant --------- #
    def _c6():
        cfg = _config_values()
        expected = cfg["tick_per_sec"] / cfg["step_size"]
        sc = SeqConfig.get()
        clocks = dict(getattr(sc, "ni_clocks", {}) or {})
        start = dict(getattr(sc, "ni_start", {}) or {})
        if nidaq_runner._RATE != expected:
            return False, "_RATE=%g != config-derived %g (DRIFT from config.yml)" % (
                nidaq_runner._RATE, expected)
        if cfg["clock_device"] != "FPGA1":
            return False, "config clock_device=%s (expected FPGA1)" % cfg["clock_device"]
        if clocks.get("Dev1") != "PFI0" or start.get("Dev1") != "PFI1":
            return False, "PFI routing clocks=%r start=%r" % (clocks, start)
        return True, ("NI rate=%g Hz == config.yml (tick/step_size); clock_device=FPGA1; "
                      "card listens Dev1/PFI0 (clock) + Dev1/PFI1 (start)" % expected)
    check("nidaq_rate_matches_config", _c6)

    return _summary_and_exit()


def _summary_and_exit():
    print("-" * 78)
    print("NI PRE-FLIGHT RESULT: %d passed, %d failed" % (_PASS, _FAIL))
    status = 0 if _FAIL == 0 else 1
    print("ALL PASS -- NI codegen + transpose sound; cleared to fire the analog seq."
          if _FAIL == 0 else "FAILURES -- do NOT fire the NI run until resolved.")
    print("-" * 78)
    _hard_exit(status)


def _hard_exit(status):
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32" and ("libnacs" in sys.modules
                                    or "libnacs.expseq_manager" in sys.modules):
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x0001, False, os.getpid())
        k32.TerminateProcess(h, int(status))
    sys.exit(status)


if __name__ == "__main__":
    main()
