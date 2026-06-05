"""preflight_hwtest1.py -- HARDWARE-FREE pre-flight for LIVE-test #1.

Compiles the smallest runnable seq (``get_my_seq``) + a known multi-channel reference
through the REAL libnacs engine and ASSERTS the device-codegen artifacts exist and are
well-shaped -- WITHOUT ever firing the FPGA or the NI card and WITHOUT opening the camera.
Safe to run while the live pyctrl backend (or MATLAB) is up: it runs in a SEPARATE process
and sends no command to the board.

WHAT IS / IS NOT TOUCHED (verified against pyctrl source AND the libnacs C++ on 2026-06-04):
  * COMPILE-ONLY (no hardware): SeqConfig.load_real, seq_manager.load_config_string,
    ExpSeq.generate (create_sequence + get_nidaq_channel_info), eseq.init_run(), eseq.pre_run(),
    get_zynq_bytecode/_clock, run_seq2._reshape_sample_major, nidaq_runner._to_channel_major.
      - C++ proof: the zynq Backend has NO init_run override (base no-op, device.cpp:78-80);
        pre_run() only runs bc_gen->generate() (backend.cpp:633-659, NO ZMQ). The ONLY calls
        that send a message to the FPGA are start()/run_bytecode ("run_seq"), wait()
        ("wait_seq"), and cancel() -- none called here. get_zynq_bytecode/_clock key off
        host_seq.cur_seq_idx (set by init_run), so init_run+pre_run are REQUIRED to populate
        them -- still board-free. (A passive, command-free ZMQ DEALER connect happens at
        create_sequence; it sends nothing and cannot trigger a run.)
  * NEVER CALLED HERE (the only hardware-firing calls): pyseq.start()/wait()/post_run()/cancel()
    and NiDAQRunner.run()/.wait(). No camera open. NOTE get_nidaq_data is NOT called on the
    smallest seq: the real engine RAISES "Device NiDAQ cannot be found" for a no-NiDAQ seq, so
    the faithful None-guard is "ni_channels is empty after generate()", which is what run_bseq keys on.

Run (repo root = cwd):
    pyctrl\\.venv-engine\\Scripts\\python.exe pyctrl\\tools\\preflight_hwtest1.py

Exit 0 = all HARD checks PASS (WARN/diagnostic lines do not fail the run). The libnacs+libzmq
DLL-detach wedge hangs normal CPython exit once the engine is loaded, so we TerminateProcess at
the very end (after all results print) -- same trick as tests/conftest.py. That is NOT a crash.
"""

import json
import os
import re
import sys
import traceback

# --- bootstrap sys.path exactly like tests/conftest.py + pyproject pythonpath ---
_THIS = os.path.dirname(os.path.abspath(__file__))          # .../pyctrl/tools
_PYCTRL = os.path.dirname(_THIS)                            # .../pyctrl
for _p in (_PYCTRL,
           os.path.join(_PYCTRL, "lib"),
           os.path.join(_PYCTRL, "tools"),
           os.path.join(_PYCTRL, "YbSteps"),
           os.path.join(_PYCTRL, "YbSeqs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG_PATH = os.path.join(_PYCTRL, "config.yml")           # the SAME config serve() loads
ENGINE_REF_DIR = os.path.join(_PYCTRL, "tests", "reference_engine")
YBSEQS_REF = os.path.join(_PYCTRL, "tests", "reference_ybseqs", "ybseqs_reference.json")


def _config_values():
    """Read every number we check FROM config.yml (the source of truth) -- never hardcode them.

    config.yml is small + stable; a targeted per-key parse avoids a pyyaml dependency. The
    engine itself parses the SAME file in C++ via load_config_string, so the engine's
    tick_per_sec() must equal the value here -- that equality is check 2.
    """
    txt = open(CONFIG_PATH).read()

    def scalar(key):
        m = re.search(r"^\s*%s:\s*([^\s#]+)" % re.escape(key), txt, re.M)
        return m.group(1) if m else None

    fpga = re.search(r"FPGA1:.*?\burl:\s*([^\s#]+)", txt, re.S)   # FPGA1 block's url (not AWG1's)
    return {
        "tick_per_sec": int(scalar("tick_per_sec")),
        "step_size": int(scalar("step_size")),
        "clock_device": scalar("clock_device"),
        "start_ttl_chn": int(scalar("start_ttl_chn")),
        "fpga_url": fpga.group(1) if fpga else None,
    }


_PASS = 0
_FAIL = 0
_WARN = 0


def check(name, fn):
    """A HARD check: FAIL increments the failure count (exit 1). Never raises out."""
    global _PASS, _FAIL
    try:
        ok, detail = fn()
    except Exception as e:                       # noqa: BLE001
        ok, detail = False, "EXCEPTION: %s\n%s" % (e, traceback.format_exc())
    if ok:
        _PASS += 1
        print("  [PASS] %-32s %s" % (name, detail))
    else:
        _FAIL += 1
        print("  [FAIL] %-32s %s" % (name, detail))
    return ok


def diag(name, fn):
    """A SOFT/diagnostic check: a problem WARNs (does not fail the run) -- used for
    first-contact engine/config codegen observations flagged as 'diagnostic, not a hard
    blocker' (e.g. NI clock-pulse emission on a captured reference seq)."""
    global _PASS, _WARN
    try:
        ok, detail = fn()
    except Exception as e:                       # noqa: BLE001
        ok, detail = False, "EXCEPTION: %s\n%s" % (e, traceback.format_exc())
    if ok:
        _PASS += 1
        print("  [PASS] %-32s %s" % (name, detail))
    else:
        _WARN += 1
        print("  [WARN] %-32s %s" % (name, detail))
    return ok


def _clk_list(x):
    """get_zynq_clock returns None (not []) when a device has no NI clock -- normalize."""
    return list(x) if x is not None else []


def _bc_bytes(x):
    return bytes(x) if x is not None else b""


# =========================================================================== #
def main():
    print("=" * 78)
    print("LIVE-TEST #1 PRE-FLIGHT  (hardware-free: NO start(), NO NI card, NO camera)")
    print("interpreter:", sys.executable)
    print("config.yml :", CONFIG_PATH)
    print("=" * 78)

    import seq_manager
    import compare_bytes
    import run_seq2
    import devices.nidaq.nidaq_runner as nidaq_runner
    import numpy as np

    # ---- 1. engine importable in THIS interpreter -------------------------- #
    def _c1():
        if not seq_manager.engine_available():
            return False, ("libnacs NOT importable -- you are NOT on .venv-engine/Python38; "
                           "every getter would be the dummy (get_zynq_bytecode->b''). ABORT.")
        seq_manager.get()                         # force the lazy import now
        return True, "libnacs engine present in this interpreter"
    if not check("engine_importable", _c1):
        return _summary_and_exit()                # nothing downstream is meaningful

    # ---- 2. configs load + tick == 1e12 (matches the live serve() path) ---- #
    from seq_config import SeqConfig

    def _c2():
        SeqConfig.reset()
        SeqConfig.load_real()                     # real expConfig snapshot (channel aliases/defaults)
        with open(CONFIG_PATH) as f:
            seq_manager.load_config_string(f.read())   # engine channel + timing config
        cfg = _config_values()
        tps = seq_manager.tick_per_sec()
        if tps != cfg["tick_per_sec"]:
            return False, ("engine tick_per_sec()=%r != config.yml tick_per_sec=%r -- the engine did "
                           "not load THIS config.yml (time quantization would differ from the byte ref)."
                           % (tps, cfg["tick_per_sec"]))
        return True, "expConfig + config.yml loaded; tick_per_sec=%d (== config.yml)" % tps
    if not check("configs_load_tick_1e12", _c2):
        return _summary_and_exit()                # the build below needs tick loaded

    mgr = seq_manager.get()

    # ---- 3. THE ONE RULE: get_my_seq serialize() == committed MATLAB bytes -- #
    from exp_seq import ExpSeq
    from get_my_seq import get_my_seq

    def _c3():
        if not os.path.exists(YBSEQS_REF):
            return False, "no committed YbSeqs reference at %s" % YBSEQS_REF
        with open(YBSEQS_REF) as f:
            refs = {e["name"]: bytes.fromhex(e["bytes"])
                    for e in json.load(f) if e.get("status") == "ok"}
        want = refs.get("get_my_seq")
        if want is None:
            return False, "get_my_seq missing from the committed ok-corpus (ybseqs_reference.json)"
        got = bytes(get_my_seq(ExpSeq()).serialize())   # nargin-1: takes a configured ExpSeq
        if got != want:
            d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
            return False, ("get_my_seq bytes != MATLAB reference (%d vs %d); first diff: %s"
                           % (len(got), len(want), d))
        if bytes(get_my_seq(ExpSeq()).serialize()) != got:
            return False, "get_my_seq build NOT repeatable"
        return True, "get_my_seq serialize() == MATLAB reference, %d bytes (ONE RULE holds)" % len(got)
    check("smallest_seq_byte_equal_matlab", _c3)

    # ---- 4. engine COMPILES get_my_seq via generate() + NO NI (None-guard) -- #
    # Faithful to the live loop (prepare_seq -> ExpSeq(); seqfn(s); s.generate()). generate()
    # create_sequence's the bytes AND populates ni_channels from get_nidaq_channel_info. The live
    # NI None-guard keys on ni_channels being EMPTY -- NOT on get_nidaq_data (which the real engine
    # RAISES "Device NiDAQ cannot be found" on for a no-NiDAQ seq). This also proves get_my_seq
    # generate()s cleanly on the real engine -- it would CRASH here if get_nidaq_channel_info raised.
    gen_seq = {"val": None}

    def _c4():
        s = get_my_seq(ExpSeq())
        s.generate()                               # create_sequence + get_nidaq_channel_info
        if s.pyseq is None:
            return False, "generate() left pyseq None (engine did not compile get_my_seq)"
        if s.ni_channels:
            return False, ("get_my_seq ni_channels=%r -- expected EMPTY (no NI); the live loop "
                           "would wrongly arm the NI card for an analog-free seq." % s.ni_channels)
        gen_seq["val"] = s
        return True, ("generate() OK; pyseq compiled; ni_channels=[] -> live loop SKIPS the NI arm "
                      "(get_nidaq_channel_info returned None gracefully, no crash)")
    check("engine_generate_no_ni", _c4)

    # ---- 5. FPGA1 zynq bytecode present for get_my_seq (init_run+pre_run) --- #
    def _c5():
        s = gen_seq["val"]
        if s is None:
            return False, "skipped: get_my_seq did not generate"
        pyseq = s.pyseq
        pyseq.init_run()                           # board-free (C++ verified)
        pyseq.pre_run()                            # board-free: bc_gen->generate, NO ZMQ
        bc = _bc_bytes(pyseq.get_zynq_bytecode("FPGA1"))
        clk = _clk_list(pyseq.get_zynq_clock("FPGA1"))
        if len(bc) == 0:
            return False, "FPGA1 bytecode EMPTY for get_my_seq (expected non-empty compiled bytecode)"
        return True, ("FPGA1 bytecode=%d bytes (compiled OK); clock pulses=%d "
                      "(0 expected -- get_my_seq has no NI analog)" % (len(bc), len(clk)))
    check("zynq_bytecode_present_smallest", _c5)

    # ---- 6. NI transpose column-order -- STRONG synthetic multi-channel proof #
    # (The real-engine refs have only 1 NI channel -> a degenerate transpose; this synthetic
    #  [3 samps x 2 chns] case is what actually distinguishes a correct .T from an omitted one.)
    def _c6():
        m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]   # sample-major [nsamps=3, nchns=2]
        cm = np.asarray(nidaq_runner._to_channel_major(m))
        if tuple(cm.shape) != (2, 3):
            return False, "channel-major shape %r != (2,3) -- transpose direction WRONG" % (cm.shape,)
        if not cm.flags["C_CONTIGUOUS"]:
            return False, "channel-major buffer not C-contiguous (nidaqmx needs C order)"
        if list(cm[0]) != [1.0, 3.0, 5.0] or list(cm[1]) != [2.0, 4.0, 6.0]:
            return False, "ao-channel order wrong: cm=%r (expected ch0=[1,3,5], ch1=[2,4,6])" % cm.tolist()
        return True, "[nsamps,nchns]->[nchns,nsamps] correct: ch0=all-ch0-samps, C-contiguous"
    check("nidaq_transpose_column_order", _c6)

    # ---- 7. NI rate DERIVED FROM config.yml matches the code constant ------ #
    # The NI sample clock is EXTERNAL: the FPGA (clock_device) emits one edge every
    # NiDAQ.step_size ticks, so the authoritative rate = tick_per_sec / step_size. The card is
    # merely told to LISTEN at nidaq_runner._RATE -- a HARDCODED constant that can DRIFT from
    # config.yml. Derive the truth from config and assert the constant matches it.
    def _c7():
        cfg = _config_values()
        if cfg["step_size"] is None or not cfg["clock_device"]:
            return False, "could not find NiDAQ step_size / clock_device in config.yml"
        tps = cfg["tick_per_sec"]                  # from config.yml (== engine, checked in c2)
        step_ticks = cfg["step_size"]              # from config.yml
        expected = tps / step_ticks                # e.g. 1e12 / 2.5e6 = 400000.0
        sc = SeqConfig.get()
        clocks = dict(getattr(sc, "ni_clocks", {}) or {})
        start = dict(getattr(sc, "ni_start", {}) or {})
        problems = []
        if nidaq_runner._RATE != expected:
            problems.append("nidaq_runner._RATE=%g != config-derived %g (tick_per_sec/step_size="
                            "%d/%d) -- the HARDCODED rate has DRIFTED from config.yml"
                            % (nidaq_runner._RATE, expected, tps, step_ticks))
        if cfg["clock_device"] != "FPGA1":
            problems.append("config.yml clock_device=%s (expected FPGA1)" % cfg["clock_device"])
        if clocks.get("Dev1") != "PFI0" or start.get("Dev1") != "PFI1":
            problems.append("PFI routing clocks=%r start=%r (expected Dev1->PFI0, Dev1->PFI1)"
                            % (clocks, start))
        if problems:
            return False, "; ".join(problems)
        return True, ("NI rate=%g Hz == config.yml (tick_per_sec %d / step_size %d); clock_device=%s; "
                      "FPGA start_ttl_chn=%d -> NI PFI1 trig; board=%s; route Dev1/PFI0 clock"
                      % (expected, tps, step_ticks, cfg["clock_device"], cfg["start_ttl_chn"],
                         cfg["fpga_url"]))
    check("nidaq_rate_matches_config", _c7)

    # ---- 8. submit payload: NumImages=0 actually propagates (handle semantics) #
    def _c8():
        from scan_group import ScanGroup
        from scan_export import scangroup_to_descriptor
        g = ScanGroup()
        g.runp().NumImages = 0                     # assignment under test (DynProps handle)
        back = g.runp().NumImages(99)              # fresh-handle read-back (99 = sentinel default)
        desc = scangroup_to_descriptor(g, "get_my_seq", opts={"rep": 1}, label="HWTEST1")
        ni = desc.get("runp", {}).get("NumImages", "MISSING")
        if int(back) != 0:
            return False, ("runp().NumImages=0 did NOT persist across runp() calls (read back %r) -- "
                           "the camera would ARM; pass NumImages via the descriptor instead." % (back,))
        if desc.get("seq") != "get_my_seq":
            return False, "descriptor seq=%r (expected get_my_seq)" % (desc.get("seq"),)
        if ni == "MISSING" or int(ni) != 0:
            return False, ("descriptor runp NumImages=%r (expected 0) -> _num_images falls back to 1 "
                           "and the camera ARMS for the FPGA-only smoke" % (ni,))
        return True, ("NumImages=0 persists + lands in descriptor (runp.NumImages=0); seq=get_my_seq, "
                      "opts.rep=1 -> camera will NOT arm")
    check("descriptor_numimages_0", _c8)

    # ====================== DIAGNOSTIC (soft) checks ========================= #
    # First-contact engine/config codegen observations on a captured multi-channel reference
    # (FPGA1 TTL+DDS + NiDAQ). A problem here WARNs, it does not block the live run.
    print("  " + "-" * 40 + " diagnostics (non-blocking)")
    multi = {"eseq": None, "nchns": 0}

    def _compile_multi():
        ref = os.path.join(ENGINE_REF_DIR, "eng_multi.bin")
        if not os.path.exists(ref):
            return None
        e = mgr.create_sequence(bytearray(compare_bytes.load(ref)))
        info = e.get_nidaq_channel_info("NiDAQ")
        multi["nchns"] = len(info or [])
        e.init_run()
        e.pre_run()
        multi["eseq"] = e
        return e

    def _d_clock():
        e = multi["eseq"] or _compile_multi()
        if e is None:
            return False, "eng_multi.bin missing -- cannot positively prove NI clock pulses"
        bc = _bc_bytes(e.get_zynq_bytecode("FPGA1"))
        clk = _clk_list(e.get_zynq_clock("FPGA1"))
        if len(bc) == 0:
            return False, "eng_multi FPGA1 bytecode EMPTY (unexpected -- it has FPGA1 channels)"
        if len(clk) == 0:
            return False, ("eng_multi FPGA1 NI clock pulses=0 -- the reference has only 1 NI sample, "
                           "so revisit clock emission on the REAL analog run (step 5) before trusting it")
        return True, "eng_multi FPGA1 bytecode=%d bytes, NI clock pulses=%d (present)" % (len(bc), len(clk))
    diag("ni_clock_pulses_present", _d_clock)

    def _d_flow():
        e = multi["eseq"] or _compile_multi()
        if e is None:
            return False, "eng_multi.bin missing -- cannot validate the real-engine transpose flow"
        nchns = multi["nchns"]
        if not nchns:
            return False, "eng_multi reports 0 NI channels (get_nidaq_channel_info)"
        ni = e.get_nidaq_data("NiDAQ")
        if ni is None:
            return False, "eng_multi get_nidaq_data returned None (expected data for a NiDAQ seq)"
        ni = list(ni)
        if len(ni) % nchns != 0:
            return False, "ni len %d not a multiple of nchns %d" % (len(ni), nchns)
        nsamps = len(ni) // nchns
        m = np.asarray(run_seq2._reshape_sample_major(ni, nchns))
        cm = np.asarray(nidaq_runner._to_channel_major(m))
        if tuple(cm.shape) != (nchns, nsamps):
            return False, "channel-major shape %r != (%d,%d)" % (cm.shape, nchns, nsamps)
        if not cm.flags["C_CONTIGUOUS"]:
            return False, "real-engine channel-major buffer not C-contiguous"
        if not np.array_equal(cm[0], m[:, 0]):
            return False, "real-engine transpose mismatch cm[0] != m[:,0]"
        note = " (nchns=1: degenerate -- shape only)" if nchns == 1 else ""
        return True, ("real-engine NI flow ok: nchns=%d nsamps=%d, reshape->transpose C-contig%s"
                      % (nchns, nsamps, note))
    diag("real_engine_nidaq_flow", _d_flow)

    def _d_refs():
        if not os.path.isdir(ENGINE_REF_DIR):
            return False, "no reference_engine dir"
        files = sorted(f for f in os.listdir(ENGINE_REF_DIR) if f.endswith(".bin"))
        if not files:
            return False, "no *.bin engine references"
        bad = []
        for f in files:
            raw = compare_bytes.load(os.path.join(ENGINE_REF_DIR, f))
            if compare_bytes.encode(compare_bytes.decode(raw)) != raw:
                bad.append(f)
        if bad:
            return False, "byte round-trip drift in: %s" % ", ".join(bad)
        return True, "%d engine refs decode->encode byte-stable (compare_bytes faithful)" % len(files)
    diag("engine_refs_roundtrip", _d_refs)

    return _summary_and_exit()


def _summary_and_exit():
    print("-" * 78)
    print("PRE-FLIGHT RESULT: %d passed, %d failed, %d warn" % (_PASS, _FAIL, _WARN))
    status = 0 if _FAIL == 0 else 1
    if _FAIL == 0:
        print("ALL HARD CHECKS PASS -- compile/codegen path is sound; cleared for the LIVE runbook.")
        if _WARN:
            print("(%d diagnostic WARN above -- review before the NI/analog run, not a blocker.)" % _WARN)
    else:
        print("FAILURES PRESENT -- do NOT proceed to the live run until resolved.")
    print("-" * 78)
    _hard_exit(status)


def _hard_exit(status):
    """TerminateProcess past the libnacs+libzmq DLL-detach wedge (conftest.py parity)."""
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32" and ("libnacs" in sys.modules
                                    or "libnacs.expseq_manager" in sys.modules):
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x0001, False, os.getpid())   # PROCESS_TERMINATE
        k32.TerminateProcess(h, int(status))
    sys.exit(status)


if __name__ == "__main__":
    main()
