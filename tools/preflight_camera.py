"""preflight_camera.py -- HARDWARE-FREE check that the imaging seq triggers the Orca camera.

Engine-free (override_tick_per_sec, like the byte tests -- no libnacs, no DLL-detach wedge):
builds the imaging seq, serializes, decodes, and confirms the Orca trigger channel
(FPGA1/TTL54 = TTLOrcaTrig) is present AND driven (has outputs), and that NumImages propagates
to the descriptor. The actual capture (external rising-edge trigger -> frame -> store_imgs) is
the LIVE run-loop test; the camera hardware/wrapper was already live-verified (runtime-design.md).

Run:
    pyctrl\\.venv-engine-py312\\Scripts\\python.exe pyctrl\\tools\\preflight_camera.py [SeqName] [NumImages]
Default SeqName=TweezerLoadingSeq, NumImages=1 (what LACScan images).
"""

import os
import sys
import traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_THIS)
for _p in (_PYCTRL, os.path.join(_PYCTRL, "lib"), os.path.join(_PYCTRL, "tools"),
           os.path.join(_PYCTRL, "YbSteps"), os.path.join(_PYCTRL, "YbSeqs"),
           os.path.join(_PYCTRL, "YbExptCtrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "TweezerLoadingSeq"
NUMIMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 1

_PASS = 0
_FAIL = 0


def check(name, fn):
    global _PASS, _FAIL
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, "EXCEPTION: %s\n%s" % (e, traceback.format_exc())
    if ok:
        _PASS += 1
        print("  [PASS] %-28s %s" % (name, detail))
    else:
        _FAIL += 1
        print("  [FAIL] %-28s %s" % (name, detail))
    return ok


def main():
    print("=" * 78)
    print("LIVE-TEST #1 CAMERA PRE-FLIGHT (hardware-free)  seq=%s NumImages=%d" % (SEQ, NUMIMAGES))
    print("=" * 78)

    import seq_manager
    import compare_bytes
    from seq_config import SeqConfig
    from exp_seq import ExpSeq

    decoded = {"seq": None}

    def _c1():
        SeqConfig.reset()
        SeqConfig.load_real()
        seq_manager.override_tick_per_sec(int(1e12))   # engine-free build (byte-test path)
        mod = __import__(SEQ)
        fn = getattr(mod, SEQ)
        blob = fn(ExpSeq()).serialize()
        decoded["seq"] = compare_bytes.decode(blob)
        return True, "%s built + serialized (%d bytes), decoded" % (SEQ, len(blob))
    if not check("seq_builds", _c1):
        return _summary_and_exit()

    def _c2():
        seq = decoded["seq"]
        channels = seq["channels"]
        idxs = [i for i, c in enumerate(channels) if "TTL54" in c]
        if not idxs:
            return False, ("no FPGA1/TTL54 (TTLOrcaTrig) channel in %s -- this seq does NOT trigger "
                           "the Orca; pick an imaging seq" % SEQ)
        idx = idxs[0]
        name = channels[idx]
        nout = sum(1 for b in seq["basicseqs"] for o in b["outputs"] if o["chn"] == idx)
        if nout == 0:
            return False, "%s present but has 0 outputs (camera trigger never driven)" % name
        return True, ("Orca trigger %s present + driven (%d output events across bseqs) -> the seq "
                      "pulses TTL54 to trigger the camera" % (name, nout))
    check("camera_trigger_present", _c2)

    def _c3():
        from scan_group import ScanGroup
        from scan_export import scangroup_to_descriptor
        g = ScanGroup()
        g.runp().NumImages = NUMIMAGES
        back = g.runp().NumImages(-1)
        desc = scangroup_to_descriptor(g, SEQ, opts={"rep": 1}, label="cam")
        ni = desc.get("runp", {}).get("NumImages", "MISSING")
        if int(back) != NUMIMAGES:
            return False, "NumImages=%d did not stick (read back %r)" % (NUMIMAGES, back)
        if ni == "MISSING" or int(ni) != NUMIMAGES:
            return False, "descriptor NumImages=%r (expected %d)" % (ni, NUMIMAGES)
        return True, ("NumImages=%d sticks + lands in descriptor -> make_engine_run WILL arm the "
                      "camera and the post_cb reads %d frame(s)" % (NUMIMAGES, NUMIMAGES))
    check("numimages_propagates", _c3)

    return _summary_and_exit()


def _summary_and_exit():
    print("-" * 78)
    print("CAMERA PRE-FLIGHT: %d passed, %d failed" % (_PASS, _FAIL))
    print("ALL PASS -- seq triggers the camera + NumImages set; cleared to fire the imaging shot."
          if _FAIL == 0 else "FAILURES -- resolve before the camera run.")
    print("-" * 78)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
