"""check_ab_byte_equality.py -- Layer A oracle for the BlueLAC + Spectrum556 A/B comparison.

THE rigorous, hardware-free proof. For each scan it builds the SAME ScanGroup the live
submission uses (imported from ``YbScans/{BlueLACScan,Spectrum556Scan}.py`` ``build()``), expands
it point by point exactly as RunScans/runSeq2 does per shot --

    params = g.getseq(n);  s = ExpSeq(params);  seqfn(s);  bytes = s.serialize()

-- and asserts each point's bytes are BYTE-IDENTICAL to the MATLAB capture
(``tools/scan_point_list_ab.m`` via ``capture_scan_point_reference.m`` ->
``tests/reference_scan_point/ab_reference.json``). Engine-free (override_tick_per_sec; no libnacs)
and serialize() never runs the deferred camera/AWG/server/MemoryMap callbacks.

If every point matches, the live result MUST match MATLAB within shot noise (THE ONE RULE) --
this is strictly stronger than any single live run.

To (re)generate the MATLAB reference first (engine-free, ~maintenance window):
    matlab -batch "cd <pyctrl/tools>; capture_scan_point_reference( ...
        fullfile(pwd,'..','tests','reference_scan_point','ab_reference.json'), @scan_point_list_ab)"

Then run this (any python with pyctrl importable; engine-free -- base/anaconda is fine):
    python pyctrl/tools/check_ab_byte_equality.py
"""

import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_THIS)
for _p in (_PYCTRL, *[os.path.join(_PYCTRL, d) for d in
                      ("lib", "YbSteps", "YbSeqs", "YbScans", "YbExptCtrl", "tools")]):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_REF = os.path.join(_PYCTRL, "tests", "reference_scan_point", "ab_reference.json")


def _pairs():
    import BlueLACScan
    import Spectrum556Scan
    return {
        "bluelac":     (BlueLACScan.build,     "BlueTweezerLoadingSeq"),
        "spectrum556": (Spectrum556Scan.build, "PushoutSurvivalSeq"),
    }


def main():
    import compare_bytes
    import seq_manager
    from exp_seq import ExpSeq
    from seq_config import SeqConfig

    if not os.path.exists(_REF):
        print("MISSING reference json: %s" % _REF)
        print("  Generate it (engine-free) with:")
        print("    matlab -batch \"cd '%s'; capture_scan_point_reference("
              "fullfile(pwd,'..','tests','reference_scan_point','ab_reference.json'), "
              "@scan_point_list_ab)\"" % _THIS)
        sys.exit(2)
    ref_all = json.load(open(_REF))

    # Real expConfig + production tick rate, engine never loaded.
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(int(1e12))

    print("=" * 74)
    print("LAYER A -- per-point byte-equality vs MATLAB (engine-free)")
    print("=" * 74)

    pairs = _pairs()
    fails = 0
    for name, (build, seqname) in pairs.items():
        if name not in ref_all:
            print("[FAIL] %-12s not in reference json (rerun the MATLAB capture)" % name)
            fails += 1
            continue
        ref = ref_all[name]
        if ref.get("seq") != seqname:
            print("[FAIL] %-12s seq mismatch: ref=%r py=%r" % (name, ref.get("seq"), seqname))
            fails += 1
            continue
        g = build()
        ns = g.nseq()
        if ns != ref["nseq"]:
            print("[FAIL] %-12s nseq mismatch: py=%d ref=%d" % (name, ns, ref["nseq"]))
            fails += 1
            continue
        seqfn = getattr(__import__(seqname), seqname)
        want_hex = ref["points"]
        seen = set()
        bad = 0
        first = None
        for n in range(1, ns + 1):
            got = seqfn(ExpSeq(g.getseq(n))).serialize()
            want = bytes.fromhex(want_hex[n - 1])
            if got != want:
                bad += 1
                if first is None:
                    d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
                    first = (n, len(got), len(want), d)
            seen.add(got)
        if bad:
            n0, lg, lw, d = first
            print("[FAIL] %-12s %s: %d/%d points differ; first @point %d (%dB vs %dB) diff=%s"
                  % (name, seqname, bad, ns, n0, lg, lw, d))
            fails += 1
        elif ns > 1 and len(seen) <= 1:
            print("[FAIL] %-12s %s: all %d points identical to each other (sweep not driving bytes!)"
                  % (name, seqname, ns))
            fails += 1
        else:
            print("[PASS] %-12s %s: %d/%d points byte-identical to MATLAB (%d distinct blobs)"
                  % (name, seqname, ns, ns, len(seen)))

    print("-" * 74)
    if fails == 0:
        print("LAYER A: ALL PASS -- pyctrl serialize() == MATLAB per point for both scans.")
        print("  => the live result MUST match MATLAB within shot noise (THE ONE RULE).")
    else:
        print("LAYER A: %d scan(s) FAILED -- inspect the first-diff field above." % fails)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
