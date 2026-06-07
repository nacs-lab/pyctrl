"""check_ab_byte_equality.py -- per-point serialize() byte-equality oracle for the migrated scans.

For each scan it builds the SAME ScanGroup the live submission uses (imported from
``YbScans/*.py`` ``build()``), expands it point by point exactly as RunScans/runSeq2 does per shot --

    params = g.getseq(n);  s = ExpSeq(params);  seqfn(s);  bytes = s.serialize()

-- and asserts each point's bytes are BYTE-IDENTICAL to the stored reference
(``tests/reference_scan_point/ab_reference.json``). Engine-free (override_tick_per_sec; no libnacs)
and serialize() never runs the deferred camera/AWG/server/MemoryMap callbacks.

The reference is the FROZEN pyctrl golden master -- pyctrl is now the source of truth (MATLAB has
been retired as the runtime). Two modes:

  * default (compare): a regression guard -- catches when a ``lib/`` refactor or a scan edit changes
    bytes you did NOT mean to change. If every point matches, the live result is unchanged.
  * ``--capture``: re-bless the reference FROM the current pyctrl ``serialize()`` output. Run this
    when you INTENTIONALLY change the bytes (a new scan, or a deliberate param/config change) and
    have confirmed the new bytes are correct.

Migrating a NEW scan from MATLAB? Validate the port ONCE against MATLAB before blessing: regenerate
that scan's reference via the MATLAB capture (``tools/scan_point_list_ab.m`` ->
``capture_scan_point_reference.m``), confirm compare-mode PASS, then ``--capture`` freezes it in.

Run (any python with pyctrl importable; engine-free -- base/anaconda is fine):
    python pyctrl/tools/check_ab_byte_equality.py            # compare (regression guard)
    python pyctrl/tools/check_ab_byte_equality.py --capture  # re-bless from pyctrl
"""

import argparse
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
    import CoolingScan
    import CoolingScan_RNR
    import ImagingLifetimeScan
    import ReleaseRecaptureScan
    import Spectrum399Scan
    import Spectrum556Scan
    return {
        "bluelac":         (BlueLACScan.build,         "BlueTweezerLoadingSeq"),
        "spectrum556":     (Spectrum556Scan.build,     "PushoutSurvivalSeq"),
        "spectrum399":     (Spectrum399Scan.build,     "PushoutSurvival399Seq"),
        "imaginglifetime": (ImagingLifetimeScan.build, "ImagingPushoutSurvivalSeq"),
        "coolingx2d":      (CoolingScan.build,         "ImagingPushoutSurvivalSeq"),
        "coolingrnr":      (CoolingScan_RNR.build,     "ReleaseRecaptureSeq"),
        "releasetime":     (ReleaseRecaptureScan.build, "ReleaseRecaptureSeq"),
    }


def _setup_config():
    """Real expConfig + production tick rate; the engine is never loaded."""
    import seq_manager
    from seq_config import SeqConfig
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(int(1e12))


def _expand(build, seqname):
    """Build the ScanGroup and serialize every sweep point, exactly as a live shot does.

    Returns ``(nseq, [bytes_per_point])``.
    """
    from exp_seq import ExpSeq
    g = build()
    ns = g.nseq()
    seqfn = getattr(__import__(seqname), seqname)
    pts = [seqfn(ExpSeq(g.getseq(n))).serialize() for n in range(1, ns + 1)]
    return ns, pts


def capture():
    """Regenerate ab_reference.json from the CURRENT pyctrl serialize() output (re-bless)."""
    _setup_config()
    out = {}
    print("=" * 74)
    print("CAPTURE -- re-blessing ab_reference.json from pyctrl serialize()")
    print("=" * 74)
    for name, (build, seqname) in _pairs().items():
        ns, pts = _expand(build, seqname)
        out[name] = {"seq": seqname, "nseq": ns, "points": [b.hex() for b in pts]}
        print("[capture] %-16s %-28s %d points" % (name, seqname, ns))
    with open(_REF, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("-" * 74)
    print("wrote %s (%d scans)" % (_REF, len(out)))
    print("  re-run WITHOUT --capture to confirm the new baseline compares clean.")


def main():
    import compare_bytes

    if not os.path.exists(_REF):
        print("MISSING reference json: %s" % _REF)
        print("  Bless it from pyctrl with:   python %s --capture" % os.path.basename(__file__))
        sys.exit(2)
    ref_all = json.load(open(_REF))

    _setup_config()

    print("=" * 74)
    print("LAYER A -- per-point byte-equality vs the frozen pyctrl reference (engine-free)")
    print("=" * 74)

    pairs = _pairs()
    fails = 0
    for name, (build, seqname) in pairs.items():
        if name not in ref_all:
            print("[FAIL] %-12s not in reference json (rerun --capture)" % name)
            fails += 1
            continue
        ref = ref_all[name]
        if ref.get("seq") != seqname:
            print("[FAIL] %-12s seq mismatch: ref=%r py=%r" % (name, ref.get("seq"), seqname))
            fails += 1
            continue
        ns, pts = _expand(build, seqname)
        if ns != ref["nseq"]:
            print("[FAIL] %-12s nseq mismatch: py=%d ref=%d" % (name, ns, ref["nseq"]))
            fails += 1
            continue
        want_hex = ref["points"]
        seen = set()
        bad = 0
        first = None
        for n, got in enumerate(pts, 1):
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
            print("[PASS] %-12s %s: %d/%d points byte-identical to reference (%d distinct blobs)"
                  % (name, seqname, ns, ns, len(seen)))

    print("-" * 74)
    if fails == 0:
        print("LAYER A: ALL PASS -- pyctrl serialize() matches the frozen reference per point.")
    else:
        print("LAYER A: %d scan(s) FAILED -- inspect the first-diff field above." % fails)
        print("  If the change was INTENTIONAL, re-bless with: python %s --capture"
              % os.path.basename(__file__))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true",
                    help="regenerate ab_reference.json from the current pyctrl serialize() output "
                         "(pyctrl becomes the golden reference) instead of comparing")
    args = ap.parse_args()
    if args.capture:
        capture()
    else:
        main()
