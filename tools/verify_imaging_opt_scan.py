"""verify_imaging_opt_scan.py -- offline check of YbScans/SLMRearrangeImagingOptScan.py.

Touches NO hardware, NO backend, NO SLM server, NO git: it only builds the ScanGroup, exports the
descriptor, and BUILDS + serializes every scan point through the production seq
(RearrangeCommSeq2) over the real expConfig. Run it before submitting the scan:

    cd pyctrl
    python tools/verify_imaging_opt_scan.py
    YB_IMGOPT_MODE=fin_ratio python tools/verify_imaging_opt_scan.py

What it asserts:
  * build() runs and yields the expected point count = len(MID pairs) x len(FIN pairs)
  * the descriptor JSON exports (scan_export.scangroup_to_descriptor) and carries 3 image
    patterns + the scan-long SLM lock
  * every point's c_ovr carries the intended (MidImgAmp1/2, FinImgAmp1/2) -- and the enumeration
    order matches the COLUMN-MAJOR convention (axis 1 fastest), which is how the analysis must
    reshape the flat scan index (yb_skills/memory/gotcha-2d-scan-reshape-column-major)
  * every point serializes, and all points are byte-DISTINCT
  * the baseline cell (all amps 1.0) is byte-identical to a no-override build = today's production
"""

import json
import os
import sys

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../pyctrl
for _d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps", "YbScans", ""):
    _p = os.path.join(_PY, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seq_manager                                    # noqa: E402
from exp_seq import ExpSeq                            # noqa: E402
from seq_config import SeqConfig                      # noqa: E402

import SLMRearrangeImagingOptScan as S                # noqa: E402


def _get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main():
    fails = []

    def check(cond, msg):
        print(("  OK   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    seq_name, g = S.build()
    print(S.describe())
    print()

    print("[1] scan shape")
    npts = g.nseq()
    if S.IMGOPT_MODE == "grid":
        want = len(S.MID_AMP_PAIRS) * len(S.FIN_AMP_PAIRS)
    else:
        want = len(S.RATIO_AMP1) * len(S.RATIO_AMP2)
    check(seq_name == "RearrangeCommSeq2", "seq is the production RearrangeCommSeq2")
    check(npts == want, "points = %d (expected %d)" % (npts, want))
    check(g.scandim(1) >= 2, "at least 2 scan dimensions (%d)" % g.scandim(1))

    print("[2] descriptor export")
    from scan_export import scangroup_to_descriptor
    desc = scangroup_to_descriptor(g, seq_name, opts={"rep": S.SHOTS_PER_POINT},
                                   label="verify", description="offline verify")
    dj = json.dumps(desc) if not isinstance(desc, str) else desc
    d = json.loads(dj)
    check(len(dj) > 0, "descriptor JSON exports (%d bytes)" % len(dj))
    rp = d.get("runp") or {}
    pats = json.loads(rp.get("imagePatternsJson", "[]"))
    check(len(pats) == 3, "imagePatternsJson has 3 frames: %s"
          % [p.get("name") for p in pats])
    check(int(rp.get("NumImages", 0)) == 3, "NumImages = 3")
    check(int(rp.get("useScanLongSlmLock", 0)) == 1, "scan-long SLM lock requested")
    check(int(rp.get("Scramble", 0)) == 1, "Scramble = 1 (drift decorrelated from the grid)")
    check(d.get("seq") == "RearrangeCommSeq2", "descriptor seq = %r" % d.get("seq"))
    opts = dict(d.get("opts") or [])          # exported as [[k, v], ...]
    check(int(opts.get("rep", 0)) == S.SHOTS_PER_POINT,
          "descriptor rep = shots/point = %d" % S.SHOTS_PER_POINT)

    print("[3] per-point parameters (column-major: axis 1 fastest)")
    got = []
    for i in range(1, npts + 1):
        p = g.getseq(i)
        ex = _get(p, "rearrange_kwargs.extras") or {}
        got.append((ex.get("MidImgAmp1"), ex.get("MidImgAmp2"),
                    ex.get("FinImgAmp1"), ex.get("FinImgAmp2")))
    if S.IMGOPT_MODE == "grid":
        exp = [(m[0], m[1], f[0], f[1]) for f in S.FIN_AMP_PAIRS for m in S.MID_AMP_PAIRS]
        check(got == exp, "all %d points carry the intended amps in column-major order" % npts)
        if got != exp:
            for i, (a, b) in enumerate(zip(got, exp), 1):
                if a != b:
                    print("       point %d: got %s want %s" % (i, a, b))
    else:
        check(all(v is not None for t in got for v in t),
              "all four amps set at every point")
    print("       point 1  (mid1,mid2,fin1,fin2) = %s" % (got[0],))
    print("       point %-2d (mid1,mid2,fin1,fin2) = %s" % (npts, got[-1]))

    print("[4] every point builds + serializes, all distinct")
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    try:
        from RearrangeCommSeq2 import RearrangeCommSeq2
        blobs = {}
        dup = None
        for i in range(1, npts + 1):
            b = RearrangeCommSeq2(ExpSeq(g.getseq(i))).serialize()
            if b in blobs:
                dup = (i, blobs[b])
            blobs[b] = i
        check(dup is None, "all %d points serialize to DISTINCT bytes%s"
              % (npts, "" if dup is None else " (points %d and %d collide)" % dup))

        print("[5] baseline cell == production (no override)")
        base_ovr = g.getseq(1)
        no_ovr = json.loads(json.dumps(base_ovr))
        for k in ("MidImgAmp1", "MidImgAmp2", "FinImgAmp1", "FinImgAmp2"):
            no_ovr["rearrange_kwargs"]["extras"].pop(k, None)
        a = RearrangeCommSeq2(ExpSeq(base_ovr)).serialize()
        b = RearrangeCommSeq2(ExpSeq(no_ovr)).serialize()
        is_unit = got[0][:4] == (1.0, 1.0, 1.0, 1.0)
        check((a == b) if is_unit else (a != b),
              "point 1 amps %s -> bytes %s no-override build"
              % (got[0], "==" if is_unit else "!="))
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()

    print()
    if fails:
        print("FAILED (%d):" % len(fails))
        for m in fails:
            print("  - " + m)
        return 1
    print("ALL CHECKS PASSED -- %d points x %d shots = %d shots"
          % (npts, S.SHOTS_PER_POINT, npts * S.SHOTS_PER_POINT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
