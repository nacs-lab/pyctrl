"""submit_ni.py -- LIVE-test #1 (NI clock-out): TweezerLoadingSeq, ONE shot, NO camera.

Mirrors LACScan.m's fixed params but with NO sweep (one shot), rep=1, and NumImages=0 so
make_engine_run does NOT arm the camera -- isolating the NI clock-out path from camera capture.
Fires ONE real loading shot that drives the 14 NI analog channels (verified by preflight_ni.py)
through the FPGA-clocked AO path: NiDAQRunner arms PCIe-6738 Dev1 to listen on PFI0 (400 kHz
clock) + PFI1 (start trig), the FPGA start() emits both (baked into the bytecode), the FINITE
task clocks out 1048 samples/channel and completes.

The seq's Imag399 step still pulses TTLOrcaTrig (TTL54), but with no camera armed that edge is
harmless. Does NOT import libnacs (pure ScanGroup/descriptor + zmq).

DEFAULT = DRY (prints the descriptor, submits nothing). Pass --fire to submit.

    pyctrl\\.venv-engine-py312\\Scripts\\python.exe pyctrl\\tools\\submit_ni.py            # dry
    <yb_analysis python> pyctrl\\tools\\submit_ni.py --fire                           # fire
"""

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_THIS)
for _p in (_PYCTRL, os.path.join(_PYCTRL, "lib"), os.path.join(_PYCTRL, "tools"),
           os.path.join(_PYCTRL, "YbSeqs"), os.path.join(_PYCTRL, "YbExptCtrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true",
                    help="actually submit (default: dry-run, prints the descriptor only)")
    ap.add_argument("--url", default="tcp://127.0.0.1:1408")
    ap.add_argument("--label", default="HWTEST1-NI")
    args = ap.parse_args()

    from scan_group import ScanGroup
    from scan_export import scangroup_to_descriptor

    g = ScanGroup()
    # active fixed params (from LACScan.m) -- NO sweep, so nseq=1 -> one shot
    g().BlueMOT.LoadingTime = 0.4
    g().GreenMOT.CoolDown.HoldTime = 0.2
    # run params -- mirror LACScan.m but NumImages=0 (no camera arm) + no scramble (single shot)
    rp = g.runp()
    rp.NumPerGroup = 500
    rp.NumImages = 0
    rp.isInit = 0
    rp.Scramble = 0
    rp.isHC = 0
    rp.isGrid2 = 0

    desc = scangroup_to_descriptor(g, "TweezerLoadingSeq", opts={"rep": 1}, label=args.label)

    # safety asserts -- bail loudly before firing
    assert int(g.runp().NumImages(99)) == 0, "NumImages=0 did not stick on the ScanGroup"
    assert int(desc.get("runp", {}).get("NumImages", -1)) == 0, "descriptor NumImages != 0"
    assert desc.get("seq") == "TweezerLoadingSeq", "descriptor seq != TweezerLoadingSeq"

    print("descriptor to submit:")
    print(json.dumps(desc, indent=2))
    print("\nEffect: ONE TweezerLoadingSeq shot -- drives the 14 NI analog channels via the "
          "FPGA-clocked AO path; NO camera arm (NumImages=0).")

    if not args.fire:
        print("\n*** DRY RUN -- nothing submitted. Re-run with --fire to submit to %s ***" % args.url)
        return

    from yb_start_scan import ybStartScan
    did = ybStartScan("TweezerLoadingSeq", g, url=args.url, label=args.label, rep=1)
    print("\n*** SUBMITTED to %s -- descriptor id = %s ***" % (args.url, did))


if __name__ == "__main__":
    main()
