"""submit_hwtest1.py -- submit the LIVE-test #1 smallest scan (get_my_seq, FPGA-only).

Builds an empty ScanGroup with runp.NumImages=0 and submits ``get_my_seq`` rep=1 to the
running pyctrl backend (ExptServer ZMQ) via the single ``submit_scan_descriptor`` verb.
This fires exactly ONE FPGA1 shot: get_my_seq adds FPGA1/DDS0/AMP=0 + a wait; it has NO NI
analog (so run_bseq's None-guard SKIPS the NI card) and NumImages=0 (so make_engine_run does
NOT arm the camera). It does NOT import libnacs (pure ScanGroup/descriptor + zmq), so there is
no engine load / DLL-detach wedge here.

DEFAULT = DRY: prints the exact descriptor and submits NOTHING. Pass --fire to submit.

    # dry (safe -- shows the payload, fires nothing):
    pyctrl\\.venv-engine-py312\\Scripts\\python.exe pyctrl\\tools\\submit_hwtest1.py
    # fire (submits one get_my_seq shot to the backend):
    pyctrl\\.venv-engine-py312\\Scripts\\python.exe pyctrl\\tools\\submit_hwtest1.py --fire
"""

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYCTRL = os.path.dirname(_THIS)
for _p in (_PYCTRL,
           os.path.join(_PYCTRL, "lib"),
           os.path.join(_PYCTRL, "tools"),
           os.path.join(_PYCTRL, "YbSteps"),
           os.path.join(_PYCTRL, "YbSeqs"),
           os.path.join(_PYCTRL, "YbExptCtrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true",
                    help="actually submit (default: dry-run, prints the descriptor only)")
    ap.add_argument("--url", default="tcp://127.0.0.1:1408", help="ExptServer ZMQ URL")
    ap.add_argument("--label", default="HWTEST1")
    args = ap.parse_args()

    from scan_group import ScanGroup
    from scan_export import scangroup_to_descriptor

    g = ScanGroup()
    g.runp().NumImages = 0                       # FPGA-only: camera NOT armed
    desc = scangroup_to_descriptor(g, "get_my_seq", opts={"rep": 1}, label=args.label)

    # safety asserts (mirror the pre-flight) -- bail loudly before firing
    assert int(g.runp().NumImages(99)) == 0, "NumImages=0 did not stick on the ScanGroup"
    assert int(desc.get("runp", {}).get("NumImages", -1)) == 0, "descriptor NumImages != 0"
    assert desc.get("seq") == "get_my_seq", "descriptor seq != get_my_seq"

    print("descriptor to submit:")
    print(json.dumps(desc, indent=2))
    print("\nEffect: ONE FPGA1 shot (get_my_seq) -- NO NI analog, NO camera arm.")

    if not args.fire:
        print("\n*** DRY RUN -- nothing submitted. Re-run with --fire to submit to %s ***" % args.url)
        return

    from yb_start_scan import ybStartScan
    did = ybStartScan("get_my_seq", g, url=args.url, label=args.label, rep=1)
    print("\n*** SUBMITTED to %s -- descriptor id = %s ***" % (args.url, did))


if __name__ == "__main__":
    main()
