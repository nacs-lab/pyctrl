"""submit_camera.py -- LIVE-test #1 (camera capture): TweezerLoadingSeq, ONE shot, NumImages=1.

Same one-shot TweezerLoadingSeq as submit_ni.py, but NumImages=1 so make_engine_run ARMS the
Orca (external rising-edge trigger) and the per-shot post_cb (make_capture_post_cb) reads ONE
frame, runs it through store_imgs (column-major wire format), and seq_finish()es. This exercises
the run-loop camera-capture wiring -- the last untested path. The seq's Imag399 step pulses
TTLOrcaTrig (FPGA1/TTL54) once (verified present by preflight_camera.py) to trigger the frame.

Confirm AFTER firing with: query_backend.py get_num_imgs  (>=1 == a frame was captured + published;
0 == short-read/seq_cancel -> no frame, check DCAM trigger polarity / TTL54 wiring).

DEFAULT = DRY. Pass --fire to submit.
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
    ap.add_argument("--label", default="HWTEST1-CAM")
    args = ap.parse_args()

    from scan_group import ScanGroup
    from scan_export import scangroup_to_descriptor

    g = ScanGroup()
    g().BlueMOT.LoadingTime = 0.4
    g().GreenMOT.CoolDown.HoldTime = 0.2
    rp = g.runp()
    rp.NumPerGroup = 500
    rp.NumImages = 1                 # <-- arm the camera + capture ONE frame
    rp.isInit = 0
    rp.Scramble = 0
    rp.isHC = 0
    rp.isGrid2 = 0

    desc = scangroup_to_descriptor(g, "TweezerLoadingSeq", opts={"rep": 1}, label=args.label)

    assert int(g.runp().NumImages(-1)) == 1, "NumImages=1 did not stick"
    assert int(desc.get("runp", {}).get("NumImages", -1)) == 1, "descriptor NumImages != 1"
    assert desc.get("seq") == "TweezerLoadingSeq", "descriptor seq != TweezerLoadingSeq"

    print("descriptor to submit:")
    print(json.dumps(desc, indent=2))
    print("\nEffect: ONE TweezerLoadingSeq shot -- arms the Orca (external trigger), captures 1 "
          "frame via the run-loop post_cb, publishes it (store_imgs).")

    if not args.fire:
        print("\n*** DRY RUN -- nothing submitted. Re-run with --fire to submit to %s ***" % args.url)
        return

    from yb_start_scan import ybStartScan
    did = ybStartScan("TweezerLoadingSeq", g, url=args.url, label=args.label, rep=1)
    print("\n*** SUBMITTED to %s -- descriptor id = %s ***" % (args.url, did))


if __name__ == "__main__":
    main()
