"""Re-capture ``tests/reference_scan_point/scan_point_reference.json`` FROM pyctrl.

Pyctrl-side twin of the MATLAB ``capture_scan_point_reference.m``. Same rationale as
``capture_ybseqs_reference.py``: pyctrl is the authoritative runtime, so the strict
pyctrl<->MATLAB per-point byte-equality is RELEASED and this captures pyctrl's OWN per-point
``serialize()`` output as the regression reference. ``test_scan_point_oracle.py`` and
``test_dispatch_descriptor.py`` (which share this file) then become pyctrl self-consistency
guards. Engine-free (override_tick_per_sec), exactly as the tests build.

The scan builders MUST mirror ``tests/test_scan_point_oracle.py`` (same swept values) so the
captured bytes equal what the tests build.

    python pyctrl/tools/capture_scan_point_reference.py
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
for _p in (os.path.join(_ROOT, "lib"), os.path.join(_ROOT, "YbSteps"),
           os.path.join(_ROOT, "YbSeqs"), os.path.join(_ROOT, "YbExptCtrl"),
           os.path.join(_ROOT, "tools"), _ROOT):   # mirror pyproject.toml pythonpath
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seq_manager                       # noqa: E402
from exp_seq import ExpSeq               # noqa: E402
from scan_group import ScanGroup         # noqa: E402
from seq_config import SeqConfig         # noqa: E402

_REF = os.path.join(_ROOT, "tests", "reference_scan_point", "scan_point_reference.json")


# --- twin builders of tests/test_scan_point_oracle.py (same scanned values) --------------- #
def build_spectrum399():
    g = ScanGroup()
    g().Pushout.Blue.Amp1 = 0.25
    g().Pushout.Blue.Freq.scan(1, [v * 1e6 for v in range(220, 361, 35)])  # 5 points
    g().Pushout.Time = 10e-3
    g.runp().NumPerGroup = 10000
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    return g


def build_imaging_hist():
    g = ScanGroup()
    g().Imag399.ExposureTime = 100e-3
    g().SLM.VServo = 1
    g().Imag399.FreqDetuning.scan(1, [-5 * 1e6, 0 * 1e6])
    g().Imag399.Amp.scan(2, [0.2, 0.3])
    g().Pushout.Green.Amp = 0
    g().Pushout.Blue.Amp1 = 0
    g().Pushout.Time = 10e-3
    g.runp().NumImages = 2
    g.runp().Scramble = 1
    return g


_PAIRS = {
    "spectrum399": (build_spectrum399, "PushoutSurvival399Seq"),
    "imaging_hist": (build_imaging_hist, "PushoutSurvivalSeq"),
}


def main():
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    out = {}
    try:
        for name, (build, seqname) in _PAIRS.items():
            g = build()
            mod = __import__(seqname)
            seqfn = getattr(mod, seqname)
            points = [seqfn(ExpSeq(g.getseq(n))).serialize().hex()
                      for n in range(1, g.nseq() + 1)]
            out[name] = {"seq": seqname, "nseq": g.nseq(), "points": points}
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()

    with open(_REF, "w") as f:
        json.dump(out, f)
    print("re-captured %d scans (%s) from pyctrl -> %s"
          % (len(out), ", ".join("%s:%d" % (k, v["nseq"]) for k, v in out.items()), _REF))


if __name__ == "__main__":
    main()
