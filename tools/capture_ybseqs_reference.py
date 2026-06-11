"""Re-capture ``tests/reference_ybseqs/ybseqs_reference.json`` FROM pyctrl.

Pyctrl-side twin of the MATLAB ``capture_ybseqs_reference.m``. Since pyctrl is now the
authoritative runtime and the MATLAB byte oracle is only a reference for un-migrated scans,
the strict pyctrl<->MATLAB byte-equality (THE ONE RULE) is RELEASED for these sequences: this
tool captures pyctrl's OWN ``serialize()`` output as the regression reference (mirrors
``tools/capture_config_reference.py`` for the config oracle). ``test_ybseqs_build.py`` then
becomes a pyctrl self-consistency guard -- "did a build's bytes change unexpectedly?" -- rather
than a MATLAB-equivalence check. Engine-free (override_tick_per_sec), like the test.

Only the pyctrl-buildable ``status == "ok"`` sequences (the ``_SEQS`` list, mirroring
``tests/test_ybseqs_build.py``) are rebuilt; every other entry (skip/hardware-driver) is
preserved verbatim so the file shape and the other consumers (test_ybseqs_roundtrip) are
unaffected.

    python pyctrl/tools/capture_ybseqs_reference.py
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
from seq_config import SeqConfig         # noqa: E402

_REF = os.path.join(_ROOT, "tests", "reference_ybseqs", "ybseqs_reference.json")

# (name, nargin) -- MUST mirror tests/test_ybseqs_build.py::_SEQS (the file == module ==
# function name; nargin 0 -> the seq builds its own ExpSeq, 1 -> it takes one).
_SEQS = [
    ("CoreShellMOTSeq", 1),
    ("GreenMOTSeq", 1),
    ("DummySeq", 0),
    ("TweezerLoadingSeq", 1),
    ("BlueTweezerLoadingSeq", 1),
    ("TweezerEnhancedLoadingSeq", 1),
    ("CoolingOptimizationSeq", 1),
    ("ImagingSurvivalSeq", 1),
    ("ReleaseRecaptureSeq", 1),
    ("PushoutSurvivalSeq", 1),
    ("PushoutSurvival399Seq", 1),
    ("ImagingPushoutSurvivalSeq", 1),
    ("RearrangeCommSeq", 1),
    ("RearrangeCommSeq2", 1),
    ("get_my_seq", 1),
]


def _build(name, nargin):
    mod = __import__(name)                # file == module == function name (MATLAB name)
    fn = getattr(mod, name)
    return fn() if nargin == 0 else fn(ExpSeq())


def main():
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)   # us-scale steps need a fine tick (engine never loaded)
    try:
        with open(_REF) as f:
            entries = json.load(f)
        by_name = {e["name"]: e for e in entries}
        n_ok = 0
        for name, nargin in _SEQS:
            b = _build(name, nargin).serialize()
            e = by_name.get(name)
            if e is None:
                e = {"name": name}
                entries.append(e)
                by_name[name] = e
            e["status"] = "ok"
            e["bytes"] = b.hex()
            n_ok += 1
    finally:
        seq_manager.override_tick_per_sec(0)
        SeqConfig.reset()

    with open(_REF, "w") as f:
        json.dump(entries, f)
    print("re-captured %d 'ok' seqs from pyctrl -> %s" % (n_ok, _REF))


if __name__ == "__main__":
    main()
