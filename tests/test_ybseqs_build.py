"""W4/W5 capstone -- BUILD real YbSeqs in pyctrl and byte-compare to the MATLAB capture.

This is the first point real production sequences are *built* in Python (distinct from
test_ybseqs_roundtrip.py, which only decode->re-encodes the same capture). Each sequence
wires the hand-ported step cone (pyctrl/YbSteps/) over the real expConfig, serializes, and
must equal its entry in tests/reference_ybseqs/ybseqs_reference.json BYTE FOR BYTE -- the
headline proof that framework (Phase 2) + config (Phase 3) + the step cone reproduce a
real ~50-channel sequence end to end. A byte match also confirms in-body purity (the build
touched no hardware) and the int->float64 pulse-value mapping.

NO-HARDWARE: real config (SeqConfig.load_real) + tick_per_sec = 1e12 (us-scale steps round
to 0 ticks at the default 1000 and the build would raise). Engine never loaded.

Covers 15 of the 16 ok-corpus sequences -- the full linear family plus both Rearrange
branch seqs (2- and 3-basic-seq). EOM616Ramp is intentionally excluded: it reads a live
MemoryMap value at BUILD time, so byte-equality needs a MemoryMap stub seeded with the
captured value (a W5 tail, not step-cone work).
"""

import json
import os

import pytest

import compare_bytes
import seq_manager
from conftest import _TESTS_DIR
from exp_seq import ExpSeq
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_REF = os.path.join(_TESTS_DIR, "reference_ybseqs", "ybseqs_reference.json")

# (name, nargin). The MATLAB name is the module, the file, AND the function name (one
# function per file). nargin 0 -> the seq builds its own ExpSeq; nargin 1 -> it takes a
# configured ExpSeq (mirrors capture_ybseqs_reference.m's nargin dispatch).
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


def _ref_bytes():
    if not os.path.exists(_REF):
        return {}
    with open(_REF) as f:
        return {e["name"]: bytes.fromhex(e["bytes"])
                for e in json.load(f) if e.get("status") == "ok"}


_REF_BYTES = _ref_bytes()
_needs_ref = pytest.mark.skipif(
    not _REF_BYTES, reason="no committed YbSeqs capture (run tools/capture_ybseqs_reference.m)")


@pytest.fixture
def real_config():
    """Real expConfig + production tick rate; reset both in teardown (process singletons)."""
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    yield
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


def _build(name, nargin):
    mod = __import__(name)                   # file == module == function name (MATLAB name)
    fn = getattr(mod, name)
    return fn() if nargin == 0 else fn(ExpSeq())


@_needs_ref
@pytest.mark.parametrize("name,nargin", _SEQS, ids=[s[0] for s in _SEQS])
def test_seq_builds_byte_identical(real_config, name, nargin):
    assert name in _REF_BYTES, "%s missing from the committed capture" % name
    want = _REF_BYTES[name]
    got = _build(name, nargin).serialize()
    if got != want:
        d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
        raise AssertionError(
            "%s: %d bytes vs reference %d; first diff at %s" % (name, len(got), len(want), d))
    # Repeatable: a fresh build of the same seq serializes identically.
    assert _build(name, nargin).serialize() == got, "%s: build not repeatable" % name
