"""Reader faithfulness against REAL experiment sequences (frozen, NO-HARDWARE).

The always-safe companion to test_ybseqs_roundtrip_live.py. Instead of spawning
MATLAB, it round-trips a committed capture of real matlab_new/YbSeqs sequences
(tests/reference_ybseqs/ybseqs_reference.json, produced by
tools/capture_ybseqs_reference.m). Every serialized sequence must decode -> encode
byte-for-byte and decode to a well-formed structure -- proving the byte-format
reader handles production-shaped sequences (~50 channels, 90-130 nodes, branches,
measures), not just the small synthetic references.

To refresh after changing sequences/config, re-capture in a separate session:
    matlab -batch "addpath pyctrl/tools; capture_ybseqs_reference"
"""

import json
import os

import pytest

import compare_bytes
from conftest import _TESTS_DIR

pytestmark = pytest.mark.no_hardware

_REF = os.path.join(_TESTS_DIR, "reference_ybseqs", "ybseqs_reference.json")
_MIN_OK = 8


def _entries():
    if not os.path.exists(_REF):
        return []
    with open(_REF) as f:
        return json.load(f)


_ENTRIES = _entries()
_BUILT = [e for e in _ENTRIES if e.get("status") == "ok"]


@pytest.mark.skipif(not _ENTRIES, reason="no committed YbSeqs capture "
                                         "(run tools/capture_ybseqs_reference.m)")
def test_capture_has_enough_built():
    assert len(_BUILT) >= _MIN_OK, (
        "committed capture only has %d serialized sequences (floor %d)"
        % (len(_BUILT), _MIN_OK))


@pytest.mark.skipif(not _BUILT, reason="no serialized YbSeqs in the committed capture")
@pytest.mark.parametrize("entry", _BUILT, ids=[e["name"] for e in _BUILT])
def test_real_seq_round_trips(entry):
    raw = bytes.fromhex(entry["bytes"])
    seq = compare_bytes.decode(raw)                    # raises on malformed / trailing
    assert compare_bytes.encode(seq) == raw, "re-encode differs for %s" % entry["name"]
    # well-formed: version 0, at least one basic sequence, every output's channel id
    # in range (channel ids are 1-based into the channel-name table).
    assert seq["version"] == 0
    assert len(seq["basicseqs"]) >= 1
    nchns = len(seq["channels"])
    for b in seq["basicseqs"]:
        for out in b["outputs"]:
            assert 1 <= out["chn"] <= nchns
