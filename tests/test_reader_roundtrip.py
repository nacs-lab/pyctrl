"""Phase 0: the byte-format reader is faithful to real MATLAB output.

For every reference, decode(bytes) -> structure -> encode(structure) must
reproduce the original bytes exactly, and decode must consume every byte.
This proves the format spec and the reader against actual serialize() output
without running MATLAB or touching hardware.
"""

import os

import pytest

import compare_bytes
from conftest import all_reference_files

pytestmark = pytest.mark.no_hardware

_REFS = all_reference_files()


@pytest.mark.skipif(not _REFS, reason="no reference files found")
@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_roundtrip_exact(path):
    data = compare_bytes.load(path)
    seq = compare_bytes.decode(data)          # raises on malformed / trailing bytes
    again = compare_bytes.encode(seq)
    assert again == data, "re-encode differs for %s" % os.path.basename(path)


@pytest.mark.skipif(not _REFS, reason="no reference files found")
@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_decode_is_well_formed(path):
    seq = compare_bytes.decode(compare_bytes.load(path))
    assert seq["version"] == 0
    assert len(seq["basicseqs"]) >= 1
    # every output references a channel id within range
    for b in seq["basicseqs"]:
        for out in b["outputs"]:
            assert out["chn"] <= len(seq["channels"])


def test_diff_finds_a_planted_change():
    """A single flipped field is reported with its path, not a raw byte offset."""
    if not _REFS:
        pytest.skip("no reference files found")
    seq = compare_bytes.decode(compare_bytes.load(_REFS[0]))
    other = compare_bytes.decode(compare_bytes.load(_REFS[0]))
    # find any node to perturb; if none, perturb the version
    if other["nodes"]:
        other["nodes"][0]["op"] = (other["nodes"][0]["op"] % 50) + 1
        expected_prefix = "seq.nodes[0].op"
    else:
        other["version"] = 1
        expected_prefix = "seq.version"
    d = compare_bytes.diff(seq, other)
    assert d is not None and d.startswith(expected_prefix)
