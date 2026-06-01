"""The byte-format reader must REJECT malformed input (NO-HARDWARE).

compare_bytes.decode is the byte-gate's validator: the round-trip and (future)
serialize() tests trust it to fault on anything that is not exactly a well-formed
serialized sequence. test_reader_roundtrip.py only proves it accepts *valid*
references; its docstring claims decode "raises on malformed / trailing bytes" but
never tested it. This file pins the rejection paths -- including the leading
serialization-version byte, which the engine rejects ("Unknown sequence
serialization version") and the reader now does too.
"""

import os

import pytest

import compare_bytes
from conftest import all_reference_files

pytestmark = pytest.mark.no_hardware

_REFS = all_reference_files()
_HAVE = bool(_REFS)


def _valid():
    return bytearray(compare_bytes.load(_REFS[0]))


@pytest.mark.skipif(not _HAVE, reason="no reference files found")
def test_valid_reference_decodes():
    # Baseline: the unmutated reference must decode cleanly (so the negatives below
    # are isolating the mutation, not a broken fixture).
    assert compare_bytes.decode(_valid())["version"] == 0


@pytest.mark.skipif(not _HAVE, reason="no reference files found")
def test_trailing_bytes_rejected():
    with pytest.raises(ValueError):
        compare_bytes.decode(_valid() + b"\x00")


@pytest.mark.skipif(not _HAVE, reason="no reference files found")
def test_truncated_rejected():
    with pytest.raises(ValueError):
        compare_bytes.decode(_valid()[:-1])


@pytest.mark.skipif(not _HAVE, reason="no reference files found")
def test_unknown_version_rejected():
    # Previously a silent gap: a bogus version byte round-tripped as "well-formed".
    data = _valid()
    data[0] = 9
    with pytest.raises(ValueError, match="version"):
        compare_bytes.decode(data)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        compare_bytes.decode(bytearray())


def test_bad_argtype_in_node_rejected():
    # A 1-node sequence whose single node carries an out-of-range arg-type tag.
    # version(0) + nnodes(1) + [opcode OP_ABS=15][argtype 99]  then empty tails.
    import struct
    blob = bytearray()
    blob += bytes([0])                      # version
    blob += struct.pack('<I', 1)            # nnodes = 1
    blob += bytes([15, 99])                 # OP_ABS, then bogus argtype 99
    with pytest.raises(ValueError):
        compare_bytes.decode(blob)
