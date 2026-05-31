"""Byte-order and constant-reuse guards for the SeqContext serializer.

The full per-sequence serialize() byte-equality check against the reference
.bin files lands in Phase 2 (it needs the ExpSeq/RootSeq tree). Here we pin the
two failure modes the plan calls out for Phase 1: wrong byte order (the single
most common serialization bug) and incorrect constant interning/reuse.
"""

import struct

import numpy as np
import pytest

from seq_val import SeqVal
from seq_context import SeqContext

pytestmark = pytest.mark.no_hardware

_COUNT1 = struct.pack('<i', 1)
_PREFIX_F64 = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_FLOAT64])
_PREFIX_I32 = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_INT32])
_PREFIX_B = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_BOOL])


class TestByteOrder:
    def test_float64_little_endian(self):
        ctx = SeqContext()
        ctx.get_val_id(1.0)
        # 1.0 as IEEE-754 float64, little-endian (matches MATLAB typecast on x86).
        assert ctx.node_serialized() == _COUNT1 + _PREFIX_F64 + bytes.fromhex('000000000000f03f')

    def test_float64_value(self):
        ctx = SeqContext()
        ctx.get_val_id(1.3)
        assert ctx.node_serialized() == _COUNT1 + _PREFIX_F64 + struct.pack('<d', 1.3)

    def test_int32_little_endian(self):
        ctx = SeqContext()
        ctx.get_val_id(np.int32(1))
        assert ctx.node_serialized() == _COUNT1 + _PREFIX_I32 + bytes.fromhex('01000000')

    def test_bool_byte(self):
        ctx = SeqContext()
        ctx.get_val_id(True)
        assert ctx.node_serialized() == _COUNT1 + _PREFIX_B + bytes([1])

    def test_data_table_little_endian(self):
        ctx = SeqContext()
        assert ctx.get_data_id([1.0, 2.0]) == 0
        # count(=1) + ndouble(=2) + 2 little-endian doubles
        assert ctx.data_serialized() == (struct.pack('<i', 1) + struct.pack('<i', 2)
                                         + struct.pack('<d', 1.0) + struct.pack('<d', 2.0))


class TestConstReuse:
    def test_same_float_one_id(self):
        ctx = SeqContext()
        a = ctx.get_val_id(1.3)
        b = ctx.get_val_id(1.3)
        assert a == b == 1
        assert len(ctx.node_serialized()) == len(_COUNT1) + len(_PREFIX_F64) + 8

    def test_distinct_floats_distinct_ids(self):
        ctx = SeqContext()
        assert ctx.get_val_id(1.3) == 1
        assert ctx.get_val_id(2.6) == 2
        assert ctx.get_val_id(1.3) == 1

    def test_int_types_share_int32_table(self):
        # int8/int16/... all intern as the same int32 constant (MATLAB int32(val)).
        ctx = SeqContext()
        assert ctx.get_val_id(np.int8(23)) == 1
        assert ctx.get_val_id(np.int16(23)) == 1
        assert ctx.get_val_id(np.int32(23)) == 1
        assert ctx.get_val_id(np.int32(24)) == 2

    def test_each_kind_distinct_node(self):
        # A float, a bool and an int32 with the "same" numeric value each get a
        # separate node with its own tag.
        ctx = SeqContext()
        assert ctx.get_val_id(1.0) == 1
        assert ctx.get_val_id(True) == 2
        assert ctx.get_val_id(np.int32(1)) == 3
        nodes = ctx.node_serialized()
        assert nodes == (struct.pack('<i', 3)
                         + _PREFIX_F64 + struct.pack('<d', 1.0)
                         + _PREFIX_B + bytes([1])
                         + _PREFIX_I32 + struct.pack('<i', 1))

    def test_data_reuse(self):
        ctx = SeqContext()
        assert ctx.get_data_id([1, 2, 7, 3, 4, 5]) == 0
        assert ctx.get_data_id([1, 2, 3, 4, 5]) == 1
        assert ctx.get_data_id([1, 2, 7, 3, 4, 5]) == 0   # reused
        assert len(ctx.datas) == 2
