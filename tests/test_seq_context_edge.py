"""SeqContext constant-interning + surface edge cases (NO-HARDWARE).

Every expectation here was checked against live MATLAB SeqContext output
(tools/probe_interning.m) so it pins THE ONE RULE behaviour, not just whatever
pyctrl happens to do today. The hand-written test_seq_context.py covers the
"happy path" tables; this file covers the boundaries that hand-written cases and
the fuzzer corpus don't pin explicitly: signed-zero / NaN / inf interning, the
int32 range boundary, cross-int-type sharing, the int32 global slot byte, data
int-vs-float dedup, and the type-dispatch error paths.
"""

import struct

import numpy as np
import pytest

from seq_val import SeqVal
from seq_context import SeqContext

pytestmark = pytest.mark.no_hardware

_COUNT = lambda n: struct.pack('<i', n)
_F64 = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_FLOAT64])
_I32 = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_INT32])
_B = bytes([SeqVal.OP_IDENTITY, SeqVal.ARG_CONST_BOOL])


class TestFloatInterning:
    def test_signed_zero_distinct(self):
        # MATLAB ground truth: id(0.0)=1, id(-0.0)=2 -> DISTINCT nodes (the f64 id
        # table is a Java Hashtable keyed by the boxed Double, and Double.equals
        # separates +0.0 from -0.0). pyctrl keys on the packed bytes, same result.
        ctx = SeqContext()
        a = ctx.get_val_id(0.0)
        b = ctx.get_val_id(-0.0)
        assert (a, b) == (1, 2)
        assert ctx.node_serialized() == (
            _COUNT(2)
            + _F64 + bytes.fromhex('0000000000000000')
            + _F64 + bytes.fromhex('0000000000000080'))

    def test_nan_reused(self):
        # MATLAB ground truth: NaN reuses one node (Double.equals treats all NaN
        # as equal). pyctrl: packed bytes of NaN are identical, so same key.
        ctx = SeqContext()
        a = ctx.get_val_id(float('nan'))
        b = ctx.get_val_id(float('nan'))
        assert a == b == 1
        assert ctx.node_serialized() == _COUNT(1) + _F64 + struct.pack('<d', float('nan'))

    def test_infinities_distinct_and_exact(self):
        # MATLAB ground truth bytes: inf=...f07f, -inf=...f0ff.
        ctx = SeqContext()
        a = ctx.get_val_id(float('inf'))
        b = ctx.get_val_id(float('-inf'))
        assert (a, b) == (1, 2)
        assert ctx.node_serialized() == (
            _COUNT(2)
            + _F64 + bytes.fromhex('000000000000f07f')
            + _F64 + bytes.fromhex('000000000000f0ff'))


class TestInt32Interning:
    def test_cross_int_types_share_one_node(self):
        # MATLAB ground truth: int8(23), int16(23), int64(23) all intern to the
        # same int32 const node (getValID does int32(val) before keying).
        ctx = SeqContext()
        a = ctx.get_val_id(np.int8(23))
        b = ctx.get_val_id(np.int16(23))
        c = ctx.get_val_id(np.int64(23))
        assert a == b == c == 1
        assert ctx.node_serialized() == _COUNT(1) + _I32 + struct.pack('<i', 23)

    def test_int32_boundaries_roundtrip(self):
        # intmin/intmax must serialize to the exact little-endian int32 bytes.
        for val, hexbytes in ((2147483647, 'ffffff7f'), (-2147483648, '00000080')):
            ctx = SeqContext()
            ctx.get_val_id(np.int32(val))
            assert ctx.node_serialized() == _COUNT(1) + _I32 + bytes.fromhex(hexbytes)

    def test_out_of_range_int32_is_guarded_by_numpy(self):
        # MATLAB int32(2^40) SATURATES to intmax silently; pyctrl can never reach
        # that divergence because numpy refuses to build an out-of-range int32 at
        # construction. This pins that the divergence is unreachable, not latent.
        with pytest.raises((OverflowError, ValueError)):
            np.int32(2 ** 40)


class TestBoolInterning:
    def test_true_and_false_intern(self):
        # NB MATLAB getValID(false) is BROKEN (const_b_ids(int8(false)) indexes
        # element 0 and throws); only getValID(true) works on the MATLAB side.
        # pyctrl keys on 0/1 in a dict, so BOTH work -- pyctrl is strictly more
        # correct here. We pin that pyctrl serializes each as its own bool node.
        ctx = SeqContext()
        assert ctx.get_val_id(True) == 1
        assert ctx.get_val_id(False) == 2
        assert ctx.get_val_id(True) == 1     # reused
        assert ctx.node_serialized() == _COUNT(2) + _B + bytes([1]) + _B + bytes([0])

    def test_each_numeric_kind_distinct_node(self):
        # float / bool / int32 of the "same" value 1 each get their own tagged node.
        ctx = SeqContext()
        assert ctx.get_val_id(1.0) == 1
        assert ctx.get_val_id(True) == 2
        assert ctx.get_val_id(np.int32(1)) == 3
        assert ctx.node_serialized() == (
            _COUNT(3)
            + _F64 + struct.pack('<d', 1.0)
            + _B + bytes([1])
            + _I32 + struct.pack('<i', 1))


class TestSurface:
    def test_global_int32_slot_byte(self):
        # The int32 global slot type byte (==2) is never exercised by the happy-path
        # tables; pin it. MATLAB ground truth: globalSerialized == 01000000 02.
        ctx = SeqContext()
        ctx.new_global(SeqVal.TYPE_INT32)
        assert ctx.global_serialized() == struct.pack('<I', 1) + bytes([SeqVal.TYPE_INT32])

    def test_all_global_slot_types(self):
        ctx = SeqContext()
        ctx.new_global(SeqVal.TYPE_BOOL)
        ctx.new_global(SeqVal.TYPE_INT32)
        ctx.new_global(SeqVal.TYPE_FLOAT64)
        assert ctx.global_serialized() == struct.pack('<I', 3) + bytes(
            [SeqVal.TYPE_BOOL, SeqVal.TYPE_INT32, SeqVal.TYPE_FLOAT64])

    def test_new_global_rejects_bad_type(self):
        with pytest.raises((AssertionError, Exception)):
            SeqContext().new_global(99)

    def test_data_dedup_is_value_based_int_vs_float(self):
        # MATLAB keys the data array on num2hex(double(data)); pyctrl coerces to
        # float first. So an int list and the equal float list share one data id.
        ctx = SeqContext()
        assert ctx.get_data_id([1, 2, 3]) == 0
        assert ctx.get_data_id([1.0, 2.0, 3.0]) == 0     # reused, not a 2nd entry
        assert len(ctx.datas) == 1

    def test_unknown_value_type_raises(self):
        with pytest.raises(TypeError):
            SeqContext().get_val_id("not a value")
        with pytest.raises(TypeError):
            SeqContext().serialize_arg(bytearray(), "not a value")
