"""Port of matlab_new/lib/test/TestSeqContext.m.

Covers SeqVal construction, operator overloading + construction-time constant
folding, to_string, and -- the Phase 1 byte-equality gate -- the exact node /
data / global serialized tables produced by SeqContext.

SeqVal overrides ``==`` to build a comparison node (it cannot return a plain
bool), so structural comparisons here go through ``seqval_isequal`` and a
kind-check, never Python ``==``.
"""

import struct

import numpy as np
import pytest

from seq_val import (SeqVal, seqval_isequal, _arg_kind, to_string,
                     hypot, xor, max, min, abs_, ceil, floor, exp, expm1,
                     log, log1p, log2, log10, sqrt, asin, acos, atan, atan2,
                     acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch,
                     sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech,
                     csch, erf, erfc, gamma, gammaln, rem, ldivide)
from seq_context import SeqContext
from interpolate import interpolate
from ifelse import ifelse

pytestmark = pytest.mark.no_hardware


# --- comparison helpers (mirror MATLAB verifyEqual: class + value) ---------- #
def _kind_match(x, y):
    kx, ky = _arg_kind(x), _arg_kind(y)
    if kx is None and ky is None:
        return True               # raw value tables / non-scalars
    return kx == ky


def eq_(a, b):
    assert _kind_match(a, b), "kind mismatch: %r vs %r" % (a, b)
    assert seqval_isequal(a, b), "value mismatch: %r vs %r" % (a, b)


def args_(actual, expected):
    assert len(actual) == len(expected), "arg count %d vs %d" % (len(actual), len(expected))
    for x, y in zip(actual, expected):
        eq_(x, y)


# --- byte builders (mirror MATLAB typecast(...,'int8'), little-endian) ------ #
def u8(x):
    return bytes([x & 0xFF])


def i32(x):
    return struct.pack('<i', x)


def f64(x):
    return struct.pack('<d', x)


def f64arr(vals):
    return struct.pack('<%dd' % len(vals), *[float(v) for v in vals])


class TestSeqContext:
    def test_dotest(self):
        ctx = SeqContext()

        # Creating argument nodes
        arg0 = ctx.get_arg(0)
        arg1 = ctx.get_arg(1)
        arg10 = ctx.get_arg(10)
        assert arg0.head == SeqVal.H_ARG
        assert arg0.args == [0]
        assert arg1.head == SeqVal.H_ARG
        assert arg1.args == [1]
        assert arg10.head == SeqVal.H_ARG
        assert arg10.args == [10]
        assert arg0 is ctx.get_arg(0)
        assert arg1 is ctx.get_arg(1)
        assert arg10 is ctx.get_arg(10)
        assert arg0 is ctx.arg0
        assert arg1 is ctx.arg1
        assert to_string(arg0) == 'arg(0)'
        assert to_string(arg1) == 'arg(1)'

        # Creating measure nodes
        m0, m0id = ctx.new_measure()
        assert m0id == 0
        assert m0.head == SeqVal.H_MEASURE
        assert len(m0.args) == 1
        assert m0.args[0] == 0
        assert to_string(m0) == 'm(0)'
        m1, m1id = ctx.new_measure()
        assert m1id == 1
        assert m1.head == SeqVal.H_MEASURE
        assert len(m1.args) == 1
        assert m1.args[0] == 1
        assert to_string(m1) == 'm(1)'

        # Creating global nodes
        g0, g0id = ctx.new_global(SeqVal.TYPE_BOOL)
        assert g0id == 0
        assert g0.head == SeqVal.H_GLOBAL
        assert len(g0.args) == 1
        assert g0.args[0] == 0
        assert to_string(g0) == 'g(0)'
        g1, g1id = ctx.new_global(SeqVal.TYPE_FLOAT64)
        assert g1id == 1
        assert g1.head == SeqVal.H_GLOBAL
        assert len(g1.args) == 1
        assert g1.args[0] == 1
        assert to_string(g1) == 'g(1)'

        # Expressions
        e1 = arg0 + arg1
        assert e1.head == SeqVal.OP_ADD
        assert len(e1.args) == 2
        eq_(e1.args[0], arg0)
        eq_(e1.args[1], arg1)
        assert to_string(e1) == 'arg(0) + arg(1)'

        e2 = g0 * m1
        assert e2.head == SeqVal.OP_MUL
        args_(e2.args, [g0, m1])
        assert to_string(e2) == 'g(0) * m(1)'

        e3 = arg0 / m0
        assert e3.head == SeqVal.OP_DIV
        args_(e3.args, [arg0, m0])
        assert to_string(e3) == 'arg(0) / m(0)'

        e5 = interpolate(arg0, e2, e3, [1, 2, 7, 3, 4, 5])
        assert e5.head == SeqVal.OP_INTERP
        assert len(e5.args) == 4
        eq_(e5.args[0], arg0)
        eq_(e5.args[1], e2)
        e4 = e5.args[2]
        assert e4.head == SeqVal.OP_SUB
        args_(e4.args, [e3, e2])
        assert seqval_isequal(e5.args[3], [1, 2, 7, 3, 4, 5])
        assert to_string(e4) == 'arg(0) / m(0) - g(0) * m(1)'
        assert to_string(e5) == \
            'interp(arg(0), g(0) * m(1), arg(0) / m(0) - g(0) * m(1), [1,2,7,3,4,5])'

        e6 = hypot(4.5, e5)
        assert e6.head == SeqVal.OP_HYPOT
        args_(e6.args, [4.5, e5])
        assert to_string(e6) == \
            'hypot(4.5, interp(arg(0), g(0) * m(1), arg(0) / m(0) - g(0) * m(1), [1,2,7,3,4,5]))'

        e8 = interpolate(arg0, arg1, g0, [1, 2, 7, 3, 4, 5])
        assert e8.head == SeqVal.OP_INTERP
        assert len(e8.args) == 4
        eq_(e8.args[0], arg0)
        eq_(e8.args[1], arg1)
        e7 = e8.args[2]
        assert e7.head == SeqVal.OP_SUB
        args_(e7.args, [g0, arg1])
        assert seqval_isequal(e8.args[3], [1, 2, 7, 3, 4, 5])
        assert to_string(e7) == 'g(0) - arg(1)'
        assert to_string(e8) == 'interp(arg(0), arg(1), g(0) - arg(1), [1,2,7,3,4,5])'

        e10 = interpolate(arg0, arg1, g0, [1, 2, 3, 4, 5])
        assert e10.head == SeqVal.OP_INTERP
        assert len(e10.args) == 4
        eq_(e10.args[0], arg0)
        eq_(e10.args[1], arg1)
        e9 = e10.args[2]
        assert e9.head == SeqVal.OP_SUB
        args_(e9.args, [g0, arg1])
        assert seqval_isequal(e10.args[3], [1, 2, 3, 4, 5])
        assert to_string(e9) == 'g(0) - arg(1)'
        assert to_string(e10) == 'interp(arg(0), arg(1), g(0) - arg(1), [1,2,3,4,5])'

        c0 = ctx.get_val_id(1.3)
        assert c0 == 1
        assert ctx.get_val_id(1.3) == 1
        c1 = ctx.get_val_id(True)
        assert c1 == 2
        assert ctx.get_val_id(True) == 2
        c2 = ctx.get_val_id(np.int8(23))
        assert c2 == 3
        assert ctx.get_val_id(np.int16(23)) == 3

        n0 = ctx.get_val_id(e6)
        assert n0 == 8
        assert ctx.get_val_id(e2) == 4
        assert ctx.get_val_id(e3) == 5
        assert ctx.get_val_id(e4) == 6
        assert ctx.get_val_id(e5) == 7
        n1 = ctx.get_val_id(e10)
        assert n1 == 10
        n2 = ctx.get_val_id(e8)
        assert n2 == 12

        na0 = ctx.get_val_id(arg0)
        assert na0 == 13
        assert ctx.get_val_id(arg0) == na0

        nm1 = ctx.get_val_id(m1)
        assert nm1 == 14
        assert ctx.get_val_id(m1) == nm1

        ng0 = ctx.get_val_id(g0)
        assert ng0 == 15
        assert ctx.get_val_id(g0) == ng0

        expected_nodes = (
            i32(15) +
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_CONST_FLOAT64) + f64(1.3) +         # 1
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_CONST_BOOL) + u8(1) +               # 2
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_CONST_INT32) + i32(23) +            # 3
            u8(SeqVal.OP_MUL) +                                                        # 4
            u8(SeqVal.ARG_GLOBAL) + i32(0) + u8(SeqVal.ARG_MEASURE) + i32(1) +
            u8(SeqVal.OP_DIV) +                                                        # 5
            u8(SeqVal.ARG_ARG) + i32(0) + u8(SeqVal.ARG_MEASURE) + i32(0) +
            u8(SeqVal.OP_SUB) +                                                        # 6
            u8(SeqVal.ARG_NODE) + i32(5) + u8(SeqVal.ARG_NODE) + i32(4) +
            u8(SeqVal.OP_INTERP) +                                                     # 7
            u8(SeqVal.ARG_ARG) + i32(0) + u8(SeqVal.ARG_NODE) + i32(4) +
            u8(SeqVal.ARG_NODE) + i32(6) + i32(0) +
            u8(SeqVal.OP_HYPOT) +                                                      # 8
            u8(SeqVal.ARG_CONST_FLOAT64) + f64(4.5) + u8(SeqVal.ARG_NODE) + i32(7) +
            u8(SeqVal.OP_SUB) +                                                        # 9
            u8(SeqVal.ARG_GLOBAL) + i32(0) + u8(SeqVal.ARG_ARG) + i32(1) +
            u8(SeqVal.OP_INTERP) +                                                     # 10
            u8(SeqVal.ARG_ARG) + i32(0) + u8(SeqVal.ARG_ARG) + i32(1) +
            u8(SeqVal.ARG_NODE) + i32(9) + i32(1) +
            u8(SeqVal.OP_SUB) +                                                        # 11
            u8(SeqVal.ARG_GLOBAL) + i32(0) + u8(SeqVal.ARG_ARG) + i32(1) +
            u8(SeqVal.OP_INTERP) +                                                     # 12
            u8(SeqVal.ARG_ARG) + i32(0) + u8(SeqVal.ARG_ARG) + i32(1) +
            u8(SeqVal.ARG_NODE) + i32(11) + i32(0) +
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_ARG) + i32(0) +                     # 13
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_MEASURE) + i32(1) +                 # 14
            u8(SeqVal.OP_IDENTITY) + u8(SeqVal.ARG_GLOBAL) + i32(0)                    # 15
        )
        assert ctx.node_serialized() == expected_nodes

        expected_data = (i32(2) +
                         i32(6) + f64arr([1, 2, 7, 3, 4, 5]) +
                         i32(5) + f64arr([1, 2, 3, 4, 5]))
        assert ctx.data_serialized() == expected_data

        expected_global = i32(2) + u8(SeqVal.TYPE_BOOL) + u8(SeqVal.TYPE_FLOAT64)
        assert ctx.global_serialized() == expected_global

    def test_constarg(self):
        ctx = SeqContext()

        e1 = ctx.arg0 * np.int32(2)
        assert e1.head == SeqVal.OP_MUL
        args_(e1.args, [ctx.arg0, np.int32(2)])
        e2 = e1 + True
        assert e2.head == SeqVal.OP_ADD
        args_(e2.args, [e1, True])

        ie2 = ctx.get_val_id(e2)
        assert ie2 == 2
        ie1 = ctx.get_val_id(e1)
        assert ie1 == 1

        expected_nodes = (
            i32(2) +
            u8(SeqVal.OP_MUL) +                                                        # 1
            u8(SeqVal.ARG_ARG) + i32(0) + u8(SeqVal.ARG_CONST_INT32) + i32(2) +
            u8(SeqVal.OP_ADD) +                                                        # 2
            u8(SeqVal.ARG_NODE) + i32(1) + u8(SeqVal.ARG_CONST_BOOL) + u8(1)
        )
        assert ctx.node_serialized() == expected_nodes
        assert ctx.data_serialized() == i32(0)
        assert ctx.global_serialized() == i32(0)

    def test_equal(self):
        ctx = SeqContext()

        g0, g0id = ctx.new_global(SeqVal.TYPE_BOOL)
        assert g0id == 0
        g1, g1id = ctx.new_global(SeqVal.TYPE_BOOL)
        assert g1id == 1
        # == returns a SeqVal node
        assert isinstance(g0 == g1, SeqVal)
        # seqval_isequal returns a bool
        assert isinstance(seqval_isequal(g0, g1), bool)
        assert not seqval_isequal(g0, g1)

    def test_operations(self):
        ctx = SeqContext()
        g0, _ = ctx.new_global(SeqVal.TYPE_BOOL)
        g1, _ = ctx.new_global(SeqVal.TYPE_BOOL)

        def check(v, head, expect_args, s):
            assert v.head == head
            args_(v.args, expect_args)
            assert to_string(v) == s

        # plus
        check(g0 + g1, SeqVal.OP_ADD, [g0, g1], 'g(0) + g(1)')
        check(1 + g1, SeqVal.OP_ADD, [1, g1], '1 + g(1)')
        check(g0 + 1, SeqVal.OP_ADD, [g0, 1], 'g(0) + 1')

        # minus
        check(g0 - g1, SeqVal.OP_SUB, [g0, g1], 'g(0) - g(1)')
        check(3 - g1, SeqVal.OP_SUB, [3, g1], '3 - g(1)')
        check(g0 - 3, SeqVal.OP_SUB, [g0, 3], 'g(0) - 3')

        # times
        check(g0 * g1, SeqVal.OP_MUL, [g0, g1], 'g(0) * g(1)')
        check(3 * g1, SeqVal.OP_MUL, [3, g1], '3 * g(1)')
        check(g0 * 3, SeqVal.OP_MUL, [g0, 3], 'g(0) * 3')

        # uplus / uminus
        eq_(+g0, g0)
        check(-g0, SeqVal.OP_MUL, [np.int32(-1), g0], '-1 * g(0)')

        # divide
        check(g0 / g1, SeqVal.OP_DIV, [g0, g1], 'g(0) / g(1)')
        check(3 / g1, SeqVal.OP_DIV, [3, g1], '3 / g(1)')
        check(g0 / 3, SeqVal.OP_DIV, [g0, 3], 'g(0) / 3')

        # left divide (MATLAB \) -- ldivide(a, b) == b / a
        check(ldivide(g0, g1), SeqVal.OP_DIV, [g1, g0], 'g(1) / g(0)')
        check(ldivide(3, g1), SeqVal.OP_DIV, [g1, 3], 'g(1) / 3')
        check(ldivide(g0, 3), SeqVal.OP_DIV, [3, g0], '3 / g(0)')

        # comparisons. SeqVal-on-left and SeqVal-on-both match MATLAB exactly.
        # A constant on the LEFT reflects: Python has no __rlt__, so ``3 < g1``
        # dispatches to ``g1.__gt__(3)`` -> GT{g1,3}, the mathematical mirror of
        # MATLAB's LT{3,g1}. That is fine -- compare_bytes.normalize() canonicalizes
        # swappable comparisons so the serialized forms verify as equal (proven in
        # tests/test_compare_canonical.py). Here we just pin the reflected result.
        check(g0 < g1, SeqVal.OP_CMP_LT, [g0, g1], 'g(0) < g(1)')
        check(g0 < 3, SeqVal.OP_CMP_LT, [g0, 3], 'g(0) < 3')
        check(3 < g1, SeqVal.OP_CMP_GT, [g1, 3], 'g(1) > 3')     # mirror of LT{3,g1}

        check(g0 > g1, SeqVal.OP_CMP_GT, [g0, g1], 'g(0) > g(1)')
        check(g0 > 3, SeqVal.OP_CMP_GT, [g0, 3], 'g(0) > 3')
        check(3 > g1, SeqVal.OP_CMP_LT, [g1, 3], 'g(1) < 3')     # mirror of GT{3,g1}

        check(g0 <= g1, SeqVal.OP_CMP_LE, [g0, g1], 'g(0) <= g(1)')
        check(g0 <= 3, SeqVal.OP_CMP_LE, [g0, 3], 'g(0) <= 3')
        check(3 <= g1, SeqVal.OP_CMP_GE, [g1, 3], 'g(1) >= 3')   # mirror of LE{3,g1}

        check(g0 >= g1, SeqVal.OP_CMP_GE, [g0, g1], 'g(0) >= g(1)')
        check(g0 >= 3, SeqVal.OP_CMP_GE, [g0, 3], 'g(0) >= 3')
        check(3 >= g1, SeqVal.OP_CMP_LE, [g1, 3], 'g(1) <= 3')   # mirror of GE{3,g1}

        check(g0 != g1, SeqVal.OP_CMP_NE, [g0, g1], 'g(0) ~= g(1)')
        check(g0 != 3, SeqVal.OP_CMP_NE, [g0, 3], 'g(0) ~= 3')
        check(3 != g1, SeqVal.OP_CMP_NE, [g1, 3], 'g(1) ~= 3')   # commutative; args mirror

        check(g0 == g1, SeqVal.OP_CMP_EQ, [g0, g1], 'g(0) == g(1)')
        check(g0 == 3, SeqVal.OP_CMP_EQ, [g0, 3], 'g(0) == 3')
        check(3 == g1, SeqVal.OP_CMP_EQ, [g1, 3], 'g(1) == 3')   # commutative; args mirror

        # logical
        check(g0 & g1, SeqVal.OP_AND, [g0, g1], 'g(0) & g(1)')
        check(g0 | g1, SeqVal.OP_OR, [g0, g1], 'g(0) | g(1)')
        check(xor(g0, g1), SeqVal.OP_XOR, [g0, g1], 'xor(g(0), g(1))')
        check(~g0, SeqVal.OP_NOT, [g0], '~g(0)')

        # unary math
        check(abs(g0), SeqVal.OP_ABS, [g0], 'abs(g(0))')
        check(ceil(g0), SeqVal.OP_CEIL, [g0], 'ceil(g(0))')
        check(exp(g0), SeqVal.OP_EXP, [g0], 'exp(g(0))')
        check(expm1(g0), SeqVal.OP_EXPM1, [g0], 'expm1(g(0))')
        check(floor(g0), SeqVal.OP_FLOOR, [g0], 'floor(g(0))')
        check(log(g0), SeqVal.OP_LOG, [g0], 'log(g(0))')
        check(log1p(g0), SeqVal.OP_LOG1P, [g0], 'log1p(g(0))')
        check(log2(g0), SeqVal.OP_LOG2, [g0], 'log2(g(0))')
        check(log10(g0), SeqVal.OP_LOG10, [g0], 'log10(g(0))')

        # pow
        check(g0 ** g1, SeqVal.OP_POW, [g0, g1], 'g(0) ^ g(1)')
        check(g0 ** 4, SeqVal.OP_POW, [g0, 4], 'g(0) ^ 4')
        check(2 ** g1, SeqVal.OP_POW, [2, g1], '2 ^ g(1)')

        check(sqrt(g0), SeqVal.OP_SQRT, [g0], 'sqrt(g(0))')
        check(asin(g0), SeqVal.OP_ASIN, [g0], 'asin(g(0))')
        check(acos(g0), SeqVal.OP_ACOS, [g0], 'acos(g(0))')
        check(atan(g0), SeqVal.OP_ATAN, [g0], 'atan(g(0))')
        check(atan2(g0, g1), SeqVal.OP_ATAN2, [g0, g1], 'atan2(g(0), g(1))')
        check(atan2(g0, 4), SeqVal.OP_ATAN2, [g0, 4], 'atan2(g(0), 4)')
        check(atan2(2, g1), SeqVal.OP_ATAN2, [2, g1], 'atan2(2, g(1))')

        # derived inverse/reciprocal trig (compositions)
        v = acot(g0)
        check(v, SeqVal.OP_ATAN, [v.args[0]], 'atan(1 / g(0))')
        check(v.args[0], SeqVal.OP_DIV, [1, g0], '1 / g(0)')
        v = asec(g0)
        check(v, SeqVal.OP_ACOS, [v.args[0]], 'acos(1 / g(0))')
        v = acsc(g0)
        check(v, SeqVal.OP_ASIN, [v.args[0]], 'asin(1 / g(0))')

        check(asinh(g0), SeqVal.OP_ASINH, [g0], 'asinh(g(0))')
        check(acosh(g0), SeqVal.OP_ACOSH, [g0], 'acosh(g(0))')
        check(atanh(g0), SeqVal.OP_ATANH, [g0], 'atanh(g(0))')
        v = acoth(g0)
        check(v, SeqVal.OP_ATANH, [v.args[0]], 'atanh(1 / g(0))')
        v = asech(g0)
        check(v, SeqVal.OP_ACOSH, [v.args[0]], 'acosh(1 / g(0))')
        v = acsch(g0)
        check(v, SeqVal.OP_ASINH, [v.args[0]], 'asinh(1 / g(0))')

        check(sin(g0), SeqVal.OP_SIN, [g0], 'sin(g(0))')
        check(cos(g0), SeqVal.OP_COS, [g0], 'cos(g(0))')
        check(tan(g0), SeqVal.OP_TAN, [g0], 'tan(g(0))')
        v = cot(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / tan(g(0))')
        check(v.args[1], SeqVal.OP_TAN, [g0], 'tan(g(0))')
        v = sec(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / cos(g(0))')
        v = csc(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / sin(g(0))')

        check(sinh(g0), SeqVal.OP_SINH, [g0], 'sinh(g(0))')
        check(cosh(g0), SeqVal.OP_COSH, [g0], 'cosh(g(0))')
        check(tanh(g0), SeqVal.OP_TANH, [g0], 'tanh(g(0))')
        v = coth(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / tanh(g(0))')
        v = sech(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / cosh(g(0))')
        v = csch(g0)
        check(v, SeqVal.OP_DIV, [1, v.args[1]], '1 / sinh(g(0))')

        # hypot
        check(hypot(g0, g1), SeqVal.OP_HYPOT, [g0, g1], 'hypot(g(0), g(1))')
        check(hypot(g0, 4), SeqVal.OP_HYPOT, [g0, 4], 'hypot(g(0), 4)')
        check(hypot(2, g1), SeqVal.OP_HYPOT, [2, g1], 'hypot(2, g(1))')

        check(erf(g0), SeqVal.OP_ERF, [g0], 'erf(g(0))')
        check(erfc(g0), SeqVal.OP_ERFC, [g0], 'erfc(g(0))')
        check(gamma(g0), SeqVal.OP_GAMMA, [g0], 'gamma(g(0))')
        check(gammaln(g0), SeqVal.OP_LGAMMA, [g0], 'gammaln(g(0))')
        check(round(g0), SeqVal.OP_RINT, [g0], 'round(g(0))')

        check(max(g0, g1), SeqVal.OP_MAX, [g0, g1], 'max(g(0), g(1))')
        check(max(g0, 4), SeqVal.OP_MAX, [g0, 4], 'max(g(0), 4)')
        check(max(2, g1), SeqVal.OP_MAX, [2, g1], 'max(2, g(1))')
        check(min(g0, g1), SeqVal.OP_MIN, [g0, g1], 'min(g(0), g(1))')
        check(min(g0, 4), SeqVal.OP_MIN, [g0, 4], 'min(g(0), 4)')
        check(min(2, g1), SeqVal.OP_MIN, [2, g1], 'min(2, g(1))')

        check(rem(g0, g1), SeqVal.OP_MOD, [g0, g1], 'rem(g(0), g(1))')
        check(rem(g0, 4), SeqVal.OP_MOD, [g0, 4], 'rem(g(0), 4)')
        check(rem(2, g1), SeqVal.OP_MOD, [2, g1], 'rem(2, g(1))')

        check(ifelse(g0, g1, 3), SeqVal.OP_SELECT, [g0, g1, 3], 'ifelse(g(0), g(1), 3)')

    def test_optimizations(self):
        # Constant folding at construction time.
        ctx = SeqContext()
        g0, _ = ctx.new_global(SeqVal.TYPE_BOOL)
        g1, _ = ctx.new_global(SeqVal.TYPE_BOOL)

        eq_(g0 + 0, g0)
        eq_(0 + g1, g1)

        eq_(g0 - g0, 0)
        eq_(g1 - 0, g1)

        eq_(g0 * 0, False)
        eq_(g1 * 1, g1)
        eq_(0 * g1, False)
        eq_(1 * g0, g0)

        eq_(+g0, g0)

        eq_(0 / g1, False)
        eq_(ldivide(g0, 0), False)     # g0 \ 0
        eq_(g0 / 1, g0)
        eq_(ldivide(1, g1), g1)        # 1 \ g1

        eq_(g0 < g0, False)
        eq_(g1 > g1, False)
        eq_(g1 >= g1, True)
        eq_(g0 <= g0, True)
        eq_(g1 == g1, True)
        eq_(g0 != g0, False)

        eq_(g0 & 1, g0)
        eq_(g1 & 0, False)
        eq_(1 & g1, g1)
        eq_(0 & g0, False)
        eq_(g0 & g0, g0)

        eq_(g0 | 1, True)
        eq_(g1 | 0, g1)
        eq_(1 | g1, True)
        eq_(0 | g0, g0)
        eq_(g1 | g1, g1)

        eq_(xor(g0, g0), False)
        eq_(xor(g0, False), g0)
        eq_(xor(False, g1), g1)
        ng0 = xor(g0, 1)
        assert ng0.head == SeqVal.OP_NOT
        args_(ng0.args, [g0])
        ng1 = xor(1, g1)
        assert ng1.head == SeqVal.OP_NOT
        args_(ng1.args, [g1])

        eq_(1 ** g1, 1)
        eq_(g0 ** 0, np.int32(1))
        eq_(g1 ** 1, g1)
        g0_2 = g0 ** 2
        assert g0_2.head == SeqVal.OP_MUL
        args_(g0_2.args, [g0, g0])

        rg0 = round(g0)
        eq_(round(rg0), rg0)

        eq_(max(g1, g1), g1)
        eq_(min(g0, g0), g0)

        eq_(ifelse(True, g0, g1), g0)
        eq_(ifelse(False, g0, g1), g1)
        eq_(ifelse(g0, 1, 1), 1)
        eq_(ifelse(g1, g0, g0), g0)
