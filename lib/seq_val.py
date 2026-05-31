"""seq_val.py -- build-time value algebra for sequence parameters.

Faithful transliteration of ``matlab_new/lib/SeqVal.m`` (class name ``SeqVal``
kept; methods/functions snake_case). A ``SeqVal`` is an expression tree
(``head`` opcode + ``args``); it is *never evaluated here* -- the libnacs engine
evaluates it after deserialization. So this module is a pure build-time AST,
which trivially satisfies brassboard-seq's rtval build/eval separation (there is
no local evaluator to split off). The byte layout each node serializes to lives
in ``seq_context.py`` (``SeqContext.serialize_arg`` / ``ensure_serialize``).

Design note (brassboard-seq, https://github.com/euriqa-brassboard/brassboard-seq):
the build/eval split is inspired by ``src/rtval.h`` + ``src/rtval_interp.h``; no
code is copied. THE ONE RULE (byte-identical ``serialize()`` vs MATLAB) is the
acceptance gate; this file reproduces MATLAB's construction-time constant folding
exactly so the node graph -- and therefore the bytes -- match.

Type mapping (so the serialized constant tags match MATLAB's isfloat/islogical/
isinteger dispatch):
  * Python ``float`` / ``numpy.floating``  -> float64  (ArgConstFloat64)
  * Python ``int``                         -> float64  (a bare MATLAB literal is a
                                                         ``double``; use ``numpy``
                                                         ints for an int32 const)
  * Python ``bool`` / ``numpy.bool_``      -> bool     (ArgConstBool)
  * ``numpy.integer`` (int8/16/32/64, ...) -> int32    (ArgConstInt32)
"""

import builtins
import math

import numpy as np

from num_to_str import num_to_str


class SeqVal:
    # --- opcodes (matlab_new/lib/SeqVal.m) ---------------------------------- #
    OP_ADD = 1
    OP_SUB = 2
    OP_MUL = 3
    OP_DIV = 4
    OP_CMP_LT = 5
    OP_CMP_GT = 6
    OP_CMP_LE = 7
    OP_CMP_GE = 8
    OP_CMP_NE = 9
    OP_CMP_EQ = 10
    OP_AND = 11
    OP_OR = 12
    OP_XOR = 13
    OP_NOT = 14
    OP_ABS = 15
    OP_CEIL = 16
    OP_EXP = 17
    OP_EXPM1 = 18
    OP_FLOOR = 19
    OP_LOG = 20
    OP_LOG1P = 21
    OP_LOG2 = 22
    OP_LOG10 = 23
    OP_POW = 24
    OP_SQRT = 25
    OP_ASIN = 26
    OP_ACOS = 27
    OP_ATAN = 28
    OP_ATAN2 = 29
    OP_ASINH = 30
    OP_ACOSH = 31
    OP_ATANH = 32
    OP_SIN = 33
    OP_COS = 34
    OP_TAN = 35
    OP_SINH = 36
    OP_COSH = 37
    OP_TANH = 38
    OP_HYPOT = 39
    OP_ERF = 40
    OP_ERFC = 41
    OP_GAMMA = 42
    OP_LGAMMA = 43
    OP_RINT = 44
    OP_MAX = 45
    OP_MIN = 46
    OP_MOD = 47
    OP_INTERP = 48
    OP_SELECT = 49
    OP_IDENTITY = 50

    # --- argument type tags ------------------------------------------------- #
    ARG_CONST_BOOL = 0
    ARG_CONST_INT32 = 1
    ARG_CONST_FLOAT64 = 2
    ARG_NODE = 3
    ARG_MEASURE = 4
    ARG_GLOBAL = 5
    ARG_ARG = 6

    # --- global slot value types -------------------------------------------- #
    TYPE_BOOL = 1
    TYPE_INT32 = 2
    TYPE_FLOAT64 = 3

    # --- non-opcode head tags (negative so they cannot clash with opcodes) -- #
    H_MEASURE = -1
    H_GLOBAL = -2
    H_ARG = -3

    # Defer numpy ufuncs (e.g. ``np.int32(2) + seqval``) to our reflected ops.
    __array_ufunc__ = None

    def __init__(self, head, args, ctx):
        self.head = head
        self.args = args          # list (MATLAB cell array)
        self.ctx = ctx
        self.node_id = 0          # serialization id; 0 == unassigned

    # Identity hashing: ``__eq__`` builds a comparison node (below), so the
    # default value-equality hash is unavailable -- keep handle-style identity.
    __hash__ = object.__hash__

    def __repr__(self):
        return 'SeqVal(%s)' % to_string(self)

    # --- arithmetic / comparison operators (dispatch to the module funcs) --- #
    # ``__r*__`` swaps operand order so ``2 - seqval`` == ``minus(2, seqval)``.
    def __add__(self, o): return plus(self, o)
    def __radd__(self, o): return plus(o, self)
    def __sub__(self, o): return minus(self, o)
    def __rsub__(self, o): return minus(o, self)
    def __mul__(self, o): return times(self, o)
    def __rmul__(self, o): return times(o, self)
    def __truediv__(self, o): return rdivide(self, o)
    def __rtruediv__(self, o): return rdivide(o, self)
    def __pow__(self, o): return power(self, o)
    def __rpow__(self, o): return power(o, self)
    def __pos__(self): return self
    def __neg__(self): return SeqVal(SeqVal.OP_MUL, [np.int32(-1), self], self.ctx)

    def __lt__(self, o): return lt(self, o)
    def __gt__(self, o): return gt(self, o)
    def __le__(self, o): return le(self, o)
    def __ge__(self, o): return ge(self, o)
    def __eq__(self, o): return eq(self, o)
    def __ne__(self, o): return ne(self, o)

    def __and__(self, o): return and_(self, o)
    def __rand__(self, o): return and_(o, self)
    def __or__(self, o): return or_(self, o)
    def __ror__(self, o): return or_(o, self)
    def __xor__(self, o): return xor(self, o)
    def __rxor__(self, o): return xor(o, self)
    def __invert__(self): return not_(self)

    def __abs__(self): return SeqVal(SeqVal.OP_ABS, [self], self.ctx)

    def __round__(self, ndigits=None):
        # Note: this rounding mode is actually slightly different from MATLAB.
        if self.head == SeqVal.OP_RINT:
            return self
        return SeqVal(SeqVal.OP_RINT, [self], self.ctx)

    def __floor__(self): return SeqVal(SeqVal.OP_FLOOR, [self], self.ctx)
    def __ceil__(self): return SeqVal(SeqVal.OP_CEIL, [self], self.ctx)

    # MATLAB left-division a\b == b./a, as a method (Python has no ``\`` op).
    def ldivide(self, o):
        return rdivide(o, self)


# --------------------------------------------------------------------------- #
# Type classification -- mirrors MATLAB's isfloat / islogical / isinteger.
# --------------------------------------------------------------------------- #
def _arg_kind(v):
    if isinstance(v, SeqVal):
        return 'node'
    if isinstance(v, (bool, np.bool_)):
        return 'bool'
    if isinstance(v, np.integer):
        return 'int32'
    if isinstance(v, (int, float, np.floating)):
        return 'float64'          # bare Python int == MATLAB double
    return None


# --------------------------------------------------------------------------- #
# Structural equality -- MATLAB ``isequal`` (value-based, type-insensitive for
# numerics; recurses into SeqVal head/args). Used by the construction-time
# folding below. Deliberately ignores ``ctx`` (always the same handle within a
# sequence) and ``node_id`` (transient; 0 during construction when folding runs).
# --------------------------------------------------------------------------- #
def seqval_isequal(a, b):
    a_sv = isinstance(a, SeqVal)
    b_sv = isinstance(b, SeqVal)
    if a_sv != b_sv:
        return False
    if not a_sv:
        return _plain_isequal(a, b)
    if a is b:
        return True
    if a.head != b.head:
        return False
    if len(a.args) != len(b.args):
        return False
    for x, y in zip(a.args, b.args):
        if not seqval_isequal(x, y):
            return False
    return True


def _plain_isequal(a, b):
    a_arr = isinstance(a, (list, tuple, np.ndarray))
    b_arr = isinstance(b, (list, tuple, np.ndarray))
    if a_arr or b_arr:
        if a_arr != b_arr:
            return False
        try:
            return bool(np.array_equal(np.asarray(a, dtype=float),
                                       np.asarray(b, dtype=float)))
        except Exception:
            return False
    try:
        return bool(a == b)        # NaN compares unequal, matching isequal
    except Exception:
        return a is b


# --------------------------------------------------------------------------- #
# Binary operators (module-level, mirroring the MATLAB methods of the same name).
# Each reproduces MATLAB's construction-time constant folding exactly.
# --------------------------------------------------------------------------- #
def plus(a, b):
    if not isinstance(a, SeqVal):
        if a == 0:
            return b
        return SeqVal(SeqVal.OP_ADD, [a, b], b.ctx)
    if not isinstance(b, SeqVal) and b == 0:
        return a
    return SeqVal(SeqVal.OP_ADD, [a, b], a.ctx)


def minus(a, b):
    if seqval_isequal(a, b):
        return 0
    if not isinstance(b, SeqVal):
        if b == 0:
            return a
        return SeqVal(SeqVal.OP_SUB, [a, b], a.ctx)
    return SeqVal(SeqVal.OP_SUB, [a, b], b.ctx)


def times(a, b):
    if not isinstance(a, SeqVal):
        if a == 0:
            return False
        if a == 1:
            return b
        return SeqVal(SeqVal.OP_MUL, [a, b], b.ctx)
    if not isinstance(b, SeqVal):
        if b == 0:
            return False
        if b == 1:
            return a
    return SeqVal(SeqVal.OP_MUL, [a, b], a.ctx)


def rdivide(a, b):
    if not isinstance(a, SeqVal):
        if a == 0:
            return False
        return SeqVal(SeqVal.OP_DIV, [a, b], b.ctx)
    if not isinstance(b, SeqVal) and b == 1:
        return a
    return SeqVal(SeqVal.OP_DIV, [a, b], a.ctx)


def ldivide(a, b):
    # MATLAB a\b == b./a
    return rdivide(b, a)


def power(a, b):
    ctx = a.ctx if isinstance(a, SeqVal) else b.ctx
    if not isinstance(a, SeqVal):
        if a == 1:
            return a
    elif not isinstance(b, SeqVal):
        if b == 0:
            return np.int32(1)
        if b == 1:
            return a
        if b == 2:
            return SeqVal(SeqVal.OP_MUL, [a, a], ctx)
    return SeqVal(SeqVal.OP_POW, [a, b], ctx)


def _ctx_of(a, b):
    return a.ctx if isinstance(a, SeqVal) else b.ctx


def lt(a, b):
    if seqval_isequal(a, b):
        return False
    return SeqVal(SeqVal.OP_CMP_LT, [a, b], _ctx_of(a, b))


def gt(a, b):
    if seqval_isequal(a, b):
        return False
    return SeqVal(SeqVal.OP_CMP_GT, [a, b], _ctx_of(a, b))


def le(a, b):
    if seqval_isequal(a, b):
        return True
    return SeqVal(SeqVal.OP_CMP_LE, [a, b], _ctx_of(a, b))


def ge(a, b):
    if seqval_isequal(a, b):
        return True
    return SeqVal(SeqVal.OP_CMP_GE, [a, b], _ctx_of(a, b))


def ne(a, b):
    if seqval_isequal(a, b):
        return False
    return SeqVal(SeqVal.OP_CMP_NE, [a, b], _ctx_of(a, b))


def eq(a, b):
    if seqval_isequal(a, b):
        return True
    return SeqVal(SeqVal.OP_CMP_EQ, [a, b], _ctx_of(a, b))


def and_(a, b):
    if not isinstance(a, SeqVal):
        return b if a else False
    if not isinstance(b, SeqVal):
        return a if b else False
    if seqval_isequal(a, b):
        return a
    return SeqVal(SeqVal.OP_AND, [a, b], a.ctx)


def or_(a, b):
    if not isinstance(a, SeqVal):
        return True if a else b
    if not isinstance(b, SeqVal):
        return True if b else a
    if seqval_isequal(a, b):
        return a
    return SeqVal(SeqVal.OP_OR, [a, b], a.ctx)


def xor(a, b):
    if not isinstance(a, SeqVal):
        return not_(b) if a else b
    if not isinstance(b, SeqVal):
        return not_(a) if b else a
    if seqval_isequal(a, b):
        return False
    return SeqVal(SeqVal.OP_XOR, [a, b], a.ctx)


def not_(a):
    if not isinstance(a, SeqVal):
        return not a
    return SeqVal(SeqVal.OP_NOT, [a], a.ctx)


# --------------------------------------------------------------------------- #
# Unary math functions. On a SeqVal they build a node; on a plain number they
# fall back to the stdlib so the same name works on both (as MATLAB's globals do).
# --------------------------------------------------------------------------- #
def _unary(op, fallback):
    def f(a):
        if isinstance(a, SeqVal):
            return SeqVal(op, [a], a.ctx)
        return fallback(a)
    return f


abs_ = _unary(SeqVal.OP_ABS, builtins.abs)
ceil = _unary(SeqVal.OP_CEIL, math.ceil)
exp = _unary(SeqVal.OP_EXP, math.exp)
expm1 = _unary(SeqVal.OP_EXPM1, math.expm1)
floor = _unary(SeqVal.OP_FLOOR, math.floor)
log = _unary(SeqVal.OP_LOG, math.log)
log1p = _unary(SeqVal.OP_LOG1P, math.log1p)
log2 = _unary(SeqVal.OP_LOG2, math.log2)
log10 = _unary(SeqVal.OP_LOG10, math.log10)
sqrt = _unary(SeqVal.OP_SQRT, math.sqrt)
asin = _unary(SeqVal.OP_ASIN, math.asin)
acos = _unary(SeqVal.OP_ACOS, math.acos)
atan = _unary(SeqVal.OP_ATAN, math.atan)
asinh = _unary(SeqVal.OP_ASINH, math.asinh)
acosh = _unary(SeqVal.OP_ACOSH, math.acosh)
atanh = _unary(SeqVal.OP_ATANH, math.atanh)
sin = _unary(SeqVal.OP_SIN, math.sin)
cos = _unary(SeqVal.OP_COS, math.cos)
tan = _unary(SeqVal.OP_TAN, math.tan)
sinh = _unary(SeqVal.OP_SINH, math.sinh)
cosh = _unary(SeqVal.OP_COSH, math.cosh)
tanh = _unary(SeqVal.OP_TANH, math.tanh)
erf = _unary(SeqVal.OP_ERF, math.erf)
erfc = _unary(SeqVal.OP_ERFC, math.erfc)
gamma = _unary(SeqVal.OP_GAMMA, math.gamma)
gammaln = _unary(SeqVal.OP_LGAMMA, math.lgamma)


def round(a):
    if isinstance(a, SeqVal):
        return a.__round__()
    return builtins.round(a)


# Derived inverse/reciprocal trig (compositions, exactly as MATLAB defines them).
def acot(a): return atan(1 / a)
def asec(a): return acos(1 / a)
def acsc(a): return asin(1 / a)
def acoth(a): return atanh(1 / a)
def asech(a): return acosh(1 / a)
def acsch(a): return asinh(1 / a)
def cot(a): return 1 / tan(a)
def sec(a): return 1 / cos(a)
def csc(a): return 1 / sin(a)
def coth(a): return 1 / tanh(a)
def sech(a): return 1 / cosh(a)
def csch(a): return 1 / sinh(a)


# --------------------------------------------------------------------------- #
# Binary math functions.
# --------------------------------------------------------------------------- #
def atan2(a, b):
    if not isinstance(a, SeqVal) and not isinstance(b, SeqVal):
        return math.atan2(a, b)
    return SeqVal(SeqVal.OP_ATAN2, [a, b], _ctx_of(a, b))


def hypot(a, b):
    if not isinstance(a, SeqVal) and not isinstance(b, SeqVal):
        return math.hypot(a, b)
    return SeqVal(SeqVal.OP_HYPOT, [a, b], _ctx_of(a, b))


def rem(a, b):
    if not isinstance(a, SeqVal) and not isinstance(b, SeqVal):
        return math.fmod(a, b)
    return SeqVal(SeqVal.OP_MOD, [a, b], _ctx_of(a, b))


def max(a, b):
    if not isinstance(a, SeqVal) and not isinstance(b, SeqVal):
        return builtins.max(a, b)
    if seqval_isequal(a, b):
        return a
    return SeqVal(SeqVal.OP_MAX, [a, b], _ctx_of(a, b))


def min(a, b):
    if not isinstance(a, SeqVal) and not isinstance(b, SeqVal):
        return builtins.min(a, b)
    if seqval_isequal(a, b):
        return a
    return SeqVal(SeqVal.OP_MIN, [a, b], _ctx_of(a, b))


# --------------------------------------------------------------------------- #
# Pretty-printer (display only -- never affects serialized bytes).
# Port of SeqVal.toString / toStringArg / operator_precedence.
# --------------------------------------------------------------------------- #
def operator_precedence(head):
    return {
        SeqVal.OP_ADD: 3, SeqVal.OP_SUB: 3,
        SeqVal.OP_MUL: 2, SeqVal.OP_DIV: 2,
        SeqVal.OP_CMP_LT: 4, SeqVal.OP_CMP_GT: 4,
        SeqVal.OP_CMP_LE: 4, SeqVal.OP_CMP_GE: 4,
        SeqVal.OP_CMP_NE: 5, SeqVal.OP_CMP_EQ: 5,
        SeqVal.OP_AND: 6, SeqVal.OP_OR: 8, SeqVal.OP_POW: 7,
        SeqVal.OP_NOT: 1,
        SeqVal.OP_IDENTITY: 99,
    }.get(head, 0)


def _to_string_arg(self, parent_head):
    res = to_string(self)
    if not isinstance(self, SeqVal):
        return res
    op_self = operator_precedence(self.head)
    op_parent = operator_precedence(parent_head)
    if op_self == 0 or op_parent == 0 or op_self < op_parent:
        return res
    if op_self > op_parent:
        return '(' + res + ')'
    if parent_head == SeqVal.OP_ADD or parent_head == SeqVal.OP_MUL:
        return res
    return '(' + res + ')'


_BINARY_STR = {
    SeqVal.OP_ADD: ' + ', SeqVal.OP_SUB: ' - ', SeqVal.OP_MUL: ' * ',
    SeqVal.OP_DIV: ' / ', SeqVal.OP_CMP_LT: ' < ', SeqVal.OP_CMP_GT: ' > ',
    SeqVal.OP_CMP_LE: ' <= ', SeqVal.OP_CMP_GE: ' >= ', SeqVal.OP_CMP_NE: ' ~= ',
    SeqVal.OP_CMP_EQ: ' == ', SeqVal.OP_AND: ' & ', SeqVal.OP_OR: ' | ',
    SeqVal.OP_POW: ' ^ ',
}
_FUNC_STR = {
    SeqVal.OP_ABS: 'abs', SeqVal.OP_CEIL: 'ceil', SeqVal.OP_EXP: 'exp',
    SeqVal.OP_EXPM1: 'expm1', SeqVal.OP_FLOOR: 'floor', SeqVal.OP_LOG: 'log',
    SeqVal.OP_LOG1P: 'log1p', SeqVal.OP_LOG2: 'log2', SeqVal.OP_LOG10: 'log10',
    SeqVal.OP_SQRT: 'sqrt', SeqVal.OP_ASIN: 'asin', SeqVal.OP_ACOS: 'acos',
    SeqVal.OP_ATAN: 'atan', SeqVal.OP_ASINH: 'asinh', SeqVal.OP_ACOSH: 'acosh',
    SeqVal.OP_ATANH: 'atanh', SeqVal.OP_SIN: 'sin', SeqVal.OP_COS: 'cos',
    SeqVal.OP_TAN: 'tan', SeqVal.OP_SINH: 'sinh', SeqVal.OP_COSH: 'cosh',
    SeqVal.OP_TANH: 'tanh', SeqVal.OP_ERF: 'erf', SeqVal.OP_ERFC: 'erfc',
    SeqVal.OP_GAMMA: 'gamma', SeqVal.OP_LGAMMA: 'gammaln', SeqVal.OP_RINT: 'round',
}
_FUNC2_STR = {
    SeqVal.OP_ATAN2: 'atan2', SeqVal.OP_HYPOT: 'hypot',
    SeqVal.OP_MAX: 'max', SeqVal.OP_MIN: 'min', SeqVal.OP_MOD: 'rem',
}


def to_string(self):
    if not isinstance(self, SeqVal):
        if isinstance(self, (bool, np.bool_)):
            return 'true' if self else 'false'
        if _arg_kind(self) is not None:
            return num_to_str(self)
        raise ValueError('Unknown value type.')

    head = self.head
    args = self.args
    if head == SeqVal.OP_INTERP:
        args = args[0:3]
    strargs = ([_to_string_arg(x, head) for x in args] if head >= 0 else None)

    if head in _BINARY_STR:
        return strargs[0] + _BINARY_STR[head] + strargs[1]
    if head in _FUNC_STR:
        return '%s(%s)' % (_FUNC_STR[head], strargs[0])
    if head in _FUNC2_STR:
        return '%s(%s, %s)' % (_FUNC2_STR[head], strargs[0], strargs[1])
    if head == SeqVal.OP_XOR:
        return 'xor(%s, %s)' % (strargs[0], strargs[1])
    if head == SeqVal.OP_NOT:
        return '~' + strargs[0]
    if head == SeqVal.OP_INTERP:
        return 'interp(%s, %s, %s, %s)' % (strargs[0], strargs[1], strargs[2],
                                           _json_vector(self.args[3]))
    if head == SeqVal.OP_SELECT:
        return 'ifelse(%s, %s, %s)' % (strargs[0], strargs[1], strargs[2])
    if head == SeqVal.OP_IDENTITY:
        return strargs[0]
    if head == SeqVal.H_MEASURE:
        return 'm(%d)' % int(args[0])
    if head == SeqVal.H_GLOBAL:
        return 'g(%d)' % int(args[0])
    if head == SeqVal.H_ARG:
        return 'arg(%d)' % int(args[0])
    raise ValueError('Unknown value type')


def _json_vector(vals):
    # MATLAB jsonencode of a numeric row vector: comma-separated, no spaces, and
    # integer-valued doubles print without a fractional part (num_to_str matches).
    return '[' + ','.join(num_to_str(float(v)) for v in np.atleast_1d(vals)) + ']'
