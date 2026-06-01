"""sqrt_ramp.py -- transliteration of ``matlab_new/lib/sqrtRamp.m``.

Factory returning a 3-arg pulse callable = ``sqrt(<linear ramp body>)``. Uses the
SeqVal-aware ``sqrt`` (builds an OP_SQRT node on a SeqVal, math.sqrt on a number).
Constant-folds to a scalar when both endpoints are equal numbers.
"""

from mat_utils import is_numeric
from seq_val import sqrt


def sqrt_ramp(vstart, vend):
    if is_numeric(vstart) and is_numeric(vend) and vstart == vend:
        return vstart

    def func(t, length, old_val):
        return sqrt((vstart * (length - t) + vend * t) / length)

    return func
