"""fld.py -- floor-divide for floats, robust to rounding (port of fld.m).

MATLAB's ``rem`` (sign of dividend) is ``math.fmod`` in Python; MATLAB's
``round`` is round-half-away-from-zero (not Python's banker's rounding).
"""

import math


def _mround(v):
    # MATLAB round(): half away from zero.
    return math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)


def fld(x, y):
    r0 = _mround((x - math.fmod(x, y)) / y)
    r1 = r0 - 1
    while r1 <= r0 + 1:
        if r1 * y > x:
            return r1 - 1
        r1 += 1
    return r0 + 1
