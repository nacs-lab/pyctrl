"""cld.py -- ceil-divide for floats (port of cld.m)."""

from fld import fld


def cld(x, y):
    return -fld(-x, y)
