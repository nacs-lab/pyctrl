"""end_time.py -- transliteration of ``matlab_new/lib/endTime.m``.

Convenience wrapper: the end (anchor=1) of a (sub)sequence.
"""

from time_point import TimePoint


def end_time(seq, offset=0):
    return TimePoint(seq, 1, offset)
