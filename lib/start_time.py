"""start_time.py -- transliteration of ``matlab_new/lib/startTime.m``.

Convenience wrapper: the start (anchor=0) of a (sub)sequence.
"""

from time_point import TimePoint


def start_time(seq, offset=0):
    return TimePoint(seq, 0, offset)
