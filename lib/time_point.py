"""time_point.py -- transliteration of ``matlab_new/lib/TimePoint.m``.

A value-type marker ``(seq, anchor, offset)`` consumed by
``ExpSeqBase.get_time_point_offset``. ``anchor`` is 0 (start), 1 (end), or a
fraction/SeqVal. MATLAB ``TimePoint`` is a value class (not a handle); it is built
fresh and never mutated, so a plain immutable-by-convention object is faithful.
"""


class TimePoint:
    def __init__(self, seq, anchor, offset=0):
        self.seq = seq
        self.anchor = anchor
        self.offset = offset
