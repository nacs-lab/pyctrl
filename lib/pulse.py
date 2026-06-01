"""pulse.py -- transliteration of ``matlab_new/lib/Pulse.m``.

A trivial output record (``id``, ``val``, ``cond``). It emits no bytes itself; its
fields are interned later by ``ExpSeqBase.collect_serialized_pulses`` into the
24-byte output record. ``to_string`` matches the ``seqN.txt`` golden dumps.
"""

from mat_utils import is_logical
from seq_val import to_string


class Pulse:
    def __init__(self, id, val, cond):
        self.id = id
        self.val = val
        self.cond = cond

    def to_string(self):
        if is_logical(self.cond) and self.cond:
            return 'Pulse(id=%d, val=%s)' % (self.id, to_string(self.val))
        return 'Pulse(id=%d, val=%s, cond=%s)' % (
            self.id, to_string(self.val), to_string(self.cond))
