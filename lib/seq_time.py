"""seq_time.py -- transliteration of ``matlab_new/lib/SeqTime.m``.

A time offset from the start of the owning basic sequence, represented as a
**singly-linked chain of additive terms** (``parent`` chain). Each term is either a
number (in ticks) or a ``SeqVal``, and carries its own ``sign`` and obj-counter
``id`` -- both of which are serialized per-term as
``[sign:1B][id:4B][delta_node:4B][prev_id:4B]`` by ``RootSeq.get_time_id``.

This is deliberately NOT the brassboard ``EventTime{static|rt}`` model: collapsing
the chain into a single offset slot would lose the per-term ``id``/``sign`` bytes.
Byte stability depends on reproducing ``create``/``combine``'s exact term-splitting
and obj-id consumption (see PYTHON_FRONTEND_PLAN.md Phase-2 resolved design).

NOTE: the caller rounds ``term`` before calling ``create`` (MATLAB ``round`` is
half-away-from-zero -- see ``mat_utils.mat_round``); ``SeqTime`` never rounds.
"""

from mat_utils import is_logical, is_numeric
from seq_val import SeqVal
from seq_val import to_string as sv_to_string


def is_nan(x):
    """MATLAB ``isnan`` over a tOffset: a SeqTime answers False (SeqTime.isnan),
    ``None`` (empty/at-parent-zero) is not nan, and the literal float nan answers
    True. Lets ``~isnan(tOffset)`` distinguish a positioned SeqTime from a floating
    placeholder."""
    if isinstance(x, SeqTime):
        return False
    if x is None:
        return False
    return x != x


class SeqTime:
    # Sign of the current term (serialized as a 1-byte int8).
    UNKNOWN = 0
    NONNEG = 1
    POS = 2

    def __init__(self, seq, id, sign, parent, term):
        # `term` is pre-scaled (ticks) by the caller. Private ctor: build only
        # via zero()/create()/combine().
        self.seq = seq
        self.id = id
        self.sign = sign
        self.parent = parent
        if parent is not None:
            assert parent.seq is seq
        self.term = term
        self.time_id = 0  # serialization id; 0 == not yet numbered (per-bseq)

    @staticmethod
    def zero(seq):
        return SeqTime(seq, 0, SeqTime.UNKNOWN, None, 0)

    @staticmethod
    def get_var(time):
        # Referenced by TimeSeq.set_time's non-step anchor branch but never defined
        # in MATLAB SeqTime.m (latent bug, only reachable for fractional-anchor
        # subsequence positioning, which TestExpSeq never exercises).
        raise AttributeError("SeqTime.get_var is undefined (matches MATLAB)")

    def iszero(self):
        return (self.parent is None
                and (is_numeric(self.term) or is_logical(self.term))
                and self.term == 0)

    def isnan(self):
        # Always false: lets `~isnan(tOffset)` distinguish a SeqTime from the
        # literal nan-float floating placeholder. (SeqTime.m:44-46)
        return False

    def get_val(self):
        res = self.term
        node = self.parent
        while node is not None:
            term = node.term
            node = node.parent
            if not isinstance(term, SeqVal):
                if isinstance(res, SeqVal) and res.head == SeqVal.OP_ADD:
                    # Merge numerical terms together, keeping the numeric grouped.
                    a0 = res.args[0]
                    a1 = res.args[1]
                    if is_numeric(a0) or is_logical(a0):
                        res = (a0 + term) + a1
                        continue
                    if is_numeric(a1) or is_logical(a1):
                        res = (a1 + term) + a0
                        continue
            res = res + term
        return res

    @staticmethod
    def combine(time1, time2):
        # Combine the terms of time1 and time2, with base sequence = time1.seq.
        if time2.iszero():
            return time1
        if time1.iszero():
            if time2.seq is time1.seq:
                return time2
            return time2._resequence(time1.seq)
        if time2.parent is not None:
            time1 = SeqTime.combine(time1, time2.parent)
        seq = time1.seq
        if is_numeric(time1.term) or is_logical(time1.term):
            if is_numeric(time2.term) or is_logical(time2.term):
                return SeqTime(seq, time2.id, SeqTime.UNKNOWN, time1.parent,
                               time1.term + time2.term)
            # Try to always put the numerical term at the end.
            res = SeqTime(seq, time2.id, time2.sign, time1.parent, time2.term)
            return SeqTime(seq, time1.id, time1.sign, res, time1.term)
        return SeqTime(seq, time2.id, time2.sign, time1, time2.term)

    def create(self, sign, term):
        # The caller is in charge of rounding the `term`.
        seq = self.seq
        selfterm = self.term
        if is_numeric(term) or is_logical(term):
            if term <= 0:
                if sign == SeqTime.POS:
                    raise ValueError('Time offset/length must be positive')
                elif sign == SeqTime.NONNEG:
                    if term < 0:
                        raise ValueError('Time offset/length must not be negative')
                if term == 0:
                    return self
            if is_numeric(selfterm) or is_logical(selfterm):
                # The SeqTime ID is only used for error reporting; no new one needed.
                return SeqTime(seq, 0, SeqTime.UNKNOWN, self.parent, selfterm + term)
        elif is_numeric(selfterm) or is_logical(selfterm):
            # term is a SeqVal onto a numeric self: put the numerical term at the end.
            oldid = self.id
            oldsign = self.sign
            id_ = seq.top_level.seq_ctx.next_obj_id()
            node = SeqTime(seq, id_, sign, self.parent, term)
            return SeqTime(seq, oldid, oldsign, node, selfterm)
        parent = self
        if self.iszero():
            parent = None
        id_ = seq.top_level.seq_ctx.next_obj_id()
        return SeqTime(seq, id_, sign, parent, term)

    def _resequence(self, seq):
        parent = self.parent
        if parent is not None:
            parent = parent._resequence(seq)
        return SeqTime(seq, self.id, self.sign, parent, self.term)

    def to_string(self, ignore_zero=False):
        parent = self.parent
        seq = self.seq
        term = self.term
        if parent is not None:
            parentstr = parent.to_string(True)
        elif seq.parent is not None:
            parentstr = seq.t_offset.to_string(True)
        else:
            parentstr = ''
        if (is_numeric(term) or is_logical(term)) and term == 0:
            termstr = ''
        else:
            termstr = sv_to_string(term)
            if self.sign == SeqTime.POS:
                termstr = termstr + '/p'
            elif self.sign == SeqTime.NONNEG:
                termstr = termstr + '/nn'
        if parentstr == '':
            if termstr == '':
                res = '' if ignore_zero else '0'
            else:
                res = termstr
        else:
            if termstr == '':
                res = parentstr
            else:
                res = parentstr + ' + ' + termstr
        return res
