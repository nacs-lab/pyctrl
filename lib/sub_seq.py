"""sub_seq.py -- transliteration of ``matlab_new/lib/SubSeq.m``.

A child sub-sequence: a thin ctor over ``ExpSeqBase`` that caches shared fields
from the parent and appends itself to ``parent.sub_seqs``. (MATLAB relies on
property defaults for ``sub_seqs``/``n_sub_seqs``; Python initializes them here.)
"""

from exp_seq_base import ExpSeqBase
from seq_time import SeqTime


class SubSeq(ExpSeqBase):
    def __init__(self, parent, toffset, cond):
        self.parent = parent
        self.t_offset = toffset
        self.cond = cond
        self.config = parent.config
        self.top_level = parent.top_level
        self.root = parent.root
        self.C = parent.C
        self.G = parent.G
        self.sub_seqs = []
        self.n_sub_seqs = 0
        self.cur_seq_time = SeqTime.zero(self)
        self.latest_seq = parent.latest_seq
        parent.n_sub_seqs += 1
        parent.sub_seqs.append(self)
