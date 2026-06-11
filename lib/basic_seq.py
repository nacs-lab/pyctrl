"""basic_seq.py -- transliteration of ``matlab_new/lib/BasicSeq.m``.

A non-top-level basic sequence (a branch target). A thin ctor over ``RootSeq``: it
is its own ``root`` but shares the top-level's ``config``/``C``/``seq_ctx``. It
appends itself to the top-level's flat ``basic_seqs`` list and takes
``bseq_id = len(basic_seqs) + 1`` AFTER appending -- so the first one is id 2 (the
``ExpSeq`` itself is id 1 and is always serialized as basic-sequence #1). That id
is what a branch ``target_id`` encodes.
"""

from root_seq import RootSeq
from seq_time import SeqTime


class BasicSeq(RootSeq):
    def __init__(self, parent):
        self.config = parent.config
        self.top_level = parent
        self.root = self
        self.C = parent.C
        self._init_root()
        self.pattern = getattr(parent, "pattern", None)   # inherit top-level's pattern (override via set_pattern)
        self.zero_time = SeqTime.zero(self)
        self.cur_seq_time = self.zero_time
        parent.basic_seqs.append(self)
        self.bseq_id = len(parent.basic_seqs) + 1
