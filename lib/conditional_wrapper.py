"""conditional_wrapper.py -- transliteration of ``matlab_new/lib/ConditionalWrapper.m``.

A thin delegating proxy ``(seq, cond)`` that re-invokes ``ExpSeqBase``'s
add/wait API with the extra condition threaded in, so ``s.conditional(c).add_step(...)``
works. It is NOT in the sequence tree and emits no bytes of its own -- the bytes
come from the underlying step/pulse whose ``cond`` is gated. A numeric cond is
coerced to logical (``cond != 0``) in the ctor, before any ``&``/``|``.
"""

from mat_utils import is_numeric
from seq_val import and_, or_


class ConditionalWrapper:
    def __init__(self, seq, cond):
        if is_numeric(cond):
            cond = cond != 0
        self.seq = seq
        self.cond = cond

    def add_step(self, *args):
        step, end_time = self.seq.add_step_real(
            self.cond, False, self.seq.cur_seq_time, *args)
        self.seq.cur_seq_time = end_time
        step.end_after_parent = False
        if step.is_step:
            step.totallen_after_parent = False
        self.seq.end_after_parent = True
        return step

    def add_background(self, *args):
        step, _ = self.seq.add_step_real(
            self.cond, True, self.seq.cur_seq_time, *args)
        return step

    def add_floating(self, *args):
        step, _ = self.seq.add_step_real(self.cond, False, float('nan'), *args)
        return step

    def add_at(self, tp, *args):
        step, _ = self.seq.add_step_real(
            self.cond, True, self.seq.get_time_point_offset(tp), *args)
        return step

    def wait(self, t):
        self.seq.wait_with_condition(self.cond, t)
        return self

    def add(self, name, pulse):
        step, _ = self.seq.add_step_real(
            self.cond, True, self.seq.cur_seq_time,
            2 / self.seq.top_level.time_scale)
        step.add(name, pulse)
        step.end_after_parent = False
        step.totallen_after_parent = False
        return step

    def conditional(self, cond):
        return ConditionalWrapper(self.seq, and_(self.cond, cond))

    def conditional_or(self, cond):
        return ConditionalWrapper(self.seq, or_(self.cond, cond))
