"""time_seq.py -- transliteration of ``matlab_new/lib/TimeSeq.m``.

Abstract base of ``TimeStep`` (leaf) and ``ExpSeqBase`` (non-leaf). Holds the
fields shared by every node of the sequence DAG and the floating-position
(``set_time``/``set_end_time``) + global/channel forwarding API.

Field naming: MATLAB camelCase properties are snake_cased (``topLevel`` ->
``top_level``, ``tOffset`` -> ``t_offset``, ``curSeqTime`` -> ``cur_seq_time``),
matching the Phase-1 convention. ``tOffset`` is tri-state: ``None`` (starts at
parent t=0), ``float('nan')`` (floating, not yet positioned), or a ``SeqTime``.
The property ``global_path`` (storage) collides with the method ``globalPath``
under snake_case, so the storage is ``self._global_path``.
"""

from mat_utils import is_numeric, mat_round
from seq_time import SeqTime, is_nan


class TimeSeq:
    # Property defaults (subclass ctors set the wiring fields directly; these are
    # the MATLAB property defaults for everything not set in the ctor).
    config = None
    parent = None
    t_offset = None
    top_level = None
    root = None
    is_step = False
    cond = True
    end_after_parent = True
    totallen_after_parent = True
    _global_path = None

    def get_condition(self):
        return self.cond

    def translate_channel(self, name):
        # Wrapper for ExpSeq.translate_channel (overridden on ExpSeq itself).
        return self.top_level.translate_channel(name)

    def set_time(self, time, anchor=0, offset=0):
        # Position a currently-floating step/sub-sequence (self). The `anchor`
        # fraction of self is placed `offset` after `time`.
        if not is_nan(self.t_offset):
            raise ValueError('Not a floating sequence.')
        tdiff = self.parent.get_time_point_offset(time)
        tdiff = tdiff.create(SeqTime.UNKNOWN, mat_round(offset))
        if not is_numeric(anchor) or anchor != 0:
            if self.is_step:
                length = mat_round(self.len * anchor)
            else:
                length = SeqTime.get_var(self.cur_seq_time)
                if is_numeric(anchor) and anchor == 1:
                    if not is_numeric(length):
                        self.root.add_equal(self.cur_seq_time, tdiff)
                else:
                    length = mat_round(length * anchor)
            tdiff2 = tdiff.create(SeqTime.UNKNOWN, -length)
            if is_numeric(anchor) and anchor == 1:
                if not self.is_step:
                    self.root.add_equal(tdiff, self.cur_seq_time)
                elif not is_numeric(length):
                    self.root.add_equal(tdiff, tdiff2.create(self.length_sign(), length))
            tdiff = tdiff2
        self.t_offset = tdiff

    def set_end_time(self, time, offset=0):
        self.set_time(time, 1, offset)

    def new_global(self, *args):
        return self.top_level.new_global_real(False, *args)

    def new_persist_global(self, *args):
        return self.top_level.new_global_real(True, *args)

    def assign_global(self, g, val):
        self.root.assign_global(g, val)

    def global_path(self):
        p = self._global_path
        if not p:
            self._global_path = list(self.parent.global_path())
            self._global_path.append(self)
            p = self._global_path
        return p
