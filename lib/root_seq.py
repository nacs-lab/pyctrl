"""root_seq.py -- transliteration of ``matlab_new/lib/RootSeq.m``.

Root of a basic sequence (base of both ``ExpSeq`` and ``BasicSeq``): owns the
per-basic-sequence serialized state -- the time table, time orders, measures,
global assigns, and branches -- plus ``get_time_id`` (lazy 1-based per-bseq time
numbering) and ``serialize_bseq`` (the per-bseq byte block).

The property ``time_serialized`` (storage) collides with the method ``timeSerialized``
under snake_case, so the storage is ``self._time_serialized``.
"""

import struct

from exp_seq_base import ExpSeqBase
from seq_time import SeqTime
from seq_val import SeqVal
from seq_val import to_string as sv_to_string


class RootSeq(ExpSeqBase):
    def _init_root(self):
        # RootSeq mutable per-bseq state.
        self.assigns = []           # index == global slot; None == unset
        self.norders = 0
        self.orders = []            # {id, sign, before, after}
        self.default_target = None
        self.branches = []          # {cond, target, id}
        self._time_serialized = []  # list of byte records, 1-based time_id
        self.measures = []          # {time, chn, id}
        self.before_bseq_cbs = []
        self.after_bseq_cbs = []
        self.after_branch_cbs = []
        # ExpSeqBase mutable fields (no MATLAB ctor; property defaults).
        self.sub_seqs = []
        self.n_sub_seqs = 0
        self.latest_seq = True

    # -- branches / globals -------------------------------------------------- #
    def assign_global(self, g, val):
        assert isinstance(g, SeqVal) and g.head == SeqVal.H_GLOBAL
        id_ = self.top_level.seq_ctx.next_obj_id()
        slot = g.args[0]
        while len(self.assigns) <= slot:
            self.assigns.append(None)
        self.assigns[slot] = {'val': val, 'id': id_}

    def cond_branch(self, cond, target):
        self.check_branch_target(target)
        id_ = self.top_level.seq_ctx.next_obj_id()
        self.branches.append({'cond': cond, 'target': target, 'id': id_})

    def default_branch(self, target):
        self.check_branch_target(target)
        self.default_target = target

    def check_branch_target(self, target):
        if target is not None:
            if not isinstance(target, RootSeq):
                raise ValueError(
                    'Only the toplevel sequence (`ExpSeq`) or other basic sequences '
                    '(return values of `newBasicSeq`) are valid branch target')
            if self.top_level is not target.top_level:
                raise ValueError(
                    'Only basic sequences in the same top level sequence are valid '
                    'branch target. You should create new basic sequence with '
                    '`newBasicSeq()` instead of `ExpSeq()`')

    def global_path(self):
        return []

    # -- time numbering ------------------------------------------------------ #
    def get_time_id(self, time):
        time_id = time.time_id
        if time_id != 0:
            return time_id
        parent = time.parent
        if parent is not None:
            prev_id = parent.time_id
            if prev_id == 0:
                prev_id = self.get_time_id(parent)
        else:
            seq = time.seq
            if seq.parent is not None:
                parent = seq.t_offset
                prev_id = parent.time_id
                if prev_id == 0:
                    prev_id = self.get_time_id(parent)
            else:
                prev_id = 0
        time_id = len(self._time_serialized) + 1
        time.time_id = time_id
        term_id = self.top_level.seq_ctx.get_val_id(time.term)
        # [sign: 1B][id: 4B][delta_node: 4B][prev_id: 4B]
        rec = (bytes([time.sign & 0xFF])
               + struct.pack('<3I', time.id & 0xFFFFFFFF,
                             term_id & 0xFFFFFFFF, prev_id & 0xFFFFFFFF))
        self._time_serialized.append(rec)
        return time_id

    def time_serialized(self):
        # [ntimes: 4B][record x ntimes]
        return struct.pack('<I', len(self._time_serialized)) + b''.join(self._time_serialized)

    # -- orders -------------------------------------------------------------- #
    def add_order(self, sign, before, after):
        self.norders += 1
        id_ = self.top_level.seq_ctx.next_obj_id()
        self.orders.append({'id': id_, 'sign': sign, 'before': before, 'after': after})

    def add_equal(self, time1, time2):
        self.add_order(SeqTime.NONNEG, time1, time2)
        self.add_order(SeqTime.NONNEG, time2, time1)

    # -- per-basic-sequence serialization ------------------------------------ #
    def serialize_bseq(self):
        seq_ctx = self.top_level.seq_ctx
        cid_map = self.top_level.cid_map

        # [nendtimes: 4B][[time_id: 4B] x nendtimes]
        endtimes = self.collect_end_time([])
        endtimes_serialized = struct.pack('<I', len(endtimes)) + b''.join(endtimes)

        # [ntimeorders: 4B][[sign:1B][id:4B][before_id:4B][after_id:4B] x n]
        orders_parts = []
        for i in range(self.norders):
            order = self.orders[i]
            orders_parts.append(
                bytes([order['sign'] & 0xFF])
                + struct.pack('<3I', order['id'] & 0xFFFFFFFF,
                              self.get_time_id(order['before']) & 0xFFFFFFFF,
                              self.get_time_id(order['after']) & 0xFFFFFFFF))
        orders_serialized = struct.pack('<I', self.norders) + b''.join(orders_parts)

        # [noutputs: 4B][[id][time_id][len][val][cond][chn] x n]
        pulses = self.collect_serialized_pulses([])
        pulses_serialized = struct.pack('<I', len(pulses)) + b''.join(pulses)

        # [nmeasures: 4B][[id][time_id][chn] x n] -- nmeasures counts skipped ones too.
        measures = self.measures
        nmeasures = len(measures)
        measures_parts = []
        for i in range(nmeasures):
            measure = measures[i]
            cid = cid_map.get(measure['chn'], 0)
            if cid == 0:
                continue
            measures_parts.append(struct.pack(
                '<3I', measure['id'] & 0xFFFFFFFF,
                self.get_time_id(measure['time']) & 0xFFFFFFFF, cid & 0xFFFFFFFF))
        measures_serialized = struct.pack('<I', nmeasures) + b''.join(measures_parts)

        # [nassigns: 4B][[assign_id][global_id][val] x n] -- count = non-empty only.
        assigns_parts = []
        for i in range(len(self.assigns)):
            assign = self.assigns[i]
            if assign is None or assign['val'] is None:
                continue
            assigns_parts.append(struct.pack(
                '<3I', assign['id'] & 0xFFFFFFFF, i & 0xFFFFFFFF,
                seq_ctx.get_val_id(assign['val']) & 0xFFFFFFFF))
        assigns_serialized = struct.pack('<I', len(assigns_parts)) + b''.join(assigns_parts)

        # [nbranches: 4B][[branch_id][target_id][cond] x n][default_target: 4B]
        nbranches = len(self.branches)
        branches_parts = []
        for i in range(nbranches):
            branch = self.branches[i]
            target = branch['target']
            target_id = 0 if target is None else target.bseq_id
            cond_id = seq_ctx.get_val_id(branch['cond'])
            branches_parts.append(struct.pack(
                '<3I', branch['id'] & 0xFFFFFFFF, target_id & 0xFFFFFFFF,
                cond_id & 0xFFFFFFFF))
        default_target = self.default_target
        default_target_id = 0 if default_target is None else default_target.bseq_id
        branches_serialized = (struct.pack('<I', nbranches) + b''.join(branches_parts)
                               + struct.pack('<I', default_target_id))

        return (self.time_serialized() + endtimes_serialized + orders_serialized
                + pulses_serialized + measures_serialized + assigns_serialized
                + branches_serialized)

    # -- basic-sequence callbacks (register only; no byte impact) ------------ #
    def reg_before_bseq(self, cb):
        self.before_bseq_cbs.append(cb)
        return self

    def reg_after_bseq(self, cb):
        self.after_bseq_cbs.append(cb)
        return self

    def reg_after_branch(self, cb):
        self.after_branch_cbs.append(cb)
        return self

    # -- top-level forwarders ------------------------------------------------ #
    def new_basic_seq(self, *args):
        return self.top_level.new_basic_seq(*args)

    def add_ttl_mgr(self, *args):
        self.top_level.add_ttl_mgr(*args)

    def disable_channel(self, *args):
        self.top_level.disable_channel(*args)

    def check_channel_disabled(self, *args):
        return self.top_level.check_channel_disabled(*args)

    def reg_before_start(self, *args):
        self.top_level.reg_before_start(*args)
        return self

    def reg_after_end(self, *args):
        self.top_level.reg_after_end(*args)
        return self

    def get_global(self, *args):
        return self.top_level.get_global(*args)

    def set_global(self, *args):
        self.top_level.set_global(*args)

    # -- display ------------------------------------------------------------- #
    def to_string(self, indent=0):
        prefix = ' ' * indent
        res = '%sBS%d:\n' % (prefix, self.bseq_id)
        if self.assigns:
            res += prefix + '  Assigns:\n'
            for i in range(len(self.assigns)):
                assign = self.assigns[i]
                if assign is None or assign['val'] is None:
                    continue
                res += prefix + ('    g(%d) = ' % (i + 1)) + sv_to_string(assign['val']) + '\n'
        if self.norders != 0:
            res += prefix + '  Time orders:\n'
            for i in range(self.norders):
                order = self.orders[i]
                op = ' < ' if order['sign'] == SeqTime.POS else ' <= '
                res += (prefix + '    ' + order['before'].to_string() + op
                        + order['after'].to_string() + '\n')
        res += prefix + '  Branches:\n'
        for branch in self.branches:
            target = ('end' if branch['target'] is None
                      else 'BS%d' % branch['target'].bseq_id)
            res += prefix + '    ' + sv_to_string(branch['cond']) + ': ' + target + '\n'
        target = ('end' if self.default_target is None
                  else 'BS%d' % self.default_target.bseq_id)
        res += prefix + '    default: ' + target + '\n'
        res += ExpSeqBase.to_string(self, indent + 2)
        return res
