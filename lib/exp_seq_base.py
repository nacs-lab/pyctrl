"""exp_seq_base.py -- transliteration of ``matlab_new/lib/ExpSeqBase.m``.

Non-leaf node of the sequence DAG: the user-facing timing/build API (``add_step``,
``add_background``, ``add_floating``, ``add_at``, ``wait``, ``wait_all``,
``wait_for``, ``wait_background``, ``conditional``, ``add_measure``, ``add``,
``align_end``) plus the serialization walkers (``collect_serialized_pulses``,
``collect_end_time``) and the offset/total-time machinery.

Every byte hinges on (a) the global ``obj_counter`` interleaving -- so the control
flow follows MATLAB call-for-call, including the inlined id bumps in ``SeqTime`` and
``TimeStep`` -- and (b) half-away-from-zero rounding at each ``round(t*time_scale)``
site (``mat_round``), never inside ``SeqTime``.
"""

import struct

import provenance
from conditional_wrapper import ConditionalWrapper
from ifelse import ifelse
from mat_utils import is_logical, is_numeric, mat_round
from seq_time import SeqTime, is_nan
from seq_val import SeqVal
from seq_val import max as sv_max
from time_seq import TimeSeq
from time_step import TimeStep

_NO_OFFSET = object()


class ExpSeqBase(TimeSeq):
    # Tree-structure fields (subclass ctors set cur_seq_time / C / G / sub_seqs).
    n_sub_seqs = 0
    latest_seq = True
    C = None
    G = None

    # -- Add steps / sub-sequences ------------------------------------------- #
    def add_step(self, *args):
        step, end_time = self.add_step_real(True, False, self.cur_seq_time, *args)
        self.cur_seq_time = end_time
        step.end_after_parent = False
        if step.is_step:
            step.totallen_after_parent = False
        self.end_after_parent = True
        return step

    def add_background(self, *args):
        step, _ = self.add_step_real(True, True, self.cur_seq_time, *args)
        return step

    def add_floating(self, *args):
        step, _ = self.add_step_real(True, False, float('nan'), *args)
        return step

    def add_at(self, tp, *args):
        step, _ = self.add_step_real(True, True, self.get_time_point_offset(tp), *args)
        return step

    # -- Wait API ------------------------------------------------------------ #
    def wait(self, t):
        if is_numeric(t):
            if t < 0:
                if is_logical(self.cond) and self.cond:
                    raise ValueError('Wait time cannot be negative.')
            elif t == 0:
                return self
        if is_logical(self.cond) and not self.cond:
            return self
        _pt0 = provenance.wait_start(self)         # INERT unless a session is active
        self.cur_seq_time = self.cur_seq_time.create(
            SeqTime.NONNEG,
            ifelse(self.cond, mat_round(t * self.top_level.time_scale), 0))
        provenance.wait_end(self, t, _pt0)         # records the param-driven time region
        self.end_after_parent = True
        node = self
        while not node.latest_seq:
            node.totallen_after_parent = True
            node.latest_seq = True
            node = node.parent
            if node is None:
                break
        return self

    def wait_all(self):
        self.cur_seq_time = self.wait_all_time(True)
        self.end_after_parent = True
        return self

    def wait_for(self, steps, offset=_NO_OFFSET):
        if offset is _NO_OFFSET:
            offset = 0
            hasoffset = False
            nonnegoffset = True
        elif is_numeric(offset):
            hasoffset = offset != 0
            nonnegoffset = offset >= 0
        else:
            hasoffset = True
            nonnegoffset = False
        step_list = steps if isinstance(steps, (list, tuple)) else [steps]
        t = self.cur_seq_time
        tval = t.get_val()
        has_other_parent = False
        for real_step in step_list:
            if self is real_step:
                raise ValueError('Cannot wait for the sequence itself.')
            if self.check_parent(real_step):
                raise ValueError('Cannot wait for parent sequence.')
            step_toffset = real_step.t_offset
            if is_nan(step_toffset):
                raise ValueError('Cannot get offset of floating sequence.')
            if real_step.parent is not self:
                step_toffset = SeqTime.combine(
                    self.offset_diff(real_step.parent), step_toffset)
                has_other_parent = True
            assert step_toffset.seq is self
            if real_step.is_step:
                tstep = step_toffset.create(real_step.length_sign(),
                                            mat_round(real_step.len))
            else:
                tstep = SeqTime.combine(step_toffset, real_step.cur_seq_time)
            if hasoffset:
                tstep = tstep.create(SeqTime.UNKNOWN,
                                     mat_round(offset * self.top_level.time_scale))
            if nonnegoffset and real_step.parent is self:
                real_step.end_after_parent = False
                if real_step.is_step:
                    real_step.totallen_after_parent = False
            new_tval = sv_max(tval, tstep.get_val())
            new_t = SeqTime.zero(self).create(SeqTime.NONNEG, new_tval)
            self.root.add_order(SeqTime.NONNEG, tstep, new_t)
            self.root.add_order(SeqTime.NONNEG, t, new_t)
            tval = new_tval
            t = new_t
        self.cur_seq_time = t
        self.end_after_parent = True
        if (not self.latest_seq) and (
                has_other_parent or isinstance(offset, SeqVal) or offset > 0):
            node = self
            while not node.latest_seq:
                node.totallen_after_parent = True
                node.latest_seq = True
                node = node.parent
                if node is None:
                    break
        return self

    def wait_background(self):
        t = self.cur_seq_time
        tval = t.get_val()
        for i in range(self.n_sub_seqs):
            sub_seq = self.sub_seqs[i]
            step_toffset = sub_seq.t_offset
            if not sub_seq.end_after_parent:
                continue
            if is_nan(step_toffset):
                raise ValueError('Cannot get offset of floating sequence.')
            if sub_seq.is_step:
                tstep = step_toffset.create(sub_seq.length_sign(),
                                            mat_round(sub_seq.len))
            else:
                tstep = SeqTime.combine(step_toffset, sub_seq.cur_seq_time)
            sub_seq.end_after_parent = False
            new_tval = sv_max(tval, tstep.get_val())
            new_t = SeqTime.zero(self).create(SeqTime.NONNEG, new_tval)
            self.root.add_order(SeqTime.NONNEG, tstep, new_t)
            self.root.add_order(SeqTime.NONNEG, t, new_t)
            tval = new_tval
            t = new_t
        self.cur_seq_time = t
        self.end_after_parent = True
        return self

    # -- Condition ----------------------------------------------------------- #
    def conditional(self, cond, *args):
        res = ConditionalWrapper(self, cond)
        if args:
            cb = args[0]
            assert (not is_numeric(cb) and not is_logical(cb)
                    and not isinstance(cb, SeqVal))
            res = res.add_step(cb, *args[1:])
        return res

    # -- Measure ------------------------------------------------------------- #
    def add_measure(self, chn):
        seq_ctx = self.top_level.seq_ctx
        res, id_ = seq_ctx.new_measure()
        if not is_numeric(chn):
            chn = self.top_level.translate_channel(chn)
        self.root.measures.append({'time': self.cur_seq_time, 'chn': chn, 'id': id_})
        return res

    # -- Other helpers ------------------------------------------------------- #
    def add(self, name, pulse):
        if (not is_numeric(pulse) and not is_logical(pulse)
                and not isinstance(pulse, SeqVal)):
            raise ValueError('Use addStep to add a ramp pulse.')
        # The 2-tick length is just a placeholder (ignored by wait/total-time).
        step, _ = self.add_step_real(True, True, self.cur_seq_time,
                                     2 / self.top_level.time_scale)
        step.add(name, pulse)
        step.end_after_parent = False
        step.totallen_after_parent = False
        return step

    def align_end(self, *args):
        if not args:
            raise ValueError('Requires at least one sequence to align')
        if isinstance(args[-1], TimeSeq):
            subseqs = list(args)
            offset = 0
            hasoffset = False
        elif len(args) == 1:
            raise ValueError('Requires at least one sequence to align')
        else:
            subseqs = list(args[:-1])
            offset = ifelse(self.cond,
                            mat_round(args[-1] * self.top_level.time_scale), 0)
            hasoffset = not is_numeric(offset) or offset != 0
        nsubseqs = len(subseqs)
        maxlen = None
        lens = [None] * nsubseqs
        maxsign = SeqTime.NONNEG
        signs = [0] * nsubseqs
        times = [None] * nsubseqs
        for i in range(nsubseqs):
            subseq = subseqs[i]
            if not is_nan(subseq.t_offset):
                raise ValueError('alignEnd requires floating sequences as inputs.')
            assert subseq.parent is self
            if subseq.is_step:
                length = mat_round(subseq.len)
                sign = subseq.length_sign()
                if sign == SeqTime.POS:
                    maxsign = SeqTime.POS
            else:
                time = subseq.cur_seq_time
                times[i] = time
                length = time.get_val()
                sign = SeqTime.NONNEG
            maxlen = length if maxlen is None else sv_max(maxlen, length)
            lens[i] = length
            signs[i] = sign
        curtime = self.cur_seq_time
        if hasoffset:
            curtime = curtime.create(SeqTime.UNKNOWN, offset)
        endtime = curtime.create(maxsign, maxlen)
        for i in range(nsubseqs):
            subseq = subseqs[i]
            length = lens[i]
            starttime = endtime.create(SeqTime.UNKNOWN, -length)
            subseq.t_offset = starttime
            if nsubseqs > 1:
                self.root.add_order(SeqTime.NONNEG, curtime, starttime)
                time = times[i]
                if time is None:
                    if not is_numeric(length):
                        self.root.add_equal(endtime,
                                            starttime.create(signs[i], length))
                else:
                    self.root.add_equal(endtime, time)
        return subseqs

    def cur_time(self):
        return self.cur_seq_time.get_val() / self.top_level.time_scale

    def total_time(self):
        res, _ = self.total_time_raw()
        return res / self.top_level.time_scale

    def get_time_point_offset(self, time):
        from time_point import TimePoint
        if not isinstance(time, TimePoint):
            raise ValueError('`TimePoint` expected.')
        other = time.seq
        tdiff = self.offset_diff(other)
        tdiff = tdiff.create(SeqTime.UNKNOWN,
                             mat_round(time.offset * self.top_level.time_scale))
        if not is_numeric(time.anchor) or time.anchor != 0:
            if other.is_step:
                tdiff = tdiff.create(SeqTime.UNKNOWN,
                                     mat_round(other.len * time.anchor))
            elif is_numeric(time.anchor) and time.anchor == 1:
                tdiff = SeqTime.combine(tdiff, other.cur_seq_time)
            else:
                tdiff = tdiff.create(
                    SeqTime.UNKNOWN,
                    mat_round(other.cur_seq_time.get_val() * time.anchor))
        return tdiff

    def to_string(self, indent=0):
        prefix = ' ' * indent
        from seq_val import to_string as sv_to_string
        if is_logical(self.cond) and self.cond:
            res = '%s%s()' % (prefix, type(self).__name__)
        else:
            res = '%s%s(cond=%s)' % (prefix, type(self).__name__,
                                     sv_to_string(self.cond))
        if self.parent is not None:
            res = res + ' @ ' + self.t_offset.to_string()
        for i in range(self.n_sub_seqs):
            res = res + '\n' + self.sub_seqs[i].to_string(indent + 2)
        for measure in self.root.measures:
            time = measure['time']
            chn = measure['chn']
            if time.seq is not self:
                continue
            res = res + '\n' + prefix + ('  Measure(id=%d, val=m(%d), chn%d(%s))' % (
                measure['id'], measure['id'], chn,
                self.top_level.channel_name(chn))) + ' @ ' + time.to_string()
        res = res + '\n' + prefix + '  curSeqTime: ' + self.cur_seq_time.to_string()
        return res

    # -- Private helpers ----------------------------------------------------- #
    def check_parent(self, other):
        if other.parent is None:
            return True
        node = self.parent
        while node is not None:
            if node is other:
                return True
            node = node.parent
        return False

    def offset_diff(self, step):
        res = SeqTime.zero(self)
        self_path = self.global_path()
        other_path = step.global_path()
        nself = len(self_path)
        nother = len(other_path)
        has_neg = False
        for i in range(max(nself, nother)):
            if i < nself:
                self_ele = self_path[i]
                if i < nother:
                    other_ele = other_path[i]
                    if self_ele is other_ele:
                        continue
                    other_offset = other_ele.t_offset
                    if is_nan(other_offset):
                        raise ValueError(
                            'Cannot compute offset different for floating sequence')
                    res = SeqTime.combine(res, other_offset)
                    self_offset = self_ele.t_offset
                    if is_nan(self_offset):
                        raise ValueError(
                            'Cannot compute offset different for floating sequence')
                    if not self_offset.iszero():
                        has_neg = True
                        res = res.create(SeqTime.UNKNOWN, -self_offset.get_val())
                else:
                    self_offset = self_ele.t_offset
                    if is_nan(self_offset):
                        raise ValueError(
                            'Cannot compute offset different for floating sequence')
                    if not self_offset.iszero():
                        has_neg = True
                        res = res.create(SeqTime.UNKNOWN, -self_offset.get_val())
            else:
                other_ele = other_path[i]
                other_offset = other_ele.t_offset
                if is_nan(other_offset):
                    raise ValueError(
                        'Cannot compute offset different for floating sequence')
                res = SeqTime.combine(res, other_offset)
        if has_neg:
            if step.t_offset is None:
                self.root.add_equal(step.zero_time, res)
            else:
                self.root.add_equal(step.t_offset, res)
        return res

    def total_time_raw(self):
        res = self.cur_seq_time.get_val()
        curtime_only = True
        for i in range(self.n_sub_seqs):
            sub_seq = self.sub_seqs[i]
            if not sub_seq.totallen_after_parent:
                continue
            if sub_seq.is_step:
                sub_end = sub_seq.len
            else:
                sub_end, sub_curtime_only = sub_seq.total_time_raw()
                if sub_curtime_only and not sub_seq.end_after_parent:
                    continue
            if is_nan(sub_seq.t_offset):
                raise ValueError('Cannot get total time with floating sub sequence.')
            sub_end = sub_end + sub_seq.t_offset.get_val()
            curtime_only = False
            res = sv_max(res, sub_end)
        return res, curtime_only

    # -- Serialization walkers ----------------------------------------------- #
    def collect_serialized_pulses(self, res):
        toplevel = self.top_level
        seq_ctx = toplevel.seq_ctx
        cid_map = toplevel.cid_map
        for i in range(self.n_sub_seqs):
            sub_seq = self.sub_seqs[i]
            if not sub_seq.is_step:
                res = sub_seq.collect_serialized_pulses(res)
                continue
            # [id: 4B][time_id: 4B][len: 4B][val: 4B][cond: 4B][chn: 4B]
            # MATLAB's `pulses` cell is preallocated (length >= 8), so its
            # `npulses == 0` guard never fires for a TimeStep: time_id + len_id
            # (raw_len) are interned even when the step has no pulses. Reproduce
            # that -- a condition-disabled step still interns its raw_len.
            pulses = sub_seq.pulses
            time_id = self.root.get_time_id(sub_seq.t_offset)
            len_id = seq_ctx.get_val_id(sub_seq.raw_len)
            for chn in sorted(pulses.keys()):
                pulse = pulses[chn]
                cid = cid_map.get(chn, 0)
                if cid == 0:
                    continue
                if is_logical(pulse.cond):
                    if not pulse.cond:
                        continue
                    cond_id = 0
                else:
                    cond_id = seq_ctx.get_val_id(pulse.cond)
                val_id = seq_ctx.get_val_id(pulse.val)
                res.append(struct.pack('<6I', pulse.id & 0xFFFFFFFF,
                                       time_id & 0xFFFFFFFF, len_id & 0xFFFFFFFF,
                                       val_id & 0xFFFFFFFF, cond_id & 0xFFFFFFFF,
                                       cid & 0xFFFFFFFF))
        return res

    def wait_all_time(self, setflag):
        t = self.cur_seq_time
        tval = t.get_val()
        for i in range(self.n_sub_seqs):
            sub_seq = self.sub_seqs[i]
            if not sub_seq.totallen_after_parent:
                continue
            step_toffset = sub_seq.t_offset
            if is_nan(step_toffset):
                raise ValueError('Cannot get offset of floating sequence.')
            if sub_seq.is_step:
                tstep = step_toffset.create(sub_seq.length_sign(),
                                            mat_round(sub_seq.len))
            else:
                subt = sub_seq.wait_all_time(False)
                sub_seq.latest_seq = False
                if (not sub_seq.end_after_parent) and subt is sub_seq.cur_seq_time:
                    if setflag:
                        sub_seq.totallen_after_parent = False
                    continue
                tstep = SeqTime.combine(step_toffset, subt)
            if setflag:
                sub_seq.totallen_after_parent = False
                sub_seq.end_after_parent = False
            new_tval = sv_max(tval, tstep.get_val())
            new_t = SeqTime.zero(self).create(SeqTime.NONNEG, new_tval)
            self.root.add_order(SeqTime.NONNEG, tstep, new_t)
            self.root.add_order(SeqTime.NONNEG, t, new_t)
            tval = new_tval
            t = new_t
        return t

    def collect_end_time(self, res):
        root = self.root
        if self.end_after_parent:
            res.append(struct.pack('<I', root.get_time_id(self.cur_seq_time) & 0xFFFFFFFF))
        for i in range(self.n_sub_seqs):
            sub_seq = self.sub_seqs[i]
            if not sub_seq.totallen_after_parent:
                continue
            step_toffset = sub_seq.t_offset
            if is_nan(step_toffset):
                raise ValueError('Sub sequence still floating')
            if sub_seq.is_step:
                sub_time = step_toffset.create(sub_seq.length_sign(),
                                               mat_round(sub_seq.len))
                res.append(struct.pack('<I', root.get_time_id(sub_time) & 0xFFFFFFFF))
            else:
                res = sub_seq.collect_end_time(res)
        return res

    # -- ConditionalWrapper-access methods ----------------------------------- #
    def add_step_real(self, cond, allow_offset, curtime, first_arg, *varargin):
        cond = self.cond & cond
        if not is_numeric(first_arg) and not isinstance(first_arg, SeqVal):
            # Callback -> sub-sequence.
            return self.add_custom_step(cond, curtime, first_arg, *varargin)
        if not varargin:
            length = first_arg * self.top_level.time_scale
            step = TimeStep(self, curtime, length, cond)
            if is_nan(curtime):
                end_time = float('nan')
            else:
                end_time = curtime.create(step.length_sign(), mat_round(step.len))
            return step, end_time
        if is_numeric(varargin[0]) or isinstance(varargin[0], SeqVal):
            # Two value args -> time step with custom offset.
            if len(varargin) > 1:
                raise ValueError('Too many arguments to create a time step.')
            if not allow_offset:
                if is_nan(curtime):
                    raise ValueError('Floating time step with time offset not allowed.')
                raise ValueError('addStep with time offset not allowed.')
            offset = varargin[0]
            assert not is_nan(curtime)
            curtime = curtime.create(
                SeqTime.UNKNOWN,
                ifelse(cond, mat_round(offset * self.top_level.time_scale), 0))
            length = first_arg * self.top_level.time_scale
            step = TimeStep(self, curtime, length, cond)
            end_time = curtime.create(step.length_sign(), mat_round(step.len))
            return step, end_time
        # Number followed by a callback: sub-sequence with offset.
        if not allow_offset:
            if is_nan(curtime):
                raise ValueError('Floating time step with time offset not allowed.')
            raise ValueError('addStep with time offset not allowed.')
        if not self.latest_seq:
            self.totallen_after_parent = True
            self.latest_seq = True
            parent = self.parent
            while not parent.latest_seq:
                parent.totallen_after_parent = True
                parent.latest_seq = True
                parent = parent.parent
                if parent is None:
                    break
        assert not is_nan(curtime)
        curtime = curtime.create(
            SeqTime.UNKNOWN,
            ifelse(cond, mat_round(first_arg * self.top_level.time_scale), 0))
        return self.add_custom_step(cond, curtime, *varargin)

    def add_custom_step(self, cond, start_time, cb, *varargin):
        from sub_seq import SubSeq  # deferred import (ExpSeqBase <-> SubSeq cycle)
        step = SubSeq(self, start_time, cond)
        cb(step, *varargin)
        provenance.on_step(self, cb, start_time, step)   # INERT unless a session is active
        step.latest_seq = False
        if is_nan(start_time):
            end_time = float('nan')
        else:
            assert start_time.seq is self
            end_time = SeqTime.combine(start_time, step.cur_seq_time)
        return step, end_time

    def wait_with_condition(self, cond, t):
        if is_numeric(t):
            if t < 0:
                if (is_logical(self.cond) and self.cond
                        and is_logical(cond) and cond):
                    raise ValueError('Wait time cannot be negative.')
            elif t == 0:
                return
        if (is_logical(self.cond) and not self.cond) or (is_logical(cond) and not cond):
            return
        self.cur_seq_time = self.cur_seq_time.create(
            SeqTime.NONNEG,
            ifelse(self.cond & cond, mat_round(t * self.top_level.time_scale), 0))
        self.end_after_parent = True
        node = self
        while not node.latest_seq:
            node.totallen_after_parent = True
            node.latest_seq = True
            node = node.parent
            if node is None:
                break
