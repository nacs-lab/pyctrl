"""time_step.py -- transliteration of ``matlab_new/lib/TimeStep.m``.

Leaf node of the sequence DAG. Holds ``pulses`` (one per channel id), a ``len``
(condition-gated) and ``raw_len`` (un-gated). ``add``/``add_conditional`` capture a
pulse value: a number/bool -> float64, a ``SeqVal`` passthrough, or a callable
inlined immediately by declared arity (1/2/3 args -> ``t``, ``(t,len)``,
``(t,len,old_val)``) with ``t = arg0/time_scale``, ``len = raw_len/time_scale``,
``old_val = arg1``. The per-pulse ``id`` is drawn from the shared ``obj_counter``
AFTER the value is built and AFTER the disabled-cond early return -- this ordering
is byte-load-bearing.
"""

import inspect

from ifelse import ifelse
from ir_pulse import IRPulse
from mat_utils import is_logical, is_numeric
from pulse import Pulse
from seq_time import SeqTime
from seq_val import SeqVal
from seq_val import to_string as sv_to_string
from time_seq import TimeSeq


def _nargin(f):
    """Declared positional arity of a callable (MATLAB ``nargin(fh)``)."""
    try:
        sig = inspect.signature(f)
        n = 0
        for p in sig.parameters.values():
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                n += 1
        return n
    except (TypeError, ValueError):
        return f.__code__.co_argcount


class TimeStep(TimeSeq):
    def __init__(self, parent, start_time, length, cond):
        # `length` is in sequence time unit (scaled from user input).
        self.is_step = True
        self.parent = parent
        self.t_offset = start_time
        self.config = parent.config
        self.top_level = parent.top_level
        self.root = parent.root
        self.raw_len = length
        self.cond = cond
        self.len = ifelse(self.cond, length, 0)
        self.pulses = {}  # cid -> Pulse (MATLAB: cell indexed by cid)
        parent.n_sub_seqs += 1
        parent.sub_seqs.append(self)
        while not parent.latest_seq:
            parent.totallen_after_parent = True
            parent.latest_seq = True
            parent = parent.parent
            if parent is None:
                break

    def add(self, cid, pulse):
        toplevel = self.top_level
        if not is_numeric(cid):
            # Translate (and so register) the channel before any cond check, so a
            # disabled pulse still marks the channel used/initialized.
            cid = toplevel.translate_channel(cid)
        cond = self.cond
        if is_logical(cond) and not cond:
            return self
        ctx = toplevel.seq_ctx
        pulse = self._resolve_pulse(ctx, toplevel, pulse)
        id_ = ctx.next_obj_id()
        self.pulses[cid] = Pulse(id_, pulse, cond)
        return self

    def add_conditional(self, cid, pulse, cond):
        if is_numeric(cond):
            cond = cond != 0
        toplevel = self.top_level
        if not is_numeric(cid):
            cid = toplevel.translate_channel(cid)
        cond = self.cond & cond
        if is_logical(cond) and not cond:
            return self
        ctx = toplevel.seq_ctx
        pulse = self._resolve_pulse(ctx, toplevel, pulse)
        id_ = ctx.next_obj_id()
        self.pulses[cid] = Pulse(id_, pulse, cond)
        return self

    def _resolve_pulse(self, ctx, toplevel, pulse):
        if is_numeric(pulse) or is_logical(pulse):
            return float(pulse)
        if isinstance(pulse, SeqVal):
            return pulse
        if isinstance(pulse, IRPulse):
            return pulse.calc_value(ctx.arg0 / toplevel.time_scale,
                                    self.raw_len / toplevel.time_scale, ctx.arg1)
        narg = _nargin(pulse)
        if narg == 1:
            return pulse(ctx.arg0 / toplevel.time_scale)
        if narg == 2:
            return pulse(ctx.arg0 / toplevel.time_scale,
                         self.raw_len / toplevel.time_scale)
        return pulse(ctx.arg0 / toplevel.time_scale,
                     self.raw_len / toplevel.time_scale, ctx.arg1)

    def total_time(self):
        return self.len / self.top_level.time_scale

    def length_sign(self):
        if is_logical(self.cond) and self.cond:
            return SeqTime.POS
        return SeqTime.NONNEG

    def to_string(self, indent=0):
        prefix = ' ' * indent
        prefix2 = ' ' * (indent + 2)
        if is_logical(self.cond) and self.cond:
            res = '%sStep(len=%s)' % (prefix, sv_to_string(self.raw_len))
        else:
            res = '%sStep(len=%s, cond=%s)' % (
                prefix, sv_to_string(self.raw_len), sv_to_string(self.cond))
        res = res + ' @ ' + self.t_offset.to_string()
        for cid in sorted(self.pulses.keys()):
            pulse = self.pulses[cid]
            res = res + '\n' + prefix2 + ('chn%d(%s): ' % (
                cid, self.top_level.channel_name(cid))) + pulse.to_string()
        return res
