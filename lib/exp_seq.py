"""exp_seq.py -- transliteration of ``matlab_new/lib/ExpSeq.m``.

The top-level sequence (basic-sequence #1) + channel management + the master
``serialize()`` byte assembler.

``serialize()`` layout (ExpSeq.m:220-317), all little-endian:
    [version=0:1B][nodes][channels][defvals][slots][noramp=0:4B][basicseqs][datas][backenddata]
CRITICAL ordering: the channel pass builds ``cid_map`` (consumed by every bseq +
backend block), and the basic-sequence walk *interns* nodes/datas as a side effect,
so the node/data tables MUST be read AFTER the bseqs are serialized even though they
are placed earlier in the byte stream. ``noramp`` is always 4 zero bytes.
"""

import struct

import numpy as np

import seq_manager
from dyn_props import DynProps
from mat_utils import is_logical, is_numeric, mat_round
from root_seq import RootSeq
from seq_config import SeqConfig
from seq_context import SeqContext
from seq_time import SeqTime
from seq_val import SeqVal


class ExpSeq(RootSeq):
    def __init__(self, c_ovr=None, c_def=None):
        self.config = SeqConfig.get(1)
        self.top_level = self
        self.root = self
        C = {}
        consts = self.config.consts
        for fn in consts:
            C[fn] = consts[fn]
        if c_ovr is not None:
            if not isinstance(c_ovr, dict):
                raise TypeError('Constant input must be a struct.')
            for fn in c_ovr:
                C[fn] = c_ovr[fn]
        self.time_scale = seq_manager.tick_per_sec()
        self.C = DynProps(C)
        # c_def debug path skipped (empty config -> debug=0).
        self.G = self.config.G

        # Channel management.
        self.orig_channel_names = []     # 0-based; cid = index + 1
        self.channel_names = []          # translated names; cid = index + 1
        self.cid_cache = {}              # name (orig + translated) -> cid
        self.disabled_channels = {}
        self.cid_map = {}                # orig cid -> compacted cid (0 = disabled)
        self.inverse_chn_map = {}

        # Output / defaults / backend.
        self.default_override = {}       # cid -> True
        self.default_override_val = {}
        self.ttl_managers = []
        self.trigger_device = ''
        self.trigger_config = {'channel': 0, 'raise': 0, 'timeout': 0}

        self.seq_ctx = SeqContext(self.C.debug(0))
        self.basic_seqs = []
        self.globals = []
        self.before_start_cbs = []
        self.after_end_cbs = []

        self._init_root()
        self.bseq_id = 1
        self.zero_time = SeqTime.zero(self)
        self.cur_seq_time = self.zero_time

    # -- top-level builders -------------------------------------------------- #
    def new_basic_seq(self, cb=None, *args):
        from basic_seq import BasicSeq
        bseq = BasicSeq(self)
        if cb is not None:
            cb(bseq, *args)
        return bseq

    def add_ttl_mgr(self, chn, off_delay, on_delay, skip_time, min_time, off_val=False):
        if off_delay < 0 or on_delay < 0 or skip_time < 0 or min_time < 0:
            raise ValueError('TTL Manager input times must be positive or zero')
        # Translate but do NOT assign a cid (specifying props must not mark it used).
        chn = self.config.translate_channel(chn)
        self.ttl_managers.append({'chn': chn, 'off_delay': off_delay,
                                  'on_delay': on_delay, 'skip_time': skip_time,
                                  'min_time': min_time, 'off_val': off_val})

    def enable_global_wait_trigger(self, devname, channel, raise_, timeout):
        if self.trigger_device != '':
            raise ValueError('Wait trigger already enabled')
        self.trigger_device = devname
        self.trigger_config = {'channel': channel, 'raise': raise_, 'timeout': timeout}

    def translate_channel(self, name):
        return self.get_channel_id(name)

    def set_default(self, name, val):
        cid = name if is_numeric(name) else self.translate_channel(name)
        self.default_override[cid] = True
        self.default_override_val[cid] = val
        return self

    def channel_name(self, cid):
        return self.channel_names[cid - 1]

    def get_default(self, cid):
        if self.default_override.get(cid):
            return self.default_override_val[cid]
        name = self.channel_name(cid)
        if name in self.config.default_vals:
            return self.config.default_vals[name]
        return 0

    def disable_channel(self, name):
        name = self.config.translate_channel(name)
        if not self.disabled_channels and not self.G.localDisableWarned(False):
            self.G.localDisableWarned = True  # warning omitted (no byte effect)
        self.disabled_channels[name] = 0

    def check_channel_disabled(self, name):
        for key in self.disabled_channels:
            if name == key or name.startswith(key + '/'):
                return True
        return self.config.check_channel_disabled(name)

    def reg_before_start(self, cb):
        self.before_start_cbs.append(cb)
        return self

    def reg_after_end(self, cb):
        self.after_end_cbs.append(cb)
        return self

    # -- globals ------------------------------------------------------------- #
    def new_global_real(self, persist, type_=None, init_val=0):
        if type_ is None:
            type_ = SeqVal.TYPE_FLOAT64
        g, id_ = self.seq_ctx.new_global(type_)
        self.globals.append({'id': id_, 'persist': persist,
                             'init_val': float(init_val)})
        return g

    # -- serialize ----------------------------------------------------------- #
    def serialize(self):
        nchns = len(self.channel_names)

        # [nchns: 4B][chnname\0 ...] -- count is the COMPACTED count (cid-1).
        chn_serialized = bytearray()
        cid_map = {}
        cid = 1
        for i in range(1, nchns + 1):
            chnname = self.channel_names[i - 1]
            if self.check_channel_disabled(chnname):
                cid_map[i] = 0
                continue
            cid_map[i] = cid
            cid += 1
            assert chnname != ''
            chn_serialized += chnname.encode('latin-1') + b'\x00'
        chn_serialized = struct.pack('<I', cid - 1) + bytes(chn_serialized)
        self.cid_map = cid_map

        # [ndefvals: 4B][[chnid: 4B][type: 1B][value] ...] -- skip 0 / disabled.
        ndefvals = 0
        defval_serialized = bytearray()
        for i in range(1, nchns + 1):
            cid = cid_map.get(i, 0)
            if cid == 0:
                continue
            defval = self.get_default(i)
            if defval == 0:
                continue
            if is_logical(defval):
                assert defval
                defval_serialized += (struct.pack('<I', cid)
                                      + bytes([SeqVal.TYPE_BOOL]) + bytes([1]))
            elif isinstance(defval, np.integer):
                defval_serialized += (struct.pack('<I', cid)
                                      + bytes([SeqVal.TYPE_INT32])
                                      + struct.pack('<i', int(np.int32(defval))))
            else:
                defval_serialized += (struct.pack('<I', cid)
                                      + bytes([SeqVal.TYPE_FLOAT64])
                                      + struct.pack('<d', float(defval)))
            ndefvals += 1
        defval_serialized = struct.pack('<I', ndefvals) + bytes(defval_serialized)

        # [nbasicseqs: 4B][bseq ...] -- ExpSeq is #1, then basic_seqs. Interns nodes.
        bseqs = [self.serialize_bseq()]
        for bs in self.basic_seqs:
            bseqs.append(bs.serialize_bseq())
        bseqs_serialized = struct.pack('<I', len(bseqs)) + b''.join(bseqs)

        # [nbackenddatas: 4B][...]  (reads cid_map -> built last)
        backenddata_serialized = self.serialize_backend_data()

        seq_ctx = self.seq_ctx
        res = bytearray()
        res += bytes([0])                       # version
        res += seq_ctx.node_serialized()        # read AFTER bseqs (interned)
        res += chn_serialized
        res += defval_serialized
        res += seq_ctx.global_serialized()
        res += bytes([0, 0, 0, 0])              # noramp count = 0
        res += bseqs_serialized
        res += seq_ctx.data_serialized()
        res += backenddata_serialized
        return bytes(res)

    # -- backend data (ZYNQZYNQ blocks) -------------------------------------- #
    def serialize_trigger_data(self):
        trig_type = 2 if self.trigger_config['raise'] else 1
        trig_timeout_ns = int(mat_round(self.trigger_config['timeout'] * 1e9))
        return (bytes([trig_type & 0xFF])
                + bytes([int(self.trigger_config['channel']) & 0xFF])
                + struct.pack('<q', trig_timeout_ns))

    def collect_backend_data(self):
        res = []
        device_ttl_managers = {}
        cid_map = self.cid_map
        for ttl_manager in self.ttl_managers:
            chnname = ttl_manager['chn']
            devname, sep, subname = chnname.partition('/')
            if devname == '' or sep == '':
                raise ValueError('Invalid channel name "%s"' % chnname)
            if chnname not in self.cid_cache:
                continue
            cid = cid_map.get(self.cid_cache[chnname], 0)
            rec = (struct.pack('<I', cid & 0xFFFFFFFF)
                   + struct.pack('<q', int(mat_round(ttl_manager['off_delay'] * self.time_scale)))
                   + struct.pack('<q', int(mat_round(ttl_manager['on_delay'] * self.time_scale)))
                   + struct.pack('<q', int(mat_round(ttl_manager['skip_time'] * self.time_scale)))
                   + struct.pack('<q', int(mat_round(ttl_manager['min_time'] * self.time_scale)))
                   + bytes([1 if ttl_manager['off_val'] != 0 else 0]))
            device_ttl_managers.setdefault(devname, []).append(rec)
        found_trigger = False
        for devname in sorted(device_ttl_managers.keys()):  # containers.Map keys are sorted
            ttl_mgr = device_ttl_managers[devname]
            if devname == self.trigger_device:
                ver = 2
                trig_serialized = self.serialize_trigger_data()
                found_trigger = True
            else:
                ver = 1
                trig_serialized = b''
            dev_serialized = (b'ZYNQZYNQ' + bytes([ver & 0xFF])
                              + bytes([len(ttl_mgr) & 0xFF]) + b''.join(ttl_mgr)
                              + trig_serialized)
            res.append(devname.encode('latin-1') + b'\x00'
                       + struct.pack('<i', len(dev_serialized)) + dev_serialized)
        if not found_trigger and self.trigger_device != '':
            dev_serialized = (b'ZYNQZYNQ' + bytes([2]) + bytes([0])
                              + self.serialize_trigger_data())
            res.append(self.trigger_device.encode('latin-1') + b'\x00'
                       + struct.pack('<i', len(dev_serialized)) + dev_serialized)
        return res

    def serialize_backend_data(self):
        datas = self.collect_backend_data()
        res = bytearray(struct.pack('<i', len(datas)))
        for d in datas:
            res += d
        return bytes(res)

    def to_string(self, indent=0):
        res = RootSeq.to_string(self, indent)
        for bs in self.basic_seqs:
            res += '\n\n' + bs.to_string(indent)
        return res

    # -- private channel factory --------------------------------------------- #
    def get_channel_id(self, name):
        if name in self.cid_cache:
            return self.cid_cache[name]
        orig_name = name
        name = self.config.translate_channel(name)
        if name not in self.inverse_chn_map:
            self.inverse_chn_map[name] = [orig_name]
        elif orig_name not in self.inverse_chn_map[name]:
            self.inverse_chn_map[name].append(orig_name)
        if name in self.cid_cache:
            return self.cid_cache[name]
        cid = len(self.channel_names) + 1
        self.channel_names.append(name)
        self.cid_cache[name] = cid
        if name != orig_name:
            self.cid_cache[orig_name] = cid
        if cid > len(self.orig_channel_names):
            self.orig_channel_names.append(orig_name)
        return cid
