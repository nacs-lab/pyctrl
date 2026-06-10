"""seq_context.py -- value/node table + the node serializer.

Faithful transliteration of ``matlab_new/lib/SeqContext.m`` (class ``SeqContext``
kept; methods snake_case). It owns the build-time bookkeeping every SeqVal needs:
the serialized node graph, the interpolation data array, global slot types, and
the argument/measure/global node factories. It is the *value-level* half of the
serializer; the top-level + per-basic-sequence byte layout (``ExpSeq``/``RootSeq``)
lands in Phase 2.

Byte gate: the bytes produced here (``node_serialized`` / ``data_serialized`` /
``global_serialized`` and the per-arg encoding in ``serialize_arg``) must match the
blessed golden master -- a regression guard now; matched against MATLAB's
``SeqContext`` at the port. All packing is little-endian (``struct '<'``) to match
the ``typecast(...,'int8')`` layout on x86 -- the single most common serialization bug.

Naming: MATLAB has both a private property ``node_serialized`` (the storage) and a
method ``nodeSerialized()``. They collide under snake_case, so the storage is
``self._node_serialized`` (it is ``Access=private`` in MATLAB anyway) and the
method is ``node_serialized()``.
"""

import struct

import numpy as np

from seq_val import SeqVal, _arg_kind


def _b(x):
    """An int8-valued opcode/tag as a single unsigned byte."""
    return x & 0xFF


class SeqContext:
    # Node-serialization prefixes: [OP_IDENTITY, <arg tag>] (matlab_new/lib/SeqContext.m).
    _ARG_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_ARG)])
    _MEASURE_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_MEASURE)])
    _GLOBAL_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_GLOBAL)])
    _CONST_B_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_CONST_BOOL)])
    _CONST_I32_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_CONST_INT32)])
    _CONST_F64_PREFIX = bytes([_b(SeqVal.OP_IDENTITY), _b(SeqVal.ARG_CONST_FLOAT64)])

    def __init__(self, debug=0):
        self.arg_vals = []          # cached HArg SeqVals, index == arg number
        self.global_types = []      # int8 type tag per global slot (0-based id)

        self._node_serialized = []  # list of bytes, one per node (1-based ids)
        self.datas = []             # list of float lists; index == 0-based data id
        self.data_ids = {}          # packed-bytes(data) -> data id
        self.const_b_ids = {}       # 0/1 -> node id  (MATLAB: uint32([0 0]))
        self.const_f64_ids = {}     # packed-bytes(double) -> node id
        self.const_i32_ids = {}     # int32 value -> node id

        self.obj_counter = 0
        self.collect_dbg_info = bool(debug)

        # Optimized for usage in the sequence (mirror MATLAB's eager arg0/arg1).
        self.arg0 = self.get_arg(0)
        self.arg1 = self.get_arg(1)

    # --- data array (interpolation tables) ---------------------------------- #
    def get_data_id(self, data):
        data = [float(v) for v in np.atleast_1d(data)]
        key = struct.pack('<%dd' % len(data), *data)
        if key in self.data_ids:
            return self.data_ids[key]
        id_ = len(self.datas)       # 0-based, before append
        self.datas.append(data)
        self.data_ids[key] = id_
        return id_

    # --- per-argument encoding ---------------------------------------------- #
    def serialize_arg(self, res, arg):
        """Append the encoding of one node argument to ``res`` (a bytearray)."""
        kind = _arg_kind(arg)
        if kind == 'float64':
            res += bytes([_b(SeqVal.ARG_CONST_FLOAT64)])
            res += struct.pack('<d', float(arg))
        elif kind == 'node':
            head = arg.head
            if head == SeqVal.H_ARG:
                res += bytes([_b(SeqVal.ARG_ARG)])
                res += struct.pack('<I', int(arg.args[0]) & 0xFFFFFFFF)
            elif head == SeqVal.H_MEASURE:
                res += bytes([_b(SeqVal.ARG_MEASURE)])
                res += struct.pack('<I', int(arg.args[0]) & 0xFFFFFFFF)
            elif head == SeqVal.H_GLOBAL:
                res += bytes([_b(SeqVal.ARG_GLOBAL)])
                res += struct.pack('<I', int(arg.args[0]) & 0xFFFFFFFF)
            else:
                node_id = arg.node_id
                if node_id == 0:
                    self.ensure_serialize(arg)
                    node_id = arg.node_id
                res += bytes([_b(SeqVal.ARG_NODE)])
                res += struct.pack('<I', int(node_id) & 0xFFFFFFFF)
        elif kind == 'bool':
            res += bytes([_b(SeqVal.ARG_CONST_BOOL), 1 if arg else 0])
        elif kind == 'int32':
            res += bytes([_b(SeqVal.ARG_CONST_INT32)])
            res += struct.pack('<i', int(np.int32(arg)))
        else:
            raise TypeError('Argument with unknown type.')
        return res

    def ensure_serialize(self, val):
        """Assign ``val`` a node id and record its serialized bytes (assumes unset)."""
        head = val.head
        if head == SeqVal.H_ARG:
            serial = self._ARG_PREFIX + struct.pack('<I', int(val.args[0]) & 0xFFFFFFFF)
        elif head == SeqVal.H_MEASURE:
            serial = self._MEASURE_PREFIX + struct.pack('<I', int(val.args[0]) & 0xFFFFFFFF)
        elif head == SeqVal.H_GLOBAL:
            serial = self._GLOBAL_PREFIX + struct.pack('<I', int(val.args[0]) & 0xFFFFFFFF)
        elif head == SeqVal.OP_INTERP:
            serial = bytearray([_b(head)])
            for i in range(3):
                serial = self.serialize_arg(serial, val.args[i])
            serial += struct.pack('<I', int(self.get_data_id(val.args[3])) & 0xFFFFFFFF)
            serial = bytes(serial)
        else:
            serial = bytearray([_b(head)])
            for arg in val.args:
                serial = self.serialize_arg(serial, arg)
            serial = bytes(serial)
        val.node_id = len(self._node_serialized) + 1
        self._node_serialized.append(serial)

    # --- object counter + node factories ------------------------------------ #
    def next_obj_id(self):
        res = self.obj_counter
        self.obj_counter = res + 1
        # Debug backtrace capture (collect_dbg_info) is omitted: it has no effect
        # on the serialized bytes. Mirrors SeqContext.nextObjID's debug branch.
        return res

    def get_arg(self, i):
        # 0-based input.
        self._fill_args(i + 1)
        return self.arg_vals[i]

    def new_measure(self):
        id_ = self.next_obj_id()
        return SeqVal(SeqVal.H_MEASURE, [id_], self), id_

    def new_global(self, type_):
        assert type_ in (SeqVal.TYPE_BOOL, SeqVal.TYPE_INT32, SeqVal.TYPE_FLOAT64)
        id_ = len(self.global_types)   # 0-based
        res = SeqVal(SeqVal.H_GLOBAL, [id_], self)
        self.global_types.append(type_)
        return res, id_

    # --- table serializers -------------------------------------------------- #
    def node_serialized(self):
        out = bytearray(struct.pack('<I', len(self._node_serialized)))
        for s in self._node_serialized:
            out += s
        return bytes(out)

    def global_serialized(self):
        out = bytearray(struct.pack('<I', len(self.global_types)))
        out += bytes(_b(t) for t in self.global_types)
        return bytes(out)

    def data_serialized(self):
        out = bytearray(struct.pack('<I', len(self.datas)))
        for data in self.datas:
            out += struct.pack('<I', len(data))
            out += struct.pack('<%dd' % len(data), *data)
        return bytes(out)

    def get_val_id(self, val):
        """Intern ``val`` as a top-level node; return its 1-based node id."""
        kind = _arg_kind(val)
        if kind == 'float64':
            v = float(val)
            key = struct.pack('<d', v)
            vid = self.const_f64_ids.get(key)
            if vid is None:
                vid = len(self._node_serialized) + 1
                self.const_f64_ids[key] = vid
                self._node_serialized.append(self._CONST_F64_PREFIX + struct.pack('<d', v))
            return vid
        if kind == 'node':
            if val.node_id == 0:
                self.ensure_serialize(val)
            return val.node_id
        if kind == 'bool':
            b = 1 if val else 0
            vid = self.const_b_ids.get(b, 0)
            if vid != 0:
                return vid
            vid = len(self._node_serialized) + 1
            self.const_b_ids[b] = vid
            self._node_serialized.append(self._CONST_B_PREFIX + bytes([b]))
            return vid
        if kind == 'int32':
            v = int(np.int32(val))
            vid = self.const_i32_ids.get(v)
            if vid is None:
                vid = len(self._node_serialized) + 1
                self.const_i32_ids[v] = vid
                self._node_serialized.append(self._CONST_I32_PREFIX + struct.pack('<i', v))
            return vid
        raise TypeError('Value with unknown type.')

    # --- private ------------------------------------------------------------ #
    def _fill_args(self, nargs):
        old = len(self.arg_vals)
        if nargs <= old:
            return
        for i in range(old, nargs):
            self.arg_vals.append(SeqVal(SeqVal.H_ARG, [i], self))
