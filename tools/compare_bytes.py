#!/usr/bin/env python3
"""compare_bytes.py -- read and compare serialized Yb experiment sequences.

This is the runnable form of the byte-format spec documented in
``matlab_new/lib/SeqContext.m`` (and the appendix of PYTHON_FRONTEND_PLAN.md).
It decodes the byte array that MATLAB's ``ExpSeq.serialize()`` produces into a
structured form, can re-encode it (to prove the reader is faithful), and can
report the first field where two serialized sequences differ.

No MATLAB, no hardware, standard library only -- so it is safe to run at any
time, including while an experiment is in progress.

Usage:
    python compare_bytes.py FILE                 # decode + print a summary
    python compare_bytes.py FILE --tree          # decode + dump full structure
    python compare_bytes.py FILE_A FILE_B        # first differing field
    python compare_bytes.py --selftest DIR       # round-trip every seq*.json in DIR

FILE may be a raw ``.bin`` (int8/uint8 bytes) or a MATLAB ``.json`` array
(with ``//`` comments, as the lib/test reference files use).
"""

import json
import os
import re
import struct
import sys

# --- opcode -> number of serialized args : from matlab_new/lib/SeqVal.m -------
_ARITY = {
    1: 2, 2: 2, 3: 2, 4: 2,                       # add sub mul div
    5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2,          # cmplt cmpgt cmple cmpge cmpne cmpeq
    11: 2, 12: 2, 13: 2,                          # and or xor
    14: 1,                                        # not
    15: 1, 16: 1, 17: 1, 18: 1, 19: 1,            # abs ceil exp expm1 floor
    20: 1, 21: 1, 22: 1, 23: 1,                   # log log1p log2 log10
    24: 2,                                        # pow
    25: 1,                                        # sqrt
    26: 1, 27: 1, 28: 1,                          # asin acos atan
    29: 2,                                        # atan2
    30: 1, 31: 1, 32: 1, 33: 1, 34: 1,            # asinh acosh atanh sin cos
    35: 1, 36: 1, 37: 1, 38: 1,                   # tan sinh cosh tanh
    39: 2,                                        # hypot
    40: 1, 41: 1, 42: 1, 43: 1, 44: 1,            # erf erfc gamma lgamma rint
    45: 2, 46: 2, 47: 2,                          # max min mod
    48: 3,                                        # interp (+ 4-byte data_id)
    49: 3,                                        # select
    50: 1,                                        # identity
}
_OP_INTERP = 48

# --- argtype -> name : from SeqVal.m (ArgConst*/ArgNode/ArgMeasure/...) --------
_ARG_NAME = {0: "bool", 1: "int32", 2: "float64",
             3: "node", 4: "measure", 5: "global", 6: "arg"}
_ARG_CODE = {v: k for k, v in _ARG_NAME.items()}

# --- default-value Type tag -> name : from SeqVal.m (TypeBool/Int32/Float64) ---
_TYPE_NAME = {1: "bool", 2: "int32", 3: "float64"}


# --------------------------------------------------------------------------- #
# Cursor over a list of unsigned bytes.  All multibyte values are little-endian
# (MATLAB ``typecast(...,'int8')`` on x86).
# --------------------------------------------------------------------------- #
class _Cur:
    def __init__(self, data):
        self.d = bytes(data)
        self.i = 0

    def take(self, n):
        if self.i + n > len(self.d):
            raise ValueError("overrun at byte %d (+%d > %d)" % (self.i, n, len(self.d)))
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def u8(self):
        return self.take(1)[0]

    def i8(self):
        return struct.unpack("<b", self.take(1))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def f64(self):
        return struct.unpack("<d", self.take(8))[0]

    def cstr(self):
        start = self.i
        while self.i < len(self.d) and self.d[self.i] != 0:
            self.i += 1
        if self.i >= len(self.d):
            raise ValueError("unterminated string from byte %d" % start)
        s = self.d[start:self.i].decode("latin1")
        self.i += 1  # skip the NUL
        return s


# --------------------------------------------------------------------------- #
# Encoder helpers (mirror the cursor so encode(decode(x)) == x).
# --------------------------------------------------------------------------- #
def _u8(out, v): out.append(v & 0xFF)
def _i8(out, v): out += struct.pack("<b", v)
def _u32(out, v): out += struct.pack("<I", v & 0xFFFFFFFF)
def _i32(out, v): out += struct.pack("<i", v)
def _f64(out, v): out += struct.pack("<d", v)
def _cstr(out, s): out += s.encode("latin1"); out.append(0)


# --------------------------------------------------------------------------- #
# Node (SeqVal) decode / encode.
# --------------------------------------------------------------------------- #
def _decode_arg(c):
    t = c.u8()
    if t not in _ARG_NAME:
        raise ValueError("bad argtype %d at byte %d" % (t, c.i - 1))
    name = _ARG_NAME[t]
    if name == "bool":
        val = c.u8()
    elif name == "int32":
        val = c.i32()
    elif name == "float64":
        val = c.f64()
    else:  # node / measure / global / arg are all uint32 ids
        val = c.u32()
    return {"argtype": name, "val": val}


def _encode_arg(out, a):
    name = a["argtype"]
    _u8(out, _ARG_CODE[name])
    if name == "bool":
        _u8(out, a["val"])
    elif name == "int32":
        _i32(out, a["val"])
    elif name == "float64":
        _f64(out, a["val"])
    else:
        _u32(out, a["val"])


def _decode_node(c):
    op = c.u8()
    if op not in _ARITY:
        raise ValueError("bad opcode %d at byte %d" % (op, c.i - 1))
    node = {"op": op, "args": [_decode_arg(c) for _ in range(_ARITY[op])]}
    if op == _OP_INTERP:
        node["data_id"] = c.u32()
    return node


def _encode_node(out, node):
    _u8(out, node["op"])
    for a in node["args"]:
        _encode_arg(out, a)
    if node["op"] == _OP_INTERP:
        _u32(out, node["data_id"])


# --------------------------------------------------------------------------- #
# Basic sequence decode / encode.
# --------------------------------------------------------------------------- #
def _decode_bseq(c):
    b = {}
    b["times"] = [{"sign": c.i8(), "id": c.u32(), "delta_node": c.u32(), "prev_id": c.u32()}
                  for _ in range(c.u32())]
    b["endtimes"] = [c.u32() for _ in range(c.u32())]
    b["timeorders"] = [{"sign": c.i8(), "id": c.u32(), "before_id": c.u32(), "after_id": c.u32()}
                       for _ in range(c.u32())]
    b["outputs"] = [{"id": c.u32(), "time_id": c.u32(), "len": c.u32(),
                     "val": c.u32(), "cond": c.u32(), "chn": c.u32()}
                    for _ in range(c.u32())]
    b["measures"] = [{"id": c.u32(), "time_id": c.u32(), "chn": c.u32()}
                     for _ in range(c.u32())]
    b["assigns"] = [{"assign_id": c.u32(), "global_id": c.u32(), "val": c.u32()}
                    for _ in range(c.u32())]
    b["branches"] = [{"branch_id": c.u32(), "target_id": c.u32(), "cond": c.u32()}
                     for _ in range(c.u32())]
    b["default_target"] = c.u32()
    return b


def _encode_bseq(out, b):
    _u32(out, len(b["times"]))
    for t in b["times"]:
        _i8(out, t["sign"]); _u32(out, t["id"]); _u32(out, t["delta_node"]); _u32(out, t["prev_id"])
    _u32(out, len(b["endtimes"]))
    for e in b["endtimes"]:
        _u32(out, e)
    _u32(out, len(b["timeorders"]))
    for o in b["timeorders"]:
        _i8(out, o["sign"]); _u32(out, o["id"]); _u32(out, o["before_id"]); _u32(out, o["after_id"])
    _u32(out, len(b["outputs"]))
    for p in b["outputs"]:
        for k in ("id", "time_id", "len", "val", "cond", "chn"):
            _u32(out, p[k])
    _u32(out, len(b["measures"]))
    for m in b["measures"]:
        _u32(out, m["id"]); _u32(out, m["time_id"]); _u32(out, m["chn"])
    _u32(out, len(b["assigns"]))
    for a in b["assigns"]:
        _u32(out, a["assign_id"]); _u32(out, a["global_id"]); _u32(out, a["val"])
    _u32(out, len(b["branches"]))
    for br in b["branches"]:
        _u32(out, br["branch_id"]); _u32(out, br["target_id"]); _u32(out, br["cond"])
    _u32(out, b["default_target"])


# --------------------------------------------------------------------------- #
# Top-level sequence decode / encode.
# --------------------------------------------------------------------------- #
def decode(data):
    """Decode serialized sequence bytes into a nested structure.

    Raises ``ValueError`` on any malformed field or trailing bytes.
    """
    c = _Cur(data)
    seq = {"version": c.u8()}
    seq["nodes"] = [_decode_node(c) for _ in range(c.u32())]
    seq["channels"] = [c.cstr() for _ in range(c.u32())]

    defvals = []
    for _ in range(c.u32()):
        chnid = c.u32()
        t = c.u8()
        if t == 1:
            v = c.u8()
        elif t == 2:
            v = c.i32()
        elif t == 3:
            v = c.f64()
        else:
            raise ValueError("bad defval type %d at byte %d" % (t, c.i - 1))
        defvals.append({"chnid": chnid, "type": t, "val": v})
    seq["defvals"] = defvals

    seq["slots"] = [c.u8() for _ in range(c.u32())]          # global slot types
    seq["noramp"] = [c.u32() for _ in range(c.u32())]
    seq["basicseqs"] = [_decode_bseq(c) for _ in range(c.u32())]
    seq["datas"] = [[c.f64() for _ in range(c.u32())] for _ in range(c.u32())]

    backend = []
    for _ in range(c.u32()):
        name = c.cstr()
        size = c.u32()
        backend.append({"device": name, "size": size, "data": list(c.take(size))})
    seq["backenddatas"] = backend

    if c.i != len(c.d):
        raise ValueError("trailing %d bytes after decode" % (len(c.d) - c.i))
    return seq


def encode(seq):
    """Re-encode a decoded structure back to bytes. Inverse of ``decode``."""
    out = bytearray()
    _u8(out, seq["version"])
    _u32(out, len(seq["nodes"]))
    for n in seq["nodes"]:
        _encode_node(out, n)
    _u32(out, len(seq["channels"]))
    for s in seq["channels"]:
        _cstr(out, s)
    _u32(out, len(seq["defvals"]))
    for d in seq["defvals"]:
        _u32(out, d["chnid"]); _u8(out, d["type"])
        if d["type"] == 1:
            _u8(out, d["val"])
        elif d["type"] == 2:
            _i32(out, d["val"])
        else:
            _f64(out, d["val"])
    _u32(out, len(seq["slots"]))
    for t in seq["slots"]:
        _u8(out, t)
    _u32(out, len(seq["noramp"]))
    for r in seq["noramp"]:
        _u32(out, r)
    _u32(out, len(seq["basicseqs"]))
    for b in seq["basicseqs"]:
        _encode_bseq(out, b)
    _u32(out, len(seq["datas"]))
    for arr in seq["datas"]:
        _u32(out, len(arr))
        for v in arr:
            _f64(out, v)
    _u32(out, len(seq["backenddatas"]))
    for be in seq["backenddatas"]:
        _cstr(out, be["device"]); _u32(out, be["size"])
        out += bytes(be["data"])
    return bytes(out)


# --------------------------------------------------------------------------- #
# Loading + comparison.
# --------------------------------------------------------------------------- #
def load(path):
    """Load bytes from a raw ``.bin`` or a MATLAB ``.json`` int8 array."""
    if path.endswith(".json"):
        text = open(path, "r").read()
        text = re.sub(r"//[^\n]*", "", text)     # strip // comments (as MATLAB does)
        text = re.sub(r",\s*]", "]", text)        # tolerate a trailing comma
        arr = json.loads(text)
        return bytes((x + 256) & 0xFF if x < 0 else x for x in arr)  # int8 -> ubyte
    with open(path, "rb") as f:
        return f.read()


def diff(a, b, path="seq"):
    """Return the first differing field between two decoded structures, or None."""
    if type(a) is not type(b):
        return "%s: type %s != %s" % (path, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        for k in a:
            if k not in b:
                return "%s.%s: missing on right" % (path, k)
            d = diff(a[k], b[k], "%s.%s" % (path, k))
            if d:
                return d
        for k in b:
            if k not in a:
                return "%s.%s: missing on left" % (path, k)
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: length %d != %d" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    if a != b:
        return "%s: %r != %r" % (path, a, b)
    return None


def summary(seq):
    n = seq["basicseqs"]
    return ("version=%d  nodes=%d  channels=%d  defvals=%d  slots=%d  "
            "basicseqs=%d  datas=%d  backenddatas=%d\n"
            "  bseq[0]: times=%d outputs=%d measures=%d assigns=%d branches=%d"
            % (seq["version"], len(seq["nodes"]), len(seq["channels"]),
               len(seq["defvals"]), len(seq["slots"]), len(n), len(seq["datas"]),
               len(seq["backenddatas"]),
               len(n[0]["times"]) if n else 0,
               len(n[0]["outputs"]) if n else 0,
               len(n[0]["measures"]) if n else 0,
               len(n[0]["assigns"]) if n else 0,
               len(n[0]["branches"]) if n else 0))


def _selftest(directory):
    files = sorted(f for f in os.listdir(directory)
                   if f.startswith("seq") and f.endswith(".json"))
    if not files:
        print("no seq*.json reference files in %s" % directory)
        return 1
    ok = 0
    for f in files:
        path = os.path.join(directory, f)
        try:
            data = load(path)
            seq = decode(data)
            again = encode(seq)
            if again != data:
                print("FAIL  %-22s round-trip mismatch (%d vs %d bytes)"
                      % (f, len(again), len(data)))
                continue
            print("ok    %-22s %5d bytes  | %s" % (f, len(data), summary(seq).split("\n")[0]))
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print("FAIL  %-22s %s" % (f, e))
    print("\n%d/%d reference files decoded and round-tripped exactly" % (ok, len(files)))
    return 0 if ok == len(files) else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--selftest":
        return _selftest(argv[1] if len(argv) > 1 else ".")
    if len(argv) == 1 or (len(argv) == 2 and argv[1] == "--tree"):
        seq = decode(load(argv[0]))
        if len(argv) == 2:
            print(json.dumps(seq, indent=2))
        else:
            print(summary(seq))
        return 0
    # two files: compare
    a = decode(load(argv[0]))
    b = decode(load(argv[1]))
    d = diff(a, b)
    if d is None:
        print("identical")
        return 0
    print("first difference: %s" % d)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
