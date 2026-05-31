#!/usr/bin/env python3
"""compare_seq_bytes.py -- read and compare the *flattened* ``.seq`` files that
feed SeqPlotter (https://github.com/nacs-lab/SeqPlotter).

This is the sibling of ``compare_bytes.py``. Where ``compare_bytes.py`` decodes
the *symbolic* byte blob ``ExpSeq.serialize()`` hands to the libnacs engine, this
tool decodes the *evaluated* byte blob ``ExpSeq.dump_output_to_file()`` writes for
plotting -- a per-channel list of ``(time, value, pulse_id)`` points produced by
the engine's ``get_nominal_output(pts_per_ramp)`` and packed with channel
names + an optional parameters-JSON block + an optional debug backtrace.

It is the runnable form of the byte-format spec documented in the MATLAB source:
  * top level + params block : matlab_new/lib/ExpSeq.m:628 (dump_output_to_file)
  * channel block            : matlab_new/lib/ExpSeq.m:686 (get_nominal_output)
  * backtrace block          : matlab_new/lib/ExpSeq.m:735 (get_debug_output)

No MATLAB, no hardware, standard library only -- safe to run at any time. Used to
(a) validate the reader against a known-good ``.seq`` by round-trip, and (b) report
the first field where a pyctrl-produced ``.seq`` differs from a MATLAB one.

Usage:
    python compare_seq_bytes.py FILE                 # decode + print a summary
    python compare_seq_bytes.py FILE --tree          # decode + dump full structure
    python compare_seq_bytes.py FILE_A FILE_B        # first differing field
    python compare_seq_bytes.py FILE_A FILE_B --no-name  # ignore seq_name (timestamp)
    python compare_seq_bytes.py --selftest DIR       # round-trip every *.seq in DIR

All multibyte values are little-endian (MATLAB ``typecast(...,'int8')`` on x86).
Per point the layout is ``[time int64 8B][value float64 8B][pulse_id uint32 4B]``
== 20 bytes, matching SeqPlotter's ``struct.unpack('<qdI', ...)``.
"""

import json
import os
import struct
import sys


# --------------------------------------------------------------------------- #
# Cursor over a byte buffer.
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

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i64(self):
        return struct.unpack("<q", self.take(8))[0]

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


def _u8(out, v): out.append(v & 0xFF)
def _u32(out, v): out += struct.pack("<I", v & 0xFFFFFFFF)
def _i64(out, v): out += struct.pack("<q", v)
def _f64(out, v): out += struct.pack("<d", v)
def _cstr(out, s): out += s.encode("latin1"); out.append(0)


# --------------------------------------------------------------------------- #
# Channel block  (matlab_new/lib/ExpSeq.m:686, get_nominal_output)
#   [nchns: 4B][ chn_name\0, npts: 4B, [time i64, value f64, pulse_id u32] x npts ]
# --------------------------------------------------------------------------- #
def _decode_channels(c):
    chns = []
    for _ in range(c.u32()):
        name = c.cstr()
        npts = c.u32()
        pts = [{"t": c.i64(), "v": c.f64(), "pid": c.u32()} for _ in range(npts)]
        chns.append({"name": name, "points": pts})
    return chns


def _encode_channels(out, chns):
    _u32(out, len(chns))
    for ch in chns:
        _cstr(out, ch["name"])
        _u32(out, len(ch["points"]))
        for p in ch["points"]:
            _i64(out, p["t"]); _f64(out, p["v"]); _u32(out, p["pid"])


# --------------------------------------------------------------------------- #
# Backtrace block  (matlab_new/lib/ExpSeq.m:735, get_debug_output)
#   [nfilenames:4B][ name\0 x nfilenames]
#   [nnames:4B][ name\0 x nnames]
#   [nobjs:4B][ [nframes:4B][ fname_id:4B, name_id:4B, line:4B ] x nframes ] x nobjs
# NOTE: the comment at ExpSeq.m:663 mentions a trailing [has_params] after the
# objs, but the *code* (get_debug_output) does not emit it -- we follow the code.
# --------------------------------------------------------------------------- #
def _decode_backtrace_payload(c):
    filenames = [c.cstr() for _ in range(c.u32())]
    names = [c.cstr() for _ in range(c.u32())]
    objs = []
    for _ in range(c.u32()):
        frames = [{"fname_id": c.u32(), "name_id": c.u32(), "line": c.u32()}
                  for _ in range(c.u32())]
        objs.append(frames)
    return {"filenames": filenames, "names": names, "objs": objs}


def _encode_backtrace_payload(out, bt):
    _u32(out, len(bt["filenames"]))
    for s in bt["filenames"]:
        _cstr(out, s)
    _u32(out, len(bt["names"]))
    for s in bt["names"]:
        _cstr(out, s)
    _u32(out, len(bt["objs"]))
    for frames in bt["objs"]:
        _u32(out, len(frames))
        for f in frames:
            _u32(out, f["fname_id"]); _u32(out, f["name_id"]); _u32(out, f["line"])


# --------------------------------------------------------------------------- #
# Top level  (matlab_new/lib/ExpSeq.m:628, dump_output_to_file)
#   [nseqs:4B]
#   per seq: [seq_name\0][seq_idx:4B][channel block][has_params:1B][params\0?]
#   [has_bt_info:1B]
#   if set: [bt_idx:4B x nseqs][n_bts:4B][backtrace payload x n_bts]
# --------------------------------------------------------------------------- #
def decode(data):
    """Decode ``.seq`` bytes into a nested structure. Raises on malformed input.

    The parameters block is kept BOTH as the exact raw string (``params_raw``,
    used by ``encode`` for an exact round-trip) and parsed (``params``, for
    inspection / diffing), since ``json.dumps`` would not reproduce MATLAB's
    ``jsonencode`` formatting byte-for-byte.
    """
    c = _Cur(data)
    seqs = []
    for _ in range(c.u32()):
        s = {"seq_name": c.cstr(), "seq_idx": c.u32(), "channels": _decode_channels(c)}
        s["has_params"] = c.u8()
        if s["has_params"]:
            raw = c.cstr()
            s["params_raw"] = raw
            try:
                s["params"] = json.loads(raw) if raw else None
            except ValueError:
                s["params"] = None  # keep raw; not all blobs are strict JSON
        seqs.append(s)
    seq = {"seqs": seqs}

    seq["has_bt_info"] = c.u8()
    if seq["has_bt_info"]:
        seq["bt_idx"] = [c.u32() for _ in range(len(seqs))]
        seq["backtraces"] = [_decode_backtrace_payload(c) for _ in range(c.u32())]

    if c.i != len(c.d):
        raise ValueError("trailing %d bytes after decode" % (len(c.d) - c.i))
    return seq


def encode(seq):
    """Re-encode a decoded structure back to bytes. Inverse of ``decode``."""
    out = bytearray()
    _u32(out, len(seq["seqs"]))
    for s in seq["seqs"]:
        _cstr(out, s["seq_name"])
        _u32(out, s["seq_idx"])
        _encode_channels(out, s["channels"])
        _u8(out, s["has_params"])
        if s["has_params"]:
            _cstr(out, s.get("params_raw", ""))
    _u8(out, seq["has_bt_info"])
    if seq["has_bt_info"]:
        for x in seq["bt_idx"]:
            _u32(out, x)
        _u32(out, len(seq["backtraces"]))
        for bt in seq["backtraces"]:
            _encode_backtrace_payload(out, bt)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Loading + comparison.
# --------------------------------------------------------------------------- #
def load(path):
    with open(path, "rb") as f:
        return f.read()


def diff(a, b, path="seq", ignore=()):
    """Return the first differing field between two decoded structures, or None.

    ``ignore`` is a set of leaf key names to skip (e.g. ``{"seq_name"}`` to
    compare modulo the nondeterministic timestamp embedded by MATLAB).
    """
    if type(a) is not type(b):
        return "%s: type %s != %s" % (path, type(a).__name__, type(b).__name__)
    if isinstance(a, dict):
        for k in a:
            if k in ignore:
                continue
            if k not in b:
                return "%s.%s: missing on right" % (path, k)
            d = diff(a[k], b[k], "%s.%s" % (path, k), ignore)
            if d:
                return d
        for k in b:
            if k not in a and k not in ignore:
                return "%s.%s: missing on left" % (path, k)
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: length %d != %d" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, "%s[%d]" % (path, i), ignore)
            if d:
                return d
        return None
    if a != b:
        return "%s: %r != %r" % (path, a, b)
    return None


def summary(seq):
    lines = ["nseqs=%d  has_bt_info=%d%s"
             % (len(seq["seqs"]), seq["has_bt_info"],
                ("  backtraces=%d" % len(seq["backtraces"])) if seq["has_bt_info"] else "")]
    for i, s in enumerate(seq["seqs"]):
        npts = sum(len(ch["points"]) for ch in s["channels"])
        nparam = (len(s["params"]) if isinstance(s.get("params"), dict) else
                  ("raw" if s["has_params"] else 0))
        lines.append("  seq[%d] idx=%d name=%r channels=%d points=%d params=%s"
                     % (i, s["seq_idx"], s["seq_name"], len(s["channels"]), npts, nparam))
    return "\n".join(lines)


def _selftest(directory):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".seq"))
    if not files:
        print("no *.seq files in %s" % directory)
        return 1
    ok = 0
    for f in files:
        path = os.path.join(directory, f)
        try:
            data = load(path)
            seq = decode(data)
            again = encode(seq)
            if again != data:
                print("FAIL  %-34s round-trip mismatch (%d vs %d bytes)"
                      % (f, len(again), len(data)))
                continue
            print("ok    %-34s %7d bytes  | %s" % (f, len(data), summary(seq).split("\n")[0]))
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue
            print("FAIL  %-34s %s" % (f, e))
    print("\n%d/%d .seq files decoded and round-tripped exactly" % (ok, len(files)))
    return 0 if ok == len(files) else 1


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--selftest":
        return _selftest(argv[1] if len(argv) > 1 else ".")
    if len(argv) == 1 or (len(argv) == 2 and argv[1] == "--tree"):
        seq = decode(load(argv[0]))
        print(json.dumps(seq, indent=2) if len(argv) == 2 else summary(seq))
        return 0
    ignore = {"seq_name"} if "--no-name" in argv else ()
    files = [a for a in argv if not a.startswith("--")]
    a = decode(load(files[0]))
    b = decode(load(files[1]))
    d = diff(a, b, ignore=ignore)
    if d is None:
        print("identical" + (" (ignoring seq_name)" if ignore else ""))
        return 0
    print("first difference: %s" % d)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
