"""dump_output.py -- emit the flattened ``.seq`` file SeqPlotter consumes.

Port of ``ExpSeq.dump_output_to_file`` (matlab_new/lib/ExpSeq.m:585) and the
channel packing inside ``get_nominal_output`` (matlab_new/lib/ExpSeq.m:683).

KEY FACT (verified against the MATLAB source): the per-channel point data is
produced by the libnacs **engine** call ``get_nominal_output(pts_per_ramp)``.
This module does NOT re-evaluate the sequence in Python -- it only orchestrates
the engine call and packs the result into the ``.seq`` byte layout. The byte
packing itself is delegated to ``compare_seq_bytes.encode`` so the reader and the
writer can never drift apart (one source of truth for the format).

``.seq`` is an OUTPUT-ONLY plotting artifact consumed by SeqPlotter, *not* fed to the
engine -- so it is not on the serialize path and not bound by byte-equality. (Byte-equality
is still a handy *validation*; see tests.) That freedom lets us optionally enrich the
parameters block with a ``scanned`` marker for the dashboard's scanned-parameter
highlighting (see ``mark_scanned``); strict-parity callers simply omit the marker.

Where this lands: when ``ExpSeq`` is ported (Phase 2), ``dump_output(...)`` becomes
``ExpSeq.dump_output_to_file``; the channel-alias decoration needs the channel map
(Phase 3) and per-scan-point params need ``ScanGroup`` (Phase 4). Until then the
pieces here work standalone against a compiled engine handle.
"""

import json
import os
import sys

# The byte layout lives in tools/compare_seq_bytes.py (stdlib-only reader/encoder).
# Make it importable whether or not the test conftest already put tools/ on path.
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import compare_seq_bytes  # noqa: E402  (path set up just above)


# --------------------------------------------------------------------------- #
# Channel extraction from the engine's get_nominal_output return.
# MATLAB: res = self.pyseq.get_nominal_output(pts); res{i} = {name, times, values, pulse_ids}
#         (matlab_new/lib/ExpSeq.m:683-731)
# --------------------------------------------------------------------------- #
def _as_str(name):
    return name.decode("latin1") if isinstance(name, (bytes, bytearray)) else str(name)


def decorate_channel_name(nominal_name, inverse_chn_map=None):
    """Reproduce ``get_nominal_output``'s alias-decorated channel name.

    Port of matlab_new/lib/ExpSeq.m:699-721. ``inverse_chn_map`` maps an engine
    (nominal) channel name to the list of user-facing aliases for it. With no map
    (channel map is a Phase-3 concern), the nominal name is used verbatim.

    Result: ``nominal`` if there are no extra aliases, else
    ``"alias1, alias2 (nominal)"``.
    """
    if not inverse_chn_map:
        return nominal_name
    all_names = inverse_chn_map.get(nominal_name, [nominal_name])
    additional = [n for n in all_names if n != nominal_name]
    if not additional:
        return nominal_name
    return "%s (%s)" % (", ".join(additional), nominal_name)


def channels_from_nominal_output(res, inverse_chn_map=None):
    """Turn the engine's ``get_nominal_output`` return into channel structures.

    ``res`` is an iterable of 4-element entries ``(name, times, values, pulse_ids)``
    (lists / tuples / numpy arrays all accepted). Returns the channel list the
    ``.seq`` encoder expects: ``[{name, points:[{t,v,pid}, ...]}, ...]``.
    """
    channels = []
    for entry in res:
        name = decorate_channel_name(_as_str(entry[0]), inverse_chn_map)
        times, values, pulse_ids = entry[1], entry[2], entry[3]
        points = [{"t": int(t), "v": float(v), "pid": int(pid)}
                  for t, v, pid in zip(times, values, pulse_ids)]
        channels.append({"name": name, "points": points})
    return channels


# --------------------------------------------------------------------------- #
# Parameters block + the scanned-parameter highlight hook.
# Real leaf schema (from a captured .seq): {"value": <num>, "type": <int>,
# optionally "config_value": <num>}. There is NO native "scanned" flag, so for
# highlighting we inject one (safe: .seq is a viewer artifact, not engine input).
# --------------------------------------------------------------------------- #
def mark_scanned(params, scanned, scan_dims=None):
    """Return a copy of ``params`` with scanned leaves flagged for the UI.

    ``scanned`` is an iterable of dotted parameter paths (e.g. ``"AWG.AWG556.
    pulse_width_us"``). For each, the leaf dict gains ``"scanned": true`` and,
    if ``scan_dims`` maps that path to an int, ``"scan_dim": <n>``. Leaves are the
    ``{"value", "type", ...}`` dicts; intermediate dicts are walked by dotted key.

    SeqPlotter (or a small patch to it) reads ``leaf["scanned"]`` to highlight the
    parameter. Omit this call entirely for a strict byte-parity ``.seq``.
    """
    import copy
    out = copy.deepcopy(params)
    scan_dims = scan_dims or {}
    for path in scanned:
        node = out
        keys = path.split(".")
        ok = True
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, dict):
            node["scanned"] = True
            if path in scan_dims:
                node["scan_dim"] = int(scan_dims[path])
    return out


# --------------------------------------------------------------------------- #
# Packing (delegates byte layout to compare_seq_bytes.encode).
# --------------------------------------------------------------------------- #
def build_seq_struct(seq_name, seq_idx, channels, params=None):
    """Build one decoded-seq dict (the shape compare_seq_bytes.encode consumes)."""
    s = {"seq_name": seq_name, "seq_idx": int(seq_idx), "channels": channels}
    if params is None:
        s["has_params"] = 0
    else:
        s["has_params"] = 1
        # MATLAB jsonencode emits compact JSON (no spaces); match separators.
        s["params_raw"] = json.dumps(params, separators=(",", ":"))
        s["params"] = params
    return s


def pack(seqs, has_bt_info=False, backtraces=None, bt_idx=None):
    """Pack a list of seq dicts into ``.seq`` bytes via the shared encoder.

    ``seqs`` entries come from ``build_seq_struct``. The backtrace block is
    optional (debug-only; not needed for plotting).
    """
    struct_ = {"seqs": list(seqs), "has_bt_info": 1 if has_bt_info else 0}
    if has_bt_info:
        struct_["bt_idx"] = list(bt_idx) if bt_idx is not None else [0] * len(seqs)
        struct_["backtraces"] = list(backtraces or [])
    return compare_seq_bytes.encode(struct_)


def format_seq_name(name, dt=None):
    """Build MATLAB's ``act_seq_name`` = ``yyyymmdd_HHMMSS:name`` (ExpSeq.m:595).

    ``dt`` is a ``datetime``; pass one explicitly in production (e.g.
    ``datetime.now()``). When ``None`` the bare ``name`` is returned so callers /
    tests stay deterministic (the timestamp is the only nondeterministic field in
    the whole ``.seq``, and the byte-equality tests compare modulo it).
    """
    if dt is None:
        return name
    return "%s_%s:%s" % (dt.strftime("%Y%m%d"), dt.strftime("%H%M%S"), name)


# --------------------------------------------------------------------------- #
# Orchestration: compiled engine handle -> .seq bytes.
# --------------------------------------------------------------------------- #
def dump_output(engine_seq, pts_per_ramp=100, seq_name="", seq_idx=1,
                params=None, inverse_chn_map=None):
    """Single-basic-sequence dump (compile/eval-only -- no init_run/start).

    ``engine_seq`` is the handle returned by ``SeqManager.create_sequence`` (real
    engine or the test dummy). Calls ``get_nominal_output(pts_per_ramp)`` and packs
    the one basic sequence. This path does not drive hardware and is safe to run
    on a compiled sequence. For a sequence with branches/multiple basic sequences
    use ``dump_output_branches`` (needs the real engine + a downtime window).
    """
    res = engine_seq.get_nominal_output(pts_per_ramp)
    channels = channels_from_nominal_output(res, inverse_chn_map)
    s = build_seq_struct(seq_name, seq_idx, channels, params)
    return pack([s])


def dump_output_branches(engine_seq, pts_per_ramp=100, seq_name="",
                         params_for=None, inverse_chn_map=None):
    """Multi-basic-sequence dump, walking branches like ExpSeq.dump_output_to_file.

    Mirrors matlab_new/lib/ExpSeq.m:598-624: init_run -> loop {pre_run ->
    get_nominal_output -> post_run(branch idx)}. ``init_run``/``post_run`` are the
    engine's run-control entry points; per PYTHON_FRONTEND_PLAN.md they are
    treated as NEEDS-HARDWARE, so run this only with the real engine in a downtime
    window (the board-free dummy raises on init_run by design).

    ``params_for`` is an optional ``callable(seq_idx) -> params dict``.
    """
    engine_seq.init_run()
    seqs = []
    idx = 1
    while idx != 0:
        engine_seq.pre_run()
        res = engine_seq.get_nominal_output(pts_per_ramp)
        channels = channels_from_nominal_output(res, inverse_chn_map)
        params = params_for(idx) if params_for else None
        seqs.append(build_seq_struct(seq_name, idx, channels, params))
        idx = int(engine_seq.post_run())
    return pack(seqs)


def dump_output_to_file(path, engine_seq, pts_per_ramp=100, seq_name="",
                        seq_idx=1, params=None, inverse_chn_map=None):
    """Convenience: ``dump_output`` then write the bytes to ``path``."""
    data = dump_output(engine_seq, pts_per_ramp, seq_name, seq_idx,
                       params, inverse_chn_map)
    with open(path, "wb") as f:
        f.write(data)
    return data
