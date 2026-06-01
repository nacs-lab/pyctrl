"""seq_config.py -- SeqConfig: empty test config by default, real expConfig on opt-in.

Phase-3 (config & globals) port of ``matlab_new/lib/SeqConfig.m``.

MATLAB selects the config by which ``expConfig.m`` is on the path: the empty
``lib/test/config/expConfig.m`` for ``TestExpSeq`` (no consts, identity channel
translation), and the real ``matlab_new/expConfig.m`` for production. pyctrl mirrors
that with a process singleton that is **empty by default** -- so the Phase-2
``seq*.json`` byte tests keep building against an empty config untouched -- and the
real config is **opt-in** via :meth:`SeqConfig.load_real` (used by the real-sequence
byte builds; reset in teardown so later tests get the empty default back).

The real config is a committed JSON capture (``tests/reference/config_reference.json``,
produced engine-free by ``tools/capture_config_reference.m``) -- the HYBRID decision in
PYTHON_FRONTEND_PLAN.md Phase 3; an executable ``exp_config.py`` is deferred to Phase 5/6.

Byte-load-bearing surface of the real config (bare build, no scan):
  * ``consts``        -- read as both ``s.C.X`` and ``Consts().X`` (the same tree)
  * ``channel_alias`` -- drives :meth:`translate_channel` -> the cid map / ``[nchns]`` block
  * ``default_vals``  -- drives the ``[ndefvals]`` block

CRITICAL (audit fix #3): every numeric leaf is coerced to ``float`` on load. MATLAB
consts/defaults are all class ``double``; JSON emits integer-valued doubles as ``1``
(-> Python ``int``), which would serialize as ``ARG_CONST_INT32`` instead of
``ARG_CONST_FLOAT64`` and diverge byte-for-byte.
"""

import json
import os

from dyn_props import DynProps

_CONFIG_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "tests", "reference", "config_reference.json")


def _coerce_floats(x):
    # MATLAB stores all numeric config as double -> force float so value nodes
    # serialize as ARG_CONST_FLOAT64. Guard bool first (bool is a subclass of int);
    # expConfig has no logical consts, but stay faithful if one ever appears.
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return float(x)
    if isinstance(x, dict):
        return {k: _coerce_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce_floats(v) for v in x]
    return x


class SeqConfig:
    # MATLAB caches ONE SeqConfig (a single MutableRef); is_seq only gates a
    # disabled-channel warning on first construction, which never fires here.
    _singleton = None

    @staticmethod
    def get(is_seq=0):
        if SeqConfig._singleton is None:
            SeqConfig._singleton = SeqConfig()        # default: empty test config
        return SeqConfig._singleton

    @staticmethod
    def reset():
        SeqConfig._singleton = None

    @staticmethod
    def load_real(config_path=None):
        """Activate the captured real expConfig as the singleton (real-seq builds)."""
        with open(config_path or _CONFIG_JSON) as f:
            raw = json.load(f)
        SeqConfig._singleton = SeqConfig(raw)
        return SeqConfig._singleton

    def __init__(self, raw=None):
        self._name_map = {}        # translate_channel memo + alias-loop sentinel
        if raw is None:
            # Empty test config (matches lib/test/config/expConfig.m): no aliases
            # (identity translation), no consts, no defaults. Preserves Phase-2 bytes.
            self.channel_alias = {}
            self.consts = {}
            self.default_vals = {}
        else:
            # channel_alias values are already trailing-slash-trimmed at capture
            # (SeqConfig.m:82-85).
            self.channel_alias = dict(zip(raw["channel_alias_keys"],
                                          raw["channel_alias_vals"]))
            self.consts = _coerce_floats(raw["consts"])
            # default_vals captured RAW (alias keys); re-key by TRANSLATED backend
            # name (SeqConfig.m:89-97), float-coerced, with MATLAB's conflict check.
            self.default_vals = {}
            for k, v in zip(raw["default_vals_keys"], raw["default_vals_vals"]):
                name = self.translate_channel(k)
                v = float(v)
                if name in self.default_vals and self.default_vals[name] != v:
                    raise ValueError(
                        'Conflict default values for channel "%s" (%s).' % (k, name))
                self.default_vals[name] = v
        self.G = DynProps({})      # shared global context (ExpSeqBase.G)

    def translate_channel(self, name):
        # Recursive alias expansion: only the FIRST '/'-component is aliased, then
        # re-joined and re-resolved; name_map memoizes and a None sentinel set before
        # recursing detects alias loops. Port of SeqConfig.m:138-155.
        nm = self._name_map
        if name in nm:
            res = nm[name]
            if res is None:
                raise ValueError("Alias loop detected: %s." % name)
            return res
        cpath = name.split("/")
        nm[name] = None            # loop sentinel
        if cpath[0] in self.channel_alias:
            cpath[0] = self.channel_alias[cpath[0]]
            res = self.translate_channel("/".join(cpath))
        else:
            res = name
        nm[name] = res
        return res

    def check_channel_disabled(self, name):
        # disabledChannels is empty in expConfig.m (no disableChannel() calls).
        return False
