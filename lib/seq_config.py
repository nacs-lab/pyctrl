"""seq_config.py -- MINIMAL Phase-2 stub of ``matlab_new/lib/SeqConfig.m``.

``TestExpSeq.m`` runs against ``lib/test/config/expConfig.m``, which is **empty**
(``%% Empty one for testing``). With an empty config there are no consts, no
channel aliases (so channel translation is the identity), no default values, and
nothing disabled. This stub reproduces exactly that empty-config surface so Phase 2
can build + serialize ``ExpSeq`` byte-identically to the committed ``seq*.json``.

The real port -- loading ``expConfig.m``, the ``TTL/V/Freq/Amp`` alias rule,
recursive alias expansion, default values, disabled-channel prefixes -- is
**Phase 3**. See PYTHON_FRONTEND_PLAN.md.
"""

from dyn_props import DynProps


class SeqConfig:
    # Mirror SeqConfig.get(is_seq)'s cached singleton (MATLAB uses a MutableRef).
    _cache = {}

    @staticmethod
    def get(is_seq):
        cfg = SeqConfig._cache.get(is_seq)
        if cfg is None:
            cfg = SeqConfig()
            SeqConfig._cache[is_seq] = cfg
        return cfg

    def __init__(self):
        self.consts = {}            # empty test config -> no C fields
        self.G = DynProps({})       # shared global context (unused in Phase 2)
        self.default_vals = {}      # translated name -> default value (none)

    def translate_channel(self, name):
        # Empty config has no aliases: names pass through unchanged.
        return name

    def check_channel_disabled(self, name):
        return False
