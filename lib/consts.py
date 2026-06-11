"""consts.py -- Consts: a DynProps over the active config's consts tree.

Faithful transliteration of ``matlab_new/lib/Consts.m`` (the whole class is 8 lines):
``Consts()`` is a :class:`DynProps` constructed over ``SeqConfig.get().consts``. With the
real config active (``SeqConfig.load_real()``) it reads the real expConfig consts; with the
empty default config it is an empty DynProps.

It deliberately wraps the SAME consts tree that ``ExpSeq`` copies into ``s.C``, so a bare
build step reads identical values via both ``Consts().X`` and ``s.C.X``.

``DynProps.__init__`` deep-copies the store (audit fix #2), so every ``Consts()`` instance
owns an independent copy -- a default-write through one neither leaks into another nor
mutates ``SeqConfig.get().consts``.
"""

from dyn_props import DynProps
from seq_config import SeqConfig


class Consts(DynProps):
    def __init__(self):
        # SeqConfig.get() with the default is_seq (matches MATLAB Consts(): conf =
        # SeqConfig.get()); DynProps deep-copies conf.consts into this instance.
        store = SeqConfig.get().consts
        # Per-pattern overlay: when a build has an active SLM pattern (expconfig_helper's
        # current_pattern, set by the runner per scan and by the per-bseq build hook), overlay
        # that pattern's c["ByPattern"] entry onto the consts BEFORE DynProps deep-copies. No
        # active pattern (or an empty/unknown entry) -> apply_pattern returns the store unchanged
        # -> byte-identical. Lazy import keeps a JSON-snapshot config (no helper on path) working.
        try:
            import expConfig_helper
            pat = expConfig_helper.current_pattern()
            if pat:
                store = expConfig_helper.apply_pattern(store, pat)
        except Exception:  # noqa: BLE001 -- overlay is best-effort; fall back to base consts
            pass
        super().__init__(store)
