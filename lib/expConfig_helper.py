"""expConfig_helper.py -- per-pattern config overlay + cross-ref resolution for expConfig.

Lives in ``lib/`` (the config/framework machinery, next to seq_config/dyn_props/consts) and is
kept OUT of ``expConfig.py`` so that module stays the pure data/config source; this holds the
logic that PROCESSES that config. Imported by ``expConfig._consts`` (for the const-to-const
cross-refs) and by ``lib/consts.py`` + the runner + the per-bseq build hook (for the per-pattern
overlay). Pure (stdlib only) -- so no import cycle with ``seq_config``->``expConfig``, and it is
NOT dropped by the per-job ``expConfig`` hot-reload (``SeqConfig.load_real(reload=True)`` only
pops ``expConfig``); deliberate, so the active-pattern context below survives a config reload.

Per-pattern overlay model: ``consts["ByPattern"][name]`` (built in ``expConfig._consts``) is a
SPARSE override of the base consts for an SLM loading pattern -- the cooling/imaging/VSLMServo
leaves that differ for that array. RUNTIME-ONLY: no sequence reads ``ByPattern``, so it has no
serialize() byte effect until a leaf is actually overlaid AND a scan resolves to that pattern.
The active pattern during a build is :func:`set_current_pattern` (the runner sets the
scan-default per scan; the per-bseq build hook overrides it per bseq); ``Consts()`` and the hook
overlay ``ByPattern[current]`` via :func:`apply_pattern`. Final precedence: base < ByPattern <
scan ``g()`` (the scan's params are layered last, in ExpSeq).
"""

import copy

# --- active SLM pattern context ------------------------------------------------------------ #
_CURRENT_PATTERN = None


def set_current_pattern(name):
    """Set (None clears) the active SLM pattern for subsequent Consts()/overlay reads."""
    global _CURRENT_PATTERN
    _CURRENT_PATTERN = name or None


def current_pattern():
    """The active SLM pattern name, or None."""
    return _CURRENT_PATTERN


# --- merge / overlay ------------------------------------------------------------------------ #
def deep_merge(base, overlay):
    """A NEW dict = base with overlay applied (overlay leaves win, sub-dicts recurse). Neither
    input is mutated; merged-in values are deep-copied."""
    out = dict(base)
    for k, ov in overlay.items():
        bv = out.get(k)
        if isinstance(bv, dict) and isinstance(ov, dict):
            out[k] = deep_merge(bv, ov)
        else:
            out[k] = copy.deepcopy(ov)
    return out


def pattern_params(consts, pattern_name):
    """The sparse per-pattern override dict for ``pattern_name`` (or {} if absent/unnamed)."""
    if not pattern_name:
        return {}
    return (consts.get("ByPattern", {}) or {}).get(pattern_name, {}) or {}


def apply_pattern(consts, pattern_name):
    """Return ``consts`` deep-merged with its per-pattern overrides, cross-refs re-resolved.
    Returns ``consts`` UNCHANGED (identity) when the pattern is unknown or its entry is empty,
    so the no-pattern path is byte-identical. Never mutates ``consts``."""
    ov = pattern_params(consts, pattern_name)
    if not ov:
        return consts
    return apply_cross_refs(copy.deepcopy(deep_merge(consts, ov)))


def apply_cross_refs(c):
    """Re-resolve expConfig.m's const-to-const links. Idempotent, so it runs both at config
    build (end of ``expConfig._consts``) and after a per-pattern overlay -- the overlay may
    change a source leaf (e.g. Init.VSLMServo, which feeds SLM.VServo)."""
    c["SLM"]["VServo"] = c["Init"]["VSLMServo"]
    c["LAC"]["BlueLAC"]["Resonance556mj0Freq"] = c["Resonance556mj0Freq"]
    c["Imag399"]["ExposureTime"] = c["Orca"]["ExposureTime"]
    return c


def build_seq_consts(base, c_ovr, pattern_name):
    """Build an ExpSeq's ``s.C`` store: ``base`` overlaid with the per-pattern entry (deep
    merge), then the scan's params ``c_ovr`` TOP-LEVEL replaced on top -- mirroring
    ExpSeq.__init__'s base-then-c_ovr merge with the pattern layer inserted between. Final
    precedence: base < ByPattern < scan ``g()``.

    Returns a FULLY INDEPENDENT dict (safe to assign straight to ``DynProps._store`` -- the
    per-bseq build hook does exactly that, bypassing the DynProps deep-copy). When
    ``pattern_name`` is unknown/empty this equals the plain base (+) c_ovr merge, so the
    no-pattern path is byte-identical."""
    c = apply_pattern(base, pattern_name)          # base (shared) or fresh base(+)pattern
    c = copy.deepcopy(c) if c is base else c        # independent of base for the identity case
    if c_ovr:
        for fn in c_ovr:
            c[fn] = copy.deepcopy(c_ovr[fn])
    return c
