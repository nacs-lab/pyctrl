"""scan_programs.py -- turn a ScanGroup into QICK programs + per-shot selection keys.

The QICK analog of the Siglent ``AWGManager``'s scangroup walk (``devices/sigilent_awg/awg_manager.py``
``_seq_awg_overrides`` / ``_build_key``). A scan declares QICK behavior via ``g().QICK.*`` (the params
in ``expConfig`` ``c["QICK"]``) and opts in with ``g().runp().QICK = True`` -- exactly mirroring
``g().AWG.<name>.*`` + ``g().runp().AWGs``.

  * :func:`qick_enabled`  -- does this scan activate QICK? (``runp().QICK``)
  * :func:`resolve_params` -- merge the live ``c["QICK"]`` defaults with a seq's ``QICK`` overrides.
  * :func:`build_programs` -- walk every seq in the group -> one :class:`QickProgram` per seq (the
    manager dedups by ``key``, so a swept scalar mints one program per unique value).
  * :func:`seq_qick_key`  -- the per-shot key for a single seq (identical to the builder's key, so
    ``arm_for_seq`` selects the program ``setup`` uploaded).

Out-of-band device: none of this touches the serialized byte blob (THE ONE RULE does not apply).
"""
from .templates import build_program


def _live_defaults():
    """The live ``c["QICK"]`` defaults dict from the active SeqConfig (empty if absent)."""
    try:
        from seq_config import SeqConfig
        return dict(SeqConfig.get().consts.get("QICK", {}))
    except Exception:  # noqa: BLE001 - no active config (e.g. a unit test) -> caller passes defaults
        return {}


def qick_enabled(scangroup):
    """True iff the scan opts into QICK via ``runp().QICK`` (mirrors ``runp().AWGs`` gating).

    Any truthy value enables. Defensive: a missing field / no runp -> False (non-QICK scans pay
    nothing).
    """
    try:
        return bool(scangroup.runp().QICK(False))
    except Exception:  # noqa: BLE001
        return False


def _seq_qick_overrides(seq):
    """Extract a getseq() result's ``QICK`` override dict (empty if none)."""
    if not isinstance(seq, dict):
        return {}
    q = seq.get("QICK")
    return dict(q) if isinstance(q, dict) else {}


def resolve_params(seq, defaults=None):
    """Merge the live ``c["QICK"]`` defaults with this seq's ``QICK`` overrides (overrides win)."""
    params = dict(defaults if defaults is not None else _live_defaults())
    params.update(_seq_qick_overrides(seq))
    return params


def build_programs(scangroup, defaults=None):
    """One :class:`QickProgram` per seq in the group (1-based ``getseq``), for ``FPGAAWGManager.setup``.

    Duplicates (same ``key``) are uploaded once by the manager. ``defaults`` overrides the live config
    source (for tests).
    """
    base = defaults if defaults is not None else _live_defaults()
    programs = []
    total = scangroup.nseq()
    for n in range(1, total + 1):
        params = resolve_params(scangroup.getseq(n), base)
        programs.append(build_program(params))
    return programs


def seq_qick_key(seq, defaults=None):
    """The per-shot program-selection key for one seq -- identical to the builder's ``QickProgram.key``.

    Built via the same ``build_program`` path so ``arm_for_seq(seq_qick_key(seq))`` always names a
    program that ``setup(build_programs(...))`` uploaded.
    """
    return build_program(resolve_params(seq, defaults)).key
