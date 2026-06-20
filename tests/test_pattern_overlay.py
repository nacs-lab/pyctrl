"""Per-pattern config overlay (expConfig ByPattern) -- unit + integration.

NO-HARDWARE. The full suite's byte oracles already prove the overlay is INERT when ByPattern is
empty (no behaviour change). These tests prove it WORKS when populated: the deep-merge + per-leaf
fallback + cross-ref re-resolution (expConfig_helper), the Consts() overlay keyed by the active
pattern, and the per-bseq build hook (set_pattern -> the step sees that pattern in both Consts()
and s.C). Precedence asserted end to end: base < ByPattern < scan g().
"""

import pytest

import expConfig_helper as H
import seq_manager
from consts import Consts
from exp_seq import ExpSeq
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware


# Minimal base consts with the leaves apply_cross_refs touches + a sparse ByPattern entry.
def _base():
    return {
        "Init": {"VSLMServo": 6.0},
        "Imag399": {"Amp": 0.18, "ExposureTime": None,
                    "Cool556": {"X": {"Amp": 0.20}, "h": {"Amp": 0.13}}},
        "SLM": {"VServo": None},
        "LAC": {"BlueLAC": {"Resonance556mj0Freq": None}},
        "Orca": {"ExposureTime": 0.05},
        "Resonance556mj0Freq": 1.07e8,
        "ByPattern": {
            "47x47_uniform": {
                "Init": {"VSLMServo": 3.7},
                "Imag399": {"Amp": 0.16, "Cool556": {"X": {"Amp": 0.18}}},
            },
        },
    }


# --------------------------------------------------------------------------- #
# expConfig_helper -- pure merge / overlay
# --------------------------------------------------------------------------- #
def test_apply_pattern_overlay_fallback_crossref():
    base = _base()
    c = H.apply_pattern(base, "47x47_uniform")
    assert c["Init"]["VSLMServo"] == 3.7                      # overridden by pattern
    assert c["Imag399"]["Amp"] == 0.16                        # overridden
    assert c["Imag399"]["Cool556"]["X"]["Amp"] == 0.18        # overridden (nested)
    assert c["Imag399"]["Cool556"]["h"]["Amp"] == 0.13        # per-leaf FALLBACK to base
    assert c["SLM"]["VServo"] == 3.7                          # cross-ref re-resolved to overlaid VSLMServo
    # base is never mutated
    assert base["Init"]["VSLMServo"] == 6.0
    assert base["SLM"]["VServo"] is None


def test_apply_pattern_identity_when_absent():
    base = _base()
    assert H.apply_pattern(base, "no_such_pattern") is base   # identity -> byte-identical
    assert H.apply_pattern(base, None) is base
    # a present-but-empty entry is also identity
    base["ByPattern"]["empty"] = {}
    assert H.apply_pattern(base, "empty") is base


def test_build_seq_consts_precedence_base_pattern_scan():
    base = _base()
    # scan c_ovr (top-level Imag399) must win over the pattern AND base
    c = H.build_seq_consts(base, {"Imag399": {"Amp": 0.25}}, "47x47_uniform")
    assert c["Imag399"]["Amp"] == 0.25                        # scan g() wins (pattern was 0.16, base 0.18)
    assert c["Init"]["VSLMServo"] == 3.7                      # pattern wins over base (c_ovr didn't touch Init)
    # independent of base (safe to drop straight into DynProps._store)
    assert base["Imag399"]["Amp"] == 0.18
    # no-pattern + no c_ovr == plain base (byte-identical path)
    assert H.build_seq_consts(base, {}, None)["Imag399"]["Amp"] == 0.18


# --------------------------------------------------------------------------- #
# integration -- Consts() overlay + per-bseq build hook
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg_with_pattern():
    """Real expConfig + production tick, with a per-pattern entry injected into the live consts."""
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    SeqConfig.get().consts["ByPattern"] = {
        "PAT": {"Imag399": {"Amp1": 0.999}, "Init": {"VSLMServo": 1.23}}}
    base_amp = SeqConfig.get().consts["Imag399"]["Amp1"]
    yield base_amp
    H.set_current_pattern(None)
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


def test_consts_overlay_follows_current_pattern(cfg_with_pattern):
    base_amp = cfg_with_pattern
    H.set_current_pattern(None)
    assert Consts().Imag399.Amp1() == base_amp               # base
    H.set_current_pattern("PAT")
    assert Consts().Imag399.Amp1() == 0.999                  # overlaid
    assert Consts().SLM.VServo() == 1.23                     # cross-ref to overlaid VSLMServo
    H.set_current_pattern(None)
    assert Consts().Imag399.Amp1() == base_amp               # restored


def test_expseq_flags_by_pattern(cfg_with_pattern):
    H.set_current_pattern(None)
    assert ExpSeq()._has_by_pattern is True                  # picked up from consts["ByPattern"]


def test_per_bseq_overlay_in_build(cfg_with_pattern):
    base_amp = cfg_with_pattern
    H.set_current_pattern(None)                              # scan-default = base
    s = ExpSeq()
    seen = {}

    def cap(sub):
        # capture BOTH the Consts() overlay path and the s.C (g) path the steps actually read
        seen[sub.root.pattern] = (Consts().Imag399.Amp1(), sub.C.Imag399.Amp1(0))

    s.add_step(cap)                                          # root bseq: pattern None -> base
    s2 = s.new_basic_seq(pattern="PAT")
    s2.add_step(cap)                                         # bseq2: pattern PAT -> overlaid
    assert seen[None] == (base_amp, base_amp)
    assert seen["PAT"] == (0.999, 0.999)
    # context restored to the scan-default after the build (no leak to the next shot)
    assert H.current_pattern() is None


def test_set_pattern_chaining_and_clear(cfg_with_pattern):
    s = ExpSeq()
    assert s.set_pattern("X") is s and s.pattern == "X"
    assert s.set_pattern(None).pattern is None
