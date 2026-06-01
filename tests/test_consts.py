"""W3 (Phase-3 config) -- Consts(): a DynProps over the active config's consts.

NO-HARDWARE: pure config math; never loads the engine. The real config is opt-in
(SeqConfig.load_real); every test resets the process singleton in teardown so the empty
default is restored for the rest of the suite (mirrors test_seq_config.py).

Covers: real-config reads (scalar + nested + nested default-fallback, all float-typed),
the deepcopy-independence guarantee (Consts instances are isolated from each other and
from SeqConfig.get().consts), and the empty-default config behaving as an empty DynProps.
"""

import os

import pytest

from consts import Consts
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "reference", "config_reference.json")
_needs_capture = pytest.mark.skipif(
    not os.path.exists(_REF),
    reason="no config_reference.json (run tools/capture_config_reference.m)")


@pytest.fixture
def real_config():
    """The captured real expConfig as the active singleton; reset afterwards."""
    SeqConfig.reset()
    cfg = SeqConfig.load_real()
    yield cfg
    SeqConfig.reset()


@pytest.fixture
def empty_config():
    """The empty default config as the active singleton; reset afterwards."""
    SeqConfig.reset()
    yield SeqConfig.get()
    SeqConfig.reset()


# --------------------------------------------------------------------------- #
# Real captured config -- Consts() reads it, and reads it as float.
# --------------------------------------------------------------------------- #
@_needs_capture
class TestRealConfig:
    def test_scalar_const(self, real_config):
        # Resolve with no default (parens) -> raw value; check value AND float type.
        v = Consts().Resonance399Freq()
        assert v == 310e6
        assert type(v) is float

    def test_nested_const(self, real_config):
        v = Consts().Init.TwoDMOT.FreqDetuning()
        assert v == -20e6
        assert type(v) is float

    def test_nested_default_fallback(self, real_config):
        # Amp already exists (= 1.0) so the parens default 99 is ignored: the
        # default-fallback syntax resolves to the existing value, not the default.
        v = Consts().Init.TwoDMOT.Amp(99)
        assert v == 1.0
        assert type(v) is float

    def test_same_tree_as_seqconfig(self, real_config):
        # Consts() wraps the same consts SeqConfig exposes (the tree ExpSeq copies to s.C).
        assert Consts().Resonance399Freq() == real_config.consts["Resonance399Freq"]

    # -- deepcopy independence (audit fix #2) ------------------------------- #
    def test_instances_are_independent(self, real_config):
        c1 = Consts()
        c2 = Consts()
        # A default-write of a MISSING field persists into c1's store only.
        assert c1.NewConst(42.0) == 42.0
        assert not c2.isfield("NewConst")
        assert "NewConst" not in SeqConfig.get().consts

    def test_nested_default_write_isolated(self, real_config):
        c1 = Consts()
        c2 = Consts()
        assert c1.Init.TwoDMOT.NewField(7.0) == 7.0
        assert not c2.Init.TwoDMOT.isfield("NewField")
        assert "NewField" not in SeqConfig.get().consts["Init"]["TwoDMOT"]

    def test_default_write_does_not_mutate_source(self, real_config):
        # The shared SeqConfig.consts tree is untouched by Consts() default-writes.
        before = real_config.consts["Init"]["TwoDMOT"]["Amp"]
        Consts().Init.TwoDMOT.Amp(99)          # existing value wins; nothing persisted
        Consts().BrandNewLeaf(123.0)           # persisted into that instance only
        assert real_config.consts["Init"]["TwoDMOT"]["Amp"] == before
        assert "BrandNewLeaf" not in real_config.consts


# --------------------------------------------------------------------------- #
# Empty default config -- Consts() is an empty DynProps.
# --------------------------------------------------------------------------- #
class TestEmptyDefault:
    def test_empty_has_no_fields(self, empty_config):
        assert Consts().fieldnames() == []

    def test_missing_without_default_raises(self, empty_config):
        with pytest.raises(KeyError):
            Consts().Anything()

    def test_missing_with_default_returns_default(self, empty_config):
        assert Consts().Anything(5.0) == 5.0
