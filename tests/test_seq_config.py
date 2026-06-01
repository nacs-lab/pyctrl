"""W1 (Phase-3 config) -- SeqConfig loader: channel translation, float-coercion,
default-value re-keying, and the empty-config default that keeps Phase 2 building.

NO-HARDWARE: pure config math against the committed engine-free capture
(tests/reference/config_reference.json from tools/capture_config_reference.m). Never
loads the engine. The real config is opt-in (SeqConfig.load_real); every test resets
the singleton in teardown so the empty default is restored for the rest of the suite.
"""

import os

import pytest

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


def _ints_in(x):
    """All genuine ints (not bools) anywhere in a nested dict/list -- should be none
    after float-coercion (MATLAB doubles must not serialize as ARG_CONST_INT32)."""
    if isinstance(x, bool):
        return []
    if isinstance(x, int):
        return [x]
    if isinstance(x, dict):
        return [i for v in x.values() for i in _ints_in(v)]
    if isinstance(x, list):
        return [i for v in x for i in _ints_in(v)]
    return []


# --------------------------------------------------------------------------- #
# Empty default config -- the Phase-2 contract (must stay identity / empty).
# --------------------------------------------------------------------------- #
class TestEmptyDefault:
    def test_default_is_empty(self):
        SeqConfig.reset()
        cfg = SeqConfig.get(1)
        assert cfg.consts == {}
        assert cfg.default_vals == {}
        assert cfg.check_channel_disabled("anything") is False

    def test_default_translation_is_identity(self):
        SeqConfig.reset()
        cfg = SeqConfig.get()
        assert cfg.translate_channel("Freq2DMOT") == "Freq2DMOT"
        assert cfg.translate_channel("FPGA1/TTL5") == "FPGA1/TTL5"

    def test_singleton_and_reset(self):
        SeqConfig.reset()
        a = SeqConfig.get()
        b = SeqConfig.get(1)
        assert a is b                      # one shared singleton, is_seq irrelevant
        SeqConfig.reset()
        assert SeqConfig.get() is not a


# --------------------------------------------------------------------------- #
# Real captured config.
# --------------------------------------------------------------------------- #
@_needs_capture
class TestRealConfig:
    def test_load_real_is_active_singleton(self, real_config):
        assert SeqConfig.get() is real_config
        assert SeqConfig.get(1) is real_config
        assert len(real_config.channel_alias) > 0
        assert len(real_config.consts) > 0

    # -- channel translation (recursive alias expansion) -------------------- #
    @pytest.mark.parametrize("alias,expected", [
        ("Freq2DMOT", "FPGA1/DDS21/FREQ"),     # one-hop DDS alias
        ("Amp2DMOT", "FPGA1/DDS21/AMP"),
        ("TTL556MOTaShutter", "FPGA1/TTL5"),   # one-hop TTL alias
        ("VMOTCoil", "NiDAQ/Dev1/0"),          # two-hop: VMOTCoil->Dev1/0->NiDAQ/Dev1/0
        ("VElectrode1", "NiDAQ/Dev1/12"),
        ("Dev1/0", "NiDAQ/Dev1/0"),            # device-prefix alias on first component
        ("FreqCatsEye", "FreqCatsEye"),        # unaliased (commented out) -> identity
        ("FPGA1/TTL5", "FPGA1/TTL5"),          # backend name -> identity
    ])
    def test_translate_channel(self, real_config, alias, expected):
        assert real_config.translate_channel(alias) == expected

    def test_translate_channel_memoized(self, real_config):
        first = real_config.translate_channel("VMOTCoil")
        assert real_config.translate_channel("VMOTCoil") == first
        assert real_config._name_map["VMOTCoil"] == "NiDAQ/Dev1/0"

    # -- float-coercion (audit fix #3: no int leaks -> ARG_CONST_INT32) ------ #
    def test_consts_have_no_int_leaves(self, real_config):
        assert _ints_in(real_config.consts) == []

    def test_default_vals_are_all_float(self, real_config):
        assert real_config.default_vals, "expected some defaults"
        for name, v in real_config.default_vals.items():
            assert type(v) is float, "%s default %r is %s" % (name, v, type(v).__name__)

    @pytest.mark.parametrize("path,expected", [
        (("Resonance399Freq",), 310e6),
        (("GreenMOT", "BFieldGradient"), 3.0),
        (("Init", "TwoDMOT", "Amp"), 1.0),
        (("Init", "TwoDMOT", "FreqDetuning"), -20e6),
        (("AbsImag", "TOF"), 0.0),
    ])
    def test_consts_values_resolved_as_float(self, real_config, path, expected):
        v = real_config.consts
        for k in path:
            v = v[k]
        assert v == expected
        assert type(v) is float

    def test_consts_cross_reference_resolved(self, real_config):
        # consts.SLM.VServo = consts.Init.VSLMServo (expConfig.m) -> same resolved value
        assert real_config.consts["SLM"]["VServo"] == real_config.consts["Init"]["VSLMServo"]
        assert real_config.consts["Init"]["VSLMServo"] == 4.0

    def test_consts_non_numeric_leaves_preserved(self, real_config):
        assert isinstance(real_config.consts["AWG556"]["resource_address"], str)
        roi = real_config.consts["Orca"]["ROI"]
        assert isinstance(roi, list) and len(roi) == 4
        assert all(type(x) is float for x in roi)   # numeric list elements coerced too

    # -- default_vals keyed by TRANSLATED backend name ---------------------- #
    @pytest.mark.parametrize("alias,translated,value", [
        ("Amp2DMOT", "FPGA1/DDS21/AMP", 1.0),
        ("TTL556MOTaShutter", "FPGA1/TTL5", 1.0),
        ("VElectrode1", "NiDAQ/Dev1/12", 0.0),
        ("AmpAOM308", "FPGA1/DDS4/AMP", 0.0),
    ])
    def test_default_vals_keyed_by_translated_name(self, real_config, alias, translated, value):
        assert real_config.translate_channel(alias) == translated
        assert real_config.default_vals[translated] == value
        assert type(real_config.default_vals[translated]) is float
