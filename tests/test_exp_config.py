"""Phase-5 expConfig.py: executable config drift oracle + hot-reload.

The executable ``expConfig.py`` is pyctrl's production config source AND (since 2026-06-05, the
gradual MATLAB -> pyctrl switch) the human-edited SOURCE OF TRUTH for the front-end config. The
committed snapshot ``tests/reference/config_reference.json`` is a frozen copy of
``expConfig.build_config()`` (regenerated engine-free by ``tools/capture_config_reference.py``).
This test is the DRIFT ORACLE: an accidental/unintended edit to ``expConfig.py`` that is not a
deliberate recalibration fails loudly until the snapshot is re-captured on purpose -- converting
the old SILENT snapshot staleness (bug-pyctrl-config-snapshot-staleness) into a caught error. NO
MATLAB needed at test time or to regenerate the snapshot.

(THE ONE RULE byte-equality vs MATLAB is enforced separately by the byte oracles --
``ybseqs_reference.json`` / ``scan_point_reference.json`` -- which stay MATLAB-captured. A
recalibration still re-captures those from MATLAB; this config oracle no longer depends on MATLAB.)

Also checks that ``SeqConfig.load_real()`` now sources the executable config, and that the
per-job hot-reload updates in place while preserving runtime globals (``SeqConfig.G``).
"""

import json
import os

import pytest

import expConfig
from conftest import _TESTS_DIR
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_SNAP = os.path.join(_TESTS_DIR, "reference", "config_reference.json")


def _snapshot_config():
    with open(_SNAP) as f:
        return SeqConfig(json.load(f))


# --------------------------------------------------------------------------- #
# drift oracle: expConfig.py resolves identically to the committed snapshot
# (the snapshot is a frozen capture of expConfig.py itself -- the source of truth)
# --------------------------------------------------------------------------- #
def test_exec_config_matches_committed_snapshot():
    snap = _snapshot_config()
    exe = SeqConfig(expConfig.build_config())
    # Compare each resolved table; pinpoint the first differing key for a useful failure.
    assert exe.channel_alias == snap.channel_alias, _diff(exe.channel_alias, snap.channel_alias)
    assert exe.default_vals == snap.default_vals, _diff(exe.default_vals, snap.default_vals)
    assert exe.ni_clocks == snap.ni_clocks
    assert exe.ni_start == snap.ni_start
    assert exe.consts == snap.consts, _const_diff(exe.consts, snap.consts)


def test_cross_references_resolve():
    c = SeqConfig(expConfig.build_config()).consts
    # The three const-to-const links in expConfig.m must resolve, not stay None.
    assert c["SLM"]["VServo"] == c["Init"]["VSLMServo"]
    assert c["Imag399"]["ExposureTime"] == c["Orca"]["ExposureTime"]
    assert c["LAC"]["BlueLAC"]["Resonance556mj0Freq"] == c["Resonance556mj0Freq"]


# --------------------------------------------------------------------------- #
# load_real now sources the executable config; reload preserves G
# --------------------------------------------------------------------------- #
def test_load_real_uses_exec_config():
    SeqConfig.reset()
    try:
        cfg = SeqConfig.load_real()           # default source = expConfig.py
        snap = _snapshot_config()
        assert cfg.consts == snap.consts
        assert cfg.channel_alias == snap.channel_alias
        assert cfg.default_vals == snap.default_vals
    finally:
        SeqConfig.reset()


def test_hot_reload_in_place_preserves_globals():
    SeqConfig.reset()
    try:
        cfg = SeqConfig.load_real()
        cfg.G.MyRuntimeGlobal = 42            # a sequence-internal runtime global
        cfg2 = SeqConfig.load_real(reload=True)
        assert cfg2 is cfg                    # in-place update keeps the singleton identity
        assert cfg2.G.MyRuntimeGlobal(0) == 42   # runtime globals survive the config reload
        assert cfg2.consts == _snapshot_config().consts   # ...and the config is (re)applied
    finally:
        SeqConfig.reset()


def test_json_path_still_supported():
    # The drift oracle / back-compat path: an explicit snapshot path bypasses expConfig.py.
    SeqConfig.reset()
    try:
        cfg = SeqConfig.load_real(config_path=_SNAP)
        assert cfg.consts == _snapshot_config().consts
    finally:
        SeqConfig.reset()


# --------------------------------------------------------------------------- #
# helpers (first-diff reporters for readable failures)
# --------------------------------------------------------------------------- #
def _diff(a, b):
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            return "first diff at %r: exec=%r snapshot=%r" % (k, a.get(k), b.get(k))
    return "dicts differ but no scalar key diff found"


def _const_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return "%s.%s missing from exec config" % (path, k)
            if k not in b:
                return "%s.%s missing from snapshot" % (path, k)
            d = _const_diff(a[k], b[k], "%s.%s" % (path, k))
            if d:
                return d
        return ""
    if a != b:
        return "const %s: exec=%r snapshot=%r" % (path, a, b)
    return ""
