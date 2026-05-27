"""Shared pytest fixtures for the pyctrl test suite.

Reference outputs come from two places:
  * matlab_new/lib/test/seq*.json -- committed MATLAB serialize() output,
    captured engine-free (SeqManager.override_tick_per_sec). Used by the
    byte round-trip tests. No MATLAB run required.
  * pyctrl/tests/reference/*.bin   -- references captured for real-config
    sequences by tools/capture_matlab_reference.m (used by the engine check).
"""

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYCTRL_DIR = os.path.dirname(_TESTS_DIR)
REPO_ROOT = os.path.dirname(_PYCTRL_DIR)

# Make the flat lib/ and tools/ importable (mirrors pyproject pythonpath, so the
# suite also runs without an editable install).
for _p in (os.path.join(_PYCTRL_DIR, "lib"), os.path.join(_PYCTRL_DIR, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MATLAB_REF_DIR = os.path.join(REPO_ROOT, "matlab_new", "lib", "test")
PYCTRL_REF_DIR = os.path.join(_TESTS_DIR, "reference")
ENGINE_REF_DIR = os.path.join(_TESTS_DIR, "reference_engine")


def matlab_reference_files():
    """Existing MATLAB serialize() references (seq*.json)."""
    if not os.path.isdir(MATLAB_REF_DIR):
        return []
    return [os.path.join(MATLAB_REF_DIR, f)
            for f in sorted(os.listdir(MATLAB_REF_DIR))
            if f.startswith("seq") and f.endswith(".json")]


def pyctrl_reference_files():
    """Real-config references captured by capture_matlab_reference.m (*.bin)."""
    if not os.path.isdir(PYCTRL_REF_DIR):
        return []
    return [os.path.join(PYCTRL_REF_DIR, f)
            for f in sorted(os.listdir(PYCTRL_REF_DIR))
            if f.endswith(".bin")]


def engine_reference_files():
    """Real-config references for the engine-accepts check (*.bin)."""
    if not os.path.isdir(ENGINE_REF_DIR):
        return []
    return [os.path.join(ENGINE_REF_DIR, f)
            for f in sorted(os.listdir(ENGINE_REF_DIR))
            if f.endswith(".bin")]


def all_reference_files():
    return matlab_reference_files() + pyctrl_reference_files()


@pytest.fixture
def repo_root():
    return REPO_ROOT
