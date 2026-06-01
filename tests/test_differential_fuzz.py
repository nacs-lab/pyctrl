"""Randomized differential byte-equality: Python SeqContext vs live MATLAB.

The plan (PYTHON_FRONTEND_PLAN.md, Phase 1) calls for a randomized test that builds
the same value trees in *both* MATLAB and Python and asserts byte equality -- the one
test that catches folding-order / interning / DFS-id divergences the hand-written
cases miss.

How it works (NO-HARDWARE):
  * tools/fuzz_programs.py deterministically generates a corpus of language-neutral
    "program" specs -> tests/reference_fuzz/programs.json (committed).
  * tools/capture_fuzz_reference.m replays that SAME corpus in MATLAB with the real
    SeqVal/SeqContext and writes the node/data/global tables as hex ->
    tests/reference_fuzz/fuzz_reference.json (committed ground truth).
  * Here we rebuild every program in Python and assert each of the three tables
    equals the MATLAB hex, byte for byte.

This test loads only the committed JSON (no MATLAB, no engine), so it runs in the
default safe suite. To refresh the corpus + ground truth (e.g. after extending the
op set), regenerate and re-capture in a separate MATLAB session:

    python tools/fuzz_programs.py gen tests/reference_fuzz/programs.json
    matlab -batch "cd tools; capture_fuzz_reference"
"""

import json
import os

import pytest

import fuzz_programs
from conftest import _TESTS_DIR

pytestmark = pytest.mark.no_hardware

_FUZZ_DIR = os.path.join(_TESTS_DIR, "reference_fuzz")
_PROGRAMS = os.path.join(_FUZZ_DIR, "programs.json")
_REFERENCE = os.path.join(_FUZZ_DIR, "fuzz_reference.json")


def _load():
    if not (os.path.exists(_PROGRAMS) and os.path.exists(_REFERENCE)):
        return None
    with open(_PROGRAMS) as f:
        programs = json.load(f)
    with open(_REFERENCE) as f:
        reference = json.load(f)
    return programs, reference


_LOADED = _load()


@pytest.mark.skipif(_LOADED is None,
                    reason="fuzz corpus / MATLAB reference not captured "
                           "(run tools/fuzz_programs.py gen + capture_fuzz_reference.m)")
def test_corpus_is_paired():
    programs, reference = _LOADED
    assert len(programs) == len(reference) >= 1
    assert len(programs) >= 30, "expected a non-trivial corpus"


def _cases():
    if _LOADED is None:
        return []
    programs, reference = _LOADED
    return list(enumerate(zip(programs, reference)))


_CASES = _cases()


@pytest.mark.skipif(_LOADED is None, reason="fuzz corpus / MATLAB reference not captured")
@pytest.mark.parametrize("idx, pair", _CASES,
                         ids=["prog%02d" % i for i, _ in _CASES] or None)
def test_program_matches_matlab(idx, pair):
    program, expected = pair
    tables = fuzz_programs.build_python(program)
    for key in ("node", "data", "global"):
        assert tables[key] == expected[key], (
            "program %d: %s table differs from MATLAB\n  py : %s\n  mat: %s"
            % (idx, key, tables[key], expected[key]))


@pytest.mark.skipif(_LOADED is None, reason="fuzz corpus / MATLAB reference not captured")
def test_op_coverage():
    """The corpus must actually exercise a broad operation surface."""
    programs, _ = _LOADED
    ops = {s["op"] for p in programs for s in p["steps"]}
    # a representative spread across arithmetic, comparison, logical, transcendental,
    # and the ternary forms -- not an exhaustive list, just a floor.
    for must in ("add", "sub", "mul", "div", "pow", "interp", "ifelse",
                 "max", "min", "atan2", "hypot", "xor", "neg"):
        assert must in ops, "fuzz corpus never exercises %r" % must
    assert len(ops) >= 25, "fuzz corpus op surface too narrow: %d" % len(ops)
