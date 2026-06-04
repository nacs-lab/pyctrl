"""LIVE randomized differential: Python SeqContext vs a freshly-spawned MATLAB.

This is the opt-in companion to test_differential_fuzz.py. Where that test compares
against a *committed frozen* MATLAB reference (always-safe, no MATLAB needed), this
one generates a BRAND-NEW random corpus on every run and verifies it against ground
truth captured from a headless ``matlab -batch`` right now -- so it explores value
trees the frozen corpus never contained.

Marked ``needs_matlab`` and DESELECTED by default; run it explicitly:

    pytest -m needs_matlab pyctrl

It is engine-free and hardware-free (touches only SeqVal/SeqContext via
tools/capture_fuzz_reference.m) and spawns its OWN MATLAB process, so it does not
disturb a live experiment's MATLAB session. It needs a MATLAB on PATH, or the
``MATLAB_EXE`` env var pointing at the executable; it skips cleanly if neither is
present. Knobs (env): ``FUZZ_N`` corpus size (default 25), ``FUZZ_SEED`` to
reproduce a specific run (otherwise a random seed is chosen and printed on failure).

    MATLAB_EXE="C:\\Program Files\\MATLAB\\R2024b\\bin\\matlab.exe" \\
        FUZZ_SEED=12345 pytest -m needs_matlab pyctrl/tests/test_differential_fuzz_live.py
"""

import json
import os
import random
import shutil
import subprocess
import tempfile

import pytest

import fuzz_programs
from conftest import _PYCTRL_DIR

pytestmark = pytest.mark.needs_matlab

_TOOLS_DIR = os.path.join(_PYCTRL_DIR, "tools")
_CAPTURE_TIMEOUT_S = 600


def _matlab_exe():
    """Locate the MATLAB launcher: $MATLAB_EXE, else 'matlab' on PATH, else None."""
    env = os.environ.get("MATLAB_EXE")
    if env and os.path.exists(env):
        return env
    return shutil.which("matlab")


def _fwd(path):
    # MATLAB accepts forward slashes on Windows and they need no escaping inside a
    # single-quoted char array in the -batch statement.
    return path.replace("\\", "/")


def _run_matlab_capture(programs, workdir):
    """Write the corpus, spawn matlab -batch to capture tables, return parsed JSON."""
    progs_path = os.path.join(workdir, "programs.json")
    out_path = os.path.join(workdir, "fuzz_reference.json")
    with open(progs_path, "w") as f:
        json.dump(programs, f)

    exe = _matlab_exe()
    stmt = "addpath('%s'); capture_fuzz_reference('%s','%s')" % (
        _fwd(_TOOLS_DIR), _fwd(progs_path), _fwd(out_path))
    proc = subprocess.run([exe, "-batch", stmt], cwd=workdir, timeout=_CAPTURE_TIMEOUT_S,
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("matlab -batch capture failed (rc=%d)\nSTDOUT:\n%s\nSTDERR:\n%s"
                           % (proc.returncode, proc.stdout, proc.stderr))
    with open(out_path) as f:
        return json.load(f)


def test_live_matlab_differential():
    if _matlab_exe() is None:
        pytest.skip("no MATLAB on PATH and $MATLAB_EXE unset")

    n = int(os.environ.get("FUZZ_N", "25"))
    seed = int(os.environ.get("FUZZ_SEED", str(random.randrange(1 << 31))))
    programs = fuzz_programs.generate_programs(seed=seed, n=n)

    with tempfile.TemporaryDirectory(prefix="pyctrl_fuzz_") as workdir:
        reference = _run_matlab_capture(programs, workdir)

    assert len(reference) == len(programs), (
        "MATLAB returned %d tables for %d programs (seed=%d)"
        % (len(reference), len(programs), seed))

    mismatches = []
    for i, (program, expected) in enumerate(zip(programs, reference)):
        tables = fuzz_programs.build_python(program)
        for key in ("node", "data", "global"):
            if tables[key] != expected[key]:
                mismatches.append(
                    "prog %d %s:\n    py : %s\n    mat: %s"
                    % (i, key, tables[key], expected[key]))

    assert not mismatches, (
        "live MATLAB differential FAILED for seed=%d, n=%d "
        "(reproduce with FUZZ_SEED=%d FUZZ_N=%d):\n%s"
        % (seed, n, seed, n, "\n".join(mismatches[:6])))
