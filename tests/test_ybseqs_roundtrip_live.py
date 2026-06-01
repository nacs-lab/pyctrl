"""LIVE reader faithfulness against REAL experiment sequences.

Spawns a headless ``matlab -batch`` that builds every sequence in
``matlab_new/YbSeqs`` engine-free (tools/capture_ybseqs_reference.m) and serializes
it, then asserts the pyctrl byte-format reader round-trips each blob byte-for-byte
(decode -> encode == original) and decodes to a well-formed structure.

This is NOT a Python-rebuild byte-equality test -- pyctrl cannot build a full
ExpSeq until Phase 2. It proves the *reader* (the byte-format spec it encodes) is
faithful to real production sequences, which carry far more structure than the
synthetic references: ~50 channels, 90-130 value nodes, branches, and measures.

Marked ``needs_matlab`` and DESELECTED by default; run it explicitly:

    MATLAB_EXE="C:\\Program Files\\MATLAB\\R2023a\\bin\\matlab.exe" \\
        pytest -m needs_matlab pyctrl/tests/test_ybseqs_roundtrip_live.py

SAFETY: the capture only BUILDS + serialize()s (it never runs the deferred
camera/AWG/server callbacks), runs in its OWN headless process, and skips
sequences that drive AWG/SLM/AOD/camera hardware in-body. It is engine-free. Still
prefer a maintenance window -- it executes real sequence-building code.
"""

import json
import os
import shutil
import subprocess
import tempfile

import pytest

import compare_bytes
from conftest import _PYCTRL_DIR

pytestmark = pytest.mark.needs_matlab

_TOOLS_DIR = os.path.join(_PYCTRL_DIR, "tools")
_CAPTURE_TIMEOUT_S = 600
_MIN_OK = 8           # floor: catch a catastrophic build/serialize regression


def _matlab_exe():
    env = os.environ.get("MATLAB_EXE")
    if env and os.path.exists(env):
        return env
    return shutil.which("matlab")


def _fwd(path):
    return path.replace("\\", "/")


def _run_capture(workdir):
    out_path = os.path.join(workdir, "ybseqs_reference.json")
    exe = _matlab_exe()
    stmt = "addpath('%s'); capture_ybseqs_reference('', '%s')" % (
        _fwd(_TOOLS_DIR), _fwd(out_path))
    proc = subprocess.run([exe, "-batch", stmt], cwd=workdir, timeout=_CAPTURE_TIMEOUT_S,
                          capture_output=True, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError("matlab capture produced no output (rc=%d)\nSTDOUT:\n%s\nSTDERR:\n%s"
                           % (proc.returncode, proc.stdout, proc.stderr))
    with open(out_path) as f:
        return json.load(f)


def test_real_ybseqs_round_trip_live():
    if _matlab_exe() is None:
        pytest.skip("no MATLAB on PATH and $MATLAB_EXE unset")

    with tempfile.TemporaryDirectory(prefix="ybctrl_ybseqs_") as workdir:
        entries = _run_capture(workdir)

    built = [e for e in entries if e["status"] == "ok"]
    assert len(built) >= _MIN_OK, (
        "only %d/%d YbSeqs serialized (floor %d) -- a build/serialize regression? "
        "statuses: %s" % (len(built), len(entries), _MIN_OK,
                          {e["name"]: e["status"] for e in entries if e["status"] != "ok"}))

    failures = []
    for e in built:
        raw = bytes.fromhex(e["bytes"])
        try:
            seq = compare_bytes.decode(raw)               # raises on malformed/trailing
            if compare_bytes.encode(seq) != raw:
                failures.append("%s: re-encode != original" % e["name"])
                continue
            assert seq["version"] == 0
            assert len(seq["basicseqs"]) >= 1
        except Exception as exc:                          # noqa: BLE001
            failures.append("%s: %s" % (e["name"], exc))

    assert not failures, "reader failed on real sequences:\n  " + "\n  ".join(failures)
