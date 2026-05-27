"""Phase 0: prove the engine accepts an externally-produced byte array.

This is the only Phase 0 test that loads the libnacs engine. It is marked
`needs_engine` and is DESELECTED by default, so the normal `pytest` run on the
shared lab PC never loads the engine. Run it explicitly in a safe window:

    pytest -m needs_engine

It compiles only (create_sequence) and never calls init_run / start, so it
does not drive hardware. It uses real-config references from
pyctrl/tests/reference/*.bin (captured by tools/capture_matlab_reference.m with
channel names that match config.yml); it skips if none have been captured yet.
The MATLAB lib/test seq*.json references use placeholder device names that the
real engine config does not know, so they are intentionally not used here.
"""

import os

import pytest

import compare_bytes
import seq_manager
from conftest import REPO_ROOT, engine_reference_files

pytestmark = pytest.mark.needs_engine

_REFS = engine_reference_files()


@pytest.fixture(scope="module")
def manager():
    if not seq_manager.engine_available():
        pytest.skip("libnacs engine not importable in this interpreter")
    mgr = seq_manager.get()
    config_path = os.path.join(REPO_ROOT, "matlab_new", "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            mgr.load_config_string(f.read())
    return mgr


@pytest.mark.skipif(not _REFS, reason="no real-config references captured yet "
                                       "(run tools/capture_matlab_reference.m)")
@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_engine_accepts_bytes(manager, path):
    data = bytearray(compare_bytes.load(path))
    eseq = manager.create_sequence(data)
    assert eseq is not None
    dump = eseq.get_builder_dump()
    assert dump  # non-empty -> the engine parsed and built our bytes


def test_corrupt_bytes_raise(manager):
    """A deliberately truncated byte array must be rejected by the engine."""
    if not _REFS:
        pytest.skip("no real-config references captured yet")
    data = bytearray(compare_bytes.load(_REFS[0]))[: max(1, len(_REFS[0]) // 2)]
    with pytest.raises(Exception):
        manager.create_sequence(bytearray(data) + bytearray([255, 255, 255, 255]))
