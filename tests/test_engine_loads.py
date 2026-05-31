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
    # The builder/seq dumps are only populated when dump is enabled; without
    # this get_builder_dump() returns None even for a valid sequence.
    mgr.enable_dump(True)
    return mgr


@pytest.mark.skipif(not _REFS, reason="no real-config references captured yet "
                                       "(run tools/capture_matlab_reference.m)")
@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_engine_accepts_bytes(manager, path):
    data = bytearray(compare_bytes.load(path))
    # The accept proof: create_sequence returns a non-null compiled handle and
    # does not raise (the binding's `guarded` wrapper raises on a SeqError). A
    # non-empty builder dump (dump enabled in the fixture) further confirms the
    # engine actually parsed and built our externally-produced bytes.
    eseq = manager.create_sequence(data)
    assert eseq is not None, "engine returned a null handle for %s" % path
    assert eseq.get_builder_dump()


def test_corrupt_bytes_raise(manager):
    """A deliberately corrupted byte array must be rejected by the engine.

    Flip the leading version byte (valid == 0) to an unknown value: the engine
    rejects it immediately ("Unknown sequence serialization version"). NB: do
    NOT truncate with an oversized count field -- the engine would try to read
    billions of records and hang rather than raise.
    """
    if not _REFS:
        pytest.skip("no real-config references captured yet")
    data = bytearray(compare_bytes.load(_REFS[0]))
    data[0] = 99  # bogus serialization version
    with pytest.raises(Exception):
        manager.create_sequence(data)
