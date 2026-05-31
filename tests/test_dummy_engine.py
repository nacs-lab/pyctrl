"""Phase 0: the board-free engine stand-in accepts externally-produced bytes.

This is the always-safe analogue of test_engine_loads.py: it runs under the
default `pytest` invocation on any interpreter (no libnacs, no Zynq board). It
proves the harness can hand a MATLAB-captured byte array to a Manager-shaped
object, get a compiled-sequence handle back, and that the run path is refused.

The real engine-accepts proof lives in test_engine_loads.py (marked
needs_engine); this guards the test infrastructure itself. It builds the dummy
directly (never the shared `engine` fixture) so the run-path assertions can
never reach a real engine, regardless of --real-engine.
"""

import os

import pytest

import compare_bytes
import dummy_libnacs
from conftest import all_reference_files

pytestmark = pytest.mark.no_hardware

_REFS = all_reference_files()


@pytest.mark.skipif(not _REFS, reason="no reference files found")
@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_dummy_accepts_reference_bytes(path):
    mgr = dummy_libnacs.Manager()
    data = bytearray(compare_bytes.load(path))
    eseq = mgr.create_sequence(data)
    assert eseq is not None
    assert eseq.get_builder_dump()           # non-empty -> "parsed" our bytes
    assert ("create_sequence", (len(data),)) in mgr.transcript


def test_dummy_rejects_corrupt_bytes():
    """A truncated/garbage blob must raise, mirroring the real engine."""
    mgr = dummy_libnacs.Manager()
    with pytest.raises(Exception):
        mgr.create_sequence(bytearray([0, 255, 255, 255, 255]))


def test_dummy_refuses_to_run():
    """The dummy never drives hardware: the run path raises loudly."""
    if not _REFS:
        pytest.skip("no reference files found")
    mgr = dummy_libnacs.Manager()
    eseq = mgr.create_sequence(bytearray(compare_bytes.load(_REFS[0])))
    with pytest.raises(NotImplementedError):
        eseq.start()
    with pytest.raises(NotImplementedError):
        eseq.init_run()
