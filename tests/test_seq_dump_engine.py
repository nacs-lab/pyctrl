"""Task-2 (Q-F) verification against the REAL libnacs engine (compile-only).

The engine-free tests (test_seq_dump.py) prove GlobalsCaptureSession's dedup / schema /
run-loop arming with a FAKE ``get_global``. This closes the one thing they can't: that the
capture reads a runtime global's **genuinely-injected value back out of the real engine
handle** via ``pyseq.get_global`` -- the exact path the 616-EOM ``freq616global`` rides.

It mirrors the real pattern (YbSeqs/PushoutSurvivalSeq): ``g = s.new_global()`` used as a
``FreqEOM616`` output, then a ``before_start`` callback injects the "from" frequency. Here we
``generate()`` (compile-only: create_sequence + reset_globals -- NO init_run/start, so NO
hardware), simulate the injection with ``set_global`` (what register_eom616_persistence's
``_pre`` does), then run the capture and confirm globals.json carries the injected value.

Run in a maintenance window (MATLAB off):
    .venv-engine\\Scripts\\python -m pytest tests/test_seq_dump_engine.py -m needs_engine --real-engine
"""

import json
import os

import pytest

import seq_manager
from conftest import CONFIG_PATH
from exp_seq import ExpSeq
from seq_config import SeqConfig
from seq_dump import GlobalsCaptureSession

pytestmark = pytest.mark.needs_engine


@pytest.fixture(scope="module")
def engine_cfg():
    if not seq_manager.engine_available():
        pytest.skip("libnacs engine not importable in this interpreter")
    mgr = seq_manager.get()
    with open(CONFIG_PATH) as f:
        mgr.load_config_string(f.read())
    mgr.enable_dump(True)
    SeqConfig.reset()
    SeqConfig.load_real()
    yield mgr
    SeqConfig.reset()
    seq_manager.reset()


def _build_seq_with_global(target=3.21e8):
    """A minimal real seq that uses a runtime global as a channel output (the
    freq616global pattern, stripped to one global + one DDS-freq channel)."""
    s = ExpSeq()
    g = s.new_global()                     # non-persist float64 global, init 0
    s.add('FreqEOM616', g)                 # USE the global as a real output value
    s.add_step(1e-3).add('FreqEOM616', target)
    return s, g


def test_capture_reads_injected_value_from_real_engine(engine_cfg, tmp_path):
    s, g = _build_seq_with_global()
    s.generate()                           # compile-only: real engine handle, no hardware

    INJECTED = 3.07e8                       # what a before_start callback would inject
    s.set_global(g, INJECTED)              # == register_eom616_persistence._pre
    # Pre-check: the real engine returns exactly what we injected (the crux).
    assert s.get_global(g) == INJECTED

    seq_dir = str(tmp_path / "sequence")
    sess = GlobalsCaptureSession(seq_dir, scan_id="20250619170317", seq_name="probe")
    sess.on_globals(7, 42, s)              # arg0=7, seqid=42
    doc = sess.finalize()

    assert doc is not None
    on_disk = json.load(open(os.path.join(seq_dir, "globals.json")))
    entries = on_disk["globals"]["42"]
    assert len(entries) == 1
    assert entries[0]["value"] == INJECTED          # captured == injected (via real get_global)
    assert entries[0]["persist"] is False
    assert entries[0]["id"] == int(g.args[0])


def test_capture_reads_init_value_when_not_injected(engine_cfg, tmp_path):
    """With no injection, the capture reads the global's reset (init) value -- confirming the
    read reflects real engine state, not a stale Python-side copy."""
    s, g = _build_seq_with_global()
    s.generate()                           # reset_globals(first=True) sets init_val (0.0)

    seq_dir = str(tmp_path / "sequence")
    sess = GlobalsCaptureSession(seq_dir, scan_id="20250619170317", seq_name="probe")
    sess.on_globals(1, 1, s)
    doc = sess.finalize()

    assert doc is not None
    assert doc["globals"]["1"][0]["value"] == 0.0
