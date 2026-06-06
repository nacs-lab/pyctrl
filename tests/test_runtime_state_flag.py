"""The runtime_state mmap store carries the 'save sequence dumps' toggle (offset 8)
alongside the 616-EOM freq (offset 0), and migrates a legacy 8-byte file losslessly.

    python -m pytest pyctrl/tests/test_runtime_state_flag.py -v
"""

import struct

import pytest

import runtime_state


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Point runtime_state at a throwaway file and reset its cached handle."""
    # Close any handle a prior test/use opened, then redirect + reset the cache.
    if runtime_state._mm is not None:
        try:
            runtime_state._mm.close()
            runtime_state._fh.close()
        except Exception:
            pass
    monkeypatch.setattr(runtime_state, "_PATH", str(tmp_path / "rs.dat"))
    monkeypatch.setattr(runtime_state, "_mm", None)
    monkeypatch.setattr(runtime_state, "_fh", None)
    yield
    if runtime_state._mm is not None:
        try:
            runtime_state._mm.close()
            runtime_state._fh.close()
        except Exception:
            pass


def test_flag_defaults_off(fresh_store):
    assert runtime_state.get_save_sequence_dumps() is False
    assert runtime_state.get_save_sequence_dumps(default=True) is False  # fresh file -> 0


def test_flag_set_and_clear(fresh_store):
    runtime_state.set_save_sequence_dumps(True)
    assert runtime_state.get_save_sequence_dumps() is True
    runtime_state.set_save_sequence_dumps(False)
    assert runtime_state.get_save_sequence_dumps() is False


def test_flag_independent_of_eom616(fresh_store):
    runtime_state.set_eom616_old(123456.0)
    runtime_state.set_save_sequence_dumps(True)
    assert runtime_state.get_eom616_old(default=0.0) == 123456.0
    assert runtime_state.get_save_sequence_dumps() is True


def test_legacy_8byte_file_migrates_losslessly(tmp_path, monkeypatch):
    # A pre-toggle store is exactly 8 bytes (one float64).
    path = tmp_path / "legacy.dat"
    path.write_bytes(struct.pack("<d", 222333.0))
    if runtime_state._mm is not None:
        try:
            runtime_state._mm.close(); runtime_state._fh.close()
        except Exception:
            pass
    monkeypatch.setattr(runtime_state, "_PATH", str(path))
    monkeypatch.setattr(runtime_state, "_mm", None)
    monkeypatch.setattr(runtime_state, "_fh", None)

    # First access migrates 8 -> 9 bytes, preserving the freq and defaulting the flag off.
    assert runtime_state.get_eom616_old(default=0.0) == 222333.0
    assert runtime_state.get_save_sequence_dumps() is False
    assert path.stat().st_size == 9
    try:
        runtime_state._mm.close(); runtime_state._fh.close()
    except Exception:
        pass
