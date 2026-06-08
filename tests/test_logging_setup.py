"""logging_setup: the terminal-mirror tee + root-logger file handlers.

NO-HARDWARE: exercises the _LineTee wrapper directly (copy-not-replace + per-line timestamps),
and drives setup_logging() with the log dir at tmp_path to assert it installs the tee on the
std streams, splits INFO vs DEBUG into the right files, and is idempotent / fail-safe. The tee
content is tested on _LineTee directly rather than through ``print`` so pytest's own stdout
capture doesn't interfere.
"""

import io
import logging
import os
import threading

import pytest

import logging_setup
from logging_setup import _LineTee, setup_logging

pytestmark = pytest.mark.no_hardware


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --- _LineTee unit tests (no global state) -----------------------------------

def test_tee_copies_verbatim_to_stream_and_stamps_file(tmp_path):
    stream = io.StringIO()
    fpath = tmp_path / "mirror.log"
    with open(fpath, "w", encoding="utf-8") as fh:
        tee = _LineTee(stream, fh, threading.Lock())
        tee.write("hello world\n")
    # Terminal side: byte-for-byte, no timestamp.
    assert stream.getvalue() == "hello world\n"
    # File side: same text, timestamp-prefixed.
    mirror = _read(fpath)
    assert mirror.rstrip().endswith("hello world")
    assert mirror != "hello world\n"  # a stamp was prepended


def test_tee_partial_line_gets_single_stamp(tmp_path):
    stream = io.StringIO()
    fpath = tmp_path / "mirror.log"
    with open(fpath, "w", encoding="utf-8") as fh:
        tee = _LineTee(stream, fh, threading.Lock())
        tee.write("abc")     # no newline yet
        tee.write("def\n")   # completes the line
    assert stream.getvalue() == "abcdef\n"
    mirror = _read(fpath)
    assert "abcdef" in mirror
    assert mirror.count("abcdef") == 1  # exactly one logical line -> one stamp


def test_tee_delegates_attrs_to_real_stream():
    stream = io.StringIO()
    tee = _LineTee(stream, io.StringIO(), threading.Lock())
    # isatty/etc. pass through to the wrapped stream.
    assert tee.isatty() == stream.isatty()


# --- setup_logging integration ----------------------------------------------

@pytest.fixture
def clean_logging(monkeypatch, tmp_path):
    """Isolate global state: reset the install flag, capture std streams, restore everything."""
    monkeypatch.setenv("YB_PYCTRL_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("YB_PYCTRL_LOG", raising=False)
    monkeypatch.setattr(logging_setup, "_INSTALLED", False)
    # monkeypatch records/restores the originals even though setup_logging reassigns them.
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield tmp_path
    for h in root.handlers[:]:
        if h not in saved_handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved_level)


def test_setup_installs_tee_and_writes_through(clean_logging):
    import sys
    paths = setup_logging()
    assert paths is not None
    assert isinstance(sys.stdout, _LineTee)
    assert isinstance(sys.stderr, _LineTee)

    sys.stdout.write("from-stdout\n")
    sys.stderr.write("from-stderr\n")
    # Both streams share one mirror file (stdout + stderr interleaved there); the verbatim
    # terminal-side copy is covered by the _LineTee unit tests above (pytest re-grabs the real
    # sys.stdout, so we don't assert on its type here).
    mirror = _read(paths["mirror"])
    assert "from-stdout" in mirror
    assert "from-stderr" in mirror


def test_logging_info_and_debug_split(clean_logging):
    paths = setup_logging()
    logging.getLogger("pyctrl.devices").info("dev-info")
    logging.getLogger("pyctrl.x").debug("dbg-only")
    logging.getLogger("pyctrl.y").warning("warn-line")
    for h in logging.getLogger().handlers:
        h.flush()

    info_txt = _read(paths["logging"])
    debug_txt = _read(paths["debug"])
    assert "dev-info" in info_txt
    assert "warn-line" in info_txt        # WARNING >= INFO
    assert "dbg-only" not in info_txt     # DEBUG excluded from the INFO file
    assert "dbg-only" in debug_txt
    assert "dev-info" not in debug_txt     # DEBUG file is DEBUG-only (no dup)


def test_idempotent(clean_logging):
    assert setup_logging() is not None
    assert setup_logging() is None  # second call is a no-op


def test_disabled_via_env(clean_logging, monkeypatch):
    monkeypatch.setenv("YB_PYCTRL_LOG", "0")
    assert setup_logging() is None


def test_unwritable_dir_returns_none(clean_logging, monkeypatch):
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert setup_logging() is None
