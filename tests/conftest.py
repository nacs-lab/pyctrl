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

# Make the flat lib/ + tools/ + the matlab_new-mirroring YbSteps/ YbSeqs/ importable
# (mirrors pyproject pythonpath, so the suite also runs without an editable install).
for _p in (os.path.join(_PYCTRL_DIR, "lib"), os.path.join(_PYCTRL_DIR, "tools"),
           os.path.join(_PYCTRL_DIR, "YbSteps"), os.path.join(_PYCTRL_DIR, "YbSeqs")):
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


# --------------------------------------------------------------------------- #
# Engine selection: the board-free dummy by default, the real libnacs engine
# only when --real-engine is passed (and only importable under an interpreter
# that has the libnacs build -- see README). This lets byte-equality / harness
# tests run anywhere, while the engine-accepts proof opts in explicitly.
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--real-engine", action="store_true", default=False,
        help="use the real libnacs engine instead of tools/dummy_libnacs "
             "(requires an interpreter with libnacs installed; compile-only)")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """On the real-engine path, force a clean exit past a broken C++ teardown.

    The libnacs engine bundles its own libzmq for device comms. When the engine
    is loaded but never started (our compile-only checks never call new_run /
    init_run), libzmq's network layer is never initialised, so its static
    destructor asserts at process exit ("Successful WSASTARTUP not yet
    performed"). On Windows the CRT abort() that follows hangs the pytest
    process *after* the results are already printed -- and os._exit doesn't help
    because ExitProcess still runs the engine DLL's detach routine, which is
    where it wedges. TerminateProcess on our own handle skips DLL detach
    entirely, exiting immediately with pytest's real status (0 on success).
    trylast=True so the terminal summary is printed first. Only fires on Windows
    under --real-engine; the default (dummy) run exits normally.
    """
    if session.config.getoption("--real-engine") and sys.platform == "win32":
        import ctypes
        sys.stdout.flush()
        sys.stderr.flush()
        kernel32 = ctypes.windll.kernel32
        PROCESS_TERMINATE = 0x0001
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, os.getpid())
        kernel32.TerminateProcess(handle, int(exitstatus))


@pytest.fixture
def engine(request):
    """A Manager-shaped object: real libnacs if --real-engine, else the dummy.

    Config from matlab_new/config.yml is loaded into whichever is returned.
    """
    config_path = os.path.join(REPO_ROOT, "matlab_new", "config.yml")
    if request.config.getoption("--real-engine"):
        import seq_manager
        if not seq_manager.engine_available():
            pytest.skip("libnacs engine not importable in this interpreter")
        mgr = seq_manager.get()
    else:
        import dummy_libnacs
        mgr = dummy_libnacs.Manager()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            mgr.load_config_string(f.read())
    return mgr
