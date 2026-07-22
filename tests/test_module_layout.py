"""test_module_layout.py -- naming-lock for the run-loop module split (YbExptCtrl restructure).

WHY: ``YbExptCtrl/runner.py`` (the monolithic host) was split into five flat-importable
modules -- ``run_loop`` (host orchestration), ``run_job`` (per-job orchestration, the old
``sequence_runner``), ``engine_run`` (per-scan engine + scan-prep), ``slm_runtime`` (loading
patterns / rearrangement), ``camera_runtime`` (Orca pump), ``awg_runtime`` (Siglent/QICK) --
and ``set_chns`` moved from ``YbExptCtrl/`` to ``lib/``. There are NO re-export shims: the old
flat names ``runner`` / ``sequence_runner`` are gone, and every reference in the tree has been
retargeted to the owning module.

These tests LOCK that structure so a regression (a resurrected ``runner.py``, an accidental
re-export shim, a symbol drifting back to the wrong module, or a leaf module importing "up" and
re-forming the monolith) fails loudly:
  1. each new module exports the public/tests-depended-on names it owns,
  2. the old flat module names stay gone (launcher's spawn module is the only ``runner`` left,
     reachable ONLY as ``launcher.run_loop.runner``),
  3. ``set_chns`` lives in ``lib/`` (not ``YbExptCtrl/``),
  4. the dependency DAG points one way -- leaf runtimes never import ``run_loop`` / ``engine_run``,
  5. no source file anywhere in the tree still imports the deleted flat modules.

NO-HARDWARE: pure import-spec + source-text inspection; nothing loads the engine or a device.
"""

import importlib
import importlib.util
import os
import re

import pytest

pytestmark = pytest.mark.no_hardware


# The owning module -> the names that module must export. Public API names plus the private
# helpers the test suite reaches for directly (so a rename/move that would silently break a
# test is caught here first, with a clear message).
_EXPECTED = {
    "run_loop": [
        "main", "serve", "resolve_url", "assert_single_backend", "consume_loop",
        "handle_descriptor_pop", "make_idle", "DEFAULT_URL",
        "_teardown", "_pid_alive", "_start_parent_watchdog", "_force_dummy_off",
    ],
    "run_job": [
        "run_job", "IdleScheduler", "JobResult", "_build_scan_order",
        "_build_run_kwargs", "_extract_code_snapshot", "_snapshot_replay_ctx",
    ],
    "engine_run": [
        "make_engine_run", "load_configs", "_write_scan_prep", "_new_scan_id",
        "_num_images", "_line_trigger_config", "_ttl_managers_config",
    ],
    "slm_runtime": [
        "_loading_patterns_json", "_first_loading_pattern", "_n_rounds",
        "DEFAULT_LOADING_PATTERN_PHASE",
    ],
    "camera_runtime": [
        "handle_camera_cmd", "open_camera", "sync_camera_exposure", "make_camera_pump",
    ],
    "awg_runtime": [
        "awg_names", "setup", "make_pre_cb", "cleanup",
    ],
}


@pytest.mark.parametrize("mod_name", sorted(_EXPECTED))
def test_new_modules_export_expected_names(mod_name):
    mod = importlib.import_module(mod_name)
    missing = [n for n in _EXPECTED[mod_name] if not hasattr(mod, n)]
    assert not missing, "%s is missing expected names: %s" % (mod_name, missing)


def test_old_module_names_gone():
    # No re-export shim, no resurrected file: the old flat modules must not resolve at all.
    assert importlib.util.find_spec("runner") is None
    assert importlib.util.find_spec("sequence_runner") is None
    # The launcher's spawn module keeps the basename runner.py, but it is reachable ONLY as the
    # dotted package path (its imports were already updated); that path must still import.
    assert importlib.util.find_spec("launcher.run_loop.runner") is not None


def test_set_chns_lives_in_lib():
    spec = importlib.util.find_spec("set_chns")
    assert spec is not None and spec.origin, "set_chns must be importable"
    origin = os.path.normcase(spec.origin)
    marker = os.path.normcase(os.sep + "lib" + os.sep)
    assert marker in origin, "set_chns should live under lib/, got %s" % spec.origin


def _module_source(mod_name):
    spec = importlib.util.find_spec(mod_name)
    assert spec is not None and spec.origin, "%s must be importable" % mod_name
    with open(spec.origin, "r", encoding="utf-8") as f:
        return f.read()


def test_leaf_modules_do_not_import_up():
    # Lock the dependency DAG direction: the leaf runtimes + the per-job orchestrator must not
    # import the host (run_loop) or the per-scan engine builder (engine_run); and engine_run
    # itself must not import the host. Any such edge would re-form the old monolith.
    up_re = re.compile(r"^\s*(?:import|from)\s+(engine_run|run_loop)\b")
    for leaf in ("camera_runtime", "slm_runtime", "awg_runtime", "run_job"):
        offenders = [ln for ln in _module_source(leaf).splitlines() if up_re.match(ln)]
        assert not offenders, "%s imports up the DAG: %s" % (leaf, offenders)
    # engine_run may import the leaves, but never the host.
    host_re = re.compile(r"^\s*(?:import|from)\s+run_loop\b")
    offenders = [ln for ln in _module_source("engine_run").splitlines() if host_re.match(ln)]
    assert not offenders, "engine_run imports the host run_loop: %s" % offenders


def test_no_stale_imports_in_source_tree():
    # Walk pyctrl's source dirs and assert nothing still imports the deleted flat modules.
    # The pattern is built by concatenation so THIS file never matches its own scan.
    dead = "runner" + "|" + "sequence_" + "runner"
    stale_re = re.compile(
        r"^\s*(?:from\s+(?:" + dead + r")\s+import|import\s+(?:" + dead + r")\b)")
    pyctrl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scan_dirs = ("YbExptCtrl", "lib", "tools", "launcher", "tests")
    hits = []
    for sub in scan_dirs:
        base = os.path.join(pyctrl_root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune caches and any virtualenv dirs in place.
            dirnames[:] = [d for d in dirnames
                           if d != "__pycache__" and not d.startswith(".venv")]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if stale_re.match(line):
                            hits.append("%s:%d: %s" % (path, i, line.rstrip()))
    assert not hits, "stale imports of the deleted flat modules:\n" + "\n".join(hits)
