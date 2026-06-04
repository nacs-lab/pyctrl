"""Phase-5 seq_reload: per-job hot-reload of ported experiment modules (rehash()+str2func analog).

NO-HARDWARE. Uses throwaway modules written to tmp dirs (added to sys.path) so we can edit a file
on disk and prove the next import re-reads it -- without touching the real YbSeqs. Verifies:
  * an experiment-dir module is dropped + re-read after an edit;
  * a module OUTSIDE the roots (the "lib" analog) is KEPT cached (not reloaded);
  * the reload is TRANSITIVE -- editing a step a seq imports is picked up (the whole point of
    invalidate-and-reimport vs a shallow importlib.reload);
  * a brand-new module added after the first import is discoverable.
"""

import os
import textwrap
import time

import pytest

from seq_reload import reload_experiment_modules

pytestmark = pytest.mark.no_hardware

# Force a strictly-increasing, far-future source mtime on every write so Python's mtime+size
# .pyc validation always recompiles on re-import -- deterministic regardless of filesystem
# timestamp resolution. (In real use a human-saved edit advances the mtime naturally, which is
# the same standard invalidation dev auto-reloaders rely on; the test just makes it explicit so
# a same-tick / same-size rewrite can't serve stale bytecode.)
_mtime = [time.time() + 1000.0]


def _write(path, body):
    path.write_text(textwrap.dedent(body))
    _mtime[0] += 10.0
    os.utime(path, (_mtime[0], _mtime[0]))


def test_edit_picked_up_and_lib_kept(tmp_path, monkeypatch):
    exp = tmp_path / "ExpDir"
    lib = tmp_path / "LibDir"
    exp.mkdir()
    lib.mkdir()
    _write(exp / "demo_seq.py", "VALUE = 1\n")
    _write(lib / "framework_mod.py", "VALUE = 100\n")
    monkeypatch.syspath_prepend(str(lib))
    monkeypatch.syspath_prepend(str(exp))

    import demo_seq            # noqa: F401 - imported for cache population
    import framework_mod
    assert demo_seq.VALUE == 1 and framework_mod.VALUE == 100
    fw_id = id(framework_mod)

    # Edit BOTH files on disk, then hot-reload only the experiment root.
    _write(exp / "demo_seq.py", "VALUE = 2\n")
    _write(lib / "framework_mod.py", "VALUE = 200\n")
    dropped = reload_experiment_modules(roots=[str(exp)])

    assert "demo_seq" in dropped and "framework_mod" not in dropped
    import demo_seq as demo2
    import framework_mod as fw2
    assert demo2.VALUE == 2           # experiment edit picked up
    assert fw2.VALUE == 100           # lib analog NOT reloaded (still cached)
    assert id(fw2) == fw_id           # ...same module object


def test_reload_is_transitive_over_imported_steps(tmp_path, monkeypatch):
    exp = tmp_path / "Exp2"
    exp.mkdir()
    _write(exp / "dep_step.py", "STEP_VAL = 1\n")
    _write(exp / "top_seq.py", "from dep_step import STEP_VAL\n\ndef val():\n    return STEP_VAL\n")
    monkeypatch.syspath_prepend(str(exp))

    import top_seq
    assert top_seq.val() == 1

    # Edit the STEP the seq imports (not the seq itself). A shallow importlib.reload(top_seq)
    # would miss this; invalidate-and-reimport picks it up.
    _write(exp / "dep_step.py", "STEP_VAL = 9\n")
    reload_experiment_modules(roots=[str(exp)])

    import top_seq as t2
    assert t2.val() == 9


def test_new_module_discoverable_after_invalidate(tmp_path, monkeypatch):
    exp = tmp_path / "Exp3"
    exp.mkdir()
    _write(exp / "first.py", "X = 1\n")
    monkeypatch.syspath_prepend(str(exp))
    import first            # noqa: F401
    # A file created after the first import: invalidate_caches() (inside reload) makes it findable.
    _write(exp / "second.py", "Y = 2\n")
    reload_experiment_modules(roots=[str(exp)])
    import second
    assert second.Y == 2


def test_no_dropped_when_nothing_under_roots(tmp_path):
    # An empty/unused root drops nothing and never raises.
    assert reload_experiment_modules(roots=[str(tmp_path / "nope")]) == []
