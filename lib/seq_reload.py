"""seq_reload.py -- per-job hot-reload of ported experiment modules (rehash()+str2func analog).

The long-lived pyctrl runner (``YbExptCtrl/runner.py``) imports a seq the first time a job needs
it, and Python CACHES the module in ``sys.modules``. Without intervention, editing a ported
seq/step file would NOT take effect until the runner restarts. MATLAB's runner does
``rehash()`` + ``str2func`` every job, so ``.m`` edits go live immediately on the next job;
:func:`reload_experiment_modules` gives pyctrl the same behavior -- no restart, no button.

Approach (NOT ``importlib.reload``): drop from ``sys.modules`` every module loaded from the
experiment directories (``YbSteps`` / ``YbSeqs`` / ``YbScans`` / ``YbRearrangement``). The NEXT
import (the dispatcher's import-by-convention resolver) re-executes the seq AND transitively
re-imports its (now-also-dropped) experiment dependencies in correct order -- so an edit to a STEP
a seq imports is picked up too. A shallow ``importlib.reload`` of just the seq module would miss
that (it re-binds ``from dep_step import X`` to the still-cached ``dep_step``), and is also
identity-unsafe.

``lib/`` is intentionally KEPT: it is the stable framework (the ``SeqConfig`` singleton, the
``ExpSeq`` / ``ScanGroup`` base classes). Reloading it would create NEW class objects -- breaking
``isinstance`` against live instances and duplicating singletons -- so edits to ``lib/`` (and the
expConfig snapshot, loaded once at startup) still require a runner restart. The run loop,
``ExptServer``, and the dispatcher (``YbExptCtrl/``) are likewise never reloaded.

Call it ONCE per job, before the dispatcher resolves anything (``run_job`` does this when handed
the reloader). Best-effort: it never raises into the run loop (a failure just means stale modules,
i.e. the pre-reload behavior).

Relies on Python's standard source-mtime bytecode invalidation: a saved edit advances the file
mtime, so the fresh re-import recompiles and picks up the change (the same mechanism dev
auto-reloaders use). A sub-resolution, same-size rewrite (mtime unchanged) would serve the cached
``.pyc`` -- not a concern for human-paced edits.
"""

import importlib
import os
import sys

# Directories whose modules are experiment DEFINITIONS -- safe to hot-reload per job.
_EXPERIMENT_DIRS = ("YbSteps", "YbSeqs", "YbScans", "YbRearrangement")


def _pyctrl_root():
    # …/pyctrl/lib/seq_reload.py -> …/pyctrl
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _experiment_roots():
    root = _pyctrl_root()
    return [os.path.join(root, d) for d in _EXPERIMENT_DIRS]


def _under_any(path, roots):
    ap = os.path.abspath(path)
    for r in roots:
        rr = os.path.abspath(r)
        if ap == rr or ap.startswith(rr + os.sep):
            return True
    return False


def reload_experiment_modules(roots=None, log=None):
    """Drop cached experiment modules so the next import re-reads them from disk.

    Args:
        roots: directories whose modules to drop (default: the pyctrl experiment dirs). Injected
            by tests.
        log: optional ``message -> None`` sink for a one-line summary.

    Returns:
        The list of dropped module names (for logging / tests). Best-effort: never raises.
    """
    if roots is None:
        roots = _experiment_roots()
    dropped = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)        # None for builtins / namespace pkgs / failed imports
        if f and _under_any(f, roots):
            try:
                del sys.modules[name]
                dropped.append(name)
            except KeyError:                       # already gone (concurrent drop) -- fine
                pass
    try:
        importlib.invalidate_caches()              # so brand-new files are discoverable too
    except Exception:  # noqa: BLE001 - advisory; never raise into the run loop
        pass
    if log and dropped:
        log("hot-reloaded %d experiment module(s) for live edits" % len(dropped))
    return dropped
