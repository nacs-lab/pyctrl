"""enable_scan.py -- EnableScan: a process-global "may scans actually run?" switch.

Faithful transliteration of ``matlab_new/lib/EnableScan.m`` (which extends
``FacyOnCleanup``). The flag is a single class-shared ``MutableRef`` (MATLAB ``Constant``
static), read/written globally via ``check()`` / ``set()`` and scoped via the
constructor. ``RunScans2`` short-circuits when ``check()`` is false -- before any
``getseq`` / build / hardware -- so it gates *whether a scan runs*, not the expansion math.

MATLAB relies on deterministic object destruction (``delete``) to restore the previous
value when a scoped guard leaves scope. Python has no deterministic destructor, so the
RAII role of ``FacyOnCleanup`` (run a restore callback on delete, unless disabled) is
provided here by the **context-manager protocol** (PYTHON_FRONTEND_PLAN.md Phase-4 W1):

    with EnableScan(False):       # disables scans for the block...
        ...
    # ...the previous value is restored on exit -- even if the block raises.

Faithful to the MATLAB constructor, ``EnableScan(enable)`` captures the prior value and
sets the new one **immediately** (so ``a = EnableScan(False)`` outside a ``with`` still
disables, matching the test). The restore runs once -- on ``__exit__`` or an explicit
``delete()`` (mirroring MATLAB ``delete(a)``); ``disable()`` cancels it (mirroring
``FacyOnCleanup.disable``). Prefer the ``with`` form; ``__del__`` is only a best-effort
backstop because CPython refcount-collection timing is not guaranteed.

FacyOnCleanup itself is not ported as a standalone class: its only behaviours used here
(run-on-cleanup / cancel) collapse into this context manager, and its other consumer
(ScanAccessTracker, Phase-4 W9) will use a plain ``try/finally``.
"""

from mutable_ref import MutableRef


class EnableScan:
    # Class-shared flag (MATLAB: properties(Constant) enabled = MutableRef(true)).
    enabled = MutableRef(True)

    def __init__(self, enable):
        # Mirror the MATLAB ctor: capture the old value, then set the new one NOW.
        self._old = EnableScan.check()
        self._enable_restore = True          # FacyOnCleanup.enable guard (run cb once)
        EnableScan.set(enable)

    # -- static global accessors (MATLAB static methods) ----------------------- #
    @staticmethod
    def check():
        return EnableScan.enabled.get()

    @staticmethod
    def set(enable):
        # MATLAB note: "Only for testing, use the scoped version EnableScan(val) instead."
        EnableScan.enabled.set(enable)

    # -- RAII restore (FacyOnCleanup.delete / .disable, via the CM protocol) ---- #
    def delete(self):
        # Restore the prior value at most once (mirrors MATLAB delete() on a handle:
        # the object becomes invalid after the first delete, so the callback runs once).
        if self._enable_restore:
            self._enable_restore = False
            EnableScan.set(self._old)

    def disable(self):
        # Cancel the restore (mirrors FacyOnCleanup.disable: enable = 0).
        self._enable_restore = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.delete()
        return False                         # never swallow exceptions

    def __del__(self):
        # Best-effort backstop for the non-context-manager path. Python GC is not
        # deterministic, so prefer `with EnableScan(...)`. Guard against interpreter
        # teardown, where the class/global may already be torn down.
        try:
            self.delete()
        except Exception:
            pass
