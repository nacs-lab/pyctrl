"""Phase-4 W1 -- EnableScan (+ MutableRef), the process-global "may scans run?" switch.

NO-HARDWARE: pure global-flag math; never loads the engine or touches devices.

Ports matlab_new/lib/test/TestEnableScan.m (the single ``dotest``) and adds focused
tests for the Python context-manager form that replaces MATLAB's RAII destructor
(PYTHON_FRONTEND_PLAN.md Phase-4 W1). An autouse fixture saves/restores the global flag
around every test (mirroring the MATLAB test's ``onCleanup(@() EnableScan.set(true))``)
so a failing test can't leave scans globally disabled for the rest of the suite.
"""

import pytest

from enable_scan import EnableScan
from mutable_ref import MutableRef

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _restore_enable_scan():
    saved = EnableScan.check()
    yield
    EnableScan.set(saved)


class TestEnableScan:
    def test_dotest(self):
        # Faithful port of TestEnableScan.dotest.
        # static check/set toggling
        assert EnableScan.check() is True
        EnableScan.set(False)
        assert EnableScan.check() is False
        EnableScan.set(True)
        assert EnableScan.check() is True

        # scoped guard, restored by an explicit delete (MATLAB: a0 = EnableScan(false); delete(a0))
        a0 = EnableScan(False)
        assert EnableScan.check() is False
        a0.delete()
        assert EnableScan.check() is True

        # the guard restores even when its scope is left via an exception
        # (MATLAB's nested disable(): construct guard, observe disabled, error()).
        disabled = False
        try:
            with EnableScan(False):
                disabled = not EnableScan.check()
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert disabled is True
        assert EnableScan.check() is True

    def test_context_manager_restores(self):
        assert EnableScan.check() is True
        with EnableScan(False):
            assert EnableScan.check() is False
        assert EnableScan.check() is True

    def test_context_manager_does_not_swallow_exceptions(self):
        with pytest.raises(ValueError):
            with EnableScan(False):
                raise ValueError("boom")
        assert EnableScan.check() is True   # still restored

    def test_constructor_restores_prior_value_not_default(self):
        # The guard restores whatever was set when it was created, not a hard-coded True.
        EnableScan.set(False)
        with EnableScan(True):
            assert EnableScan.check() is True
        assert EnableScan.check() is False  # restored the False that preceded the guard

    def test_nested_guards_restore_lifo(self):
        with EnableScan(False):
            assert EnableScan.check() is False
            with EnableScan(True):
                assert EnableScan.check() is True
            assert EnableScan.check() is False   # inner restored
        assert EnableScan.check() is True        # outer restored

    def test_disable_cancels_restore(self):
        # FacyOnCleanup.disable: the restore callback never fires.
        a = EnableScan(False)
        assert EnableScan.check() is False
        a.disable()
        a.delete()                               # no-op now
        assert EnableScan.check() is False       # NOT restored to True

    def test_delete_is_idempotent(self):
        EnableScan.set(True)
        a = EnableScan(False)
        a.delete()
        assert EnableScan.check() is True
        EnableScan.set(False)                    # someone else changes it after
        a.delete()                               # second delete must do nothing
        assert EnableScan.check() is False


class TestMutableRef:
    def test_get_returns_constructed_value(self):
        assert MutableRef(7).get() == 7

    def test_set_then_get(self):
        r = MutableRef(1)
        r.set(2)
        assert r.get() == 2

    def test_set_returns_self_for_chaining(self):
        r = MutableRef(0)
        assert r.set(5) is r

    def test_default_is_none(self):
        assert MutableRef().get() is None
