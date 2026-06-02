"""Phase-4 W9 -- ScanAccessTracker: warn about unused scan parameters.

NO-HARDWARE: pure bool-tree arithmetic + a per-scan completion counter; never loads the
engine or touches devices. Direct port of ``matlab_new/lib/test/TestScanAccessTracker.m``
(``test_reset`` is MATLAB-warning-prefix plumbing and is skipped, per the Phase-4 plan).

MATLAB captures ``warning(...)`` text via ``evalc`` and compares it (with a version-specific
prefix). Python instead captures the ``warnings`` list and compares each message body
directly -- cleaner and prefix-free. ``record_access``/``force_check`` are fed synthetic
``accessed`` dicts here (the live run-loop wiring is Phase 5).

The DSL diverges only where Python lacks MATLAB syntax: ``sg(1).A.C.scan(2) = [..]`` becomes
``sg(1).A.C.scan(2, [..])``.
"""

import warnings

import pytest

from scan_group import ScanGroup
from scan_access_tracker import ScanAccessTracker

pytestmark = pytest.mark.no_hardware


def _warnings_of(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught]


def _assert_no_warn(fn):
    assert _warnings_of(fn) == []


def _assert_warns(fn, *expected):
    assert _warnings_of(fn) == list(expected)


class TestScanAccessTracker:
    def test_no_unused(self):
        sg = ScanGroup()
        sg(1).A.B = 1
        sg(1).A.C.scan(2, [2, 3, 4])
        sg(2).A.C = 1
        sg(2).A.D.scan(2, [2, 3, 4, 5])
        sg(3).B.E = 1

        st = ScanAccessTracker(sg)
        _assert_no_warn(lambda: st.record_access(7, {"B": {"C": True}}))
        _assert_no_warn(lambda: st.record_access(5, {"A": {"D": True}}))
        _assert_no_warn(lambda: st.record_access(6, {"A": {"C": True}}))
        _assert_no_warn(lambda: st.record_access(4, {"C": {"D": True}}))
        _assert_no_warn(lambda: st.record_access(1, {"C": {"D": True}}))
        _assert_no_warn(lambda: st.record_access(8, True))
        _assert_no_warn(lambda: st.record_access(3, {"A": True}))
        _assert_no_warn(lambda: st.record_access(2, {}))
        _assert_no_warn(st.force_check)

    def test_use_subparams(self):
        # Can happen if the user supplied a struct default for a NaN-valued parameter: the
        # access reaches A.B.C but the parameter is only A.B, so A.B reads as unused.
        sg = ScanGroup()
        sg(1).A.B = float("nan")

        st = ScanAccessTracker(sg)
        _assert_warns(
            lambda: st.record_access(1, {"A": {"B": {"C": True}}}),
            "Unused fixed parameters in scan #1:\n  A.B")
        _assert_no_warn(st.force_check)

    def test_rep_access(self):
        sg = ScanGroup()
        sg(1).A.B.scan(1, [1, 2, 3])

        st = ScanAccessTracker(sg)
        # Re-accessing an already-collected sequence shouldn't double-count nor warn.
        _assert_no_warn(lambda: st.record_access(1, {}))
        _assert_no_warn(lambda: st.record_access(1, {"A": {"C": True}}))
        _assert_no_warn(lambda: st.record_access(2, {}))
        _assert_no_warn(lambda: st.record_access(2, {"A": {"B": True}}))
        _assert_no_warn(lambda: st.record_access(3, {}))
        _assert_no_warn(st.force_check)

    def test_force(self):
        sg = ScanGroup()
        sg(1).A.B = 1
        sg(1).A.C.scan(2, [2, 3, 4])
        sg(2).A.C = 1
        sg(2).A.D.scan(2, [2, 3, 4, 5])
        sg(3).B.E = 1

        st = ScanAccessTracker(sg)
        _assert_no_warn(lambda: st.record_access(4, {"A": {"D": True}}))
        _assert_warns(
            st.force_check,
            "Unused fixed parameters in scan #1:\n  A.B",
            "Unused scanning parameters in scan #1:\n  A.C",
            "Unused fixed parameters in scan #2:\n  A.C",
            "Unused fixed parameters in scan #3:\n  B.E")

    def test_multiline(self):
        sg = ScanGroup()
        sg(1).A.B.A = 1
        sg(1).A.B.C = 1
        sg(1).A.B.X = 1
        sg(1).A.Y.Z.K.scan(2, [2, 3, 4])
        sg(1).A.Y.A.K.scan(1, [1, 2, 3, 4])
        sg(1).A.Y.M.K.scan(4, [3, 2, 3, 4])

        st = ScanAccessTracker(sg)
        _assert_warns(
            st.force_check,
            "Unused fixed parameters in scan #1:\n  A.B.A\n     .C\n     .X",
            "Unused scanning parameters in scan #1:\n  A.Y.A.K\n     .Z.K\n     .M.K")
