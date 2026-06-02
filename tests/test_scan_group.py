"""Phase-4 W2 -- ScanGroup core data model + fluent authoring DSL.

NO-HARDWARE: pure data-model math; never loads the engine or touches devices.

W2 covers building a ScanGroup (fixed params, scan axes, nested fields, multiple scans,
whole-scan assignment, base index) and reading back the low-level ``dump()`` / ``groupsize``
/ ``size``. Materialization (``getseq`` & the column-major expansion), the query surface
(``get_scan``/``get_scanaxis``/``nseq``/``scansize``...), ``usevar`` and ``load`` are later
weeks, so this file verifies the authored structure via ``dump()`` only.

Anchored to ``matlab_new/lib/test/TestScanGroup.m``: ``test_scan_nonarray``,
``test_param_size`` and ``test_newempty`` are direct ports (their MATLAB bodies use exactly
the W2 surface); ``test_setbase`` ports the ``dump`` assertions (its ``checked_disp`` lines
are display-only, dropped as for DynProps ``test_disp``). The remaining tests are focused
unit checks of the authoring DSL + its errors, as the plan asks ("unit-test W2/W3 against
tiny hand-built scans before the W4 battery").

The Python DSL diverges from MATLAB syntax only where Python lacks an equivalent:
``b.c.scan(dim) = vals`` becomes ``b.c.scan(dim, vals)``, ``b.c.scan(vals)`` stays, and the
whole-scan ``g(i) = rhs`` becomes ``g(i).assign(rhs)`` (see scan_param.py).
"""

import pytest

from scan_group import ScanGroup

pytestmark = pytest.mark.no_hardware


# Canonical "empty" pieces of a dump, reused across expectations.
DEF_USE_VAR = {"def": 0, "dims": [], "field": {}}


def empty_dump():
    return {
        "version": 1,
        "scans": [{"baseidx": 0, "params": {}, "vars": []}],
        "base": {"params": {}, "vars": []},
        "runparam": {},
        "use_var_base": dict(DEF_USE_VAR),
        "use_var_scans": [],
    }


class TestScanGroupCore:
    def test_empty_group_dump(self):
        g = ScanGroup()
        assert g.groupsize() == 1
        assert g.dump() == empty_dump()

    def test_fixed_params_on_base(self):
        g = ScanGroup()
        g().a = 1
        g().b = 2
        d = g.dump()
        assert d["base"]["params"] == {"a": 1, "b": 2}
        assert d["base"]["vars"] == []
        # The (untouched) first real scan is still empty -- base holds the values.
        assert d["scans"] == [{"baseidx": 0, "params": {}, "vars": []}]
        assert g.groupsize() == 1

    def test_nested_fixed_param(self):
        g = ScanGroup()
        g(2).k.a.b.c = 2
        assert g.groupsize() == 2
        d = g.dump()
        assert d["scans"][1]["params"] == {"k": {"a": {"b": {"c": 2}}}}

    def test_scan_axes_on_base(self):
        # Mirrors the default-scan shape built early in TestScanGroup.dotest.
        g = ScanGroup()
        g().a = 1
        g().b = 2
        g().c.scan(1, [1, 2, 3])
        g().d.scan(2, [1, 2])
        d = g.dump()
        assert d["base"] == {
            "params": {"a": 1, "b": 2},
            "vars": [
                {"size": 3, "params": {"c": [1, 2, 3]}},
                {"size": 2, "params": {"d": [1, 2]}},
            ],
        }

    def test_scan_dim1_default(self):
        # scan(vals) with no explicit dim -> dimension 1.
        g = ScanGroup()
        g().c.scan([5, 6, 7])
        d = g.dump()
        assert d["base"]["vars"] == [{"size": 3, "params": {"c": [5, 6, 7]}}]

    def test_two_params_same_dimension(self):
        g = ScanGroup()
        g().c.scan(1, [1, 2, 3])
        g().e.scan(1, [4, 5, 6])
        d = g.dump()
        assert d["base"]["vars"] == [
            {"size": 3, "params": {"c": [1, 2, 3], "e": [4, 5, 6]}}
        ]

    def test_g_end_indexes_last_scan(self):
        g = ScanGroup()
        g().a = 1
        assert g.end == 1
        g(g.end).x = 7        # g(end).x = 7  -> first real scan
        assert g.groupsize() == 1
        assert g.dump()["scans"][0]["params"] == {"x": 7}

    def test_stored_value_is_copied(self):
        # MATLAB structs/arrays copy by value on assignment; mutating the caller's list
        # afterwards must not change the stored scan axis.
        g = ScanGroup()
        vals = [1, 2, 3]
        g().c.scan(1, vals)
        vals.append(4)
        assert g.dump()["base"]["vars"][0]["params"]["c"] == [1, 2, 3]


class TestScanNonArray:
    """Direct port of TestScanGroup.test_scan_nonarray."""

    def test_scan_nonarray(self):
        g = ScanGroup()
        g().A.B.scan(1, 2)          # scalar -> decays to a fixed param
        g().A.C.scan(10)            # scalar, default dim -> fixed param
        g().B.scan(3, "abcdef")     # char row -> not an array -> fixed param
        assert g.dump() == {
            "version": 1,
            "scans": [{"baseidx": 0, "params": {}, "vars": []}],
            "base": {
                "params": {"A": {"B": 2, "C": 10}, "B": "abcdef"},
                "vars": [],
            },
            "runparam": {},
            "use_var_base": dict(DEF_USE_VAR),
            "use_var_scans": [],
        }


class TestParamSize:
    """Direct port of TestScanGroup.test_param_size (uses ScanParam.size(dim))."""

    def test_param_size(self):
        g = ScanGroup()
        assert g().size(1) == 1
        assert g().size(2) == 1
        assert g(1).size(1) == 1
        assert g(1).size(2) == 1
        assert g(2).size(1) == 1
        assert g(2).size(2) == 1

        p0 = g()
        p0.A.scan(2, [1, 2, 3])
        p0.B.scan([5, 6])
        assert g().size(1) == 2
        assert g().size(2) == 3
        assert g(1).size(1) == 1
        assert g(1).size(2) == 1

        p2 = g(2)
        p2.K.scan(3, [1, 2, 3, 4, 5])
        assert g(1).size(1) == 1
        assert g(1).size(2) == 1
        assert g(2).size(1) == 1
        assert g(2).size(2) == 1
        assert g(2).size(3) == 5


class TestNewEmpty:
    """Direct port of TestScanGroup.test_newempty."""

    def test_untouched_first_scan(self):
        g = ScanGroup()
        assert g.groupsize() == 1
        assert g.new_empty() == 1
        assert g.groupsize() == 1
        assert g.new_empty() == 2
        assert g.groupsize() == 2

    def test_base_only_does_not_count_as_touching(self):
        g = ScanGroup()
        g().A.C = 2
        assert g.groupsize() == 1
        assert g.new_empty() == 1
        assert g.groupsize() == 1
        assert g.new_empty() == 2
        assert g.groupsize() == 2

    def test_touched_first_scan_fixed(self):
        g = ScanGroup()
        g(1).A.C = 2
        assert g.groupsize() == 1
        assert g.new_empty() == 2
        assert g.groupsize() == 2

    def test_touched_first_scan_axis(self):
        g = ScanGroup()
        g(1).A.C.scan(2, [2, 3, 4])
        assert g.groupsize() == 1
        assert g.new_empty() == 2
        assert g.groupsize() == 2


class TestSetbase:
    """Ports the dump() assertions of TestScanGroup.test_setbase (display lines dropped)."""

    def test_setbase(self):
        g = ScanGroup()
        g.setbase(2, 1)
        assert g.dump() == {
            "version": 1,
            "scans": [
                {"baseidx": 0, "params": {}, "vars": []},
                {"baseidx": 1, "params": {}, "vars": []},
            ],
            "base": {"params": {}, "vars": []},
            "runparam": {},
            "use_var_base": dict(DEF_USE_VAR),
            "use_var_scans": [],
        }
        g.setbase(2, 0)
        assert g.dump() == {
            "version": 1,
            "scans": [
                {"baseidx": 0, "params": {}, "vars": []},
                {"baseidx": 0, "params": {}, "vars": []},
            ],
            "base": {"params": {}, "vars": []},
            "runparam": {},
            "use_var_base": dict(DEF_USE_VAR),
            "use_var_scans": [],
        }

    def test_setbase_loop_detection(self):
        g = ScanGroup()
        g.setbase(2, 1)
        with pytest.raises(ValueError, match="loop"):
            g.setbase(1, 2)

    def test_setbase_validation(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="non-negative integer"):
            g.setbase(2, -1)
        with pytest.raises(ValueError, match="non-existing scan"):
            g.setbase(2, 5)


class TestWholeScanAssign:
    def test_assign_param_copies_base(self):
        g = ScanGroup()
        g().a = 1
        g(2).assign(g())                 # g(2) = g()  -> copy the fallback into scan 2
        d = g.dump()
        assert g.groupsize() == 2
        assert d["scans"][1]["params"] == {"a": 1}
        assert d["scans"][1]["baseidx"] == 0
        # assigning from the fallback seeds two default use_var entries (def 0 each).
        assert d["use_var_scans"] == [dict(DEF_USE_VAR), dict(DEF_USE_VAR)]

    def test_assign_is_a_copy_not_an_alias(self):
        g = ScanGroup()
        g().a = 1
        g(2).assign(g())
        g().a = 99                       # mutate the source afterwards
        assert g.dump()["scans"][1]["params"] == {"a": 1}   # scan 2 unaffected

    def test_assign_struct_rhs(self):
        g = ScanGroup()
        g(2).assign({"c": 1, "d": 123})
        d = g.dump()
        assert d["scans"][1] == {"baseidx": 0, "params": {"c": 1, "d": 123}, "vars": []}

    def test_assign_struct_with_array_field_errors(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="Mixing fixed and variable"):
            g(2).assign({"x": [1, 2, 3]})

    def test_assign_from_other_group_errors(self):
        g = ScanGroup()
        g2 = ScanGroup()
        with pytest.raises(ValueError, match="different group"):
            g(1).assign(g2())

    def test_assign_to_base_from_scan(self):
        g = ScanGroup()
        g(1).x = 5
        g().assign(g(1))                 # g(:) = g(1) -> base gets scan-1's fixed params
        assert g.dump()["base"]["params"] == {"x": 5}


class TestRunParam:
    def test_runp_roundtrips_into_dump(self):
        g = ScanGroup()
        rp = g.runp()
        g.runp().a = 3
        rp.b = 2                          # same handle -> same store
        assert g.dump()["runparam"] == {"a": 3, "b": 2}


class TestAuthoringErrors:
    def test_cannot_scan_a_fixed_parameter(self):
        g = ScanGroup()
        g().x = 1
        with pytest.raises(ValueError, match="Cannot scan a fixed parameter"):
            g().x.scan(1, [1, 2, 3])

    def test_cannot_fix_a_scanned_parameter(self):
        g = ScanGroup()
        g().y.scan(1, [1, 2, 3])
        with pytest.raises(ValueError, match="Cannot fix a scanned parameter"):
            g().y = 5

    def test_cannot_scan_in_multiple_dimensions(self):
        g = ScanGroup()
        g().z.scan(1, [1, 2, 3])
        with pytest.raises(ValueError, match="multiple dimensions"):
            g().z.scan(2, [1, 2])

    def test_scan_size_mismatch(self):
        g = ScanGroup()
        g().p.scan(1, [1, 2, 3])
        with pytest.raises(ValueError, match="size does not match"):
            g().q.scan(1, [1, 2])

    def test_scan_dimension_must_be_positive(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="positive integer"):
            g().x.scan(0, [1, 2, 3])

    def test_override_struct_not_allowed(self):
        g = ScanGroup()
        g().s.x = 1
        with pytest.raises(ValueError, match="Override struct"):
            g().s = {"y": 2}

    def test_struct_to_nonstruct_not_allowed(self):
        g = ScanGroup()
        g().s.x = 1
        with pytest.raises(ValueError, match="struct to non-struct"):
            g().s = 5

    def test_nonstruct_to_struct_not_allowed(self):
        g = ScanGroup()
        g().a = 1
        with pytest.raises(ValueError, match="non-struct to struct"):
            g().a = {"b": 2}

    def test_assign_field_of_nonstruct_not_allowed(self):
        g = ScanGroup()
        g().a = 1
        with pytest.raises(ValueError, match="non-struct not allowed"):
            g().a.b = 2

    def test_scan_index_must_be_positive(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="must be positive"):
            g(0)
        with pytest.raises(ValueError, match="must be positive"):
            g(-1)

    def test_too_many_scan_index(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="Too many scan index"):
            g(1, 2)

    def test_scan_on_bare_param_errors(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="Must specify parameter to scan"):
            g().scan(1, [1, 2, 3])
