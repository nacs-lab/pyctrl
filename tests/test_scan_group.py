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
from scan_info import ScanInfo

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


# =========================================================================== #
# W3 -- materialization (getseq column-major expansion) + the query surface.
# =========================================================================== #
def dotest_group():
    """The 2-scan group built early in TestScanGroup.dotest (pre-setbase)."""
    g = ScanGroup()
    g().a = 1
    g().b = 2
    b = g()
    b.c.scan(1, [1, 2, 3])
    g(1).c = 3                 # scan 1 fixes c -> shadows the base scan axis
    b.d.scan(2, [1, 2])
    g(2).d = 0                 # scan 2 fixes d -> shadows the base scan axis
    g(g.end).k.a.b.c = 2       # nested fixed param on scan 2
    return g


KSTRUCT = {"a": {"b": {"c": 2}}}


class TestGetSeq1D:
    def test_single_axis_expansion(self):
        g = ScanGroup()
        g().a = 1
        g().c.scan(1, [10, 20, 30])
        assert g.nseq() == 3
        assert g.scansize(1) == 3
        assert g.scandim(1) == 1
        assert g.getseq(1) == {"a": 1, "c": 10}
        assert g.getseq(2) == {"a": 1, "c": 20}
        assert g.getseq(3) == {"a": 1, "c": 30}

    def test_getseq_out_of_bound(self):
        g = ScanGroup()
        g().c.scan(1, [1, 2, 3])
        with pytest.raises(ValueError, match="out of bound"):
            g.getseq(4)


class TestGetSeq2D:
    def test_column_major_dim1_fastest(self):
        # Dimension 1 varies fastest, then dimension 2 (ScanGroup.m:278-294).
        g = ScanGroup()
        g().c.scan(1, [1, 2, 3])
        g().d.scan(2, [10, 20])
        assert g.nseq() == 6
        assert g.scandim(1) == 2
        order = [g.getseq(n) for n in range(1, 7)]
        assert order == [
            {"c": 1, "d": 10},
            {"c": 2, "d": 10},
            {"c": 3, "d": 10},
            {"c": 1, "d": 20},
            {"c": 2, "d": 20},
            {"c": 3, "d": 20},
        ]


class TestGetSeqBaseMerge:
    """getseq across a 2-scan group with base-fallback merge (port of dotest)."""

    def test_pre_setbase_battery(self):
        g = dotest_group()
        assert g.groupsize() == 2
        assert g.scansize(1) == 2          # scan 1: dummy dim1, dim2 size 2
        assert g.scansize(2) == 3          # scan 2: dim1 size 3
        assert g.nseq() == 5
        assert g.getseq(1) == {"c": 3, "a": 1, "b": 2, "d": 1}
        assert g.getseq(2) == {"c": 3, "a": 1, "b": 2, "d": 2}
        assert g.getseq(3) == {"d": 0, "k": KSTRUCT, "a": 1, "b": 2, "c": 1}
        assert g.getseq(4) == {"d": 0, "k": KSTRUCT, "a": 1, "b": 2, "c": 2}
        assert g.getseq(5) == {"d": 0, "k": KSTRUCT, "a": 1, "b": 2, "c": 3}

    def test_post_setbase_battery(self):
        g = dotest_group()
        g.setbase(2, 1)                    # scan 2 now falls back to scan 1
        assert g.scansize(1) == 2
        assert g.scansize(2) == 1          # c is fixed via scan 1 -> dim1 collapses
        assert g.nseq() == 3
        assert g.getseq(1) == {"c": 3, "a": 1, "b": 2, "d": 1}
        assert g.getseq(2) == {"c": 3, "a": 1, "b": 2, "d": 2}
        assert g.getseq(3) == {"d": 0, "k": KSTRUCT, "c": 3, "a": 1, "b": 2}

    def test_getseq_independent_of_internals(self):
        # getseq returns an independent dict; mutating it must not corrupt the cache.
        g = dotest_group()
        s = g.getseq(1)
        s["a"] = 999
        assert g.getseq(1)["a"] == 1


class TestAxisNum:
    """Direct port of TestScanGroup.test_axisnum."""

    def test_axisnum(self):
        g = ScanGroup()
        assert g.axisnum() == 0
        g(g.end).A.scan(1, [1, 2, 3, 4])
        g(g.end).B.scan(1, [1, 2, 3, 4])
        g(g.end).C.scan(2, [1, 2, 3, 4])
        assert g.axisnum() == 2
        assert g.axisnum(1, 1) == 2
        assert g.axisnum(1, 2) == 1
        assert g.axisnum(1, 100) == 0


class TestGetScanAxis:
    """Direct port of TestScanGroup.test_get_scanaxis."""

    def _group(self):
        g = ScanGroup()
        g(g.end).A.B.C = 2
        g(g.end).A.C.D = 3
        g(g.end).K.L = 4
        g(g.end).K.Z = 42
        g(g.end).A.B.E.scan([1, 2, 3, 4])
        g(g.end).A.C.F.scan(1, [2, 3, 4, 5])
        g(g.end).A.C.X.scan(1, [3, 2, 1, 0])
        g(g.end).K.M.scan(2, [1, 2, 3])
        g(g.end).K.Y.scan(2, [2, 3, 4])
        return g

    def test_dim1_by_index(self):
        g = self._group()
        assert g.get_scanaxis(1, 1, 1) == ([1, 2, 3, 4], "A.B.E")
        assert g.get_scanaxis(1, 1, 2) == ([2, 3, 4, 5], "A.C.F")
        assert g.get_scanaxis(1, 1, 3) == ([3, 2, 1, 0], "A.C.X")

    def test_dim1_by_name(self):
        g = self._group()
        assert g.get_scanaxis(1, 1, "A.B.E")[0] == [1, 2, 3, 4]
        assert g.get_scanaxis(1, 1, "A.C.F")[0] == [2, 3, 4, 5]
        assert g.get_scanaxis(1, 1, "A.C.X")[0] == [3, 2, 1, 0]

    def test_dim2(self):
        g = self._group()
        assert g.get_scanaxis(1, 2, 1)[0] == [1, 2, 3]
        assert g.get_scanaxis(1, 2, 2)[0] == [2, 3, 4]
        assert g.get_scanaxis(1, 2, "K.M")[0] == [1, 2, 3]
        assert g.get_scanaxis(1, 2, "K.Y")[0] == [2, 3, 4]

    def test_dim3_decays_to_fixed(self):
        # An out-of-bound dimension falls back to the fixed parameters.
        g = self._group()
        assert g.get_scanaxis(1, 3, 1)[0] == 2
        assert g.get_scanaxis(1, 3, 2)[0] == 3
        assert g.get_scanaxis(1, 3, 3)[0] == 4
        assert g.get_scanaxis(1, 3, 4)[0] == 42
        assert g.get_scanaxis(1, 3, "A.B.C")[0] == 2
        assert g.get_scanaxis(1, 3, "A.C.D")[0] == 3
        assert g.get_scanaxis(1, 3, "K.L")[0] == 4
        assert g.get_scanaxis(1, 3, "K.Z")[0] == 42

    def test_missing_field_errors(self):
        g = self._group()
        with pytest.raises(ValueError, match="Cannot find scan field"):
            g.get_scanaxis(1, 1, "no.such.path")
        with pytest.raises(ValueError, match="Cannot find scan field"):
            g.get_scanaxis(1, 1, 99)


class TestSizeQueries:
    def test_empty_group(self):
        g = ScanGroup()
        assert g.groupsize() == 1
        assert g.nseq() == 1               # one scan, no axes -> one sequence
        assert g.scansize(1) == 1
        assert g.scandim(1) == 0

    def test_get_fixed_and_get_vars(self):
        g = ScanGroup()
        g().a = 1
        g().c.scan(1, [1, 2, 3])
        assert g.get_fixed(1) == {"a": 1}
        params, sz = g.get_vars(1, 1)
        assert params == {"c": [1, 2, 3]}
        assert sz == 3

    def test_get_fixed_out_of_bound(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="Out of bound"):
            g.get_fixed(0)
        with pytest.raises(ValueError, match="Out of bound"):
            g.get_vars(0)


class TestGetScan:
    """ScanInfo read-only view (the W3 surface of dotest's get_scan asserts)."""

    def test_fixed_leaf_returns_value_and_dim0(self):
        g = ScanGroup()
        g().a = 5
        val, dim = g.get_scan(1).a()
        assert val == 5
        assert dim == 0

    def test_swept_leaf_returns_value_and_dim(self):
        g = ScanGroup()
        g().c.scan(1, [1, 2, 3])
        val, dim = g.get_scan(1).c()
        assert val == [1, 2, 3]
        assert dim == 1

    def test_missing_returns_proxy_dim_neg1(self):
        g = ScanGroup()
        g().k.a.b.c = 2
        x, dim = g.get_scan(1).k()         # k is a sub-tree, not a leaf
        assert dim == -1
        assert isinstance(x, ScanInfo)
        x2, dim2 = g.get_scan(1).e()       # absent entirely
        assert dim2 == -1
        assert isinstance(x2, ScanInfo)

    def test_default_form(self):
        g = ScanGroup()
        val, dim = g.get_scan(1).e(2)      # absent -> default value, dim 0
        assert val == 2
        assert dim == 0

    def test_fieldnames(self):
        g = ScanGroup()
        g().a = 1
        g().b = 2
        g().c.scan(1, [1, 2, 3])
        assert g.get_scan(1).fieldnames() == ["a", "b", "c"]

    def test_subfieldnames(self):
        g = ScanGroup()
        g().k.a.b.c = 2
        assert g.get_scan(1).k.fieldnames() == ["a"]

    def test_get_scan_out_of_bound(self):
        g = ScanGroup()
        with pytest.raises(ValueError, match="Out of bound"):
            g.get_scan(0)

    def test_read_only(self):
        g = ScanGroup()
        with pytest.raises(TypeError, match="read-only"):
            g.get_scan(1).a = 5
