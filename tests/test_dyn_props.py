"""W2 -- DynProps / SubProps parity port of matlab_new/lib/test/TestDynProps.m.

NO-HARDWARE: pure data-model math, no engine. Every MATLAB test method is ported with
the same name and assertions. Adaptations (no behavior change, just Python syntax):
  * brace default `c.X{default}`  -> `c.X[default]` (SubProps.__getitem__).
  * a parens call `c.X(default)` returns a plain dict; subsequent `.field` navigation of
    that returned struct becomes `['field']` dict indexing.
  * MATLAB free functions get_accessed/clear_accessed/isfield/fieldnames/getfields are
    methods here (dp.get_accessed() etc.).
  * NaN -> float('nan'); a MATLAB 1xN struct array -> a Python list of dicts.
  * SKIPPED: the 1-based auto-grow `dp0.A(3)=2` line (no Python lvalue-call syntax;
    test-only edge), and test_disp (YAML.sprint/cprintf formatting -- human-facing, not
    byte-load-bearing; per the Phase-3 plan it is dropped rather than ported verbatim).
"""

import pytest

from dyn_props import DynProps

pytestmark = pytest.mark.no_hardware

nan = float("nan")


class TestDynProps:
    def test_dotest(self):
        s = {"A": 1, "B": 2, "C": {"B": 3}}

        dp0 = DynProps()
        dp1 = DynProps(s)

        assert dp0.get_accessed() == {}
        assert dp1.get_accessed() == {}

        # Pre-populated values
        assert dp1.A == 1
        assert dp1.B == 2
        assert dp1.C.B == 3
        assert dp1.get_accessed() == {"A": True, "B": True, "C": {"B": True}}

        # Simple default values
        assert dp0.A(1) == 1
        assert dp0.B(2) == 2
        assert dp0.get_accessed() == {"A": True, "B": True}

        dp0.clear_accessed()
        dp1.clear_accessed()
        assert dp0.get_accessed() == {}
        assert dp1.get_accessed() == {}

        # Struct assignments
        dp0.C = {"A": 1, "B": 2}
        assert dp0.C.A == 1
        assert dp0.C.B == 2
        assert dp0.get_accessed() == {"C": {"A": True, "B": True}}
        assert dp0.C() == {"A": 1, "B": 2}
        assert dp0.get_accessed() == {"C": True}
        dp0.clear_accessed()
        assert dp0.get_accessed() == {}
        d0c = dp0.C({"C": 3})
        assert dp0.get_accessed() == {"C": True}
        assert d0c["C"] == 3
        assert d0c == {"A": 1, "B": 2, "C": 3}
        assert dp0.C() == {"A": 1, "B": 2, "C": 3}
        d0c2 = dp0.C({"D": 4})
        assert "C" in d0c and "D" not in d0c
        assert "C" in d0c2 and "D" in d0c2
        assert d0c2["D"] == 4
        assert d0c2 == {"A": 1, "B": 2, "C": 3, "D": 4}
        assert dp0.C() == {"A": 1, "B": 2, "C": 3, "D": 4}
        d0c3 = dp0.C("D", 5)
        assert d0c3["D"] == 4
        assert d0c3 == {"A": 1, "B": 2, "C": 3, "D": 4}
        assert dp0.C() == {"A": 1, "B": 2, "C": 3, "D": 4}
        dp0.C.D = nan
        d0c4 = dp0.C("D", 5)
        assert d0c4["D"] == 5
        assert d0c4 == {"A": 1, "B": 2, "C": 3, "D": 5}
        assert dp0.C() == {"A": 1, "B": 2, "C": 3, "D": 5}

        dp0.clear_accessed()
        assert dp0.get_accessed() == {}
        dp0.C = {"A": 1, "B": 2}
        assert dp0.get_accessed() == {}
        d0c = dp0.C["C", 3]                  # brace -> SubProps proxy
        assert dp0.get_accessed() == {}      # brace does NOT mark accessed
        assert d0c.C == 3
        assert dp0.get_accessed() == {"C": {"C": True}}
        assert d0c() == {"A": 1, "B": 2, "C": 3}
        assert dp0.get_accessed() == {"C": True}
        d0c.C = 2                            # write through the proxy
        dp0.clear_accessed()
        assert dp0.get_accessed() == {}
        assert dp0.C() == {"A": 1, "B": 2, "C": 2}
        assert dp0.get_accessed() == {"C": True}
        d0c.C = 3
        d0c2 = dp0.C["D", 4]
        assert d0c2.D == 4
        assert d0c2() == {"A": 1, "B": 2, "C": 3, "D": 4}

        dp0.clear_accessed()
        assert dp0.get_accessed() == {}

        # Create new nested field
        dp1.D.E.F = 3
        assert dp1.get_accessed() == {}
        assert dp1.D.E.F == 3
        assert dp1.get_accessed() == {"D": {"E": {"F": True}}}

        dp1.clear_accessed()
        assert dp1.get_accessed() == {}

        # Create new nested field with default value
        assert dp0.D.E.F.G(4) == 4
        assert dp0.get_accessed() == {"D": {"E": {"F": {"G": True}}}}
        dp0.clear_accessed()
        assert dp0.get_accessed() == {}
        assert dp0.D.E.F.G == 4
        assert dp0.get_accessed() == {"D": {"E": {"F": {"G": True}}}}

        dp0.clear_accessed()
        assert dp0.get_accessed() == {}

        # (MATLAB lines 146-153, `dp0.A(3) = 2` array auto-grow, skipped: no Python
        #  lvalue-call syntax; test-only edge, not a real-sequence pattern.)

        # Reference to subfield (handle aliasing) -- dp0.C is {A:1,B:2,C:3,D:4} here.
        c0 = dp0.C
        assert c0.A == 1
        assert c0.B == 2
        assert dp0.get_accessed() == {"C": {"A": True, "B": True}}
        assert c0.A(3) == 1
        assert c0.C(3) == 3
        assert dp0.get_accessed() == {"C": {"A": True, "B": True, "C": True}}
        assert c0.C == 3
        c0.D = 4
        assert c0.D(3) == 4
        assert c0.D == 4
        c0.A = 2
        assert c0.A == 2
        assert dp0.get_accessed() == {"C": {"A": True, "B": True, "C": True, "D": True}}

        # Mutation through the alias is reflected on the original.
        assert dp0.C.A == 2
        assert dp0.C.B == 2
        assert dp0.C.C == 3
        assert dp0.C.D == 4

        c0.A = nan
        assert c0.A(1) == 1
        assert c0.A() == 1
        assert c0.A == 1

    def test_brace_multiaccess(self):
        dp = DynProps()
        dp.C.A = 2
        c = dp.C[{"A": 4, "B": 3}, "C", 1]
        assert dp.get_accessed() == {}
        assert c() == {"A": 2, "B": 3, "C": 1}
        assert dp.get_accessed() == {"C": True}
        assert dp.C() == {"A": 2, "B": 3, "C": 1}

    def test_paren_multiaccess(self):
        dp = DynProps()
        dp.C.A = 2
        assert dp.C({"A": 4, "B": 3}, "C", 1) == {"A": 2, "B": 3, "C": 1}
        assert dp.get_accessed() == {"C": True}

    def test_isfield(self):
        dp = DynProps()
        assert not dp.isfield("abc")
        assert not dp.XYZ.isfield("abc")
        assert dp.get_accessed() == {}
        assert dp() == {}
        assert dp.get_accessed() is True

        dp.clear_accessed()
        assert dp.get_accessed() == {}

        dp.A = nan
        assert not dp.isfield("A")
        dp.B = {"C": nan}
        assert not dp.B.isfield("C")
        assert dp.get_accessed() == {}
        assert dp() == {"B": {}}
        assert dp.get_accessed() is True
        assert dp.B() == {}
        assert dp.get_accessed() is True
        assert dp.isfield("B")

        assert dp.B.C("abc") == "abc"
        assert dp.B.isfield("C")

    def test_getfields(self):
        dp = DynProps({"A": {"C": nan}, "B": {}, "C": {"X": [1, 2, 3]}, "D": nan})

        assert dp.getfields() == {}
        assert dp.get_accessed() == {}
        assert dp.getfields("A", "C") == {"A": {}, "C": {"X": [1, 2, 3]}}
        assert dp.get_accessed() == {"A": True, "C": True}
        assert dp.getfields({"X": 123}, "B") == {"B": {}, "X": 123}
        assert dp.get_accessed() == {"A": True, "C": True, "B": True}

        dp.clear_accessed()
        assert dp.get_accessed() == {}

        # getfields marks correctly when the whole struct is already marked.
        dp()
        assert dp.get_accessed() is True
        assert dp.getfields("A", "C") == {"A": {}, "C": {"X": [1, 2, 3]}}
        assert dp.get_accessed() is True

    def test_fieldnames(self):
        dp = DynProps({"A": {"C": nan}, "B": 2, "D": nan})
        assert dp.fieldnames() == ["A", "B"]
        assert dp.get_accessed() == {}
        assert list(dp().keys()) == ["A", "B"]
        assert dp.get_accessed() is True

        dp.clear_accessed()
        assert dp.get_accessed() == {}

        assert dp.A.fieldnames() == []
        assert dp.get_accessed() == {}
        assert list(dp.A().keys()) == []
        assert dp.get_accessed() == {"A": True}

    def test_merge(self):
        dp = DynProps()
        result = dp.A.B.C(
            {"A": 1, "B": {"C": {"K": 4}, "D": 5}},
            {"A": 3, "B": {"C": {"M": "abc"}, "D": ""}},
        )
        assert result == {"A": 1, "B": {"C": {"K": 4, "M": "abc"}, "D": 5}}
        assert dp.get_accessed() == {"A": {"B": {"C": True}}}
        assert dp.A.B.C() == {"A": 1, "B": {"C": {"K": 4, "M": "abc"}, "D": 5}}

    def test_chain(self):
        dp = DynProps()

        # parens returns a value (dict); navigate the returned struct with [].
        assert dp.A.B.C({"A": 3, "B": {"C": {"M": "abc"}, "D": ""}})["B"]["C"] == {"M": "abc"}
        assert dp.get_accessed() == {"A": {"B": {"C": True}}}
        assert dp.A.B.C() == {"A": 3, "B": {"C": {"M": "abc"}, "D": ""}}

        # brace returns a SubProps proxy; continue navigating + injecting.
        assert dp.X.Y[{"A": 3, "B": {"C": {"M": "abc"}, "D": ""}}].B.F("M", 10) == {"M": 10}
        assert dp.get_accessed() == {"A": {"B": {"C": True}},
                                     "X": {"Y": {"B": {"F": True}}}}
        assert dp.X.Y() == {"A": 3, "B": {"C": {"M": "abc"}, "D": "", "F": {"M": 10}}}
        assert dp.get_accessed() == {"A": {"B": {"C": True}}, "X": {"Y": True}}

        assert dp("C", "a")["A"]["B"]["C"]["A"] == 3
        assert dp.get_accessed() is True
        dp.clear_accessed()
        assert dp.get_accessed() == {}
        assert dp["D", 100].D == 100
        assert dp.get_accessed() == {"D": True}

    def test_struct_array(self):
        dp = DynProps()
        dp.A = [{"A": 1, "B": 1}, {"A": 1, "B": 2}]   # MATLAB 1x2 struct array
        assert dp.get_accessed() == {}
        assert dp.A == [{"A": 1, "B": 1}, {"A": 1, "B": 2}]
        assert dp.get_accessed() == {"A": True}
