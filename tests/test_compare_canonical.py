"""compare_bytes comparison-node canonicalization.

Python has no ``__rlt__``, so ``const < seqval`` reflects to
``seqval.__gt__(const)`` and the front-end serializes ``GT{seqval, const}`` where
MATLAB's ``const < seqval`` is ``LT{const, seqval}``. They are mathematically
identical, so the byte comparator canonicalizes swappable comparisons rather than
forcing the porter to write ``lt()``/``gt()``. These tests pin that behavior --
including that it does NOT mask a genuine opcode mistake.
"""

import pytest

import compare_bytes as cbz
from seq_val import SeqVal, lt, gt, le, ge
from seq_context import SeqContext

pytestmark = pytest.mark.no_hardware


def _g(argtype, val):
    return {"argtype": argtype, "val": val}


class TestCanonicalNode:
    def test_lt_gt_mirror_equal(self):
        # GT{g,3} (Python `3 < g`) canonicalizes to the same form as LT{3,g}
        # (MATLAB `3 < g`).
        gt_node = {"op": SeqVal.OP_CMP_GT, "args": [_g("global", 0), _g("float64", 3.0)]}
        lt_node = {"op": SeqVal.OP_CMP_LT, "args": [_g("float64", 3.0), _g("global", 0)]}
        assert cbz.canonical_node(gt_node) == cbz.canonical_node(lt_node)

    def test_le_ge_mirror_equal(self):
        ge_node = {"op": SeqVal.OP_CMP_GE, "args": [_g("global", 0), _g("float64", 3.0)]}
        le_node = {"op": SeqVal.OP_CMP_LE, "args": [_g("float64", 3.0), _g("global", 0)]}
        assert cbz.canonical_node(ge_node) == cbz.canonical_node(le_node)

    def test_eq_ne_commutative(self):
        for op in (SeqVal.OP_CMP_EQ, SeqVal.OP_CMP_NE):
            ab = {"op": op, "args": [_g("global", 0), _g("float64", 3.0)]}
            ba = {"op": op, "args": [_g("float64", 3.0), _g("global", 0)]}
            assert cbz.canonical_node(ab) == cbz.canonical_node(ba)

    def test_does_not_mask_real_opcode_mistake(self):
        # Same arg order but the wrong operator (LT vs GT) must STILL differ:
        # a<b is not a>b. Only the swap-equivalent forms are unified.
        lt_ab = {"op": SeqVal.OP_CMP_LT, "args": [_g("global", 0), _g("global", 1)]}
        gt_ab = {"op": SeqVal.OP_CMP_GT, "args": [_g("global", 0), _g("global", 1)]}
        assert cbz.canonical_node(lt_ab) != cbz.canonical_node(gt_ab)

    def test_non_comparison_nodes_untouched(self):
        add = {"op": SeqVal.OP_ADD, "args": [_g("global", 0), _g("float64", 3.0)]}
        assert cbz.canonical_node(add) == add


class TestReflectedOperatorEquivalence:
    def _node_table(self, build):
        ctx = SeqContext()
        g, _ = ctx.new_global(SeqVal.TYPE_FLOAT64)
        ctx.get_val_id(build(g))
        return ctx.node_serialized()

    @pytest.mark.parametrize("operator_form, function_form", [
        (lambda g: 3 < g, lambda g: lt(3, g)),   # Python `3 < g` vs MATLAB form
        (lambda g: 3 > g, lambda g: gt(3, g)),
        (lambda g: 3 <= g, lambda g: le(3, g)),
        (lambda g: 3 >= g, lambda g: ge(3, g)),
    ])
    def test_reflected_serializes_equivalently(self, operator_form, function_form):
        raw_py = self._node_table(operator_form)     # reflected (e.g. GT{g,3})
        raw_ml = self._node_table(function_form)     # MATLAB form (e.g. LT{3,g})
        # The raw bytes differ (that's the whole point) ...
        assert raw_py != raw_ml
        # ... but canonicalized node tables are equal.
        canon_py = [cbz.canonical_node(n) for n in cbz.decode_nodes(raw_py)]
        canon_ml = [cbz.canonical_node(n) for n in cbz.decode_nodes(raw_ml)]
        assert canon_py == canon_ml
