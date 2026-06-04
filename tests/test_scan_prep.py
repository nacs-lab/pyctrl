"""Phase-5 scan_prep: the run-order construction (ybBuildScanJob's Scan.Params) + the
scan-config Params field.

NO-HARDWARE: pure index math (stack / scramble_groups / build_scan_order) with a seeded PRNG,
plus write_scan_config's Params persistence written to a tmp prefix. The randomization mirrors
MATLAB production (scramble WITHIN each pass, then stack -- scrambleGroups.m + stack.m), NOT
runSeq2's global randperm.
"""

import json
import random

import pytest

from scan_prep import build_scan_order, scramble_groups, stack, write_scan_config

pytestmark = pytest.mark.no_hardware


# --------------------------------------------------------------------------- #
# stack.m
# --------------------------------------------------------------------------- #
class TestStack:
    def test_repeats_row(self):
        assert stack([1, 2, 3], 2) == [1, 2, 3, 1, 2, 3]

    def test_one_copy(self):
        assert stack([1, 2, 3], 1) == [1, 2, 3]

    def test_zero_copies(self):
        assert stack([1, 2, 3], 0) == []


# --------------------------------------------------------------------------- #
# scrambleGroups.m -- shuffle WITHIN each block, boundaries intact
# --------------------------------------------------------------------------- #
class TestScrambleGroups:
    def test_each_block_is_a_permutation_of_its_sweep(self):
        # 3 passes over 1..4; every block stays a permutation of {1,2,3,4}.
        seq = stack([1, 2, 3, 4], 3)
        out = scramble_groups(seq, 4, random.Random(1))
        assert len(out) == 12
        for start in (0, 4, 8):
            assert sorted(out[start:start + 4]) == [1, 2, 3, 4]   # boundaries intact

    def test_group_len_one_is_noop(self):
        seq = [1, 2, 3, 1, 2, 3]
        assert scramble_groups(seq, 1, random.Random(0)) == seq   # blocks of size 1

    def test_empty(self):
        assert scramble_groups([], 3, random.Random(0)) == []

    def test_does_not_mutate_input(self):
        seq = stack([1, 2, 3], 2)
        before = list(seq)
        scramble_groups(seq, 3, random.Random(0))
        assert seq == before

    def test_actually_shuffles_some_block(self):
        # With this seed at least one block is reordered (guards against a no-op regression).
        out = scramble_groups(stack([1, 2, 3, 4, 5], 4), 5, random.Random(3))
        assert out != stack([1, 2, 3, 4, 5], 4)


# --------------------------------------------------------------------------- #
# build_scan_order -- ybBuildScanJob's Scan.Params
# --------------------------------------------------------------------------- #
class TestBuildScanOrder:
    def test_unscrambled_is_plain_stack(self):
        assert build_scan_order(3, stack_num=2, scramble=False) == [1, 2, 3, 1, 2, 3]

    def test_single_pass(self):
        assert build_scan_order(4, stack_num=1, scramble=False) == [1, 2, 3, 4]

    def test_scrambled_keeps_per_block_invariant(self):
        order = build_scan_order(3, stack_num=2, scramble=True, rng=random.Random(0))
        assert len(order) == 6
        assert sorted(order) == [1, 1, 2, 2, 3, 3]              # every point twice overall
        assert sorted(order[0:3]) == [1, 2, 3]                 # ...once per pass
        assert sorted(order[3:6]) == [1, 2, 3]

    def test_empty_when_no_points(self):
        assert build_scan_order(0, stack_num=5, scramble=True) == []

    def test_empty_when_no_passes(self):
        assert build_scan_order(3, stack_num=0) == []

    def test_single_point_scramble_is_degenerate(self):
        # nseqs=1: each "block" is size 1, so scramble_groups is a no-op -> degenerate [1, 1].
        # (MATLAB guards NumPerGroup<2 a layer up; pyctrl is permissive here.)
        assert build_scan_order(1, stack_num=2, scramble=True, rng=random.Random(0)) == [1, 1]

    def test_seeded_reproducible(self):
        a = build_scan_order(5, stack_num=3, scramble=True, rng=random.Random(7))
        b = build_scan_order(5, stack_num=3, scramble=True, rng=random.Random(7))
        assert a == b


# --------------------------------------------------------------------------- #
# write_scan_config -- the Scan.Params persistence (the live-curve fix)
# --------------------------------------------------------------------------- #
class TestWriteScanConfigParams:
    def test_params_written(self, tmp_path):
        order = [1, 2, 3, 1, 2, 3]
        path = write_scan_config(20260603120000, (10, 20), 1,
                                 num_per_group=len(order), params=order, prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert cfg["Params"] == order              # seq_id (1-based) -> scan-point index map
        assert cfg["NumPerGroup"] == 6             # = length(Scan.Params)
        assert cfg["frameSize"] == [10, 20] and cfg["NumImages"] == 1

    def test_no_params_key_when_absent(self, tmp_path):
        path = write_scan_config(20260603120001, (0, 0), 1, prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert "Params" not in cfg                 # backward compatible (pre-2026-06 behavior)
