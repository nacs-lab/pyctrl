"""Phase-5 scan_prep: the run-order construction (ybBuildScanJob's Scan.Params) + the
scan-config Params field.

NO-HARDWARE: pure index math (stack / scramble_groups / build_scan_order) with a seeded PRNG,
plus write_scan_config's Params persistence written to a tmp prefix. The randomization mirrors
MATLAB production (scramble WITHIN each pass, then stack -- scrambleGroups.m + stack.m), NOT
runSeq2's global randperm.
"""

import json
import os
import random

import pytest

from scan_prep import (build_scan_order, scramble_groups, stack, write_scan_config,
                       _pattern_detection_box)

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


# --------------------------------------------------------------------------- #
# write_scan_config -- the self-contained reconstruction descriptor (SeqPlotter).
# --------------------------------------------------------------------------- #
class TestWriteScanConfigDescriptor:
    def test_descriptor_stored(self, tmp_path):
        desc = {"schema_version": 1, "seq": "RydDetSeq",
                "params": {"Pushout.Time": {"scan": 1, "values": [1e-3, 2e-3]}}}
        path = write_scan_config(20260603120002, (0, 0), 1,
                                 descriptor=desc, prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert cfg["descriptor"] == desc           # rebuilds via dispatch_descriptor offline

    def test_no_descriptor_key_when_absent(self, tmp_path):
        path = write_scan_config(20260603120003, (0, 0), 1, prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert "descriptor" not in cfg             # backward compatible / best-effort


# --------------------------------------------------------------------------- #
# write_scan_config -- per-run code snapshot (#2). Additive: cfg['code_snapshot']
# + content-addressed blobs/manifest under <prefix>/Data/_code_snapshots.
# --------------------------------------------------------------------------- #
class TestWriteScanConfigCodeSnapshot:
    def test_snapshot_block_and_manifest(self, tmp_path, monkeypatch):
        monkeypatch.delenv("YB_CODE_SNAPSHOT", raising=False)
        sid = 20260605120000
        path = write_scan_config(sid, (10, 20), 1, num_per_group=4, prefix=str(tmp_path))
        cfg = json.load(open(path))
        cs = cfg.get("code_snapshot")
        assert cs is not None and cs["scan_id"] == sid
        assert cs["n_experiment"] >= 1             # the real YbSeqs/YbSteps were captured
        assert isinstance(cs.get("hashes"), dict)
        # Per-run manifest exists under <prefix>/Data/_code_snapshots/_runs/<sid>/.
        man = tmp_path / "Data" / cs["run_manifest"].replace("/", os.sep)
        assert man.is_file()

    def test_snapshot_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YB_CODE_SNAPSHOT", "0")
        path = write_scan_config(20260605120001, (0, 0), 1, prefix=str(tmp_path))
        assert "code_snapshot" not in json.load(open(path))   # opt-out is honored

    def test_replay_run_records_pointer_not_resnapshot(self, tmp_path, monkeypatch):
        # A '+code' re-queue executes OLD code; its own sidecar must honestly point at the
        # replayed run (not re-snapshot the live disk, which is not what ran).
        import code_snapshot
        monkeypatch.delenv("YB_CODE_SNAPSHOT", raising=False)
        monkeypatch.setattr(code_snapshot, "_active_replay_source", 20260101000000)
        path = write_scan_config(20260605120003, (5, 5), 1, prefix=str(tmp_path))
        cs = json.load(open(path))["code_snapshot"]
        assert cs["replayed_from_scan_id"] == 20260101000000
        assert "hashes" not in cs                  # did NOT re-snapshot live experiment code

    def test_snapshot_failure_never_breaks_the_sidecar(self, tmp_path, monkeypatch):
        # A failure deep in capture must leave a valid sidecar (provenance is best-effort):
        # _capture_code_snapshot swallows it and returns None, so no code_snapshot key.
        import code_snapshot
        monkeypatch.delenv("YB_CODE_SNAPSHOT", raising=False)
        monkeypatch.setattr(code_snapshot, "snapshot_code",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        path = write_scan_config(20260605120002, (5, 5), 1, prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert "code_snapshot" not in cfg and cfg["frameSize"] == [5, 5]


# --------------------------------------------------------------------------- #
# write_scan_config -- detection calibration baked into the json (mirrors
# MATLAB ybBuildScanPayload), so the offline analysis gets per-site maps +
# thresholds + discrimination for pyctrl runs.
# --------------------------------------------------------------------------- #
class TestWriteScanConfigCalibration:
    def _day_folder(self, tmp_path, scan_id, n_sites=5):
        np = pytest.importorskip("numpy")
        savemat = pytest.importorskip("scipy.io").savemat
        day = str(scan_id)[:8]
        day_dir = tmp_path / "Data" / day
        day_dir.mkdir(parents=True)
        with open(day_dir / "gridLocations.txt", "w") as f:
            f.write("Y\tX\n")
            for i in range(n_sites):
                f.write("%f\t%f\n" % (10.0 + i, 20.0 + i))
        savemat(str(day_dir / "threshold.mat"),
                {"thresholds": np.arange(100.0, 100.0 + n_sites),
                 "infidelities": np.linspace(1e-3, 5e-3, n_sites)})
        return n_sites

    def test_calibration_written_for_production_scan(self, tmp_path):
        sid = 20260603120010
        n = self._day_folder(tmp_path, sid)
        path = write_scan_config(sid, (2100, 2100), 2, is_init=0,
                                 num_per_group=10, prefix=str(tmp_path))
        cfg = json.load(open(path))
        for k in ("initGridLocationsX", "initGridLocationsY",
                  "initThresholds", "initInfidelities"):
            assert len(cfg[k]) == n
        assert cfg["initGridLocationsY"][0] == 10.0   # gridLocations.txt col Y
        assert cfg["initGridLocationsX"][0] == 20.0   # col X
        assert cfg["boxSize"] == 9 and cfg["maskSigma"] == 2

    def test_pattern_detection_box_gated_override(self):
        # Per-pattern detection-box hook: ONLY a pattern that sets boxSize/maskSigma in its
        # expConfig ByPattern overlay differs; everything else stays at the global (9, 2).
        assert _pattern_detection_box("2x15x15_xyoffset_5um") == (13, 3)  # defocused 2-layer array
        assert _pattern_detection_box("33x33_uniform") == (9, 2)          # no key -> global default
        assert _pattern_detection_box("") == (9, 2)
        assert _pattern_detection_box("nonexistent_pattern") == (9, 2)

    def test_is_init_scan_has_no_calibration(self, tmp_path):
        sid = 20260603120011
        self._day_folder(tmp_path, sid)
        path = write_scan_config(sid, (2100, 2100), 1, is_init=1,
                                 prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert not any(k.startswith("initGrid") for k in cfg)

    def test_missing_day_folder_is_graceful(self, tmp_path):
        # No day folder created -> no calibration keys, no crash.
        path = write_scan_config(20260603120012, (2100, 2100), 2, is_init=0,
                                 prefix=str(tmp_path))
        cfg = json.load(open(path))
        assert not any(k.startswith("initGrid") for k in cfg)
