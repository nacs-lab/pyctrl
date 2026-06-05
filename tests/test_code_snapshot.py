"""code_snapshot: per-run source capture (#2) + snapshot replay (#3).

Mirrors the SLM server's code_snapshot mechanism on the pyctrl side: content-addressed
blobs + a readable/importable per-run tree + git state, then REPLAY of the captured
experiment code via sys.path injection (the seq_reload-safe boundary: YbSeqs/YbSteps/
YbScans/YbRearrangement only; lib/expConfig are record-only).

NO hardware / engine. Run in any pyctrl interpreter:
    pytest pyctrl/tests/test_code_snapshot.py
"""
import json
import os
import sys

import pytest

import code_snapshot

pytestmark = pytest.mark.no_hardware


# --- a fake pyctrl source tree -------------------------------------------------

def _make_tree(root, *, seq_value=1, lib_value=10):
    """A minimal project root: YbSeqs/YbSteps (experiment, replayable), lib/expConfig
    (record-only)."""
    os.makedirs(os.path.join(root, "YbSeqs"), exist_ok=True)
    os.makedirs(os.path.join(root, "YbSteps"), exist_ok=True)
    os.makedirs(os.path.join(root, "lib"), exist_ok=True)
    with open(os.path.join(root, "YbSeqs", "ProbeSeq.py"), "w") as f:
        f.write("VALUE = %d\n" % seq_value)
    with open(os.path.join(root, "YbSteps", "ProbeStep.py"), "w") as f:
        f.write("STEP = 'a'\n")
    with open(os.path.join(root, "lib", "ProbeLib.py"), "w") as f:
        f.write("LIBV = %d\n" % lib_value)
    with open(os.path.join(root, "expConfig.py"), "w") as f:
        f.write("CFG = 1\n")


def _snap_count(data_root):
    d = os.path.join(data_root, "_code_snapshots")
    return len([f for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f))]) if os.path.isdir(d) else 0


# --- #2 capture ---------------------------------------------------------------

def test_snapshot_writes_blobs_and_manifest(tmp_path):
    root = str(tmp_path / "proj")
    data_root = str(tmp_path / "data")
    _make_tree(root)
    res = code_snapshot.snapshot_code(root, data_root, run_id=20260605120000,
                                      seq_name="ProbeSeq")
    # Compact result for the sidecar.
    assert res["scan_id"] == 20260605120000
    assert res["n_files"] == 4 and res["n_experiment"] == 2
    assert not res["errors"]
    # hashes dict carries ONLY experiment files (the rest live in the manifest).
    assert set(res["hashes"]) == {"YbSeqs/ProbeSeq.py", "YbSteps/ProbeStep.py"}
    # Per-run manifest: full provenance, roles, original rel paths.
    man = os.path.join(data_root, res["run_manifest"])
    with open(man) as f:
        m = json.load(f)
    assert m["scan_id"] == 20260605120000 and m["seq_name"] == "ProbeSeq"
    roles = {r["src_rel"]: r["role"] for r in m["files"]}
    assert roles["YbSeqs/ProbeSeq.py"] == "experiment"
    assert roles["lib/ProbeLib.py"] == "framework"
    assert roles["expConfig.py"] == "config"
    # Per-run tree reconstructs files at their original rel paths (readable + importable).
    assert os.path.isfile(os.path.join(data_root, res["run_dir"], "YbSeqs", "ProbeSeq.py"))


def test_blobs_dedup_across_runs(tmp_path):
    root = str(tmp_path / "proj")
    data_root = str(tmp_path / "data")
    _make_tree(root, seq_value=1)
    code_snapshot.snapshot_code(root, data_root, run_id=1)
    n1 = _snap_count(data_root)
    # Identical content -> no new blobs.
    code_snapshot.snapshot_code(root, data_root, run_id=2)
    assert _snap_count(data_root) == n1
    # Change one experiment file -> exactly one new blob.
    _make_tree(root, seq_value=999)
    code_snapshot.snapshot_code(root, data_root, run_id=3)
    assert _snap_count(data_root) == n1 + 1


# --- #3 replay ----------------------------------------------------------------

def _evict(name):
    return lambda: sys.modules.pop(name, None)


def test_snapshot_syspath_replays_experiment_code(tmp_path):
    root = str(tmp_path / "proj")
    data_root = str(tmp_path / "data")
    _make_tree(root, seq_value=1)
    code_snapshot.snapshot_code(root, data_root, run_id=42)

    # Live tree now diverges: ProbeSeq.VALUE 1 -> 2.
    with open(os.path.join(root, "YbSeqs", "ProbeSeq.py"), "w") as f:
        f.write("VALUE = 2\n")

    live_dir = os.path.join(root, "YbSeqs")
    saved = list(sys.path)
    sys.modules.pop("ProbeSeq", None)
    try:
        sys.path.insert(0, live_dir)
        import importlib
        assert importlib.import_module("ProbeSeq").VALUE == 2          # live
        # Inside the replay context, the SAME import resolves to the snapshot (VALUE==1).
        with code_snapshot.snapshot_syspath(data_root, 42,
                                            reload_modules=_evict("ProbeSeq")) as active:
            assert active is True
            assert importlib.import_module("ProbeSeq").VALUE == 1      # snapshot
        # Restored: sys.path back to exactly what it was, live import again.
        sys.modules.pop("ProbeSeq", None)
        assert importlib.import_module("ProbeSeq").VALUE == 2
    finally:
        sys.path[:] = saved
        sys.modules.pop("ProbeSeq", None)


def test_active_replay_source_marker(tmp_path):
    root = str(tmp_path / "proj")
    data_root = str(tmp_path / "data")
    _make_tree(root)
    code_snapshot.snapshot_code(root, data_root, run_id=55)
    assert code_snapshot.active_replay_source() is None
    with code_snapshot.snapshot_syspath(data_root, 55, reload_modules=_evict("ProbeSeq")):
        assert code_snapshot.active_replay_source() == 55      # scan-prep reads this
    assert code_snapshot.active_replay_source() is None        # cleared on exit


def test_replay_missing_snapshot_falls_back_to_live(tmp_path):
    data_root = str(tmp_path / "data")
    saved = list(sys.path)
    with code_snapshot.snapshot_syspath(data_root, 999) as active:
        assert active is False               # no snapshot -> caller uses live code
    assert sys.path == saved                 # sys.path untouched


def test_lib_mismatch_detects_framework_drift(tmp_path):
    root = str(tmp_path / "proj")
    data_root = str(tmp_path / "data")
    _make_tree(root, lib_value=10)
    code_snapshot.snapshot_code(root, data_root, run_id=7)
    assert code_snapshot.lib_mismatch(data_root, 7, project_root=root) == []
    # Change the framework file -> reported (replay does NOT swap lib).
    with open(os.path.join(root, "lib", "ProbeLib.py"), "w") as f:
        f.write("LIBV = 11\n")
    assert "lib/ProbeLib.py" in code_snapshot.lib_mismatch(data_root, 7, project_root=root)


# --- git state (read-only, best-effort) ---------------------------------------

def test_read_git_state_is_readonly_and_safe():
    # The real pyctrl root is a git repo; this must return a dict and never raise / mutate.
    gs = code_snapshot.read_git_state(code_snapshot.pyctrl_root())
    if gs is not None:
        assert "commit" in gs and "dirty" in gs and "status" in gs


def test_read_git_state_non_repo_returns_none(tmp_path):
    assert code_snapshot.read_git_state(str(tmp_path)) is None
