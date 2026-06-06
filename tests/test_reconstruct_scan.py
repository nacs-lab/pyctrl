"""reconstruct_scan: engine-free guard rails (no_hardware).

The FULL reconstruction (descriptor -> snapshot replay -> per-point build/dump) needs the
real engine + a scan that has a descriptor + a code snapshot, and is verified in a
maintenance window. These cover the early-exit guards that run BEFORE any engine import, so
they're safe in the default suite.

    python -m pytest pyctrl/tests/test_reconstruct_scan.py -v
"""

import importlib.util
import json
import os

import pytest

pytestmark = pytest.mark.no_hardware

_PYCTRL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "reconstruct_scan", os.path.join(_PYCTRL, "tools", "reconstruct_scan.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_no_sidecar(tmp_path):
    r = _load().reconstruct(str(tmp_path))
    assert r["ok"] is False and "sidecar" in r["error"]


def test_no_descriptor(tmp_path):
    scan = tmp_path / "data_20250619_170317"
    scan.mkdir()
    (scan / "data_20250619_170317.json").write_text(json.dumps({"scan_id": 20250619170317}))
    r = _load().reconstruct(str(scan))
    assert r["ok"] is False and "descriptor" in r["error"]


def test_no_snapshot(tmp_path):
    # descriptor present but no _code_snapshots/_runs/<id> -> graceful "no code snapshot"
    scan = tmp_path / "Data" / "20250619" / "data_20250619_170317"
    scan.mkdir(parents=True)
    (scan / "data_20250619_170317.json").write_text(json.dumps(
        {"scan_id": 20250619170317, "descriptor": {"seq": "X", "params": {}}}))
    r = _load().reconstruct(str(scan))
    assert r["ok"] is False and "snapshot" in r["error"]


def test_result_marker_constant():
    assert _load().RESULT_PREFIX == "RECONSTRUCT_RESULT:"
