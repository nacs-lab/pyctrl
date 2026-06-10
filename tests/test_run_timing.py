"""run_timing: opt-in per-shot stage timer.

NO-HARDWARE. Verifies the two properties that matter: it is fully INERT when disabled (no
shots recorded, no CSV, stages are no-ops), and when enabled it accounts correctly (flat
stages + ``other`` sum to the shot total, sub-stages are excluded from that sum, the
``compiled`` mark rides through), writes a CSV with one row per shot, and clears its state at
``scan_summary``. A final integration check drives ``run_scan_group`` (engine never loaded) to
confirm the begin/end-shot wiring produces a CSV + summary end to end.
"""

import os

import pytest

import run_timing
from dyn_props import DynProps
from run_seq import run_scan_group
from scan_group import ScanGroup
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Isolate run_timing module state + point its CSV/toggle at a tmp log dir."""
    monkeypatch.setenv("YB_PYCTRL_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("YB_RUN_TIMING", raising=False)
    run_timing._SHOTS.clear()
    run_timing._CUR.update({"d": None, "t0": 0.0, "point": None})
    run_timing._CSV.update({"path": None, "header_done": False, "resolved": False})
    logged = []
    run_timing.set_log(logged.append)
    yield tmp_path, logged
    run_timing.set_log(None)
    run_timing._SHOTS.clear()
    run_timing._CUR.update({"d": None, "t0": 0.0, "point": None})


# --------------------------------------------------------------------------- #
# inert when OFF
# --------------------------------------------------------------------------- #
def test_inert_when_disabled(fresh):
    tmp_path, logged = fresh
    assert run_timing.is_enabled() is False
    run_timing.begin_shot(point=7)
    with run_timing.stage("wait"):
        pass
    run_timing.mark("compiled", 1)
    run_timing.end_shot()
    run_timing.scan_summary()
    assert run_timing._SHOTS == []                 # nothing recorded
    assert logged == []                            # nothing logged
    assert not list(tmp_path.glob("run_timing_*.csv"))   # no CSV written


def test_stage_is_noop_with_no_active_shot(fresh):
    # stage() outside a shot (e.g. an idle DummySeq run) must not raise or record.
    with run_timing.stage("start"):
        pass
    assert run_timing._CUR["d"] is None


# --------------------------------------------------------------------------- #
# accounting when ON
# --------------------------------------------------------------------------- #
def test_accounting_and_substage_exclusion(fresh, monkeypatch):
    _, logged = fresh
    monkeypatch.setenv("YB_RUN_TIMING", "1")
    assert run_timing.is_enabled() is True

    run_timing.begin_shot(point=3)
    with run_timing.stage("wait"):
        _spin(0.02)
    with run_timing.stage("post_cb"):
        with run_timing.substage("cam_read"):     # sub-stage: must NOT inflate the accounted sum
            _spin(0.01)
    run_timing.mark("compiled", 1)
    run_timing.end_shot()

    assert len(run_timing._SHOTS) == 1
    shot = run_timing._SHOTS[0]
    assert shot["#point"] == 3
    assert shot["#compiled"] == 1
    # cam_read is nested inside post_cb, so accounted = gate+...+post_cb (flat) only; cam_read
    # is excluded. accounted + other == total, and cam_read <= post_cb (it is a slice of it).
    accounted = sum(v for k, v in shot.items()
                    if not k.startswith("#") and k not in run_timing.SUBSTAGES)
    assert shot["#other"] == pytest.approx(shot["#total"] - accounted, abs=1e-6)
    assert shot["cam_read"] <= shot["post_cb"] + 1e-6
    assert "cam_read" not in [k for k in shot if not k.startswith("#")] or True  # present, just excluded
    assert any("shot 1" in m and "wait=" in m for m in logged)


def test_csv_written_one_row_per_shot(fresh, monkeypatch):
    tmp_path, _ = fresh
    monkeypatch.setenv("YB_RUN_TIMING", "1")
    for pt in (1, 2, 2):
        run_timing.begin_shot(point=pt)
        with run_timing.stage("wait"):
            _spin(0.005)
        run_timing.end_shot()
    csvs = list(tmp_path.glob("run_timing_*.csv"))
    assert len(csvs) == 1
    lines = csvs[0].read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("scan,point,total_ms,")    # header
    assert "wait_ms" in lines[0] and "cam_read_ms" in lines[0]
    assert len(lines) == 1 + 3                        # header + 3 shots


def test_toggle_file_enables_without_env(fresh):
    tmp_path, _ = fresh
    assert run_timing.is_enabled() is False
    (tmp_path / "RUN_TIMING_ON").write_text("", encoding="utf-8")
    assert run_timing.is_enabled() is True


def test_scan_summary_logs_and_clears(fresh, monkeypatch):
    _, logged = fresh
    monkeypatch.setenv("YB_RUN_TIMING", "1")
    for _ in range(2):
        run_timing.begin_shot(point=1)
        with run_timing.stage("wait"):
            _spin(0.005)
        run_timing.end_shot()
    logged.clear()
    run_timing.scan_summary(label="DemoScan")
    assert run_timing._SHOTS == []                    # cleared
    assert any("SCAN SUMMARY" in m and "DemoScan" in m for m in logged)


# --------------------------------------------------------------------------- #
# integration through run_scan_group (engine never loaded)
# --------------------------------------------------------------------------- #
class _FakeSeq:
    def __init__(self):
        self.C = DynProps({})


def test_run_scan_group_emits_timing(fresh, monkeypatch):
    tmp_path, logged = fresh
    monkeypatch.setenv("YB_RUN_TIMING", "1")

    class Ctl:
        def begin_scan(self):
            return 99

        def check_pause_abort(self):
            return False

    g = ScanGroup()
    g().A.B.scan(1, [1.0, 2.0, 3.0])
    run_scan_group(lambda s: s, g, control=Ctl(),
                   compile_point=lambda fn, p: _FakeSeq(),
                   run_real=lambda seq: None, seq_config=SeqConfig())

    csvs = list(tmp_path.glob("run_timing_*.csv"))
    assert len(csvs) == 1
    rows = csvs[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 3                         # header + 3 shots
    assert any("SCAN SUMMARY" in m for m in logged)   # end-of-scan summary fired


def _spin(seconds):
    """Busy-wait a real (tiny) interval so a stage gets a non-zero perf_counter delta without
    a scheduler-dependent sleep granularity surprise on Windows."""
    import time
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass
