"""Unit tests for the autocal (continuous background-calibration) package -- no hardware/engine.

Covers the Phase-0 foundation end to end with injected fakes (no backend, no SLM, no yb_analysis,
no data files):
  * pooling math (inverse-variance combine of short yielded runs) + the cross-run accumulator;
  * ledger atomic IO + corrupt-file safety;
  * ingest decision logic (insufficient -> fit -> auto-apply in-band / flag out-of-band / flag
    edge-pinned / reference-only no-apply) + commit_apply + rollback;
  * the change journal + dashboard command queue (settings, rollback intent);
  * the rotation scheduler (active-overdue first, auto-cycle non-home only when idle);
  * descriptor building (background+cycle+rep+self-asserting loading_phase) + label parse;
  * a full Controller.tick() that pools a synthetic Lorentzian dip, fits, auto-applies, journals.

    pytest pyctrl/tests/test_autocal.py
"""
import json
import math
import os
import types

import pytest

from autocal import paths, ledger, pooling, fit, rollback, rotation, submit, controller

pytestmark = pytest.mark.no_hardware


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def caldir(tmp_path, monkeypatch):
    d = tmp_path / "calibration"
    monkeypatch.setenv("YB_AUTOCAL_DIR", str(d))
    return str(d)


def _lorentz_dip(x, center, hw, baseline=0.9, depth=0.6):
    return [baseline - depth / (1.0 + ((xi - center) / hw) ** 2) for xi in x]


def _grid(lo, step, hi):
    n = int(round((hi - lo) / step)) + 1
    return [lo + i * step for i in range(n)]


# =========================================================================== #
# pooling
# =========================================================================== #
def test_scan_point_weights_skips_nonfinite_and_floors_sem():
    mean = [0.5, float("nan"), 0.8]
    sem = [0.1, 0.1, 0.0]            # last sem 0 -> floored to min_sem
    w0, w1 = pooling.scan_point_weights(mean, sem, min_sem=0.02)
    assert w0[1] == 0.0 and w1[1] == 0.0          # NaN mean -> no weight
    assert w0[0] == pytest.approx(1 / 0.1 ** 2)
    assert w0[2] == pytest.approx(1 / 0.02 ** 2)  # floored
    assert w1[0] == pytest.approx(0.5 / 0.1 ** 2)


def test_combine_sums_recovers_and_tightens():
    # two identical measurements of mean 0.4, sem 0.1 -> mean 0.4, sem 0.1/sqrt(2)
    w0a, w1a = pooling.scan_point_weights([0.4], [0.1])
    w0 = [w0a[0] * 2]
    w1 = [w1a[0] * 2]
    m, s = pooling.combine_sums(w0, w1)
    assert m[0] == pytest.approx(0.4)
    assert s[0] == pytest.approx(0.1 / math.sqrt(2), rel=1e-6)


def test_combine_unmeasured_point_is_nonfinite():
    m, s = pooling.combine_sums([0.0], [0.0])
    assert not math.isfinite(m[0]) and not math.isfinite(s[0])


def test_scan_to_points_adapter_uses_injected_reader():
    def reader(sid):
        return {"sweep": {"values": [[1.0, 2.0, 3.0]]},
                "summary": {"survival_mean": [0.9, 0.3, 0.9], "survival_sem": [0.05, 0.05, 0.05]},
                "n_shots": 30}
    x, mean, sem, n = pooling.scan_to_points("sid", reader=reader)
    assert x == [1.0, 2.0, 3.0] and mean[1] == 0.3 and n == 30


def test_scan_to_points_empty_on_no_params():
    assert pooling.scan_to_points("x", reader=lambda _s: {"sweep": {"values": []},
                                                          "summary": {}}) == ([], [], [], 0)


# =========================================================================== #
# accumulator (combine many short runs)
# =========================================================================== #
def test_accumulate_pools_shots_across_runs_until_ready():
    x = _grid(107.5e6, 0.01e6, 107.9e6)
    y = _lorentz_dip(x, 107.7e6, 0.03e6)
    sem = [0.08] * len(x)
    acc = ledger.default_accumulator()
    for i in range(5):
        acc, info = ledger.accumulate(acc, x, y, sem, 40, "sid%d" % i, stamp="t%d" % i)
        assert info["action"] in ("open", "added")
    assert acc["shots"] == 200
    assert ledger.accumulator_ready(acc, 200)
    assert len(acc["scan_ids"]) == 5
    xs, mean, _ = ledger.accumulator_curve(acc)
    # pooled curve still has the dip at the right place
    assert xs == pytest.approx(x)
    assert mean[mean.index(min(mean))] < 0.5


def test_accumulate_grid_mismatch_resets():
    acc = ledger.default_accumulator()
    acc, _ = ledger.accumulate(acc, [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], 10, "a")
    acc, info = ledger.accumulate(acc, [1.0, 2.0], [0.5, 0.5], [0.1, 0.1], 10, "b")
    assert info["action"] == "reset"
    assert acc["scan_ids"] == ["b"] and acc["shots"] == 10


# =========================================================================== #
# ledger IO
# =========================================================================== #
def test_ledger_save_load_roundtrip(caldir):
    led = ledger.default_ledger(home_pattern="33x33_uniform")
    rotation.seed_pattern_cals(led, "33x33_uniform")
    led["patterns"]["33x33_uniform"]["eligible"] = True
    p = ledger.save(led, stamp="2026-06-22T10:00:00")
    assert os.path.exists(p)
    led2 = ledger.load()
    assert led2["settings"]["home_pattern"] == "33x33_uniform"
    assert "556mj0" in led2["patterns"]["33x33_uniform"]["cals"]


def test_ledger_corrupt_file_returns_default(caldir):
    paths.ensure_dir()
    with open(paths.ledger_path(), "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    led = ledger.load()
    assert led["_version"] == ledger.SCHEMA_VERSION and led["patterns"] == {}


# =========================================================================== #
# ingest decision logic
# =========================================================================== #
def _feed_until_fit(led, cal_key, pattern, center, baseline_center, *, band=50e3,
                    applies=True, hw=0.03e6, edge=False):
    cal_def = dict(rotation.CAL_DEFS[cal_key])
    cal_def["applies_config"] = applies
    # pre-seed the cal entry with a baseline so drift is judged
    ce = ledger.ensure_cal(led, pattern, cal_key, cadence_s=cal_def["cadence_s"],
                           reps_per_run=cal_def["reps_per_run"],
                           target_shots=cal_def["target_shots"], ref=baseline_center,
                           baseline=baseline_center, band=band,
                           applies_config=applies, config_key=cal_def["config_key"])
    lo, hi = (center - 0.18e6, center + 0.18e6)
    if edge:                      # put the true center right at the low edge of the window
        lo, hi = center, center + 0.4e6
    x = _grid(lo, 0.01e6, hi)
    y = _lorentz_dip(x, center, hw)
    sem = [0.07] * len(x)
    out = None
    # feed short runs until the pool reaches target_shots and a fit is cut (don't run past it --
    # a fit resets the accumulator, so a further run would read back as 'accumulated').
    for i in range(12):
        led, out = controller.ingest_scan(led, pattern=pattern, cal_key=cal_key, cal_def=cal_def,
                                          x=x, mean=y, sem=sem, n_shots=40, scan_id="s%d" % i,
                                          stamp="t%d" % i, auto_apply=True)
        if out["action"] in ("fit", "fit_failed"):
            break
    return led, out


def test_ingest_insufficient_then_fit_and_apply_in_band():
    led = ledger.default_ledger(home_pattern="P")
    led, out = _feed_until_fit(led, "556mj0", "P", center=107.71e6, baseline_center=107.70e6)
    assert out["action"] == "fit"
    assert out["fit"]["center"] == pytest.approx(107.71e6, abs=10e3)
    assert out["apply"] is not None and out["apply"]["config_key"] == "Resonance556mj0Freq"
    # commit it and check ref moved + history marked applied
    led = controller.commit_apply(led, out["apply"], change_id="chg-1", stamp="tX")
    ce = led["patterns"]["P"]["cals"]["556mj0"]
    assert ce["ref"] == pytest.approx(107.71e6, abs=10e3)
    assert ce["history"][-1]["applied"] is True


def test_ingest_out_of_band_flags_not_applies():
    led = ledger.default_ledger(home_pattern="P")
    # center 300 kHz from baseline, band 50 kHz -> out of band
    led, out = _feed_until_fit(led, "556mj0", "P", center=108.00e6, baseline_center=107.70e6,
                               band=50e3)
    assert out["status"] == "out_of_band"
    assert out["apply"] is None
    assert out["alert"] is not None and out["alert"]["id"].startswith("oob::")


def test_ingest_edge_pinned_flags():
    led = ledger.default_ledger(home_pattern="P")
    led, out = _feed_until_fit(led, "556mj0", "P", center=107.70e6, baseline_center=107.70e6,
                               edge=True)
    assert out["apply"] is None
    assert out["alert"] is not None and out["alert"]["id"].startswith("edge::")


def test_ingest_reference_cal_never_applies():
    led = ledger.default_ledger(home_pattern="P")
    led, out = _feed_until_fit(led, "556mj1", "P", center=105.0e6, baseline_center=105.0e6,
                               band=200e3, applies=False, hw=0.3e6)
    assert out["action"] == "fit"
    assert out["apply"] is None          # reference-only: trend, never write config


def test_ingest_low_loading_raises_alert():
    led = ledger.default_ledger(home_pattern="P")
    cal_def = rotation.CAL_DEFS["556mj0"]
    x = _grid(107.6e6, 0.01e6, 107.8e6)
    led, out = controller.ingest_scan(led, pattern="P", cal_key="556mj0", cal_def=cal_def,
                                      x=x, mean=_lorentz_dip(x, 107.7e6, 0.03e6),
                                      sem=[0.07] * len(x), n_shots=20, scan_id="s",
                                      stamp="t", loading_mean=0.02)
    assert out["alert"]["id"].startswith("loading::")


# =========================================================================== #
# rollback journal + command queue
# =========================================================================== #
def test_change_journal_and_revert(caldir):
    rollback.record_change(change_id="chg-1", pattern="P", cal="556mj0",
                           config_key="Resonance556mj0Freq", old=107.70e6, new=107.71e6,
                           scan_id="s1", kind="apply", ts="t1")
    changes = rollback.list_changes()
    assert len(changes) == 1
    rev = rollback.build_revert(changes[0], change_id="chg-2", ts="t2")
    assert rev["old"] == 107.71e6 and rev["new"] == 107.70e6 and rev["reverts"] == "chg-1"
    rollback.record_change(**{k: rev[k] for k in ("change_id", "pattern", "cal", "config_key",
                                                  "old", "new", "scan_id")},
                           kind="revert", reverts=rev["reverts"], ts="t2")
    all_changes = rollback.list_changes()
    # the revert event references chg-1
    assert "chg-1" in rollback.reverted_ids(all_changes)
    assert rollback.current_values(all_changes)[("P", "556mj0", "Resonance556mj0Freq")] == 107.70e6


def test_command_queue_pending_and_process_settings(caldir):
    rollback.enqueue_command(command_id="c1", type="set_setting",
                             args={"key": "auto_apply", "value": False}, ts="t")
    pend = rollback.pending_commands(0)
    assert len(pend) == 1
    led = ledger.default_ledger(home_pattern="P")
    led, intents = controller.process_command(led, pend[0])
    assert led["settings"]["auto_apply"] is False and intents == []


def test_process_command_rollback_makes_revert_intent():
    led = ledger.default_ledger()
    orig = {"change_id": "chg-1", "pattern": "P", "cal": "556mj0",
            "config_key": "Resonance556mj0Freq", "old": 1.0, "new": 2.0}
    cmd = {"type": "rollback", "args": {"change_id": "chg-1"}}
    led, intents = controller.process_command(led, cmd, changes_lookup={"chg-1": orig}.get)
    assert intents and intents[0]["kind"] == "revert" and intents[0]["orig"] is orig


def test_process_command_unknown_is_noop():
    led = ledger.default_ledger()
    led, intents = controller.process_command(led, {"type": "bogus"})
    assert intents and intents[0]["kind"] == "noop"


# =========================================================================== #
# rotation scheduler
# =========================================================================== #
def _eligible_ledger():
    led = ledger.default_ledger(home_pattern="home")
    led["settings"]["cycle_patterns"] = ["home", "other"]
    led["settings"]["active_pattern"] = "home"
    for p in ("home", "other"):
        rotation.seed_pattern_cals(led, p)
        led["patterns"][p]["eligible"] = True
    return led


def test_select_next_prefers_active_overdue():
    led = _eligible_ledger()
    plan = rotation.select_next(led, "2026-06-22T12:00:00", controller.iso_to_epoch)
    assert plan["pattern"] == "home" and plan["requires_switch"] is False


def test_select_next_auto_cycles_other_when_home_fresh_and_idle():
    led = _eligible_ledger()
    # mark all home cals fresh (just ran), leave 'other' never-run
    for ce in led["patterns"]["home"]["cals"].values():
        ce["last_run_ts"] = "2026-06-22T11:59:30"
    plan = rotation.select_next(led, "2026-06-22T12:00:00", controller.iso_to_epoch,
                                foreground_idle=True)
    assert plan["pattern"] == "other" and plan["requires_switch"] is True
    assert plan["restore_home_after"] is True


def test_select_next_no_cycle_when_foreground_busy():
    led = _eligible_ledger()
    for ce in led["patterns"]["home"]["cals"].values():
        ce["last_run_ts"] = "2026-06-22T11:59:30"
    plan = rotation.select_next(led, "2026-06-22T12:00:00", controller.iso_to_epoch,
                                foreground_idle=False)
    # home is fresh and we can't switch while foreground busy -> nothing
    assert plan is None


def test_select_next_disabled_lane_returns_none():
    led = _eligible_ledger()
    led["settings"]["lane_enabled"] = False
    assert rotation.select_next(led, "t", controller.iso_to_epoch) is None


def test_check_eligibility_flags_missing_wiring():
    ok, reasons = rotation.check_eligibility(
        "P", checker=lambda _p: {"record": True, "threshold": False})
    assert ok is False and any("threshold" in r for r in reasons)
    ok, _ = rotation.check_eligibility("P", checker=lambda _p: {"record": True, "threshold": True})
    assert ok is True


# =========================================================================== #
# submit / descriptor building
# =========================================================================== #
def test_parse_label_roundtrip():
    lbl = submit.make_label("556mj0", "33x33_uniform")
    assert submit.parse_label(lbl) == ("556mj0", "33x33_uniform")
    assert submit.parse_label("PushoutSurvival_556") is None


def test_build_descriptor_sets_background_cycle_rep_and_phase():
    captured = {}

    class _RP:
        pass

    class _G:
        def __init__(self):
            self._rp = _RP()

        def runp(self):
            return self._rp

    def fake_build(**kw):
        captured["build_kwargs"] = kw
        return _G()

    def fake_to_desc(g, seq, opts=None, label=None, description=None, background=False, cycle=True):
        captured["g"] = g
        return {"seq": seq, "opts": opts, "label": label, "description": description,
                "background": background, "cycle": cycle}

    cal_def = rotation.CAL_DEFS["556mj0"]
    desc = submit.build_descriptor("556mj0", cal_def, "47x47_uniform", reps=20,
                                   loading_phase="phase/47x47_uniform.pt", loading_defocus=-5,
                                   cycle=True, requires_switch=True,
                                   build_fn=fake_build, to_descriptor=fake_to_desc)
    assert desc["background"] is True and desc["cycle"] is True
    assert desc["opts"] == {"rep": 20}
    assert desc["label"] == "autocal::556mj0::47x47_uniform"
    assert captured["build_kwargs"] == {"mj": 0}
    # self-asserting pattern was wired onto the runp
    assert captured["g"].runp().loading_phase == "phase/47x47_uniform.pt"
    assert captured["g"].runp().loading_defocus == -5
    assert "[auto-calibration]" in desc["description"]


def test_build_descriptor_home_pattern_no_phase_write():
    class _G:
        def __init__(self):
            self._rp = types.SimpleNamespace()

        def runp(self):
            return self._rp

    g = _G()
    submit.build_descriptor("556mj0", rotation.CAL_DEFS["556mj0"], "home", reps=10,
                            build_fn=lambda **kw: g, to_descriptor=lambda *a, **k: {})
    assert not hasattr(g.runp(), "loading_phase")  # home pattern: no extra SLM write


# =========================================================================== #
# full Controller.tick() (fakes for all IO; real pooling + real fit)
# =========================================================================== #
def test_controller_tick_pools_fits_applies_and_journals(caldir):
    pattern, cal_key = "home", "556mj0"
    x = _grid(107.6e6, 0.01e6, 107.85e6)
    y = _lorentz_dip(x, 107.74e6, 0.03e6)
    scan_data = {"x": x, "mean": y, "sem": [0.07] * len(x), "n_shots": 60, "loading_mean": 0.45}

    # backend hands back one completed mj0 scan per tick (different scan_id each time)
    state = {"i": 0}

    def query_backend():
        state["i"] += 1
        return {"foreground_idle": True,
                "completed": [{"scan_id": "S%03d" % state["i"],
                               "label": submit.make_label(cal_key, pattern)}],
                "queued_labels": [], "lane_enabled": True}

    writes = []

    def write_config(p, key, value):
        writes.append((p, key, value))
        return True

    submits = []

    led = ledger.default_ledger(home_pattern=pattern)
    rotation.seed_pattern_cals(led, pattern)
    led["patterns"][pattern]["eligible"] = True
    led["patterns"][pattern]["cals"][cal_key]["baseline"] = 107.735e6
    led["patterns"][pattern]["cals"][cal_key]["ref"] = 107.735e6
    ledger.save(led)

    ctrl = controller.Controller(query_backend=query_backend,
                                  read_scan=lambda sid: dict(scan_data),
                                  submit=lambda plan: submits.append(plan),
                                  write_config=write_config,
                                  clock=_fixed_clock())
    # target_shots is 200 -> need ceil(200/60)=4 ticks to reach the fit
    for _ in range(5):
        led = ctrl.tick()

    # auto-applied the in-band center
    assert writes, "expected a config write after pooling reached target shots"
    p, key, value = writes[-1]
    assert key == "Resonance556mj0Freq"
    assert value == pytest.approx(107.74e6, abs=15e3)
    # journaled for rollback
    changes = rollback.list_changes()
    assert any(c.get("config_key") == "Resonance556mj0Freq" and c.get("kind") == "apply"
               for c in changes)
    # ledger persisted with the new ref + alive runtime
    led2 = ledger.load()
    assert led2["runtime"]["controller_alive"] is True
    assert led2["patterns"][pattern]["cals"][cal_key]["ref"] == pytest.approx(107.74e6, abs=15e3)
    # idempotent: a scan already ingested is not double-counted (accumulator reset after fit)
    assert led2["patterns"][pattern]["cals"][cal_key]["accumulator"]["shots"] < 200


def test_controller_tick_executes_dashboard_rollback(caldir):
    # seed a prior applied change + a queued rollback command; tick should revert config.
    rollback.record_change(change_id="chg-000001", pattern="home", cal="556mj0",
                           config_key="Resonance556mj0Freq", old=107.70e6, new=107.74e6,
                           scan_id="S1", kind="apply", ts="t0")
    rollback.enqueue_command(command_id="cmd1", type="rollback",
                             args={"change_id": "chg-000001"}, ts="t1")
    led = ledger.default_ledger(home_pattern="home")
    rotation.seed_pattern_cals(led, "home")
    led["patterns"]["home"]["cals"]["556mj0"]["ref"] = 107.74e6
    ledger.save(led)

    writes = []
    ctrl = controller.Controller(
        query_backend=lambda: {"foreground_idle": True, "completed": [], "queued_labels": []},
        read_scan=lambda sid: None,
        write_config=lambda p, k, v: writes.append((p, k, v)) or True,
        clock=_fixed_clock())
    ctrl.tick()
    assert writes and writes[-1] == ("home", "Resonance556mj0Freq", 107.70e6)
    # a revert event was journaled
    assert any(c.get("kind") == "revert" for c in rollback.list_changes())
    led2 = ledger.load()
    assert led2["patterns"]["home"]["cals"]["556mj0"]["ref"] == 107.70e6


def _fixed_clock():
    """A monotonically-advancing fake clock (avoids real wall-clock in tests)."""
    from datetime import datetime, timedelta
    base = datetime(2026, 6, 22, 12, 0, 0)
    state = {"n": 0}

    def _clock():
        state["n"] += 1
        return base + timedelta(seconds=state["n"])
    return _clock
