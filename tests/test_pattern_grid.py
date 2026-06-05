"""NO-HARDWARE tests for the per-pattern registry path (loading-pattern affine migration).

Covers, with on-disk fakes (no SLM server, no monitor):
  * pattern_grid       -- registry/affine readers, the affine knm->camera math, and
                          resolve_pattern_calibration (grid + per-pattern thresholds).
  * scan_prep          -- resolve_calibration prefers the per-pattern registry, falls back to the
                          day folder; write_scan_config emits imagePatternsJson + per-pattern calib.
  * runner             -- _loading_patterns_json (port of ybLoadingPatternsJson).
  * rearrange_runtime  -- the mid-shot detector uses per-pattern grid+thresholds.
"""

import json
import os

import pytest

pytestmark = pytest.mark.no_hardware

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy.io import savemat   # noqa: E402


# =========================================================================== #
# fixtures: an on-disk registry (patterns dir + affine file)
# =========================================================================== #
def _make_registry(tmp_path, name="33x33", knm=((10, 10), (10, 30)),
                   thresholds=(5.0, 1.0e9), infidelities=(0.01, 0.02),
                   A=((0, 1, 0), (1, 0, 0))):
    """Write a fake registry: <patterns>/<name>/record.json + threshold.mat, and an affine file.

    The default affine A maps knm [x,y] -> camera [Y,X] = [y, x] (so knm [y,x]=[10,10] lands at
    camera [10,10]); identity-ish for a predictable grid."""
    patterns = tmp_path / "patterns"
    pdir = patterns / name
    pdir.mkdir(parents=True)
    with open(pdir / "record.json", "w") as f:
        json.dump({"name": name, "knm": [list(p) for p in knm]}, f)
    savemat(str(pdir / "threshold.mat"),
            {"thresholds": np.asarray(thresholds, dtype=float),
             "infidelities": np.asarray(infidelities, dtype=float)})
    affine = tmp_path / "affine_transform.json"
    with open(affine, "w") as f:
        json.dump({"current": {"A": [list(r) for r in A]}, "history": []}, f)
    return str(patterns), str(affine)


def _set_registry_env(monkeypatch, patterns_dir, affine_file):
    monkeypatch.setenv("YB_PATTERNS_DIR", patterns_dir)
    monkeypatch.setenv("YB_AFFINE_PATH", affine_file)


# =========================================================================== #
# pattern_grid: affine math
# =========================================================================== #
def test_affine_knm_to_camera_cropped():
    import pattern_grid as pg
    A = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)   # [x,y,1] -> [Y=y, X=x]
    knm = [[10, 10], [10, 30]]                           # [y, x]
    yx = pg._apply_affine_cropped(pg._knm_to_xy(knm), A, roi=[0, 0, 40, 40])
    assert yx.tolist() == [[10.0, 10.0], [10.0, 30.0]]
    # ROI offset is subtracted from [Y, X].
    yx2 = pg._apply_affine_cropped(pg._knm_to_xy(knm), A, roi=[5, 3, 40, 40])
    assert yx2.tolist() == [[10.0 - 3, 10.0 - 5], [10.0 - 3, 30.0 - 5]]


# =========================================================================== #
# pattern_grid: registry readers + resolve_pattern_calibration
# =========================================================================== #
def test_resolve_pattern_calibration(tmp_path, monkeypatch):
    import pattern_grid as pg
    pdir, affine = _make_registry(tmp_path)
    _set_registry_env(monkeypatch, pdir, affine)

    pc = pg.resolve_pattern_calibration("33x33", roi=[0, 0, 40, 40])
    assert pc is not None
    assert pc["n_sites"] == 2
    assert pc["grid"].tolist() == [[10.0, 10.0], [10.0, 30.0]]
    assert pc["thresholds"] == [5.0, 1.0e9]
    assert pc["infidelities"] == [0.01, 0.02]


def test_resolve_pattern_calibration_none_without_affine(tmp_path, monkeypatch):
    import pattern_grid as pg
    pdir, _ = _make_registry(tmp_path)
    monkeypatch.setenv("YB_PATTERNS_DIR", pdir)
    monkeypatch.setenv("YB_AFFINE_PATH", str(tmp_path / "nonexistent_affine.json"))
    assert pg.resolve_pattern_calibration("33x33", roi=[0, 0, 40, 40]) is None


def test_resolve_pattern_calibration_none_unknown_pattern(tmp_path, monkeypatch):
    import pattern_grid as pg
    pdir, affine = _make_registry(tmp_path)
    _set_registry_env(monkeypatch, pdir, affine)
    assert pg.resolve_pattern_calibration("does_not_exist", roi=[0, 0, 40, 40]) is None


# =========================================================================== #
# rearrange_runtime: mid-shot detector via the per-pattern source
# =========================================================================== #
def test_detector_uses_pattern_source(tmp_path, monkeypatch):
    from rearrange_runtime import _Detector
    pdir, affine = _make_registry(tmp_path)
    _set_registry_env(monkeypatch, pdir, affine)

    img = np.zeros((40, 40))
    img[5:14, 5:14] = 100.0          # light the box around site 0 (Y=10, X=10)

    det = _Detector("unused_day_root", pattern_name="33x33",
                    roi_provider=lambda: [0, 0, 40, 40])
    assert det.bits(img) == "10"     # site 0 fires, site 1 (thr 1e9) does not
    assert det._key[0] == "pattern"  # built from the per-pattern source


def test_detector_falls_back_to_day_folder(tmp_path, monkeypatch):
    import time
    from rearrange_runtime import _Detector
    # No registry env -> the pattern source is unavailable; a real day folder exists instead.
    monkeypatch.delenv("YB_PATTERNS_DIR", raising=False)
    monkeypatch.delenv("YB_AFFINE_PATH", raising=False)
    day = tmp_path / time.strftime("%Y%m%d")
    day.mkdir()
    (day / "gridLocations.txt").write_text("Y\tX\n10\t10\n10\t30\n")
    savemat(str(day / "threshold.mat"), {"thresholds": np.array([5.0, 1.0e9])})

    img = np.zeros((40, 40))
    img[5:14, 5:14] = 100.0

    det = _Detector(str(tmp_path), pattern_name="missing_pattern",
                    roi_provider=lambda: [0, 0, 40, 40])
    assert det.bits(img) == "10"
    assert det._key[0] == "day"      # fell back to the day folder


# =========================================================================== #
# scan_prep: resolve_calibration prefers per-pattern, falls back to day-folder
# =========================================================================== #
def test_scan_prep_resolve_calibration_prefers_pattern(tmp_path, monkeypatch):
    import scan_prep
    pdir, affine = _make_registry(tmp_path)
    _set_registry_env(monkeypatch, pdir, affine)

    day_dir = tmp_path / "day"
    day_dir.mkdir()
    image_patterns = [{"name": "33x33", "base_phase_path": "phase/33x33.pt", "order": "col"}]
    calib = scan_prep.resolve_calibration(str(day_dir), image_patterns, roi=[0, 0, 40, 40])
    assert calib["calibrationSource"] == "pattern:33x33"
    assert calib["initGridLocationsY"] == [10.0, 10.0]
    assert calib["initGridLocationsX"] == [10.0, 30.0]
    assert calib["initThresholds"] == [5.0, 1.0e9]
    assert calib["initInfidelities"] == [0.01, 0.02]


def test_scan_prep_resolve_calibration_falls_back_to_day(tmp_path, monkeypatch):
    import scan_prep
    monkeypatch.delenv("YB_PATTERNS_DIR", raising=False)
    monkeypatch.delenv("YB_AFFINE_PATH", raising=False)
    day_dir = tmp_path / "day"
    day_dir.mkdir()
    (day_dir / "gridLocations.txt").write_text("Y\tX\n10\t10\n10\t30\n")
    savemat(str(day_dir / "threshold.mat"), {"thresholds": np.array([5.0, 7.0])})
    # No pattern declared -> day-folder calibration (no calibrationSource key).
    calib = scan_prep.resolve_calibration(str(day_dir), image_patterns=None, roi=None)
    assert "calibrationSource" not in calib
    assert calib["initThresholds"] == [5.0, 7.0]


def test_write_scan_config_emits_imagepatternsjson(tmp_path, monkeypatch):
    import scan_prep
    pdir, affine = _make_registry(tmp_path)
    _set_registry_env(monkeypatch, pdir, affine)
    monkeypatch.setenv("YB_DATA_PREFIX", str(tmp_path))

    image_patterns = [{"name": "33x33", "base_phase_path": "phase/33x33.pt", "order": "col",
                       "legacy_zerniked": False}]
    path = scan_prep.write_scan_config(
        20260605120000, (40, 40), 2, is_init=0, num_per_group=10,
        image_patterns=image_patterns, roi=[0, 0, 40, 40])
    with open(path) as f:
        cfg = json.load(f)
    assert json.loads(cfg["imagePatternsJson"]) == image_patterns
    assert cfg["calibrationSource"] == "pattern:33x33"
    assert cfg["initThresholds"] == [5.0, 1.0e9]


# =========================================================================== #
# runner: _loading_patterns_json (port of ybLoadingPatternsJson)
# =========================================================================== #
def test_loading_patterns_json_from_warmup_kwargs():
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    import runner
    from scan_group import ScanGroup

    g = ScanGroup()
    rp = g.runp()
    rp.warmup_kwargs.initial_phase = "phase/33x33_uniform.pt"
    rp.warmup_kwargs.final_phase = "phase/3270_z4eq4.pt"
    rp.warmup_kwargs.extras.final_phase_zernike = [0, 0, 0, 0, -4]

    items = runner._loading_patterns_json(rp, num_images=2)
    assert items[0] == {"name": "33x33_uniform",
                        "base_phase_path": "phase/33x33_uniform.pt",
                        "order": "col", "legacy_zerniked": False}
    assert items[1]["name"] == "3270_z4eq4"
    assert items[1]["legacy_zerniked"] is True
    assert items[1]["baked_zernike"] == [0.0, 0.0, 0.0, 0.0, -4.0]


def test_loading_patterns_json_none_without_pattern():
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    import runner
    from scan_group import ScanGroup
    g = ScanGroup()
    assert runner._loading_patterns_json(g.runp(), num_images=1) is None
