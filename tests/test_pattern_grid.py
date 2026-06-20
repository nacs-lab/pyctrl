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
def _gauss_fits_struct(gauss_params):
    """Build the MATLAB-struct-array 'gaussFitsStruct' the way hist_init.save_calibration_outputs
    does: a (M,) struct array with object field 'params' (a (6,) vector or empty [] per site)."""
    gs = np.empty(len(gauss_params), dtype=[("params", "O")])
    for s, p in enumerate(gauss_params):
        gs[s]["params"] = np.asarray(p, dtype=float) if p is not None else np.array([])
    return gs


def _make_registry(tmp_path, name="33x33", knm=((10, 10), (10, 30)),
                   thresholds=(5.0, 1.0e9), infidelities=(0.01, 0.02),
                   A=((0, 1, 0), (1, 0, 0)), gauss_params=None):
    """Write a fake registry: <patterns>/<name>/record.json + threshold.mat, and an affine file.

    The default affine A maps knm [x,y] -> camera [Y,X] = [y, x] (so knm [y,x]=[10,10] lands at
    camera [10,10]); identity-ish for a predictable grid. ``gauss_params`` (per-site
    [mu_e,s_e,A_e,mu_a,s_a,A_a] or None) writes a 'gaussFitsStruct' for the posterior path."""
    patterns = tmp_path / "patterns"
    pdir = patterns / name
    pdir.mkdir(parents=True)
    with open(pdir / "record.json", "w") as f:
        json.dump({"name": name, "knm": [list(p) for p in knm]}, f)
    mat = {"thresholds": np.asarray(thresholds, dtype=float),
           "infidelities": np.asarray(infidelities, dtype=float)}
    if gauss_params is not None:
        mat["gaussFitsStruct"] = _gauss_fits_struct(gauss_params)
    savemat(str(pdir / "threshold.mat"), mat)
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
# posterior path: gaussFitsStruct loading + _atom_posterior + detector.probs()
# =========================================================================== #
def test_atom_posterior_matches_mixture_formula():
    from rearrange_runtime import _atom_posterior
    # Two well-separated peaks; intensity right on the atom peak -> posterior ~1; on the empty
    # peak -> ~0. params = [mu_e, s_e, A_e, mu_a, s_a, A_a].
    params = [0.0, 5.0, 0.5, 100.0, 10.0, 0.5]
    post = _atom_posterior(np.array([0.0, 100.0, 50.0]), [params, params, params])
    assert post[0] < 0.01            # at the empty peak
    assert post[1] > 0.99            # at the atom peak
    assert 0.0 <= post[2] <= 1.0
    # Manual Bayes check at the crossover-ish point.
    def _n(x, mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    pe, pa = 0.5 * _n(50.0, 0.0, 5.0), 0.5 * _n(50.0, 100.0, 10.0)
    assert abs(post[2] - pa / (pe + pa)) < 1e-9


def test_atom_posterior_degenerate_site_is_zero():
    from rearrange_runtime import _atom_posterior
    good = [0.0, 5.0, 0.5, 100.0, 10.0, 0.5]
    post = _atom_posterior(np.array([100.0, 100.0, 100.0]),
                           [good, None, [0.0, 0.0, 0.5, 100.0, 0.0, 0.5]])  # None + s=0
    assert post[0] > 0.99            # valid fit -> real posterior
    assert post[1] == 0.0            # missing params -> 0.0
    assert post[2] == 0.0            # degenerate (sigma 0) -> 0.0


def test_detector_probs_uses_gauss_fits(tmp_path, monkeypatch):
    from rearrange_runtime import _Detector
    # Site 0 lit (real fit, atom peak ~100); site 1 unlit + DEGENERATE fit (empty params) -> 0.0.
    gp = [[0.0, 5.0, 0.5, 100.0, 20.0, 0.5], None]
    pdir, affine = _make_registry(tmp_path, gauss_params=gp)
    _set_registry_env(monkeypatch, pdir, affine)

    img = np.zeros((40, 40))
    img[5:14, 5:14] = 100.0          # light the box around site 0 (Y=10, X=10)

    det = _Detector("unused_day_root", pattern_name="33x33",
                    roi_provider=lambda: [0, 0, 40, 40])
    probs = det.probs(img)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] > 0.5            # lit site with a real fit -> present
    assert probs[1] == 0.0           # degenerate fit -> 0.0 (uncertain treated as empty)
    # Rounding the probs at 0.5 reproduces the hard bitstring (what the server does for now).
    assert "".join("1" if p >= 0.5 else "0" for p in probs) == det.bits(img) == "10"


def test_detector_probs_falls_back_to_hard_cut_without_gauss_fits(tmp_path, monkeypatch):
    from rearrange_runtime import _Detector
    # Registry threshold.mat WITHOUT gaussFitsStruct -> probs fall back to 1.0/0.0 (intensity>thr).
    pdir, affine = _make_registry(tmp_path)        # no gauss_params
    _set_registry_env(monkeypatch, pdir, affine)

    img = np.zeros((40, 40))
    img[5:14, 5:14] = 100.0

    det = _Detector("unused_day_root", pattern_name="33x33",
                    roi_provider=lambda: [0, 0, 40, 40])
    assert det.probs(img) == [1.0, 0.0]            # same as bits() "10", encoded as floats


def test_resolve_pattern_calibration_surfaces_gauss_params(tmp_path, monkeypatch):
    import pattern_grid as pg
    gp = [[0.0, 5.0, 0.5, 100.0, 20.0, 0.5], [1.0, 2.0, 0.5, 50.0, 5.0, 0.5]]
    pdir, affine = _make_registry(tmp_path, gauss_params=gp)
    _set_registry_env(monkeypatch, pdir, affine)

    pc = pg.resolve_pattern_calibration("33x33", roi=[0, 0, 40, 40])
    assert pc["gauss_params"] is not None and len(pc["gauss_params"]) == 2
    assert list(pc["gauss_params"][0]) == gp[0]
    # And the standalone reader returns the same per-site params.
    params = pg.read_gauss_params(str(tmp_path / "patterns" / "33x33" / "threshold.mat"))
    assert list(params[1]) == gp[1]


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


def test_loading_patterns_json_from_loading_phase():
    """A non-rearrange scan that overrides the loading hologram via runp().loading_phase
    (e.g. LACScan) must declare it -- a single base-phase entry, defocus NOT baked in."""
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
    rp.loading_phase = "phase/47x47_uniform.pt"
    rp.loading_defocus = -5    # write-only; extraction is defocus-independent

    items = runner._loading_patterns_json(rp, num_images=1)
    assert items == [{"name": "47x47_uniform",
                      "base_phase_path": "phase/47x47_uniform.pt",
                      "order": "col", "legacy_zerniked": False}]


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
