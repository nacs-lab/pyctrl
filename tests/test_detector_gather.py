"""_Detector intensity path: the C-order gather must equal the original column-major matvec.

NO-HARDWARE. The detector's per-shot math was ``W.dot(img.astype(float).ravel("F"))`` with
``W``'s columns indexed column-major. It is now a gather over C-order indices. These tests pin
the equivalence against an INDEPENDENT reference computed from the grid + mask directly, so a
column-ordering mistake (which would silently score the wrong pixels and rearrange the wrong
atoms) cannot pass.
"""
import pytest

pytestmark = pytest.mark.no_hardware

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

import rearrange_runtime as rr


def _detector():
    det = rr._Detector.__new__(rr._Detector)
    det._log = lambda *a, **k: None
    det._W = None
    det._gidx = det._gwts = det._gseg = det._gempty = None
    det._img_shape = None
    det._key = None
    return det


def _reference(img, grid, box, sigma):
    """Independent per-site masked sum: straight 2-D indexing, no sparse matrix, no ravel."""
    mask = rr._fspecial_gaussian(box, sigma)
    half = box // 2
    H, W = img.shape
    out = np.zeros(grid.shape[0], dtype=float)
    for i in range(grid.shape[0]):
        y0, x0 = int(round(grid[i, 0])), int(round(grid[i, 1]))
        y_min, y_max = max(y0 - half, 1), min(y0 + half, H)
        x_min, x_max = max(x0 - half, 1), min(x0 + half, W)
        if y_min > y_max or x_min > x_max:
            continue
        my0, mx0 = y_min - (y0 - half), x_min - (x0 - half)
        acc = 0.0
        for dy in range(y_max - y_min + 1):
            for dx in range(x_max - x_min + 1):
                acc += mask[my0 + dy, mx0 + dx] * float(img[y_min + dy - 1, x_min + dx - 1])
        out[i] = acc
    return out


def _build(det, grid, shape):
    thr = np.full(grid.shape[0], 100.0)
    det._build(grid, thr, None, shape, key="test")


def test_gather_matches_independent_reference():
    shape = (64, 96)                      # non-square on purpose: H != W catches order bugs
    grid = np.array([[10, 12], [30, 50], [45, 80], [20, 33]], dtype=float)
    det = _detector()
    _build(det, grid, shape)
    assert det._gidx is not None, "gather should be active (self-check passed)"
    rng = np.random.default_rng(7)
    img = rng.integers(0, 4000, size=shape, dtype=np.uint16)
    got = det._apply(img)
    ref = _reference(img, grid, rr._BOX, rr._SIGMA)
    assert np.allclose(got, ref, rtol=0, atol=1e-9)


def test_gather_matches_matvec_fallback():
    """The fast path and the retained W matvec must agree element-for-element."""
    shape = (64, 96)
    grid = np.array([[10, 12], [30, 50], [45, 80]], dtype=float)
    det = _detector()
    _build(det, grid, shape)
    rng = np.random.default_rng(11)
    img = rng.integers(0, 4000, size=shape, dtype=np.uint16)
    fast = det._apply(img)
    det._gidx = None                       # force the fallback branch
    slow = det._apply(img)
    assert np.allclose(fast, slow, rtol=0, atol=1e-9)


def test_sites_clipped_off_image_score_zero():
    """A site whose whole box falls outside the frame must read 0, not its neighbour's pixels
    (np.add.reduceat reads the next element for an empty segment if not handled)."""
    shape = (40, 40)
    grid = np.array([[20, 20], [-50, -50], [25, 25]], dtype=float)
    det = _detector()
    _build(det, grid, shape)
    rng = np.random.default_rng(3)
    img = rng.integers(1000, 4000, size=shape, dtype=np.uint16)
    got = det._apply(img)
    ref = _reference(img, grid, rr._BOX, rr._SIGMA)
    assert np.allclose(got, ref, rtol=0, atol=1e-9)
    assert got[1] == 0.0


def test_non_contiguous_frame_is_handled():
    """A cropped/strided view must give the same answer as its contiguous copy."""
    shape = (64, 96)
    grid = np.array([[10, 12], [30, 50]], dtype=float)
    det = _detector()
    _build(det, grid, shape)
    rng = np.random.default_rng(5)
    big = rng.integers(0, 4000, size=(64, 192), dtype=np.uint16)
    view = big[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.allclose(det._apply(view), det._apply(np.ascontiguousarray(view)),
                       rtol=0, atol=1e-9)


def test_bits_unchanged_across_paths():
    """The externally visible product -- the bitstring -- must be identical either way."""
    shape = (64, 96)
    grid = np.array([[10, 12], [30, 50], [45, 80], [20, 33]], dtype=float)
    det = _detector()
    thr = np.array([50.0, 5e4, 50.0, 5e4])
    det._build(grid, thr, None, shape, key="test")
    det._thresholds = thr
    rng = np.random.default_rng(13)
    img = rng.integers(0, 4000, size=shape, dtype=np.uint16)
    fast = "".join("1" if b else "0" for b in det._apply(img) > det._thresholds)
    det._gidx = None
    slow = "".join("1" if b else "0" for b in det._apply(img) > det._thresholds)
    assert fast == slow
