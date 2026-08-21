"""Per-shot LOADING-DEFOCUS sweep: one scan, N focal planes.

Covers the three pieces that make ``g().SLM.LoadingDefocus.scan(...)`` work end to end:
  * ``SlmScanSession.set_defocus`` -- re-declare the z4 Zernike + rewrite ON CHANGE only.
  * ``slm_runtime.defocus_scan_values`` -- per-point plane list (None when not swept).
  * ``slm_runtime.make_slm_defocus_pre_cb`` -- the per-shot pre_cb, driven with the SAME
    (seq_num, point_idx) signature the run loop passes (1-based point index, as awg_runtime).

Plus the byte invariant: the swept param is read by NO Step, so every point serializes
identically -- THE ONE RULE is untouched by the sweep (that half lives in
tests/test_ybseqs_build.py's reference; here we assert point-to-point equality of the
materialized params instead, which needs no engine).
"""

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_ROOT, os.path.join(_ROOT, "lib"), os.path.join(_ROOT, "YbExptCtrl"),
           os.path.join(_ROOT, "YbScans"), os.path.join(_ROOT, "YbSeqs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import slm_runtime                                      # noqa: E402
from scan_group import ScanGroup                        # noqa: E402
from devices.slm.slm_scan_session import SlmScanSession  # noqa: E402

pytestmark = pytest.mark.no_hardware


class FakeClient:
    def __init__(self):
        self.writes = []

    def acquire_lock(self, device, description="", timeout_s=60, block_timeout=30):
        pass

    def release_lock(self, device="all"):
        pass

    def heartbeat(self, device="all"):
        pass

    def write_loading_phase(self, phase_path, loading_zernike=None, name=None,
                            legacy_zerniked=False, baked_zernike=None):
        self.writes.append(list(loading_zernike or []))


def _session():
    c = FakeClient()
    s = SlmScanSession(c, lease_s=10.0, clock=lambda: 0.0)
    s.set_loading_pattern("33x33", "phase/33.pt", [0, 0, 0, 0, -5])
    s.begin()
    return c, s


def _group(values):
    g = ScanGroup()
    g().SLM.LoadingDefocus.scan(1, list(values))
    return g


# --------------------------------------------------------------------------- #
# SlmScanSession.set_defocus
# --------------------------------------------------------------------------- #
def test_set_defocus_rewrites_on_change_only():
    c, s = _session()
    assert c.writes == [[0, 0, 0, 0, -5]]               # the scan-start write
    assert s.set_defocus(-3.0) is True
    assert c.writes[-1] == [0, 0, 0, 0, -3.0]
    assert s.set_defocus(-3.0) is False                 # unchanged -> no second write
    assert len(c.writes) == 2


def test_set_defocus_noop_without_declared_pattern():
    c = FakeClient()
    s = SlmScanSession(c, clock=lambda: 0.0)            # lock-only session, nothing declared
    s.begin()
    assert s.set_defocus(-3.0) is False
    assert c.writes == []


def test_set_defocus_zero_clears_the_zernike():
    c, s = _session()
    assert s.set_defocus(0.0) is True
    assert c.writes[-1] == []                           # z4 = 0 -> no Zernike layered on


# --------------------------------------------------------------------------- #
# slm_runtime.defocus_scan_values
# --------------------------------------------------------------------------- #
def test_defocus_scan_values_reads_every_point():
    assert slm_runtime.defocus_scan_values(_group([-2.0, -1.0, 0.0])) == [-2.0, -1.0, 0.0]


def test_defocus_scan_values_none_when_not_swept():
    g = ScanGroup()
    g().BlueMOT.LoadingTime.scan(1, [0.2, 0.3])
    assert slm_runtime.defocus_scan_values(g) is None


# --------------------------------------------------------------------------- #
# the per-shot pre_cb
# --------------------------------------------------------------------------- #
def test_pre_cb_writes_this_points_plane():
    c, s = _session()
    cb = slm_runtime.make_slm_defocus_pre_cb(_group([-2.0, -1.0, 0.0]), s)
    assert cb is not None
    c.writes.clear()
    for point in (1, 2, 3, 3):                          # point index is 1-BASED; 3 repeats
        cb(0, point)
    assert c.writes == [[0, 0, 0, 0, -2.0], [0, 0, 0, 0, -1.0], []]


def test_pre_cb_absent_when_nothing_to_sweep():
    _c, s = _session()
    assert slm_runtime.make_slm_defocus_pre_cb(_group([-5.0]), s) is None       # single plane
    g = ScanGroup()
    g().BlueMOT.LoadingTime.scan(1, [0.2, 0.3])
    assert slm_runtime.make_slm_defocus_pre_cb(g, s) is None                    # not swept


def test_pre_cb_survives_a_bad_point_index():
    c, s = _session()
    cb = slm_runtime.make_slm_defocus_pre_cb(_group([-2.0, -1.0]), s)
    c.writes.clear()
    cb(0, 99)                                           # out of range -> ignored, no raise
    cb(0, None)
    assert c.writes == []


# --------------------------------------------------------------------------- #
# byte invariant: the swept plane is not a sequence parameter any Step reads
# --------------------------------------------------------------------------- #
def test_swept_points_differ_only_in_the_slm_node():
    g = _group([-10.0, 0.0])
    p1, p2 = g.getseq(1), g.getseq(2)
    assert p1 == {"SLM": {"LoadingDefocus": -10.0}}
    assert p2 == {"SLM": {"LoadingDefocus": 0.0}}
    assert set(p1) == {"SLM"} and set(p1["SLM"]) == {"LoadingDefocus"}
