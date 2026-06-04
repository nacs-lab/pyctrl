"""Phase-5 scan metadata for the dashboard: scan_summary.build_descriptor_summary (queue panel)
and scangroup_scan_config (DataManager live scan-info). NO-HARDWARE -- pure ScanGroup +
descriptor math, mirroring the LACScan the live backend runs.
"""

import json

import pytest

from scan_export import linspace, scangroup_to_descriptor
from scan_group import ScanGroup
from scan_summary import (build_descriptor_summary, scangroup_scan_config)

pytestmark = pytest.mark.no_hardware


def _lac_group():
    """Mirror YbScans/LACScan.py: a fixed g() override + a 1-D dim-1 sweep + runp."""
    g = ScanGroup()
    g().BlueMOT.LoadingTime = 0.4                                   # g() fixed override
    g().GreenMOT.CoolDown.HoldTime = 0.2
    g(1).GreenMOT.BiasCoilCurrent.Y.scan(1, linspace(0.24, 0.32, 17))   # dim-1, 17-pt sweep
    rp = g.runp()
    rp.NumPerGroup = 500
    rp.NumImages = 1
    rp.isInit = 0
    return g


# --------------------------------------------------------------------------- #
# Surface A -- the queue-panel summary (from the descriptor)
# --------------------------------------------------------------------------- #
class TestDescriptorSummary:
    def setup_method(self):
        g = _lac_group()
        self.desc = scangroup_to_descriptor(g, "TweezerLoadingSeq", label="LACScan")
        self.s = build_descriptor_summary(self.desc)

    def test_axis_extracted(self):
        ax = self.s["axes"]
        assert len(ax) == 1
        assert ax[0]["name"] == "GreenMOT.BiasCoilCurrent.Y"
        assert ax[0]["dim"] == 1 and ax[0]["npts"] == 17
        assert ax[0]["min"] == pytest.approx(0.24)
        assert ax[0]["max"] == pytest.approx(0.32)

    def test_set_params_are_g_overrides(self):
        sp = self.s["set_params"]
        assert sp["BlueMOT.LoadingTime"] == pytest.approx(0.4)
        assert sp["GreenMOT.CoolDown.HoldTime"] == pytest.approx(0.2)
        # a swept path is NOT a set-param
        assert "GreenMOT.BiasCoilCurrent.Y" not in sp

    def test_reps_use_stacknum_formula(self):
        # nseqs=17, NumPerGroup=500 -> StackNum=max(ceil(500/17),2)=30 -> total=510 (ybScanSummary)
        assert self.s["num_per_group"] == 500
        assert self.s["nseqs"] == 17
        assert self.s["total_per_group"] == 510

    def test_scan_name_from_label(self):
        assert self.s["scan_name"] == "LACScan"
        assert self.s["scan_filename"] == "LACScan"

    def test_accepts_json_string(self):
        s2 = build_descriptor_summary(json.dumps(self.desc))
        assert s2 == self.s

    def test_malformed_descriptor_degrades(self):
        s = build_descriptor_summary("{not json")
        assert s["axes"] == [] and s["set_params"] == {}


# --------------------------------------------------------------------------- #
# Surface B -- the DataManager scan-config fields (from the dispatched ScanGroup)
# --------------------------------------------------------------------------- #
class TestScanGroupScanConfig:
    def setup_method(self):
        self.g = _lac_group()
        self.cfg = scangroup_scan_config(
            self.g, scan_name="LACScan", expconfig={"Orca": {"ROI": [1000, 100, 2100, 2100]}})

    def test_base_vars_shape(self):
        vars_ = self.cfg["ScanGroup"]["base"]["vars"]
        assert vars_["size"] == [17]
        assert isinstance(vars_["params"], list) and len(vars_["params"]) == 1
        # the nested struct reaches the swept leaf
        leaf = vars_["params"][0]["GreenMOT"]["BiasCoilCurrent"]["Y"]
        assert len(leaf) == 17

    def test_base_vars_read_by_extract_logic(self):
        # Replicate the monitor's extract_scan_dims/_find_first_numeric contract after the
        # mat_reader JSON->ndarray normalization, so we know DataManager will find the axis.
        cfg = _json_roundtrip_with_arrays(self.cfg)
        name, vals = _extract_first_dim(cfg["ScanGroup"]["base"]["vars"])
        assert name == "GreenMOT.BiasCoilCurrent.Y"
        assert len(vals) == 17

    def test_base_params_carry_g_overrides(self):
        bp = self.cfg["ScanGroup"]["base"]["params"]
        assert bp["BlueMOT"]["LoadingTime"] == pytest.approx(0.4)

    def test_scan_name_is_uint16_codes(self):
        assert self.cfg["ScanName"]["scanname"] == [ord(c) for c in "LACScan"]

    def test_plotscale_default(self):
        assert self.cfg["PlotScale"] == pytest.approx(1.0)

    def test_expconfig_snapshot_embedded(self):
        assert self.cfg["expConfig"] == {"Orca": {"ROI": [1000, 100, 2100, 2100]}}

    def test_json_serializable(self):
        # the whole config must survive a JSON round-trip (it is written to disk as JSON)
        json.dumps(self.cfg)


# --------------------------------------------------------------------------- #
# helpers replicating the monitor-side consumers (kept local; yb_analysis isn't on path)
# --------------------------------------------------------------------------- #
def _json_roundtrip_with_arrays(cfg):
    import numpy as np

    def arrays(obj):
        if isinstance(obj, dict):
            return {k: arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
                return np.asarray(obj, dtype=float)
            return [arrays(x) for x in obj]
        return obj

    return arrays(json.loads(json.dumps(cfg)))


def _extract_first_dim(vars_):
    """Minimal port of extract_scan_dims/_find_first_numeric for the list (N-D) form."""
    import numpy as np
    params = vars_["params"]

    def find(obj, path):
        if isinstance(obj, np.ndarray) and obj.size > 1:
            return obj.ravel().astype(float), path
        if isinstance(obj, dict):
            for k, v in obj.items():
                r, rp = find(v, path + [k])
                if r is not None:
                    return r, rp
        return None, []

    vals, path = find(params[0], [])
    return ".".join(path), vals
