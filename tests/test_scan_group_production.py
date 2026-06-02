"""Phase-4 W5 -- runp round-trip + production scan-shape smoke.

NO-HARDWARE: pure data-model math; never loads the engine or touches devices.

Two concerns, both from the real ``matlab_new/YbScans/*`` surface:

  * ``runp()`` round-trip -- the run parameters (``NumPerGroup`` / ``NumImages`` /
    ``Scramble`` / ``isInit`` / ``isHC`` / ``isGrid2`` / the ``AWGs`` list) are a DynProps
    shared per group. Confirm scalars, a list, nested keys, and the handle-aliasing all
    round-trip into ``dump()['runparam']`` and back via ``runp()``.

  * Production query surface -- build scans shaped like FreqPushOut399Scan /
    imagingScan / STIRAPAWGScan and exercise the W3 query API (``nseq`` / ``scandim`` /
    ``axisnum`` / ``get_fixed`` / ``get_vars`` / ``get_scanaxis``) the runner + analysis
    actually call. The element-for-element getseq() expansion of these exact shapes is
    pinned against real MATLAB in test_scan_group_oracle.py; here we lock the *query* side.
"""

import pytest

from scan_group import ScanGroup

pytestmark = pytest.mark.no_hardware


class TestRunParamRoundTrip:
    def test_scalar_runp_into_dump(self):
        g = ScanGroup()
        g.runp().NumPerGroup = 10000
        g.runp().NumImages = 2
        g.runp().Scramble = 1
        g.runp().isInit = 0
        g.runp().isHC = 0
        g.runp().isGrid2 = 0
        assert g.dump()["runparam"] == {
            "NumPerGroup": 10000, "NumImages": 2, "Scramble": 1,
            "isInit": 0, "isHC": 0, "isGrid2": 0,
        }

    def test_runp_read_back_via_runp(self):
        g = ScanGroup()
        g.runp().NumImages = 2
        # Reading the leaf with empty parens resolves the DynProps value.
        assert g.runp().NumImages() == 2

    def test_runp_is_one_shared_handle(self):
        # runp() returns the same DynProps every call (MATLAB: one per group).
        g = ScanGroup()
        rp = g.runp()
        g.runp().a = 3
        rp.b = 2
        assert g.runp().a() == 3
        assert g.runp().b() == 2
        assert g.dump()["runparam"] == {"a": 3, "b": 2}

    def test_runp_list_value(self):
        # The AWGs run-parameter is a list (MATLAB cell array).
        g = ScanGroup()
        g.runp().AWGs = ["AWG556", "AWG308"]
        assert g.dump()["runparam"] == {"AWGs": ["AWG556", "AWG308"]}

    def test_runp_list_value_is_copied(self):
        g = ScanGroup()
        names = ["AWG556"]
        g.runp().AWGs = names
        names.append("AWG308")            # mutate the caller's list afterwards
        assert g.dump()["runparam"]["AWGs"] == ["AWG556"]

    def test_runp_nested(self):
        g = ScanGroup()
        g.runp().detail.queue = 5
        g.runp().detail.name = "run-a"
        assert g.dump()["runparam"] == {"detail": {"queue": 5, "name": "run-a"}}

    def test_runp_independent_of_scan_params(self):
        # Run parameters live in their own store, not in the scan/base params.
        g = ScanGroup()
        g().a = 1
        g.runp().NumImages = 2
        d = g.dump()
        assert d["runparam"] == {"NumImages": 2}
        assert d["base"]["params"] == {"a": 1}

    def test_empty_runp(self):
        assert ScanGroup().dump()["runparam"] == {}


class TestProductionQuerySurface:
    def _spectrum399(self):
        # FreqPushOut399Scan shape: nested fixed + a 1-D scan on a 3-level path.
        g = ScanGroup()
        g().Pushout.Blue.Amp = 0.25
        g().Pushout.Blue.Freq.scan(1, [v * 1e6 for v in range(220, 361, 20)])
        g().Pushout.Time = 10e-3
        g.runp().NumPerGroup = 10000
        g.runp().Scramble = 1
        return g

    def _imaging_hist(self):
        # imagingScan shape: nested fixed + a 2-D grid.
        g = ScanGroup()
        g().Imag399.ExposureTime = 100e-3
        g().Imag399.FreqDetuning.scan(1, [-5e6, 0.0])
        g().Imag399.Amp.scan(2, [0.2, 0.3])
        g().Pushout.Time = 10e-3
        return g

    def test_spectrum399_shape(self):
        g = self._spectrum399()
        assert g.groupsize() == 1
        assert g.scandim(1) == 1
        assert g.nseq() == 8
        assert g.scansize(1) == 8
        assert g.axisnum(1, 1) == 1
        # get_fixed returns the merged fixed tree (the scanned Freq is NOT in it).
        assert g.get_fixed(1) == {"Pushout": {"Blue": {"Amp": 0.25}, "Time": 10e-3}}
        val, path = g.get_scanaxis(1, 1, 1)
        assert path == "Pushout.Blue.Freq"
        assert val == [v * 1e6 for v in range(220, 361, 20)]
        # first / last expansion points
        assert g.getseq(1)["Pushout"]["Blue"]["Freq"] == 220e6
        assert g.getseq(8)["Pushout"]["Blue"]["Freq"] == 360e6

    def test_imaging_hist_shape(self):
        g = self._imaging_hist()
        assert g.scandim(1) == 2
        assert g.nseq() == 4              # 2 x 2 grid
        assert g.axisnum(1, 1) == 1
        assert g.axisnum(1, 2) == 1
        # dim-1 varies fastest: FreqDetuning cycles within a fixed Amp.
        assert g.getseq(1)["Imag399"] == {"ExposureTime": 100e-3, "FreqDetuning": -5e6, "Amp": 0.2}
        assert g.getseq(2)["Imag399"] == {"ExposureTime": 100e-3, "FreqDetuning": 0.0, "Amp": 0.2}
        assert g.getseq(3)["Imag399"] == {"ExposureTime": 100e-3, "FreqDetuning": -5e6, "Amp": 0.3}
        params1, sz1 = g.get_vars(1, 1)
        assert params1 == {"Imag399": {"FreqDetuning": [-5e6, 0.0]}}
        assert sz1 == 2
        params2, sz2 = g.get_vars(1, 2)
        assert params2 == {"Imag399": {"Amp": [0.2, 0.3]}}
        assert sz2 == 2
        assert g.get_scanaxis(1, 2, "Imag399.Amp")[0] == [0.2, 0.3]

    def test_boolean_fixed_param_survives_expansion(self):
        # STIRAPAWGScan sets g().Pushout.STIRAP.ifReverse = true.
        g = ScanGroup()
        g().Pushout.STIRAP.ifReverse = True
        g().Pushout.STIRAP.gap.scan(1, [100e-9, 200e-9, 300e-9])
        assert g.nseq() == 3
        seq = g.getseq(2)
        assert seq["Pushout"]["STIRAP"]["ifReverse"] is True
        assert seq["Pushout"]["STIRAP"]["gap"] == 200e-9
