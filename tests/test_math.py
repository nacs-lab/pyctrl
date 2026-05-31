"""Port of matlab_new/lib/test/TestMath.m (the numeric helpers).

The SeqVal-node forms of ifelse/interpolate are exercised in test_seq_context;
here we pin the plain-number evaluation against MATLAB.
"""

import math

import pytest

from fld import fld
from cld import cld
from ifelse import ifelse
from interpolate import interpolate
from rabi_line import rabi_line

pytestmark = pytest.mark.no_hardware


class TestMath:
    def test_fldcld(self):
        assert fld(5.5, 2.2) == 2
        assert cld(5.5, 2.2) == 3
        # fld(6.0, 0.1) == 59 is broken in MATLAB too; only cld is checked.
        assert cld(6.0, 0.1) == 60
        assert fld(7.3, 5.5) == 1
        assert cld(7.3, 5.5) == 2
        assert fld(9.0, 3.0) == 3
        assert cld(9.0, 3.0) == 3
        assert fld(-0.097076000000000023, 0.0000020000000000000003) == -48539
        assert cld(-0.097076000000000023, 0.0000020000000000000003) == -48538

    def test_ifelse(self):
        assert ifelse(True, 1, 2) == 1
        assert ifelse(False, 1, 2) == 2

    def test_interpolate(self):
        vals = [0, 9, 2, 3, -10, 5]
        assert interpolate(-1, 0, 5, vals) == 0
        assert interpolate(0, 0, 5, vals) == 0
        assert interpolate(1, 0, 5, vals) == 9
        assert interpolate(2, 0, 5, vals) == 2
        assert interpolate(3, 0, 5, vals) == 3
        assert interpolate(4, 0, 5, vals) == -10
        assert interpolate(5, 0, 5, vals) == 5
        assert interpolate(6, 0, 5, vals) == 5
        assert interpolate(2.5, 0, 5, vals) == 2.5

    def test_rabi_line(self):
        # Zero Omega
        assert rabi_line(0, 0, 0) == 0
        assert rabi_line(0, 1, 0) == 0
        assert rabi_line(1, 0, 0) == 0
        assert rabi_line(1, 1, 0) == 0
        # Zero t
        assert rabi_line(0, 0, 1) == 0
        assert rabi_line(1, 0, 1) == 0
        # Zero det (on resonance)
        assert rabi_line(0, 1, 1) == math.sin(0.5) ** 2
        assert rabi_line(0, 2, 1) == math.sin(1) ** 2
        assert rabi_line(0, 3, 1) == math.sin(1.5) ** 2
        assert rabi_line(0, 1, 2) == math.sin(1) ** 2
        assert rabi_line(0, 1, 3) == math.sin(1.5) ** 2
        # On resonance Pi time
        assert rabi_line(0, math.pi, 1) == pytest.approx(1, abs=1e-15)
        assert rabi_line(0, math.pi / 2, 2) == pytest.approx(1, abs=1e-15)
        # On resonance 2Pi time
        assert rabi_line(0, math.pi, 2) == pytest.approx(0, abs=1e-16)
        assert rabi_line(0, math.pi / 2, 4) == pytest.approx(0, abs=1e-16)
        # Node at Pi time
        assert rabi_line(math.sqrt(3), math.pi, 1) == pytest.approx(0, abs=1e-16)
        assert rabi_line(math.sqrt(3) * 2, math.pi / 2, 2) == pytest.approx(0, abs=1e-16)
        assert rabi_line(math.sqrt(15), math.pi, 1) == pytest.approx(0, abs=1e-16)
        assert rabi_line(math.sqrt(15) * 2, math.pi / 2, 2) == pytest.approx(0, abs=1e-16)
        assert rabi_line(-math.sqrt(3), math.pi, 1) == pytest.approx(0, abs=1e-16)
        assert rabi_line(-math.sqrt(15), math.pi, 1) == pytest.approx(0, abs=1e-16)

        assert rabi_line(math.sqrt(2.5 ** 2 - 1), math.pi, 1) == pytest.approx(0.5 * 0.16, abs=1e-15)
        assert rabi_line(math.sqrt(2.5 ** 2 - 1) * 2, math.pi / 2, 2) == pytest.approx(0.5 * 0.16, abs=1e-15)
