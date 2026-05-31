"""Port of matlab_new/lib/test/TestNumToStr.m."""

import math

import numpy as np
import pytest

from num_to_str import num_to_str

pytestmark = pytest.mark.no_hardware


class TestNumToStr:
    def test_special(self):
        assert num_to_str(np.uint32(10)) == '10'
        assert num_to_str(np.int32(-10)) == '-10'
        assert num_to_str(True) == '1'
        assert num_to_str(math.inf) == 'inf'
        assert num_to_str(-math.inf) == '-inf'
        assert num_to_str(math.nan) == 'nan'

    def test_integer(self):
        assert num_to_str(11) == '11'
        assert num_to_str(-11) == '-11'
        assert num_to_str(0) == '0'
        assert num_to_str(-0) == '0'

    def test_unity(self):
        assert num_to_str(1.1) == '1.1'
        assert num_to_str(-1.1) == '-1.1'
        assert num_to_str(0.3) == '0.3'
        assert num_to_str(-0.3) == '-0.3'
        assert num_to_str(0.1 + 0.2) == '0.30000000000000004'
        assert num_to_str(-0.1 - 0.2) == '-0.30000000000000004'

    def test_small(self):
        assert num_to_str(1.1e-7) == '1.1e-7'
        assert num_to_str(-1.1e-7) == '-1.1e-7'
        assert num_to_str(3e-8) == '3e-8'
        assert num_to_str(-3e-8) == '-3e-8'
        assert num_to_str(1e-8 + 2e-8) == '3.0000000000000004e-8'
        assert num_to_str(-1e-8 - 2e-8) == '-3.0000000000000004e-8'

    def test_large(self):
        assert num_to_str(1.1e7) == '1.1e7'
        assert num_to_str(-1.1e7) == '-1.1e7'
        assert num_to_str(1.234e7 + 2.345e7) == '3.579e7'
        assert num_to_str(-1.234e7 - 2.345e7) == '-3.579e7'
