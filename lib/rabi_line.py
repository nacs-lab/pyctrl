"""rabi_line.py -- Rabi lineshape (port of rabiLine.m).

det:   detuning in angular-frequency units
t:     time in 1/frequency units
Omega: Rabi frequency in angular-frequency units
"""

import math


def rabi_line(det, t, Omega):
    Omega2 = Omega ** 2
    OmegaG2 = det ** 2 + Omega2
    if OmegaG2 == 0 and Omega2 == 0:
        return 0
    return Omega2 / OmegaG2 * math.sin(math.sqrt(OmegaG2) * t / 2) ** 2
