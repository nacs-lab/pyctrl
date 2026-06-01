"""ir_pulse.py -- transliteration of ``matlab_new/lib/IRPulse.m``.

A deprecated, empty compatibility stub ("For compatibility only."). It defines no
fields and no ``calc_value``; ``TimeStep.add`` only dispatches into it for a concrete
subclass that overrides ``calc_value``. No such subclass exists, so this path is
dead in Phase 2 -- kept for ``isinstance`` fidelity.
"""


class IRPulse:
    def __init__(self, id):
        pass
