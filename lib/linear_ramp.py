"""linear_ramp.py -- transliteration of ``matlab_new/lib/linearRamp.m``.

Factory returning a 3-arg pulse callable that linearly interpolates from
``vstart`` to ``vend`` over the step. Constant-folds to a bare scalar when both
endpoints are equal numbers. The SeqVal node graph follows from the already-ported
operators; ``TimeStep.add`` calls the closure with ``(t, len, old_val)``.
"""

from mat_utils import is_numeric


def linear_ramp(vstart, vend):
    if is_numeric(vstart) and is_numeric(vend) and vstart == vend:
        return vstart

    def func(t, length, old_val):
        return (vstart * (length - t) + vend * t) / length

    return func
