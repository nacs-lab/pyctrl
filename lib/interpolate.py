"""interpolate.py -- piecewise-linear lookup over a value table.

Faithful transliteration of ``matlab_new/lib/interpolate.m``. With any SeqVal
argument it builds an ``OP_INTERP`` node (carrying ``x``, ``x0``, ``dx=x1-x0``
and the value table, which ``SeqContext`` interns into the data array); with
plain numbers it evaluates the interpolation directly, matching MATLAB.
"""

import math

import numpy as np

from seq_val import SeqVal


def interpolate(x, x0, x1, vals):
    dx = x1 - x0
    if isinstance(x, SeqVal):
        return SeqVal(SeqVal.OP_INTERP, [x, x0, dx, vals], x.ctx)
    if isinstance(x0, SeqVal):
        return SeqVal(SeqVal.OP_INTERP, [x, x0, dx, vals], x0.ctx)
    if isinstance(dx, SeqVal):
        return SeqVal(SeqVal.OP_INTERP, [x, x0, dx, vals], dx.ctx)

    vals = [float(v) for v in np.atleast_1d(vals)]
    nv = len(vals)
    scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    xs = [float(x)] if scalar else [float(v) for v in np.atleast_1d(x)]
    out = []
    for xv in xs:
        xe = (xv - x0) * (nv - 1) / dx
        if xe <= 0:
            out.append(vals[0])
        elif xe >= (nv - 1):
            out.append(vals[-1])
        else:
            lo = int(math.floor(xe))
            xrem = xe - lo
            vlo = vals[lo]
            if xrem == 0:
                out.append(vlo)
            else:
                out.append(vlo * (1 - xrem) + vals[lo + 1] * xrem)
    return out[0] if scalar else out
