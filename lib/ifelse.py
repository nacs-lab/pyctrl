"""ifelse.py -- branch-free conditional select.

Combines the two MATLAB originals: ``SeqVal.ifelse`` (when ``cond`` is a SeqVal,
build an ``OP_SELECT`` node) and the free function ``ifelse.m`` (when ``cond`` is
a plain scalar or vector, pick element-wise). Splitting them out is unnecessary
in Python since a single function can dispatch on type.
"""

import numpy as np

from seq_val import SeqVal, seqval_isequal


def ifelse(cond, v1, v2):
    if isinstance(cond, SeqVal):
        if seqval_isequal(v1, v2):
            return v1
        return SeqVal(SeqVal.OP_SELECT, [cond, v1, v2], cond.ctx)

    # Plain cond. Scalar (incl. SeqVal v1/v2) returns the chosen branch as-is;
    # only a genuine vector cond takes the element-wise numeric path.
    conds = np.atleast_1d(cond)
    if conds.size <= 1:
        return v1 if cond else v2

    res = np.zeros(conds.size)
    a1 = np.atleast_1d(v1)
    a2 = np.atleast_1d(v2)
    for i in range(conds.size):
        if conds[i]:
            res[i] = a1[i] if a1.size > 1 else v1
        else:
            res[i] = a2[i] if a2.size > 1 else v2
    return res
