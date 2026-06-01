"""mat_utils.py -- MATLAB-semantics helpers used by the Phase-2 timing port.

No MATLAB class counterpart: these reproduce three MATLAB built-in behaviors that
differ from their Python lookalikes and are byte-/timing-load-bearing.

* ``mat_round`` -- MATLAB ``round`` is **half-away-from-zero**; Python ``round`` and
  numpy are **banker's (half-to-even)**. Every ``round(t * time_scale)`` call site in
  ``ExpSeqBase``/``TimeSeq`` must use this. On a ``SeqVal`` it builds an ``OP_RINT``
  node (the runtime rounds), matching MATLAB ``round(SeqVal)``.
* ``is_numeric`` -- MATLAB ``isnumeric`` is ``false`` for logicals; a Python ``bool``
  is an ``int`` subclass, so it must be excluded explicitly.
* ``is_logical`` -- MATLAB ``islogical``.
"""

import math

import numpy as np

from seq_val import SeqVal


def mat_round(x):
    """Round half-away-from-zero (MATLAB ``round``). SeqVal -> OP_RINT node."""
    if isinstance(x, SeqVal):
        return x.__round__()
    # math.floor/ceil keep this exact for the .5 boundary in both signs.
    if x >= 0:
        return float(math.floor(x + 0.5))
    return float(math.ceil(x - 0.5))


def is_numeric(x):
    """MATLAB ``isnumeric``: real/complex numeric, but NOT logical and NOT a SeqVal."""
    if isinstance(x, bool) or isinstance(x, np.bool_):
        return False
    return isinstance(x, (int, float, np.integer, np.floating))


def is_logical(x):
    """MATLAB ``islogical``."""
    return isinstance(x, (bool, np.bool_))
