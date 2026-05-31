"""num_to_str.py -- shortest round-tripping decimal string for a number.

Faithful transliteration of matlab_new/lib/num_to_str.py's MATLAB original
(``num_to_str.m``). Used by ``SeqVal.to_string`` to render numeric leaves the
same way MATLAB does (e.g. ``0.1 + 0.2`` -> ``'0.30000000000000004'``). This is
display-only and never affects the serialized bytes, but the ports of
``TestNumToStr`` / ``TestSeqContext`` pin the exact strings.
"""

import math

import numpy as np


def num_to_str(num):
    # Integer- and logical-typed values format with %d (MATLAB isinteger/islogical).
    # A bare Python int maps to a MATLAB double (see seq_val._arg_kind), so it is
    # handled by the float path below, NOT here -- only numpy integers / bools do.
    if isinstance(num, (bool, np.bool_)):
        return '%d' % int(num)
    if isinstance(num, np.integer):
        return '%d' % int(num)

    num = float(num)
    if not math.isfinite(num):
        if math.isnan(num):
            return 'nan'
        return 'inf' if num > 0 else '-inf'
    if num == 0:
        # Ignore signed zero for now... (matches MATLAB)
        return '0'

    anum = abs(num)
    if anum >= 1e6 or anum < 1e-4:
        s = _shortest(num, 'e')
        # This is stupid but whatever........ (matches MATLAB strrep cleanup)
        if anum > 1:
            s = s.replace('e+', 'e').replace('e0', 'e')
        else:
            s = s.replace('e-0', 'e-')
        return s
    return _shortest(num, 'f')


def _shortest(num, fmt):
    # Smallest number of digits whose formatted form parses back to exactly num.
    for ndig in range(21):
        s = ('%.' + str(ndig) + fmt) % num
        if float(s) == num:
            return s
    return s
