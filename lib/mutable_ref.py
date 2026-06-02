"""mutable_ref.py -- MutableRef: a mutable single-value box (handle semantics).

Faithful transliteration of ``matlab_new/lib/MutableRef.m``. MATLAB only allows
*constant* static class members, so a value that must be shared at class level yet stay
mutable is wrapped in this handle box -- see ``EnableScan.enabled`` (enable_scan.py),
the one consumer in this migration.
"""


class MutableRef:
    def __init__(self, x=None):
        # MATLAB guards with exist('x','var'); a no-arg MutableRef leaves the property
        # as [] (here: None). Every real use constructs MutableRef(value).
        self.x = x

    def set(self, x):
        self.x = x
        return self          # MATLAB returns self (handle -- the mutation persists anyway)

    def get(self):
        return self.x
