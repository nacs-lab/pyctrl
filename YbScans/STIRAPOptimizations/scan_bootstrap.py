"""scan_bootstrap.py -- path-setup shim for STIRAP scans moved into STIRAPOptimizations/.

These scans are Claude-run STIRAP-optimization campaign forks moved out of ``YbScans/`` to
declutter it (the production bases -- ``RearrangeSTIRAPScan.py`` and ``STIRAPAWGScan.py`` --
stayed in ``YbScans/``). A scan run as ``python YbScans/STIRAPOptimizations/Foo.py`` gets THIS
directory as ``sys.path[0]``, so its unchanged ``import scan_bootstrap`` resolves to this file
instead of ``YbScans/scan_bootstrap.py``.

This shim mirrors ``YbScans/scan_bootstrap.bootstrap()`` but one directory deeper: it adds the
pyctrl flat-path roots (pyctrl/ + lib, YbExptCtrl, YbSeqs, YbSteps) AND the parent ``YbScans/``
to ``sys.path``, so the moved scans need no edits.
"""

import os
import sys


def bootstrap():
    """Idempotently add the pyctrl flat-path roots + YbScans/ to ``sys.path``."""
    here = os.path.dirname(os.path.abspath(__file__))   # .../YbScans/STIRAPOptimizations
    ybscans = os.path.dirname(here)                     # .../YbScans
    root = os.path.dirname(ybscans)                     # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    for p in (root, ybscans):
        if p not in sys.path:
            sys.path.insert(0, p)


bootstrap()
