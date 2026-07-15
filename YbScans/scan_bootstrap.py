"""scan_bootstrap.py -- shared sys.path setup for YbScans scan files (import side effect).

Importing this module puts the pyctrl flat-path roots (pyctrl/ itself + lib, YbExptCtrl,
YbSeqs, YbSteps) on ``sys.path``, replacing the per-file ``_bootstrap()`` copy every scan
used to carry. A scan file starts with::

    import scan_bootstrap  # noqa: F401  (side effect: pyctrl dirs on sys.path)

    from MySeq import MySeq            # now importable; editor go-to-definition works

This resolves when the scan is run as a script (``python YbScans/MyScan.py`` puts YbScans
first on ``sys.path``) and when the scan module is imported by anything that could already
find the scan module itself (same directory).

Scans in subdirectories (e.g. ``RearrangeDiagnostics/``) run as scripts get the SUBdir, not
YbScans, as ``sys.path[0]`` -- they cannot ``import scan_bootstrap`` directly and keep their
local ``_bootstrap()`` (or add YbScans to the path first).
"""

import os
import sys


def bootstrap():
    """Idempotently add the pyctrl flat-path roots to ``sys.path`` (runs once at import)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    if root not in sys.path:
        sys.path.insert(0, root)


bootstrap()
