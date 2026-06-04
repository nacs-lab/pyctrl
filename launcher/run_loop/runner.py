"""launcher.run_loop.runner -- the ``python -m launcher.run_loop.runner <url>`` entry point.

This is the module ``PyctrlLauncher`` spawns (``yb_analysis/config.py`` ``PYCTRL_MODULE``).
It is a THIN shim: ``-m`` puts only the pyctrl package root on ``sys.path``, but the run loop
and every class it builds are imported FLAT (``from sequence_runner import ...``,
``from exp_seq import ExpSeq``) off the pyproject ``pythonpath`` dirs. So before importing the
real host we prepend those dirs (mirroring ``pyproject.toml`` ``pythonpath`` and
``tests/conftest.py``), then delegate to ``YbExptCtrl/runner.py``'s :func:`main`.

The real run-loop host -- ExptServer hosting, the consume loop, engine wiring, camera
release-on-terminate, the single-backend guard -- all lives in ``YbExptCtrl/runner.py``
(the faithful port of ``SequenceRunner.m``). Keeping the path bootstrap separate keeps that
module flat-importable + NO-HARDWARE-testable exactly like the rest of the codebase.
"""

import os
import sys


def _bootstrap_path():
    """Prepend the flat pyctrl source dirs to ``sys.path`` (mirror pyproject pythonpath)."""
    # __file__ = <pyctrl>/launcher/run_loop/runner.py  ->  pyctrl root is three dirs up.
    pyctrl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # pyctrl_root itself carries expConfig.py (the executable config, mirroring matlab_new/expConfig.m).
    for p in [pyctrl_root] + [os.path.join(pyctrl_root, name)
                              for name in ("lib", "YbSteps", "YbSeqs", "YbExptCtrl", "tools")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv=None):
    _bootstrap_path()
    from runner import main as _main  # YbExptCtrl/runner.py (now flat-importable)
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
