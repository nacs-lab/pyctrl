"""launcher -- launch-shim package for the pyctrl run loop.

The pyctrl modules are imported FLAT (``from exp_seq import ExpSeq``) off the pyproject
``pythonpath`` (lib / YbSteps / YbSeqs / YbExptCtrl / tools). This ``launcher`` package exists
only to provide the ``python -m launcher.run_loop.runner <url>`` entry point the monitor's
``PyctrlLauncher`` spawns (``yb_analysis/config.py`` ``PYCTRL_MODULE``): when launched via
``-m`` with only the pyctrl root on ``sys.path``, those flat dirs are NOT importable, so
``run_loop/runner.py`` bootstraps them before delegating to ``YbExptCtrl/runner.py``.
"""
