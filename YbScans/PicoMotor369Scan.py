"""PicoMotor369Scan.py -- submit ``PicoMotor369`` to the RUNNING pyctrl backend, once.

Sibling of ``PicoMotor308Scan.py`` for the 369 steering mirror. Runs the picomotor-mirror move
sequence (``YbSeqs/PicoMotor369.py``) a SINGLE time through the live backend (so it shares the
engine the backend already owns -- no second-process engine collision). The axis (``v``/``h``),
kick voltage + hold time are set in ``PicoMotor369Scan`` below and passed via the scan config
(``PicoMotor369.Axis`` / ``.Volts`` / ``.HoldTime``); the seq reads them off ``s.C``. The
ScanGroup has exactly ONE point and ``rep=1`` -> ``run_scan_group`` walks ``for i in 1..1`` once
= ONE shot. Every build also appends to ``log/pyctrl_log/picomotor_history.jsonl``.

Two modes (axes): ``v`` drives ``VPicoMotor369v`` (Dev1/23, vertical), ``h`` drives
``VPicoMotor369h`` (Dev1/22, horizontal). Edit ``axis`` / ``volts`` / ``hold_s`` in
``PicoMotor369Scan`` to change the move.

SAFETY: ``volts`` defaults to 0 -> drives the NI channel at 0 V (no physical move). Set a
nonzero ``volts`` for a real move; sign sets direction.

This only BUILDS the (1-point) ScanGroup + sends the descriptor JSON; it does NOT load the
engine, so any interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend must already be live):
    cd pyctrl
    python YbScans/PicoMotor369Scan.py
"""

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from PicoMotor369 import PicoMotor369


def build(axis, volts, hold_s):
    """A 1-point ScanGroup carrying the picomotor axis + kick (volts) + hold time."""
    from scan_group import ScanGroup

    g = ScanGroup()
    g().PicoMotor369.Axis = axis                # seq reads s.C.PicoMotor369.Axis ("v"/"h")
    g().PicoMotor369.Volts = float(volts)       # seq reads s.C.PicoMotor369.Volts
    g().PicoMotor369.HoldTime = float(hold_s)   # seq reads s.C.PicoMotor369.HoldTime
    rp = g.runp()
    rp.NumPerGroup = 1     # one shot per group
    rp.NumImages = 0       # PicoMotor369 has no imaging step
    rp.Scramble = 0
    rp.useScanLongSlmLock = 0   # picomotor test doesn't touch the SLM: skip the lock + the
                                # expConfig default loading-phase write
    return g


def PicoMotor369Scan(url=None):
    """Build + submit ONE PicoMotor369 shot (rep=1). Returns the queued descriptor id.

    Set the move here:
    """
    # ---- the move (edit these) ----
    axis = "h"        # "v" = VPicoMotor369v (vertical), "h" = VPicoMotor369h (horizontal)
    volts = 2         # SAFETY: 0 = no physical move. Set nonzero (e.g. 2) for a real kick; sign = direction.
    hold_s = 1        # seconds to hold the kick before ramping back to 0
    # -------------------------------

    from yb_start_scan import ybStartScan

    g = build(axis=axis, volts=volts, hold_s=hold_s)
    description = "one-shot picomotor move: axis=%s, %g V, %g s hold" % (axis, volts, hold_s)
    did = ybStartScan(PicoMotor369, g, url=url, label="PicoMotor369_%s" % axis,
                      description=description,
                      rep=1)   # rep=1 + 1 scan point -> exactly one shot (rep=0 would be forever)
    print("submitted PicoMotor369 -> descriptor id %s (url=%s, 1 shot; %s)"
          % (did, url or "default", description))
    return did


if __name__ == "__main__":
    PicoMotor369Scan()
