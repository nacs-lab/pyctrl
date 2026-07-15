"""PicoMotor308Scan.py -- submit ``PicoMotor308`` to the RUNNING pyctrl backend, once.

Runs the picomotor-mirror move sequence (``YbSeqs/PicoMotor308.py``) a SINGLE time through the
live backend (so it shares the engine the backend already owns -- no second-process engine
collision). The axis (``v``/``h``), kick voltage + hold time are set in ``PicoMotor308Scan``
below and passed via the scan config (``PicoMotor308.Axis`` / ``.Volts`` / ``.HoldTime``); the
seq reads them off ``s.C``. The ScanGroup has exactly ONE point and ``rep=1`` ->
``run_scan_group`` walks ``for i in 1..1`` once = ONE shot. Every build also appends to
``log/pyctrl_log/picomotor_history.jsonl``.

Two modes (axes): ``v`` drives ``VPicoMotor308v`` (vertical), ``h`` drives ``VPicoMotor308h``
(horizontal). Edit ``axis`` / ``volts`` / ``hold_s`` in ``PicoMotor308Scan`` to change the move.

SAFETY: ``volts`` defaults to 0 -> drives the NI channel at 0 V (no physical move). Set a
nonzero ``volts`` for a real move; sign sets direction.

This only BUILDS the (1-point) ScanGroup + sends the descriptor JSON; it does NOT load the
engine, so any interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend must already be live):
    cd pyctrl
    python YbScans/PicoMotor308Scan.py
"""

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from PicoMotor308 import PicoMotor308


def build(axis, volts, hold_s):
    """A 1-point ScanGroup carrying the picomotor axis + kick (volts) + hold time."""
    from scan_group import ScanGroup

    g = ScanGroup()
    g().PicoMotor308.Axis = axis                # seq reads s.C.PicoMotor308.Axis ("v"/"h")
    g().PicoMotor308.Volts = float(volts)       # seq reads s.C.PicoMotor308.Volts
    g().PicoMotor308.HoldTime = float(hold_s)   # seq reads s.C.PicoMotor308.HoldTime
    rp = g.runp()
    rp.NumPerGroup = 1     # one shot per group
    rp.NumImages = 0       # PicoMotor308 has no imaging step
    rp.Scramble = 0
    rp.useScanLongSlmLock = 0   # picomotor test doesn't touch the SLM: skip the lock + the
                                # expConfig default loading-phase write (which 404'd: 47x47_feedbackwarm3.pt)
    return g


def PicoMotor308Scan(url=None):
    """Build + submit ONE PicoMotor308 shot (rep=1). Returns the queued descriptor id.

    Set the move here:
    """
    # ---- the move (edit these) ----
    axis = "h"        # "v" = VPicoMotor308v (vertical), "h" = VPicoMotor308h (horizontal)
    volts = -2         # SAFETY: 0 = no physical move. Set nonzero (e.g. 2) for a real kick; sign = direction.
    hold_s = 1      # seconds to hold the kick before ramping back to 0
    # -------------------------------

    from yb_start_scan import ybStartScan

    g = build(axis=axis, volts=volts, hold_s=hold_s)
    description = "one-shot picomotor move: axis=%s, %g V, %g s hold" % (axis, volts, hold_s)
    did = ybStartScan(PicoMotor308, g, url=url, label="PicoMotor308_%s" % axis,
                      description=description,
                      rep=1)   # rep=1 + 1 scan point -> exactly one shot (rep=0 would be forever)
    print("submitted PicoMotor308 -> descriptor id %s (url=%s, 1 shot; %s)"
          % (did, url or "default", description))
    return did


if __name__ == "__main__":
    PicoMotor308Scan()
