"""PicoMotor369.py -- 369 picomotor-mirror kick seq (sibling of ``PicoMotor308.py``).

nargin-1 seq: takes a configured ``ExpSeq`` and drives one *relative* picomotor-mirror kick on a
piezo channel (NI DAQ): hold ``Volts`` for ``HoldTime`` s, then ramp to 0. The kick's sign sets
the move direction. Same structure as ``PicoMotor308`` (which see), on the 369 mirror channels
(``expConfig.py`` aliases added 2026-07-06):
  * "v" -> ``VPicoMotor369v`` (Dev1/23), vertical
  * "h" -> ``VPicoMotor369h`` (Dev1/22), horizontal

Axis/kick/hold come from the scan config (``s.C.PicoMotor369.Axis`` / ``.Volts`` / ``.HoldTime``)
with safe defaults (v axis, 0 V = no physical move). Every build records the move in the
picomotor history (``log/pyctrl_log/picomotor_history.jsonl``) so the mirror's relative travel
per axis is traceable.
"""

from picomotor_history import log_move

CHANNELS = {"v": "VPicoMotor369v", "h": "VPicoMotor369h"}
DEFAULT_AXIS = "v"
DEFAULT_VOLTS = 0.0    # SAFETY default: 0 V = no physical move (the scan sets the real kick).
DEFAULT_HOLD_S = 0.5


def PicoMotor369(s):
    axis = str(s.C.PicoMotor369.Axis(DEFAULT_AXIS)).lower()
    if axis not in CHANNELS:
        raise ValueError("PicoMotor369 Axis must be 'v' or 'h', got %r" % (axis,))
    channel = CHANNELS[axis]
    volts = s.C.PicoMotor369.Volts(DEFAULT_VOLTS)
    hold_s = s.C.PicoMotor369.HoldTime(DEFAULT_HOLD_S)

    s.add(channel, volts)
    s.wait(hold_s)
    s.add_step(1e-6).add(channel, 0)

    log_move(channel, volts, hold_s, note="PicoMotor369 axis=%s" % axis)
    return s
