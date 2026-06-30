"""PicoMotor308.py -- transliteration of ``matlab_new/YbSeqs/PicoMotor308.m``.

nargin-1 seq: takes a configured ``ExpSeq`` and drives one *relative* picomotor-mirror kick on a
piezo channel (NI DAQ): hold ``Volts`` for ``HoldTime`` s, then ramp to 0. The kick's sign sets
the move direction (per the MATLAB note: "+2V move twice faster than -2V for horizontal, +2V move
x1.5 faster than -2V for vertical").

Two axes / modes, selected by ``s.C.PicoMotor308.Axis`` ("v" or "h"):
  * "v" -> ``VPicoMotor308v`` (Dev1/11), vertical
  * "h" -> ``VPicoMotor308h`` (Dev1/10), horizontal

The MATLAB source hardcodes the v axis at +2 V / 0.5 s and is nargin-0; here axis/kick/hold are
read from the scan config (``s.C.PicoMotor308.Axis`` / ``.Volts`` / ``.HoldTime``) so the scan
supplies them, with safe defaults (v axis, 0 V = no physical move). The MATLAB inner ``rampTo``
(defined, never called) is dropped (pyctrl has ``ramp_to`` if a ramp pulse is ever wanted). Every
build records the move in the picomotor history (``log/pyctrl_log/picomotor_history.jsonl``) so
the mirror's relative travel per axis is traceable.
"""

from picomotor_history import log_move

CHANNELS = {"v": "VPicoMotor308v", "h": "VPicoMotor308h"}
DEFAULT_AXIS = "v"
DEFAULT_VOLTS = 0.0    # SAFETY default: 0 V = no physical move (the scan sets the real kick).
DEFAULT_HOLD_S = 0.5


def PicoMotor308(s):
    axis = str(s.C.PicoMotor308.Axis(DEFAULT_AXIS)).lower()
    if axis not in CHANNELS:
        raise ValueError("PicoMotor308 Axis must be 'v' or 'h', got %r" % (axis,))
    channel = CHANNELS[axis]
    volts = s.C.PicoMotor308.Volts(DEFAULT_VOLTS)
    hold_s = s.C.PicoMotor308.HoldTime(DEFAULT_HOLD_S)

    # +V moves ~2x faster than -V (horizontal); ~1.5x faster than -V (vertical).
    s.add(channel, volts)
    s.wait(hold_s)
    s.add_step(1e-6).add(channel, 0)

    log_move(channel, volts, hold_s, note="PicoMotor308 axis=%s" % axis)
    return s
