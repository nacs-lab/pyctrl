"""sigilent_awg -- Siglent SDG6X arbitrary-waveform generators (556 nm SDG6022X, 308 nm SDG6052X).

USB-VISA (pyvisa) driver + scan coordinator for the gated Gaussian Rydberg pulses. DDS mode
(fixed num_points, FREQ in the WVDT). Three pieces, ported from
``matlab_new/YbExptCtrl/sigilentAWG/``:

  * :func:`pulse_waveform`          -- pure waveform generator (big-endian int16 bytes); ``shape``
    param selects gaussian / rise_gaussian / fall_gaussian / rise_linear / fall_linear
    (+ ``smooth_width_us`` cosine edge). :func:`gaussian_pulse_waveform` is the byte-exact
    Gaussian-only legacy entry.
  * :class:`AWGConnection`          -- one USB channel; build/send WVDT, amplitude, gated burst.
  * :class:`AWGManager`             -- batch-upload every unique waveform at scan start, switch the
    active waveform per shot (~2 ms), clean up at scan end. Process-global state.

Import-safe with no VISA backend present (pyvisa is imported lazily inside ``AWGConnection.connect``).
"""
from .awg_connection import AWGConnection
from .awg_manager import AWGManager, WAVEFORM_FIELDS
from .gaussian_pulse_waveform import gaussian_pulse_waveform
from .pulse_waveform import (pulse_waveform, pulse_envelope, pulse_total_us,
                             pulse_carrier_freq_MHz, SHAPES, CHIRP_PROFILES)

__all__ = ["AWGConnection", "AWGManager", "WAVEFORM_FIELDS", "gaussian_pulse_waveform",
           "pulse_waveform", "pulse_envelope", "pulse_total_us", "pulse_carrier_freq_MHz",
           "SHAPES", "CHIRP_PROFILES"]

# Registry: create("sigilent_awg", resource, channel) -> an AWGConnection handle
# (NEEDS-HARDWARE on connect()). The AWGManager is the scan-level orchestrator.
from ..device_registry import register

register("sigilent_awg")(AWGConnection)
