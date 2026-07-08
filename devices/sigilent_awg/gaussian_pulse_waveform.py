"""gaussian_pulse_waveform.py -- port of
``matlab_new/YbExptCtrl/sigilentAWG/gaussianPulseWaveform.m``.

Generate a Gaussian-enveloped sinusoidal waveform for upload to a Siglent SDG6X AWG in
**DDS mode** (fixed ``num_points``). The AWG plays all ``num_points`` in ``pulse_width_us``
microseconds, so the effective sample rate = ``num_points / pulse_width_us`` and the WVDT
``FREQ = 1e6 / pulse_width_us`` (set on the wire by :meth:`AWGConnection.build_waveform_cmd`).

Pure / hardware-free: NumPy only, no vendor package -- safe to import and unit-test anywhere.

Byte layout (matches the MATLAB original, which the Siglent SDG6X requires): samples are
**big-endian int16** (MATLAB ``typecast(swapbytes(int16(...)), 'uint8')``). Little-endian gives a
flat envelope + random phase on the SDG6X -- the single most common AWG upload bug.

NOTE (2026-07-03): the shape math was generalized into :mod:`pulse_waveform` (``shape`` param:
gaussian / rise_gaussian / fall_gaussian / rise_linear / fall_linear + ``smooth_width_us``).
This module stays as the byte-exact Gaussian back-compat entry point: it FORCES
``shape="gaussian"`` regardless of any ``shape`` key in ``params``. New code should call
:func:`pulse_waveform.pulse_waveform` directly (the AWGManager does).
"""
from .pulse_waveform import pulse_waveform, _AWG_MAX_CODE  # noqa: F401  (re-export for old importers)


def gaussian_pulse_waveform(params):
    """Return ``(binary_data, info)`` for a Gaussian pulse (byte-exact legacy entry).

    Args:
        params: a mapping with the waveform-shaping fields --
            ``num_points`` (int, fixed sample count, e.g. 10000),
            ``pulse_width_us`` (float, pulse duration in microseconds),
            ``carrier_freq_MHz`` (float, carrier frequency in MHz),
            ``steepness`` (float, envelope steepness factor),
            ``amplitude_scale`` (float 0-1, default 1.0),
            ``max_amplitude_vpp`` (float, optional -- only for the voltage trace in ``info``).

    Returns:
        binary_data (bytes): big-endian int16 samples, ready to append to a WVDT command.
        info (dict): ``num_points``, ``freq_hz``, ``t_us`` (ndarray), ``waveform`` (ndarray),
            and ``voltage`` (ndarray) when ``max_amplitude_vpp`` is given.
    """
    p = dict(params)
    p["shape"] = "gaussian"        # legacy contract: always the symmetric Gaussian
    p.pop("smooth_width_us", None)  # ignored for gaussian anyway; drop for strict back-compat
    return pulse_waveform(p)
