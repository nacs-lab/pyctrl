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
"""
import numpy as np

_AWG_MAX_CODE = 32767  # double(intmax('int16'))


def gaussian_pulse_waveform(params):
    """Return ``(binary_data, info)`` for a Gaussian pulse.

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
    num_points = int(params["num_points"])
    pulse_width_us = float(params["pulse_width_us"])
    carrier_freq_MHz = float(params["carrier_freq_MHz"])
    steepness = float(params["steepness"])
    amplitude_scale = float(params.get("amplitude_scale", 1.0))

    # Normalized time axis [0, 1] (matches the MATLAB GuassianPulseWaveforms convention).
    t = np.linspace(0.0, 1.0, num_points)

    # Number of carrier oscillations across the pulse window.
    num_oscillations = carrier_freq_MHz * pulse_width_us
    carrier = np.sin(2.0 * np.pi * num_oscillations * t)

    # Gaussian envelope: exp(-((t-0.5) * steepness)^2).
    envelope = np.exp(-((t - 0.5) * steepness) ** 2)

    waveform = carrier * envelope
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    waveform = waveform * amplitude_scale

    # int16 with MATLAB int16() semantics (round-to-nearest + saturate), then BIG-endian bytes.
    scaled = np.clip(np.round(waveform * _AWG_MAX_CODE), -32768, 32767).astype(np.int16)
    binary_data = scaled.astype(">i2").tobytes()   # '>i2' == big-endian int16

    freq_hz = 1e6 / pulse_width_us                 # DDS playback: all points in pulse_width_us

    info = {
        "num_points": num_points,
        "freq_hz": freq_hz,
        "t_us": t * pulse_width_us,
        "waveform": waveform,
    }
    if "max_amplitude_vpp" in params and params["max_amplitude_vpp"] is not None:
        info["voltage"] = waveform * float(params["max_amplitude_vpp"]) / 2.0
    return binary_data, info
