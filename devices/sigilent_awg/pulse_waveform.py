"""pulse_waveform.py -- general pulse-shape generator for the Siglent SDG6X AWGs.

Generalizes :mod:`gaussian_pulse_waveform` (the port of
``matlab_new/YbExptCtrl/sigilentAWG/gaussianPulseWaveform.m``) to a family of envelope shapes,
selected by the ``shape`` param (design agreed 2026-07-03; see ``pyctrl/tmp/pulse_10_examples.png``):

  * ``gaussian``       -- exp(-((x-0.5)*steepness)^2), peak mid-window (the original; default).
  * ``rise_gaussian``  -- exp(-((x-1)*steepness)^2), Gaussian flank rising to its peak at the END
                          of the main window (convention (a): same formula, center moved -> the
                          envelope starts at exp(-steepness^2), a true-zero start for steepness>=3).
  * ``fall_gaussian``  -- exp(-(x*steepness)^2), peak at the START (at the trigger when sharp).
  * ``rise_linear``    -- x       (ramp 0 -> 1; ``steepness`` ignored).
  * ``fall_linear``    -- 1 - x   (ramp 1 -> 0; ``steepness`` ignored).

where ``x = t/pulse_width_us`` is normalized MAIN-window time.

**Smooth window** (``smooth_width_us``, default 0 = sharp): rise/fall shapes end (rise) or start
(fall) at full amplitude -- a hard step on the AWG output. A nonzero ``smooth_width_us`` appends
(rise_*) or prepends (fall_*) an EXTRA half-cosine ramp window so the envelope is continuous:

  * rise_*: main window [0, pw] then 0.5*(1+cos(pi*xs)) over [pw, pw+sm]   (1 -> 0)
  * fall_*: 0.5*(1-cos(pi*xs)) over [0, sm] then main window [sm, sm+pw]   (0 -> 1)

``pulse_width_us`` always means the MAIN window; total playback = pulse_width_us + smooth_width_us.
NOTE (hardware): for fall_* the AWG can only play from its trigger, so the smooth pre-rise shifts
the envelope peak ``smooth_width_us`` AFTER the trigger edge. And the gated burst
(``BTWV GATE_NCYC,GATE``) only plays while the TTL gate is high -> the gate must cover the TOTAL
width or the tail is cut. ``smooth_width_us`` is ignored for ``gaussian`` (its edges are already
~0). The carrier stays exactly ``carrier_freq_MHz`` over the TOTAL width and the DDS playback
``FREQ = 1e6 / total`` (all ``num_points`` play in the total width).

Pure / hardware-free: NumPy only. Byte layout unchanged: **big-endian int16** samples (little-endian
gives a flat envelope + random phase on the SDG6X -- the single most common AWG upload bug).
"""
import numpy as np

_AWG_MAX_CODE = 32767  # double(intmax('int16'))

#: shape name -> needs ``steepness``?
SHAPES = {
    "gaussian": True,
    "rise_gaussian": True,
    "fall_gaussian": True,
    "rise_linear": False,
    "fall_linear": False,
}


def pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness):
    """Envelope for ``shape`` on ``num_points`` samples. Returns ``(t_us, envelope)``.

    ``t_us`` runs [0, total] where total = pulse_width_us (+ smooth_width_us for rise_*/fall_*);
    t=0 is the AWG trigger. Pure helper -- also used by plotting/diagnostic scripts.
    """
    if shape not in SHAPES:
        raise ValueError("unknown pulse shape %r (valid: %s)" % (shape, ", ".join(sorted(SHAPES))))
    if smooth_width_us < 0:
        raise ValueError("smooth_width_us must be >= 0 (got %g)" % smooth_width_us)
    if shape == "gaussian":
        smooth_width_us = 0.0                       # symmetric pulse: edges already ~0
    total = pulse_width_us + smooth_width_us

    t_norm = np.linspace(0.0, 1.0, int(num_points))
    t_us = t_norm * total
    e = np.zeros(t_us.shape)

    if shape == "gaussian":
        # x = t_norm directly (not t_us/pw): byte-exact vs the original gaussianPulseWaveform
        # (avoids the 1-ulp (t*pw)/pw round-trip for non-power-of-2 pulse widths).
        e = np.exp(-((t_norm - 0.5) * steepness) ** 2)
    elif shape in ("rise_gaussian", "rise_linear"):
        # main window first, smooth 1->0 cosine tail appended after the peak
        m = t_us <= pulse_width_us
        x = t_us[m] / pulse_width_us
        e[m] = np.exp(-((x - 1.0) * steepness) ** 2) if shape == "rise_gaussian" else x
        if smooth_width_us > 0:
            xs = (t_us[~m] - pulse_width_us) / smooth_width_us
            e[~m] = 0.5 * (1.0 + np.cos(np.pi * xs))
    else:  # fall_gaussian / fall_linear
        # smooth 0->1 cosine pre-rise prepended, then the main window
        m = t_us < smooth_width_us
        if smooth_width_us > 0:
            xs = t_us[m] / smooth_width_us
            e[m] = 0.5 * (1.0 - np.cos(np.pi * xs))
        x = (t_us[~m] - smooth_width_us) / pulse_width_us
        e[~m] = np.exp(-(x * steepness) ** 2) if shape == "fall_gaussian" else 1.0 - x
    return t_us, e


def pulse_waveform(params):
    """Return ``(binary_data, info)`` for a shaped pulse (superset of gaussian_pulse_waveform).

    Args:
        params: a mapping with the waveform-shaping fields --
            ``shape`` (str, one of :data:`SHAPES`; default ``"gaussian"``),
            ``num_points`` (int, fixed sample count, e.g. 10000),
            ``pulse_width_us`` (float, MAIN-window duration in microseconds),
            ``smooth_width_us`` (float >= 0, EXTRA cosine window; default 0 = sharp),
            ``carrier_freq_MHz`` (float, carrier frequency in MHz),
            ``steepness`` (float, Gaussian-envelope factor; ignored by *_linear),
            ``amplitude_scale`` (float 0-1, default 1.0),
            ``max_amplitude_vpp`` (float, optional -- only for the voltage trace in ``info``).

    Returns:
        binary_data (bytes): big-endian int16 samples, ready to append to a WVDT command.
        info (dict): ``num_points``, ``freq_hz`` (= 1e6 / total width), ``t_us`` (ndarray, us
            since trigger), ``waveform`` (ndarray), ``shape``, ``total_width_us``, and
            ``voltage`` (ndarray) when ``max_amplitude_vpp`` is given.
    """
    shape = str(params.get("shape", "gaussian"))
    num_points = int(params["num_points"])
    pulse_width_us = float(params["pulse_width_us"])
    smooth_width_us = float(params.get("smooth_width_us", 0.0))
    carrier_freq_MHz = float(params["carrier_freq_MHz"])
    amplitude_scale = float(params.get("amplitude_scale", 1.0))
    if shape not in SHAPES:
        raise ValueError("unknown pulse shape %r (valid: %s)" % (shape, ", ".join(sorted(SHAPES))))
    steepness = float(params["steepness"]) if SHAPES[shape] else 0.0

    t_us, envelope = pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness)
    total = t_us[-1]

    # Carrier: exactly carrier_freq_MHz over the TOTAL width. Phrased as oscillations x normalized
    # time so shape="gaussian" reproduces gaussianPulseWaveform byte-for-byte (num_osc = f * pw).
    t_norm = np.linspace(0.0, 1.0, num_points)
    num_oscillations = carrier_freq_MHz * total
    carrier = np.sin(2.0 * np.pi * num_oscillations * t_norm)

    waveform = carrier * envelope
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    waveform = waveform * amplitude_scale

    # int16 with MATLAB int16() semantics (round-to-nearest + saturate), then BIG-endian bytes.
    scaled = np.clip(np.round(waveform * _AWG_MAX_CODE), -32768, 32767).astype(np.int16)
    binary_data = scaled.astype(">i2").tobytes()   # '>i2' == big-endian int16

    freq_hz = 1e6 / total                          # DDS playback: all points in the TOTAL width

    info = {
        "num_points": num_points,
        "freq_hz": freq_hz,
        "t_us": t_us,
        "waveform": waveform,
        "shape": shape,
        "total_width_us": total,
    }
    if "max_amplitude_vpp" in params and params["max_amplitude_vpp"] is not None:
        info["voltage"] = waveform * float(params["max_amplitude_vpp"]) / 2.0
    return binary_data, info
