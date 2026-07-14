"""pulse_waveform.py -- general pulse-shape generator for the Siglent SDG6X AWGs.

Generalizes :mod:`gaussian_pulse_waveform` (the port of
``matlab_new/YbExptCtrl/sigilentAWG/gaussianPulseWaveform.m``) to a family of envelope shapes,
selected by the ``shape`` param (design agreed 2026-07-03; see ``pyctrl/tmp/pulse_10_examples.png``):

  * ``gaussian``       -- exp(-((x-0.5)*steepness)^2), peak mid-window (the original; default).
  * ``rise_gaussian``  -- ``exp(-((t-peak)/pw)^2)`` rising to its peak at the END; single half-
                          Gaussian lobe, SAME formula + position as the double's forward 556 lobe:
                          ``pulse_width_us`` is the 1/e half-width, total = 3*pw (peak at 3*pw =
                          pad+pw), ``steepness``/``smooth_width_us`` IGNORED. ``f_delay`` (us) shifts
                          the peak INTO the window (rise: 3*pw - f_delay; fall: 0 + f_delay), so the
                          HalfPulse pair tunes its overlap just like the two-lobe shapes. (Unified
                          2026-07-13 so the same AWG556/AWG308 params drive rise/fall and the double.)
  * ``fall_gaussian``  -- mirror of ``rise_gaussian``: peak at the START (t=0), falling 1 -> 0.
  * ``rise_linear``    -- x       (ramp 0 -> 1; ``steepness`` ignored; keeps the smooth window).
  * ``fall_linear``    -- 1 - x   (ramp 1 -> 0; ``steepness`` ignored; keeps the smooth window).

where ``x = t/pulse_width_us`` is normalized MAIN-window time (linear shapes only).

**Two-lobe STIRAP shapes** (``double_half_gaussian_inner`` / ``double_half_gaussian_outer``, design
agreed 2026-07-13; gallery ``pyctrl/tmp/claudeoutput.png``): a single AWG waveform holding TWO
half-Gaussian lobes -- lobe A = the forward-STIRAP pulse, lobe B = the reverse (iSTIRAP) pulse --
separated by a pure-zero ``stirap_gap`` of dead time. Each lobe is ``exp(-((t-peak)/pw)^2)`` so
``pulse_width_us`` (pw) is the lobe **1/e half-width** (``steepness``/``smooth_width_us`` are
IGNORED for these shapes). The two lobes are combined by ``MAX`` (an overlap caps at 1 -> a clean
merge into one continuous pulse, never >1). The playback window is zero-padded ``pad = 2*pw`` on
each side, so ``total = 6*pw + stirap_gap`` (a pure function of pw+gap -> the 556 and 308 waveforms
share the same total width and DDS FREQ automatically, so gate-triggered playback stays aligned).

  * ``double_half_gaussian_inner`` -- lobe peaks face the GAP (rise up to A's peak at the gap's
    left edge, fall away from B's peak at the gap's right edge). This is the ANCHOR shape: its two
    inner peaks sit exactly ``stirap_gap`` apart, so ``stirap_gap`` IS defined by the inner-pulse
    edges. ``f_delay`` / ``r_delay`` are IGNORED here (the anchor never moves -> the gap is fixed).
  * ``double_half_gaussian_outer`` -- lobe peaks sit at the core ENDS (fall away from A's peak,
    rise up to B's peak). Its lobes slide by ``f_delay`` (forward, lobe A) and ``r_delay`` (reverse,
    lobe B): peak A = ``pad - f_delay``, peak B = ``pad + 2*pw + gap + r_delay``. **Sign:** POSITIVE
    delay = more lead (fwd) / lag (rev) = a normal, well-separated STIRAP; NEGATIVE = toward the gap
    (the two outer lobes collapse into one continuous pulse). Clamped at the collapse floor
    ``f_delay + r_delay >= -(2*pw + gap)`` (the meet point; past it the lobes would cross).

Physics use: 556 = ``double_half_gaussian_inner`` (anchor), 308 = ``double_half_gaussian_outer``.
The inner/outer pairing bakes in the STIRAP counterintuitive order (308 leads fwd, 556 leads rev);
``f_delay``/``r_delay`` on the 308 tune each lobe's overlap. See ``YbScans/STIRAPAWGScan.py``.

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

#: shape name -> needs ``steepness``?  (only the symmetric ``gaussian`` does now; rise_/fall_
#: gaussian + the two-lobe STIRAP shapes use pw=1/e half-width and ignore steepness/smooth)
SHAPES = {
    "gaussian": True,
    "rise_gaussian": False,
    "fall_gaussian": False,
    "rise_linear": False,
    "fall_linear": False,
    "double_half_gaussian_inner": False,
    "double_half_gaussian_outer": False,
}

#: the two-lobe STIRAP shapes (built by :func:`_double_half_gaussian_envelope`, not the single-
#: envelope path). They read ``stirap_gap`` / ``f_delay`` / ``r_delay`` instead of steepness/smooth.
DOUBLE_HALF_GAUSSIAN_SHAPES = ("double_half_gaussian_inner", "double_half_gaussian_outer")


def _double_half_gaussian_envelope(shape, num_points, pw, stirap_gap, f_delay, r_delay):
    """Two-lobe STIRAP envelope. Returns ``(t_us, envelope)``; ``t_us`` runs [0, total].

    lobe = ``exp(-((t-peak)/pw)^2)`` (pw = 1/e half-width). pad = 2*pw each side ->
    total = 6*pw + gap. inner peaks face the gap (anchor, f/r IGNORED); outer peaks at the core
    ends and slide by f_delay/r_delay (+ = outward/more lead-lag = normal STIRAP; - = toward gap).
    Combined by MAX. Collapse floor: f_delay + r_delay >= -(2*pw + gap) (clamped).
    """
    if pw <= 0:
        raise ValueError("pulse_width_us must be > 0 (got %g)" % pw)
    if stirap_gap < 0:
        raise ValueError("stirap_gap must be >= 0 (got %g)" % stirap_gap)
    pad = 2.0 * pw
    total = 2.0 * pad + 2.0 * pw + stirap_gap                # = 6*pw + gap
    floor = -(2.0 * pw + stirap_gap)                          # collapse meet point
    if f_delay + r_delay < floor:                            # clamp past the meet (would cross)
        s = floor / (f_delay + r_delay)
        f_delay, r_delay = f_delay * s, r_delay * s
    t_us = np.linspace(0.0, 1.0, int(num_points)) * total
    if shape == "double_half_gaussian_inner":
        pA, pB = pad + pw, pad + pw + stirap_gap             # peaks face the gap (fixed anchor)
        gA = np.where(t_us <= pA, np.exp(-((t_us - pA) / pw) ** 2), 0.0)   # rise up to A
        gB = np.where(t_us >= pB, np.exp(-((t_us - pB) / pw) ** 2), 0.0)   # fall after B
    else:  # double_half_gaussian_outer
        pA = pad - f_delay                                   # peak at core-left, +f -> outward
        pB = pad + 2.0 * pw + stirap_gap + r_delay           # peak at core-right, +r -> outward
        gA = np.where(t_us >= pA, np.exp(-((t_us - pA) / pw) ** 2), 0.0)   # fall after A
        gB = np.where(t_us <= pB, np.exp(-((t_us - pB) / pw) ** 2), 0.0)   # rise up to B
    return t_us, np.maximum(gA, gB)


def _single_half_gaussian_envelope(shape, num_points, pw, f_delay=0.0):
    """One half-Gaussian lobe -- the SAME lobe formula as one double_half_gaussian lobe (a single
    ``exp(-((t-peak)/pw)^2)``, pw = 1/e half-width). total = 3*pw so the (undelayed) peak sits at
    pad+pw = 3*pw, IDENTICAL to the double's forward 556 lobe (pad=2*pw). Half-masked so the pulse
    stays a proper rise/fall as the peak moves.

    ``f_delay`` (us) shifts the peak INTO the window (both edges are pinned, so only inward moves
    are clip-free -- use f_delay >= 0):
      * ``rise_gaussian`` -- peak at 3*pw - f_delay  (base at the END; +f_delay -> EARLIER)
      * ``fall_gaussian`` -- peak at 0 + f_delay      (base at the START; +f_delay -> LATER)
    In a HalfPulse pair (556=rise anchor f=0, 308=fall with f_delay) a positive f_delay slides the
    308 later, toward the 556. ``r_delay`` has no effect (single lobe = no reverse). Only ``pw`` +
    ``f_delay`` -- no steepness / smooth (unified with the two-lobe shapes; t_wait for the 556 rise
    = 3*pw - f_delay, i.e. 3*pw when the 556 is the anchor).
    """
    if pw <= 0:
        raise ValueError("pulse_width_us must be > 0 (got %g)" % pw)
    total = 3.0 * pw
    t_us = np.linspace(0.0, 1.0, int(num_points)) * total
    if shape == "rise_gaussian":
        peak = 3.0 * pw - f_delay                            # base END; +f_delay -> earlier
        e = np.where(t_us <= peak, np.exp(-((t_us - peak) / pw) ** 2), 0.0)   # rise up to peak
    else:  # fall_gaussian
        peak = f_delay                                       # base START; +f_delay -> later
        e = np.where(t_us >= peak, np.exp(-((t_us - peak) / pw) ** 2), 0.0)   # fall after peak
    return t_us, e


def pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness,
                   *, stirap_gap=0.0, f_delay=0.0, r_delay=0.0):
    """Envelope for ``shape`` on ``num_points`` samples. Returns ``(t_us, envelope)``.

    ``t_us`` runs [0, total] where total = pulse_width_us (+ smooth_width_us for rise_*/fall_*);
    t=0 is the AWG trigger. For the two-lobe STIRAP shapes total = 6*pulse_width_us + stirap_gap
    and ``smooth_width_us``/``steepness`` are ignored (see :func:`_double_half_gaussian_envelope`).
    Pure helper -- also used by plotting/diagnostic scripts.
    """
    if shape not in SHAPES:
        raise ValueError("unknown pulse shape %r (valid: %s)" % (shape, ", ".join(sorted(SHAPES))))
    if shape in DOUBLE_HALF_GAUSSIAN_SHAPES:
        return _double_half_gaussian_envelope(shape, num_points, pulse_width_us,
                                              stirap_gap, f_delay, r_delay)
    if shape in ("rise_gaussian", "fall_gaussian"):
        return _single_half_gaussian_envelope(shape, num_points, pulse_width_us, f_delay)
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
    elif shape == "rise_linear":
        # linear ramp 0->1 over the main window, smooth 1->0 cosine tail appended after the peak
        m = t_us <= pulse_width_us
        e[m] = t_us[m] / pulse_width_us
        if smooth_width_us > 0:
            xs = (t_us[~m] - pulse_width_us) / smooth_width_us
            e[~m] = 0.5 * (1.0 + np.cos(np.pi * xs))
    else:  # fall_linear
        # smooth 0->1 cosine pre-rise prepended, then the linear ramp 1->0 over the main window
        m = t_us < smooth_width_us
        if smooth_width_us > 0:
            xs = t_us[m] / smooth_width_us
            e[m] = 0.5 * (1.0 - np.cos(np.pi * xs))
        e[~m] = 1.0 - (t_us[~m] - smooth_width_us) / pulse_width_us
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
            ``steepness`` (float, Gaussian-envelope factor; ignored by *_linear + the two-lobe
                shapes),
            ``amplitude_scale`` (float 0-1, default 1.0),
            ``max_amplitude_vpp`` (float, optional -- only for the voltage trace in ``info``).
            For ``double_half_gaussian_*`` also: ``stirap_gap`` (float >= 0, us of zero dead time
                between the two lobes; = the inner-peak separation), ``f_delay`` / ``r_delay``
                (float, us; slide the OUTER shape's fwd/rev lobe -- + = more lead/lag, ignored by
                the inner anchor). ``pulse_width_us`` is then the lobe 1/e half-width.

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
    stirap_gap = float(params.get("stirap_gap", 0.0))
    f_delay = float(params.get("f_delay", 0.0))
    r_delay = float(params.get("r_delay", 0.0))

    t_us, envelope = pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness,
                                    stirap_gap=stirap_gap, f_delay=f_delay, r_delay=r_delay)
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
