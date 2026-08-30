"""pulse_waveform.py -- general pulse-shape generator for the Siglent SDG6X AWGs.

Generalizes :mod:`gaussian_pulse_waveform` (the port of
``matlab_new/YbExptCtrl/sigilentAWG/gaussianPulseWaveform.m``) to a family of envelope shapes,
selected by the ``shape`` param (design agreed 2026-07-03; see ``pyctrl/tmp/pulse_10_examples.png``):

  * ``flat``           -- envelope == 1 over the whole window: a CONSTANT-amplitude carrier burst
                          at ``carrier_freq_MHz`` for ``pulse_width_us`` (total = pw). The output
                          peak reaches full ``max_amplitude_vpp * amplitude_scale``.
                          ``steepness`` / ``smooth_width_us`` / ``f_delay`` all IGNORED.
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
  * ``rise_cubic``     -- cubic Hermite smoothstep ``3x^2 - 2x^3`` rising 0 -> 1 with the peak at
                          the END of the window (design agreed 2026-07-15). COMPACT support:
                          ``total = pulse_width_us`` (NOT 3*pw like rise_gaussian), exactly 0 at
                          t=0 and exactly 1 at t=pw with zero slope at both ends (S'(0)=S'(1)=0).
                          ``steepness``/``smooth_width_us``/``f_delay`` all IGNORED.
  * ``fall_cubic``     -- mirror of ``rise_cubic``: ``S(1-x)`` = 1 - 3x^2 + 2x^3, peak at the
                          START (t=0), falling 1 -> 0. ``total = pulse_width_us``.
  * ``rise_quintic``   -- quintic smootherstep ``6x^5 - 15x^4 + 10x^3`` rising 0 -> 1, peak at the
                          END. Like rise_cubic but ALSO zero curvature at both ends
                          (S''(0)=S''(1)=0) -> no acceleration kink at turn-on or peak (the
                          smoothest adiabatic ramp). ``total = pulse_width_us``.
  * ``fall_quintic``   -- mirror of ``rise_quintic``: ``S(1-x)``, peak at the START, 1 -> 0.
                          Optional ``pad_time_us`` (us, default 0) PREPENDS a flat hold at amplitude 1
                          before the fall: hold 1 over [0, pad_time_us], then the quintic fall 1 -> 0
                          over ``pulse_width_us``. Extends total: ``total = pad_time_us + pulse_width_us``
                          (like ``smooth_width_us``). ``pad_time_us`` is IGNORED by the other spline
                          shapes (rise/fall cubic, rise_quintic).
  * ``chirped_rise_quintic`` / ``chirped_fall_quintic`` -- the SAME envelope as the un-prefixed
                          base shape (same total, same ``pad_time_us`` semantics); only the CARRIER
                          changes: it sweeps ``chirp_freq_MHz`` (a SIGNED TOTAL SPAN, final -
                          initial, in MHz; default 0) across the amplitude-ramp window. The ramp
                          coordinate is ``u = clip((t - pad)/pulse_width_us, 0, 1)`` where ``pad`` =
                          ``pad_time_us`` (only ``chirped_fall_quintic`` has one). During the pad
                          hold u = 0, so the carrier sits EXACTLY at ``carrier_freq_MHz`` and the
                          sweep begins where the ramp begins -> frequency is continuous across the
                          junction. ``chirp_profile`` picks the normalized sweep shape P(u):
                          ``"linear"`` (default) P(u) = u -> constant chirp rate; ``"quintic"``
                          P(u) = 6u^5 - 15u^4 + 10u^3 (the same smootherstep as the envelope) ->
                          chirp rate starts AND ends at zero. The carrier comes from the EXACT
                          analytic phase integral ``phase_cycles = f0*t + chirp*pw*Q(u)`` with
                          Q = integral of P (linear: ``u^2/2``; quintic: ``u^6 - 3u^5 + 2.5u^4``;
                          units MHz*us = cycles) -- NOT a stepped frequency -- so d(phase)/dt is
                          exactly the intended instantaneous frequency and the phase is continuous
                          everywhere. ``chirp_freq_MHz = 0`` reproduces the base shape BYTE-FOR-BYTE.

where ``x = t/pulse_width_us`` is normalized MAIN-window time (linear + spline shapes).

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
    "flat": False,
    "gaussian": True,
    "rise_gaussian": False,
    "fall_gaussian": False,
    "rise_linear": False,
    "fall_linear": False,
    "rise_cubic": False,
    "fall_cubic": False,
    "rise_quintic": False,
    "fall_quintic": False,
    "chirped_rise_quintic": False,
    "chirped_fall_quintic": False,
    "double_half_gaussian_inner": False,
    "double_half_gaussian_outer": False,
}

#: cubic/quintic spline shapes (built by :func:`_spline_envelope`). total = pulse_width_us, one
#: monotone ramp 0<->1 over the window with zero endpoint slope (quintic also zero endpoint
#: curvature). No steepness / smooth / f_delay. The ``chirped_`` variants share this envelope path
#: verbatim (see :func:`_base_shape`) -- only their carrier differs.
SPLINE_SHAPES = ("rise_cubic", "fall_cubic", "rise_quintic", "fall_quintic",
                 "chirped_rise_quintic", "chirped_fall_quintic")

#: the two-lobe STIRAP shapes (built by :func:`_double_half_gaussian_envelope`, not the single-
#: envelope path). They read ``stirap_gap`` / ``f_delay`` / ``r_delay`` instead of steepness/smooth.
DOUBLE_HALF_GAUSSIAN_SHAPES = ("double_half_gaussian_inner", "double_half_gaussian_outer")

#: shapes whose CARRIER sweeps (``chirp_freq_MHz`` / ``chirp_profile``); envelope identical to the
#: un-prefixed base shape. Every other shape ignores the two chirp fields entirely.
CHIRPED_SHAPES = ("chirped_rise_quintic", "chirped_fall_quintic")

_CHIRP_PREFIX = "chirped_"

#: ``chirp_profile`` -> ``(P, Q)``. P(u) = the normalized instantaneous-frequency profile (0 -> 1
#: across the ramp window, so f(t) = f0 + chirp*P(u)); Q = its integral with Q(0) = 0, which gives
#: the phase in cycles once scaled by ``chirp * pulse_width_us`` (MHz*us = cycles).
CHIRP_PROFILES = {
    "linear":  (lambda u: u,
                lambda u: 0.5 * u ** 2),
    "quintic": (lambda u: 6.0 * u ** 5 - 15.0 * u ** 4 + 10.0 * u ** 3,
                lambda u: u ** 6 - 3.0 * u ** 5 + 2.5 * u ** 4),
}


def _base_shape(shape):
    """``"chirped_fall_quintic"`` -> ``"fall_quintic"`` (any other name passes through unchanged).

    The chirp touches ONLY the carrier, so every envelope / total-width code path reuses the base
    shape's implementation verbatim by stripping the prefix first.
    """
    return shape[len(_CHIRP_PREFIX):] if shape.startswith(_CHIRP_PREFIX) else shape


def _chirp_profile_fns(chirp_profile):
    """``(P, Q)`` for ``chirp_profile``; raises ValueError on an unknown name."""
    try:
        return CHIRP_PROFILES[chirp_profile]
    except KeyError:
        raise ValueError("unknown chirp_profile %r (valid: %s)"
                         % (chirp_profile, ", ".join(sorted(CHIRP_PROFILES)))) from None


def _chirp_ramp_u(shape, t_us, pw, pad_time_us=0.0):
    """Ramp coordinate ``u = clip((t - pad)/pw, 0, 1)`` -- 0 through the pad hold, 0 -> 1 across the
    amplitude ramp. ``pad`` is ``pad_time_us`` for the fall (the flat pre-hold) and 0 for the rise,
    i.e. exactly the pad the envelope uses, so the sweep starts where the ramp starts.
    """
    pad = pad_time_us if _base_shape(shape) == "fall_quintic" else 0.0
    return np.clip((np.asarray(t_us, dtype=float) - pad) / pw, 0.0, 1.0)


def pulse_carrier_freq_MHz(shape, t_us, pulse_width_us, carrier_freq_MHz, *,
                           chirp_freq_MHz=0.0, chirp_profile="linear", pad_time_us=0.0):
    """Instantaneous carrier frequency (MHz) at each sample time in ``t_us``.

    ``f(t) = carrier_freq_MHz + chirp_freq_MHz * P(u)`` for the chirped shapes (u from
    :func:`_chirp_ramp_u`), a flat ``carrier_freq_MHz`` for every other shape. This is the EXACT
    derivative of the phase :func:`pulse_waveform` builds, so it is the analytic reference to
    compare a frequency measured off the uploaded blob against (``info["inst_freq_MHz"]`` is this
    same curve). Pure -- for plots/diagnostics as well as internal use.
    """
    t = np.asarray(t_us, dtype=float)
    if shape not in CHIRPED_SHAPES:
        return np.full(t.shape, float(carrier_freq_MHz))
    P, _Q = _chirp_profile_fns(chirp_profile)
    u = _chirp_ramp_u(shape, t, pulse_width_us, pad_time_us)
    return float(carrier_freq_MHz) + float(chirp_freq_MHz) * P(u)


def pulse_total_us(shape, pulse_width_us, smooth_width_us=0.0, stirap_gap=0.0, pad_time_us=0.0):
    """Total playback width (us) for ``shape`` -- the span all ``num_points`` are stretched over.

    Independent of ``num_points`` (it only sets sampling density), so it can be used to pick
    ``num_points`` from a target sample rate BEFORE building the envelope. Mirrors the totals in
    :func:`_double_half_gaussian_envelope` (6*pw+gap), :func:`_single_half_gaussian_envelope`
    (3*pw) and :func:`pulse_envelope` (gaussian=pw, *_linear=pw+smooth, fall_quintic=pw+pad_time_us).

    The ``chirped_`` variants have the SAME total as their base shape (the chirp only rewrites the
    carrier), so the prefix is stripped up front -- callers threading a config ``shape`` through
    (e.g. ``YbSteps/STIRAPPushoutStep``) need no change when a channel is switched to a chirped one.
    """
    shape = _base_shape(shape)
    if shape in DOUBLE_HALF_GAUSSIAN_SHAPES:
        return 6.0 * pulse_width_us + stirap_gap
    if shape in ("rise_gaussian", "fall_gaussian"):
        return 3.0 * pulse_width_us
    if shape == "fall_quintic":
        return pulse_width_us + pad_time_us             # flat hold prepended before the quintic fall
    if shape == "gaussian" or shape == "flat" or shape in SPLINE_SHAPES:
        return pulse_width_us                        # flat/gaussian/splines: one window of pw
    return pulse_width_us + smooth_width_us          # rise_linear / fall_linear


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


def _spline_envelope(shape, num_points, pw, pad_time_us=0.0):
    """Cubic/quintic-spline ramp over ONE window [0, pw]. Returns ``(t_us, envelope)``.

    x = t/pw in [0,1]. rise = S(x) (0 -> 1, peak at the END); fall = S(1-x) (1 -> 0, peak at the
    START). S is the Hermite smoothstep (cubic) or smootherstep (quintic):
      * cubic   S(x) = 3x^2 - 2x^3          -> S'(0)=S'(1)=0  (zero endpoint slope)
      * quintic S(x) = 6x^5 - 15x^4 + 10x^3 -> also S''(0)=S''(1)=0 (zero endpoint curvature)
    total = pw (compact support; exactly 0 / exactly 1 at the ends). No steepness/smooth/f_delay.

    ``fall_quintic`` only: ``pad_time_us`` (us) prepends a flat hold at amplitude 1 over [0, pad_time_us]
    before the quintic fall over the following ``pw`` -> total = pad_time_us + pw. ``pad_time_us`` is
    ignored by every other spline shape (they keep total = pw, pad_time_us=0 -> byte-identical).

    A ``chirped_`` prefix is stripped first: the chirped shapes have IDENTICAL envelopes to their
    base shape (only :func:`pulse_waveform`'s carrier differs).
    """
    shape = _base_shape(shape)
    if pw <= 0:
        raise ValueError("pulse_width_us must be > 0 (got %g)" % pw)
    if pad_time_us < 0:
        raise ValueError("pad_time_us must be >= 0 (got %g)" % pad_time_us)
    pad = pad_time_us if shape == "fall_quintic" else 0.0
    total = pw + pad
    t_us = np.linspace(0.0, 1.0, int(num_points)) * total
    if pad > 0.0:
        # flat hold at 1 over [0, pad], quintic fall over the remaining pw: x = (t-pad)/pw
        x = np.clip((t_us - pad) / pw, 0.0, 1.0)      # 0 during the hold, 0->1 over the fall window
        x = 1.0 - x                                   # mirror: 1 during the hold, 1 -> 0 over pw
    else:
        x = np.linspace(0.0, 1.0, int(num_points))    # normalized window time
        if shape.startswith("fall_"):
            x = 1.0 - x                               # mirror: peak at the start, 1 -> 0
    if shape in ("rise_cubic", "fall_cubic"):
        e = 3.0 * x ** 2 - 2.0 * x ** 3               # cubic Hermite smoothstep
    else:                                             # rise_quintic / fall_quintic
        e = 6.0 * x ** 5 - 15.0 * x ** 4 + 10.0 * x ** 3   # quintic smootherstep
    return t_us, e


def pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness,
                   *, stirap_gap=0.0, f_delay=0.0, r_delay=0.0, pad_time_us=0.0):
    """Envelope for ``shape`` on ``num_points`` samples. Returns ``(t_us, envelope)``.

    ``t_us`` runs [0, total] where total = pulse_width_us (+ smooth_width_us for rise_*/fall_*,
    + pad_time_us for fall_quintic); t=0 is the AWG trigger. For the two-lobe STIRAP shapes total =
    6*pulse_width_us + stirap_gap and ``smooth_width_us``/``steepness`` are ignored (see
    :func:`_double_half_gaussian_envelope`). Pure helper -- also used by plotting/diagnostic scripts.
    """
    if shape not in SHAPES:
        raise ValueError("unknown pulse shape %r (valid: %s)" % (shape, ", ".join(sorted(SHAPES))))
    if shape in DOUBLE_HALF_GAUSSIAN_SHAPES:
        return _double_half_gaussian_envelope(shape, num_points, pulse_width_us,
                                              stirap_gap, f_delay, r_delay)
    if shape in ("rise_gaussian", "fall_gaussian"):
        return _single_half_gaussian_envelope(shape, num_points, pulse_width_us, f_delay)
    if shape in SPLINE_SHAPES:
        return _spline_envelope(shape, num_points, pulse_width_us, pad_time_us)
    if smooth_width_us < 0:
        raise ValueError("smooth_width_us must be >= 0 (got %g)" % smooth_width_us)
    if shape in ("gaussian", "flat"):
        smooth_width_us = 0.0                       # gaussian edges ~0; flat is a bare hold
    total = pulse_width_us + smooth_width_us

    t_norm = np.linspace(0.0, 1.0, int(num_points))
    t_us = t_norm * total
    e = np.zeros(t_us.shape)

    if shape == "flat":
        e = np.ones(t_us.shape)                     # constant-amplitude carrier burst over [0, pw]
    elif shape == "gaussian":
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
            ``num_points`` (int, sample count, e.g. 10000 -- a FLOOR when ``sample_rate_MHz`` set),
            ``sample_rate_MHz`` (float >= 0, optional; target min effective sample rate in MSa/s.
                When >0, num_points is raised to ``ceil(sample_rate_MHz * total_width_us)`` so the
                density -- hence carrier Nyquist margin -- stays fixed as the STIRAP gap grows total.
                Default 0 = off = legacy fixed num_points, byte-identical),
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
            For ``fall_quintic`` also: ``pad_time_us`` (float >= 0, default 0 -- flat hold at
                amplitude 1 prepended before the quintic fall; total = pad_time_us + pulse_width_us;
                ignored by every other shape).
            For ``chirped_rise_quintic`` / ``chirped_fall_quintic`` also: ``chirp_freq_MHz``
                (float, default 0 -- the SIGNED TOTAL SPAN final-minus-initial, in MHz, swept across
                the amplitude-ramp window; 0 = byte-identical to the un-prefixed base shape) and
                ``chirp_profile`` (``"linear"`` (default) or ``"quintic"``, see
                :data:`CHIRP_PROFILES`). Both are IGNORED by every other shape.

    Returns:
        binary_data (bytes): big-endian int16 samples, ready to append to a WVDT command.
        info (dict): ``num_points``, ``freq_hz`` (= 1e6 / total width), ``t_us`` (ndarray, us
            since trigger), ``waveform`` (ndarray), ``shape``, ``total_width_us``,
            ``inst_freq_MHz`` (ndarray -- the analytic instantaneous carrier frequency per sample;
            flat ``carrier_freq_MHz`` for the unchirped shapes), and ``voltage`` (ndarray) when
            ``max_amplitude_vpp`` is given.
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
    pad_time_us = float(params.get("pad_time_us", 0.0))
    chirp_freq_MHz = float(params.get("chirp_freq_MHz", 0.0))      # SIGNED total span, MHz
    chirp_profile = str(params.get("chirp_profile", "linear"))

    # Optional: scale num_points so the effective sample rate = num_points/total stays >= a target
    # (MSa/s). Since total = 6*pw + gap grows with the STIRAP gap while num_points is otherwise a
    # fixed constant, a big gap silently collapses the sample rate below the carrier Nyquist ->
    # aliased carrier / coarse lobes. sample_rate_MHz>0 pins the density (num_points ~ total); it
    # only ever RAISES num_points, so shapes with sample_rate_MHz unset (0) stay byte-identical.
    sample_rate_MHz = float(params.get("sample_rate_MHz", 0.0))
    if sample_rate_MHz > 0.0:
        total_us = pulse_total_us(shape, pulse_width_us, smooth_width_us, stirap_gap, pad_time_us)
        num_points = max(num_points, int(np.ceil(sample_rate_MHz * total_us)))

    t_us, envelope = pulse_envelope(shape, num_points, pulse_width_us, smooth_width_us, steepness,
                                    stirap_gap=stirap_gap, f_delay=f_delay, r_delay=r_delay,
                                    pad_time_us=pad_time_us)
    total = t_us[-1]

    # Carrier: exactly carrier_freq_MHz over the TOTAL width. Phrased as oscillations x normalized
    # time so shape="gaussian" reproduces gaussianPulseWaveform byte-for-byte (num_osc = f * pw).
    t_norm = np.linspace(0.0, 1.0, num_points)
    num_oscillations = carrier_freq_MHz * total
    phase_cycles = num_oscillations * t_norm
    inst_freq_MHz = pulse_carrier_freq_MHz(shape, t_us, pulse_width_us, carrier_freq_MHz,
                                           chirp_freq_MHz=chirp_freq_MHz,
                                           chirp_profile=chirp_profile,
                                           pad_time_us=pad_time_us)
    if shape in CHIRPED_SHAPES and chirp_freq_MHz != 0.0:
        # Chirped carrier from the EXACT analytic phase integral (cycles):
        #     phase(t) = f0*t + chirp*pw*Q(u),   u = clip((t-pad)/pw, 0, 1),  Q = integral of P.
        # d(phase)/dt is then EXACTLY inst_freq_MHz, and the phase is continuous across the
        # pad->ramp junction (a stepped-frequency carrier would jump there). chirp == 0 skips this
        # branch entirely -> byte-identical to the un-prefixed base shape.
        _P, Q = _chirp_profile_fns(chirp_profile)
        u = _chirp_ramp_u(shape, t_us, pulse_width_us, pad_time_us)
        phase_cycles = phase_cycles + chirp_freq_MHz * pulse_width_us * Q(u)
    carrier = np.sin(2.0 * np.pi * phase_cycles)

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
        "inst_freq_MHz": inst_freq_MHz,
    }
    if "max_amplitude_vpp" in params and params["max_amplitude_vpp"] is not None:
        info["voltage"] = waveform * float(params["max_amplitude_vpp"]) / 2.0
    return binary_data, info
