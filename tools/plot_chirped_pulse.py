"""plot_chirped_pulse.py -- plot the waveform ACTUALLY UPLOADED to a Siglent SDG6X channel.

Diagnostic for the chirped AWG pulse shapes (``chirped_rise_quintic`` / ``chirped_fall_quintic`` /
``chirped_flat``, see ``devices/sigilent_awg/pulse_waveform.py``). It does NOT re-evaluate an
idealized waveform: it calls :func:`pulse_waveform`, takes the BIG-ENDIAN int16 blob that
``AWGConnection`` would append to the WVDT command, decodes it back to volts, and measures the
instantaneous frequency off those samples. What you see is what the box plays (module the
DAC/analog front end).

Default example = the RearrangeSTIRAPScan forward-Stokes pulse: AWG308.Ch1, 200 MHz carrier,
pulse_width_us=3, pad_time_us=2, amplitude_scale=1, 8 Vpp, num_points=10000 floored to
sample_rate_MHz=2500 (-> 12500 pts over the 5 us total), chirped by 0.2 MHz.

Frequency measurement -- HETERODYNE, not FFT-Hilbert. Multiply by exp(-i*2*pi*f0*t), low-pass with
a Hann boxcar to kill the 2*f0 image, unwrap the phase, differentiate. An FFT-based analytic signal
puts ~1 MHz of bogus error on this waveform (the AM envelope + the non-periodic ends smear it),
which is 5x the whole 0.2 MHz chirp span.

Run (base anaconda python is enough -- numpy + matplotlib only, no engine, no hardware):
    python tools/plot_chirped_pulse.py
    python tools/plot_chirped_pulse.py --profile quintic --chirp 0.5 --out tmp/chirp_quintic.png
    python tools/plot_chirped_pulse.py --shape chirped_flat --chirp 5 --out tmp/chirped_flat.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devices.sigilent_awg.pulse_waveform import (pulse_waveform, pulse_envelope,  # noqa: E402
                                                 pulse_pad_us, CHIRP_PROFILES)

_PYCTRL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT = os.path.dirname(_PYCTRL)
#: mirror target read by the operator's viewer (see the claude-display convention)
_MIRROR = os.path.join(_PROJECT, "tmp", "claude_display.png")


def decode_blob(binary_data, max_amplitude_vpp):
    """Decode the uploaded big-endian int16 blob back to VOLTS.

    The SDG6X maps the full int16 code range onto the configured peak-to-peak amplitude, so
    code 32767 = +Vpp/2. This is the inverse of the ``np.round(w * 32767).astype('>i2')`` step in
    :func:`pulse_waveform` -- i.e. it includes the real quantization the box receives.
    """
    codes = np.frombuffer(binary_data, dtype=">i2").astype(float)
    return codes / 32767.0 * (max_amplitude_vpp / 2.0)


def heterodyne_freq_MHz(t_us, x, f0_MHz, win_pts):
    """Instantaneous frequency (MHz) measured off real samples ``x`` sampled at ``t_us`` (us).

    Mix down to baseband by ``f0_MHz``, Hann-boxcar low-pass (removes the 2*f0 image the real-valued
    mix creates), unwrap, differentiate. ``win_pts`` should span several carrier periods and stay
    well under the timescale the chirp moves on.
    """
    z = x.astype(complex) * np.exp(-2j * np.pi * f0_MHz * t_us)
    k = np.hanning(win_pts)
    z = np.convolve(z, k / k.sum(), mode="same")
    return f0_MHz + np.gradient(np.unwrap(np.angle(z)), t_us) / (2.0 * np.pi), np.abs(z)


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="chirped_fall_quintic",
                    help="chirped_fall_quintic (default, AWG308 fwd Stokes), chirped_rise_quintic, "
                         "or chirped_flat (constant amplitude, sweep spans the whole burst)")
    ap.add_argument("--f0", type=float, default=200.0, help="carrier_freq_MHz (default 200)")
    ap.add_argument("--chirp", type=float, default=0.2,
                    help="chirp_freq_MHz -- SIGNED TOTAL SPAN final-initial (default 0.2)")
    ap.add_argument("--profile", default="linear", choices=sorted(CHIRP_PROFILES),
                    help="chirp_profile (default linear)")
    ap.add_argument("--pw", type=float, default=3.0, help="pulse_width_us (default 3)")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="pad_time_us -- hold before the swept window; honored by "
                         "chirped_fall_quintic + chirped_flat, ignored by the rise (default 2)")
    ap.add_argument("--amp-scale", type=float, default=1.0, help="amplitude_scale (default 1)")
    ap.add_argument("--vpp", type=float, default=8.0, help="max_amplitude_vpp (default 8)")
    ap.add_argument("--num-points", type=int, default=10000, help="num_points floor (default 10000)")
    ap.add_argument("--sample-rate", type=float, default=2500.0,
                    help="sample_rate_MHz floor (default 2500)")
    ap.add_argument("--out", default=os.path.join("tmp", "chirped_pulse_awg308.png"),
                    help="output PNG, relative to pyctrl/ (default tmp/chirped_pulse_awg308.png)")
    ap.add_argument("--no-mirror", action="store_true", help="skip the claude_display.png mirror")
    return ap.parse_args()


def main():
    a = build_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = {
        "shape": a.shape, "carrier_freq_MHz": a.f0, "pulse_width_us": a.pw,
        "pad_time_us": a.pad, "chirp_freq_MHz": a.chirp, "chirp_profile": a.profile,
        "amplitude_scale": a.amp_scale, "max_amplitude_vpp": a.vpp,
        "num_points": a.num_points, "sample_rate_MHz": a.sample_rate,
        "smooth_width_us": 0.0, "steepness": 3.5,
    }
    blob, info = pulse_waveform(params)

    t = info["t_us"]
    v = decode_blob(blob, a.vpp)                      # VOLTS, from the uploaded bytes
    n, total = info["num_points"], info["total_width_us"]
    dt = total / (n - 1)                              # us per sample
    # ~20 carrier periods: long enough to bury the 2*f0 mixing image, short enough not to
    # smear the chirp curvature (swept 10/20/40/80/160 -- 20 is the residual minimum).
    win = max(9, int(round(20.0 / a.f0 / dt)) | 1)
    meas, mag = heterodyne_freq_MHz(t, v, a.f0, win)
    analytic = info["inst_freq_MHz"]

    # Valid region for the measurement: away from the boxcar edge transients and out of the deep
    # AM tail (where the carrier phase is quantization noise, not signal).
    ok = np.zeros(t.shape, dtype=bool)
    ok[win:-win] = True
    ok &= mag > 0.10 * mag.max()
    err_kHz = (meas - analytic) * 1e3

    # ask the library which shapes honor pad_time_us (fall_quintic + flat); rise ignores it
    pad = pulse_pad_us(a.shape, a.pad)
    ramp0, ramp1 = pad, pad + a.pw

    # AMPLITUDE envelope (info["waveform"] is the MODULATED trace, not the envelope). Scaled to the
    # blob's own peak so it is the true outer bound of the decoded samples.
    _, env = pulse_envelope(a.shape, n, a.pw, 0.0, 0.0, pad_time_us=pad)
    env_v = env * (np.abs(v).max() / env.max())

    # The same pulse with the chirp OFF -- byte-identical to the un-prefixed base shape. Overlaid in
    # the zooms it makes the accumulated chirp phase visible directly in the samples.
    blob0, _ = pulse_waveform(dict(params, chirp_freq_MHz=0.0))
    v0 = decode_blob(blob0, a.vpp)

    fig = plt.figure(figsize=(11.0, 9.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.1, 1.5, 1.8, 1.1], hspace=0.42, wspace=0.22,
                          left=0.085, right=0.975, top=0.925, bottom=0.085)
    ax_v = fig.add_subplot(gs[0, :])
    ax_z0 = fig.add_subplot(gs[1, 0])
    ax_z1 = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[2, :])
    ax_r = fig.add_subplot(gs[3, :], sharex=ax_f)

    for ax in (ax_v, ax_f, ax_r):
        if pad > 0:
            ax.axvspan(0, pad, color="0.93", zorder=0)
        ax.axvline(ramp0, color="0.55", lw=0.9, ls="--", zorder=1)
        ax.axvline(ramp1, color="0.55", lw=0.9, ls="--", zorder=1)

    # ---- 1. the uploaded waveform, in volts ------------------------------------------------
    ax_v.plot(t, v, lw=0.35, color="#1f77b4", label="uploaded samples (decoded int16)")
    ax_v.plot(t, env_v, lw=1.4, color="#d62728", label="envelope")
    ax_v.plot(t, -env_v, lw=1.4, color="#d62728")
    ax_v.set_ylabel("output [V]")
    ax_v.set_xlim(0, total)
    ax_v.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_v.set_title("%s   f0=%.4f MHz, chirp=%+g MHz (%s), pw=%g us, pad=%g us, "
                   "%g Vpp x %g\n%d pts over %.3f us (%.0f MSa/s), DDS FREQ=%.1f Hz"
                   % (a.shape, a.f0, a.chirp, a.profile, a.pw, pad, a.vpp, a.amp_scale,
                      n, total, n / total, info["freq_hz"]), fontsize=10)

    # ---- 2. zooms: the junction (no phase step) and mid-ramp (accumulated chirp phase) --------
    ncyc = 5.0 / a.f0                                  # ~5 carrier periods each side
    P, Q = CHIRP_PROFILES[a.profile]
    u_mid = 0.6                                        # still ~1/3 amplitude on a quintic fall
    t_mid = ramp0 + u_mid * a.pw
    slip_deg = 360.0 * a.chirp * a.pw * float(Q(np.array(u_mid)))
    # with no pad the left zoom is the burst START, not a pad->sweep junction
    z0_ttl = ("burst start: carrier exactly f0" if pad == 0 else
              "pad -> sweep junction: carrier still f0, NO phase step")
    for ax, c, ttl in ((ax_z0, ramp0, z0_ttl),
                       (ax_z1, t_mid,
                        "mid-sweep (u=%.1f): %+.0f deg ahead of the unchirped pulse" % (u_mid, slip_deg))):
        m = (t >= c - ncyc) & (t <= c + ncyc)
        ax.plot(t[m], v0[m], lw=1.6, color="0.72", label="chirp = 0")
        ax.plot(t[m], v[m], lw=0.9, color="#1f77b4", marker=".", ms=2.2,
                label="chirp = %+g MHz" % a.chirp)
        ax.plot(t[m], env_v[m], lw=1.0, color="#d62728")
        ax.plot(t[m], -env_v[m], lw=1.0, color="#d62728")
        ax.axvline(c, color="0.55", lw=0.9, ls="--")
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xlabel("t [us]")
        ax.set_ylabel("output [V]")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # ---- 3. instantaneous frequency: measured vs analytic -------------------------------------
    ax_f.plot(t, analytic, lw=3.2, color="#7fd18a",
              label="analytic  f0 + chirp*P(u)   (= d(phase)/dt)")
    ax_f.plot(t[ok], meas[ok], lw=1.0, color="#111111", ls="--",
              label="measured off the uploaded samples (heterodyne)")
    ax_f.set_ylabel("inst. frequency [MHz]")
    ax_f.legend(loc="upper left", fontsize=8, framealpha=0.9)
    span = abs(a.chirp) if a.chirp else 1.0
    ax_f.set_ylim(min(a.f0, a.f0 + a.chirp) - 0.15 * span,
                  max(a.f0, a.f0 + a.chirp) + 0.25 * span)

    # ---- 4. residual --------------------------------------------------------------------------
    ax_r.axhline(0.0, color="0.5", lw=0.8)
    ax_r.plot(t[ok], err_kHz[ok], lw=0.9, color="#9467bd")
    ax_r.set_ylabel("measured - analytic\n[kHz]")
    ax_r.set_xlabel("t [us]  (0 = AWG trigger)")
    ax_r.set_xlim(0, total)
    rms = float(np.sqrt(np.mean(err_kHz[ok] ** 2)))
    ax_r.set_title("residual: max |err| = %.2f kHz, rms = %.2f kHz  (chirp span = %.0f kHz)"
                   % (np.abs(err_kHz[ok]).max(), rms, abs(a.chirp) * 1e3), fontsize=9)

    out = a.out if os.path.isabs(a.out) else os.path.join(_PYCTRL, a.out)
    out = os.path.abspath(out)
    outdir = os.path.dirname(out)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fig.text(0.005, 0.006, out, fontsize=6.5, color="0.35", ha="left", va="bottom")
    fig.savefig(out, dpi=130)
    print("wrote %s" % out)

    if not a.no_mirror:
        try:
            if not os.path.isdir(os.path.dirname(_MIRROR)):
                os.makedirs(os.path.dirname(_MIRROR))
            fig.savefig(_MIRROR, dpi=130)
            print("mirrored %s" % _MIRROR)
        except OSError as e:                       # a mirror failure must not lose the real figure
            print("mirror skipped (%s)" % e)
    plt.close(fig)

    print("total_width_us = %.6f   num_points = %d   freq_hz = %.4f" % (total, n, info["freq_hz"]))
    print("f(start) = %.6f MHz   f(end) = %.6f MHz   span = %+.6f MHz"
          % (analytic[0], analytic[-1], analytic[-1] - analytic[0]))
    print("measured vs analytic over %d valid samples: max %.3f kHz, rms %.3f kHz"
          % (ok.sum(), np.abs(err_kHz[ok]).max(), rms))
    # chirp = 0 must be byte-identical to the un-prefixed base shape
    base = dict(params, shape=a.shape.replace("chirped_", ""))
    zero, _ = pulse_waveform(dict(params, chirp_freq_MHz=0.0))
    ref, _ = pulse_waveform(base)
    print("chirp=0 byte-identical to %s: %s" % (base["shape"], zero == ref))


if __name__ == "__main__":
    main()
