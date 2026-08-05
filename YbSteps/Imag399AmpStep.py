"""Imag399AmpStep.py -- ``Imag399Step`` with the two 399 DDS amplitudes AND the 399 pulse
length overridable per image.

WHY THIS EXISTS (the imaging-power policy, 2026-07-27). In a MULTI-ROUND rearrangement shot
(RearrangeCommSeq2: loading -> middle -> final, three camera frames) the 399 imaging-power PID
is engaged ONCE, in the ROOT ``BlueMOTStep``, at the LOADING pattern's ``BlueMOT.Img1/Img2PIDSet``
-- and then HELD (PIDMode TTL back to 0) for the rest of the shot. There is no time to re-PID
mid-shot, and trying to rails the integrator (see the memory note
``gotcha-imaging-pid-held-multiround-rearrange``). So the middle and final frames are physically
taken at the LOADING pattern's held optical power, and the ONLY per-frame brightness knob left is
the DDS amplitude of each frame's imaging beams:

    beam 1 -> ``AmpAbsImag``    (``Imag399.Amp1``, the 369-fiber-output imaging beam)
    beam 2 -> ``Amp399Imag2``   (``Imag399.Amp2``, the second imaging beam)

AOM knee (measured 2026-07-16, R212 _214049): amps 0.5-1.0 are optically FLAT; only <= 0.5
actually attenuates. DDS amp -> optical power is NONLINEAR -- sweep it, never compute a ratio.

PULSE LENGTH (``t_exp``, added 2026-07-28). Once the amps are railed at 1/1 and the PID setpoints
are pinned (policy above), the ONLY brightness knob left for a given frame is TIME: a longer 399
pulse deposits more photons inside the camera's exposure window -> larger per-site separation /
d' -> fewer false-empty mis-reads. The base step takes the pulse length from
``g.ExposureTime`` (expConfig cross-refs it to ``Orca.ExposureTime``, so it normally tracks the
camera window exactly); ``t_exp >= 0`` replaces it for THIS image only. Note the CAMERA exposure
is a single global hardware setting the runner syncs once per scan
(``camera_runtime.sync_camera_exposure``), so this override moves the 399 PULSE inside a FIXED
window -- lengthening the pulse past ``Orca.ExposureTime`` collects nothing extra (the tail falls
outside the frame) and just costs shot time. Extra 399 heating is harmless on the LAST image of a
shot (no atoms are needed after it) but is NOT harmless on any earlier one.

This step is byte-identical to :func:`Imag399Step.Imag399Step` whenever ``a1``/``a2``/``t_exp``
are all negative (the "no override" sentinel) -- pinned by
``tests/test_rearrange_imaging_amps.py::test_imag399_amp_step_default_equals_imag399_step``. Keep
the two bodies in lockstep: any edit to ``Imag399Step`` must be mirrored here (the test will fail
loudly otherwise).

NOTE: ``YbSeqs/RearrangeCommSeq2Dev.py`` carries a private, identical copy of this body
(``Imag399DevStep``, added 2026-07-19). Collapse it onto this module when that sandbox is next
touched; it is deliberately left alone here to keep this change minimal.
"""

from consts import Consts


def Imag399AmpStep(s, g, a1=-1.0, a2=-1.0, t_exp=-1.0):
    """``Imag399Step`` with per-image DDS-amp and pulse-length overrides.

    Args:
        s, g: the usual step args (``g = s.C.Imag399``, resolved through this bseq's
            per-pattern ByPattern overlay).
        a1, a2: the 399 beam-1 / beam-2 DDS amplitudes for THIS image. ``< 0`` (the default)
            means "no override" -> fall back to ``g.Amp1``/``g.Amp2`` exactly as the base step
            does. Both amps are set back to 0 inside the step's own turn-off block (identical
            to the base step), so nothing leaks into the next image.
        t_exp: the 399 PULSE length (seconds) for THIS image. ``< 0`` (the default) means "no
            override" -> fall back to ``g.ExposureTime`` exactly as the base step does. Purely
            local: the pulse ends inside this step, so nothing leaks into the next image. See
            the pulse-length note in the module docstring (fixed camera window; heating only
            harmless on a shot's LAST image).
    """
    t_Imag399 = t_exp if t_exp >= 0 else g.ExposureTime(Consts().Imag399.ExposureTime)

    Freq_Resonance399 = Consts().Resonance399Freq
    Freq_Imag399Detuning = g.FreqDetuning(Consts().Imag399.FreqDetuning)
    Freq_Imag399 = Freq_Resonance399 + Freq_Imag399Detuning
    Amp_Imag399_1 = a1 if a1 >= 0 else g.Amp1(Consts().Imag399.Amp1)
    Amp_Imag399_2 = a2 if a2 >= 0 else g.Amp2(Consts().Imag399.Amp2)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq

    Freq_Cool556DetuningX = g.Cool556.X.FreqDetuning(Consts().Imag399.Cool556.X.FreqDetuning)
    Freq_Cool556Detuningh = g.Cool556.h.FreqDetuning(Consts().Imag399.Cool556.h.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Freq_Cool556h = Freq_Resonance556mj0Freq + Freq_Cool556Detuningh

    Amp_Cool556X = g.Cool556.X.Amp(Consts().Imag399.Cool556.X.Amp)
    Amp_Cool556h = g.Cool556.h.Amp(Consts().Imag399.Cool556.h.Amp)

    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    s.add('TTL556RydbergShutter', 0)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', Amp_Cool556h)

    s.wait(3e-3)  # wait for the shutter

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', Amp_Imag399_1)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', Amp_Imag399_2)
    s.add_step(100e-6).add('TTLOrcaTrig', 1)

    s.add('TTLOrcaTrig', 0)

    s.wait(t_Imag399)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', 0)

    s.add('FreqAbsImag', Freq_Imag399).add('AmpAbsImag', 0)
    s.add('Freq399Imag2', Freq_Imag399).add('Amp399Imag2', 0)

    s.add('TTL399AbsImagShutter', 0)
    s.add('TTL399Imag2Shutter', 0)
    s.wait(3e-3)  # wait for the shutter
