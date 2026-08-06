"""RearrangeCool556hXStep.py -- the post-rearrangement 556 X+h recool, on its OWN config block.

``g = s.C.RearrangeCool556``. Functionally identical to :mod:`Cool556hXStep` (power-balanced green
556 cooling on the X and h MOT beams: set freq = resonance + per-beam detuning, set per-beam amp,
hold, then zero the amps with the frequencies held) but it reads a **separate** config namespace so
the recool that precedes ``STIRAPPushoutStep`` can be optimized INDEPENDENTLY of the release-recapture
``Cool556`` block.

Why it exists (2026-08-05/06). The STIRAP science block used to recool via ``Cool556Step``, which
drives both beams from the single top-level ``Cool556.FreqDetuning``/``Amp`` (0.14 MHz / 0.08) rather
than the per-beam ``Cool556.X``/``.h`` sub-blocks that the 2026-07-20 RNR campaign optimized
(0.12 MHz / 0.14 on both). Swapping the science block to ``Cool556hXStep`` lifted mid->final survival
0.341 -> 0.514 (scan 20260805_231444 vs 20260806_000351, every gap point +6..+11 sigma). That
confirmed the recool was the dominant limiter -- but it also tied the STIRAP recool to the RNR-tuned
numbers, and the two want different values: the STIRAP atoms arrive HOTTER (they have been transported
by the SLM with ``RearrCoolAmp = 0``, and imaged twice) so they plausibly want a longer / stronger
recool than the RNR release-recapture does. This step gives that knob its own home.

Scannable via ``RearrangeSTIRAPScan.py``: ``RearrangeCool556.Time``, ``.X.{FreqDetuning,Amp}``,
``.h.{FreqDetuning,Amp}``.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by ``_resolve_pulse``, so the bare
``0`` amplitudes serialize as ARG_CONST_FLOAT64 -- faithful to MATLAB's doubles. No globals/measures
here, so no explicit-float operand coercion is needed. Channel set + ordering mirror
``Cool556hXStep`` exactly, so with matched config values this step is byte-identical to it.
"""

from consts import Consts


def RearrangeCool556hXStep(s, g):
    t_Cool556 = g.Time(Consts().RearrangeCool556.Time)

    Freq_Resonance556mj0Freq = Consts().Resonance556mj0Freq

    Freq_Cool556DetuningX = g.X.FreqDetuning(Consts().RearrangeCool556.X.FreqDetuning)
    Freq_Cool556X = Freq_Resonance556mj0Freq + Freq_Cool556DetuningX
    Amp_Cool556X = g.X.Amp(Consts().RearrangeCool556.X.Amp)

    Freq_Cool556Detuningh = g.h.FreqDetuning(Consts().RearrangeCool556.h.FreqDetuning)
    Freq_Cool556h = Freq_Resonance556mj0Freq + Freq_Cool556Detuningh
    Amp_Cool556h = g.h.Amp(Consts().RearrangeCool556.h.Amp)

    s.add('AmpAbsImag', 0)

    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', Amp_Cool556X)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', Amp_Cool556h)
    s.wait(t_Cool556)
    s.add('Freq556MOTX', Freq_Cool556X).add('Amp556MOTX', 0)
    s.add('Freq556RydbergMOTh', Freq_Cool556h).add('Amp556RydbergMOTh', 0)
