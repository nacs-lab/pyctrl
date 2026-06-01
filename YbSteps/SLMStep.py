"""SLMStep.py -- transliteration of ``matlab_new/YbSteps/SLMStep.m``.

``g = s.C.SLM``. Turn on the SLM tweezer AOM and set the servo voltage. Reads
resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().SLM.X.Y)``);
``Consts()`` default args are SubProps that DynProps eagerly resolves to a number.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced, so the resolved
config values serialize as float64 -- faithful to MATLAB's doubles.
"""

from consts import Consts


def SLMStep(s, g):
    # Turn on SLM Tweezer AOM and set the servo voltage
    Freq_SLM = g.AOM.Freq(Consts().SLM.AOM.Freq)
    Amp_SLM = g.AOM.Amp(Consts().SLM.AOM.Amp)
    V_SLMServo = g.VServo(Consts().SLM.VServo)

    (s.add('FreqSLM', Freq_SLM)
        .add('AmpSLM', Amp_SLM)
        .add('VSLMservo', V_SLMServo))
