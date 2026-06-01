"""ReleaseRecaptureStep.py -- transliteration of ``matlab_new/YbSteps/ReleaseRecaptureStep.m``.

``g = s.C.ReleaseRecapture``. Release-and-recapture: drop the SLM trap power (release
the atoms), hold for the release time, then restore the SLM AOM amplitude (recapture)
and re-enable the sample-and-hold. A scope trigger brackets the release window.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by ``_resolve_pulse``,
so bare ints (``0``/``1``) serialize as ARG_CONST_FLOAT64 -- faithful to MATLAB's doubles.
"""

from consts import Consts


def ReleaseRecaptureStep(s, g):
    t_release = g.Time(0)

    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)

    s.add('AmpSLM', 0).add('TTLSampleAndHold', 0)
    s.add('TTLScopeTrig', 1)
    s.wait(t_release)
    s.add('TTLScopeTrig', 0)
    s.add('AmpSLM', Amp_SLM)
    s.wait(3e-6)  # wait the response time of our 532 AOM
    s.add('TTLSampleAndHold', 1)
