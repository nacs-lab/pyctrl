"""SLMTrapModulationStep.py -- transliteration of ``matlab_new/YbSteps/SLMTrapModulationStep.m``.

``g = s.C.SLMTrapModulation``. Modulates the SLM trap AOM (frequency + amplitude) for a
set modulation time while firing a scope trigger, then turns the modulation off, lowers
the trap depth via the SLM servo voltage (ramped) to let hot atoms fly out, waits, and
ramps the servo back to its init value.

All arithmetic here is on concrete config floats (no globals/measures), so the int->float
SeqVal-operand exception does not apply. Pulse values (``1``/``0``) are written verbatim
and float()-coerced by the framework.
"""

from consts import Consts
from ramp_to import ramp_to


def SLMTrapModulationStep(s, g):
    Amp_SLMmodulation_factor = g.AmpFactor(Consts().SLMTrapModulation.AmpFactor)
    Amp_SLMmodulation_real = Amp_SLMmodulation_factor * Consts().SLM.AOM.Amp

    Freq_SLMmodulation = g.Freq(Consts().SLMTrapModulation.Freq)
    Freq_SLMmodulation_real = Freq_SLMmodulation + Consts().SLM.AOM.Freq

    # ideally t_Modulation should be inversely propotional to
    # Freq_SLMmodulation, and we assume this is set in the Scan file
    # If not, we will use a constant modulation time
    t_Modulation = g.Time(Consts().SLMTrapModulation.Time)

    V_lowerTrapDepth = g.lowerTrapDepth.Vservo(Consts().SLMTrapModulation.lowerTrapDepth.Vservo)
    t_lowerTrapDepth = g.lowerTrapDepth.Time(Consts().SLMTrapModulation.lowerTrapDepth.Time)

    s.add('FreqSLMmodulation', Freq_SLMmodulation_real).add('AmpSLMmodulation', Amp_SLMmodulation_real)
    s.add('TTLScopeTrig', 1)

    s.wait(t_Modulation)  # might need to change the modulation to be dynamical depending on the modulating freq.

    s.add('FreqSLMmodulation', 0).add('AmpSLMmodulation', 0)
    s.add('TTLScopeTrig', 0)

    # lower the trap depth and wait for hot atom to fly out
    s.add_step(10e-6).add('VSLMservo', ramp_to(V_lowerTrapDepth))
    s.wait(t_lowerTrapDepth)
    s.add_step(10e-6).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))
