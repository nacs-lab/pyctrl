"""RydbergHighFieldPushoutStep.py -- High field (> 62G) Rydberg pushout using double double pass AOM

``g = s.C.Pushout``. Rydberg / high-field variant of the push-out step. Applies the Ryd bias
coil field (current->voltage), opens the 369 + 556-Rydberg shutters (closing the 556 MOTa
shutter), lowers the SLM trap depth, fires the 556 *Rydberg* (h) beam + 308 + the QICK
microwave trigger for the push-out time, then restores trap depth / shutters and zeroes the coil.

Differs from ``PushoutStep`` (which pushes with the 556 MOT beams + 399 at the standing trap
depth and applies NO field): this step applies a field (> 60 A on ``BiasCoilCurrent.Ryd``),
pushes with the Rydberg 556 beam only (not MOTX/399), lowers the trap (``VSLMservo`` ramp to 0.4),
and pulses 308 + the QICK microwave (``TTLQickTrig``). Used for high-field / Rydberg push-out
spectroscopy.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``);
``Consts()`` default args are SubProps that DynProps resolves to a number. As in ``PushoutStep``,
some reads are computed but unused (``Freq/Amp_Pushout399``).
"""

from consts import Consts
from ramp_to import ramp_to


def RydbergHighFieldPushoutStep(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    Freq_Pushout556 = g.Green.Freq(Consts().Pushout.Green.Freq)
    Amp_Pushout556 = g.Green.Amp(Consts().Pushout.Green.Amp)
    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)

    Amp_Pushout308 = g.Ryd308.Amp(Consts().Pushout.Ryd308.Amp)
    Amp_Pushout369 = g.Amp369(Consts().Pushout.Ionization.Amp)

    Amp_AOM616Divert = Consts().AOM616Divert.Amp()

    I_RydCoil = g.BiasCoilCurrent.Ryd(0)
    V_RydCoil = 5 * I_RydCoil / 100
    s.add('VRydCoil', V_RydCoil)

    # Turn on the 369 shutter
    s.add('TTL369Shutter', 1)

    # Turn off the low field 556 rydberg shutter, turn off the 556 MOT shutter. 
    # The high field 556 rydberg path currently has no shutter.
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL556MOTaShutter', 0) #h 
    s.add('TTL556MOTbShutter', 0)
    s.add('TTL556MOTcShutter', 0)
    
    s.wait(50e-3)

    V_RydTrap = g.VRydTrap(0.03)                       # pushout trap depth; default 0.03 (0.5 for STIRAP)
    s.add_step(1e-3).add('VSLMservo', ramp_to(V_RydTrap))

    # prevent trap depth flucations during pushout
    s.wait(1e-3)

    s.add('TTLScopeTrig', 1)

    # Now we switch on the 556 Rydberg beam
    # The second double pass AOM is default at lower end and highest amp. It's used as a switch
    s.add('Freq556RydbergHF', 120e6).add('Amp556RydbergHF', 0.9)
    # We still use the first double pass AOM to set the frequency and amplitude of the HF 556 Rydberg beam
    s.add('Freq556RydbergMOTh', Freq_Pushout556).add('Amp556RydbergMOTh', Amp_Pushout556)
    
    # Dump all the power to the 308 cavity
    s.add('AmpAOM616', 0)
    # Use the 308 AOM to determine whether we want 308 during the pushout
    s.add('AmpAOM308', Amp_Pushout308)
    
    # Turn on microwave during pushout
    s.add('TTLQickTrig', 1)

    s.wait(t_Pushout)

    s.add('AmpSLM', Amp_SLM)
    s.add('TTLScopeTrig', 0)
    s.add('AmpAbsImag', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)
    s.add('Amp556RydbergHF', 0)

    s.add('AmpAOM308', 0)
    s.add('AmpAOM616', Amp_AOM616Divert)   # restore 616 idle
    s.add('TTLScopeTrig', 0)

    s.add('TTLQickTrig', 0)
    s.add('TTL369Shutter', 0)
    s.add('Amp369', 0)

    # Ramp the tweezer back up and wait
    s.add_step(1e-3).add('VSLMservo', ramp_to(Consts().Init.VSLMServo))

    # Turn the 556 rydberg shutter off
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL556MOTaShutter', 1)
    s.add('TTL556MOTbShutter', 1)
    s.add('TTL556MOTcShutter', 1)

    s.add('VRydCoil', 0)
    s.wait(50e-3)
