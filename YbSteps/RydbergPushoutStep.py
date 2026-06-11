"""RydbergPushoutStep.py -- transliteration of ``matlab_new/YbSteps/RydbergPushoutStep.m``.

``g = s.C.Pushout``. Rydberg / high-field variant of the push-out step. Applies the Ryd bias
coil field (current->voltage), opens the 369 + 556-Rydberg shutters (closing the 556 MOTa
shutter), lowers the SLM trap depth, fires the 556 *Rydberg* (h) beam + 308 + the QICK
microwave trigger for the push-out time, then restores trap depth / shutters and zeroes the coil.

Differs from ``PushoutStep`` (which pushes with the 556 MOT beams + 399 at the standing trap
depth and applies NO field): this step applies a field (default 5 A on ``BiasCoilCurrent.Ryd``),
pushes with the Rydberg 556 beam only (not MOTX/399), lowers the trap (``VSLMservo`` ramp to 0.4),
and pulses 308 + the QICK microwave (``TTLQickTrig``). Used for high-field / Rydberg push-out
spectroscopy.

Reads resolve config with a ``Consts()`` fallback default (``g.X.Y(Consts().Pushout...)``);
``Consts()`` default args are SubProps that DynProps resolves to a number. As in ``PushoutStep``,
some reads are computed but unused (``Freq/Amp_Pushout399``); kept verbatim from the .m -- a
fallback read adds no pulse, so it has no byte effect.

Deviations from the .m (pyctrl-only, per the porting decision):
  * the 369 amp default ``g.Amp369(0)`` reads the named const ``Consts().Pushout.Ionization.Amp``
    (0 by default) instead of a bare literal 0;
  * the post-pushout 616 idle write ``add('AmpAOM616', 0.11)`` reads the named const
    ``Consts().AOM616Divert.Amp`` (0.11) and the .m's DUPLICATE ``add('AmpAOM616', 0.11)`` (it
    appears twice at the same time point) is collapsed to a single write.
These make pyctrl's bytes differ from the (unchanged) MATLAB twin, but no seq is byte-blessed
against ``RydbergPushoutStep`` yet, so no oracle is affected.

Byte note: all arithmetic is on concrete config floats (no globals/measures), so bare int
literals (``5``/``100``/``0``/``1``) stay concrete Python floats -- faithful to MATLAB's doubles;
no explicit-float coercion is needed.
"""

from consts import Consts
from ramp_to import ramp_to


def RydbergPushoutStep(s, g):
    t_Pushout = g.Time(Consts().Pushout.Time)

    Freq_Pushout399 = g.Blue.Freq(Consts().Pushout.Blue.Freq)
    Amp_Pushout399 = g.Blue.Amp(Consts().Pushout.Blue.Amp)
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

    # Turn on the 556 rydberg shutter, turn off the 556 MOT shutter
    s.add('TTL556RydbergShutter', 0)
    s.add('TTL556MOTaShutter', 1)
    s.add('TTL556MOTbShutter', 0)
    s.add('TTL556MOTcShutter', 0)
    
    s.wait(50e-3)

    s.add_step(1e-3).add('VSLMservo', ramp_to(0.4))  # 0.5 for STIRAP

    s.add('TTLScopeTrig', 1)

    # Now we switch to the Rydberg beam
    s.add('Freq556RydbergMOTh', Freq_Pushout556).add('Amp556RydbergMOTh', Amp_Pushout556)
    #s.add('Freq556MOTX', Freq_Pushout556).add('Amp556MOTX', Amp_Pushout556)

    s.add('AmpAOM308', Amp_Pushout308)
    s.add('AmpAOM616', 0)

    # Turn on microwave during pushout
    s.add('TTLQickTrig', 1)

    s.wait(t_Pushout)

    s.add('AmpSLM', Amp_SLM)
    s.add('TTLScopeTrig', 0)
    s.add('AmpAbsImag', 0)
    s.add('AmpBlueMOT', 0)

    s.add('Amp556MOTX', 0)
    s.add('Amp556RydbergMOTh', 0)

    s.add('AmpAOM308', 0)
    s.add('AmpAOM616', Amp_AOM616Divert)   # restore 616 idle (.m had this twice -- deduped)
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
