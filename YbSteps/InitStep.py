"""InitStep.py -- transliteration of ``matlab_new/YbSteps/InitStep.m``.

Step signature is ``step(s, g)`` (MATLAB ``function InitStep(s, g)``): ``s`` is the
(sub-)sequence, ``g`` is the config sub-tree the parent seq passes in -- here
``g = s.C.Init``. Reads resolve config with a ``Consts()`` fallback default
(``g.X.Y(Consts().Init.X.Y)``); ``Consts()`` default args are SubProps that DynProps
eagerly resolves to a number (W2 fix #1).

Turn on 2D MOT & Zeeman slower (always kept on); set initial shutters; set most AOM
amplitudes and analog voltages to 0.

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by ``_resolve_pulse``,
so bare ints (``0``/``1``/``200e6``) serialize as ARG_CONST_FLOAT64 -- faithful to
MATLAB's doubles. (Only literals that become SeqVal operands need explicit floats.)
"""

from consts import Consts


def InitStep(s, g):
    Freq_Resonance399Freq = g.Resonance399Freq(Consts().Resonance399Freq)

    Freq_2DMOTDetuning = g.TwoDMOT.FreqDetuning(Consts().Init.TwoDMOT.FreqDetuning)
    Freq_2DMOT = Freq_Resonance399Freq + Freq_2DMOTDetuning
    Amp_2DMOT = g.TwoDMOT.Amp(Consts().Init.TwoDMOT.Amp)

    Freq_ZeemanDetuning = g.Zeeman.FreqDetuning(Consts().Init.Zeeman.FreqDetuning)
    Freq_Zeeman = Freq_Resonance399Freq + Freq_ZeemanDetuning
    Amp_Zeeman = g.Zeeman.Amp(Consts().Init.Zeeman.Amp)

    Vx = g.Electrodes.Vx(Consts().Init.Electrodes.Vx)
    Vy = g.Electrodes.Vy(Consts().Init.Electrodes.Vy)
    Vz = g.Electrodes.Vz(Consts().Init.Electrodes.Vz)
    VSLMservo = g.VSLMServo(Consts().Init.VSLMServo)

    # s.wait(10e-6)
    (s.add_step(10e-6)
        .add('Freq2DMOT', Freq_2DMOT).add('Amp2DMOT', Amp_2DMOT)
        .add('FreqZeeman', Freq_Zeeman).add('AmpZeeman', Amp_Zeeman)
        .add('FreqAOM308', 200e6).add('AmpAOM308', 0)
        .add('FreqAOM616', 120e6).add('AmpAOM616', 0.11)
        .add('FreqAODv', 100e6)
        .add('Amp556MOTX', 0)
        .add('Amp556RydbergMOTh', 0)
        .add('AmpSLM', 0.55)
        .add('Amp369', 0)
        .add('AmpAbsImag', 0)
        .add('VSLMservo', VSLMservo)
        .add('AmpAODs', 0)
        .add('VMOTCoil', 0)
        .add('VRydCoil', 0)
        .add('VBiasCoilX', 0)
        .add('VBiasCoilY', 0)
        .add('VBiasCoilZ', 0)
        .add('TTLThorCamTrig', 0)
        .add('TTLOrcaTrig', 0)
        .add('TTL556RydbergShutter', 0)
        .add('TTL556MOTaShutter', 1)
        .add('TTL556MOTbShutter', 1)
        .add('TTL556MOTcShutter', 1)
        .add('TTL399AbsImagShutter', 0)
        .add('TTL369Shutter', 0)
        .add('TTL399MOTShutter', 1)
        .add('TTL3992DMOTShutter', 1)
        .add('TTL369Switch', 0)
        .add('TTL556RydAWGSwitch', 0)
        .add('TTL308RydAWGSwitch', 0)
        .add('TTLSampleAndHold', 1))

    (s.add_step(1e-3)
        .add('VElectrode1', +Vx + Vy - Vz)
        .add('VElectrode2', 0 + Vy - Vz)
        .add('VElectrode3', 0 + Vy + Vz)
        .add('VElectrode4', -Vx + Vy + Vz)
        .add('VElectrode5', 0 - Vy - Vz)
        .add('VElectrode6', -Vx - Vy - Vz)
        .add('VElectrode7', +Vx - Vy + Vz)
        .add('VElectrode8', 0 - Vy + Vz))
