"""nidaq_io_handler.py -- static DC output on a single NI AO channel (no sequence / no FPGA).

pyctrl mirror of the matlab_new static-DC "set channel" path
(``NIUSBDAQ.setV`` / ``NIDAQReadWriteLib.dcoutNow``, called via
``NIUSBDAQWrapper.setV`` / ``NIDAQIOHandler.aoVoltage``): drive ONE analog-output channel to
a DC voltage immediately, on the card's own on-demand timing -- NO external FPGA clock or
trigger. This is distinct from :mod:`nidaq_runner` (the SEQUENCE NI path, externally clocked
+ triggered by the FPGA): use this to park / set a single channel by hand, not to clock out a
sequence waveform.

NEEDS-HARDWARE: drives a real DAC. The output PERSISTS after the task closes (NI AO holds its
last written value), so the value stays until it is re-set or the device is reset -- exactly
how ``setV`` parks the B-field-comp DAQ. (The MATLAB ``setV`` hardcoded ``Dev1/ao0`` ignoring
its ``channelName`` arg -- a bug; this port honors the passed channel.)
"""


def set_channel(channel, voltage):
    """Set a single AO channel to ``voltage`` volts (on-demand, immediate, holds after close).

    Args:
        channel: channel in the codebase convention ``"Dev1/7"`` (device/number, NO ``ao`` --
            matches expConfig ``channelAlias`` and the engine's ``{dev, chn}``). The nidaqmx
            ``aoN`` physical form is derived internally; an explicit ``"Dev1/ao7"`` also works.
        voltage: DC volts (within the channel's range; the PCIe-6738 is +-10 V).

    Returns the voltage written. NEEDS-HARDWARE (lazy ``nidaqmx`` import).
    """
    import nidaqmx
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(_phys(channel))
        # On-demand single-sample write (no sample clock) -> outputs immediately; the DAC
        # holds the value after the task closes. Mirrors NIUSBDAQ.setV's task.write(V).
        task.write(float(voltage), auto_start=True)
    return voltage


def read_channel(channel, samples=1):
    """Read back an AO channel's ACTUAL output voltage via the card's internal monitor.

    The PCIe-6738 has no external analog input, but exposes an internal AI loopback per AO
    channel (``Dev1/_aoN_vs_aognd``) -- so an AO output can be verified electronically with no
    external meter. Averages ``samples`` reads. NEEDS-HARDWARE.

    Args:
        channel: the AO channel in ``"Dev1/7"`` convention (or ``"Dev1/ao7"``), or its
            ``_aoN_vs_aognd`` monitor name directly.
    """
    import nidaqmx
    mon = channel if "_vs_aognd" in channel else _ao_monitor_name(channel)
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(mon)
        if samples > 1:
            vals = task.read(number_of_samples_per_channel=samples)
            return sum(vals) / len(vals)
        return float(task.read())


def _phys(channel):
    """``"Dev1/7"`` -> ``"Dev1/ao7"`` (nidaqmx physical AO name). Idempotent if ``ao`` present."""
    dev, _, chn = channel.rpartition("/")
    return channel if chn.startswith("ao") else "%s/ao%s" % (dev, chn)


def _ao_monitor_name(channel):
    """``"Dev1/7"`` (or ``"Dev1/ao7"``) -> ``"Dev1/_ao7_vs_aognd"`` (internal AO-monitor AI)."""
    dev, _, chn = channel.rpartition("/")
    num = chn[2:] if chn.startswith("ao") else chn
    return "%s/_ao%s_vs_aognd" % (dev, num)
