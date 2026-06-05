"""nidaq -- NI PCIe-6738 analog output.

Two paths: :mod:`devices.nidaq.nidaq_runner` (the externally-clocked SEQUENCE
runner, the one real device-driver port onto ``nidaqmx``) and
:mod:`devices.nidaq.nidaq_io_handler` (static on-demand DC set/read on a single
AO channel). Import-safe with no nidaqmx present (lazy import inside the run paths).
"""
from . import nidaq_io_handler as _io
from .nidaq_runner import NiDAQRunner
from .nidaq_io_handler import set_channel, read_channel
from ..device_registry import register

__all__ = ["NiDAQRunner", "set_channel", "read_channel"]

# Registry. NiDAQRunner is a process-global namespace (classmethods + class state), NOT
# instantiated -- so its factory returns the class itself. The static-DC path is stateless
# module functions, so "nidaq_dc" resolves to that module: create("nidaq_dc").set_channel(...).
register("nidaq_seq")(lambda *a, **k: NiDAQRunner)
register("nidaq_dc")(lambda *a, **k: _io)
