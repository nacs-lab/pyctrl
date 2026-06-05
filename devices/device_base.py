"""Forward-looking base contract for pyctrl hardware backends.

This formalizes the connect/close + injectable-seam pattern the existing drivers
already follow by convention, so future devices have a common shape. The current
drivers (OrcaCamera, NiDAQRunner, the NI DC handler) are NOT retrofitted onto this
yet -- it is scaffolding for new hardware, adopted incrementally.
"""
from abc import ABC, abstractmethod


class Device(ABC):
    """A hardware backend.

    Import-safe with no vendor package present; the vendor import + handle open
    happen lazily in :meth:`connect` (NEEDS-HARDWARE), never at import time.
    """

    name = ""        # registry key, e.g. "orca", "nidaq_seq"

    @abstractmethod
    def connect(self):
        """Lazily import the vendor package and open the device handle."""

    @abstractmethod
    def close(self):
        """Release the device handle."""
