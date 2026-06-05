"""pyctrl hardware backends, grouped by device family.

Each family lives in its own subpackage (``devices/orca/``, ``devices/nidaq/``, ...).
To add hardware: drop a new ``devices/<family>/`` package, re-export its public API
from that family's ``__init__`` and add a guarded import block below.

Families are import-GUARDED: a broken family is logged and skipped so the rest stay
usable and ``import devices`` never hard-fails. They are also import-safe with NO
vendor package present -- ``pylablib``/``nidaqmx`` are imported lazily inside the run
paths, so importing this package touches no hardware.
"""
import logging

from .device_base import Device
from .device_registry import register, create, available

_log = logging.getLogger(__name__)
__all__ = ["Device", "register", "create", "available"]

try:
    from . import orca
    from .orca import (DEFAULT_ROI, OrcaCamera, open_orca_from_config,
                       orca_config_defaults, to_store_array)
    __all__ += ["orca", "DEFAULT_ROI", "OrcaCamera", "open_orca_from_config",
                "orca_config_defaults", "to_store_array"]
except Exception as e:                       # one bad family must not break the rest
    _log.warning("devices: 'orca' family unavailable: %s", e)

try:
    from . import nidaq
    from .nidaq import NiDAQRunner, set_channel, read_channel
    __all__ += ["nidaq", "NiDAQRunner", "set_channel", "read_channel"]
except Exception as e:
    _log.warning("devices: 'nidaq' family unavailable: %s", e)
