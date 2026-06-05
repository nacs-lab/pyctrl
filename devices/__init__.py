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

try:
    from . import slm
    from .slm import (SlmClient, SlmHTTPError, get_client,
                      SlmScanSession, SlmLockUnavailable)
    __all__ += ["slm", "SlmClient", "SlmHTTPError", "get_client",
                "SlmScanSession", "SlmLockUnavailable"]
except Exception as e:
    _log.warning("devices: 'slm' family unavailable: %s", e)

try:
    from . import sigilent_awg
    from .sigilent_awg import (AWGConnection, AWGManager, WAVEFORM_FIELDS,
                               gaussian_pulse_waveform)
    __all__ += ["sigilent_awg", "AWGConnection", "AWGManager", "WAVEFORM_FIELDS",
                "gaussian_pulse_waveform"]
except Exception as e:
    _log.warning("devices: 'sigilent_awg' family unavailable: %s", e)

try:
    from . import qick_awg
    from .qick_awg import (FPGAAWGClient, FPGAABSClient, FPGAAWGManager, QickProgram,
                           simple_pulse_cfg, simple_prog_cfg, compile_chn, loop)
    __all__ += ["qick_awg", "FPGAAWGClient", "FPGAABSClient", "FPGAAWGManager", "QickProgram",
                "simple_pulse_cfg", "simple_prog_cfg", "compile_chn", "loop"]
except Exception as e:
    _log.warning("devices: 'qick_awg' family unavailable: %s", e)
