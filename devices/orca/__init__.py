"""orca -- Hamamatsu Orca-Quest qCMOS camera (pyctrl capture).

pylablib (DCAM) wrapper + the ``store_imgs`` wire-format helper. Import-safe with
no pylablib present (the DCAM import is lazy inside the open/connect path).
"""
from .orca_camera import (
    DEFAULT_ROI,
    OrcaCamera,
    open_orca_from_config,
    orca_config_defaults,
    to_store_array,
    to_store_frame_u16,
)

__all__ = ["DEFAULT_ROI", "OrcaCamera", "open_orca_from_config",
           "orca_config_defaults", "to_store_array", "to_store_frame_u16"]

# Registry: create("orca", **kw) constructs + opens an OrcaCamera handle (NEEDS-HARDWARE
# unless a fake backend is injected via cam=...). Registering here keeps the driver clean.
from ..device_registry import register

register("orca")(OrcaCamera)
