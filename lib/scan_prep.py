"""scan_prep.py -- write the scan-config the monitor's DataManager reads (JSON, not .mat).

In MATLAB, ``ybBuildScanJob`` writes a ``Scan``-struct ``.mat`` into the dated data dir; the
yb_analysis monitor's ``DataManager`` reads it (via ``scan_id_to_stamps`` -> ``make_scan_dir``)
to learn the scan metadata (frame size, NumImages, isInit/isHC/isGrid2, ...). Without it the
monitor raises "Cannot load <path>" and its ``_process_once`` dies.

pyctrl is a Python backend, so rather than emit a MATLAB ``.mat`` we dump a **JSON** sidecar
(``data_<stamp>.json``) next to where the ``.mat``/``.h5`` live, and the monitor was taught to
prefer that JSON (falling back to ``.mat`` for MATLAB-written scans). Cleaner, no scipy.

For an **isInit scan** (e.g. LACScan) ``DataManager`` reads only ``frameSize`` / ``NumImages`` /
``isInit`` / ``isHC`` / ``isGrid2`` (then creates the HDF5 + saves frames; no detection) -- so
that is what we write, plus ``NumPerGroup`` for display. Path mirrors yb_analysis
``make_scan_dir``: ``<prefix>/Data/<YYYYMMDD>/data_<stamp>/data_<stamp>.json``.

The data root (monitor ``PATH_PREFIX``) is ``$YB_DATA_PREFIX`` if set, else the lab default;
keep it in sync with ``yb_analysis.config.PATH_PREFIX``. Design inspired by the MATLAB original.
"""

import json
import os

# Mirrors yb_analysis.config.PATH_PREFIX (the monitor's data root). Override per-machine with
# $YB_DATA_PREFIX so the backend writes where the monitor reads.
DEFAULT_DATA_PREFIX = r"D:\OneDrive - Harvard University\Documents - Yb"


def scan_dir(scan_id, prefix=None):
    """The dated scan dir for ``scan_id`` (14-digit YYYYMMDDHHMMSS), mirroring make_scan_dir:
    ``<prefix>/Data/<YYYYMMDD>/data_<YYYYMMDD>_<HHMMSS>``."""
    prefix = prefix or os.environ.get("YB_DATA_PREFIX", DEFAULT_DATA_PREFIX)
    s = str(int(scan_id))
    if len(s) != 14:
        raise ValueError("scan_id must be 14 digits (YYYYMMDDHHMMSS), got %r" % (scan_id,))
    date, time = s[:8], s[8:]
    return os.path.join(prefix, "Data", date, "data_%s_%s" % (date, time))


def scan_config_path(scan_id, prefix=None):
    """The JSON scan-config path: ``<scan_dir>/data_<stamp>.json``."""
    d = scan_dir(scan_id, prefix=prefix)
    return os.path.join(d, "data_%s.json" % os.path.basename(d)[len("data_"):])


def write_scan_config(scan_id, frame_wh, num_images, *, is_init=0, is_hc=0, is_grid2=0,
                      num_per_group=0, prefix=None):
    """Write the JSON scan-config for ``scan_id``; return the path.

    Args:
        scan_id: 14-digit YYYYMMDDHHMMSS (the value frames are stamped with).
        frame_wh: ``(W, H)`` of the imaging ROI (MATLAB ``frameSize = [W, H]``; the monitor
            transposes to ``(H, W)``). ``(0, 0)`` when no camera -> monitor skips HDF5 but
            does not crash.
        num_images / is_init / is_hc / is_grid2 / num_per_group: from the descriptor runp.
    """
    path = scan_config_path(scan_id, prefix=prefix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = int(frame_wh[0]), int(frame_wh[1])
    cfg = {
        "frameSize": [w, h],                 # MATLAB [W, H]
        "NumImages": int(num_images),
        "NumPerGroup": int(num_per_group),
        "isInit": int(is_init),
        "isHC": int(is_hc),
        "isGrid2": int(is_grid2),
        "scan_id": int(scan_id),
        "source": "pyctrl",
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, path)                    # atomic publish
    return path
