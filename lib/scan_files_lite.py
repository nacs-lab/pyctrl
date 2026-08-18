"""Dependency-light mirror of ``yb_analysis/io/scan_files.py`` for pyctrl tools.

A scan directory holds either the LEGACY combined layout (one ``data_<stamp>.h5``
carrying ``/imgs`` plus logicals / intensities / seq_ids) or the SPLIT layout (a
small ``data_<stamp>.h5`` without ``/imgs`` beside a bulk ``image_<stamp>.h5``
holding ``/imgs`` + ``/seq_ids`` + ``/frame_seq_ids`` and the
``num_images_per_seq`` / ``frame_size`` attrs).

pyctrl tools run in other environments (the engine venv, a bare base python) and
must not import ``yb_analysis``, so the two helpers they actually need are
restated here. Semantics are IDENTICAL to ``yb_analysis/io/scan_files.py``
(``IMGS_PREFIX``, ``imgs_path_for``, ``image_source``) -- change the naming there
and mirror it here.

stdlib only; ``frames_per_seq`` accepts an already-open h5py handle so a caller
that has one does not need h5py imported here at all.
"""

import os

# Must match yb_analysis.io.scan_files.IMGS_PREFIX.
IMGS_PREFIX = 'image'


def _stamp_of(path):
    """``data_20260818_101500.h5`` -> ``20260818_101500`` ('' when unstamped)."""
    base = os.path.basename(str(path).rstrip('\\/'))
    base = os.path.splitext(base)[0]
    _, sep, stamp = base.partition('_')
    return stamp if sep else ''


def imgs_path_for(data_path):
    """Map ``.../data_<stamp>.h5`` -> ``.../image_<stamp>.h5`` (pure string)."""
    data_path = str(data_path)
    d, base = os.path.split(data_path)
    stamp = _stamp_of(base)
    name = '%s_%s.h5' % (IMGS_PREFIX, stamp) if stamp else '%s.h5' % IMGS_PREFIX
    return os.path.join(d, name) if d else name


def image_source(path):
    """Redirect a data-file path to its image file when one exists on disk.

    Returns ``path`` unchanged for a combined scan, a ``.mat`` file, or a path
    that already is the image file.
    """
    candidate = imgs_path_for(path)
    if candidate != str(path) and os.path.exists(candidate):
        return candidate
    return str(path)


def frames_per_seq(image_h5, default=None):
    """Frames per sequence (``num_images_per_seq``) of an image/combined h5.

    ``image_h5`` may be an open h5py handle or a path (opened read-only). Order:
    the ``num_images_per_seq`` attr, then ``len(frame_seq_ids) // len(seq_ids)``
    when both datasets are present, else ``default``. Never raises.
    """
    if isinstance(image_h5, (str, bytes, os.PathLike)):
        os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
        try:
            import h5py
        except ImportError:
            return default
        try:
            with h5py.File(image_h5, 'r') as f:
                return frames_per_seq(f, default=default)
        except Exception:
            return default

    f = image_h5
    try:
        if 'num_images_per_seq' in f.attrs:
            n = int(f.attrs['num_images_per_seq'])
            if n > 0:
                return n
    except Exception:
        pass
    try:
        if 'frame_seq_ids' in f and 'seq_ids' in f:
            n_fs = int(f['frame_seq_ids'].shape[0])
            n_sq = int(f['seq_ids'].shape[0])
            if n_sq > 0 and n_fs % n_sq == 0 and n_fs // n_sq > 0:
                return n_fs // n_sq
    except Exception:
        pass
    return default
