"""rearrange_runtime.py -- process-global bridge + atom detector for the rearrangement scan.

MATLAB's rearrangement callbacks reach the camera (``vid``) and the experiment server through the
base workspace (``evalin('base','vid')`` / ``ExptServer.get(...)``). pyctrl seq callbacks only get
the seq object ``s1`` -- they have no handle to the camera/server/scan-session that the run loop
owns. This module is the pyctrl analog of that base workspace: a tiny process-global
:class:`ScanContext` the runner populates at scan start (camera, ExptServer, the
:class:`SlmScanSession`, scan_id, the SLM client) and the seq callbacks read.

It also hosts the atom detector ported from ``RearrangeCommSeq.m::slm_detect_init`` /
``detect_bits``: it loads the day-folder calibration (``gridLocations.txt`` + ``threshold.mat``),
builds the per-site sparse weight matrix once, and turns a camera frame into a '0'/'1' bitstring
with a single sparse matvec + threshold compare -- the same machinery, reloaded when yb_analysis
rewrites the calibration on disk.

Finally it carries the SLM-kwarg helpers (``collect_kwargs`` / ``translate_zernike_zN``, ports of
the MATLAB seq helpers) and the pause/resume hooks the control channel calls so a pause actively
drops the scan-long ``slm`` lock and a resume reacquires + rewrites it.

NO byte-path impact: nothing here is serialized. The detector's scipy/numpy + ``.mat`` reads are
only exercised on the live (NEEDS-HARDWARE) rearrangement path; importing the module is cheap.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import os
import time

# Day-folder calibration root (mirrors RearrangeCommSeq.m). Overridable for portability/tests.
_DATA_ROOT = os.environ.get(
    "YB_DATA_ROOT",
    r"D:\OneDrive - Harvard University\Documents - Yb\Data")

# Detection mask defaults (mirror arrayConfig / slm_detect_init).
_BOX = 9
_SIGMA = 2


# =========================================================================== #
# the process-global scan context
# =========================================================================== #
class ScanContext:
    """Live handles for the active scan's rearrangement callbacks (set by the runner)."""

    def __init__(self, *, session, camera, server, client, scan_id,
                 is_rearrange=False, n_rounds=1, calib_root=None, log=None):
        self.session = session          # SlmScanSession (scan-long slm lock owner)
        self.camera = camera            # OrcaCamera (or None)
        self.server = server            # ExptServer (store_imgs / seq_finish / seq_cancel)
        self.client = client            # SlmClient (shared lock owner)
        self.scan_id = scan_id          # 14-digit YYYYMMDDHHMMSS (frame routing)
        self.is_rearrange = bool(is_rearrange)
        self.n_rounds = int(n_rounds)
        self.log = log or (lambda _m: None)
        self._detector = _Detector(calib_root or _DATA_ROOT, log=self.log)

    def detect_bits(self, img):
        """Detect atoms in ``img`` -> '0'/'1' string, or '' on a calibration mismatch (so the
        caller bails without rearranging on a stale grid)."""
        return self._detector.bits(img)


_CTX = None   # the active ScanContext, or None when no rearrangement-capable scan is running


def set_context(ctx):
    """Install the active scan context (runner, at scan start)."""
    global _CTX
    _CTX = ctx


def clear_context():
    """Remove the active scan context (runner finally, at scan end)."""
    global _CTX
    _CTX = None


def context():
    """The active :class:`ScanContext`, or None."""
    return _CTX


# =========================================================================== #
# pause / resume hooks (called by control_channel.py)
# =========================================================================== #
def on_pause():
    """Active drop of the scan-long slm lock on pause. No-op when no session is active."""
    ctx = _CTX
    if ctx is None or ctx.session is None:
        return
    try:
        ctx.session.on_pause()
    except Exception as e:  # noqa: BLE001 - the pause gate must never crash on this
        ctx.log("[rearrange_runtime] on_pause failed: %s" % e)


def on_resume():
    """Reacquire + rewrite the loading phase on resume. No-op when no session is active.
    Best-effort: the next shot's ``ensure_held`` enforces the mandatory regrab."""
    ctx = _CTX
    if ctx is None or ctx.session is None:
        return
    try:
        ctx.session.on_resume()
    except Exception as e:  # noqa: BLE001
        ctx.log("[rearrange_runtime] on_resume failed: %s" % e)


# =========================================================================== #
# camera frame grab (port of grab_one_frame / the nFrames==1 guard)
# =========================================================================== #
def grab_one_frame(camera, timeout=0.1, sleep=time.sleep, clock=time.monotonic):
    """Wait for EXACTLY one frame and return ``(img, True)``; ``(None, False)`` on timeout or a
    stale-frame surplus (>1). Mirrors the MATLAB ``nFrames ~= 1`` cancel-and-drain guard: any
    surplus is consumed by ``read_frames`` so it can't pollute the next shot."""
    if camera is None:
        return None, False
    collected = []
    deadline = clock() + float(timeout)
    while clock() < deadline:
        try:
            frames = camera.read_frames()
        except Exception:  # noqa: BLE001 - a read error is treated as "no frame"
            frames = []
        if frames:
            collected.extend(frames)
            break
        sleep(0.001)
    if len(collected) != 1:
        return None, False
    return collected[0], True


# =========================================================================== #
# SLM-kwarg helpers (ports of the MATLAB seq helpers)
# =========================================================================== #
def collect_kwargs(subprops):
    """Convert a ``rearrange_kwargs`` SubProps / dict into a setup_rearrangement kwargs dict.

    Port of ``collect_kwargs.m``: top-level leaves become kwargs; a nested ``extras`` namespace is
    bundled into ``kwargs['extras']`` (the escape hatch for non-signature server kwargs); any
    OTHER nested namespace is skipped (the server only takes top-level scalars)."""
    d = _as_plain_dict(subprops)
    if not isinstance(d, dict):
        return {}
    kwargs = {}
    extras = {}
    for name, v in d.items():
        if name == "extras":
            if isinstance(v, dict):
                extras.update(v)
            continue
        if isinstance(v, dict):
            continue                      # nested namespace -> skip (use extras.X)
        kwargs[name] = v
    if extras:
        kwargs["extras"] = extras
    return kwargs


def translate_zernike_zN(kwargs):
    """Fold ``extras.z<N>`` scalars (ANSI index N) into ``extras.zernike_coeffs`` (port of the
    MATLAB ``translate_zernike_zN``). Lets a scan declare per-coefficient sweeps
    (``extras.z4 = -4``) while talking the server's bundled-vector contract. An explicit
    ``zernike_coeffs`` wins (no overwrite)."""
    extras = kwargs.get("extras")
    if not isinstance(extras, dict):
        return kwargs
    zn = {}
    for key in list(extras.keys()):
        if len(key) >= 2 and key[0] == "z" and key[1:].isdigit():
            zn[int(key[1:])] = float(extras.pop(key))
    if not zn:
        return kwargs
    coeffs = [0.0] * (max(zn) + 1)
    for idx, val in zn.items():
        coeffs[idx] = val
    extras.setdefault("zernike_coeffs", coeffs)
    return kwargs


def _as_plain_dict(subprops):
    """Resolve a SubProps/DynProps subtree (or accept a plain dict) to a nested dict; {} on miss."""
    if isinstance(subprops, dict):
        return subprops
    to_struct = getattr(subprops, "to_struct", None)
    if to_struct is not None:
        try:
            v = to_struct()
            return v if isinstance(v, dict) else {}
        except Exception:  # noqa: BLE001 - path absent -> nothing to forward
            return {}
    return {}


# =========================================================================== #
# atom detector (port of slm_detect_init + detect_bits)
# =========================================================================== #
class _Detector:
    """Day-folder calibration -> per-site sparse weight matrix -> '0'/'1' bits via one matvec.

    Rebuilt when ``gridLocations.txt`` / ``threshold.mat`` mtimes change (yb_analysis drift
    correction / threshold refit) or when the image shape changes."""

    def __init__(self, data_root, log=None):
        self._data_root = data_root
        self._log = log or (lambda _m: None)
        self._W = None                 # scipy.sparse (M, H*W)
        self._thresholds = None        # (M,)
        self._img_shape = None         # (H, W)
        self._calib_folder = None
        self._grid_mtime = -1.0
        self._thr_mtime = -1.0

    def bits(self, img):
        import numpy as np
        a = np.asarray(img, dtype=float)
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        shape = a.shape
        try:
            self._ensure(shape)
        except Exception as e:  # noqa: BLE001 - missing calibration -> bail (no rearrange)
            self._log("[rearrange_runtime] detector unavailable: %s" % e)
            return ""
        if self._W is None or self._img_shape != shape:
            self._log("[rearrange_runtime] image shape %s != calibrated %s; skipping"
                      % (shape, self._img_shape))
            return ""
        intensities = self._W.dot(a.ravel(order="F"))     # MATLAB W * img(:) (column-major)
        logicals = intensities > self._thresholds
        return "".join("1" if b else "0" for b in logicals)

    def _ensure(self, img_shape):
        folder = self._today_folder()
        grid_file = os.path.join(folder, "gridLocations.txt")
        thr_file = os.path.join(folder, "threshold.mat")
        grid_mt = _mtime(grid_file)
        thr_mt = _mtime(thr_file)
        if (self._W is not None and self._img_shape == img_shape
                and self._calib_folder == folder
                and grid_mt <= self._grid_mtime and thr_mt <= self._thr_mtime):
            return                          # warm + unchanged
        self._build(folder, grid_file, thr_file, img_shape)
        self._calib_folder = folder
        self._grid_mtime = grid_mt
        self._thr_mtime = thr_mt

    def _build(self, folder, grid_file, thr_file, img_shape):
        from scipy import sparse
        grid = _read_grid_locations(grid_file)            # (M, 2) [Y, X], 1-based pixel coords
        thresholds = _read_thresholds(thr_file)           # (M,)
        m = grid.shape[0]
        if thresholds.shape[0] != m:
            raise ValueError("gridLocations has %d sites but thresholds has %d"
                             % (m, thresholds.shape[0]))
        H, W = int(img_shape[0]), int(img_shape[1])
        mask = _fspecial_gaussian(_BOX, _SIGMA)
        half = _BOX // 2
        rows, cols, vals = [], [], []
        for i in range(m):
            y0 = int(round(grid[i, 0]))
            x0 = int(round(grid[i, 1]))
            y_min = max(y0 - half, 1)                       # 1-based, clipped to the image
            y_max = min(y0 + half, H)
            x_min = max(x0 - half, 1)
            x_max = min(x0 + half, W)
            if y_min > y_max or x_min > x_max:
                continue                                   # site entirely off-image
            my0 = y_min - (y0 - half)                       # 0-based offset into the mask
            mx0 = x_min - (x0 - half)
            for dy in range(y_max - y_min + 1):
                yy = y_min + dy                             # 1-based pixel row
                for dx in range(x_max - x_min + 1):
                    xx = x_min + dx                         # 1-based pixel col
                    lin = (xx - 1) * H + (yy - 1)           # column-major linear (0-based)
                    rows.append(i)
                    cols.append(lin)
                    vals.append(mask[my0 + dy, mx0 + dx])
        self._W = sparse.csr_matrix((vals, (rows, cols)), shape=(m, H * W))
        self._thresholds = thresholds.astype(float)
        self._img_shape = (H, W)
        self._log("[rearrange_runtime] detector built: %s M=%d imgSize=[%d %d]"
                  % (folder, m, H, W))

    def _today_folder(self):
        return os.path.join(self._data_root, time.strftime("%Y%m%d"))


def _read_grid_locations(path):
    """Read the tab-delimited ``gridLocations.txt`` (header ``Y\\tX``) -> (M, 2) [Y, X]."""
    import numpy as np
    rows = []
    with open(path, "r") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", "\t").split("\t")
            if ln == 0 and not _is_number(parts[0]):
                continue                                   # header row
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def _read_thresholds(path):
    """Read ``thresholds`` from a MATLAB ``threshold.mat`` (v7 via scipy, v7.3 via h5py) -> (M,)."""
    import numpy as np
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        return np.asarray(d["thresholds"], dtype=float).ravel()
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            return np.asarray(f["thresholds"], dtype=float).ravel()


def _fspecial_gaussian(n, sigma):
    """MATLAB ``fspecial('gaussian', n, sigma)``: normalised centred 2-D Gaussian (n x n)."""
    import numpy as np
    siz = (n - 1) / 2.0
    ax = np.arange(-siz, siz + 1)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
