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

# Sentinel scan_id for FAILING-shot frames published for LIVE DISPLAY ONLY (never persisted /
# accumulated). Distinct from dummy-mode's -1 so the monitor can label these "failing" (red chip)
# rather than "dummy". Mirrored on the lab side in
# yb_analysis/gui/control_panel.py (FAILING_DISPLAY_SCAN_ID) -- keep the two in sync.
FAILING_DISPLAY_SCAN_ID = -2


# =========================================================================== #
# the process-global scan context
# =========================================================================== #
class ScanContext:
    """Live handles for the active scan's rearrangement callbacks (set by the runner)."""

    def __init__(self, *, session, camera, server, client, scan_id,
                 is_rearrange=False, n_rounds=1, pattern_name=None,
                 server_grid_knm=None, calib_root=None, log=None):
        self.session = session          # SlmScanSession (scan-long slm lock owner)
        self.camera = camera            # OrcaCamera (or None)
        self.server = server            # ExptServer (store_imgs / seq_finish / seq_cancel)
        self.client = client            # SlmClient (shared lock owner)
        self.scan_id = scan_id          # 14-digit YYYYMMDDHHMMSS (frame routing)
        self.is_rearrange = bool(is_rearrange)
        self.n_rounds = int(n_rounds)
        self.pattern_name = pattern_name  # frame-0 loading pattern (per-pattern detection)
        self.log = log or (lambda _m: None)
        # The mid-shot detector's grid source, in priority order:
        #   (1) SINGLE SOURCE OF TRUTH -- the SERVER's actual init_grid (``server_grid_knm``, the
        #       exact array ``rearrange(bits)`` scores ``bits[i]`` against), mapped to camera px
        #       through the global affine. Using it GUARANTEES the lab detects in the same site
        #       order the server scores -> ``bits[i]`` corresponds to ``init_grid[i]`` BY
        #       CONSTRUCTION, so the detection order can NEVER desync from the server (independent
        #       re-derivation with a divergent sort order -- e.g. col vs col_up -- was the prior
        #       desync hazard).
        #   (2) the per-pattern registry grid+thresholds (knm -> affine -> ROI-crop), and
        #   (3) the day-folder grid+thresholds.
        # The ROI comes from the live camera.
        roi_provider = (camera.current_roi if camera is not None else None)
        self._detector = _Detector(calib_root or _DATA_ROOT, pattern_name=pattern_name,
                                   roi_provider=roi_provider, server_grid_knm=server_grid_knm,
                                   log=self.log)

    def detect_bits(self, img):
        """Detect atoms in ``img`` -> '0'/'1' string, or '' on a calibration mismatch (so the
        caller bails without rearranging on a stale grid)."""
        return self._detector.bits(img)

    def detect_probs(self, img):
        """Detect atoms in ``img`` -> list of per-site posterior probabilities P(atom present) in
        [0,1] (same site order as :meth:`detect_bits`), or ``[]`` on a calibration mismatch. Sent to
        the SLM server's rearrange call IN PLACE OF the bitstring; a missing/degenerate per-site fit
        -> 0.0 so the server (which rounds at 0.5) drops uncertain sites."""
        return self._detector.probs(img)

    # ----------------------------------------------------------------------- #
    # shot-error reporting (feeds the dashboard's "shots failing" banner)
    # ----------------------------------------------------------------------- #
    def record_error(self, message, kind=None, seq_id=None):
        """Log ``message`` (exactly as ``self.log`` would) AND record it as a failed shot on the
        ExptServer, so the live monitor can surface "shots are failing" instead of a bare
        "Running / no data" when every rearrange shot errors. Best-effort on both halves -- a
        callback must never crash on its own error reporting."""
        try:
            self.log(message)
        except Exception:  # noqa: BLE001
            pass
        _safe_server_call(self.server, "record_shot_error", message, self.scan_id, seq_id, kind)

    def record_ok(self):
        """Mark that a rearrange shot completed without error, so the dashboard sees a recovery
        (clears the "shots failing" banner promptly rather than waiting out the staleness window).
        Best-effort -- a missing method (older/MATLAB server) is a harmless no-op."""
        _safe_server_call(self.server, "record_shot_ok")

    def publish_failed_shot(self, frames, seq_id=None):
        """Publish whatever frames a FAILING shot captured for LIVE DISPLAY ONLY.

        A failing shot used to ``cancel_shot`` (drop its staged frames) so the live view froze on
        the last good pair. Instead we re-publish the captured frame(s) under the
        :data:`FAILING_DISPLAY_SCAN_ID` sentinel so the monitor still flashes img1 (+ img2 if it was
        captured, else "no data") while the shot-health chip stays red -- WITHOUT persisting to HDF5
        or feeding the accumulators (the negative scan_id routes to the lab's show-without-persist
        path). The shot-health "failing" state is driven separately via :meth:`record_error`.

        ``frames`` is the ordered list of captured raw frames (img1 first, img2 if present); ``None``
        entries are dropped. Leading ``cancel_shot`` discards any real-scan_id frames still staged
        for this shot (FIFO-ordered before the re-stage), so a partially-staged real shot is cleanly
        converted to a display-only one. No-op if nothing was captured."""
        srv = self.server
        if srv is None:
            return
        _safe_server_call(srv, "cancel_shot")          # drop any real-scan_id staged frames first
        frames = [f for f in (frames or []) if f is not None]
        if not frames:
            return
        sid = FAILING_DISPLAY_SCAN_ID
        sq = int(seq_id) if seq_id is not None else -1
        for f in frames:
            _safe_server_call(srv, "stage_frame", f, sid, sq)
        _safe_server_call(srv, "finish_shot")


def _safe_server_call(server, method, *args):
    """Call ``server.method(*args)`` best-effort (missing hook / failure never crashes a callback)."""
    if server is None:
        return
    fn = getattr(server, method, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass


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
def grab_one_frame(camera, timeout=0.2, sleep=time.sleep, clock=time.monotonic):
    """Wait for EXACTLY one frame and return ``(img, True, 1)``; ``(None, False, n_seen)`` on a
    timeout or a stale-frame surplus. Mirrors the MATLAB ``nFrames ~= 1`` cancel-and-drain guard:
    any surplus is consumed by ``read_frames`` so it can't pollute the next shot.

    ``n_seen`` is the number of frames drained, so callers can SURFACE the two failure modes
    distinctly: ``0`` = nothing arrived within ``timeout`` (readout latency / a missed trigger),
    ``>=2`` = a surplus, i.e. the frame stream has DESYNCED (a straggler from a prior shot or a
    spurious trigger). A desync is what flips img1/img2: the one-frame-per-grab pairing in
    ``hand_over_slm`` (img1) / ``post_run`` (img2) slips by one and the loading frame lands in img2.
    """
    if camera is None:
        return None, False, 0
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
        return None, False, len(collected)
    return collected[0], True, 1


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
    """Grid + thresholds -> per-site sparse weight matrix -> '0'/'1' bits via one matvec.

    PREFERS the per-pattern registry (the scan's frame-0 pattern: knm -> global affine -> ROI-crop
    grid + ``<pattern>/threshold.mat``), the same source yb_analysis's live detection uses, so the
    rearrange bits score with PATTERN thresholds. Falls back to the DAY-FOLDER ``gridLocations.txt``
    + ``threshold.mat`` when no pattern is set or the registry/affine isn't available -- nothing is
    lost. Rebuilt when the source calibration changes (mtimes / ROI / pattern) or the image shape
    changes."""

    def __init__(self, data_root, pattern_name=None, roi_provider=None, server_grid_knm=None,
                 log=None):
        self._data_root = data_root
        self._pattern_name = pattern_name
        self._roi_provider = roi_provider
        self._server_grid_knm = server_grid_knm   # server init_grid (N,2 [y,x] knm); single source
        self._log = log or (lambda _m: None)
        self._W = None                 # scipy.sparse (M, H*W)
        self._thresholds = None        # (M,)
        self._gauss_params = None      # list[M] of (6,) [mu_e,s_e,A_e,mu_a,s_a,A_a] or None/site
        self._img_shape = None         # (H, W)
        self._key = None               # cache identity of the built calibration

    def _intensities(self, img):
        """Per-site masked intensity vector for ``img`` (one sparse matvec), or None on a
        calibration mismatch. Shared by :meth:`bits` (hard threshold) and :meth:`probs`
        (posterior) so both score the SAME intensities in the SAME server site order."""
        import numpy as np
        a = np.asarray(img, dtype=float)
        if a.ndim != 2:
            a = a.reshape(a.shape[0], -1)
        shape = a.shape
        try:
            self._ensure(shape)
        except Exception as e:  # noqa: BLE001 - missing calibration -> bail (no rearrange)
            self._log("[rearrange_runtime] detector unavailable: %s" % e)
            return None
        if self._W is None or self._img_shape != shape:
            self._log("[rearrange_runtime] image shape %s != calibrated %s; skipping"
                      % (shape, self._img_shape))
            return None
        return self._W.dot(a.ravel(order="F"))            # MATLAB W * img(:) (column-major)

    def bits(self, img):
        intensities = self._intensities(img)
        if intensities is None:
            return ""
        logicals = intensities > self._thresholds
        return "".join("1" if b else "0" for b in logicals)

    def probs(self, img):
        """Per-site posterior ``P(atom present | intensity)`` as a list of floats in [0,1], or ``[]``
        on a calibration mismatch (mirrors :meth:`bits` returning ``""`` so the caller bails). Same
        intensities + same site order as :meth:`bits`, so ``probs[i]`` lines up with the server's
        ``init_grid[i]``.

        A site with a MISSING/degenerate Gaussian fit -> 0.0 (an uncertain site is treated as empty
        -- the conservative choice; the SLM server rounds <0.5 to "no atom"). If the WHOLE
        calibration lacks Gaussian fits (older threshold.mat with no gaussFitsStruct), falls back to
        the hard ``intensity > threshold`` decision encoded as 1.0/0.0 so the scan still
        rearranges instead of sending all-zeros."""
        intensities = self._intensities(img)
        if intensities is None:
            return []
        import numpy as np
        gp = self._gauss_params
        if gp is None or all(p is None for p in gp):
            self._log("[rearrange_runtime] no Gaussian fits in calibration; probs fall back to "
                      "hard intensity>threshold (1.0/0.0)")
            logicals = intensities > self._thresholds
            return [1.0 if b else 0.0 for b in logicals]
        post = _atom_posterior(np.asarray(intensities, dtype=float), gp)
        return [float(p) for p in post]

    def _ensure(self, img_shape):
        """(Re)build the sparse weight matrix from the best available calibration source."""
        src = self._server_source() or self._pattern_source() or self._day_source()
        if src is None:
            raise RuntimeError("no server / per-pattern / day-folder calibration available")
        grid, thresholds, gauss_params, key = src
        if self._W is not None and self._img_shape == img_shape and self._key == key:
            return                          # warm + unchanged
        self._build(grid, thresholds, gauss_params, img_shape, key)

    def _server_source(self):
        """(grid [Y,X], thresholds, cache-key) from the SERVER's init_grid -- THE single source of
        truth for the rearrange bit ordering, or None (caller falls back to the registry/day grid).

        ``setup_rearrangement`` derived + sorted the grid (its ``sweep_order``, e.g. ``col_up``) and
        ``rearrange(bits)`` scores ``bits[i]`` against ``init_grid[i]``. We map that EXACT grid
        (``self._server_grid_knm``, knm [y,x]) through the global affine to camera px, so the lab
        detects in the SAME site order the server scores -> ``bits[i]`` corresponds to
        ``init_grid[i]`` BY CONSTRUCTION (no independently re-derived grid whose sort could diverge).
        Per-site thresholds come from the per-pattern registry, reordered to the server's site order
        by a position match (same physical points, possibly different sort) -- so a stale-ordered
        threshold.mat can't misalign them either."""
        if self._server_grid_knm is None or self._roi_provider is None:
            return None
        import numpy as np
        try:
            import pattern_grid
            roi = list(self._roi_provider())
            A = pattern_grid.load_affine_matrix()
            if A is None:
                return None
            sknm = np.asarray(self._server_grid_knm, dtype=float).reshape(-1, 2)   # [y, x] knm
            if sknm.shape[0] == 0:
                return None
            grid = pattern_grid._apply_affine_cropped(pattern_grid._knm_to_xy(sknm), A, roi)  # [Y,X]
        except Exception as e:  # noqa: BLE001 - affine/registry unavailable -> fall back
            self._log("[rearrange_runtime] server-grid affine map failed (%s); registry/day" % e)
            return None
        n = sknm.shape[0]
        # Thresholds: per-pattern registry values, position-matched to the server's site order.
        if not self._pattern_name:
            return None
        try:
            rec = pattern_grid.get_pattern_record(self._pattern_name)
            td = pattern_grid.load_pattern_thresholds(self._pattern_name)
            if not rec or not rec.get("knm") or td is None:
                return None
            rknm = np.asarray(rec["knm"], dtype=float).reshape(-1, 2)
            rthr = np.asarray(td["thresholds"], dtype=float).ravel()
            if rknm.shape[0] != n or rthr.shape[0] != n:
                return None                  # site-count mismatch -> let registry/day handle it
            from scipy.spatial import cKDTree
            dist, idx = cKDTree(rknm).query(sknm)   # for each server site, nearest record site
            if float(np.max(dist)) > 5.0 or len(set(idx.tolist())) != n:
                self._log("[rearrange_runtime] server-grid<->record position match failed "
                          "(max=%.2f, bijection=%s); registry/day"
                          % (float(np.max(dist)), len(set(idx.tolist())) == n))
                return None
            thr = rthr[idx]
            # Gaussian fits (for the posterior path), reordered to the server's site order by the
            # SAME position match -> gp[i] aligns with grid[i]/thr[i]. Absent -> None (probs() then
            # falls back to the hard cut).
            rgp = td.get("gauss_params")
            gp = ([rgp[int(j)] for j in idx]
                  if rgp is not None and len(rgp) == n else None)
        except Exception as e:  # noqa: BLE001
            self._log("[rearrange_runtime] server-grid threshold match failed (%s); registry/day" % e)
            return None
        thr_mtime = _mtime(pattern_grid._pattern_threshold_path(self._pattern_name))
        key = ("server", self._pattern_name, tuple(float(v) for v in roi[:4]), n, thr_mtime)
        return grid, thr, gp, key

    def _pattern_source(self):
        """(grid [Y,X], thresholds, cache-key) from the per-pattern registry, or None."""
        if not self._pattern_name or self._roi_provider is None:
            return None
        try:
            roi = list(self._roi_provider())
            import pattern_grid
            pc = pattern_grid.resolve_pattern_calibration(self._pattern_name, roi)
        except Exception as e:  # noqa: BLE001 - registry/affine unavailable -> day-folder
            self._log("[rearrange_runtime] pattern calibration unavailable (%s); day folder" % e)
            return None
        if pc is None:
            return None
        import numpy as np
        grid = np.asarray(pc["grid"], dtype=float).reshape(-1, 2)
        thr = np.asarray(pc["thresholds"], dtype=float).ravel()
        # Gaussian fits (for the posterior path), already aligned to the grid by
        # resolve_pattern_calibration (record.json knm + threshold.mat share the registry order).
        gp = pc.get("gauss_params")
        if gp is not None and len(gp) != int(grid.shape[0]):
            gp = None
        # Key on the per-pattern threshold.mat mtime so the detector REBUILDS when the live
        # monitor refits + re-saves the pattern thresholds mid-scan (the day-folder source keys
        # on mtime too, see _day_source). Without it the pattern thresholds would be frozen for
        # the whole backend session -- the rearrange bits would never pick up a refit.
        thr_mtime = _mtime(pattern_grid._pattern_threshold_path(self._pattern_name))
        key = ("pattern", self._pattern_name, tuple(float(v) for v in roi[:4]),
               int(grid.shape[0]), thr_mtime)
        return grid, thr, gp, key

    def _day_source(self):
        """(grid [Y,X], thresholds, cache-key) from the day folder, or None."""
        folder = self._today_folder()
        grid_file = os.path.join(folder, "gridLocations.txt")
        thr_file = os.path.join(folder, "threshold.mat")
        if not (os.path.isfile(grid_file) and os.path.isfile(thr_file)):
            return None
        grid = _read_grid_locations(grid_file)            # (M, 2) [Y, X], 1-based pixel coords
        thresholds = _read_thresholds(thr_file)           # (M,)
        if grid.shape[0] != thresholds.shape[0]:
            raise ValueError("gridLocations has %d sites but thresholds has %d"
                             % (grid.shape[0], thresholds.shape[0]))
        # Gaussian fits (for the posterior path); the day-folder grid + threshold.mat share the
        # same site order, so gp[i] aligns with grid[i]/thresholds[i]. None -> probs() falls back
        # to the hard cut. Read via pattern_grid (pure pyctrl .mat reader); failure -> None.
        gp = None
        try:
            import pattern_grid
            gp = pattern_grid.read_gauss_params(thr_file)
            if gp is not None and len(gp) != int(grid.shape[0]):
                gp = None
        except Exception:  # noqa: BLE001 - gauss params optional; hard-cut fallback covers it
            gp = None
        key = ("day", folder, _mtime(grid_file), _mtime(thr_file), int(grid.shape[0]))
        return grid, thresholds, gp, key

    def _build(self, grid, thresholds, gauss_params, img_shape, key):
        from scipy import sparse
        m = grid.shape[0]
        if thresholds.shape[0] != m:
            raise ValueError("grid has %d sites but thresholds has %d"
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
        # gauss_params: list[M] of (6,) [mu_e,s_e,A_e,mu_a,s_a,A_a] (or None/site), or None for the
        # whole calibration when no gaussFitsStruct was available (probs() falls back to hard cut).
        if gauss_params is not None and len(gauss_params) != m:
            self._log("[rearrange_runtime] gauss_params len %d != M %d; dropping (hard-cut probs)"
                      % (len(gauss_params), m))
            gauss_params = None
        self._gauss_params = gauss_params
        self._img_shape = (H, W)
        self._key = key
        self._log("[rearrange_runtime] detector built: source=%s M=%d imgSize=[%d %d] gaussFits=%s"
                  % (key[0], m, H, W, "yes" if gauss_params is not None else "no"))

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


def _gauss_pdf(x, mu, sigma):
    """Scalar/array Gaussian pdf N(x | mu, sigma) (sigma > 0)."""
    import numpy as np
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _atom_posterior(intensities, params_list):
    """Vectorised per-site posterior ``P(atom present | intensity)`` under the per-site two-Gaussian
    mixture ``params = [mu_e, s_e, A_e, mu_a, s_a, A_a]`` (empty peak first; fitted areas A_e/A_a are
    the mixing weights, so the posterior folds in the site's loading rate).

    Mirrors ``yb_analysis/detection/dynamical_threshold.py:atom_posterior`` -- reimplemented locally
    (numpy only) so the engine-venv backend runtime takes NO yb_analysis import dependency. A site
    whose params are missing (None) or degenerate (s_e<=0 / s_a<=0 / vanishing mixture density) ->
    0.0 (uncertain -> treated as empty, the conservative choice for rearrangement)."""
    import numpy as np
    x = np.asarray(intensities, dtype=float).ravel()
    out = np.zeros(x.shape[0], dtype=float)
    for i, params in enumerate(params_list):
        if params is None:
            continue
        p = np.asarray(params, dtype=float).ravel()
        if p.size < 6:
            continue
        mu_e, s_e, A_e, mu_a, s_a, A_a = p[0], p[1], p[2], p[3], p[4], p[5]
        if not (s_e > 0 and s_a > 0):
            continue
        pe = A_e * _gauss_pdf(x[i], mu_e, s_e)
        pa = A_a * _gauss_pdf(x[i], mu_a, s_a)
        denom = pe + pa
        if denom > 0:
            out[i] = min(1.0, max(0.0, float(pa / denom)))
        # denom == 0 (both peaks vanish at this intensity) -> leave 0.0
    return out


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
