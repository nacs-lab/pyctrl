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

import shot_time                      # per-shot wall clock (TEMPORARY: RP-N correlation)

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

# Flag file that turns on the per-frame common-mode probs normalization (same effect as env
# YB_NORM_PROBS=1) WITHOUT a backend restart: the env can't be injected into the long-lived
# backend process, the flag file can. Checked per probs() call (a stat(); negligible).
# DEFAULT OFF: normalization runs ONLY while this file exists (Dev campaign 2026-07-19).
_NORM_PROBS_FLAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tmp", "yb_norm_probs.flag")


def _norm_probs_flag():
    try:
        return os.path.isfile(_NORM_PROBS_FLAG)
    except Exception:  # noqa: BLE001
        return False


# =========================================================================== #
# the process-global scan context
# =========================================================================== #
class ScanContext:
    """Live handles for the active scan's rearrangement callbacks (set by the runner)."""

    def __init__(self, *, session, camera, server, client, scan_id,
                 is_rearrange=False, n_rounds=1, pattern_name=None,
                 server_grid_knm=None, calib_root=None, log=None,
                 frame_patterns=None, loading_defocus=None):
        self.session = session          # SlmScanSession (scan-long slm lock owner)
        self.camera = camera            # OrcaCamera (or None)
        self.server = server            # ExptServer (store_imgs / seq_finish / seq_cancel)
        self.client = client            # SlmClient (shared lock owner)
        self.scan_id = scan_id          # 14-digit YYYYMMDDHHMMSS (frame routing)
        self.is_rearrange = bool(is_rearrange)
        self.n_rounds = int(n_rounds)
        self.pattern_name = pattern_name  # frame-0 loading pattern (per-pattern detection)
        # Per-camera-frame pattern names (loading, [middle...], final) for the multi-round scan
        # (RearrangeCommSeq2). Each frame is detected with its OWN per-pattern registry grid +
        # thresholds (independent site counts / orderings), NOT the single server grid -- the
        # caller (the seq callback) is responsible for keeping the site count in agreement with
        # what the SLM server scores that round (a mismatch is surfaced, never silently off-by-one).
        # None / single-round -> the single-detector path below is used unchanged (ground truth).
        self.frame_patterns = list(frame_patterns) if frame_patterns else None
        self.log = log or (lambda _m: None)
        # ---- probability-path diagnostics (the SLM server's prob_hungarian) --------------
        # Whether THIS scan asked the server for the imaging-weighted Hungarian, and the beta
        # it asked for (None -> the server's survival-calibrated (nsteps*1.5)^2 default). Set
        # by rearrange_callbacks.pre_run from the per-shot setup extras; purely diagnostic on
        # this side (the flag itself travels to the server inside setup_rearrangement).
        self.prob_hungarian = False
        self.prob_hungarian_beta = None
        # Last per-round probability summary, keyed by the round's callback tag; see
        # :meth:`note_probs`. Readable post-hoc from the runner log, and live from the context.
        self.prob_diag = {}
        self._prob_shots = {}           # tag -> shots seen (log throttle)
        self._site_counts = {}          # pattern name -> registry site count (cached)
        # Cache of per-pattern detectors (built lazily on first use), keyed by pattern name. The
        # frame-0 detector is the ``_detector`` built below (server-grid-anchored when available);
        # additional per-round detectors are pure per-pattern-registry (server_grid_knm=None) so
        # each round scores against its own pattern's grid + fits.
        self._detector_cache = {}
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
        self._calib_root = calib_root or _DATA_ROOT
        self._roi_provider = (camera.current_roi if camera is not None else None)
        self._server_grid_knm = server_grid_knm
        self._loading_defocus = loading_defocus   # scan carrier z4 (rad); for the 3-D dz term
        self._detector = _Detector(self._calib_root, pattern_name=pattern_name,
                                   roi_provider=self._roi_provider,
                                   server_grid_knm=server_grid_knm, log=self.log,
                                   loading_defocus=loading_defocus)
        # The frame-0 detector doubles as the cache entry for the loading pattern name so
        # detect_*_for(pattern_name, ...) reuses it (server-grid-anchored) instead of rebuilding.
        if pattern_name:
            self._detector_cache[pattern_name] = self._detector

    def detect_bits(self, img):
        """Detect atoms in ``img`` -> '0'/'1' string, or '' on a calibration mismatch (so the
        caller bails without rearranging on a stale grid)."""
        return self._detector.bits(img)

    def detect_probs(self, img):
        """Detect atoms in ``img`` -> list of per-site posterior probabilities P(atom present) in
        [0,1] (same site order as :meth:`detect_bits`), or ``[]`` on a calibration mismatch. Sent to
        the SLM server's rearrange call IN PLACE OF the bitstring: the server rounds at 0.5 to get
        the source bitstring and (with ``extras.prob_hungarian``) feeds the raw floats to the
        assignment as ``site_probs``. A missing/degenerate per-site fit falls back to that site's
        hard ``intensity > threshold`` cut, so ``round(probs) == bits()`` site-for-site."""
        return self._detector.probs(img)

    # ----------------------------------------------------------------------- #
    # per-pattern detection (multi-round scan: each frame its OWN pattern grid)
    # ----------------------------------------------------------------------- #
    def detector_for(self, pattern_name):
        """The cached :class:`_Detector` for ``pattern_name`` (built from that pattern's registry
        grid + thresholds, INDEPENDENT of any other frame's pattern), building it on first use.

        ``pattern_name`` None / "" -> the frame-0 detector (server-grid-anchored when available;
        the single-round behaviour). Any OTHER name -> a pure per-pattern-registry detector
        (``server_grid_knm=None``): a distinct pattern round scores against its own derived grid,
        so a middle/final pattern with a different site count than the loading pattern detects
        correctly. Falls back to the day folder inside the detector when the registry is absent."""
        if not pattern_name:
            return self._detector
        det = self._detector_cache.get(pattern_name)
        if det is None:
            det = _Detector(self._calib_root, pattern_name=pattern_name,
                            roi_provider=self._roi_provider, server_grid_knm=None, log=self.log,
                            loading_defocus=self._loading_defocus)
            self._detector_cache[pattern_name] = det
        return det

    def detect_probs_for(self, pattern_name, img):
        """:meth:`detect_probs` against ``pattern_name``'s own registry grid (multi-round scan)."""
        return self.detector_for(pattern_name).probs(img)

    def detect_bits_for(self, pattern_name, img):
        """:meth:`detect_bits` against ``pattern_name``'s own registry grid (multi-round scan)."""
        return self.detector_for(pattern_name).bits(img)

    # ----------------------------------------------------------------------- #
    # probability-path diagnostics (prob_hungarian)
    # ----------------------------------------------------------------------- #
    def set_prob_hungarian(self, enabled, beta=None):
        """Record whether this scan requested the server's imaging-weighted Hungarian (and its
        beta), so :meth:`note_probs` can warn when the probabilities we post carry no confidence
        information for it to act on. Called once per shot by ``rearrange_callbacks.pre_run``;
        logs only on a CHANGE so the per-shot path stays quiet."""
        want = bool(enabled)
        b = None if beta is None else float(beta)
        if want == self.prob_hungarian and b == self.prob_hungarian_beta:
            return
        self.prob_hungarian = want
        self.prob_hungarian_beta = b
        self.log("[rearrange_runtime] prob_hungarian=%s beta=%s (server default when None: "
                 "(nsteps*1.5)^2)" % (want, "default" if b is None else b))

    def pattern_site_count(self, pattern_name):
        """Site count of ``pattern_name`` from the per-pattern registry record (cached), or None.
        Used ONLY by the diagnostics below to say whether a round runs in atom SURPLUS -- the
        only regime in which ``prob_hungarian`` can change the assignment (its ``-beta*log(p)``
        term is a per-ROW constant, so a square/under-filled solve is unaffected)."""
        if not pattern_name:
            return None
        if pattern_name in self._site_counts:
            return self._site_counts[pattern_name]
        n = None
        try:
            import pattern_grid
            rec = pattern_grid.get_pattern_record(pattern_name)
            knm = (rec or {}).get("knm")
            if knm:
                n = int(len(knm))
        except Exception:  # noqa: BLE001 - registry absent -> no surplus hint, never fatal
            n = None
        self._site_counts[pattern_name] = n
        return n

    def note_probs(self, tag, pattern_name, probs, target_pattern=None, log_every=200):
        """Summarise the per-site probabilities THIS round is about to post and stash them in
        :attr:`prob_diag` (+ a throttled log line) so it is visible post-hoc that probs were sent
        and whether they were informative.

        The summary is what ``prob_hungarian`` actually consumes: the loaded sites' ``-log(p)``.
        ``sum_neglog_p == 0`` means every loaded atom looked fully confident, so the server's
        ``-beta*log(p)`` term is identically zero and the assignment is the plain distance-only
        Hungarian no matter what beta is. ``surplus`` (loaded minus the NEXT pattern's site count)
        is the other gate: with ``surplus <= 0`` every loaded atom is placed and the per-row
        penalty cannot change the pairing.

        Best-effort: any failure here is swallowed (a diagnostic must never fail a shot)."""
        try:
            d = _prob_summary(probs)
        except Exception:  # noqa: BLE001
            return None
        d["tag"] = tag
        d["pattern"] = pattern_name
        n_targets = self.pattern_site_count(target_pattern)
        d["n_targets"] = n_targets
        d["surplus"] = (None if n_targets is None else int(d["n_loaded"]) - int(n_targets))
        self.prob_diag[tag] = d
        n = self._prob_shots.get(tag, 0) + 1
        self._prob_shots[tag] = n
        if n == 1 or (log_every and n % int(log_every) == 0):
            self.log(
                "[rearrange_runtime] %s probs[%s]: n=%d loaded=%d marginal=%d p_min_loaded=%.3f "
                "sum_-log(p)=%.2f surplus=%s (shot %d)"
                % (tag, pattern_name or "frame0", d["n"], d["n_loaded"], d["n_marginal"],
                   d["p_min_loaded"], d["sum_neglog_p"],
                   "?" if d["surplus"] is None else d["surplus"], n))
        if n == 1 and self.prob_hungarian and not d["informative"]:
            # Requested but toothless: the floats are hard 0/1 (no Gaussian fits, or every loaded
            # site saturated), so the server's prob_hungarian_applied will be True while the cost
            # offset is all-zero -- indistinguishable from OFF in the results. Say so loudly ONCE.
            self.log("[rearrange_runtime] WARNING %s: prob_hungarian is ON but the posted probs "
                     "carry NO confidence information (all 0/1) -- the -beta*log(p) term is "
                     "identically zero for this round. Check that %s/threshold.mat has a "
                     "gaussFitsStruct."
                     % (tag, pattern_name or "the frame-0 pattern"))
        if n == 1 and self.prob_hungarian and d["surplus"] is not None and d["surplus"] <= 0:
            self.log("[rearrange_runtime] NOTE %s: no atom surplus (loaded %d <= targets %d) -- "
                     "prob_hungarian cannot change the assignment this round (its penalty is a "
                     "per-row constant)." % (tag, d["n_loaded"], n_targets))
        return d

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
    # Per-frame wall clock for the RP-N correlation campaign (TEMPORARY). This is the SINGLE
    # funnel every rearrangement grab goes through (hand_over_slm / mid / finalize), so one stamp
    # here times all of them -- a ~1 s rearrange shot's pre/post bracket is far too coarse for a
    # 100 ms correlation window. Inert when no scan is active; never raises.
    shot_time.stamp_frame()
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
                 log=None, loading_defocus=None):
        self._data_root = data_root
        self._pattern_name = pattern_name
        self._roi_provider = roi_provider
        self._server_grid_knm = server_grid_knm   # server init_grid (N,2 [y,x] knm); single source
        self._loading_defocus = loading_defocus   # scan carrier z4 (rad); for the 3-D dz term
        self._log = log or (lambda _m: None)
        self._W = None                 # scipy.sparse (M, H*W), columns in C(row-major) order
        # Gather fast path derived from _W (see _build_gather); None -> _apply uses the matvec.
        self._gidx = None              # flat pixel indices, CSR column order
        self._gwts = None              # matching mask weights
        self._gseg = None              # per-site segment starts (CSR indptr[:-1])
        self._gempty = None            # sites with no in-image pixels
        self._thresholds = None        # (M,)
        self._gauss_params = None      # list[M] of (6,) [mu_e,s_e,A_e,mu_a,s_a,A_a] or None/site
        self._gp_packed = None         # (M, 6) float view of _gauss_params (packed once)
        self._gp_valid = None          # (M,) bool: posterior defined at this site
        self._img_shape = None         # (H, W)
        self._key = None               # cache identity of the built calibration

    def _intensities(self, img):
        """Per-site masked intensity vector for ``img``, or None on a calibration mismatch.
        Shared by :meth:`bits` (hard threshold) and :meth:`probs` (posterior) so both score the
        SAME intensities in the SAME server site order."""
        import numpy as np
        a = np.asarray(img)
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
        return self._apply(a)

    def _apply(self, a):
        """The per-shot math, split out from :meth:`_intensities` so it is unit-testable without
        a calibration on disk. ``a`` is the 2-D frame, ANY dtype (the camera hands us uint16).

        Gathers only the masked pixels instead of reducing the whole frame. ``W`` is 0.6% dense
        (26.7k of 4.4M pixels on the 2100x2100 ROI), so the matvec itself was never the cost --
        materialising the operand was. The old path did ``W.dot(img.astype(float).ravel("F"))``:
        a uint16->float64 cast (8.8 -> 35 MB) plus an F-order ravel of a C-contiguous array,
        which numpy cannot return as a view, so it copied 35 MB again with a cache-hostile
        stride. Measured on the live ROI: cast 8.2 ms + F-ravel 21.8 ms + matvec 0.11 ms.
        Since ``_build`` now indexes ``W``'s columns in C order, the gather below reads straight
        off a free ``ravel()`` view and costs ~0.2 ms -- the same arithmetic on the same pixels.

        Falls back to the ``W`` matvec if the gather arrays are unavailable or the build-time
        self-check failed, so a bad precompute degrades to the old behaviour, never to wrong
        intensities."""
        import numpy as np
        flat = np.ascontiguousarray(a).ravel()          # C-order; a no-op view when contiguous
        if self._gidx is None:
            return self._W.dot(flat.astype(float))      # fallback: exact old arithmetic
        vals = flat[self._gidx].astype(np.float64)
        vals *= self._gwts
        if vals.size == 0:
            return np.zeros(self._W.shape[0], dtype=float)
        # reduceat needs in-range starts; empty rows (sites clipped entirely off-image) would
        # otherwise read their neighbour's first element, so they are zeroed explicitly.
        out = np.add.reduceat(vals, np.minimum(self._gseg, vals.size - 1))
        if self._gempty is not None and self._gempty.any():
            out[self._gempty] = 0.0
        return out

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

        A site with a MISSING/degenerate Gaussian fit falls back to that site's hard
        ``intensity > threshold`` decision (1.0/0.0), so rounding the floats at 0.5 -- which is
        exactly how the SLM server derives the source bitstring -- reproduces :meth:`bits`
        site-for-site. If the WHOLE calibration lacks usable Gaussian fits (older threshold.mat
        with no gaussFitsStruct) every site takes that hard cut, so the scan still rearranges
        instead of sending all-zeros -- but note the floats then carry NO confidence information
        and ``prob_hungarian`` becomes a no-op (``-beta*log(1) == 0``)."""
        intensities = self._intensities(img)
        if intensities is None:
            return []
        import numpy as np
        gp = self._gauss_params
        if gp is None or self._gp_valid is None or not bool(self._gp_valid.any()):
            self._log("[rearrange_runtime] no usable Gaussian fits in calibration; probs fall "
                      "back to hard intensity>threshold (1.0/0.0) -- prob_hungarian will have "
                      "NO effect (-beta*log(1) == 0)")
            logicals = intensities > self._thresholds
            return [1.0 if b else 0.0 for b in logicals]
        inten = np.asarray(intensities, dtype=float)
        # OPT-IN (env YB_NORM_PROBS=1; DEFAULT OFF): per-frame COMMON-MODE brightness normalization
        # before the posterior. The stored Gaussian fits (gp) come from a BRIGHT standalone calibration;
        # an in-context frame (e.g. the dim mid/final rearrange image) is globally dimmer, so every
        # intensity falls toward the calibration EMPTY peak -> posterior ~0 -> bits collapse
        # (measured 0.175 vs true ~0.68 on the 2198 middle, 2026-07-19). Rescale this frame's
        # atom-signal so its loaded-site mean matches the calibration's mean atom brightness, then
        # apply the stored posterior. Contained, gated, reversible. (mu_e per site subtracted as the
        # per-site baseline; a single global gain applied to the signal part.)
        if os.environ.get("YB_NORM_PROBS") == "1" or _norm_probs_flag():
            try:
                mu_e = np.array([p[0] if p is not None else np.nan for p in gp])
                mu_a = np.array([p[3] if p is not None else np.nan for p in gp])
                thr = np.asarray(self._thresholds, dtype=float)
                good = np.isfinite(mu_e) & np.isfinite(mu_a) & ((mu_a - mu_e) > 0.5)
                sig = inten - np.nan_to_num(mu_e)              # per-site signal above empty
                loaded = good & (inten > thr)                 # this frame's confidently-loaded sites
                if loaded.sum() >= 30:
                    cur = sig[loaded].mean()                  # this frame's mean atom signal
                    ref = (mu_a - mu_e)[loaded].mean()        # calibration's mean atom signal
                    if cur > 0 and ref > 0:
                        gain = ref / cur
                        inten = np.nan_to_num(mu_e) + sig * gain   # rescale signal to calibration frame
                        self._log("[rearrange_runtime] YB_NORM_PROBS: common-mode gain %.3f "
                                  "(frame atom-sig %.2f -> calib %.2f, %d loaded)"
                                  % (gain, cur, ref, int(loaded.sum())))
            except Exception as e:  # noqa: BLE001 - never break detection over the normalization
                self._log("[rearrange_runtime] YB_NORM_PROBS skipped (%s)" % e)
        # Sites whose two-Gaussian fit is missing / degenerate have NO posterior. Fall back to
        # this frame's hard cut there instead of 0.0: the SLM server builds the rearrangement
        # bitstring by rounding these floats at 0.5, so a 0.0 at a site bits() calls '1' silently
        # DROPS a real atom from the source set (and leaves a target unfilled). With the fallback,
        # round(probs) == bits() site-for-site (cut on the SAME intensities the posterior saw, so
        # the opt-in common-mode rescale above applies to both). No-op on every current production
        # calibration (tri_3013_camfb / kagome_res_2198 / kagome_2078_camfb: 0 degenerate fits).
        hard = (inten > np.asarray(self._thresholds, dtype=float)).astype(float)
        post = _atom_posterior_packed(inten, self._gp_packed, self._gp_valid, fallback=hard)
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
            # Self-orienting match: record ``knm`` column order is [y,x] for some records and
            # [x,y] for others (the registry never enforced one; a centered SQUARE array is
            # near-symmetric under the swap so production 2D never noticed, but an off-diagonal
            # array -- e.g. 2x11x11_5um_back2um at knm x~631-751/y~454-569 -- hard-fails one
            # orientation, max~254). Try both; use whichever passes the gate.
            dist = idx = None
            for cand in (rknm, rknm[:, ::-1]):
                d_c, i_c = cKDTree(cand).query(sknm)   # for each server site, nearest record site
                if float(np.max(d_c)) <= 5.0 and len(set(i_c.tolist())) == n:
                    dist, idx = d_c, i_c
                    break
            if idx is None:
                d_raw, i_raw = cKDTree(rknm).query(sknm)
                self._log("[rearrange_runtime] server-grid<->record position match failed "
                          "(max=%.2f, bijection=%s; both orientations); registry/day"
                          % (float(np.max(d_raw)), len(set(i_raw.tolist())) == n))
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
        # 3-D linear-defocus correction (pattern_grid.load_dz): a site away from the camera-focus
        # plane walks laterally ~ v * z_total px. Applied ONLY for a 3-D record with a configured
        # dz block -- every flat-2-D pattern (any defocus) is byte-identical without it. The
        # record's per-site z_rad rides the SAME position match (idx) as the thresholds, so
        # z aligns with the server's site order.
        try:
            if rec.get("is_3d") and rec.get("z_rad") is not None:
                dz = pattern_grid.load_dz()
                if dz is not None:
                    z_rad = np.asarray(rec["z_rad"], dtype=float).ravel()
                    if z_rad.shape[0] == n:
                        zc = (self._loading_defocus if self._loading_defocus is not None
                              else dz["z_ref"])
                        z_tot = (float(zc) - dz["z_ref"]) + z_rad[idx]
                        grid = grid + z_tot[:, None] * dz["v"][None, :]
                        self._log("[rearrange_runtime] dz correction: v=%s px/rad carrier=%s "
                                  "z=[%.2f, %.2f]" % (list(dz["v"]), zc,
                                                      float(z_tot.min()), float(z_tot.max())))
        except Exception as e:  # noqa: BLE001 - the dz term must never break the grid source
            self._log("[rearrange_runtime] dz correction skipped (%s)" % e)
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
        # 3-D linear-defocus correction (same model as _server_source; record order, no re-match).
        try:
            rec = pattern_grid.get_pattern_record(self._pattern_name)
            if rec and rec.get("is_3d") and rec.get("z_rad") is not None:
                dz = pattern_grid.load_dz()
                if dz is not None:
                    z_rad = np.asarray(rec["z_rad"], dtype=float).ravel()
                    if z_rad.shape[0] == grid.shape[0]:
                        zc = (self._loading_defocus if self._loading_defocus is not None
                              else dz["z_ref"])
                        z_tot = (float(zc) - dz["z_ref"]) + z_rad
                        grid = grid + z_tot[:, None] * dz["v"][None, :]
        except Exception as e:  # noqa: BLE001
            self._log("[rearrange_runtime] dz correction skipped (%s)" % e)
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

    def _build_gather(self):
        """Derive the per-shot gather arrays from ``self._W`` and self-check them.

        CSR already stores exactly what the gather needs: for site ``i`` the masked pixels are
        ``W.indices[W.indptr[i]:W.indptr[i+1]]`` with weights the matching ``W.data``. So this
        is a re-view of the same numbers, not a second calibration -- there is no way for the
        two paths to disagree about WHICH pixel carries WHICH weight.

        The self-check runs once per build on a deterministic pseudo-random frame and compares
        the gather against the ``W`` matvec. On any mismatch (or any failure here) the gather is
        disabled and :meth:`_apply` falls back to the matvec, so the worst case is the old
        speed, never wrong intensities."""
        import numpy as np
        self._gidx = self._gwts = self._gseg = self._gempty = None
        try:
            W = self._W
            indptr = np.asarray(W.indptr)
            self._gidx = np.asarray(W.indices)
            self._gwts = np.asarray(W.data, dtype=np.float64)
            self._gseg = indptr[:-1].copy()
            self._gempty = (indptr[1:] == indptr[:-1])
            H, Wd = self._img_shape if self._img_shape else (0, 0)
            if H and Wd:
                rng = np.random.default_rng(12345)
                probe = rng.integers(0, 4000, size=(H, Wd), dtype=np.uint16)
                fast = self._apply(probe)
                ref = W.dot(np.ascontiguousarray(probe).ravel().astype(float))
                if not np.allclose(fast, ref, rtol=0, atol=1e-9):
                    bad = int(np.count_nonzero(~np.isclose(fast, ref, rtol=0, atol=1e-9)))
                    self._log("[rearrange_runtime] gather self-check FAILED on %d/%d site(s) "
                              "-- falling back to the sparse matvec" % (bad, len(ref)))
                    self._gidx = self._gwts = self._gseg = self._gempty = None
        except Exception as e:  # noqa: BLE001 - any precompute failure -> safe fallback
            self._log("[rearrange_runtime] gather precompute unavailable (%s); using matvec" % e)
            self._gidx = self._gwts = self._gseg = self._gempty = None

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
                    # ROW-major linear (0-based). Was column-major ((xx-1)*H + (yy-1)) to mirror
                    # MATLAB's W * img(:); that forced _apply to F-order-ravel a C-contiguous
                    # frame every shot, a full 35 MB strided copy (21.8 ms measured). Indexing
                    # in C order here picks the SAME pixel with the SAME weight and lets the
                    # per-shot path read a free ravel() view instead. Site ORDER (the rows) is
                    # untouched -- only the column numbering of an internal matrix changed.
                    lin = (yy - 1) * W + (xx - 1)
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
        # Pack the per-site fits ONCE here (not per shot): probs() then runs a single vector op
        # instead of an M-iteration Python loop on the held-atom critical path.
        self._gp_packed, self._gp_valid = _pack_gauss_params(gauss_params)
        self._img_shape = (H, W)
        self._key = key
        self._build_gather()        # AFTER _img_shape: the self-check probes at that size
        n_fit = int(self._gp_valid.sum()) if self._gp_valid is not None else 0
        self._log("[rearrange_runtime] detector built: source=%s M=%d imgSize=[%d %d] "
                  "gaussFits=%s (%d/%d sites with a usable posterior)"
                  % (key[0], m, H, W, "yes" if gauss_params is not None else "no", n_fit, m))

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


def _pack_gauss_params(params_list):
    """``(packed (M,6) float, valid (M,) bool)`` from a list of per-site
    ``[mu_e, s_e, A_e, mu_a, s_a, A_a]`` / ``None``, or ``(None, None)`` for an absent list.

    A site is VALID only with six finite params, both sigmas > 0 and both mixture areas > 0 --
    i.e. a posterior that is actually defined. Packed ONCE per calibration (in
    :meth:`_Detector._build`) so the per-shot posterior is a pure vector op: the old per-site
    Python loop cost ~10 ms per 3000-site frame ON THE HELD-ATOM CRITICAL PATH (measured
    2026-07-27: 9.8 ms for 3013 sites, 7.0 ms for 2198), i.e. ~17 ms per two-round shot between
    the image and ``rearrange()``."""
    import numpy as np
    if params_list is None:
        return None, None
    m = len(params_list)
    packed = np.zeros((m, 6), dtype=float)
    valid = np.zeros(m, dtype=bool)
    for i, params in enumerate(params_list):
        if params is None:
            continue
        p = np.asarray(params, dtype=float).ravel()
        if p.size < 6 or not np.all(np.isfinite(p[:6])):
            continue
        if not (p[1] > 0 and p[4] > 0 and p[2] > 0 and p[5] > 0):
            continue
        packed[i] = p[:6]
        valid[i] = True
    return packed, valid


def _logistic(d):
    """Numerically stable ``1 / (1 + exp(-d))`` (handles +-inf; no overflow warnings)."""
    import numpy as np
    d = np.clip(np.asarray(d, dtype=float), -700.0, 700.0)
    out = np.empty(d.shape, dtype=float)
    pos = d >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-d[pos]))
    e = np.exp(d[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _atom_posterior_packed(intensities, packed, valid, fallback=None):
    """Per-site ``P(atom present | intensity)`` from PRE-PACKED two-Gaussian params (see
    :func:`_pack_gauss_params`). Vectorised and computed in LOG space:

        p = sigmoid( log(A_a/A_e) + log(s_e/s_a)
                     - (x-mu_a)^2/(2 s_a^2) + (x-mu_e)^2/(2 s_e^2) )

    which is algebraically identical to ``A_a N_a / (A_e N_e + A_a N_a)`` but never underflows.
    The direct-ratio form did: with the production fits (s_a ~ 0.9 ADU on the 2198 middle
    pattern) BOTH densities vanish once the intensity sits ~40 sigma above the atom peak, the
    ``0/0`` guard returned 0.0, and a BRIGHT site was reported as EMPTY -- an inverted answer for
    exactly the sites we are most sure about (measured 2026-07-27: at ``mu_a + 40 ADU``,
    2069/2198 sites collapsed to p=0.0). In log space those sites correctly return 1.0.

    ``fallback`` (optional, length N) supplies the probability for sites whose posterior is
    UNDEFINED (missing / degenerate fit). Pass the hard ``intensity > threshold`` decision so
    ``round(probs) == bits()`` stays true site-for-site -- the SLM server DERIVES the bitstring by
    rounding these floats at 0.5, so a site where the two disagree is silently added to or dropped
    from the rearrangement source set. ``None`` -> 0.0 (the historical conservative default)."""
    import numpy as np
    x = np.asarray(intensities, dtype=float).ravel()
    n = x.shape[0]
    if fallback is None:
        out = np.zeros(n, dtype=float)
    else:
        fb = np.asarray(fallback, dtype=float).ravel()
        out = (np.clip(fb, 0.0, 1.0).astype(float) if fb.shape[0] == n
               else np.zeros(n, dtype=float))
    if packed is None or valid is None or n == 0:
        return out
    v = np.asarray(valid, dtype=bool).ravel()
    if v.shape[0] != n:
        return out
    v = v & np.isfinite(x)
    if not v.any():
        return out
    p = np.asarray(packed, dtype=float)[v]
    xv = x[v]
    mu_e, s_e, A_e = p[:, 0], p[:, 1], p[:, 2]
    mu_a, s_a, A_a = p[:, 3], p[:, 4], p[:, 5]
    d = (np.log(A_a) - np.log(A_e) + np.log(s_e) - np.log(s_a)
         - 0.5 * ((xv - mu_a) / s_a) ** 2
         + 0.5 * ((xv - mu_e) / s_e) ** 2)
    out[v] = _logistic(d)
    return out


def _atom_posterior(intensities, params_list, fallback=None):
    """Per-site posterior ``P(atom present | intensity)`` under the per-site two-Gaussian mixture
    ``params = [mu_e, s_e, A_e, mu_a, s_a, A_a]`` (empty peak first; fitted areas A_e/A_a are the
    mixing weights, so the posterior folds in the site's loading rate).

    Mirrors ``yb_analysis/detection/dynamical_threshold.py:atom_posterior`` -- reimplemented
    locally (numpy only) so the engine-venv backend runtime takes NO yb_analysis import
    dependency. Convenience wrapper that packs ``params_list`` then delegates to
    :func:`_atom_posterior_packed`; the live detector packs ONCE per calibration and calls that
    directly. A site whose params are missing (None) or degenerate (s<=0 / A<=0) takes
    ``fallback`` (default 0.0: uncertain -> treated as empty)."""
    packed, valid = _pack_gauss_params(params_list)
    return _atom_posterior_packed(intensities, packed, valid, fallback=fallback)


def _prob_summary(probs):
    """Compact summary of one round's posted per-site probabilities (see
    :meth:`ScanContext.note_probs`). Pure function of the list -- no I/O.

    ``sum_neglog_p`` is the exact quantity the SLM server's ``prob_hungarian`` turns into cost
    (it adds ``-beta * log(p)`` to each LOADED row), so it is the single number that says whether
    the probabilities can influence the assignment at all."""
    import numpy as np
    p = np.asarray(list(probs), dtype=float).ravel()
    n = int(p.size)
    loaded = p >= 0.5                       # the server rounds at 0.5 to build the bitstring
    n_loaded = int(loaded.sum())
    pl = p[loaded]
    neglog = -np.log(np.clip(pl, 1e-12, 1.0)) if n_loaded else np.zeros(0)
    return {
        "n": n,
        "n_loaded": n_loaded,
        # loaded but not fully confident -- the population prob_hungarian can act on
        "n_marginal": int(((pl >= 0.5) & (pl < 0.999)).sum()) if n_loaded else 0,
        "p_min_loaded": float(pl.min()) if n_loaded else 0.0,
        "sum_neglog_p": float(neglog.sum()),
        "max_neglog_p": float(neglog.max()) if n_loaded else 0.0,
        # False -> the floats are effectively a bitstring; -beta*log(p) is identically zero
        "informative": bool(n_loaded and float(neglog.sum()) > 0.0),
    }


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
