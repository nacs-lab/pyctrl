"""orca_camera.py -- pylablib wrapper for the Orca-Quest qCMOS (scenario-3 capture).

pyctrl mirror of the MATLAB Orca path (``OrcaInit.m`` / ``OrcaImag.m`` + the ``server_post_run``
capture), but a PORT, not a reuse: production capture is MATLAB IMAQ ``videoinput('hamamatsu')``;
pyctrl scenario-3 captures via **pylablib** (``pylablib.devices.DCAM.DCAMCamera``) over the
installed ``C:\\Windows\\System32\\dcamapi.dll`` (``pylablib-lightweight`` needs only that runtime
DLL -- no DCAM-SDK). The camera reports ``C15550-20UP`` (ORCA-Quest qCMOS, 4096x2304). DCAM is
ONE handle per camera, so this opens it only when MATLAB is off (the scenario design).

The three documented uses map to attributes (references/runtime-design.md):
  * cooler / temperature -- ``sensor_temperature`` (read), ``sensor_cooler*`` status.
  * exposure + ROI       -- ``get_exposure`` / ``set_exposure``, ``get_roi`` / ``set_roi``
                            (pylablib maps ROI to the DCAM ``subarray_*`` attributes).
  * capture              -- INTERNAL/software trigger for a standalone snap (NO FPGA), or
                            EXTERNAL trigger that waits on an FPGA rising edge for a real shot.

External-trigger capture (user-confirmed 2026-06-02): the Orca triggers on a **rising edge** of
``FPGA1/TTL54`` (``expConfig.m`` ``TTLOrcaTrig``; NOT TTL14, which is ``TTLQickTrig``); pulse it
on then off (the off just re-arms for the next rising edge). The channel is still a parameter
(``trigger_ttl``) for flexibility, and the pulse itself is an FPGA ``set_chns`` action (the run
loop / test harness drives it, not this wrapper).

``store_imgs`` wire format reproduced (so yb_analysis ``_process_imgs`` reads frames correctly):
each image is a flat double array ``[s1, s2, s3, <s1*s2*s3 pixels COLUMN-MAJOR>]`` (3-element
shape prefix + Fortran-order flatten); the run loop calls ``server.store_imgs(arr, scan_id,
seq_id)`` per frame and ``server.seq_finish()`` after a sequence's frames.

NEEDS-HARDWARE: opening the camera + capture drive the device (read-only temp/exposure/ROI
probing is safe). The pylablib import is lazy and the backend is injectable, so this module
imports + unit-tests with a fake DCAM and never needs pylablib present.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

# Full-frame default ROI for the C15550-20UP (4096 wide x 2304 tall): [x, y, w, h].
DEFAULT_ROI = [0, 0, 4096, 2304]

# DCAM SENSOR COOLER numeric values (fallback when pylablib won't take the string label).
_COOLER_NUM = {"off": 1, "on": 2, "max": 4}


def _open_dcam(index=0):
    """Open DCAM camera ``index`` via pylablib (lazy import; needs the runtime DLL)."""
    from pylablib.devices import DCAM
    return DCAM.DCAMCamera(idx=index)


class OrcaCamera:
    """Thin wrapper over a pylablib ``DCAMCamera`` (or an injected fake).

    Args:
        roi / exposure: optional initial settings applied on construction.
        index: DCAM device index (default 0).
        cam: an already-open backend (injected in tests) -- skips the pylablib open.
        open_cam: ``index -> backend`` factory override (tests); default :func:`_open_dcam`.
        trigger_ttl: the FPGA channel whose rising edge triggers a frame in external mode
            (UNCONFIRMED -- see module note; pass the verified channel).
    """

    def __init__(self, roi=None, exposure=None, *, index=0, cam=None, open_cam=None,
                 trigger_ttl="FPGA1/TTL54"):
        self._index = index
        self._open_cam = open_cam or _open_dcam      # remembered so reconnect() can reopen
        self._cam = cam if cam is not None else self._open_cam(index)
        self.trigger_ttl = trigger_ttl
        self._acquiring = False
        if exposure is not None:
            self.set_exposure(exposure)
        if roi is not None:
            self.set_roi(roi)

    @property
    def connected(self):
        """True iff the DCAM handle is currently open."""
        return self._cam is not None

    def disconnect(self):
        """Release the DCAM handle (== MATLAB ``stop+delete vid``). Alias of :meth:`close`."""
        self.close()

    def reconnect(self):
        """Reopen the DCAM handle after a :meth:`disconnect` (== reinitialize ``vid``).

        Reuses the remembered device index + open factory. No-op if already connected.
        """
        if self._cam is not None:
            return
        self._cam = self._open_cam(self._index)
        self._acquiring = False

    # ----------------------------------------------------------------------- #
    # handle_camera_cmd interface (runner.handle_camera_cmd contract)
    # ----------------------------------------------------------------------- #
    def init(self, roi, exposure=None):
        """Apply ROI (+ exposure) on (re)init; return the ACTUAL ``(roi, exposure)``."""
        if exposure is not None:
            self.set_exposure(exposure)
        self.set_roi(roi)
        return self.current_roi(), self.get_exposure()

    def apply_settings(self, roi, exposure=None):
        """Live ROI/exposure change (GUI "Apply Settings"); return actual ``(roi, exposure)``.

        ROI changes require acquisition to be stopped; we stop, apply, and leave it stopped
        (the run loop re-arms before a shot)."""
        was = self._acquiring
        if was:
            self.stop_acquisition()
        if exposure is not None:
            self.set_exposure(exposure)
        self.set_roi(roi)
        return self.current_roi(), self.get_exposure()

    def current_roi(self):
        """Current ROI as ``[x, y, w, h]`` (the set_camera_result / monitor convention)."""
        return self.get_roi()

    def close(self):
        """Release the DCAM handle (so the next backend can open it). Idempotent."""
        cam = self._cam
        self._cam = None
        if cam is None:
            return
        try:
            if self._acquiring:
                try:
                    cam.stop_acquisition()
                except Exception:
                    pass
                self._acquiring = False
        finally:
            cam.close()

    # ----------------------------------------------------------------------- #
    # cooler / temperature (read-only -- safe to probe live)
    # ----------------------------------------------------------------------- #
    def get_temperature(self):
        """Sensor temperature in degrees C (DCAM ``sensor_temperature``)."""
        return float(self._attr("sensor_temperature"))

    def get_cooler_status(self):
        """Cooler status string/flag (DCAM ``sensor_cooler_status``), or ``''`` if absent."""
        try:
            return self._attr("sensor_cooler_status")
        except Exception:  # noqa: BLE001 - attribute name varies by firmware
            return ""

    def get_cooler(self):
        """Current cooler MODE (DCAM ``sensor_cooler``: e.g. off/on/max)."""
        return self._attr("sensor_cooler")

    def set_cooler(self, mode):
        """Set the cooler mode (DCAM ``sensor_cooler``).

        ``mode`` may be a pylablib enum label (``"off"`` / ``"on"`` / ``"max"``) or the DCAM
        numeric (OFF=1, ON=2, MAX=4). The ORCA-Quest uses fixed cooling levels (no arbitrary
        target temperature), so this selects the level rather than a setpoint. Returns the
        read-back mode. (Exact accepted labels confirmed live -- see the wrapper notes.)
        """
        try:
            self._set_attr("sensor_cooler", mode)
        except Exception:  # noqa: BLE001 - label not accepted -> try the DCAM numeric
            self._set_attr("sensor_cooler", _COOLER_NUM.get(str(mode).lower(), mode))
        return self.get_cooler()

    # ----------------------------------------------------------------------- #
    # exposure + ROI
    # ----------------------------------------------------------------------- #
    def get_exposure(self):
        return float(self._cam.get_exposure())

    def set_exposure(self, seconds):
        self._cam.set_exposure(float(seconds))
        return self.get_exposure()

    def get_roi(self):
        """Return ROI as ``[x, y, w, h]`` (pylablib ``get_roi`` -> ``(hstart,hend,vstart,vend...)``)."""
        r = self._cam.get_roi()
        # pylablib get_roi returns (hstart, hend, vstart, vend[, hbin, vbin]).
        hstart, hend, vstart, vend = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        return [hstart, vstart, hend - hstart, vend - vstart]

    def set_roi(self, roi):
        """Set ROI from ``[x, y, w, h]`` (-> pylablib ``set_roi(hstart, hend, vstart, vend)``)."""
        x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        self._cam.set_roi(x, x + w, y, y + h)
        return self.get_roi()

    # ----------------------------------------------------------------------- #
    # trigger + capture
    # ----------------------------------------------------------------------- #
    def set_trigger_internal(self):
        """Internal/software trigger -- a standalone snap with NO FPGA (safe for testing)."""
        self._cam.set_trigger_mode("int")

    def set_trigger_external(self, polarity="positive"):
        """External trigger -- each frame waits on an EDGE of :attr:`trigger_ttl`.

        ⚠ pylablib's ``set_trigger_mode("ext")`` leaves DCAM ``trigger_polarity`` at NEGATIVE
        (falling edge); the Orca is wired RISING-edge (user-confirmed 2026-06-02), so we force
        ``trigger_polarity = positive``. Without this a rising-edge TTL pulse is silently
        ignored (verified live: a TTL54 0->1 produced no frame until polarity was set positive).
        """
        self._cam.set_trigger_mode("ext")
        try:
            self._cam.set_attribute_value("trigger_polarity", polarity)
        except Exception:  # noqa: BLE001 - fall back to the DCAM numeric (POSITIVE == 2)
            self._cam.set_attribute_value("trigger_polarity", 2)

    def snap(self):
        """Grab ONE frame on the internal trigger (standalone, no FPGA). Returns an ndarray."""
        self.set_trigger_internal()
        return self._cam.snap()

    # -- MATLAB IMAQ-parity arm/stop ("start(vid)" / "stop(vid)" + FramesAvailable/flushdata) --
    def start_video(self, nframes=256, external=True):
        """Arm continuous acquisition (== MATLAB ``start(vid)``).

        Sets the trigger (external rising-edge by default -- run-loop mode; ``external=False``
        free-runs on the internal trigger, no FPGA), allocates a circular buffer of ``nframes``,
        and starts. Frames then ACCUMULATE in the buffer as triggers arrive (read them with
        :meth:`read_frames`, count with :meth:`frames_available`), until :meth:`stop_video`.
        """
        if external:
            self.set_trigger_external()
        else:
            self.set_trigger_internal()
        self._cam.setup_acquisition(mode="sequence", nframes=nframes)
        self._cam.start_acquisition()
        self._acquiring = True

    def stop_video(self):
        """Stop acquisition (== MATLAB ``stop(vid)``). Idempotent."""
        self.stop_acquisition()

    def frames_available(self):
        """Number of unread frames in the buffer (== MATLAB ``vid.FramesAvailable``)."""
        try:
            return int(self._cam.get_frames_status().unread)
        except Exception:  # noqa: BLE001 - older pylablib / not acquiring
            return 0

    def is_running(self):
        """True iff acquisition is in progress (== ``strcmp(vid.Running,'on')``)."""
        try:
            return bool(self._cam.acquisition_in_progress())
        except Exception:  # noqa: BLE001
            return self._acquiring

    def flush(self):
        """Drop any buffered frames (== MATLAB ``flushdata(vid)``); returns the count dropped.

        SequenceRunner flushes stale frames before a scan so orphans from a prior aborted run
        aren't misattributed to the first sequence. Returns how many were discarded.
        """
        dropped = self.read_frames()
        return len(dropped)

    # -- run-loop aliases (same acquisition; named for the capture path) --
    def start_acquisition(self, nframes=256, external=True):
        """Arm acquisition (alias of :meth:`start_video`, run-loop naming)."""
        self.start_video(nframes=nframes, external=external)

    def read_frames(self):
        """Read all frames captured so far (after triggers); returns a list of ndarrays."""
        frames = self._cam.read_multiple_images()
        return list(frames) if frames is not None else []

    def stop_acquisition(self):
        if self._acquiring:
            self._cam.stop_acquisition()
            self._acquiring = False

    # ----------------------------------------------------------------------- #
    # helpers
    # ----------------------------------------------------------------------- #
    def _attr(self, name):
        """Read a DCAM attribute (pylablib ``get_attribute_value`` / ``cav`` mapping)."""
        getter = getattr(self._cam, "get_attribute_value", None)
        if getter is not None:
            return getter(name)
        return self._cam.cav[name]

    def _set_attr(self, name, value):
        """Write a DCAM attribute (pylablib ``set_attribute_value`` / ``cav`` mapping)."""
        setter = getattr(self._cam, "set_attribute_value", None)
        if setter is not None:
            setter(name, value)
            return
        self._cam.cav[name] = value


def to_store_array(frame):
    """Flatten one frame to the ``store_imgs`` wire array ``[s1, s2, s3, <pixels col-major>]``.

    Mirrors what ``ExptServer.store_imgs`` expects (a flat double iterable per image): a
    3-element shape prefix then the pixels in COLUMN-MAJOR (Fortran) order, so yb_analysis
    ``_process_imgs`` (``reshape(s1, s2, s3, order='F')``) reconstructs the image. A 2-D frame
    ``(H, W)`` becomes ``s1=H, s2=W, s3=1``. Returns a numpy float64 array.
    """
    import numpy as np
    a = np.asarray(frame)
    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    if a.ndim != 3:
        raise ValueError("frame must be 2-D or 3-D, got ndim=%d" % a.ndim)
    s1, s2, s3 = a.shape
    flat = a.reshape(-1, order="F").astype(np.float64)
    return np.concatenate(([float(s1), float(s2), float(s3)], flat))
