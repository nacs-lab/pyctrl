"""slm_client.py -- thin HTTP client for the SLM / rearrangement server (pyctrl side).

A faithful, minimal Python transliteration of the methods of ``SLMnet/.../slm_client.m`` that
the rearrangement run loop needs. The SLM server (FastAPI + slmsuite + the GPU model) runs on a
SEPARATE machine; this is purely an HTTP/JSON CLIENT -- it imports NO slmnet/torch code (the
MATLAB ``slm_client.m`` is likewise a pure HTTP client). So pyctrl talks to the server with only
``requests`` (already present in both interpreters), and the byte-builder / engine stay untouched.

Scope (what the rearrangement scan needs, mirroring ``slm_client.m``):
  * locks            -- :meth:`acquire_lock` / :meth:`release_lock` / :meth:`heartbeat`
                        (server-side blocking acquire with re-issue, per the MATLAB contract).
  * rearrange setup  -- :meth:`setup_rearrangement` / :meth:`reload_rearrange`.
  * per-shot         -- :meth:`rearrange` / :meth:`update_rearrange` / :meth:`cancel_last_shot`.
  * loading phase    -- :meth:`write_loading_phase` (scan-long SLM session).
  * health / status  -- :meth:`health` / :meth:`lock_status`.

The big binary endpoints (write_slm / capture / settle_series / eval / Fourier cal) are NOT
ported -- the rearrangement loop never touches them from pyctrl (the server owns the camera + SLM
writes during a shot). Add them here if a future pyctrl path needs them.

The :func:`get_client` singleton (mirroring MATLAB ``SLMClient.get(url, pw).client()``) hands back
ONE client per server url so the scan-long ``slm`` hold and the per-shot ``compute`` lock share a
single ``X-Client-Id`` -- which the server's ``rearrange_actual`` requires (same client must hold
``slm``; a DIFFERENT client holding ``compute`` is rejected).

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import base64
import os
import time

# Default server (this machine -> SLM PC over the LAN HTTP companion port; loopback bypasses
# auth, the LAN port needs the password). Mirrors RearrangeCommSeq.m::resolve_slm_client.
DEFAULT_URL = "http://192.168.0.171:8551"
DEFAULT_PASSWORD = "174171"


class SlmHTTPError(RuntimeError):
    """Raised on a 4xx/5xx from the server. ``status`` carries the HTTP code so callers can
    distinguish lock-contention (423) / blocking-timeout (408) from other failures."""

    def __init__(self, status, detail=""):
        self.status = int(status)
        self.detail = str(detail)
        super().__init__("HTTP %d from SLM server%s"
                         % (self.status, (": " + self.detail) if self.detail else ""))


class SlmClient:
    """HTTP client for one SLM server, identified by a stable ``client_id`` (lock owner)."""

    def __init__(self, url=DEFAULT_URL, password="", client_id=None,
                 timeout_s=30.0, verify_ssl=False, session=None):
        self.url = str(url).rstrip("/")
        self._password = str(password or "")
        self._timeout_s = float(timeout_s)
        self._verify_ssl = bool(verify_ssl)
        if client_id:
            self.client_id = str(client_id)
        else:
            # Stable per-process id so the scan-long slm hold + per-shot compute lock share one
            # owner (the server keys write-permission off X-Client-Id).
            self.client_id = "pyctrl_%s_%d" % (os.environ.get("COMPUTERNAME", "host"),
                                               os.getpid())
        # A requests.Session keeps the TCP socket warm across the per-shot calls. Lazily built
        # (so importing this module never requires `requests`); injectable for tests.
        self._session = session

    # ----------------------------------------------------------------------- #
    # transport
    # ----------------------------------------------------------------------- #
    def _sess(self):
        if self._session is None:
            import requests  # lazy: byte/structure tests never import this module
            self._session = requests.Session()
        return self._session

    def _headers(self, extra=None):
        h = {"X-Client-Id": self.client_id}
        if self._password:
            token = base64.b64encode(("admin:" + self._password).encode("utf-8")).decode("ascii")
            h["Authorization"] = "Basic " + token
        if extra:
            h.update(extra)
        return h

    def _check(self, resp):
        code = int(resp.status_code)
        if code < 400:
            return
        detail = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict) and "detail" in payload:
                detail = str(payload["detail"])
        except Exception:  # noqa: BLE001 - non-JSON body
            try:
                detail = resp.text
            except Exception:  # noqa: BLE001
                detail = ""
        raise SlmHTTPError(code, detail)

    def _get_json(self, path):
        resp = self._sess().get(self.url + path, headers=self._headers(),
                                timeout=self._timeout_s, verify=self._verify_ssl)
        self._check(resp)
        return _json_or_empty(resp)

    def _post_json(self, path, body, extra_headers=None):
        resp = self._sess().post(self.url + path, json=body,
                                 headers=self._headers(extra_headers),
                                 timeout=self._timeout_s, verify=self._verify_ssl)
        self._check(resp)
        return _json_or_empty(resp)

    # ----------------------------------------------------------------------- #
    # server info
    # ----------------------------------------------------------------------- #
    def health(self):
        return self._get_json("/health")

    def lock_status(self):
        return self._get_json("/lock/status")

    # ----------------------------------------------------------------------- #
    # locks (server-side blocking acquire with re-issue; mirrors slm_client.m)
    # ----------------------------------------------------------------------- #
    SERVER_CAP_S = 30.0   # server hard-caps block_timeout_s at 30 s regardless of what we send.

    def acquire_lock(self, device="all", description="", timeout_s=60.0,
                     block_timeout=30.0, server_block=True, mode="standard",
                     clock=time.monotonic, sleep=time.sleep):
        """Acquire ``device`` lock, blocking up to ``block_timeout`` s; raise on failure.

        ``timeout_s`` is the lock LEASE (heartbeat-expiry); ``block_timeout`` is the WAIT budget
        to acquire. With ``server_block`` the server parks the request on its condvar (up to its
        30 s cap); if our budget exceeds that we re-issue until the deadline. Raises
        :class:`SlmHTTPError` (423 contention / 408 block-timeout) once ``block_timeout`` elapses.
        """
        deadline = clock() + float(block_timeout)
        while True:
            remaining = max(0.0, deadline - clock())
            body = {"client_id": self.client_id, "device": str(device),
                    "description": str(description), "timeout_s": float(timeout_s),
                    "mode": str(mode)}
            if server_block and remaining > 0:
                body["block"] = True
                body["block_timeout_s"] = min(remaining, self.SERVER_CAP_S)
            try:
                return self._post_json("/lock/acquire", body)
            except SlmHTTPError as err:
                retriable = err.status in (408, 423)
                if not retriable or clock() >= deadline:
                    raise
                if not server_block:
                    sleep(0.2)   # client-side poll back-off when not using server blocking

    def release_lock(self, device="all"):
        return self._post_json("/lock/release",
                               {"client_id": self.client_id, "device": str(device)})

    def heartbeat(self, device="all"):
        return self._post_json("/lock/heartbeat",
                               {"client_id": self.client_id, "device": str(device)})

    # ----------------------------------------------------------------------- #
    # rearrangement setup
    # ----------------------------------------------------------------------- #
    def setup_rearrangement(self, **kwargs):
        """One-time / per-shot rearrangement config. Mirrors ``slm_client.m::setup_rearrangement``
        body shaping: grids pass through, ``initial_phase``/``final_phase`` map to
        ``*_filepath``, ``extras`` (a dict) is merged in, everything else forwarded verbatim.
        Only the keys you pass are sent -- the server keeps cached values for the rest."""
        body = _build_setup_body(kwargs)
        return self._post_json("/slm/setup_rearrangement", body)

    def reload_rearrange(self):
        """Write the cached initial phase to the SLM + prime the GPU path (start-of-shot reset).
        Caller MUST already hold the ``slm`` lock."""
        return self._post_json("/slm/reload_rearrange", {})

    # ----------------------------------------------------------------------- #
    # per-shot rearrange / results
    # ----------------------------------------------------------------------- #
    def rearrange(self, bits, target_bits=None, extras=None, scan_id=None, seq_id=None):
        """Atomic-rearrangement update from a load-pattern bitstring (model inference + paced
        SLM writes). Pre-conditions: caller holds the ``slm`` lock + setup_rearrangement done."""
        body = {"bits": _encode_bits(bits)}
        if target_bits is not None:
            body["target_bits"] = _as_float_list(target_bits)
        if extras:
            body.update(dict(extras))
        _stamp_runid(body, scan_id, seq_id)
        return self._post_json("/slm/rearrange", body)

    def update_rearrange(self, results, scan_id=None, seq_id=None):
        """Post-rearrangement detection results -> ``/slm/results`` (ledger row for the shot)."""
        body = {"results": _encode_bits(results)}
        _stamp_runid(body, scan_id, seq_id)
        return self._post_json("/slm/results", body)

    def cancel_last_shot(self, scan_id=None, seq_id=None):
        """Drop the most-recent diag ledger row for ``(scan_id[, seq_id])`` -- abort-tail cleanup
        when rearrange() committed server-side but the lab-side store_imgs failed afterwards."""
        body = {}
        _stamp_runid(body, scan_id, seq_id)
        return self._post_json("/slm/cancel_last", body)

    # ----------------------------------------------------------------------- #
    # loading phase (scan-long SLM session)
    # ----------------------------------------------------------------------- #
    def write_loading_phase(self, phase_path, loading_zernike=None, name=None,
                            legacy_zerniked=False, baked_zernike=None, block_timeout=0.0):
        """Write ``base + zernike(loading_zernike)`` to the SLM WITHOUT re-extracting positions.
        Caller MUST hold the ``slm`` lock (e.g. the scan-long hold)."""
        body = {"phase_filepath": str(phase_path),
                "legacy_zerniked": bool(legacy_zerniked)}
        if loading_zernike is not None and len(_as_float_list(loading_zernike)) > 0:
            body["loading_zernike"] = _as_float_list(loading_zernike)
        if baked_zernike is not None and len(_as_float_list(baked_zernike)) > 0:
            body["baked_zernike"] = _as_float_list(baked_zernike)
        if name:
            body["name"] = str(name)
        extra = None
        if block_timeout and block_timeout > 0:
            extra = {"X-Block-Timeout": str(block_timeout)}
        return self._post_json("/slm/write_loading_phase", body, extra_headers=extra)


# =========================================================================== #
# body-shaping helpers (mirror slm_client.m)
# =========================================================================== #
def _build_setup_body(kwargs):
    body = {}
    for key, val in kwargs.items():
        if val is None:
            continue
        if key in ("init_grid", "target_grid"):
            if isinstance(val, str):
                body[key] = val
            else:
                body[key] = _as_grid(val)             # Nx2 numeric
        elif key in ("initial_phase", "final_phase"):
            s = str(val)
            if s:
                body[key + "_filepath"] = s
        elif key == "extras":
            if isinstance(val, dict):
                for ek, ev in val.items():
                    if ev is not None:
                        body[ek] = ev
        elif key == "target_bits":
            body["target_bits"] = _as_float_list(val)
        else:
            body[key] = val                            # scalars / known params verbatim
    return body


def _encode_bits(bits):
    """Encode bits the way slm_client.m does: '0'/'1' string passthrough, logical/0-1 vectors to
    a string, an index list (any value > 1) as a numeric list, a dict passed through verbatim."""
    if isinstance(bits, str):
        return bits
    if isinstance(bits, dict):
        return bits
    seq = _as_list(bits)
    if not seq:
        return ""
    nums = [float(x) for x in seq]
    if max(nums) > 1:
        return [int(x) for x in nums]                  # index list -- server detects
    return "".join("1" if x else "0" for x in nums)


def _stamp_runid(body, scan_id, seq_id):
    # scan_id as a STRING (14-digit ids exceed the JS Number safe range).
    if scan_id is not None:
        body["scan_id"] = str(scan_id)
    if seq_id is not None:
        body["seq_id"] = int(seq_id)


def _as_list(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.ravel().tolist()
    except Exception:  # noqa: BLE001 - numpy absent
        pass
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _as_float_list(v):
    return [float(x) for x in _as_list(v)]


def _as_grid(v):
    """Coerce an Nx2 grid to a list of [r, c] pairs (server re-sorts row-major itself)."""
    try:
        import numpy as np
        a = np.asarray(v, dtype=float).reshape(-1, 2)
        return a.tolist()
    except Exception:  # noqa: BLE001
        flat = _as_float_list(v)
        return [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]


def _json_or_empty(resp):
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - empty / non-JSON body
        return {}
    return body if body is not None else {}


# =========================================================================== #
# per-url singleton (mirror MATLAB SLMClient.get(url, pw).client())
# =========================================================================== #
_CLIENTS = {}


def get_client(url=None, password=None):
    """Return the cached :class:`SlmClient` for ``url`` (one ``client_id`` per server).

    Url/password fall back to the env (``YB_SLM_URL`` / ``YB_SLM_PASSWORD``) then the module
    defaults -- mirroring RearrangeCommSeq.m's resolve_slm_client. The same instance is reused by
    the scan-long session and every per-shot callback so they share a lock owner."""
    if url is None:
        url = os.environ.get("YB_SLM_URL", DEFAULT_URL)
    if password is None:
        password = os.environ.get("YB_SLM_PASSWORD", DEFAULT_PASSWORD)
    key = str(url).rstrip("/")
    c = _CLIENTS.get(key)
    if c is None:
        c = SlmClient(url=key, password=password)
        _CLIENTS[key] = c
    return c
