"""slm -- SLM / rearrangement server (pyctrl client side).

A pure HTTP/JSON client to the remote SLM server (no slmnet/torch here) plus the
scan-long SLM hardware-lock session (port of SlmScanSession.m). Import-safe with no
server reachable: ``requests`` is imported lazily, so network calls happen only when
a client method is invoked, not at import or construction.
"""
from .slm_client import (DEFAULT_PASSWORD, DEFAULT_URL, SlmClient, SlmHTTPError,
                         get_client)
from .slm_scan_session import SlmLockUnavailable, SlmScanSession
from ..device_registry import register

__all__ = ["DEFAULT_PASSWORD", "DEFAULT_URL", "SlmClient", "SlmHTTPError",
           "get_client", "SlmLockUnavailable", "SlmScanSession"]

# Registry: create("slm", url, pw) -> the shared per-url client singleton (mirrors
# MATLAB SLMClient.get(url, pw).client()); construction is network-free.
register("slm")(get_client)
