"""Phase-5 devices registry: the name -> factory lookup (devices.register/create/available).

NO-HARDWARE: orca is constructed with an injected fake backend (``cam=...``) so no DCAM /
pylablib is touched; the NI entries resolve to a namespace / module (no device driven).
"""
import pytest

import devices

pytestmark = pytest.mark.no_hardware


def test_available_lists_all_registered_families():
    # The complete current set; adding a new device family updates this deliberately.
    assert devices.available() == ["nidaq_dc", "nidaq_seq", "orca", "slm"]


def test_create_orca_returns_an_open_camera_handle():
    cam = devices.create("orca", cam=object())          # inject a fake backend -> no pylablib
    assert isinstance(cam, devices.OrcaCamera)
    assert cam.connected


def test_create_nidaq_seq_is_the_runner_namespace():
    # NiDAQRunner is a process-global namespace, not instantiated.
    assert devices.create("nidaq_seq") is devices.NiDAQRunner


def test_create_nidaq_dc_exposes_set_and_read():
    dc = devices.create("nidaq_dc")
    assert callable(dc.set_channel) and callable(dc.read_channel)


def test_create_slm_returns_the_shared_client():
    client = devices.create("slm")                  # construction only -- network-free
    assert isinstance(client, devices.SlmClient)
    assert devices.create("slm") is client          # cached per-url singleton


def test_create_unknown_device_raises():
    with pytest.raises(KeyError):
        devices.create("no_such_device")
