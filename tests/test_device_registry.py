"""Phase-5 devices registry: the name -> factory lookup (devices.register/create/available).

NO-HARDWARE: orca is constructed with an injected fake backend (``cam=...``) so no DCAM /
pylablib is touched; the NI entries resolve to a namespace / module (no device driven).
"""
import pytest

import devices

pytestmark = pytest.mark.no_hardware


def test_available_lists_all_registered_families():
    # The complete current set; adding a new device family updates this deliberately.
    assert devices.available() == ["nidaq_dc", "nidaq_seq", "orca", "qick_awg",
                                   "sigilent_awg", "slm"]


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


def test_create_sigilent_awg_returns_a_connection_handle():
    # Construction only -- connect() (the lazy pyvisa import) is NOT called, so no VISA backend.
    conn = devices.create("sigilent_awg", "USB0::TEST::INSTR", "C1")
    assert isinstance(conn, devices.AWGConnection)
    assert conn.dev is None                              # not connected
    assert conn.channel == "C1"


def test_create_qick_awg_returns_a_client_handle():
    # Construction only -- connect() (the socket) is NOT called, so no network.
    client = devices.create("qick_awg")
    assert isinstance(client, devices.FPGAAWGClient)
    assert not client.is_connected


def test_create_unknown_device_raises():
    with pytest.raises(KeyError):
        devices.create("no_such_device")
