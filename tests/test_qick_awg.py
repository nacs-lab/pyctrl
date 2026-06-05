"""No-hardware tests for the QICK FPGA_AWG port (devices/qick_awg).

Covers the three pieces without touching a socket / the RFSoC board:
  * FPGAAWGClient transport -- big-endian int framing, length-prefixed strings, file payload,
    command verb framing -- via a FAKE socket that records bytes + replays acks.
  * simple_pulse builders -- pulse cfg fields, compile_chn (1/2 channels, loops, name_map), guards.
  * FPGAAWGManager -- batch-upload dedup + pulse namespacing + arm-first, per-shot pick
    (switch on change, skip on no-change), unknown key warns, cleanup -- via a FAKE client.
"""
import json
import struct

import pytest

from devices.qick_awg import (FPGAAWGClient, FPGAAWGManager, QickProgram, Loop,
                              compile_chn, loop, simple_prog_cfg, simple_pulse_cfg)
from devices.qick_awg.simple_pulse import render_tokens

pytestmark = pytest.mark.no_hardware


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeSocket:
    """Records everything sent; serves queued bytes for recv (preloaded with acks)."""

    def __init__(self):
        self.sent = bytearray()
        self.recv_buf = bytearray()
        self.closed = False

    # --- the bits the client uses ---
    def sendall(self, data):
        self.sent += bytes(data)

    def recv(self, n):
        if not self.recv_buf:
            return b""
        chunk = bytes(self.recv_buf[:n])
        del self.recv_buf[:n]
        return chunk

    def close(self):
        self.closed = True

    # --- test helpers ---
    def queue_ack(self, text="OK"):
        """Append a length-prefixed (big-endian) ack string the client will read back."""
        data = text.encode("utf-8")
        self.recv_buf += struct.pack(">I", len(data)) + data


def _connect_fake_client(sock=None):
    sock = sock or FakeSocket()
    client = FPGAAWGClient(socket_factory=lambda addr, timeout: sock)
    client.connect("host", 1234)
    return client, sock


class FakeClient:
    """Records the manager's command calls -- no socket. Mirrors FPGAAWGClient's surface."""

    def __init__(self):
        self.connected_to = None
        self.calls = []                 # ordered (verb, *args)
        self.uploaded_waveforms = []    # (name, cfg_dict)
        self.uploaded_programs = []     # (name, cfg_dict)
        self.started = []               # start_program names, in order
        self.stops = 0
        self.disconnected = 0

    def connect(self, host, port):
        self.connected_to = (host, port)
        self.calls.append(("connect", host, port))

    def delete_all_envelope_data(self):
        self.calls.append(("delete_all_envelope_data",))

    def delete_all_waveform_cfg(self):
        self.calls.append(("delete_all_waveform_cfg",))

    def delete_all_programs(self):
        self.calls.append(("delete_all_programs",))

    def upload_waveform_cfg(self, data, name, filename=None):
        self.uploaded_waveforms.append((name, json.loads(bytes(data).decode("utf-8"))))
        self.calls.append(("upload_waveform_cfg", name))

    def upload_program(self, data, name, filename=None):
        self.uploaded_programs.append((name, json.loads(bytes(data).decode("utf-8"))))
        self.calls.append(("upload_program", name))

    def set_trigger_mode(self, mode):
        self.calls.append(("set_trigger_mode", mode))

    def start_program(self, name):
        self.started.append(name)
        self.calls.append(("start_program", name))

    def stop_program(self):
        self.stops += 1
        self.calls.append(("stop_program",))

    def disconnect(self):
        self.disconnected += 1
        self.calls.append(("disconnect",))


@pytest.fixture(autouse=True)
def _reset_manager():
    FPGAAWGManager._state = {}
    yield
    FPGAAWGManager._state = {}


# --------------------------------------------------------------------------- #
# transport framing
# --------------------------------------------------------------------------- #
def test_send_int_is_4byte_big_endian():
    client, sock = _connect_fake_client()
    client.send_int(258)                              # 0x00000102
    assert bytes(sock.sent) == b"\x00\x00\x01\x02"


def test_send_string_is_length_prefixed_utf8():
    client, sock = _connect_fake_client()
    client.send_string("AB")
    assert bytes(sock.sent) == struct.pack(">I", 2) + b"AB"


def test_receive_string_roundtrip():
    sock = FakeSocket()
    sock.queue_ack("hello")
    client, _ = _connect_fake_client(sock)
    assert client.receive_string() == "hello"


def test_send_payload_frames_size_filename_then_bytes():
    client, sock = _connect_fake_client()
    payload = b'{"x":1}'
    client.send_payload(payload, "f.json")
    expected = (struct.pack(">I", len(payload))          # size
                + struct.pack(">I", len("f.json")) + b"f.json"  # filename (length-prefixed string)
                + payload)                                # raw bytes
    assert bytes(sock.sent) == expected


def test_upload_program_verb_framing_and_ack():
    sock = FakeSocket()
    sock.queue_ack("done")
    client, _ = _connect_fake_client(sock)
    sock.sent.clear()                                  # ignore connect (no bytes anyway)
    ack = client.upload_program(b'{"name":"p"}', "p000")
    assert ack == "done"
    # Verb string, then name string, then payload (size+filename+bytes).
    body = b'{"name":"p"}'
    expected = (struct.pack(">I", len("UPLOAD_PROGRAM")) + b"UPLOAD_PROGRAM"
                + struct.pack(">I", len("p000")) + b"p000"
                + struct.pack(">I", len(body))
                + struct.pack(">I", len("p000.json")) + b"p000.json"
                + body)
    assert bytes(sock.sent) == expected


def test_start_stop_program_verbs():
    sock = FakeSocket()
    sock.queue_ack(); sock.queue_ack()
    client, _ = _connect_fake_client(sock)
    sock.sent.clear()
    client.start_program("prog001")
    client.stop_program()
    # start_program sends verb + name; stop_program sends verb only.
    expected = (struct.pack(">I", len("START_PROGRAM")) + b"START_PROGRAM"
                + struct.pack(">I", len("prog001")) + b"prog001"
                + struct.pack(">I", len("STOP_PROGRAM")) + b"STOP_PROGRAM")
    assert bytes(sock.sent) == expected


def test_disconnect_closes_socket():
    client, sock = _connect_fake_client()
    client.disconnect()
    assert sock.closed
    assert not client.is_connected


def test_send_without_connect_raises():
    client = FPGAAWGClient()
    with pytest.raises(RuntimeError):
        client.send_string("x")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def test_simple_pulse_cfg_fields():
    cfg = simple_pulse_cfg("Pi2", freq=2.4, gain=3000, length=104.0, phase=90.0)
    assert cfg == {
        "name": "Pi2", "style": "const", "freq": 2.4, "gain": 3000,
        "phase": 90.0, "length": 104.0, "mode": "oneshot",
    }


def test_compile_chn_single_channel_with_loop():
    # Mirrors RamseySeq: {'Pi2', loop(10,[Wait]), 'Pi', loop(10,[Wait]), 'Pi2_Phase'}
    ch = compile_chn([["Pi2", loop(10, ["Wait"]), "Pi", loop(10, ["Wait"]), "Pi2_Phase"]])
    assert ch == {"ch0": "[Pi2,loop(10,[Wait]),Pi,loop(10,[Wait]),Pi2_Phase]"}


def test_compile_chn_two_channels():
    ch = compile_chn([["A"], ["B", "C"]])
    assert ch == {"ch0": "[A]", "ch1": "[B,C]"}


def test_compile_chn_rejects_three_channels():
    with pytest.raises(ValueError):
        compile_chn([["A"], ["B"], ["C"]])


def test_render_tokens_applies_name_map_through_loops():
    toks = ["Pi2", loop(5, ["Wait"])]
    out = render_tokens(toks, name_map={"Pi2": "p007_Pi2", "Wait": "p007_Wait"})
    assert out == "[p007_Pi2,loop(5,[p007_Wait])]"


def test_simple_prog_cfg_shape():
    assert simple_prog_cfg("f_Rabi", {"ch0": "[A]"}) == {
        "name": "f_Rabi", "prog_structure": {"ch0": "[A]"}}


# --------------------------------------------------------------------------- #
# manager: batch upload + dedup + namespacing
# --------------------------------------------------------------------------- #
def _prog(key, freq):
    return QickProgram(
        key=key,
        pulses={"Pi2": simple_pulse_cfg("Pi2", freq=freq, gain=3000, length=104.0)},
        channels=[["Pi2"]],
    )


def test_setup_uploads_one_program_per_unique_key_and_arms_first():
    clients = []

    def factory():
        c = FakeClient()
        clients.append(c)
        return c

    # 5 points, 3 distinct keys (a, b, a, c, b).
    progs = [_prog(k, f) for k, f in (("a", 2.0), ("b", 2.1), ("a", 2.0), ("c", 2.2), ("b", 2.1))]
    FPGAAWGManager.setup(progs, client_factory=factory)

    assert len(clients) == 1
    client = clients[0]
    assert client.connected_to == ("192.168.0.72", 1234)
    # 3 unique programs uploaded; trigger mode set; first key armed.
    assert [n for n, _ in client.uploaded_programs] == ["prog000", "prog001", "prog002"]
    assert ("set_trigger_mode", "external") in client.calls
    assert client.started == ["prog000"]
    assert FPGAAWGManager._state["last_key"] == "a"
    assert FPGAAWGManager.is_active()


def test_setup_namespaces_pulses_per_program():
    client = FakeClient()
    progs = [_prog("a", 2.0), _prog("b", 2.1)]
    FPGAAWGManager.setup(progs, client_factory=lambda: client)

    # Pulse names are namespaced by program index so they don't clash on the server.
    names = [n for n, _ in client.uploaded_waveforms]
    assert names == ["p000_Pi2", "p001_Pi2"]
    # The uploaded pulse cfg carries the namespaced name...
    assert client.uploaded_waveforms[0][1]["name"] == "p000_Pi2"
    # ...and the program's prog_structure references it.
    prog0 = dict(client.uploaded_programs)["prog000"]
    assert prog0["prog_structure"] == {"ch0": "[p000_Pi2]"}


def test_setup_deletes_all_before_upload():
    client = FakeClient()
    FPGAAWGManager.setup([_prog("a", 2.0)], client_factory=lambda: client)
    verbs = [c[0] for c in client.calls]
    # connect, then all three delete_alls, before the first upload.
    assert verbs.index("delete_all_programs") < verbs.index("upload_waveform_cfg")
    for v in ("delete_all_envelope_data", "delete_all_waveform_cfg", "delete_all_programs"):
        assert v in verbs


# --------------------------------------------------------------------------- #
# manager: per-shot pulse picking
# --------------------------------------------------------------------------- #
def test_recall_switches_on_change_and_skips_on_no_change():
    client = FakeClient()
    progs = [_prog("a", 2.0), _prog("b", 2.1), _prog("c", 2.2)]
    FPGAAWGManager.setup(progs, client_factory=lambda: client)
    # After setup: last_key = "a", started = ["prog000"].
    n_started = len(client.started)

    FPGAAWGManager.recall_for_seq("a")                 # unchanged -> skip
    assert len(client.started) == n_started

    FPGAAWGManager.recall_for_seq("b")                 # switch -> stop + start prog001
    assert client.started[-1] == "prog001"
    assert FPGAAWGManager._state["last_key"] == "b"

    FPGAAWGManager.recall_for_seq("b")                 # same -> skip
    assert client.started[-1] == "prog001"

    FPGAAWGManager.recall_for_seq("c")                 # switch -> prog002
    assert client.started[-1] == "prog002"
    # Every switch was preceded by a stop_program (2 switches).
    assert client.stops == 2


def test_recall_unknown_key_warns_and_does_not_switch():
    client = FakeClient()
    FPGAAWGManager.setup([_prog("a", 2.0)], client_factory=lambda: client)
    started_before = len(client.started)
    FPGAAWGManager.recall_for_seq("nope")              # never uploaded
    assert len(client.started) == started_before
    assert FPGAAWGManager._state["last_key"] == "a"    # unchanged


def test_recall_is_noop_without_setup():
    FPGAAWGManager._state = {}
    FPGAAWGManager.recall_for_seq("a")                 # must not raise


def test_cleanup_stops_disconnects_and_clears_state():
    client = FakeClient()
    FPGAAWGManager.setup([_prog("a", 2.0)], client_factory=lambda: client)
    FPGAAWGManager.cleanup()
    assert client.stops >= 1
    assert client.disconnected == 1
    assert FPGAAWGManager._state == {}
    assert not FPGAAWGManager.is_active()
