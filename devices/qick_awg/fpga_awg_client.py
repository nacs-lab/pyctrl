"""fpga_awg_client.py -- the QICK FPGA_AWG socket client.

The QICK RF AWG (RFSoC4x2 -- Rydberg/STIRAP/microwave) is a **separate board** from the libnacs
``zynq`` FPGA1: it is NOT in ``config.yml``, is armed **out-of-band**, and is **TTL-triggered**
(``TTLQickTrig = FPGA1/TTL14``). It runs its own socket server (``run_server.py``) with a tiny
length-prefixed protocol; the upstream Python client is ``FPGA_AWG_client.py`` from
``cliulinnaeus/FPGA_AWG@rfsoc4x2``. matlab_new's ``FPGA_AWG_Client.m`` / ``FPGA_ABS_Client.m`` is a
1:1 ``tcpclient`` reimplementation of that client (identical method set + wire format), so this
module reproduces that exact wire protocol in pure Python (stdlib ``socket`` only -- no vendor dep).

⚠ QICK output is **out-of-band** -- it is NOT in the serialized seq byte blob, so THE ONE RULE
(byte-equality) does not apply here. This is a runtime device driver, like the Orca camera or the
Siglent AWG recall path.

Wire protocol (verified against ``FPGA_ABS_Client.m``):
  * ints  -- 4-byte **big-endian** uint32.
  * strings -- ``send_int(utf8_len)`` then the UTF-8 bytes.
  * files -- ``send_int(size)``, ``send_string(filename)``, then the raw bytes.
  * every command -- ``send_string(CMD)``, then its args, then read one ack string back.

Two classes mirror the MATLAB split: :class:`FPGAABSClient` (the TCP transport) and
:class:`FPGAAWGClient` (the AWG command verbs).
"""
import logging
import socket
import struct

logger = logging.getLogger(__name__)

DEFAULT_HOST = "192.168.0.72"   # the QICK server (RamseySeq/PushoutSurvivalAWGSeq server_pre_run)
DEFAULT_PORT = 1234


class FPGAABSClient:
    """TCP transport for the QICK server -- port of ``FPGA_ABS_Client.m`` (big-endian framing).

    ``socket_factory`` is an injectable seam ``(address, timeout) -> connected socket`` (defaults to
    :func:`socket.create_connection`) so the transport framing is unit-testable with a fake socket
    and NO network (mirrors AWGConnection's ``connection_factory``).
    """

    BUFFER_SIZE = 4096
    TIMEOUT = 30.0

    def __init__(self, *, socket_factory=None):
        self._sock = None
        self._socket_factory = socket_factory or socket.create_connection
        self.host = None
        self.port = None

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #
    def connect(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        """Open the TCP connection to the QICK server (NEEDS-HARDWARE)."""
        self._sock = self._socket_factory((host, int(port)), timeout=self.TIMEOUT)
        self.host = host
        self.port = int(port)
        logger.info("FPGA_AWG connected to %s:%s", host, port)
        return self

    def disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:                       # noqa: BLE001
                pass
            logger.info("FPGA_AWG disconnected from %s:%s", self.host, self.port)
        self._sock = None

    @property
    def is_connected(self):
        return self._sock is not None

    # ------------------------------------------------------------------ #
    # low-level send/receive (big-endian, length-prefixed)
    # ------------------------------------------------------------------ #
    def send_int(self, n):
        """Send a 4-byte big-endian uint32 (== MATLAB typecast+swapbytes+fliplr)."""
        self._require_socket()
        self._sock.sendall(struct.pack(">I", int(n) & 0xFFFFFFFF))

    def send_string(self, s):
        self._require_socket()
        data = s.encode("utf-8")
        self.send_int(len(data))
        self._sock.sendall(data)

    def send_payload(self, data, filename):
        """Send a file payload (``send_file`` analog): size, filename, then the raw bytes.

        Takes the bytes directly instead of a path so callers can stream an in-memory JSON cfg
        without touching disk (the MATLAB builders wrote a temp .json first; we skip that).
        """
        self._require_socket()
        self.send_int(len(data))
        self.send_string(filename)
        self._sock.sendall(bytes(data))

    def send_file(self, path):
        """Faithful ``send_file``: stream a file from disk (size, path, bytes)."""
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_payload(data, path)

    def receive_int(self):
        return struct.unpack(">I", self._recv_exactly(4))[0]

    def receive_string(self):
        n = self.receive_int()
        return self._recv_exactly(n).decode("utf-8")

    def receive_server_ack(self):
        """Read one ack/result string from the server (returned so GET_* verbs are useful)."""
        return self.receive_string()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _require_socket(self):
        if self._sock is None:
            raise RuntimeError("FPGA_AWG client not connected.")

    def _recv_exactly(self, n):
        self._require_socket()
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(min(self.BUFFER_SIZE, n - len(buf)))
            if not chunk:
                raise ConnectionError("FPGA_AWG socket closed mid-read")
            buf += chunk
        return bytes(buf)


class FPGAAWGClient(FPGAABSClient):
    """The QICK AWG command verbs -- port of ``FPGA_AWG_Client.m`` (1:1 with the upstream client).

    Upload verbs accept **bytes** (the JSON cfg built by :mod:`simple_pulse`) plus the server-side
    ``name``; a ``filename`` is sent only because the protocol carries one (defaults to
    ``<name>.json``). ``upload_*_file`` variants stream a real file for parity with the MATLAB /
    upstream path. Every verb returns the server ack string.
    """

    # ---- uploads -------------------------------------------------------- #
    def upload_waveform_cfg(self, data, name, filename=None):
        return self._upload("UPLOAD_WAVEFORM_CFG", data, name, filename)

    def upload_envelope_data(self, data, name, filename=None):
        return self._upload("UPLOAD_ENVELOPE_DATA", data, name, filename)

    def upload_program(self, data, name, filename=None):
        return self._upload("UPLOAD_PROGRAM", data, name, filename)

    def upload_waveform_cfg_file(self, path, name):
        return self._upload_file("UPLOAD_WAVEFORM_CFG", path, name)

    def upload_envelope_data_file(self, path, name):
        return self._upload_file("UPLOAD_ENVELOPE_DATA", path, name)

    def upload_program_file(self, path, name):
        return self._upload_file("UPLOAD_PROGRAM", path, name)

    # ---- deletes -------------------------------------------------------- #
    def delete_waveform_cfg(self, name):
        return self._cmd_named("DELETE_WAVEFORM_CFG", name)

    def delete_all_waveform_cfg(self):
        return self._cmd("DELETE_ALL_WAVEFORM_CFG")

    def delete_envelope_data(self, name):
        return self._cmd_named("DELETE_ENVELOPE_DATA", name)

    def delete_all_envelope_data(self):
        return self._cmd("DELETE_ALL_ENVELOPE_DATA")

    def delete_program(self, name):
        return self._cmd_named("DELETE_PROGRAM", name)

    def delete_all_programs(self):
        return self._cmd("DELETE_ALL_PROGRAMS")

    # ---- queries -------------------------------------------------------- #
    def get_waveform_lst(self):
        return self._cmd("GET_WAVEFORM_LIST")

    def get_envelope_lst(self):
        return self._cmd("GET_ENVELOPE_LIST")

    def get_program_lst(self):
        return self._cmd("GET_PROGRAM_LIST")

    def get_state(self):
        return self._cmd("GET_STATE")

    # ---- run control ---------------------------------------------------- #
    def set_trigger_mode(self, trig_mode):
        return self._cmd_named("SET_TRIGGER_MODE", trig_mode)

    def start_program(self, name):
        return self._cmd_named("START_PROGRAM", name)

    def stop_program(self):
        return self._cmd("STOP_PROGRAM")

    # ------------------------------------------------------------------ #
    # command framing
    # ------------------------------------------------------------------ #
    def _cmd(self, verb):
        self.send_string(verb)
        return self.receive_server_ack()

    def _cmd_named(self, verb, name):
        self.send_string(verb)
        self.send_string(name)
        return self.receive_server_ack()

    def _upload(self, verb, data, name, filename):
        self.send_string(verb)
        self.send_string(name)
        self.send_payload(data, filename or (name + ".json"))
        return self.receive_server_ack()

    def _upload_file(self, verb, path, name):
        self.send_string(verb)
        self.send_string(name)
        self.send_file(path)
        return self.receive_server_ack()
