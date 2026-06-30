"""No-hardware tests for the Siglent SDG6X AWG port (devices/sigilent_awg).

Covers the three pieces without touching VISA/USB:
  * gaussian_pulse_waveform -- big-endian int16 bytes, length, DDS freq, amplitude scaling.
  * AWGConnection.build_waveform_cmd -- IEEE-488.2 WVDT framing (pure; no device handle).
  * AWGManager -- batch-upload dedup, set-amplitude-once, per-shot active-waveform switching
    (resend on change, skip on no-change), cleanup -- all via an injected FAKE connection.
"""
import struct

import numpy as np
import pytest

from devices.sigilent_awg import (AWGConnection, AWGManager, WAVEFORM_FIELDS,
                                   gaussian_pulse_waveform)

pytestmark = pytest.mark.no_hardware


_DEFAULTS = {
    "AWG556": {
        "resource_address": "USB0::TEST::AWG556::INSTR",
        "channel": "C1", "max_amplitude_vpp": 11, "num_points": 1000,
        "pulse_width_us": 4, "carrier_freq_MHz": 130.78, "steepness": 3.5,
        "amplitude_scale": 1.0,
    },
}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeConn:
    """Records SCPI/waveform calls; real build_waveform_cmd framing is reused from AWGConnection."""
    def __init__(self, resource, channel):
        self.resource = resource
        self.channel = channel
        self.connected = False
        self.sent = []            # list of WVDT cmd bytes (send_waveform)
        self.amplitudes = []      # set_amplitude calls
        self.burst_configured = 0
        self.output_enabled = 0
        self.output_disabled = 0
        self.disconnected = 0

    def connect(self):
        self.connected = True
        return "FAKE,SDG6X,0,1"

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz):
        # reuse the real framing so the test also exercises it through the manager
        return AWGConnection.build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz)

    def send_waveform(self, cmd):
        self.sent.append(cmd)

    def set_amplitude(self, amp):
        self.amplitudes.append(amp)

    def configure_burst(self):
        self.burst_configured += 1

    def enable_output(self):
        self.output_enabled += 1

    def disable_output(self):
        self.output_disabled += 1

    def disconnect(self):
        self.disconnected += 1


class FakeConnARWV(FakeConn):
    """ARWV-capable fake (fw >= 38R3): arwv_recall=True + named store / set_arb_mode / recall_by_name."""
    def __init__(self, resource, channel):
        super().__init__(resource, channel)
        self.arwv_recall = True
        self.stored = []          # named WVDT cmds via store_waveform
        self.recalled = []        # names via recall_by_name
        self.arb_mode = 0

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz, name="active"):
        return AWGConnection.build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz, name=name)

    def set_arb_mode(self):
        self.arb_mode += 1

    def store_waveform(self, cmd):
        self.stored.append(cmd)

    def recall_by_name(self, name):
        self.recalled.append(name)


class FakeScanGroup:
    """Minimal ScanGroup stand-in: getseq(n) (1-based) returns the n-th param dict."""
    def __init__(self, seqs):
        self._seqs = seqs

    def nseq(self):
        return len(self._seqs)

    def getseq(self, n):
        return self._seqs[n - 1]


@pytest.fixture(autouse=True)
def _reset_manager():
    AWGManager._state = {}
    yield
    AWGManager._state = {}


# --------------------------------------------------------------------------- #
# gaussian_pulse_waveform
# --------------------------------------------------------------------------- #
def test_waveform_is_big_endian_int16_of_expected_length():
    p = dict(_DEFAULTS["AWG556"], num_points=1000)
    data, info = gaussian_pulse_waveform(p)
    assert isinstance(data, bytes)
    assert len(data) == 2 * 1000                      # int16 -> 2 bytes/sample
    assert info["num_points"] == 1000

    be = np.frombuffer(data, dtype=">i2")             # decode as big-endian
    le = np.frombuffer(data, dtype="<i2")             # ... vs little-endian
    assert be.shape == (1000,)
    # The waveform is not endian-symmetric, so the two interpretations differ -> proves byte order.
    assert not np.array_equal(be, le)
    # Peak normalized to full int16 scale (amplitude_scale=1.0).
    assert np.max(np.abs(be)) == 32767


def test_waveform_dds_freq_and_amplitude_scale():
    # DDS playback frequency = 1e6 / pulse_width_us.
    _, info = gaussian_pulse_waveform(dict(_DEFAULTS["AWG556"], pulse_width_us=4))
    assert info["freq_hz"] == pytest.approx(250000.0)

    # Half amplitude_scale -> peak code halved (within rounding).
    data_half, _ = gaussian_pulse_waveform(dict(_DEFAULTS["AWG556"], amplitude_scale=0.5))
    peak_half = np.max(np.abs(np.frombuffer(data_half, dtype=">i2")))
    assert peak_half == pytest.approx(16384, abs=2)


# --------------------------------------------------------------------------- #
# AWGConnection.build_waveform_cmd (pure framing)
# --------------------------------------------------------------------------- #
def test_build_waveform_cmd_ieee_header_and_prefix():
    conn = AWGConnection("USB0::TEST::INSTR", "C1")    # no connect() -> no VISA
    binary = struct.pack(">5h", 1, 2, 3, 4, 5)         # 10 bytes
    cmd = conn.build_waveform_cmd(binary, 11, 250000.0)
    assert isinstance(cmd, bytes)
    text = cmd[:-10].decode("ascii")
    # IEEE block header for 10 bytes: '#' + len("10")=2 + "10" -> '#210'
    assert text.endswith("WAVEDATA,#210")
    assert text.startswith("C1:WVDT WVNM,active,WVTP,USER,AMPL,11,OFST,0,FREQ,250000")
    assert cmd.endswith(binary)


# --------------------------------------------------------------------------- #
# AWGManager batch upload + dedup
# --------------------------------------------------------------------------- #
def _seq_with_freq(freq_mhz):
    return {"AWG": {"AWG556": {"carrier_freq_MHz": freq_mhz}}}


def test_setup_uploads_one_waveform_per_unique_combo():
    conns = []

    def factory(resource, channel):
        c = FakeConn(resource, channel)
        conns.append(c)
        return c

    # 5 sequences, but only 3 DISTINCT carrier freqs (130, 131, 130, 132, 131).
    seqs = [_seq_with_freq(f) for f in (130.0, 131.0, 130.0, 132.0, 131.0)]
    AWGManager.setup("AWG556", FakeScanGroup(seqs),
                     consts=_DEFAULTS, connection_factory=factory)

    assert len(conns) == 1
    conn = conns[0]
    assert conn.connected
    # 3 unique waveforms cached.
    assert len(AWGManager._state["AWG556"]["cmd_map"]) == 3
    # First waveform sent to init output; amplitude set exactly ONCE; burst armed; output on.
    assert len(conn.sent) == 1
    assert conn.amplitudes == [11]
    assert conn.burst_configured == 1
    assert conn.output_enabled == 1
    assert AWGManager.active_awgs() == ["AWG556"]


def test_setup_no_overrides_single_waveform():
    seqs = [{}, {}, {}]                                 # all defaults -> 1 waveform
    AWGManager.setup("AWG556", FakeScanGroup(seqs),
                     consts=_DEFAULTS, connection_factory=FakeConn)
    assert len(AWGManager._state["AWG556"]["cmd_map"]) == 1


# --------------------------------------------------------------------------- #
# AWGManager per-shot switching
# --------------------------------------------------------------------------- #
def test_recall_resends_on_change_and_skips_on_no_change():
    conns = []

    def factory(resource, channel):
        c = FakeConn(resource, channel)
        conns.append(c)
        return c

    seqs = [_seq_with_freq(f) for f in (130.0, 131.0, 132.0)]
    AWGManager.setup("AWG556", FakeScanGroup(seqs),
                     consts=_DEFAULTS, connection_factory=factory)
    conn = conns[0]
    n_after_setup = len(conn.sent)                      # 1 (init send of first waveform, key=130)

    # last_key after setup is the FIRST seq's key (130) -> recalling 130 is a no-op.
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 130.0}})
    assert len(conn.sent) == n_after_setup              # skipped (unchanged)

    # Switch to 131 -> one resend.
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 131.0}})
    assert len(conn.sent) == n_after_setup + 1

    # Same 131 again -> skipped.
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 131.0}})
    assert len(conn.sent) == n_after_setup + 1

    # Switch to 132 -> resend.
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 132.0}})
    assert len(conn.sent) == n_after_setup + 2
    assert AWGManager._state["AWG556"]["last_key"] == AWGManager._build_key(
        dict(_DEFAULTS["AWG556"], carrier_freq_MHz=132.0))


def test_recall_unknown_key_warns_and_does_not_send(caplog):
    AWGManager.setup("AWG556", FakeScanGroup([_seq_with_freq(130.0)]),
                     consts=_DEFAULTS, connection_factory=FakeConn)
    conn = AWGManager._state["AWG556"]["connection"]
    sent_before = len(conn.sent)
    # 999 MHz was never uploaded -> no matching cmd; should warn, not send, not crash.
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 999.0}})
    assert len(conn.sent) == sent_before


def test_recall_is_noop_without_setup():
    AWGManager._state = {}
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 130.0}})   # must not raise


def test_cleanup_disconnects_and_clears_state():
    AWGManager.setup("AWG556", FakeScanGroup([_seq_with_freq(130.0)]),
                     consts=_DEFAULTS, connection_factory=FakeConn)
    conn = AWGManager._state["AWG556"]["connection"]
    AWGManager.cleanup()
    assert conn.output_disabled == 1        # OUTP OFF -> quiet after the scan
    assert conn.disconnected == 1
    assert AWGManager._state == {}
    assert AWGManager.active_awgs() == []


# --------------------------------------------------------------------------- #
# run-loop wiring: runp().AWGs gating + the AWG.AWG556 scan convention
# --------------------------------------------------------------------------- #
def test_runp_awgs_and_awg_dot_name_convention():
    from scan_group import ScanGroup
    from runner import _awg_names

    g = ScanGroup()
    # The confirmed convention: g().AWG.<name>.<field>.scan(...) + runp().AWGs.
    g().AWG.AWG556.carrier_freq_MHz.scan(1, [130.0, 131.0, 132.0])
    g.runp().AWGs = ["AWG556"]

    # The runner reads which AWGs to activate from runp().AWGs.
    assert _awg_names(g) == ["AWG556"]
    # A scan WITHOUT AWGs declared -> [] (the AWG path is skipped, zero overhead).
    assert _awg_names(ScanGroup()) == []

    # Per-point AWG params land under AWG.AWG556 in getseq() (what the per-shot pre_cb reads).
    assert g.getseq(1)["AWG"]["AWG556"]["carrier_freq_MHz"] == 130.0
    assert g.getseq(3)["AWG"]["AWG556"]["carrier_freq_MHz"] == 132.0

    # Batch upload over a REAL ScanGroup -> one waveform per unique swept freq.
    AWGManager.setup("AWG556", g, consts=_DEFAULTS, connection_factory=FakeConn)
    assert len(AWGManager._state["AWG556"]["cmd_map"]) == 3


# --------------------------------------------------------------------------- #
# key construction
# --------------------------------------------------------------------------- #
def test_build_key_uses_only_waveform_fields():
    key = AWGManager._build_key(_DEFAULTS["AWG556"])
    for f in WAVEFORM_FIELDS:
        assert f in key
    # hardware-config fields must NOT enter the key
    assert "resource_address" not in key
    assert "max_amplitude_vpp" not in key
    assert "channel" not in key


# --------------------------------------------------------------------------- #
# ARWV recall path (firmware >= 38R3) -- pre-store named + ARWV NAME switch (no re-upload)
# --------------------------------------------------------------------------- #
def test_setup_arwv_path_stores_named_and_recalls_first():
    conns = []

    def factory(resource, channel):
        c = FakeConnARWV(resource, channel)
        conns.append(c)
        return c

    seqs = [_seq_with_freq(f) for f in (130.0, 131.0, 130.0, 132.0)]   # 3 unique
    AWGManager.setup("AWG556", FakeScanGroup(seqs),
                     consts=_DEFAULTS, connection_factory=factory)
    conn = conns[0]
    entry = AWGManager._state["AWG556"]
    assert entry["mode"] == "arwv"
    assert len(entry["name_map"]) == 3          # one stored name per unique combo
    assert sorted(entry["name_map"].values()) == ["wf_000", "wf_001", "wf_002"]
    assert len(conn.stored) == 3                # stored once each (setup-time)
    assert conn.arb_mode == 1                   # arb DDS mode set once
    assert conn.amplitudes == [11]              # amplitude set ONCE
    assert conn.burst_configured == 1 and conn.output_enabled == 1
    assert conn.recalled == ["wf_000"]          # first waveform recalled to init output
    assert conn.sent == []                      # ARWV path NEVER re-sends WVDT to 'active'
    assert "cmd_map" not in entry               # no per-shot WVDT cache on the ARWV path


def test_recall_arwv_switches_by_name_on_change_skips_on_no_change():
    conns = []

    def factory(resource, channel):
        c = FakeConnARWV(resource, channel)
        conns.append(c)
        return c

    seqs = [_seq_with_freq(f) for f in (130.0, 131.0, 132.0)]
    AWGManager.setup("AWG556", FakeScanGroup(seqs),
                     consts=_DEFAULTS, connection_factory=factory)
    conn = conns[0]
    n0 = len(conn.recalled)                     # 1 (first recall in setup, key=130)

    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 130.0}})   # unchanged -> skip
    assert len(conn.recalled) == n0
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 131.0}})   # switch -> recall
    assert len(conn.recalled) == n0 + 1
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 131.0}})   # same -> skip
    assert len(conn.recalled) == n0 + 1
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 132.0}})   # switch -> recall
    assert len(conn.recalled) == n0 + 2
    assert conn.sent == []                      # never re-uploads on the ARWV path


def test_recall_arwv_unknown_key_warns_and_does_not_recall():
    AWGManager.setup("AWG556", FakeScanGroup([_seq_with_freq(130.0)]),
                     consts=_DEFAULTS, connection_factory=FakeConnARWV)
    conn = AWGManager._state["AWG556"]["connection"]
    n0 = len(conn.recalled)
    AWGManager.recall_for_seq({"AWG556": {"carrier_freq_MHz": 999.0}})   # never stored
    assert len(conn.recalled) == n0


def test_build_waveform_cmd_named_slot():
    conn = AWGConnection("USB0::TEST::INSTR", "C1")    # no connect() -> no VISA
    binary = struct.pack(">5h", 1, 2, 3, 4, 5)
    cmd = conn.build_waveform_cmd(binary, 11, 250000.0, name="wf_007")
    text = cmd[:-10].decode("ascii")
    assert text.startswith("C1:WVDT WVNM,wf_007,WVTP,USER,AMPL,11,OFST,0,FREQ,250000")
    assert cmd.endswith(binary)


@pytest.mark.parametrize("idn,fw", [
    ("Siglent Technologies,SDG6022X,SDG6XFCC900309,6.01.01.38R3", (6, 1, 1, 38)),
    ("Siglent Technologies,SDG6022X,SN,6.01.01.37R6", (6, 1, 1, 37)),
    ("FAKE,SDG6X,0,1", None),                          # too few numbers -> unknown
])
def test_parse_fw(idn, fw):
    assert AWGConnection._parse_fw(idn) == fw


def test_arwv_min_fw_gate():
    from devices.sigilent_awg.awg_connection import ARWV_MIN_FW
    assert (6, 1, 1, 38) >= ARWV_MIN_FW            # 38R3 -> ARWV recall
    assert not ((6, 1, 1, 37) >= ARWV_MIN_FW)      # 37R6 -> re-upload fallback
