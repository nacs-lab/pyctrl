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

from devices.sigilent_awg import (AWGConnection, AWGManager, SHAPES, WAVEFORM_FIELDS,
                                   gaussian_pulse_waveform, pulse_waveform)

pytestmark = pytest.mark.no_hardware


_DEFAULTS = {
    "AWG556": {
        "resource_address": "USB0::TEST::AWG556::INSTR",
        "channel": "C1", "max_amplitude_vpp": 11, "num_points": 1000,
        "pulse_width_us": 4, "carrier_freq_MHz": 130.78, "steepness": 3.5,
        "amplitude_scale": 1.0, "shape": "gaussian", "smooth_width_us": 0.0,
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

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz, channel=None):
        # reuse the real framing so the test also exercises it through the manager
        return AWGConnection.build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz,
                                                channel=channel)

    def send_waveform(self, cmd):
        self.sent.append(cmd)

    def set_amplitude(self, amp, channel=None):
        self.amplitudes.append(amp)
        self.amp_channels = getattr(self, "amp_channels", [])
        self.amp_channels.append(channel or self.channel)

    def configure_burst(self, channel=None, mode="GATE", ncyc=1, dlay=0.0):
        self.burst_configured += 1
        self.bursts = getattr(self, "bursts", [])
        self.bursts.append((channel or self.channel, mode, ncyc, dlay))

    def enable_output(self, channel=None):
        self.output_enabled += 1

    def disable_output(self, channel=None):
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

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz, name="active", channel=None):
        return AWGConnection.build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz,
                                                name=name, channel=channel)

    def set_arb_mode(self, channel=None):
        self.arb_mode += 1

    def store_waveform(self, cmd):
        self.stored.append(cmd)

    def recall_by_name(self, name, channel=None):
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
    from awg_runtime import awg_names

    g = ScanGroup()
    # The confirmed convention: g().AWG.<name>.<field>.scan(...) + runp().AWGs.
    g().AWG.AWG556.carrier_freq_MHz.scan(1, [130.0, 131.0, 132.0])
    g.runp().AWGs = ["AWG556"]

    # The run loop reads which AWGs to activate from runp().AWGs.
    assert awg_names(g) == ["AWG556"]
    # A scan WITHOUT AWGs declared -> [] (the AWG path is skipped, zero overhead).
    assert awg_names(ScanGroup()) == []

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
    p = _DEFAULTS["AWG556"]
    key = AWGManager._build_key(p)
    # every waveform field PRESENT in params enters the key (optional STIRAP fields absent here)
    for f in WAVEFORM_FIELDS:
        if f in p:
            assert f in key
    # hardware-config fields must NOT enter the key
    assert "resource_address" not in key
    assert "max_amplitude_vpp" not in key
    assert "channel" not in key
    # the optional two-lobe STIRAP fields DO enter the key when present (distinct waveform each)
    dbl = dict(p, shape="double_half_gaussian_outer", stirap_gap=2.0, f_delay=1.0, r_delay=1.5)
    kd = AWGManager._build_key(dbl)
    assert "stirap_gap=2" in kd and "f_delay=1" in kd and "r_delay=1.5" in kd
    assert kd != AWGManager._build_key(dict(dbl, f_delay=2.0))    # f_delay changes the key


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


# --------------------------------------------------------------------------- #
# pulse_waveform -- general shapes (gaussian / rise_* / fall_*), smooth window
# --------------------------------------------------------------------------- #
def _legacy_gaussian_oracle(params):
    """The ORIGINAL gaussianPulseWaveform math, inlined verbatim as a byte oracle."""
    num_points = int(params["num_points"])
    pulse_width_us = float(params["pulse_width_us"])
    carrier_freq_MHz = float(params["carrier_freq_MHz"])
    steepness = float(params["steepness"])
    amplitude_scale = float(params.get("amplitude_scale", 1.0))
    t = np.linspace(0.0, 1.0, num_points)
    carrier = np.sin(2.0 * np.pi * (carrier_freq_MHz * pulse_width_us) * t)
    envelope = np.exp(-((t - 0.5) * steepness) ** 2)
    waveform = carrier * envelope
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    waveform = waveform * amplitude_scale
    scaled = np.clip(np.round(waveform * 32767), -32768, 32767).astype(np.int16)
    return scaled.astype(">i2").tobytes(), 1e6 / pulse_width_us


@pytest.mark.parametrize("pw,steep,freq,amp", [
    (4, 3.5, 130.78, 1.0),
    (3.7, 4.0, 200.0, 0.7),          # non-power-of-2 width -> catches 1-ulp x-axis drift
])
def test_pulse_waveform_gaussian_matches_legacy_oracle(pw, steep, freq, amp):
    p = dict(_DEFAULTS["AWG556"], pulse_width_us=pw, steepness=steep,
             carrier_freq_MHz=freq, amplitude_scale=amp)
    want_bytes, want_freq = _legacy_gaussian_oracle(p)
    for entry in (pulse_waveform, gaussian_pulse_waveform):   # both paths byte-exact
        data, info = entry(p)
        assert data == want_bytes
        assert info["freq_hz"] == pytest.approx(want_freq)


def test_gaussian_wrapper_forces_gaussian_shape():
    p = dict(_DEFAULTS["AWG556"], shape="rise_gaussian", smooth_width_us=0.5)
    data, info = gaussian_pulse_waveform(p)                   # legacy entry ignores shape
    want, _ = _legacy_gaussian_oracle(p)
    assert data == want and info["shape"] == "gaussian"


def _env(shape, **kw):
    p = dict(_DEFAULTS["AWG556"], shape=shape, **kw)
    _, info = pulse_waveform(p)
    return info


def test_rise_gaussian_sharp_peaks_at_end():
    # unified def (2026-07-13): single half-Gaussian lobe, pw=1/e half-width, total=3*pw
    # (peak at 3*pw = the double's forward-556 peak position).
    from devices.sigilent_awg import pulse_envelope
    info = _env("rise_gaussian", steepness=4)   # steepness now IGNORED
    assert info["total_width_us"] == pytest.approx(12)   # 3*pw (pw=4)
    assert info["freq_hz"] == pytest.approx(1e6 / 12)
    _, e = pulse_envelope("rise_gaussian", 10000, 4.0, 0.0, 0.0)   # pure envelope (no carrier)
    assert e[0] < 2e-4 and e[-1] > 0.999    # ~0 (exp(-9)) at the trigger, peak at the END


def test_fall_gaussian_sharp_peaks_at_trigger():
    from devices.sigilent_awg import pulse_envelope
    info = _env("fall_gaussian", steepness=4)   # steepness IGNORED; mirror of rise
    assert info["total_width_us"] == pytest.approx(12)
    _, e = pulse_envelope("fall_gaussian", 10000, 4.0, 0.0, 0.0)
    assert e[0] > 0.999 and e[-1] < 2e-4    # peak AT the trigger, ~0 at the END


def test_half_gaussian_pw_is_1e_halfwidth_ignores_steepness_and_smooth():
    from devices.sigilent_awg import pulse_envelope
    pw = 4.0
    t, e = pulse_envelope("rise_gaussian", 30001, pw, 0.0, 0.0)   # total=3*pw=12, peak at 12
    assert t[-1] == pytest.approx(3 * pw)
    i = int(np.argmin(np.abs(t - (3 * pw - pw))))                 # peak - pw -> env = 1/e
    assert e[i] == pytest.approx(np.exp(-1), abs=1e-3)
    _, e2 = pulse_envelope("rise_gaussian", 30001, pw, 0.7, 99.0)  # smooth+steepness ignored
    assert np.allclose(e, e2)
    _, ef = pulse_envelope("fall_gaussian", 30001, pw, 0.0, 0.0)   # fall = mirror of rise
    assert np.allclose(ef, e[::-1])


def test_half_gaussian_continuity_and_ends():
    from devices.sigilent_awg import pulse_envelope
    _, er = pulse_envelope("rise_gaussian", 3000, 4.0, 0.0, 0.0)
    _, ef = pulse_envelope("fall_gaussian", 3000, 4.0, 0.0, 0.0)
    for e in (er, ef):
        assert np.max(np.abs(np.diff(e))) < 0.01                 # smooth Gaussian, no jump
    assert er[0] < 0.02 and er[-1] > 0.98                        # rise: ~0 -> peak at end
    assert ef[0] > 0.98 and ef[-1] < 0.02                        # fall: peak at start -> ~0


def test_rise_linear_smooth_appends_cosine_tail():
    info = _env("rise_linear", smooth_width_us=0.5)
    t, e = info["t_us"], np.abs(info["waveform"])
    assert info["total_width_us"] == pytest.approx(4.5)          # pw + smooth
    assert info["freq_hz"] == pytest.approx(1e6 / 4.5)           # DDS FREQ over the TOTAL width
    assert t[-1] == pytest.approx(4.5)
    # cosine tail: ~0 only AT t=4.5 (at 4.4 it is 0.095 by design); check the last sliver
    assert np.max(e[t > 4.45]) < 0.05
    assert np.max(e[(t > 3.9) & (t < 4.1)]) > 0.9                # peak still at end of MAIN window


def test_fall_linear_smooth_prepends_rise():
    info = _env("fall_linear", smooth_width_us=0.5)
    t, e = info["t_us"], np.abs(info["waveform"])
    assert info["total_width_us"] == pytest.approx(4.5)
    assert np.max(e[t < 0.05]) < 0.05                            # starts from ~0 at the trigger
    assert np.max(e[(t > 0.45) & (t < 0.6)]) > 0.9               # peak ~smooth_width after trigger
    assert np.max(e[t > 4.4]) < 0.05                             # linear ramp -> ~0 at the end


def test_linear_envelope_continuity_no_step_when_smooth():
    from devices.sigilent_awg import pulse_envelope
    for shape in ("rise_linear", "fall_linear"):
        _, e = pulse_envelope(shape, 2000, 4.0, 0.5, 0.0)
        assert np.max(np.abs(np.diff(e))) < 0.01                 # no jump anywhere
        # ... and WITH smooth the linear ramp is ~0 at both ends
        assert e[0] < 0.02 and e[-1] < 0.02


def test_linear_shapes_ramp_and_ignore_steepness():
    p = dict(_DEFAULTS["AWG556"], shape="rise_linear")
    del p["steepness"]                                           # not required for linear
    _, info = pulse_waveform(p)
    t, e = info["t_us"], np.abs(info["waveform"])
    assert np.max(e[:50]) < 0.06 and np.max(e[-50:]) > 0.9       # 0 -> 1 ramp
    p["shape"] = "fall_linear"
    _, info = pulse_waveform(p)
    e = np.abs(info["waveform"])
    assert np.max(e[:50]) > 0.9 and np.max(e[-50:]) < 0.06       # 1 -> 0 ramp


def test_spline_total_is_pw_and_exact_ends():
    # cubic/quintic splines: compact support, total = pw (NOT 3*pw), exactly 0 / 1 at the ends,
    # peak at the END (rise) / START (fall). pw=4 in _DEFAULTS["AWG556"].
    from devices.sigilent_awg import pulse_envelope
    from devices.sigilent_awg.pulse_waveform import pulse_total_us
    for shape in ("rise_cubic", "fall_cubic", "rise_quintic", "fall_quintic"):
        info = _env(shape, steepness=99)                         # steepness IGNORED
        assert info["total_width_us"] == pytest.approx(4)        # total = pw
        assert info["freq_hz"] == pytest.approx(1e6 / 4)         # DDS FREQ over pw
        assert pulse_total_us(shape, 4.0) == pytest.approx(4)
        t, e = pulse_envelope(shape, 5000, 4.0, 0.0, 0.0)
        assert t[-1] == pytest.approx(4)
        assert 0.0 <= e.min() and e.max() <= 1.0 + 1e-12
        if shape.startswith("rise_"):
            assert e[0] == pytest.approx(0.0, abs=1e-9) and e[-1] == pytest.approx(1.0, abs=1e-9)
        else:
            assert e[0] == pytest.approx(1.0, abs=1e-9) and e[-1] == pytest.approx(0.0, abs=1e-9)


def test_spline_fall_is_mirror_of_rise():
    from devices.sigilent_awg import pulse_envelope
    for rise, fall in (("rise_cubic", "fall_cubic"), ("rise_quintic", "fall_quintic")):
        _, er = pulse_envelope(rise, 4001, 4.0, 0.0, 0.0)
        _, ef = pulse_envelope(fall, 4001, 4.0, 0.0, 0.0)
        assert np.allclose(ef, er[::-1])


def test_spline_endpoint_slopes_zero_quintic_also_curvature():
    # cubic + quintic both have zero endpoint SLOPE (smoothstep); quintic additionally has zero
    # endpoint CURVATURE (smootherstep -> no acceleration kink at turn-on / peak).
    from devices.sigilent_awg import pulse_envelope
    t, ec = pulse_envelope("rise_cubic", 20001, 4.0, 0.0, 0.0)
    _, eq = pulse_envelope("rise_quintic", 20001, 4.0, 0.0, 0.0)
    dc, dq = np.gradient(ec, t), np.gradient(eq, t)
    for d in (dc, dq):                                           # zero slope at both ends
        assert abs(d[0]) < 1e-3 and abs(d[-1]) < 1e-3
    d2c, d2q = np.gradient(dc, t), np.gradient(dq, t)
    assert abs(d2q[0]) < 1e-3 and abs(d2q[-1]) < 1e-3            # quintic: zero endpoint curvature
    assert abs(d2c[0]) > 0.05                                    # cubic: curvature JUMPS at the end


def test_spline_ignores_steepness_and_smooth():
    from devices.sigilent_awg import pulse_envelope
    _, a = pulse_envelope("rise_quintic", 3000, 4.0, 0.0, 0.0)
    _, b = pulse_envelope("rise_quintic", 3000, 4.0, 0.9, 88.0)  # smooth + steepness ignored
    assert np.allclose(a, b)


def test_flat_shape():
    from devices.sigilent_awg import pulse_envelope
    from devices.sigilent_awg.pulse_waveform import pulse_total_us
    # envelope is a constant 1 over [0, pw]; smooth/steepness ignored
    t, e = pulse_envelope("flat", 5000, 5.0, 3.0, 88.0)
    assert np.all(e == 1.0) and t[-1] == pytest.approx(5.0)
    assert pulse_total_us("flat", 5.0) == pytest.approx(5.0)
    # output peak reaches full amplitude_scale; voltage = scale * vpp / 2
    _, info = pulse_waveform(dict(shape="flat", num_points=2000, pulse_width_us=5.0,
                                  carrier_freq_MHz=10.0, amplitude_scale=0.7,
                                  max_amplitude_vpp=2.0))
    assert np.max(np.abs(info["waveform"])) == pytest.approx(0.7, abs=1e-3)
    assert np.max(np.abs(info["voltage"])) == pytest.approx(0.7, abs=1e-3)
    assert info["freq_hz"] == pytest.approx(1e6 / 5.0)


def test_smooth_width_ignored_for_gaussian():
    a, ia = pulse_waveform(dict(_DEFAULTS["AWG556"]))
    b, ib = pulse_waveform(dict(_DEFAULTS["AWG556"], smooth_width_us=1.0))
    assert a == b and ib["freq_hz"] == ia["freq_hz"] == pytest.approx(1e6 / 4)


def test_pulse_waveform_validation():
    with pytest.raises(ValueError, match="unknown pulse shape"):
        pulse_waveform(dict(_DEFAULTS["AWG556"], shape="half_gaussian"))
    with pytest.raises(ValueError, match="smooth_width_us"):
        pulse_waveform(dict(_DEFAULTS["AWG556"], shape="rise_linear", smooth_width_us=-0.1))
    assert set(SHAPES) == {"flat", "gaussian", "rise_gaussian", "fall_gaussian",
                           "rise_linear", "fall_linear",
                           "rise_cubic", "fall_cubic", "rise_quintic", "fall_quintic",
                           "double_half_gaussian_inner", "double_half_gaussian_outer"}


# --------------------------------------------------------------------------- #
# shape params dispatch through the manager (key + dedup)
# --------------------------------------------------------------------------- #
def test_build_key_includes_string_shape_and_smooth():
    p = dict(_DEFAULTS["AWG556"], shape="rise_gaussian", smooth_width_us=0.5)
    key = AWGManager._build_key(p)
    assert "shape=rise_gaussian" in key
    assert "smooth_width_us=0.5" in key
    assert key != AWGManager._build_key(dict(p, shape="fall_gaussian"))
    assert key != AWGManager._build_key(dict(p, smooth_width_us=0.0))


def test_setup_dedups_by_shape():
    # 4 seqs: two shapes x two repeats -> exactly 2 unique waveforms uploaded
    seqs = [{"AWG": {"AWG556": {"shape": s}}} for s in
            ("rise_gaussian", "fall_gaussian", "rise_gaussian", "fall_gaussian")]
    consts = {"AWG556": dict(_DEFAULTS["AWG556"], shape="gaussian", smooth_width_us=0.5)}
    AWGManager.setup("AWG556", FakeScanGroup(seqs), consts=consts,
                     connection_factory=FakeConn)
    assert len(AWGManager._state["AWG556"]["cmd_map"]) == 2

    # per-shot switch keyed on shape: rise -> fall resends, same shape skips
    conn = AWGManager._state["AWG556"]["connection"]
    n0 = len(conn.sent)
    AWGManager.recall_for_seq({"AWG556": {"shape": "rise_gaussian"}})    # == first seq -> skip
    assert len(conn.sent) == n0
    AWGManager.recall_for_seq({"AWG556": {"shape": "fall_gaussian"}})    # switch -> resend
    assert len(conn.sent) == n0 + 1


# --------------------------------------------------------------------------- #
# two-channel switch scheme (Ch1/Ch2 per-channel config)
# --------------------------------------------------------------------------- #
_DEFAULTS_CH = {
    "AWG556": {
        "resource_address": "USB0::TEST::AWG556::INSTR",
        "num_points": 1000, "sample_rate_MHz": 2500,
        "Ch1": {"channel": "C1", "shape": "rise_gaussian", "carrier_freq_MHz": 143.4,
                "pulse_width_us": 1.437, "steepness": 3.5, "amplitude_scale": 1.0,
                "smooth_width_us": 0.0, "max_amplitude_vpp": 15, "trig_delay_us": 0.0},
        "Ch2": {"channel": "C2", "shape": "fall_gaussian", "carrier_freq_MHz": 143.4,
                "pulse_width_us": 1.463, "steepness": 3.5, "amplitude_scale": 1.0,
                "smooth_width_us": 0.0, "max_amplitude_vpp": 15, "trig_delay_us": 0.0},
    },
}


def test_channel_mode_arms_both_channels_ncyc():
    seqs = [{}]                                  # single point, defaults only
    AWGManager.setup("AWG556", FakeScanGroup(seqs), consts=_DEFAULTS_CH,
                     connection_factory=FakeConn)
    entry = AWGManager._state["AWG556"]
    assert entry["mode"] == "channel"
    assert set(entry["channels"]) == {"Ch1", "Ch2"}
    assert entry["channels"]["Ch1"]["scpi_ch"] == "C1"
    assert entry["channels"]["Ch2"]["scpi_ch"] == "C2"

    conn = entry["connection"]
    # both channels armed as single-cycle NCYC EXT bursts, amplitude set on each
    modes = {(ch, mode, ncyc) for (ch, mode, ncyc, _dlay) in conn.bursts}
    assert ("C1", "NCYC", 1) in modes and ("C2", "NCYC", 1) in modes
    assert conn.output_enabled == 2                       # both outputs on
    # C1 gets a rise_gaussian, C2 a fall_gaussian (different waveform bytes -> different WVDT)
    c1_wvdt = [c for c in conn.sent if c.startswith(b"C1:")]
    c2_wvdt = [c for c in conn.sent if c.startswith(b"C2:")]
    assert c1_wvdt and c2_wvdt


def test_channel_mode_per_channel_scan_and_recall():
    # scan sweeps Ch2.pulse_width_us over 2 values; Ch1 fixed
    seqs = [{"AWG": {"AWG556": {"Ch2": {"pulse_width_us": pw}}}} for pw in (1.463, 1.9)]
    AWGManager.setup("AWG556", FakeScanGroup(seqs), consts=_DEFAULTS_CH,
                     connection_factory=FakeConn)
    ch2 = AWGManager._state["AWG556"]["channels"]["Ch2"]
    assert len(ch2["cmd_map"]) == 2                       # two unique Ch2 waveforms
    assert len(AWGManager._state["AWG556"]["channels"]["Ch1"]["cmd_map"]) == 1  # Ch1 unchanged

    conn = AWGManager._state["AWG556"]["connection"]
    n0 = len(conn.sent)
    AWGManager.recall_for_seq({"AWG556": {"Ch2": {"pulse_width_us": 1.463}}})  # == setup default -> skip
    assert len(conn.sent) == n0
    AWGManager.recall_for_seq({"AWG556": {"Ch2": {"pulse_width_us": 1.9}}})    # switch Ch2 -> resend
    assert len(conn.sent) == n0 + 1
    assert conn.sent[-1].startswith(b"C2:")              # the resend targeted C2
