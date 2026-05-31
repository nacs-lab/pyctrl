"""The flattened .seq writer (lib/dump_output.py) + reader (tools/compare_seq_bytes.py).

All NO-HARDWARE: the byte packing and structure manipulation need neither the
engine nor a board. The engine-fed path (dump_output against a real compiled
handle) is exercised separately under --real-engine / downtime.

Ground truth: tests/reference/seqplotter_sample_ryddet.seq -- a real .seq produced
by MATLAB's ExpSeq.dump_output_to_file (downloaded from nacs-lab/SeqPlotter). If
the reader round-trips it byte-for-byte, the format spec is correct.
"""

import os

import pytest

import compare_seq_bytes
import dump_output
from conftest import PYCTRL_REF_DIR

pytestmark = pytest.mark.no_hardware

SAMPLE = os.path.join(PYCTRL_REF_DIR, "seqplotter_sample_ryddet.seq")


# --------------------------------------------------------------------------- #
# Reader faithfulness against real MATLAB output.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample .seq not present")
def test_sample_roundtrips_exact():
    data = compare_seq_bytes.load(SAMPLE)
    seq = compare_seq_bytes.decode(data)          # raises on malformed / trailing
    assert compare_seq_bytes.encode(seq) == data  # byte-for-byte


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample .seq not present")
def test_sample_is_well_formed():
    seq = compare_seq_bytes.decode(compare_seq_bytes.load(SAMPLE))
    assert len(seq["seqs"]) >= 1
    s = seq["seqs"][0]
    assert s["seq_idx"] == 1
    assert len(s["channels"]) > 0
    # every point is the documented (time int64, value float64, pulse_id uint32)
    for ch in s["channels"]:
        for p in ch["points"]:
            assert isinstance(p["t"], int) and isinstance(p["v"], float)
            assert p["pid"] >= 0


# --------------------------------------------------------------------------- #
# Channel extraction from a get_nominal_output-shaped return.
# --------------------------------------------------------------------------- #
def test_channels_from_nominal_output():
    res = [
        ("TTL1", [0, 1000, 2000], [0.0, 1.0, 0.0], [0, 1, 1]),
        ("AmpTweezer", [0, 500], [0.2, 0.8], [2, 3]),
    ]
    chns = dump_output.channels_from_nominal_output(res)
    assert [c["name"] for c in chns] == ["TTL1", "AmpTweezer"]
    assert chns[0]["points"][1] == {"t": 1000, "v": 1.0, "pid": 1}
    assert len(chns[1]["points"]) == 2


def test_alias_decoration():
    # No map -> nominal name verbatim.
    assert dump_output.decorate_channel_name("Dev1/0") == "Dev1/0"
    # One alias -> "alias (nominal)".
    m = {"Dev1/0": ["VShimZeroX"]}
    assert dump_output.decorate_channel_name("Dev1/0", m) == "VShimZeroX (Dev1/0)"
    # Multiple aliases -> "a, b (nominal)" (matlab_new/lib/ExpSeq.m:710-719).
    m = {"Dev1/0": ["Va", "Vb", "Dev1/0"]}
    assert dump_output.decorate_channel_name("Dev1/0", m) == "Va, Vb (Dev1/0)"


# --------------------------------------------------------------------------- #
# Packing round-trips through the reader (writer/reader symmetry).
# --------------------------------------------------------------------------- #
def test_pack_build_roundtrip():
    chns = dump_output.channels_from_nominal_output(
        [("TTL1", [0, 10], [0.0, 1.0], [0, 0])])
    s = dump_output.build_seq_struct("myseq", 1, chns)
    data = dump_output.pack([s])
    back = compare_seq_bytes.decode(data)
    assert back["has_bt_info"] == 0
    assert len(back["seqs"]) == 1
    assert back["seqs"][0]["seq_name"] == "myseq"
    assert back["seqs"][0]["has_params"] == 0
    assert back["seqs"][0]["channels"][0]["points"][1] == {"t": 10, "v": 1.0, "pid": 0}


def test_params_block_roundtrip():
    params = {"VShimZeroX": {"value": -0.04, "type": 1, "config_value": -0.04}}
    s = dump_output.build_seq_struct("s", 1, [], params=params)
    back = compare_seq_bytes.decode(dump_output.pack([s]))
    assert back["seqs"][0]["has_params"] == 1
    assert back["seqs"][0]["params"] == params


# --------------------------------------------------------------------------- #
# Scanned-parameter highlight hook.
# --------------------------------------------------------------------------- #
def test_mark_scanned_injects_marker():
    params = {
        "AWG": {"AWG556": {"pulse_width_us": {"value": 5, "type": 3},
                           "carrier_freq_MHz": {"value": 100, "type": 3}}},
        "Other": {"value": 1, "type": 1},
    }
    marked = dump_output.mark_scanned(
        params, ["AWG.AWG556.pulse_width_us"], {"AWG.AWG556.pulse_width_us": 1})
    leaf = marked["AWG"]["AWG556"]["pulse_width_us"]
    assert leaf["scanned"] is True and leaf["scan_dim"] == 1
    # untouched leaves keep no marker; original dict is not mutated
    assert "scanned" not in marked["AWG"]["AWG556"]["carrier_freq_MHz"]
    assert "scanned" not in params["AWG"]["AWG556"]["pulse_width_us"]


def test_mark_scanned_survives_seq_roundtrip():
    params = {"AWG": {"AWG556": {"pulse_width_us": {"value": 5, "type": 3}}}}
    marked = dump_output.mark_scanned(params, ["AWG.AWG556.pulse_width_us"])
    s = dump_output.build_seq_struct("s", 1, [], params=marked)
    back = compare_seq_bytes.decode(dump_output.pack([s]))
    assert back["seqs"][0]["params"]["AWG"]["AWG556"]["pulse_width_us"]["scanned"] is True


# --------------------------------------------------------------------------- #
# seq_name timestamp formatting (the one nondeterministic field).
# --------------------------------------------------------------------------- #
def test_format_seq_name():
    from datetime import datetime
    assert dump_output.format_seq_name("RydDet") == "RydDet"  # deterministic default
    dt = datetime(2025, 6, 19, 14, 26, 57)
    assert dump_output.format_seq_name("RydDet", dt) == "20250619_142657:RydDet"


# --------------------------------------------------------------------------- #
# Orchestration against a fake compiled-sequence handle (no engine needed).
# --------------------------------------------------------------------------- #
class _FakeEngineSeq:
    def __init__(self, res):
        self._res = res

    def get_nominal_output(self, pts):
        self.pts = pts
        return self._res


def test_dump_output_orchestration():
    fake = _FakeEngineSeq([("TTL1", [0, 1000], [0.0, 1.0], [0, 0])])
    data = dump_output.dump_output(fake, pts_per_ramp=50, seq_name="probe")
    assert fake.pts == 50
    back = compare_seq_bytes.decode(data)
    assert back["seqs"][0]["seq_name"] == "probe"
    assert back["seqs"][0]["channels"][0]["name"] == "TTL1"
    assert back["seqs"][0]["channels"][0]["points"][1]["v"] == 1.0


# --------------------------------------------------------------------------- #
# Full end-to-end through the board-free dummy engine: real serialize() bytes ->
# create_sequence -> get_nominal_output (synthetic) -> .seq -> decode. Proves the
# writer's whole path runs without the real engine / hardware.
# --------------------------------------------------------------------------- #
def test_dump_output_end_to_end_via_dummy():
    import compare_bytes
    import dummy_libnacs
    from conftest import all_reference_files

    refs = all_reference_files()
    if not refs:
        pytest.skip("no reference sequences found")
    mgr = dummy_libnacs.Manager()
    any_points = False
    for path in refs:
        eseq = mgr.create_sequence(bytearray(compare_bytes.load(path)))
        data = dump_output.dump_output(eseq, pts_per_ramp=20, seq_name="dummyrun")
        back = compare_seq_bytes.decode(data)             # decodes cleanly (well-formed)
        s = back["seqs"][0]
        assert s["seq_name"] == "dummyrun" and s["seq_idx"] == 1
        for ch in s["channels"]:
            # synthetic pulse_ids are the 0-indexed ordinals 0..n-1
            assert [p["pid"] for p in ch["points"]] == list(range(len(ch["points"])))
            if ch["points"]:
                any_points = True
    assert any_points, "dummy synth produced no channel points for any reference"
