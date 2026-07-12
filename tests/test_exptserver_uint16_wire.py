"""ExptServer uint16 image wire (verb ``get_imgs_uint16``) -- the parallel uint16 producer.

NO hardware/engine (ExptServer ZMQ bind only, like test_exptserver_image_publish). Stages frames
via publish_shot / store_imgs, drains via the NEW verb, and parses with an inline reference parser
that reads the stream with EXPLICIT byte offsets (the u2 pixel blocks leave the following f8
fields unaligned). Also pins: nseqs=0 -> 8 bytes, seq_cancel interleaving, the float64-staged
clip/round path, and that a uint16-staged frame drained via the OLD get_imgs is byte-for-byte the
legacy float64 Fortran stream (built from to_store_array).
"""
import array
import socket
import struct

import numpy as np
import pytest

import ExptServer as expt_mod
from ExptServer import ExptServer, _StagedFrame
from devices.orca import to_store_array, to_store_frame_u16

pytestmark = pytest.mark.no_hardware


def _free_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "tcp://127.0.0.1:%d" % port


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(expt_mod, "QUEUE_PATH", str(tmp_path / "runner_queue.json"))
    srv = ExptServer(_free_url())
    yield srv
    for teardown in (lambda: srv.stop_worker(),
                     lambda: srv._ExptServer__sock.close(linger=0),
                     lambda: srv._ExptServer__ctx.term()):
        try:
            teardown()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# inline reference parser for the uint16 wire (explicit byte offsets)
# --------------------------------------------------------------------------- #
def parse_u16_wire(buf):
    """Decode the get_imgs_uint16 stream. Returns (nseqs, [shot, ...]) where each shot is
    {scan_id, seq_id, images=[(s1,s2,s3, ndarray), ...]}. Deliberately NEVER assumes 8-byte
    alignment -- struct.unpack_from / np.frombuffer take explicit offsets."""
    off = 0

    def f8():
        nonlocal off
        (v,) = struct.unpack_from('<d', buf, off)
        off += 8
        return v

    nseqs = int(f8())
    shots = []
    for _ in range(nseqs):
        first = f8()
        if first == 0.0:                       # empty shot (no header, no images)
            shots.append({'scan_id': None, 'seq_id': None, 'images': []})
            continue
        scan_id = first
        seq_id = f8()
        images = []
        while True:
            s1 = f8()
            if s1 == 0.0:                      # shot separator (s1 is never 0)
                break
            s2 = int(f8())
            s3 = int(f8())
            s1 = int(s1)
            n = s1 * s2 * s3
            pix = np.frombuffer(bytes(buf), dtype=np.uint16, count=n, offset=off)
            off += 2 * n
            images.append((s1, s2, s3, pix.reshape(s1, s2, s3).copy()))
        shots.append({'scan_id': scan_id, 'seq_id': seq_id, 'images': images})
    assert off == len(buf), "trailing/short bytes: off=%d len=%d" % (off, len(buf))
    return nseqs, shots


def _u16(v, shape=(2, 3)):
    return np.full(shape, v, dtype=np.uint16)


# --------------------------------------------------------------------------- #
# basic: single shot / single image
# --------------------------------------------------------------------------- #
def test_single_shot_single_image(server):
    frame = np.arange(6, dtype=np.uint16).reshape(2, 3)   # [[0,1,2],[3,4,5]]
    server.publish_shot([frame], 111, 222)
    server.drain_images()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 1 and len(shots) == 1
    sh = shots[0]
    assert sh['scan_id'] == 111.0 and sh['seq_id'] == 222.0
    assert len(sh['images']) == 1
    s1, s2, s3, pix = sh['images'][0]
    assert (s1, s2, s3) == (2, 3, 1)
    assert np.array_equal(pix[:, :, 0], frame)


def test_multi_shot(server):
    server.publish_shot([_u16(7)], 1, 10)
    server.publish_shot([_u16(9)], 1, 11)
    server.drain_images()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 2
    assert shots[0]['seq_id'] == 10.0 and shots[1]['seq_id'] == 11.0
    assert np.array_equal(shots[0]['images'][0][3][:, :, 0], _u16(7))
    assert np.array_equal(shots[1]['images'][0][3][:, :, 0], _u16(9))


def test_multi_image_per_shot(server):
    f1, f2 = _u16(3), _u16(4)
    server.publish_shot([f1, f2], 5, 6)
    server.drain_images()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 1 and len(shots[0]['images']) == 2
    assert np.array_equal(shots[0]['images'][0][3][:, :, 0], f1)
    assert np.array_equal(shots[0]['images'][1][3][:, :, 0], f2)


# --------------------------------------------------------------------------- #
# empty: nothing staged -> just [nseqs=0], 8 bytes
# --------------------------------------------------------------------------- #
def test_empty_nseqs_zero(server):
    rep = server.get_imgs_uint16()
    assert len(rep) == 8
    nseqs, shots = parse_u16_wire(rep)
    assert nseqs == 0 and shots == []


# --------------------------------------------------------------------------- #
# seq_cancel interleaved with a good shot -> only the good shot survives
# --------------------------------------------------------------------------- #
def test_seq_cancel_interleaved(server):
    server.stage_frame(_u16(1), 2, 3)        # stage a frame ...
    server.cancel_shot()                     # ... then drop it (never finished)
    server.publish_shot([_u16(8)], 2, 4)     # a real shot after the cancel
    server.drain_images()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 1
    assert shots[0]['seq_id'] == 4.0
    assert np.array_equal(shots[0]['images'][0][3][:, :, 0], _u16(8))


# --------------------------------------------------------------------------- #
# unaligned offsets: odd pixel counts push the following f8 fields off 8-byte boundaries
# --------------------------------------------------------------------------- #
def test_unaligned_offsets(server):
    f_3x3 = np.arange(9, dtype=np.uint16).reshape(3, 3)        # 9 px -> 18 bytes
    f_5x7 = (np.arange(35, dtype=np.uint16) * 3).reshape(5, 7)  # 35 px -> 70 bytes
    server.publish_shot([f_3x3, f_5x7], 42, 99)                # two odd images in one shot
    server.drain_images()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 1 and len(shots[0]['images']) == 2
    a1 = shots[0]['images'][0]
    a2 = shots[0]['images'][1]
    assert (a1[0], a1[1], a1[2]) == (3, 3, 1)
    assert (a2[0], a2[1], a2[2]) == (5, 7, 1)
    assert np.array_equal(a1[3][:, :, 0], f_3x3)
    assert np.array_equal(a2[3][:, :, 0], f_5x7)


# --------------------------------------------------------------------------- #
# float64-staged frame drained via the uint16 verb -> clip(round) to uint16
# --------------------------------------------------------------------------- #
def test_float64_staged_clip_round(server):
    # A frame with fractional + out-of-range values, staged via the legacy float64 store_imgs API
    # (NOT publish_shot, which would fast-path integral uint16). Drained via get_imgs_uint16 it
    # must clip+round: 0.4->0, 1.6->2, 70000->65535, -3->0.
    frame_f = np.array([[0.4, 1.6], [70000.0, -3.0]], dtype=np.float64)
    server.store_imgs(to_store_array(frame_f), 700, 800)
    server.seq_finish()
    nseqs, shots = parse_u16_wire(server.get_imgs_uint16())
    assert nseqs == 1
    s1, s2, s3, pix = shots[0]['images'][0]
    assert (s1, s2, s3) == (2, 2, 1)
    expected = np.array([[0, 2], [65535, 0]], dtype=np.uint16)
    assert np.array_equal(pix[:, :, 0], expected)


# --------------------------------------------------------------------------- #
# uint16-staged frame via the OLD get_imgs verb -> byte-identical to the legacy float64 stream
# --------------------------------------------------------------------------- #
def test_uint16_staged_old_verb_byte_identical(server):
    frame = np.arange(6, dtype=np.uint16).reshape(2, 3)
    server.publish_shot([frame], 111, 222)    # fast uint16 staging (no upcast)
    server.drain_images()
    old = bytes(server.get_imgs())
    # Independently build the legacy float64 wire: [nseqs][scan_id][seq_id][to_store_array][0.0].
    expected = struct.pack('<d', 1.0)
    expected += struct.pack('<dd', 111.0, 222.0)
    expected += to_store_array(frame).tobytes()
    expected += struct.pack('<d', 0.0)
    assert old == expected


def test_staged_frame_dual_serialization_matches():
    # _StagedFrame from either source serializes consistently across the two wires for the same
    # integral pixels: OLD (f8 Fortran) reconstructs to_store_array; NEW (u16 C-order) is raw.
    frame = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16)
    (s1, s2, s3), raw = to_store_frame_u16(frame)
    sf_u16 = _StagedFrame.from_u16((s1, s2, s3), raw)
    sf_f64 = _StagedFrame.from_float64(array.array('d', to_store_array(frame)))
    # OLD wire identical between the two staging origins.
    assert sf_u16.tobytes() == sf_f64.tobytes() == to_store_array(frame).tobytes()
    # NEW wire: shape prefix + C-order uint16 == frame.tobytes().
    assert sf_u16.u16_bytes() == struct.pack('<3d', 2.0, 3.0, 1.0) + frame.tobytes()
    assert sf_f64.u16_bytes() == sf_u16.u16_bytes()
