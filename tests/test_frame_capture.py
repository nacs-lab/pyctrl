"""frame_capture: the default per-shot ACQUIRE side (read frames -> hand to the persister).

NO-HARDWARE: a fake camera (scripted read_frames batches) + a fake server that records
``publish_shot`` calls. Verifies the read loop (all-at-once / trickle / short-read / surplus) and
the post_cb wiring (scan_id holder, seq_id read before the bump, async_ passthrough, short-read
drops the shot). Persistence itself (encode + store + finish) is ExptServer's job, tested in
test_exptserver_image_publish.py.
"""

import numpy as np
import pytest

from frame_capture import make_capture_post_cb, read_shot_frames

pytestmark = pytest.mark.no_hardware


class FakeCamera:
    """read_frames() returns the next scripted batch each call, then [] forever."""

    def __init__(self, batches):
        self._batches = list(batches)

    def read_frames(self):
        return self._batches.pop(0) if self._batches else []


class FakeServer:
    def __init__(self):
        self.published = []        # [(frames, scan_id, seq_id, async_), ...]

    def publish_shot(self, frames, scan_id, seq_id, *, async_=True):
        self.published.append((list(frames), scan_id, seq_id, async_))


def _frame(v):
    return np.full((2, 3), v, dtype=np.uint16)


# --------------------------------------------------------------------------- #
# read_shot_frames
# --------------------------------------------------------------------------- #
class TestReadShotFrames:
    def test_all_frames_in_one_batch(self):
        cam = FakeCamera([[_frame(1), _frame(2)]])
        frames = read_shot_frames(cam, 2, sleep=lambda dt: None)
        assert frames is not None and len(frames) == 2

    def test_frames_arriving_across_batches(self):
        cam = FakeCamera([[_frame(1)], [], [_frame(2)]])
        frames = read_shot_frames(cam, 2, sleep=lambda dt: None)
        assert frames is not None and len(frames) == 2

    def test_short_read_returns_none(self):
        cam = FakeCamera([[_frame(1)]])                 # only 1 of 2 ever arrives
        clock = _fake_clock([0.0, 0.0, 5.0, 11.0])      # advance past timeout
        frames = read_shot_frames(cam, 2, timeout=10.0, sleep=lambda dt: None, clock=clock)
        assert frames is None

    def test_surplus_truncated_to_num_images(self):
        cam = FakeCamera([[_frame(1), _frame(2), _frame(3)]])
        frames = read_shot_frames(cam, 2, sleep=lambda dt: None)
        assert len(frames) == 2


# --------------------------------------------------------------------------- #
# make_capture_post_cb -- scan_id holder + seq_id from seq_config + async passthrough
# --------------------------------------------------------------------------- #
class _G:
    def __init__(self, seq_id):
        self._seq_id = seq_id

    def seq_id(self, default=None):
        return self._seq_id


class _SeqConfig:
    def __init__(self, seq_id):
        self.G = _G(seq_id)


class TestCapturePostCb:
    def test_post_cb_publishes_with_ids(self):
        cam = FakeCamera([[_frame(5)]])
        srv = FakeServer()
        cb = make_capture_post_cb(cam, srv, 1, scan_id=999, seq_config=_SeqConfig(seq_id=4))
        cb(0, 1)
        assert len(srv.published) == 1
        frames, sid, qid, async_ = srv.published[0]
        assert sid == 999 and qid == 4 and async_ is True and len(frames) == 1

    def test_scan_id_callable_and_sync_passthrough(self):
        cam = FakeCamera([[_frame(5)]])
        srv = FakeServer()
        cb = make_capture_post_cb(cam, srv, 1, scan_id=lambda: 1234,
                                  seq_config=_SeqConfig(seq_id=2), async_=False)
        cb(0, 1)
        _, sid, _, async_ = srv.published[0]
        assert sid == 1234 and async_ is False           # kill-switch flows through to publish_shot

    def test_short_read_drops_shot_no_publish(self):
        cam = FakeCamera([[_frame(5)]])                  # 1 frame, need 2 -> short read
        srv = FakeServer()
        # timeout=0 -> the read deadline is already past, so read_shot_frames returns None
        # immediately and the post_cb publishes nothing (the shot is dropped).
        cb = make_capture_post_cb(cam, srv, 2, scan_id=1, seq_config=_SeqConfig(seq_id=1),
                                  timeout=0.0)
        cb(0, 1)
        assert srv.published == []


def _fake_clock(times):
    seq = list(times)
    state = {"i": 0}

    def clock():
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]

    return clock
