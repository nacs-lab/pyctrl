"""Phase-5 frame_capture: feed a shot's frames to the ExptServer (run-loop capture core).

NO-HARDWARE: a fake camera (scripted read_frames batches) + a fake ExptServer record the
store_imgs / seq_finish / seq_cancel calls. Verifies the column-major [s1,s2,s3,pixels] format,
the per-shot publish, the short-read seq_cancel guard, and the post_cb wiring (scan_id holder +
seq_id read from seq_config before the loop bumps it).
"""

import numpy as np
import pytest

from frame_capture import make_capture_post_cb, store_shot_frames

pytestmark = pytest.mark.no_hardware


class FakeCamera:
    """read_frames() returns the next scripted batch each call, then [] forever."""

    def __init__(self, batches):
        self._batches = list(batches)

    def read_frames(self):
        return self._batches.pop(0) if self._batches else []


class FakeServer:
    def __init__(self):
        self.stored = []           # [(arr, scan_id, seq_id), ...]
        self.finished = 0
        self.cancelled = 0

    def store_imgs(self, arr, scan_id, seq_id):
        self.stored.append((np.asarray(arr), scan_id, seq_id))

    def seq_finish(self):
        self.finished += 1

    def seq_cancel(self):
        self.cancelled += 1


def _frame(v):
    return np.full((2, 3), v, dtype=np.uint16)


# --------------------------------------------------------------------------- #
# store_shot_frames
# --------------------------------------------------------------------------- #
class TestStoreShotFrames:
    def test_publishes_all_frames_then_finishes(self):
        cam = FakeCamera([[_frame(1), _frame(2)]])     # both frames in one batch
        srv = FakeServer()
        n = store_shot_frames(cam, srv, 2, scan_id=111, seq_id=7, sleep=lambda dt: None)
        assert n == 2
        assert len(srv.stored) == 2 and srv.finished == 1 and srv.cancelled == 0
        # ids stamped on each stored image; format = [s1,s2,s3, pixels col-major]
        arr0, sid, qid = srv.stored[0]
        assert sid == 111 and qid == 7
        assert (int(arr0[0]), int(arr0[1]), int(arr0[2])) == (2, 3, 1)
        assert arr0.size == 3 + 2 * 3 * 1

    def test_frames_arriving_across_batches(self):
        cam = FakeCamera([[_frame(1)], [], [_frame(2)]])   # trickle in
        srv = FakeServer()
        n = store_shot_frames(cam, srv, 2, 1, 1, sleep=lambda dt: None)
        assert n == 2 and srv.finished == 1

    def test_short_read_cancels_and_publishes_nothing(self):
        # Only 1 of 2 frames ever arrives -> timeout -> seq_cancel, no store/finish.
        cam = FakeCamera([[_frame(1)]])
        srv = FakeServer()
        clock = _fake_clock([0.0, 0.0, 5.0, 11.0])      # advance past timeout
        n = store_shot_frames(cam, srv, 2, 1, 1, timeout=10.0,
                              sleep=lambda dt: None, clock=clock)
        assert n == 1
        assert srv.stored == [] and srv.finished == 0 and srv.cancelled == 1

    def test_extra_frames_ignored(self):
        cam = FakeCamera([[_frame(1), _frame(2), _frame(3)]])   # 3 arrive, only 2 wanted
        srv = FakeServer()
        n = store_shot_frames(cam, srv, 2, 1, 1, sleep=lambda dt: None)
        assert n == 2 and len(srv.stored) == 2


# --------------------------------------------------------------------------- #
# make_capture_post_cb -- scan_id holder + seq_id from seq_config
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
    def test_post_cb_uses_scan_and_seq_ids(self):
        cam = FakeCamera([[_frame(5)]])
        srv = FakeServer()
        cb = make_capture_post_cb(cam, srv, 1, scan_id=999, seq_config=_SeqConfig(seq_id=4))
        cb(0, 1)                                        # run_scan_group's post_cb(cur, arg0)
        assert srv.finished == 1
        _, sid, qid = srv.stored[0]
        assert sid == 999 and qid == 4

    def test_scan_id_callable(self):
        cam = FakeCamera([[_frame(5)]])
        srv = FakeServer()
        cb = make_capture_post_cb(cam, srv, 1, scan_id=lambda: 1234,
                                  seq_config=_SeqConfig(seq_id=2))
        cb(0, 1)
        assert srv.stored[0][1] == 1234


def _fake_clock(times):
    seq = list(times)
    state = {"i": 0}

    def clock():
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]

    return clock
