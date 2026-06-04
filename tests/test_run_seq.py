"""Phase-5 run_seq: the scan-loop body (port of runSeq2.m).

NO-HARDWARE: a REAL ScanGroup drives ``getseq_with_var`` (Phase 4), while ``compile_point``,
``run_real``, and the control channel are injected fakes -- so the orchestration (three
memos, gate placement, dual counters, retry, rep/random loops, abort-sticky, config bracket)
is verified with the engine never loaded.
"""

import pytest

from dyn_props import DynProps
from run_seq import run_scan_group
from scan_group import ScanGroup
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeRunSeq:
    def __init__(self, tag):
        self.tag = tag
        self.C = DynProps({})


class Recorder:
    """Bundles injected fakes + their call logs for one run."""

    def __init__(self, abort_after=None, begin=4242, restart_once_tag=None):
        self.compiles = []          # swept values compiled (one per distinct point)
        self.runs = []              # tags run_real was called with (incl. retries)
        self.pre = []               # (cur_seq_num, arg) seen by pre_cb
        self.post = []
        self.seq_nums = []          # on_seq_num publications
        self.teardowns = 0
        self.checks = 0
        self._abort_after = abort_after
        self._begin = begin
        self._restart_once_tag = restart_once_tag
        self._restart_fired = set()

    # compile_point(seqfn, seqparam)
    def compile_point(self, seqfn, seqparam):
        val = seqparam["A"]["B"]
        self.compiles.append(val)
        return FakeRunSeq(val)

    # run_real(seq)
    def run_real(self, seq):
        self.runs.append(seq.tag)
        if seq.tag == self._restart_once_tag and seq.tag not in self._restart_fired:
            self._restart_fired.add(seq.tag)
            seq.C.RESTART = 1           # ask for one retry
        else:
            seq.C.RESTART = 0

    # ControlChannel surface
    def begin_scan(self):
        return self._begin

    def check_pause_abort(self):
        self.checks += 1
        return self._abort_after is not None and self.checks >= self._abort_after

    def pre_cb(self, n, arg):
        self.pre.append((n, arg))

    def post_cb(self, n, arg):
        self.post.append((n, arg))

    def on_seq_num(self, n):
        self.seq_nums.append(n)

    def teardown(self):
        self.teardowns += 1


def _scan(vals):
    g = ScanGroup()
    g().A.B.scan(1, list(vals))
    return g


def _run(g, rec, **kw):
    sc = SeqConfig()                    # isolated empty config (fresh G)
    res = run_scan_group(
        _seqfn, g, control=rec, compile_point=rec.compile_point, run_real=rec.run_real,
        seq_config=sc, on_seq_num=rec.on_seq_num, config_teardown=rec.teardown, **kw)
    return res, sc


def _seqfn(s):
    return s


# --------------------------------------------------------------------------- #
# sequential full scan + counters + memos
# --------------------------------------------------------------------------- #
def test_full_scan_runs_each_point_once():
    rec = Recorder()
    res, sc = _run(_scan([1.0, 2.0, 3.0]), rec)
    assert res == {"status": "ok", "nseq": 3}
    assert rec.compiles == [1.0, 2.0, 3.0]      # one compile per distinct point
    assert rec.runs == [1.0, 2.0, 3.0]
    assert sc.G.seq_id() == 4                    # 1 -> 4 after 3 shots
    assert rec.seq_nums[-1] == 0                 # end-of-run reset (CurrentSeqNum -> 0)
    assert 3 in rec.seq_nums                     # ...but reached 3 mid-run
    assert rec.teardowns == 1                    # config bracket teardown fired


def test_rep_reuses_compiled_points():
    rec = Recorder()
    res, _ = _run(_scan([1.0, 2.0, 3.0]), rec, rep=2)
    # 2 reps * 3 points = 6 shots, but only 3 compiles (seqlist rep-reuse).
    assert res == {"status": "ok", "nseq": 6}
    assert rec.compiles == [1.0, 2.0, 3.0]
    assert rec.runs == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_duplicate_index_dedup_via_seq_map():
    # indices with a duplicate point -> compiled once (seq_map[arg0] dedup), run twice.
    rec = Recorder()
    sc = SeqConfig()
    run_scan_group(_seqfn, _scan([10.0, 20.0]), indices=[1, 2, 1],
                   control=rec, compile_point=rec.compile_point, run_real=rec.run_real,
                   seq_config=sc, on_seq_num=rec.on_seq_num, config_teardown=rec.teardown)
    assert rec.compiles == [10.0, 20.0]          # point 1 compiled once despite appearing twice
    assert rec.runs == [10.0, 20.0, 10.0]        # but run all three slots


# --------------------------------------------------------------------------- #
# gate: abort + refused start
# --------------------------------------------------------------------------- #
def test_gate_abort_stops_and_does_not_count_aborted_shot():
    rec = Recorder(abort_after=2)               # abort at the 2nd gate check (shot 2)
    res, sc = _run(_scan([1.0, 2.0, 3.0]), rec)
    assert res == {"status": "aborted", "nseq": 1}   # only shot 1 completed
    assert rec.runs == [1.0]
    assert sc.G.seq_id() == 2                         # advanced once (shot 1 only)
    assert rec.teardowns == 1                         # teardown fires on the abort path too


def test_begin_scan_none_is_a_refused_start():
    # run_scan_group's contract: when begin_scan() returns None (the source refused to start),
    # the scan aborts at 0 iterations with the config bracket torn down. (begin_scan no longer
    # returns None for a stale abort -- that is now cleared at job start -- but the loop keeps
    # honoring a None start as the generic refusal signal.)
    rec = Recorder(begin=None)
    res, _ = _run(_scan([1.0, 2.0, 3.0]), rec)
    assert res == {"status": "aborted", "nseq": 0}
    assert rec.compiles == [] and rec.runs == []
    assert rec.teardowns == 1


# --------------------------------------------------------------------------- #
# RESTART retry + callbacks + tstartwait
# --------------------------------------------------------------------------- #
def test_restart_flag_retries_the_shot():
    rec = Recorder(restart_once_tag=2.0)        # point 2 asks for one retry
    res, _ = _run(_scan([1.0, 2.0, 3.0]), rec)
    assert res["status"] == "ok"
    # point 2 runs twice (retry), points 1 and 3 once. compiled once each (reuse on retry).
    assert rec.runs == [1.0, 2.0, 2.0, 3.0]
    assert rec.compiles == [1.0, 2.0, 3.0]


def test_callbacks_receive_seq_num_and_arg():
    rec = Recorder()
    _run(_scan([5.0, 6.0]), rec, pre_cb=[rec.pre_cb], post_cb=[rec.post_cb])
    # pre_cb sees cur_seq_num BEFORE increment (0 then 1); arg is the scan index.
    assert rec.pre == [(0, 1), (1, 2)]
    assert rec.post == [(0, 1), (1, 2)]


def test_tstartwait_sleeps_each_shot():
    rec = Recorder()
    slept = []
    sc = SeqConfig()
    run_scan_group(_seqfn, _scan([1.0, 2.0]), tstartwait=0.2,
                   control=rec, compile_point=rec.compile_point, run_real=rec.run_real,
                   seq_config=sc, sleep=lambda dt: slept.append(dt), config_teardown=rec.teardown)
    assert slept == [0.2, 0.2]


# --------------------------------------------------------------------------- #
# rep == 0 (run-forever until abort) + random
# --------------------------------------------------------------------------- #
def test_rep_zero_runs_until_abort():
    # abort after 5 gate checks -> 4 completed shots over the 2-point scan looping forever.
    rec = Recorder(abort_after=5)
    res, _ = _run(_scan([1.0, 2.0]), rec, rep=0)
    assert res == {"status": "aborted", "nseq": 4}


def test_random_rep_runs_all_shots():
    import random
    rec = Recorder()
    res, _ = _run(_scan([1.0, 2.0, 3.0]), rec, rep=2, is_random=True,
                  rng=random.Random(0))
    # order scrambled, but every (point, rep) shot runs exactly once -> 6 shots, 3 compiles.
    assert res == {"status": "ok", "nseq": 6}
    assert sorted(rec.runs) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert sorted(rec.compiles) == [1.0, 2.0, 3.0]


def test_negative_rep_rejected():
    rec = Recorder()
    with pytest.raises(ValueError, match="negative"):
        _run(_scan([1.0]), rec, rep=-1)
