"""test_basic_seq.py -- multi-basic-sequence + branch structural coverage.

TestExpSeq and reference_list never call new_basic_seq / cond_branch with a
non-end target (every committed reference has nbasicseqs=1, nbranches=0,
default_target=0), so the multi-bseq + branch byte layout is otherwise unpinned
until the Phase-3 RearrangeCommSeq* real captures. This test exercises that path:
build a 2-bseq sequence with a conditional branch + a default branch, and assert
the decoded structure (target_id == bseq_id, default_target, bseq_id arithmetic)
and that it re-encodes byte-identically. NO-HARDWARE.
"""

import pytest

import seq_manager
import compare_bytes
from basic_seq import BasicSeq
from exp_seq import ExpSeq

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)


def test_new_basic_seq_id_and_branch_layout():
    s = ExpSeq()
    s.add_step(1).add('Device1/CH1', 1)
    g = s.new_global()

    bseq2 = s.new_basic_seq()
    assert isinstance(bseq2, BasicSeq)
    # ExpSeq is bseq_id 1; the first new_basic_seq is bseq_id 2 (position+1).
    assert s.bseq_id == 1
    assert bseq2.bseq_id == 2
    bseq2.add_step(1).add('Device1/CH1', 0)

    # Conditional branch from the root to bseq2; default falls through to "end".
    s.cond_branch(g > 0, bseq2)
    s.default_branch(None)

    got = s.serialize()
    seq = compare_bytes.decode(got)

    assert len(seq['basicseqs']) == 2
    b0 = seq['basicseqs'][0]          # the ExpSeq itself
    assert len(b0['branches']) == 1
    assert b0['branches'][0]['target_id'] == 2     # == bseq2.bseq_id
    assert b0['default_target'] == 0               # end of sequence
    # The second bseq has no branches and falls through to end.
    b1 = seq['basicseqs'][1]
    assert len(b1['branches']) == 0
    assert b1['default_target'] == 0

    # Decoder is the byte gate: re-encode must match exactly.
    assert compare_bytes.encode(seq) == got


def test_default_branch_to_bseq():
    s = ExpSeq()
    s.add_step(1).add('Device1/CH1', 1)
    bseq2 = s.new_basic_seq()
    bseq2.add_step(1).add('Device1/CH1', 0)
    s.default_branch(bseq2)           # default target is a real bseq, not end

    got = s.serialize()
    seq = compare_bytes.decode(got)
    assert seq['basicseqs'][0]['default_target'] == 2
    assert compare_bytes.encode(seq) == got


def test_branch_target_must_be_rootseq():
    s = ExpSeq()
    with pytest.raises(Exception):
        s.cond_branch(True, object())     # not a RootSeq -> error
