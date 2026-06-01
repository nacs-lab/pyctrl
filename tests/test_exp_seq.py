"""test_exp_seq.py -- Phase-2 acceptance: port of matlab_new/lib/test/TestExpSeq.m.

Builds each TestExpSeq test1..test7 sequence in pyctrl and asserts serialize()
is byte-identical to the committed MATLAB oracle (matlab_new/lib/test/seq{1..6}.json),
plus the intermediate cur_time / total_time / toString checks (which catch
timing-graph drift earlier than the final byte compare). Runs on the empty test
config at tick_per_sec = 1000 (NO-HARDWARE).
"""

import os

import pytest

import seq_manager
import compare_bytes
from end_time import end_time
from exp_seq import ExpSeq
from seq_val import to_string

pytestmark = pytest.mark.no_hardware

# matlab_new/lib/test/, relative to the pyctrl submodule root (tests/ -> ../..).
_TESTDIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'matlab_new', 'lib', 'test'))


@pytest.fixture(autouse=True)
def _tick():
    # TestExpSeq runs at tick_per_sec = 1000 (engine-free).
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)


def _want(name):
    return compare_bytes.load(os.path.join(_TESTDIR, name))


def _txt(name):
    with open(os.path.join(_TESTDIR, name), encoding='latin-1') as f:
        # Normalize CRLF: the committed refs may be checked out CRLF on Windows,
        # but toString emits LF (see memory reference_matlab_test_crlf).
        return f.read().replace('\r\n', '\n')


def _assert_bytes(s, json_name, repeatable=True):
    got = s.serialize()
    want = _want(json_name)
    if got != want:
        d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
        raise AssertionError('%s: first diff at %s' % (json_name, d))
    if repeatable:
        assert s.serialize() == got, '%s: serialize not repeatable' % json_name


def test1():
    s = ExpSeq()
    s.add_step(1).add('Device1/CH1', 4)
    assert s.cur_time() == 1
    s.conditional(False).add_step(0.1).add('Device1/CH5', 3)
    assert s.cur_time() == 1
    s.conditional(True).add_step(0.1004).add('Device2/CH3', -1)
    assert s.cur_time() == 1.1
    s.wait(2.3)
    assert s.cur_time() == 3.4
    s.conditional(False).wait(100)
    assert s.cur_time() == 3.4

    g = s.new_global()
    s.wait(g)
    assert to_string(s.cur_time()) == '(3400 + round(g(0) * 1000)) / 1000'
    m1 = s.add_measure('Device2/CH5')
    s.conditional(m1 < 0).wait(3.4)
    assert to_string(s.cur_time()) == \
        '(3400 + ifelse(m(3) < 0, 3400, 0) + round(g(0) * 1000)) / 1000'
    s.add_step(1.2).add('Device1/CH5', lambda t: t - 2.3)
    assert to_string(s.total_time()) == \
        '(4600 + ifelse(m(3) < 0, 3400, 0) + round(g(0) * 1000)) / 1000'
    assert s.to_string() + '\n' == _txt('seq1.txt')

    s.wait_background()
    assert s.to_string() + '\n' == _txt('seq1.txt')
    s.wait_all()
    assert s.to_string() + '\n' == _txt('seq1.txt')
    _assert_bytes(s, 'seq1.json')


def _build_test2_3():
    s = ExpSeq()
    g = s.new_global()
    s.add('Device2/CH2', g + 2)

    def subseq(sub, length):
        m = sub.add_measure('Device2/CH2')
        (sub.add_step(length)
            .add('Device2/CH2', 3.4)
            .add('Device3/CH1', lambda t: t * 5 - m))
        sub.add('Device2/CH2', 0).add('Device3/CH1', 0)

    s.add_step(subseq, g * 0.2)
    assert to_string(s.cur_time()) == 'round(g(0) * 0.2 * 1000) / 1000'
    assert to_string(s.total_time()) == 'round(g(0) * 0.2 * 1000) / 1000'
    s.add_background(subseq, 0.4)
    assert to_string(s.cur_time()) == 'round(g(0) * 0.2 * 1000) / 1000'
    assert to_string(s.total_time()) == \
        'max(round(g(0) * 0.2 * 1000), 400 + round(g(0) * 0.2 * 1000)) / 1000'
    assert s.to_string() + '\n' == _txt('seq2.txt')
    return s


def test2():
    s = _build_test2_3()
    s.wait_background()
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    s.wait_background()  # no-op
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    s.wait_all()
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    s.wait_all()  # no-op
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    _assert_bytes(s, 'seq2_waitbg.json')


def test3():
    s = _build_test2_3()
    # waitAll and waitBackground are equivalent + idempotent on this shape.
    s.wait_all()
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    s.wait_all()
    s.wait_background()
    s.wait_background()
    assert s.to_string() + '\n' == _txt('seq2_waitbg.txt')
    _assert_bytes(s, 'seq2_waitbg.json')


def test4():
    s = ExpSeq()
    g = s.new_global()
    step = s.conditional(False).add_background(g * 2, g + 2)
    step.add('Device0/CH9', g / 2)
    assert s.cur_time() == 0
    assert s.total_time() == 0
    assert s.to_string() + '\n' == _txt('seq3.txt')
    s.wait_for(step)
    assert s.cur_time() == 0
    assert s.total_time() == 0
    assert s.to_string() + '\n' == _txt('seq3_waitfor.txt')
    s.wait_background()
    s.wait_all()
    assert s.to_string() + '\n' == _txt('seq3_waitfor.txt')
    _assert_bytes(s, 'seq3_waitfor.json')


def test5():
    s = ExpSeq()
    g = s.new_global()
    step = s.add_floating(5)
    s.wait(g * 4)
    step.set_time(end_time(s))
    s.wait_for(step)
    assert s.to_string() + '\n' == _txt('seq4.txt')
    assert to_string(s.total_time()) == \
        'max(round(g(0) * 4 * 1000), 5000 + round(g(0) * 4 * 1000)) / 1000'
    _assert_bytes(s, 'seq4.json')


def test6():
    s = ExpSeq()
    g = s.new_global()
    step = s.add_floating(5)
    s.wait(g * 4)
    step.set_end_time(end_time(s))
    s.wait_for(step)
    assert s.to_string() + '\n' == _txt('seq5.txt')
    assert to_string(s.total_time()) == 'round(g(0) * 4 * 1000) / 1000'
    _assert_bytes(s, 'seq5.json')


def test7():
    s = ExpSeq()
    g = s.new_global()
    step = s.add_floating(5)
    s.wait(g * 4)
    s.align_end(step)
    assert s.to_string() + '\n' == _txt('seq6.txt')
    assert to_string(s.total_time()) == \
        'max(round(g(0) * 4 * 1000), 5000 + round(g(0) * 4 * 1000)) / 1000'
    # alignEnd serialize is NOT repeatable (a fresh end time is minted each call).
    _assert_bytes(s, 'seq6.json', repeatable=False)
