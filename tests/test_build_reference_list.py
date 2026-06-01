"""test_build_reference_list.py -- build the reference_list.m sequences in pyctrl
and assert serialize() is byte-identical to the committed tests/reference/*.bin.

This is the broad Phase-2 build-equality net: the 12 representative shapes
(empty, single TTL, DDS set, analog ramp, two/multi channel, nested subseq,
conditional, global, measure, floating-align, 64-step long_seq) captured from
MATLAB by tools/capture_matlab_reference.m + reference_list.m. Ports of those
builders here must reproduce the bytes exactly (tick_per_sec = 1000, NO-HARDWARE).
"""

import os

import pytest

import seq_manager
import compare_bytes
from end_time import end_time
from exp_seq import ExpSeq

pytestmark = pytest.mark.no_hardware

_REFDIR = os.path.join(os.path.dirname(__file__), 'reference')


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)


def build_empty():
    return ExpSeq()


def build_single_ttl():
    s = ExpSeq()
    s.add_step(1e-3).add('Device1/CH1', True)
    return s


def build_dds_set():
    s = ExpSeq()
    s.add_step(1e-3).add('Device1/FREQ', 100e6).add('Device1/AMP', 0.5)
    return s


def build_analog_ramp():
    s = ExpSeq()
    s.add_step(2e-3).add('Device1/CH2', lambda t: t * 500)
    return s


def build_two_channel():
    s = ExpSeq()
    s.add_step(1e-3).add('Device1/CH1', True).add('Device2/CH3', -1)
    return s


def build_multi_channel():
    s = ExpSeq()
    (s.add_step(1e-3)
        .add('Device1/CH1', True)
        .add('Device2/CH3', -1)
        .add('Device3/CH7', 2.5)
        .add('Device1/CH2', lambda t: t * 100))
    return s


def build_nested_subseq():
    s = ExpSeq()

    def subseq(sub, length):
        m = sub.add_measure('Device2/CH2')
        (sub.add_step(length)
            .add('Device2/CH2', 3.4)
            .add('Device3/CH1', lambda t: t * 5 - m))
        sub.add('Device2/CH2', 0).add('Device3/CH1', 0)

    s.add_step(subseq, 0.4)
    s.add_background(subseq, 0.4)
    s.wait_all()
    return s


def build_conditional():
    s = ExpSeq()
    s.add_step(1).add('Device1/CH1', 4)
    s.conditional(False).add_step(0.1).add('Device1/CH5', 3)
    s.conditional(True).add_step(0.1004).add('Device2/CH3', -1)
    return s


def build_with_global():
    s = ExpSeq()
    g = s.new_global()
    s.add('Device2/CH2', g + 2)
    s.add_step(1e-3).add('Device2/CH2', 0)
    return s


def build_with_measure():
    s = ExpSeq()
    s.add_step(1).add('Device1/CH1', 4)
    m1 = s.add_measure('Device2/CH5')
    s.conditional(m1 < 0).wait(3.4)
    s.add_step(1.2).add('Device1/CH5', lambda t: t - 2.3)
    return s


def build_floating_align():
    s = ExpSeq()
    g = s.new_global()
    step = s.add_floating(5)
    s.wait(g * 4)
    step.set_end_time(end_time(s))
    s.wait_for(step)
    return s


def build_long_seq():
    s = ExpSeq()
    for i in range(1, 65):
        s.add_step(1e-3).add('Device1/CH1', i % 2 == 0)
    return s


_BUILDERS = {
    'empty': build_empty,
    'single_ttl': build_single_ttl,
    'dds_set': build_dds_set,
    'analog_ramp': build_analog_ramp,
    'two_channel': build_two_channel,
    'multi_channel': build_multi_channel,
    'nested_subseq': build_nested_subseq,
    'conditional': build_conditional,
    'with_global': build_with_global,
    'with_measure': build_with_measure,
    'floating_align': build_floating_align,
    'long_seq': build_long_seq,
}


@pytest.mark.parametrize('name', sorted(_BUILDERS), ids=sorted(_BUILDERS))
def test_build_matches_reference(name):
    s = _BUILDERS[name]()
    got = s.serialize()
    want = compare_bytes.load(os.path.join(_REFDIR, name + '.bin'))
    if got != want:
        d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
        raise AssertionError('%s: first diff at %s' % (name, d))
    assert got == want
