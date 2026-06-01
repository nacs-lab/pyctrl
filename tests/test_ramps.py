"""test_ramps.py -- the ramp factories build the exact MATLAB SeqVal expression.

No committed reference uses linearRamp/rampTo/sqrtRamp/rampToSqrt (TestExpSeq +
reference_list use raw @(t) callables), so these are structural: each factory's
closure, dispatched through TimeStep.add with (t=arg0/ts, len=raw_len/ts,
old=arg1), must produce a SeqVal tree structurally equal to the hand-written
formula. A byte-equality test against a MATLAB capture using these helpers is a
follow-up (would add builders to reference_list.m + recapture).
"""

import pytest

import seq_manager
from exp_seq import ExpSeq
from linear_ramp import linear_ramp
from ramp_to import ramp_to
from ramp_to_sqrt import ramp_to_sqrt
from seq_val import seqval_isequal, sqrt
from sqrt_ramp import sqrt_ramp

pytestmark = pytest.mark.no_hardware


@pytest.fixture(autouse=True)
def _tick():
    seq_manager.override_tick_per_sec(1000)
    yield
    seq_manager.override_tick_per_sec(0)


def _ramp_val(ramp):
    # Build a step (raw_len = 2000 ticks -> L = 2.0, so the outer /L does NOT fold
    # away via the b==1 rule) and capture the pulse value the ramp produced.
    s = ExpSeq()
    step = s.add_step(2)
    step.add('Device1/CH1', ramp)
    t = s.seq_ctx.arg0 / s.time_scale
    length = step.raw_len / s.time_scale
    old = s.seq_ctx.arg1
    return step.pulses[1].val, t, length, old


def test_linear_ramp():
    val, t, length, old = _ramp_val(linear_ramp(2, 5))
    assert seqval_isequal(val, (2 * (length - t) + 5 * t) / length)


def test_linear_ramp_const_fold():
    assert linear_ramp(3, 3) == 3        # equal numeric endpoints -> bare scalar
    assert sqrt_ramp(3, 3) == 3


def test_ramp_to():
    val, t, length, old = _ramp_val(ramp_to(5))
    assert seqval_isequal(val, (old * (length - t) + 5 * t) / length)


def test_sqrt_ramp():
    val, t, length, old = _ramp_val(sqrt_ramp(2, 5))
    assert seqval_isequal(val, sqrt((2 * (length - t) + 5 * t) / length))


def test_ramp_to_sqrt():
    val, t, length, old = _ramp_val(ramp_to_sqrt(5))
    assert seqval_isequal(val, sqrt((5 ** 2 - old ** 2) / length * t + old ** 2))
