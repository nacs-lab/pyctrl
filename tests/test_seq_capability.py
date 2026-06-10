"""seq_capability: declarative per-seq run-loop flags the runner reads pre-compile.

NO-HARDWARE. Verifies the decorator stamps the flag, returns the function unchanged, defaults to
False when undeclared, and that the real rearrange seqs declare owns_frames=True.
"""

import pytest

from seq_capability import has_capability, seq_capabilities

pytestmark = pytest.mark.no_hardware


def test_default_false_when_undeclared():
    def plain(s):
        return s
    assert has_capability(plain, "owns_frames") is False
    assert has_capability(plain, "owns_frames", default=True) is True


def test_decorator_sets_flag_and_returns_same_function():
    @seq_capabilities(owns_frames=True)
    def seq(s):
        return ("built", s)
    assert seq.owns_frames is True
    assert has_capability(seq, "owns_frames") is True
    assert seq("x") == ("built", "x")          # still the original callable, unchanged behaviour


def test_decorator_default_is_false():
    @seq_capabilities()
    def seq(s):
        return s
    assert seq.owns_frames is False


def test_rearrange_seqs_declare_owns_frames():
    from RearrangeCommSeq import RearrangeCommSeq
    from RearrangeCommSeq2 import RearrangeCommSeq2
    assert has_capability(RearrangeCommSeq, "owns_frames") is True
    assert has_capability(RearrangeCommSeq2, "owns_frames") is True


def test_normal_seq_does_not_declare_it():
    from PushoutSurvivalSeq import PushoutSurvivalSeq
    assert has_capability(PushoutSurvivalSeq, "owns_frames") is False
