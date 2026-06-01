"""get_my_seq.py -- transliteration of ``matlab_new/YbSeqs/get_my_seq.m``.

nargin-1 singleton seq: one DDS-amp step plus a wait sized by config defaults. The
``regAfterEnd(@dispHello)`` closure is deferred (no-op here); disp dropped. TWait3 is
computed but only used by the (dropped) closure -- kept for faithfulness.
"""


def _noop(s1):
    pass


def get_my_seq(s):
    s.reg_after_end(_noop)             # dispHello (deferred closure)

    s.add_step(1e-3).add('FPGA1/DDS0/AMP', 0)

    TWait = s.C.TWait(1.5)
    TWait2 = s.C.TWait2(1.1)
    TWait3 = s.C.KRb.TWait(2)          # noqa: F841 -- only used by the deferred closure
    s.wait(TWait + TWait2)

    return s
