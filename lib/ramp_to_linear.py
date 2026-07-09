"""ramp_to_linear.py -- transliteration of ``matlab_new/lib/rampTo.m``.

Factory returning a 3-arg pulse callable that LINEARLY ramps from the channel's
previous value (``old_val`` = ``arg1``) to ``vend`` over the step. No constant-fold.
This is the original straight-line ramp; the default ``ramp_to`` is now the quintic
smootherstep eased version.
"""


def ramp_to_linear(vend):
    def func(t, length, old_val):
        return (old_val * (length - t) + vend * t) / length

    return func
