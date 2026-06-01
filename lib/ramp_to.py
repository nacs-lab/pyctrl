"""ramp_to.py -- transliteration of ``matlab_new/lib/rampTo.m``.

Factory returning a 3-arg pulse callable that linearly ramps from the channel's
previous value (``old_val`` = ``arg1``) to ``vend`` over the step. No constant-fold.
"""


def ramp_to(vend):
    def func(t, length, old_val):
        return (old_val * (length - t) + vend * t) / length

    return func
