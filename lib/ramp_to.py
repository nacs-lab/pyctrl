"""ramp_to.py -- smootherstep ramp from the channel's previous value to ``vend``.

Factory returning a 3-arg pulse callable (``old_val`` = ``arg1``) that ramps from
``old_val`` to ``vend`` over the step using the quintic smootherstep easing
``s(u) = 6u^5 - 15u^4 + 10u^3`` (``u = t/length``): slope AND acceleration are zero
at both ends (C^2 smooth), so the channel eases in and out with no kink and no
velocity jump. This is the DEFAULT ramp. For the original straight-line ramp
(``rampTo.m`` transliteration) use ``ramp_to_linear``. No constant-fold.
"""


def ramp_to(vend):
    def func(t, length, old_val):
        u = t / length
        s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
        return old_val + (vend - old_val) * s

    return func
