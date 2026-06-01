"""ramp_to_sqrt.py -- transliteration of ``matlab_new/lib/rampToSqrt.m``.

Factory returning a 3-arg pulse callable that interpolates in the *square* domain
from the previous value ``v0`` (= ``arg1``) to ``v1``:
``sqrt((v1^2 - v0^2)/len * t + v0^2)``. ``v0 ** 2`` on the SeqVal arg folds to
OP_MUL[v0, v0] (the ``power(.,2)`` fold); ``v1 ** 2`` on a number stays numeric.
"""

from seq_val import sqrt


def ramp_to_sqrt(v1):
    def func(t, length, v0):
        return sqrt((v1 ** 2 - v0 ** 2) / length * t + v0 ** 2)

    return func
