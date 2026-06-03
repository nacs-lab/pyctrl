"""set_chns.py -- set FPGA TTL/DDS (and any engine) channels to static values.

pyctrl mirror of ``matlab_new/seqs/setChns.m``: build a one-shot ``ExpSeq`` that adds each
``(channel, value)`` at t=0 and RUN it through the engine. This is the manual "set channel"
operator tool for FPGA channels -- the counterpart to :func:`nidaq_io_handler.set_channel`
(the NI AO direct write).

Channel naming (backend, no alias needed):
    TTL:  ``"FPGA1/TTL31"``                         (value 0/1)
    DDS:  ``"FPGA1/DDS9/FREQ"`` (Hz), ``"FPGA1/DDS9/AMP"`` (0..1)

⚠ **Not an isolated write.** Unlike the NI AO direct path, this RUNS A SEQUENCE through the
engine (``init_run``/``start``), so at t=0 the engine drives EVERY configured channel to its
expConfig default, applies the channels you set, then holds. It resets the whole experiment to
its default/idle state PLUS your channels -- exactly as MATLAB ``setChns`` does (its standard
behavior). The MATLAB ``ResetMemoryMap`` is dropped (pyctrl has no memmap). NEEDS-HARDWARE.

Args accepted (mirrors setChns varargin): flat ``('FPGA1/TTL31', 1, 'FPGA1/DDS9/AMP', 0.1)``
or pairs ``[('FPGA1/TTL31', 1), ('FPGA1/DDS9/AMP', 0.1)]``.
"""


def build_set_chns(*args):
    """Build + ``generate()`` (compile) the one-shot ExpSeq, but DO NOT run it.

    Compile-only -- creates the engine handle (no ``init_run``/``start``), so it drives no
    hardware. Use to validate the channels/values before firing. Returns the generated ExpSeq.
    """
    from exp_seq import ExpSeq
    s = ExpSeq()
    for name, val in _parse_pairs(args):
        s.add(name, float(val))
    s.generate()
    return s


def set_chns(*args):
    """Mirror of ``setChns``: build + RUN. ⚠ Drives the FPGA AND resets all channels to
    their defaults (see module note). Returns the run ExpSeq."""
    from run_seq2 import run_real
    s = build_set_chns(*args)
    run_real(s)
    return s


def _parse_pairs(args):
    # pairs form: a single list of [name, val] pairs
    if (len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0]
            and isinstance(args[0][0], (list, tuple))):
        return [(str(n), float(v)) for n, v in args[0]]
    # flat form: name, val, name2, val2, ... (also tolerates inline (name, val) tuples)
    pairs = []
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, (list, tuple)) and len(a) == 2:
            pairs.append((str(a[0]), float(a[1])))
            i += 1
        else:
            if i + 1 >= len(args):
                raise ValueError("set_chns: no value for channel %r" % (a,))
            pairs.append((str(a), float(args[i + 1])))
            i += 2
    if not pairs:
        raise ValueError("set_chns: specify at least one channel")
    return pairs
