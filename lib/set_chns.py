"""set_chns.py -- set FPGA (TTL/DDS) and NI analog-out channels to static values.

pyctrl mirror of ``matlab_new/seqs/setChns.m``: build a one-shot ``ExpSeq`` that adds each
``(channel, value)`` at t=0 and RUN it through the engine. This is the manual "set channel"
operator tool. It handles BOTH backend families that the engine drives:

  * **FPGA** TTL/DDS channels -- latched directly by the FPGA bytecode (set-and-held).
  * **NI** PCIe-6738 analog-out channels (``V*`` / ``Dev1/N``) -- clocked out by the
    FPGA-driven AO path, then held by the DAC (the PCIe-6738 holds its last value).

For a single NI channel WITHOUT the full reset-to-defaults below, the lighter-weight direct
DC write :func:`devices.nidaq.nidaq_io_handler.set_channel` is an alternative.

Channel naming (an expConfig alias OR the raw backend name -- no alias needed):
    TTL:  ``"TTLBlueMOTShutter"`` or ``"FPGA1/TTL31"``                   (value 0/1)
    DDS:  ``"Freq556"``/``"FPGA1/DDS9/FREQ"`` (Hz), ``"Amp556"``/``"FPGA1/DDS9/AMP"`` (0..1)
    NI :  ``"VElectrode1"`` or ``"Dev1/12"``  -> backend ``NiDAQ/Dev1/12``  (DC volts, +-10 V)

⚠ **Not an isolated write.** Unlike the NI AO direct path, this RUNS A SEQUENCE through the
engine (``init_run``/``start``), so at t=0 the engine drives EVERY configured channel to its
expConfig default, applies the channels you set, then holds. It resets the whole experiment to
its default/idle state PLUS your channels -- exactly as MATLAB ``setChns`` does (its standard
behavior). The MATLAB ``ResetMemoryMap`` is dropped (pyctrl has no memmap). NEEDS-HARDWARE.

⚠ **NI channels must be CLOCKED to take effect.** An NI AO channel only updates while the FPGA
emits its sample clock (PFI0); the bare set-at-t=0 sequence MATLAB ``setChns`` builds has ~zero
duration, so the NI would get no clock edges and might never latch. When any channel set here
resolves to NI, this tool therefore holds the sequence for ``ni_hold`` seconds (default 1 ms,
~400 clock edges at 400 kHz) so the value clocks out; the DAC then holds it after the task
closes (like ``setV``). A call with no NI channel is left zero-length (byte-identical to before).
Pass ``ni_hold=0`` for byte-exact MATLAB ``setChns`` parity (NI may then not latch).

Args accepted (mirrors setChns varargin): flat ``('FPGA1/TTL31', 1, 'VElectrode1', 2.0)``
or pairs ``[('FPGA1/TTL31', 1), ('VElectrode1', 2.0)]``.
"""

_NI_PREFIX = "NiDAQ"      # translated NI channel names start with this (e.g. "NiDAQ/Dev1/12")
_DEFAULT_NI_HOLD = 1e-3   # s held so the FPGA clocks the NI AO value out (~400 samples @ 400 kHz)


def build_set_chns(*args, ni_hold=_DEFAULT_NI_HOLD):
    """Build + ``generate()`` (compile) the one-shot ExpSeq, but DO NOT run it.

    Compile-only -- creates the engine handle (no ``init_run``/``start``), so it drives no
    hardware. Use to validate the channels/values before firing. Returns the generated ExpSeq.

    If any channel resolves to an NI analog-out channel (translated name under ``NiDAQ/``), a
    ``ni_hold``-second hold is appended so the FPGA clocks the value out (see the module note);
    ``ni_hold=0`` skips it (and an all-FPGA call never adds one).
    """
    from exp_seq import ExpSeq
    s = ExpSeq()
    for name, val in _parse_pairs(args):
        s.add(name, float(val))
    # NI AO is a CLOCKED device: extend the sequence so the FPGA emits sample-clock edges and
    # the DAC latches the value. FPGA TTL/DDS need no hold, so a no-NI call stays zero-length
    # (byte-identical to MATLAB setChns). channel_names are already alias-translated here.
    if ni_hold and any(n.startswith(_NI_PREFIX) for n in s.channel_names):
        s.wait(ni_hold)
    s.generate()
    return s


def set_chns(*args, ni_hold=_DEFAULT_NI_HOLD):
    """Mirror of ``setChns``: build + RUN. ⚠ Drives the FPGA + NI AO AND resets all channels to
    their defaults (see module note). Returns the run ExpSeq."""
    from run_seq2 import run_real
    s = build_set_chns(*args, ni_hold=ni_hold)
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
