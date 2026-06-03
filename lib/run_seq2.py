"""run_seq2.py -- per-shot engine execution: ``run_real`` + ``run_bseq``.

Transliteration of ``ExpSeq.run_real`` / ``ExpSeq.run_bseq`` (``matlab_new/lib/ExpSeq.m:
369-463``). In MATLAB these are methods on ``ExpSeq``; pyctrl's ``ExpSeq`` is a pure
*builder* (serialize-only, Phases 1-4), so the run path lives here as module functions that
operate on a **generated runnable sequence** -- a built ``ExpSeq`` plus its libnacs engine
handle (``pyseq``) and the per-compile NI fields. This keeps the byte-only builder free of
any engine dependency (it is still tested with the engine never loaded).

The actual ``pyseq`` (the engine sequence object that ``generate()`` produces from the
serialized bytes) and the ``ni_channels`` / ``time_scale`` it carries are wired up in the
run-loop hosting step (engine integration); here everything is **duck-typed** so the call
order is verified NO-HARDWARE against a fake ``pyseq`` + fake ``nidaq``.

Runnable-sequence interface (what a generated ExpSeq must expose):

    seq.C                  config DynProps; ``seq.C.RESTART = 0`` is settable
    seq.pyseq              engine handle (see below)
    seq.basic_seqs         list of BasicSeq (idx 2.. -> basic_seqs[idx-2]); idx 1 is seq
    seq.before_start_cbs   root-level, fire once before the first bseq
    seq.after_end_cbs      root-level, fire once after the last bseq
    bseq.before_bseq_cbs / .after_bseq_cbs / .after_branch_cbs   per-bseq, passed the ROOT
    seq.ni_channels        list (possibly empty) of analog-out channels
    seq.config.ni_clocks / .ni_start external clock / trigger maps (NiDAQRunner.run args)
    seq.time_scale         ticks-per-second (tail wall-clock pause)
    seq.reset_globals(persist)       engine global reset

    pyseq.init_run() / pre_run() / start() / cur_bseq_length()->int / post_run()->int
    pyseq.get_nidaq_data(name) -> flat sequence of doubles, or None for an analog-free bseq
    pyseq.wait(timeout_ms) -> bool  (False until the shot finishes; polled in a loop)

VERIFIED call order (ExpSeq.m:369-463):
  run_real:  before_start_cbs* -> init_run -> loop[run_bseq] -> after_end_cbs* ->
             reset_globals -> tail wall-clock pause.   catch: reset_globals + re-raise.
  run_bseq:  before_bseq_cbs* -> pre_run -> [if ni: get_nidaq_data -> NiDAQRunner.run] ->
             start -> wait-poll-loop -> [if ni armed: NiDAQRunner.wait] -> after_bseq_cbs*
             -> cur_bseq_length -> post_run(next_idx) -> after_branch_cbs*.

NI None-guard (finding C, path a -- no MATLAB-style ``VMOTCoil=0`` hack at runtime): the
``ni_channels`` gate is seq-level (per-compile) but ``get_nidaq_data`` is per-bseq; an
analog-free bseq returns ``None``. pyctrl skips BOTH the NI arm and ``NiDAQRunner.wait`` for
that bseq -- physically correct (no clock is emitted), and it keeps ``get_nidaq_data``
INSIDE ``run_bseq`` so a branched seq never clocks stale analog. (Ported rearrange seqs
still carry ``VMOTCoil=0`` for *byte* parity; that is a build-time concern, not here.)

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import time


def run_bseq(seq, idx, nidaq=None):
    """Run one basic sequence; return ``(next_idx, bseq_len)``.

    ``bseq`` is the root ``seq`` when ``idx == 1`` else ``seq.basic_seqs[idx - 2]`` (MATLAB
    1-based ``basic_seqs{idx-1}``). All callbacks are invoked with the ROOT ``seq``.
    """
    pyseq = seq.pyseq
    if pyseq is None:
        raise RuntimeError("Sequence must be generated before running.")
    bseq = seq if idx == 1 else seq.basic_seqs[idx - 2]

    for cb in bseq.before_bseq_cbs:
        cb(seq)
    pyseq.pre_run()

    # --- NI DAQ: arm BEFORE start (FPGA TTL0 then triggers it). Per-bseq None-guard. ---
    ni_armed = False
    ni_channels = getattr(seq, "ni_channels", None)
    if ni_channels:
        ni_data = pyseq.get_nidaq_data("NiDAQ")
        if ni_data is not None:                 # analog-free bseq -> skip arm AND wait
            ni_nchns = len(ni_channels)
            ni_ndata = len(ni_data)
            if ni_ndata % ni_nchns != 0:
                raise ValueError(
                    "NI DAQ data length %d is not a multiple of channel count %d"
                    % (ni_ndata, ni_nchns))
            if nidaq is None:
                from nidaq_runner import NiDAQRunner as nidaq  # lazy: needs hardware pkg
            data = _reshape_sample_major(ni_data, ni_nchns)
            nidaq.run(ni_channels, seq.config.ni_clocks, seq.config.ni_start, data)
            ni_armed = True

    pyseq.start()
    while not pyseq.wait(100):                   # poll-loop (MATLAB: while ~wait(pyseq,100))
        pass
    if ni_armed:
        if nidaq is None:
            from nidaq_runner import NiDAQRunner as nidaq
        nidaq.wait()

    for cb in bseq.after_bseq_cbs:
        cb(seq)
    bseq_len = pyseq.cur_bseq_length()           # real engine call (easy to miss)
    next_idx = int(pyseq.post_run())
    for cb in bseq.after_branch_cbs:
        cb(seq)
    return next_idx, bseq_len


def run_real(seq, nidaq=None, clock=None, sleep=None):
    """Run the full (possibly multi-bseq) sequence once.

    Args:
        seq: a generated runnable sequence (see module docstring).
        nidaq: NiDAQRunner-like (injected in tests; lazily imported in production).
        clock: ``() -> seconds`` wall clock for the tail pause (default ``time.time``;
            MATLAB uses ``now()*86400``). Only time *differences* matter.
        sleep: ``seconds -> None`` (default ``time.sleep``).
    """
    if clock is None:
        clock = time.time
    if sleep is None:
        sleep = time.sleep

    seq.C.RESTART = 0
    bseq_len = 0
    start_t = clock()
    try:
        for cb in seq.before_start_cbs:
            cb(seq)
        seq.pyseq.init_run()
        idx = 1
        while idx != 0:
            start_t = clock()                    # captured PER-BSEQ; last value is used below
            idx, bseq_len = run_bseq(seq, idx, nidaq=nidaq)
        for cb in seq.after_end_cbs:             # inside the try (MATLAB ExpSeq.m:445-447)
            cb(seq)
    except BaseException:
        seq.reset_globals(False)                 # error path: reset then re-raise
        raise
    # success path: reset globals AFTER the shot (so values set before the run are observable)
    seq.reset_globals(False)

    # Tail wall-clock pause: wait until the LAST bseq's start_t + its length (- 50 ms).
    end_after = start_t + bseq_len / seq.time_scale - 50e-3
    end_t = clock()
    if end_t < end_after:
        sleep(end_after - end_t)


def _reshape_sample_major(ni_data, ni_nchns):
    """Flat channel-major engine data -> a ``[nsamps, nchns]`` matrix (MATLAB
    ``reshape(ni_data, [ni_ndata/ni_nchns, ni_nchns])``, column-major).

    The flat blob is channel-contiguous (all samples of ch1, then ch2, ...), so column ``j``
    of the result is channel ``j``. ``NiDAQRunner.run`` (nidaq_runner.py, item 6) does the
    further ``[nsamps, nchns] -> [nchns, nsamps]`` transpose that ``nidaqmx`` needs -- the #1
    silent-bug risk, kept in ONE place and covered by the separate NI shape test.
    """
    nsamps = len(ni_data) // ni_nchns
    try:
        import numpy as np
        return np.reshape(np.asarray(ni_data, dtype=float), (nsamps, ni_nchns), order="F")
    except ImportError:
        # numpy-free fallback: list of nsamps rows, each [ch0..chN] (sample-major).
        return [[float(ni_data[c * nsamps + s]) for c in range(ni_nchns)]
                for s in range(nsamps)]
