"""dummy_libnacs.py -- a board-free stand-in for the libnacs engine.

Design inspired by brassboard-seq's ``tests/dummy_artiq.py`` (LGPL-v3+; this is
a from-scratch reimplementation, no code copied). It mirrors the slice of the
``libnacs.expseq_manager.Manager`` surface that the pyctrl checks touch, so that
byte-equality / harness tests can run on a machine **without** the Zynq board,
NI-DAQ DLLs, or the real engine library.

What it does:
  * ``create_sequence(data)`` decodes the byte array with ``compare_bytes`` (so
    a malformed blob raises, exactly like the real engine rejecting it) and
    returns a ``DummyExpSeq`` whose read-back getters return deterministic text
    derived from the decoded structure.
  * Every call is appended to ``Manager.transcript`` so a test can assert what
    the front-end asked the engine to do.

What it deliberately does NOT do:
  * ``init_run`` / ``start`` / ``pre_run`` / ``post_run`` raise
    ``NotImplementedError``. The dummy never advances hardware state, so a test
    that is mis-marked ``no_hardware`` but actually drives devices fails loudly
    instead of silently faking a run.

This is NOT the engine-accepts proof (that needs the real ``libnacs`` -- see
tests/test_engine_loads.py, marked ``needs_engine``). It is the always-safe
harness used by the default ``pytest`` run.
"""

import compare_bytes


class _Recorder:
    """Mixin that appends (name, args) tuples to a shared transcript list."""

    def __init__(self, transcript):
        self._transcript = transcript

    def _record(self, name, *args):
        self._transcript.append((name, args))


class DummyExpSeq(_Recorder):
    """Stand-in for the engine's compiled-sequence handle."""

    def __init__(self, transcript, seq):
        super().__init__(transcript)
        self.seq = seq  # decoded structure from compare_bytes.decode

    # --- read-back getters (compile-time views) ---------------------------- #
    def get_builder_dump(self):
        self._record("get_builder_dump")
        n = self.seq
        return ("DUMMY builder dump: version=%d nodes=%d channels=%d "
                "basicseqs=%d backenddatas=%d"
                % (n["version"], len(n["nodes"]), len(n["channels"]),
                   len(n["basicseqs"]), len(n["backenddatas"])))

    def get_seq_dump(self):
        self._record("get_seq_dump")
        return "DUMMY seq dump: %d basicseq(s)" % len(self.seq["basicseqs"])

    def get_seq_opt_dump(self):
        self._record("get_seq_opt_dump")
        return self.get_seq_dump()

    # --- hardware-shaped read-backs (still compile-only) ------------------- #
    def get_zynq_bytecode(self, dev):
        self._record("get_zynq_bytecode", dev)
        return b""

    def get_zynq_clock(self, dev):
        self._record("get_zynq_clock", dev)
        return b""

    def get_nidaq_data(self, dev):
        self._record("get_nidaq_data", dev)
        return []

    def get_nominal_output(self, pts):
        self._record("get_nominal_output", pts)
        return []

    # --- run path: never available on the dummy ---------------------------- #
    def _no_hardware(self, *_args, **_kw):
        raise NotImplementedError(
            "dummy_libnacs does not drive hardware (init_run/start/etc.); "
            "use the real engine in a downtime window for run tests")

    init_run = pre_run = post_run = start = wait = _no_hardware


class Manager(_Recorder):
    """Board-free stand-in for ``libnacs.expseq_manager.Manager``."""

    DEFAULT_TICK_PER_SEC = 1000000000000  # 1 ps, matches config.yml

    def __init__(self, transcript=None):
        super().__init__([] if transcript is None else transcript)
        self.transcript = self._transcript
        self._tick_per_sec = self.DEFAULT_TICK_PER_SEC
        self.config = None

    def load_config_string(self, config):
        self._record("load_config_string", len(config))
        self.config = config

    def load_config_file(self, fname):
        self._record("load_config_file", fname)
        with open(fname, "r") as f:
            self.config = f.read()

    def tick_per_sec(self):
        return self._tick_per_sec

    def create_sequence(self, data):
        self._record("create_sequence", len(data))
        # Decode validates the byte format and raises on a malformed/truncated
        # blob -- the same observable behavior as the real engine rejecting it.
        seq = compare_bytes.decode(bytes(data))
        return DummyExpSeq(self._transcript, seq)
