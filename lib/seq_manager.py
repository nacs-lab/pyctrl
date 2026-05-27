"""seq_manager.py -- thin wrapper over the libnacs engine.

Mirrors matlab_new/lib/SeqManager.m: a process-wide handle to the compiled
libnacs C++ engine, reached through its ctypes binding
(`libnacs.expseq_manager.Manager`). The byte array produced by the Python
serializer is handed to `create_sequence` exactly as MATLAB does today.

NOTE ON SAFETY: importing this module does NOT load the engine. The engine is
loaded lazily by `get()`. `create_sequence` only compiles -- it does not call
init_run / start, so it does not drive hardware. Even so, on the shared lab PC
prefer to exercise it in a maintenance window (see PYTHON_FRONTEND_PLAN.md).
"""

_MANAGER = None


def get():
    """Return the process-wide engine Manager, importing libnacs on first use."""
    global _MANAGER
    if _MANAGER is None:
        from libnacs.expseq_manager import Manager  # imported lazily on purpose
        _MANAGER = Manager()
    return _MANAGER


def reset():
    """Drop the cached Manager (next get() rebuilds it)."""
    global _MANAGER
    _MANAGER = None


def engine_available():
    """True if the libnacs engine library can be imported in this interpreter."""
    try:
        import libnacs.expseq_manager  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import/load failure means "not available"
        return False


def load_config_string(config):
    get().load_config_string(config)


def load_config_file(fname):
    get().load_config_file(fname)


def tick_per_sec():
    return int(get().tick_per_sec())


def create_sequence(data):
    """Compile a serialized sequence. `data` must be a mutable bytearray.

    The C binding takes the buffer via `ctypes.c_uint8.from_buffer`, which
    requires a writable buffer -- pass `bytearray(...)`, not `bytes`.
    """
    if not isinstance(data, bytearray):
        data = bytearray(data)
    return get().create_sequence(data)
