"""Phase-5 generate() validation against the REAL libnacs engine (compile-only).

`needs_engine`, COMPILE-ONLY: builds real ported YbSeqs and calls ``ExpSeq.generate()``,
which is ``create_sequence(serialize())`` + ``get_nidaq_channel_info`` + ``reset_globals`` --
NO ``init_run``/``start``, so it drives NO hardware (create_sequence compiles to bytecode; it
does not connect to the FPGA). Proves pyctrl can turn a built sequence into a runnable engine
handle whose API matches run_seq2.py's pyseq interface -- the last piece before an actual run.

Run in a maintenance window (MATLAB off):
    .venv-engine-py312\\Scripts\\python -m pytest tests/test_generate_engine.py -m needs_engine --real-engine
"""

import inspect

import pytest

import seq_manager
from conftest import CONFIG_PATH
from exp_seq import ExpSeq
from seq_config import SeqConfig

pytestmark = pytest.mark.needs_engine


def _build(name):
    """Build a YbSeq, handling both arities: nargin-1 (CoreShellMOTSeq(s)) takes a configured
    ExpSeq; nargin-0 (DummySeq()) creates and returns its own."""
    seqfn = getattr(__import__(name), name)
    if len(inspect.signature(seqfn).parameters) == 0:
        return seqfn()
    s = ExpSeq()
    seqfn(s)
    return s

# The libnacs engine ExpSeq run API that run_seq2.run_real / run_bseq rely on.
_RUN_API = ("init_run", "pre_run", "start", "wait", "post_run",
            "cur_bseq_length", "get_nidaq_data", "get_nidaq_channel_info",
            "get_global", "set_global")

_SEQS = ["CoreShellMOTSeq", "GreenMOTSeq", "DummySeq"]


@pytest.fixture(scope="module")
def engine_cfg():
    if not seq_manager.engine_available():
        pytest.skip("libnacs engine not importable in this interpreter")
    mgr = seq_manager.get()
    cfg = CONFIG_PATH
    with open(cfg) as f:
        mgr.load_config_string(f.read())
    mgr.enable_dump(True)
    # Production path: real expConfig for the build; the engine provides the real tick rate
    # (1 ps = 1e12). Do NOT override tick -- let serialize() use the engine's own rate.
    SeqConfig.reset()
    SeqConfig.load_real()
    yield mgr
    SeqConfig.reset()
    seq_manager.reset()


@pytest.mark.parametrize("name", _SEQS)
def test_generate_compiles_and_wires_pyseq(engine_cfg, name):
    s = _build(name)
    s.generate()

    assert s.pyseq is not None, "%s: generate() produced no engine handle" % name
    for m in _RUN_API:
        assert hasattr(s.pyseq, m), "%s: engine handle missing %s()" % (name, m)
    # ni_channels parsed to {chn:int, dev:str} (empty for a TTL/DDS-only seq).
    assert isinstance(s.ni_channels, list)
    for ch in s.ni_channels:
        assert set(ch) == {"chn", "dev"} and isinstance(ch["chn"], int)
    # A non-empty builder dump confirms the engine actually parsed our bytes.
    assert s.pyseq.get_builder_dump()


def test_generate_is_idempotent(engine_cfg):
    s = _build("DummySeq")
    s.generate()
    first = s.pyseq
    s.generate()                          # second call is a no-op (isempty(pyseq) guard)
    assert s.pyseq is first


def test_reset_globals_runs_on_compiled_handle(engine_cfg):
    # reset_globals (called inside generate(true)) must not raise; for a seq with globals it
    # writes init_vals to the engine via set_global -- still compile-only (no init_run/start).
    s = _build("DummySeq")
    s.generate()
    s.reset_globals(False)                # explicit re-reset: exercises the per-shot path
