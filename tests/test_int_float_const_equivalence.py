"""needs_engine: does libnacs treat an int32 constant the same as a float64
constant of equal value?

Empirical backing for option B (let the Python front-end emit Python-natural
int constants) + a compare_bytes canonicalization that would absorb the
int/float tag difference vs MATLAB references, the way swappable comparisons are
already absorbed (see tests/test_compare_canonical.py).

We probe the engine compile-only (create_sequence; never pre_run/init_run/start)
by flipping one float64 constant in a real-config reference blob to an
equal-valued int32 and comparing the optimised IR (get_seq_opt_dump). The
device-codegen getters (get_zynq_bytecode / get_nidaq_data / get_nominal_output)
fault compile-only -- they are only populated by pre_run -- so the optimised IR
is the deepest artifact reachable without a run.

FINDINGS (this file pins them; if libnacs changes, these assertions flag it):

  1. ARITHMETIC OPERAND  -> fully equivalent. An int32 operand inside an op is
     promoted by the other operand, yielding byte-identical optimised IR. The
     "mixed int/float arithmetic might round differently" risk is absent.

  2. CHANNEL OUTPUT VALUE -> ACCEPTED and numerically equal, but the int32 tag
     is PRESERVED into the channel's compiled output (`O(...)=Int32 1` /
     `-final=Int32 1` vs `Float64 1`); the engine does NOT normalise it to the
     channel type. So unlike the comparison-swap case (provably identical at the
     engine's evaluation level), int32-vs-float64 on a channel value produces a
     genuinely different optimised IR. Whether the *hardware* bytecode is
     identical is NOT established here -- that needs a pre_run/run-level test in
     a maintenance window before option B can rely on it for channel values.

Run:  .venv-engine\\Scripts\\python -m pytest pyctrl/tests/test_int_float_const_equivalence.py -m needs_engine --real-engine -s
"""

import copy
import difflib
import os

import pytest

import compare_bytes
import seq_manager
from conftest import CONFIG_PATH, ENGINE_REF_DIR

pytestmark = pytest.mark.needs_engine


@pytest.fixture(scope="module")
def manager():
    if not seq_manager.engine_available():
        pytest.skip("libnacs engine not importable in this interpreter")
    mgr = seq_manager.get()
    config_path = CONFIG_PATH
    with open(config_path, "r") as f:
        mgr.load_config_string(f.read())
    mgr.enable_dump(True)
    return mgr


def _load(name):
    return compare_bytes.decode(compare_bytes.load(os.path.join(ENGINE_REF_DIR, name)))


def _opt(manager, seq_struct):
    """Compile (compile-only) and return the optimised-IR dump string."""
    eseq = manager.create_sequence(bytearray(compare_bytes.encode(seq_struct)))
    assert eseq is not None, "engine returned a null handle"
    dump = eseq.get_seq_opt_dump()
    assert dump, "empty optimised-IR dump"
    return dump


def _to_int32(arg):
    assert arg["argtype"] == "float64" and float(arg["val"]).is_integer()
    return {"argtype": "int32", "val": int(arg["val"])}


def _dds_freq_int32_variant():
    """eng_dds_set with the DDS-FREQ output value (node 3 = 1e8) made int32."""
    seq = _load("eng_dds_set.bin")
    seq["nodes"][2]["args"][0] = _to_int32(seq["nodes"][2]["args"][0])
    return seq


def _ttl_value_int32_variant():
    """eng_single_ttl with the TTL output value repointed to a fresh int32(1).

    A fresh node isolates the output-value position: node(1)=float64 1.0 stays in
    place for the time-delta / output-length uses, so only the *value* changes.
    """
    seq = _load("eng_single_ttl.bin")
    seq["nodes"].append({"op": 50, "args": [{"argtype": "int32", "val": 1}]})
    seq["basicseqs"][0]["outputs"][0]["val"] = len(seq["nodes"])  # 1-based id
    return seq


# --------------------------------------------------------------------------- #
# Finding 1: arithmetic operand -- int32 operand == float64 operand (identical).
# --------------------------------------------------------------------------- #
def test_int32_operand_yields_identical_ir(manager):
    base = _load("eng_analog_ramp.bin")            # node 3 = DIV(arg0, float64 1000.0)
    var = copy.deepcopy(base)
    var["nodes"][2]["args"][1] = _to_int32(var["nodes"][2]["args"][1])
    base_ir, var_ir = _opt(manager, base), _opt(manager, var)
    assert base_ir == var_ir, (
        "expected int32 operand to promote to an identical optimised IR\n"
        + "\n".join(difflib.unified_diff(base_ir.splitlines(), var_ir.splitlines(),
                                         "float64", "int32", lineterm="")))


# --------------------------------------------------------------------------- #
# Finding 2a: the engine ACCEPTS an int32 constant in a channel-value position.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", [_dds_freq_int32_variant, _ttl_value_int32_variant],
                         ids=["dds_freq", "ttl_value"])
def test_engine_accepts_int32_channel_value(manager, variant):
    # _opt asserts a non-null handle + non-empty optimised IR == accepted & built.
    assert _opt(manager, variant())


# --------------------------------------------------------------------------- #
# Finding 2b: but the int32 tag is PRESERVED (not normalised to the channel
# type) -- the value is equal, the type is not. This is the boundary that makes
# the int/float relaxation different from the comparison-swap relaxation.
# --------------------------------------------------------------------------- #
def test_int32_channel_value_type_is_preserved(manager):
    base = _opt(manager, _load("eng_dds_set.bin"))
    var = _opt(manager, _dds_freq_int32_variant())
    assert base != var, "engine unexpectedly normalised the const type"
    assert "Float64 1e+08" in base and "Int32 1e+08" not in base
    assert "Int32 100000000" in var, (
        "expected the channel output to carry the int32 tag with the same value")
