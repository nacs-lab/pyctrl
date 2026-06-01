#!/usr/bin/env python3
"""fuzz_programs.py -- randomized differential generator for the SeqContext serializer.

Plan PYTHON_FRONTEND_PLAN.md Phase 1 calls for a *randomized* byte-equality test:
build random value trees in BOTH MATLAB and Python from the same spec and assert the
serialized tables match. Hand-written tests only cover the expressions the porter
thought of; a fuzzer catches folding-order, interning, and DFS-id divergences they miss.

This module is the single source of truth for both sides:

  * ``generate_programs(seed, n)``  -> a list of language-neutral "program" specs
    (pure JSON data: which globals/measures/args exist, a list of operation steps,
    and an ordered list of getValID emits). Generation is deterministic given the
    seed (Python's Mersenne-Twister ``random`` is stable across platforms).
  * ``build_python(program)``       -> {'node','data','global'} hex of the three
    SeqContext tables, by replaying the spec with the real ``SeqVal``/``SeqContext``.
  * ``capture_fuzz_reference.m``    replays the SAME ``programs.json`` in MATLAB and
    writes ``fuzz_reference.json`` (the committed ground truth).

The replay design that keeps the two sides byte-identical:
  - Every operation is driven through the module-level ``seq_val`` functions (the exact
    implementations MATLAB's operator methods call), so construction-time constant
    folding is identical.
  - Unary ops only ever act on a *symbolic* operand (global/measure/arg/SeqVal), so no
    operation ever takes the pure-numeric stdlib fallback path -- that would risk a
    cross-platform libm ULP mismatch that has nothing to do with serialization.
  - Binary ops always take at least one symbolic operand, for the same reason; the
    other may be an interned constant (exercising inline ArgConst* encoding).

Usage:
    python fuzz_programs.py gen  tests/reference_fuzz/programs.json   # (re)generate corpus
    python fuzz_programs.py show tests/reference_fuzz/programs.json   # print python tables
"""

import json
import random
import sys

import numpy as np

import seq_val as sv
from seq_val import SeqVal
from seq_context import SeqContext
from interpolate import interpolate
from ifelse import ifelse


# --- operation tables: name -> (arity, python callable) --------------------- #
_BINARY = {
    "add": sv.plus, "sub": sv.minus, "mul": sv.times, "div": sv.rdivide,
    "pow": sv.power, "ldiv": sv.ldivide, "atan2": sv.atan2, "hypot": sv.hypot,
    "rem": sv.rem, "max": sv.max, "min": sv.min,
    "and": sv.and_, "or": sv.or_, "xor": sv.xor,
    "lt": sv.lt, "gt": sv.gt, "le": sv.le, "ge": sv.ge, "eq": sv.eq, "ne": sv.ne,
}
_UNARY = {
    "abs": sv.abs_, "exp": sv.exp, "floor": sv.floor, "log": sv.log, "sqrt": sv.sqrt,
    "sin": sv.sin, "cos": sv.cos, "atan": sv.atan, "erf": sv.erf, "round": sv.round,
    "not": sv.not_, "neg": (lambda a: -a), "uplus": (lambda a: +a),
}
BINARY_OPS = sorted(_BINARY)
UNARY_OPS = sorted(_UNARY)

# Constant pools (all exactly representable / round-trip-stable through JSON). The
# int boundaries exercise int32 min/max byte patterns.
FLOATS = [0.0, 1.0, -1.0, 0.5, -0.5, 1.3, 2.6, 3.5, -0.097076, 100.0, 0.25, 4.5, -2.0]
INTS = [-3, -1, 0, 1, 2, 4, 7, 23, -2147483648, 2147483647]
BOOLS = [0, 1]
GLOBAL_TYPES = [SeqVal.TYPE_BOOL, SeqVal.TYPE_INT32, SeqVal.TYPE_FLOAT64]


# --------------------------------------------------------------------------- #
# Replay: turn a program spec into the three SeqContext tables.
# --------------------------------------------------------------------------- #
def _ref(k, idx=0, num=0):
    return {"k": k, "idx": int(idx), "num": num}


def _resolve(ctx, env, ref):
    k = ref["k"]
    if k == "g":
        return env["globals"][ref["idx"]]
    if k == "m":
        return env["measures"][ref["idx"]]
    if k == "a":
        return ctx.get_arg(ref["idx"])
    if k == "v":
        return env["values"][ref["idx"]]
    if k == "f":
        return float(ref["num"])
    if k == "i":
        return np.int32(ref["num"])
    if k == "b":
        return bool(ref["num"])
    raise ValueError("bad ref kind %r" % k)


def _apply_step(ctx, env, step):
    op = step["op"]
    a = _resolve(ctx, env, step["a"])
    if op in _UNARY:
        val = _UNARY[op](a)
    elif op in _BINARY:
        b = _resolve(ctx, env, step["b"])
        val = _BINARY[op](a, b)
    elif op == "interp":
        b = _resolve(ctx, env, step["b"])
        c = _resolve(ctx, env, step["c"])
        val = interpolate(a, b, c, [float(x) for x in step["data"]])
    elif op == "ifelse":
        b = _resolve(ctx, env, step["b"])
        c = _resolve(ctx, env, step["c"])
        val = ifelse(a, b, c)
    else:
        raise ValueError("unknown op %r" % op)
    env["values"].append(val)
    return val


def _make_env(ctx, program):
    globals_ = [ctx.new_global(int(t))[0] for t in program["globals"]]
    measures = [ctx.new_measure()[0] for _ in range(program["nmeasure"])]
    return {"globals": globals_, "measures": measures, "values": []}


def build_python(program):
    """Replay a program spec; return hex of the node/data/global tables."""
    ctx = SeqContext()
    env = _make_env(ctx, program)
    for step in program["steps"]:
        _apply_step(ctx, env, step)
    for ref in program["emits"]:
        ctx.get_val_id(_resolve(ctx, env, ref))
    return {
        "node": ctx.node_serialized().hex(),
        "data": ctx.data_serialized().hex(),
        "global": ctx.global_serialized().hex(),
    }


# --------------------------------------------------------------------------- #
# Generation: build a random but kind-valid program (builds it in Python as it
# goes so it can pick operands of the right kind, then records the spec).
# --------------------------------------------------------------------------- #
def _symbolic_value_refs(env):
    return [_ref("v", i) for i, v in enumerate(env["values"]) if isinstance(v, SeqVal)]


def _symbolic_sources(env, program):
    """Refs that are guaranteed SeqVals: globals, measures, args, symbolic values."""
    srcs = [_ref("g", i) for i in range(len(env["globals"]))]
    srcs += [_ref("m", i) for i in range(len(env["measures"]))]
    srcs += [_ref("a", i) for i in range(program["nargs"])]
    srcs += _symbolic_value_refs(env)
    return srcs


def _any_operand_ref(rng, env, program):
    """A symbolic source OR an interned constant."""
    pool = _symbolic_sources(env, program)
    kind = rng.random()
    if kind < 0.45 or not pool:
        c = rng.random()
        if c < 0.5:
            return _ref("f", num=rng.choice(FLOATS))
        if c < 0.8:
            return _ref("i", num=rng.choice(INTS))
        return _ref("b", num=rng.choice(BOOLS))
    return rng.choice(pool)


def _generate_one(rng):
    nglobals = rng.randint(1, 3)
    program = {
        "globals": [rng.choice(GLOBAL_TYPES) for _ in range(nglobals)],
        "nmeasure": rng.randint(0, 2),
        "nargs": rng.randint(1, 3),
        "steps": [],
        "emits": [],
    }
    ctx = SeqContext()
    env = _make_env(ctx, program)

    nsteps = rng.randint(5, 11)
    for _ in range(nsteps):
        srcs = _symbolic_sources(env, program)
        roll = rng.random()
        if roll < 0.30:
            op = rng.choice(UNARY_OPS)
            step = {"op": op, "a": rng.choice(srcs), "b": _ref("none"),
                    "c": _ref("none"), "data": []}
        elif roll < 0.80:
            op = rng.choice(BINARY_OPS)
            step = {"op": op, "a": rng.choice(srcs),
                    "b": _any_operand_ref(rng, env, program),
                    "c": _ref("none"), "data": []}
        elif roll < 0.92:
            ndata = rng.randint(2, 6)
            step = {"op": "interp", "a": rng.choice(srcs), "b": rng.choice(srcs),
                    "c": rng.choice(srcs), "data": [rng.choice(FLOATS) for _ in range(ndata)]}
        else:
            step = {"op": "ifelse", "a": rng.choice(srcs),
                    "b": _any_operand_ref(rng, env, program),
                    "c": _any_operand_ref(rng, env, program), "data": []}
        _apply_step(ctx, env, step)
        program["steps"].append(step)

    # Emits: every SYMBOLIC value (in order), every global/measure/arg, plus some
    # consts and deliberate repeats (constant interning + node dedup).
    #
    # We deliberately do NOT emit a bare boolean ``false`` (nor folded-constant
    # values, which can be ``false``): MATLAB's getValID has a latent bug --
    # const_b_ids(int8(false)) indexes element 0 and throws -- so serializing a
    # standalone false const is unsupported on the MATLAB side (inline ArgConstBool
    # via serializeArg, exercised by ifelse operands below, works fine). ``true``
    # is safe. pyctrl handles both, so this restriction only keeps the differential
    # comparison on the surface MATLAB can actually produce.
    emits = [_ref("v", i) for i, v in enumerate(env["values"]) if isinstance(v, SeqVal)]
    emits += [_ref("g", i) for i in range(len(env["globals"]))]
    emits += [_ref("m", i) for i in range(len(env["measures"]))]
    emits += [_ref("a", i) for i in range(program["nargs"])]
    for _ in range(rng.randint(2, 4)):
        emits.append(_ref("f", num=rng.choice(FLOATS)))
        emits.append(_ref("i", num=rng.choice(INTS)))
    emits.append(_ref("b", num=1))      # true only (see note above)
    # repeats: re-emit a couple of earlier emits to exercise dedup paths
    if len(emits) > 3:
        emits.append(dict(emits[rng.randint(0, len(emits) - 1)]))
        emits.append(dict(emits[rng.randint(0, len(emits) - 1)]))
    program["emits"] = emits
    return program


def generate_programs(seed=20260601, n=40):
    rng = random.Random(seed)
    return [_generate_one(rng) for _ in range(n)]


# --------------------------------------------------------------------------- #
def _main(argv):
    if len(argv) >= 3 and argv[1] == "gen":
        progs = generate_programs()
        with open(argv[2], "w") as f:
            json.dump(progs, f, indent=0)
        print("wrote %d programs to %s" % (len(progs), argv[2]))
        return 0
    if len(argv) >= 3 and argv[1] == "show":
        with open(argv[2]) as f:
            progs = json.load(f)
        for i, p in enumerate(progs):
            t = build_python(p)
            print("prog %2d  node=%4dB data=%3dB global=%3dB"
                  % (i, len(t["node"]) // 2, len(t["data"]) // 2, len(t["global"]) // 2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
