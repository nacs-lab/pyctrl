# pyctrl — Python front-end for the Yb experiment control system

A Python front-end that builds experiment sequences, serializes them to the
**same byte format** the MATLAB stack uses, and drives the same `libnacs`
engine. It runs **in parallel** with `matlab_new/` (no changes to MATLAB code)
and is intended to be added to the `experiment-control` superproject as a git
submodule. See `../PYTHON_FRONTEND_PLAN.md` for the full phased plan.

## Status: Phase 0 (bootstrap & format pinning)

What exists so far:

| File | Role |
|------|------|
| `tools/compare_bytes.py` | Reader/encoder for the serialized byte format; field-level diff; `--selftest` |
| `lib/seq_manager.py` | Lazy wrapper over the `libnacs` engine (compile-only; no hardware) |
| `tools/capture_matlab_reference.m` | Engine-free MATLAB capture of `serialize()` output |
| `tools/reference_list.m` | Registry of sequences to capture |
| `tests/` | pytest suite (byte round-trip + engine-accepts) |

## Running the tests

Everything except the `needs_engine` / `needs_hardware` tests is pure
byte/structure math — safe to run at any time, including while an experiment is
in progress.

```bash
# default: no engine, no hardware (always safe on the lab PC)
pytest pyctrl

# quick reader check against the committed MATLAB references (no pytest needed)
python pyctrl/tools/compare_bytes.py --selftest matlab_new/lib/test

# decode one file / diff two files
python pyctrl/tools/compare_bytes.py matlab_new/lib/test/seq1.json
python pyctrl/tools/compare_bytes.py seq_matlab.bin seq_python.bin
```

Engine and hardware checks are opt-in and should be run in a maintenance window:

```bash
pytest pyctrl -m needs_engine     # compile-only; loads libnacs, no init_run/start
pytest pyctrl -m needs_hardware   # drives devices — stop the MATLAB experiment first
```

## Capturing more references (MATLAB side, engine-free)

Run **in a separate MATLAB session** (not the one running the experiment):

```matlab
cd pyctrl/tools
capture_matlab_reference        % writes tests/reference/<name>.bin + .params.json
```

`capture_matlab_reference.m` sets `SeqManager.override_tick_per_sec(1000)` so it
never loads the engine and never touches hardware, and calls `serialize()` only
(never `generate()`/`run()`).

## Safety model (same machine as the experiment)

- The Python process is separate from MATLAB's embedded interpreter.
- Byte round-trip / serialize comparison never loads the engine.
- The engine check compiles only (no `init_run`/`start`).
- Only one front-end may command the hardware at a time.
