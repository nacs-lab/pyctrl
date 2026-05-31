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
| `tools/reference_list.m` | Registry of ~12 sequences to capture (placeholder names, byte round-trip) |
| `tools/reference_list_engine.m` | Registry of sequences with **real `config.yml` channel names** (engine-accepts check) |
| `tools/dummy_libnacs.py` | Board-free stand-in for the engine; lets byte-equality CI run with no Zynq board |
| `tests/` | pytest suite (byte round-trip + dummy-engine + real engine-accepts) |

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

The default run uses `tools/dummy_libnacs.py` (a board-free recorder) wherever an
engine-shaped object is needed, so it never loads `libnacs`.

### Interpreter for the engine checks

`libnacs` is **only importable under the Python 3.8 install at
`C:\Users\Ybtweezer-PC2\AppData\Local\Programs\Python\Python38`** (the same build
MATLAB's `pyenv` uses — but run it as a *separate OS process*, never inside
MATLAB). The default suite above runs fine under any modern Python (it is pure
stdlib). Because Python38 has no `pytest`, create an isolated venv that inherits
`libnacs` via system-site-packages but keeps `pytest` out of MATLAB's base env:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python38\python.exe" -m venv --system-site-packages .venv-engine
.\.venv-engine\Scripts\python -m pip install pytest
```

Engine and hardware checks are opt-in and should be run in a maintenance window:

```bash
# engine-accepts proof: compile-only, loads libnacs, no init_run/start
.venv-engine/Scripts/python -m pytest pyctrl -m needs_engine --real-engine
pytest pyctrl -m needs_hardware   # drives devices — stop the MATLAB experiment first
```

## Capturing more references (MATLAB side, engine-free)

Run **in a separate MATLAB session** (not the one running the experiment):

```matlab
cd pyctrl/tools
capture_matlab_reference        % byte round-trip refs -> tests/reference/<name>.bin
capture_matlab_reference(fullfile('..','tests','reference_engine'), @reference_list_engine)
                                % engine-accepts refs (real config.yml channel names)
```

Or headless, from a shell (a fresh session, so it can never collide with the
experiment's MATLAB):

```bash
matlab -batch "cd pyctrl/tools; capture_matlab_reference; capture_matlab_reference(fullfile('..','tests','reference_engine'), @reference_list_engine)"
```

`capture_matlab_reference.m` sets `SeqManager.override_tick_per_sec(1000)` so it
never loads the engine and never touches hardware, and calls `serialize()` only
(never `generate()`/`run()`).

## Safety model (same machine as the experiment)

- The Python process is separate from MATLAB's embedded interpreter.
- Byte round-trip / serialize comparison never loads the engine.
- The engine check compiles only (no `init_run`/`start`).
- Only one front-end may command the hardware at a time.
