# pyctrl — Python front-end for the Yb experiment control system

pyctrl builds experiment sequences, serializes them to the byte format the
`libnacs` engine consumes, runs the scan loop (run / abort / pause), and drives
the FPGA / NI-DAQ / camera / AWG / SLM hardware. **It is the live runtime in
practice** — the scenario-3 backend (`python -m launcher.run_loop.runner <url>`)
drives real hardware; MATLAB (`matlab_new/`) still runs for the scans not yet
ported. pyctrl is a git submodule of the `experiment-control` superproject.

It began as a byte-identical re-implementation of the MATLAB `lib/` sequence
builder, and that heritage still governs the **serialize path**: a sequence's
`serialize()` bytes must match the blessed golden master (the single
MATLAB↔engine contract). That byte-equality is now a **regression guard** against
pyctrl's own reference, and a **one-time gate** when porting a new scan — it does
**not** constrain the run loop, the device drivers, provenance / `.seq` dumps, or
anything off the serialize path. Develop those Python-first. Full plan + porting
workflow: `../PYTHON_FRONTEND_PLAN.md` and the `pyctrl` skill.

## Status

Phases 0–5 done (value math; tree/timing; config/globals; ScanGroup; run loop —
live-verified end-to-end on real hardware). Phase 6 (experiment migration) is in
progress: scans are ported and blessed against the per-point byte oracle, then
confirmed with live A/B physics.

| Phase | What | State |
|------|------|-------|
| 0 | Bootstrap & format pinning (`compare_bytes.py`, `seq_manager.py`, reference capture) | ✅ |
| 1 | Value math & serializer (`SeqVal` + `SeqContext`, NODES/DATA/GLOBAL tables) | ✅ |
| 2 | Sequence tree & timing (`ExpSeq`/`RootSeq`/`SubSeq`/`TimeStep`) | ✅ |
| 3 | Config & globals (`SeqConfig`, `DynProps`/`SubProps`, `Consts()`) | ✅ |
| 4 | `ScanGroup` (EnableScan, DSL, materialization, `usevar`, `ScanAccessTracker`) | ✅ |
| 5 | Run loop (ExptServer / abort / pause; drives the engine + hardware) | ✅ |
| 6 | Experiment migration (`YbScans` ported + blessed) | 🔄 in progress |

Phase walkthroughs: `docs/phase{0,1,2,3,4,5}_walkthrough.html`.

**Comparison operators reflect — and that's OK.** Python has no `__rlt__`, so a
constant on the *left* of a comparison reflects: `3 < g` dispatches to
`g.__gt__(3)`, so the front-end serializes `GT{g, 3}` where MATLAB's `3 < g` is
`LT{3, g}` (and `==`/`!=` swap arg order similarly). These are the *same*
comparison (true even under IEEE-754 NaN), so **write the operators naturally** —
`3 < g` is fine. The byte comparator handles the equivalence: `compare_bytes.py`
has `normalize()` / `canonical_node()` that canonicalize swappable comparison
nodes (GT→LT, GE→LE with args swapped; EQ/NE args sorted), so a reflected form
verifies as **equivalent** to MATLAB's. Reflection only fires with a *constant* on
the left, so at most one operand is a compound sub-node and the node-graph ids stay
aligned — canonicalizing the comparison node alone is sufficient. It is *not* a
blanket relaxation: a genuine opcode mistake (e.g. `GT` where `LT` was meant, same
arg order) still diffs. Arithmetic (`+ - * / **`) and `& | xor` are byte-identical
either way — their reflected dunders preserve operand order. Use
`compare_bytes.py a.bin b.bin --strict` to require literal byte-identity.

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

The **default engine venv is now `.venv-engine-py312` (Python 3.12)** — a self-contained
venv off the anaconda 3.12 base with the full runtime stack + the (pure-Python) `libnacs`
bindings copied in (`config.PYCTRL_PYTHON` and `.vscode` point here). The default suite above
runs fine under any modern Python (it is pure stdlib); the engine/hardware checks need this
venv (it has both `libnacs` and `pylablib`).

**Legacy fallback (`.venv-engine`, Python 3.8):** `libnacs` is also importable under the Python
3.8 install at `C:\Users\Ybtweezer-PC2\AppData\Local\Programs\Python\Python38` (the same build
MATLAB's `pyenv` uses — but run it as a *separate OS process*, never inside MATLAB). That venv
was created as a `--system-site-packages` venv inheriting `libnacs`, with `pytest` added:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python38\python.exe" -m venv --system-site-packages .venv-engine
.\.venv-engine\Scripts\python -m pip install pytest
```

Engine and hardware checks are opt-in and should be run in a maintenance window:

```bash
# engine-accepts proof: compile-only, loads libnacs, no init_run/start
.venv-engine-py312/Scripts/python -m pytest pyctrl -m needs_engine --real-engine
.venv-engine-py312/Scripts/python -m pytest pyctrl -m needs_hardware   # drives devices — stop the MATLAB experiment first
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
