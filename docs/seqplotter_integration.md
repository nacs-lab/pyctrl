# SeqPlotter integration — flattened `.seq` output from pyctrl

> Goal: when a sequence is run with pyctrl, automatically save a copy of the
> *flattened* sequence (the `.seq` format SeqPlotter consumes) and display it in
> the `yb_analysis` dashboard via SeqPlotter's web UI — with **scanned parameters
> highlighted** when the run was a scan.
>
> This doc records what is already built, the verified format, and the concrete
> hooks for the later dashboard work, so the remaining implementation is
> mechanical.

## TL;DR

- `.seq` is **not** `serialize()`. `serialize()` is the *symbolic* blob the libnacs
  engine compiles. `.seq` is the *evaluated* per-channel `(time, value, pulse_id)`
  output, produced by `ExpSeq.dump_output_to_file` from the engine's
  `get_nominal_output(pts_per_ramp)`. SeqPlotter reads `.seq`.
- pyctrl does **not** need a Python evaluator for this — the **engine** produces the
  points. pyctrl only orchestrates the engine call and packs the bytes.
- `.seq` is an **output-only viewer artifact** (not fed to the engine), so it is
  *not* bound by THE ONE RULE (byte-equality with MATLAB). Byte-equality is a
  useful validation, but we are also free to **enrich** the params block with a
  `scanned` marker for highlighting.

## What is already built (verified)

| Piece | File | Status |
|---|---|---|
| Runnable `.seq` format spec — decode / encode / diff CLI | `tools/compare_seq_bytes.py` | ✅ round-trips the real 132,855-byte MATLAB sample byte-for-byte |
| Ground-truth fixture (real MATLAB `.seq` from nacs-lab/SeqPlotter) | `tests/reference/seqplotter_sample_ryddet.seq` | ✅ committed |
| `.seq` writer (engine output → bytes), alias decoration, params, scanned marker | `lib/dump_output.py` | ✅ unit-tested (no engine) |
| Tests | `tests/test_dump_output.py` | ✅ 10 passing, `no_hardware` |

Run the spec tool directly:

```
python tools/compare_seq_bytes.py FILE              # summary
python tools/compare_seq_bytes.py A B               # first differing field
python tools/compare_seq_bytes.py A B --no-name     # ignore the seq_name timestamp
python tools/compare_seq_bytes.py --selftest DIR    # round-trip every *.seq
```

## The verified `.seq` byte format

Sourced from MATLAB (`matlab_new/lib/ExpSeq.m`) and confirmed by an exact
round-trip of the real sample. All multibyte values little-endian.

```
[nseqs: uint32]
per seq:
  [seq_name\0]                       # "yyyymmdd_HHMMSS:<name>" (ExpSeq.m:595) — has a timestamp
  [seq_idx: uint32]                  # 1 for the first basic sequence, 2.. for branches
  [nchns: uint32]
  per channel:                       # from get_nominal_output (ExpSeq.m:686)
    [chn_name\0]                     # alias-decorated: "alias1, alias2 (nominal)" (ExpSeq.m:699-721)
    [npts: uint32]
    per point: [time int64 8B][value float64 8B][pulse_id uint32 4B]   # == '<qdI', 20 bytes
  [has_params: uint8]
  [params\0]                         # present iff has_params; JSON (see below)
[has_bt_info: uint8]                 # debug backtrace; not needed for plotting
if has_bt_info:
  [bt_idx: uint32] x nseqs
  [n_bts: uint32]
  per bt: [nfilenames u32][name\0..][nnames u32][name\0..]
          [nobjs u32][ [nframes u32][ fname_id u32, name_id u32, line u32 ]xnframes ]xnobjs
```

Notes / gotchas:
- The comment at `ExpSeq.m:663` claims a trailing `[has_params]` after the
  backtrace objs. The **code** (`get_debug_output`, `ExpSeq.m:735`) does not emit
  it — the exact round-trip confirms the code is right. `compare_seq_bytes.py`
  follows the code.
- `seq_name` is the **only nondeterministic field** (it embeds `datestr(clock)`).
  Byte-equality tests against a MATLAB `.seq` must inject a fixed timestamp or
  compare with `--no-name` / `diff(..., ignore={"seq_name"})`.
- `pulse_id` is 0-indexed. SeqPlotter carries it as Plotly `customdata` so a point
  can be traced back to the pulse / source line via the backtrace block.

## Params-JSON schema (drives the highlight feature)

From a real captured `.seq`, each parameter leaf is:

```json
"VShimZeroX": {"value": -0.04, "type": 1, "config_value": -0.04}
```

- `value` — the resolved value for this run. (633/633 leaves)
- `type`  — libnacs type tag (1/2/3 = bool/int32/float64). (633/633)
- `config_value` — the config default, present only when one exists. (57/633)
- (`old_value` appears in the SeqPlotter docs but not in this sample.)

**There is no native "this parameter was scanned" flag.** So highlighting needs one
of:
1. **Inference** — `value != config_value`, or `value` differs across scan points.
   Cheap, but flags *any* override, not specifically scanned axes.
2. **Explicit marker (recommended)** — pyctrl injects `"scanned": true` (and
   optionally `"scan_dim": n`) into the scanned leaves. Implemented now in
   `dump_output.mark_scanned(params, scanned_paths, scan_dims)`. Safe because
   `.seq` is a viewer artifact; a strict-parity `.seq` simply omits the call.

## How a pyctrl run will produce a `.seq` (later wiring)

`lib/dump_output.py` is the seam. When `ExpSeq`/the run loop exist:

```python
import dump_output
# eseq = SeqManager.create_sequence(serialized_bytes)   # already supported (Phase 0)
data = dump_output.dump_output(eseq, pts_per_ramp=100,
                               seq_name=dump_output.format_seq_name("RydDet", datetime.now()),
                               params=params_dict,           # Phase 3 (convert_seqval_to_string port)
                               inverse_chn_map=chn_map)      # Phase 3 (channel map)
# scan run: enrich first
params_dict = dump_output.mark_scanned(params_dict, scanned_paths, scan_dims)  # Phase 4
```

- Single basic sequence: `dump_output()` (compile/eval only — **no** `init_run`,
  safe on a compiled handle).
- Branches / multiple basic sequences: `dump_output_branches()` walks
  `init_run → pre_run → get_nominal_output → post_run` like
  `ExpSeq.dump_output_to_file` (`ExpSeq.m:598-624`). `init_run` is NEEDS-HARDWARE
  per the plan — run with the real engine in a downtime window (the board-free
  dummy raises on `init_run` by design).

Auto-save: in the run loop, after each sequence/scan-point, write the bytes next
to the run's data (e.g. `<run_dir>/<seq_name>.seq`).

## Dashboard display (later)

SeqPlotter (nacs-lab/SeqPlotter) is itself a Dash + Plotly app (`waitress`-served),
which matches the `yb_analysis` dashboard stack. Two options:

1. **Standalone (start here)** — run SeqPlotter as-is; point it at the directory
   where pyctrl writes `.seq` files (it has a "Load Latest File" path mode and a
   drag-drop upload). Zero coupling, cleanest license separation.
2. **Embedded** — fold SeqPlotter's `home.py` plotting into a page of the
   `yb_analysis` dashboard so a run's sequence is viewable alongside live images.
   More work; do after standalone is proven.

Highlighting the scanned params requires a small SeqPlotter-side change: where it
renders the params table, read `leaf.get("scanned")` and style those rows/markers.
(Confirm the exact spot in `home.py` — the params-display code path.)

## Remaining work, by phase

| Item | Needs | Phase |
|---|---|---|
| Channel-data `.seq` from pyctrl bytes (nominal names) | engine `get_nominal_output` (already provable) | **now / 1–2** |
| Alias-decorated channel names | the channel map (`inverse_chn_map`) | 3 |
| Params-JSON block | port of `convert_seqval_to_string` + `DynProps`/`Consts` | 3 |
| Per-scan-point params + `scanned` marker driven by real scan axes | `ScanGroup.get_seq(n)` + which axes are swept | 4 |
| Byte-equality vs a MATLAB `.seq` (modulo `seq_name`) | capture one MATLAB `.seq` in a downtime window | 5 (downtime) |
| Auto-save on run + dashboard wiring | run loop + dashboard | 5–6 |
| SeqPlotter highlight patch (`leaf["scanned"]`) | small change to SeqPlotter `home.py` | when embedding |

## Open item

A byte-for-byte equality test of a pyctrl `.seq` against a MATLAB `.seq` needs one
captured MATLAB reference (`s.dump_output_to_file(pts, file, name)` during
downtime, with a fixed `seq_name`). Until then, correctness rests on: (a) the exact
round-trip of the real sample through `compare_seq_bytes.py`, and (b) the existing
Phase-2 plan check that the engine's `get_nominal_output` from pyctrl bytes equals
MATLAB's — which is exactly the data the `.seq` repackages.
