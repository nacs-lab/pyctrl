# autocal -- continuous background-calibration system

Keep a tweezer pattern (or several, cycled) in calibration **continuously and autonomously**, riding
the ExptServer *background* queue lane so it never interferes with foreground experiments.

## Why this is safe (it cannot break a running experiment)

It runs only on the **background lane** (added in the `dummy queueing` commit):

- A background scan **starts only when the rig is foreground-idle** (`runner._try_pop_background`
  checks `get_background_enabled()` and `has_foreground_work()`).
- A running background scan **yields at the next shot boundary** the instant a foreground scan is
  queued (`run_seq.run_one` -> `control.should_yield()`), and the yield **never touches the
  Pause/Abort flag**, so the incoming foreground scan sees a clean slate.
- Submitting/enabling is purely additive (a queue row); it does not touch a running scan.

Because background scans are yielded *constantly*, **no single run is assumed sufficient**: results
accumulate per `(pattern, cal)` (inverse-variance pooling of per-point survival) until a target
shot count, then a fit is cut. This is the whole point of `autocal.pooling` + the ledger
accumulator.

## Two layers (running vs reacting)

- **Layer 1 -- the controller daemon** (`run_controller.py`, deterministic). Ingests finished cal
  scans, pools, fits, applies in-band config (journaled for rollback), enqueues the next cal,
  upholds the home-pattern invariant. Robust to the agent being asleep/compacted/dead.
- **Layer 2 -- the Claude agent supervisor** (a `/loop`). Reacts to the ledger: sanity-checks the
  controller's decisions, handles alerts/anomalies, retunes the rotation, and escalates to the user.
  It is **stateless across wakes** -- it rebuilds its world from the ledger every tick, so context
  compaction loses nothing.

You can run Layer 1 alone (mechanical), Layer 2 alone (agent does the mechanics too, slower), or
both (recommended).

## State (single source of truth) -- `<DATA>/yb_dashboard_state/calibration/`

- `ledger.json`   -- per-pattern cals (status / last_fit / accumulator / history), settings, alerts,
  controller runtime. **Only the controller writes it.**
- `changes.jsonl` -- append-only audit + rollback journal (every auto-applied config change + revert).
- `commands.jsonl`-- append-only command queue the **dashboard** writes and the controller consumes
  (rollback / toggle / set-baseline / ...). The dashboard never writes config itself.

## The rotation (most important, measure-only cals) -- `rotation.CAL_DEFS`

| cal | scan | catches | autonomy | cadence |
|---|---|---|---|---|
| `556mj0` | Spectrum556Scan --mj 0 | clock-line drift | **auto-apply** per-pattern resonance (in-band) | ~1 h |
| `556mj1` | Spectrum556Scan --mj 1 | trap-depth / light-shift | trend + flag (trap-depth feedback is user-gated) | ~2 h |
| `399` | Spectrum399Scan | imaging-resonance drift | trend only | ~6 h |
| loading | (derived from the above) | SLM/pattern loading health | flag if rate drops | -- |

In-band drift on a config-applying cal is auto-applied; out-of-band / edge-pinned / poor-fit /
hologram-or-cooling changes are **flagged for the user**, never done autonomously.

## Dashboard -- the "Auto-Calibrations" tab

`yb_analysis` Dash app: per-pattern status cards, center-vs-fit trend plots (band shaded, applied
points marked), open alerts, the change log with rollback ids, and controls (toggle lane /
auto-apply / auto-cycle, roll back a change). HTTP: `GET /api/autocal/state`,
`GET /api/autocal/changes`, `POST /api/autocal/command`.

## Staged bring-up (each stage is gated; the live ones need the user + a safe window)

`run_controller.py` defaults to **observe-only**. Enable one stage at a time:

```
# Stage A -- OBSERVE (no submit, no config write, no SLM switch). Validate ingest + dashboard.
python -m autocal.run_controller --home 33x33_uniform --once
# Stage B -- also enqueue home-pattern background cal scans (cycle=True)
python -m autocal.run_controller --home 33x33_uniform --enable-submit
# Stage C -- also auto-apply in-band config  (requires the UNVERIFIED write_config adapter wired)
python -m autocal.run_controller --home 33x33_uniform --enable-submit --enable-apply
# Stage D -- also auto-cycle non-home patterns (requires the UNVERIFIED SLM adapters + home-restore)
python -m autocal.run_controller --home 33x33_uniform --cycle-patterns 47x47_uniform \
    --enable-submit --enable-apply --enable-cycle
```

The Claude supervisor (Layer 2) is launched as a self-paced loop, e.g. `/loop` with a prompt like:
*"Read the autocal ledger (GET /api/autocal/state). For each pattern/cal: is the latest fit sane
(R2, edge, loading)? Is any drift real vs noise? Did the controller apply the right things? Raise/clear
alerts, retune cadence/bands or add/remove cals per the experiment-running skill, and escalate
anything out-of-band. Then schedule the next wake."* It keeps no state in context -- the ledger is
its memory.

## The live-test gate (what still needs the rig / the user)

Phase 0 (this drop) is **no-hardware** and unit-tested (`tests/test_autocal.py`). Before going live:

1. **Confirm with the user** in a safe window (no sensitive foreground run).
2. **Stage A/B** verify the read path on-rig: the `queue_list` identity field carrying the
   `autocal::cal::pattern` label (see `make_query_backend`'s NOTE), and `analyze_scan` pooling.
3. **Stage C** requires wiring `write_config` (per-pattern expConfig write + oracle re-capture) --
   marked `NotImplementedError` until done + verified.
4. **Stage D** requires wiring `switch_pattern`/`restore_home` (SLM writes + the home-pattern
   invariant), respecting the DMA-stall history -- marked `NotImplementedError` until done + verified.
