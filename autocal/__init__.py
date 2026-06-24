"""autocal -- continuous background-calibration system (the "auto-calibrations" lane).

A small, deterministic controller that keeps a tweezer pattern (or several, cycled) in
calibration by riding the ExptServer *background* queue lane (see ``YbExptCtrl/runner.py``
``_try_pop_background`` / ``lib/run_seq.py`` yield-at-shot-boundary). It submits the most
important measure-only calibration scans as low-priority ``background=True, cycle=True`` jobs,
pools their results across the many short (frequently-yielded) partial runs, fits the pooled
curve, and -- for in-band reversible quantities -- auto-applies the update while journaling
every change for one-click rollback.

DESIGN INVARIANTS (why this is safe to run during real experiments):
  * The lane only ever runs when the rig is foreground-idle and yields instantly when a
    foreground scan is submitted (enforced ENTIRELY by the ExptServer/run_seq machinery, not
    here). This package never aborts or pauses a foreground scan.
  * Because a background scan is yielded constantly, NO single run is assumed sufficient.
    Results accumulate per ``(pattern, calibration)`` until a target shot count is reached,
    THEN a fit is taken (see :mod:`autocal.pooling`). The accumulator is grid-checked so a
    changed scan definition resets it instead of mixing apples and oranges.
  * Only the controller writes config; the dashboard requests changes/rollbacks via an
    append-only command queue the controller consumes (see :mod:`autocal.rollback`).
  * Pattern switching (auto-cycle) upholds the "home pattern" invariant: a foreground scan
    always finds the home pattern on the SLM (the controller restores home after each non-home
    calibration; the hardware-touching half is gated to the on-rig phases).

Layout (this Phase-0 drop is the no-hardware FOUNDATION -- pure logic + IO adapters that are
import-guarded / dependency-injected, so the whole package is unit-testable with no engine, no
backend, no hardware, and no yb_analysis env):
  * :mod:`autocal.paths`      -- where the ledger + journals live (mirrors pattern_grid prefix).
  * :mod:`autocal.ledger`     -- the per-pattern state schema, atomic IO, cross-run accumulator.
  * :mod:`autocal.rollback`   -- the append-only change journal + revert + dashboard command queue.
  * :mod:`autocal.rotation`   -- the calibration definitions, cadence/bands, pattern eligibility,
                                 and the "what to run next" scheduler.
  * :mod:`autocal.pooling`    -- inverse-variance pooling of per-point survival + the analyze_scan
                                 adapter (guarded) that turns a scan into per-point data.
  * :mod:`autocal.fit`        -- fit a pooled spectrum (reuses yb_analysis fittings when present).
  * :mod:`autocal.submit`     -- build a background descriptor for a (cal, pattern) and submit it.
  * :mod:`autocal.controller` -- the deterministic decision logic (pure) + the gated run loop.
"""

__all__ = [
    "paths",
    "ledger",
    "rollback",
    "rotation",
    "pooling",
    "fit",
    "submit",
    "controller",
]

SCHEMA_VERSION = 1
