"""autocal.run_controller -- the deterministic background-calibration daemon (live reactor entry).

This wires the pure controller (autocal.controller.Controller) to the real lab IO and loops
``tick()``. It is the "Layer 1" mechanical reactor; a Claude agent supervises it for judgment calls
(see README.md). It is built for a STAGED, safe bring-up -- the hardware/config-mutating adapters
default to OFF, so the first live runs only OBSERVE:

  Stage A (observe):   read queue_list + finished scans, pool, fit, write the ledger + drive the
                       dashboard. NO submit, NO config write, NO SLM switch.  (flags: none)
  Stage B (+submit):   also enqueue home-pattern background cal scans (cycle=True).  (--enable-submit)
  Stage C (+apply):    also auto-apply in-band config (per-pattern resonance) + rollback.  (--enable-apply)
  Stage D (+cycle):    also auto-cycle non-home patterns with the home-restore invariant.  (--enable-cycle)

Each stage is gated by a flag AND by the adapter being wired. The two adapters that touch shared
state -- ``write_config`` (edits expConfig + re-captures the drift oracle) and
``switch_pattern``/``restore_home`` (SLM writes) -- are marked UNVERIFIED and MUST be confirmed on
the rig before their stage is enabled.

Run from ``<root>/pyctrl`` with the no-engine base python (has zmq); reading scans needs the
yb_analysis env on the path (analyze_scan) -- in practice run it under the yb_analysis env which can
import both, or split the read into a subprocess. NOTHING here runs until the user starts it in a
safe window.
"""

import argparse
import json
import os
import sys
import time

# make sibling pyctrl packages importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "lib"), os.path.join(_ROOT, "YbExptCtrl"),
           os.path.join(_ROOT, "YbScans"), os.path.join(_ROOT, "YbSeqs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autocal import ledger as _ledger          # noqa: E402
from autocal import rotation as _rotation       # noqa: E402
from autocal import pooling as _pooling         # noqa: E402
from autocal import submit as _submit           # noqa: E402
from autocal.controller import Controller        # noqa: E402

DEFAULT_URL = "tcp://127.0.0.1:1408"


# =========================================================================== #
# IO adapters
# =========================================================================== #
def make_query_backend(url):
    """ZMQ ``queue_list`` -> the snapshot the controller wants.

    NOTE (verify on-rig): the label/identity field on a finished queue entry. We try
    ``label`` -> ``summary`` -> ``seqName``; whichever carries ``autocal::<cal>::<pattern>`` wins.
    ``foreground_idle`` is derived from the queue (no direct verb): idle iff nothing
    non-background is running or queued."""
    import zmq
    ctx = zmq.Context.instance()

    def _q(verb, timeout_ms=10000):
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(url)
        s.send_string(verb)
        r = s.recv() if s.poll(timeout_ms) else None
        s.close(0)
        return r

    def query_backend():
        raw = _q("queue_list")
        q = json.loads(raw) if raw else {}
        running = q.get("running") or {}
        queued = q.get("queued") or []
        fg_running = bool(running) and running.get("priority", "normal") != "background"
        fg_queued = any(e.get("priority", "normal") != "background" for e in queued)
        completed = []
        for e in (q.get("history") or []):
            lbl = e.get("label") or e.get("summary") or e.get("seqName")
            if _submit.parse_label(lbl):
                completed.append({"scan_id": e.get("file_id"), "label": lbl})
        queued_labels = [(e.get("label") or e.get("summary") or e.get("seqName")) for e in queued]
        return {"foreground_idle": not (fg_running or fg_queued),
                "completed": completed, "queued_labels": queued_labels}

    return query_backend


def make_read_scan():
    """A finished scan -> per-point survival + loading mean (yb_analysis analyze_scan)."""
    def read_scan(scan_id):
        x, mean, sem, n = _pooling.scan_to_points(scan_id)
        if not x:
            return None
        loading_mean = None
        try:
            from yb_analysis.analysis.run_analysis import analyze_scan
            import numpy as np
            d = analyze_scan(scan_id, include_per_site=False, include_diag_aggregate=False,
                             include_per_iteration=False, sync_slm_diag=False)
            ld = np.asarray((d.get("summary") or {}).get("loading_rate") or [], float)
            loading_mean = float(np.nanmean(ld)) if ld.size else None
        except Exception:
            pass
        return {"x": x, "mean": mean, "sem": sem, "n_shots": n, "loading_mean": loading_mean}
    return read_scan


def make_submit(url, ledger_getter):
    """Submit a planned cal as a background scan. Reads per-pattern phase/defocus from the ledger
    settings ``pattern_meta`` (fallback: ``phase/<pattern>.pt`` + the home defocus)."""
    def do_submit(plan):
        pattern = plan["pattern"]
        cal_key = plan["cal_key"]
        cal_def = dict(plan["cal_def"])
        cal_def["_cal_key"] = cal_key
        loading_phase = loading_defocus = None
        if plan.get("requires_switch"):
            meta = (ledger_getter().get("settings", {}).get("pattern_meta") or {}).get(pattern, {})
            loading_phase = meta.get("phase", "phase/%s.pt" % pattern)
            loading_defocus = meta.get("defocus")
        desc = _submit.build_descriptor(cal_key, cal_def, pattern,
                                        reps=cal_def.get("reps_per_run", 6),
                                        loading_phase=loading_phase, loading_defocus=loading_defocus,
                                        cycle=True, requires_switch=plan.get("requires_switch"))
        return _submit.submit_background(desc, url)
    return do_submit


# --- UNVERIFIED adapters (wire + confirm on-rig before enabling their stage) --------------- #
def make_write_config():
    """Per-pattern config write + drift-oracle re-capture. UNVERIFIED -- Stage C.

    Must: edit pyctrl/expConfig.py to set the per-pattern resonance for ``pattern`` to ``value``
    (a per-pattern store -- the user chose this; today's single global Resonance556mj0Freq must
    grow a per-pattern map), preserving the prior value in a dated comment, then run
    tools/capture_config_reference.py so the edit is not flagged as drift. Returns True on success.
    """
    raise NotImplementedError(
        "write_config (per-pattern expConfig write + oracle re-capture) is not wired yet -- "
        "Stage C. Until then run without --enable-apply (auto-apply intents are computed and "
        "surfaced on the dashboard but NOT written).")


def make_switch_pattern_and_restore():
    """SLM switch + mandatory restore-home. UNVERIFIED -- Stage D.

    Must use devices.slm get_client to acquire the 'slm' lock, write the target pattern's hologram,
    and -- critically -- restore the home pattern after the non-home calibration (the home-pattern
    invariant). Respect the SLM-write DMA-stall history: verify the write, back off + alert on error.
    """
    raise NotImplementedError(
        "switch_pattern/restore_home (SLM writes + home-restore invariant) are not wired yet -- "
        "Stage D. Until then run without --enable-cycle (no autonomous pattern switching).")


# =========================================================================== #
# bootstrap + loop
# =========================================================================== #
def bootstrap_ledger(home, cycle_patterns, eligibility_checker=None):
    """Load (or create) the ledger and ensure home + cycle patterns are seeded + eligibility-checked."""
    led = _ledger.load()
    s = led.setdefault("settings", {})
    if home:
        s["home_pattern"] = home
        s.setdefault("active_pattern", home)
    cps = list(dict.fromkeys([p for p in ([home] + list(cycle_patterns or [])) if p]))
    if cps:
        s["cycle_patterns"] = cps
    for p in cps:
        _rotation.seed_pattern_cals(led, p)
        eligible, reasons = _rotation.check_eligibility(p, checker=eligibility_checker)
        led["patterns"][p]["eligible"] = eligible
        led["patterns"][p]["eligibility_reasons"] = reasons
    return led


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="ExptServer ZMQ url (loopback)")
    ap.add_argument("--home", default=None, help="home pattern (foreground always finds this)")
    ap.add_argument("--cycle-patterns", default="", help="comma-sep patterns to keep calibrated")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between ticks")
    ap.add_argument("--once", action="store_true", help="run a single tick and exit")
    ap.add_argument("--enable-submit", action="store_true", help="Stage B: enqueue home cal scans")
    ap.add_argument("--enable-apply", action="store_true",
                    help="Stage C: auto-apply in-band config (requires wired write_config)")
    ap.add_argument("--enable-cycle", action="store_true",
                    help="Stage D: auto-cycle non-home patterns (requires wired SLM adapters)")
    args = ap.parse_args(argv)

    cps = [p.strip() for p in args.cycle_patterns.split(",") if p.strip()]
    led = bootstrap_ledger(args.home, cps)
    _ledger.save(led)

    _state_cache = {"led": led}

    write_config = make_write_config() if args.enable_apply else None
    switch_pattern = restore_home = None
    if args.enable_cycle:
        switch_pattern, restore_home = make_switch_pattern_and_restore()

    ctrl = Controller(
        query_backend=make_query_backend(args.url),
        read_scan=make_read_scan(),
        submit=(make_submit(args.url, lambda: _state_cache["led"]) if args.enable_submit else None),
        write_config=write_config,
        switch_pattern=switch_pattern,
        restore_home=restore_home,
    )

    stage = ("D(+cycle)" if args.enable_cycle else "C(+apply)" if args.enable_apply
             else "B(+submit)" if args.enable_submit else "A(observe)")
    print("[autocal] controller starting | stage %s | home=%s cycle=%s | interval %.0fs"
          % (stage, args.home, cps, args.interval))
    while True:
        try:
            _state_cache["led"] = ctrl.tick(_state_cache["led"])
        except Exception as e:  # noqa: BLE001 - never let one tick kill the daemon
            print("[autocal] tick error: %s" % e)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
