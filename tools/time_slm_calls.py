"""time_slm_calls.py -- time the SLM rearrangement callbacks INDEPENDENTLY of the run loop / seq.

Reproduces the per-shot SLM call sequence that ``RearrangeCommSeq.pre_run`` / ``hand_over_slm``
make (ensure_held -> acquire compute -> setup_rearrangement -> reload_rearrange -> rearrange ->
release compute), driving the SAME ``SlmClient`` + ``SlmScanSession`` the run loop uses, and times
each call. This profiles which SLM call dominates a rearrangement shot WITHOUT running a full scan
or instrumenting the production seq (the seq stays clean -- the timing lives here).

It pulls the model + loading/target patterns from ``SLMRearrangementScan`` so the warmup +
per-shot ``setup_rearrangement`` match a real scan; override the swept ``--nsteps`` if you like.
Fake/zero bits are used for ``rearrange()`` (this profiles call latency, not physics).

Prereq: the SLM server reachable, and NO scan running on the backend (this briefly holds the slm
lock; ``--lease`` sets the session lease so you can see whether reload outruns it).

    cd pyctrl
    python tools/time_slm_calls.py --reps 5
    python tools/time_slm_calls.py --reps 5 --url http://192.168.0.171:8551 --lease 30
"""

import argparse
import os
import sys
import time


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    # root itself carries the ``devices`` package (devices.slm); the rest are flat module dirs.
    for d in (".", "lib", "YbExptCtrl", "YbSeqs", "YbScans"):
        p = os.path.join(root, d) if d != "." else root
        if p not in sys.path:
            sys.path.insert(0, p)


def _stats(ms):
    s = sorted(ms)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return mean, med, s[0], s[-1]


# Ordered per-shot call sequence (mirrors pre_run + hand_over_slm's rearrange()).
_SHOT_ORDER = ["ensure_held", "acquire_compute", "setup_rearrangement",
               "reload_rearrange", "rearrange", "release_compute"]


def time_slm_calls(url=None, reps=5, nsteps=100, lease_s=15.0):
    """Drive + time the per-shot SLM call sequence ``reps`` times. Returns the timings dict."""
    _bootstrap()
    from devices.slm import get_client, SlmScanSession
    import SLMRearrangementScan as RS

    init = RS._pattern_cfg(RS.INIT_PATTERN)
    final = RS._pattern_cfg(RS.TARGET_PATTERN)

    timings = {k: [] for k in (["begin", "initial_setup"] + _SHOT_ORDER)}
    errors = []

    def timed(key, fn):
        t0 = time.perf_counter()
        err = None
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - record the time + error, keep profiling
            err = e
            return None
        finally:
            timings[key].append((time.perf_counter() - t0) * 1e3)
            if err is not None:
                errors.append("%s: %s" % (key, err))

    client = get_client(url)
    ses = SlmScanSession(client, lease_s=lease_s, description="time_slm_calls",
                         log=lambda m: print("[session] %s" % m))
    ses.set_loading_pattern(RS.INIT_PATTERN, init["phase_path"], [0, 0, 0, 0, -5],
                            legacy_zerniked=init["legacy"], baked_zernike=init["baked_zernike"])

    # --- dequeue-time warmup: grab the lock + write the loading phase, then load the model
    #     (reset_params=True -> the one-time model load + max-autotune compile). ----------------
    timed("begin", ses.begin)
    warm = {
        "model_filename": RS.MODEL_FILENAME,
        "initial_phase": init["phase_path"], "final_phase": final["phase_path"],
        "reset_params": True, "client_scan_id": "timing",
        "extras": {"grid_rotation": 90,
                   "initial_phase_zernike": init["baked_zernike"],
                   "final_phase_zernike": final["baked_zernike"],
                   "loading_zernike": [0, 0, 0, 0, -5]},
        "compile_mode": "max-autotune-no-cudagraphs", "use_fp16": True,
        "use_channels_last": True, "use_compile": True, "compile_fullgraph": True,
        "cuda_graph": True, "derive_threshold": 0.35,
    }
    print("warmup (initial setup_rearrangement, reset_params=True) ...")
    timed("initial_setup", lambda: client.setup_rearrangement(**warm))

    # --- per-shot loop (mirror pre_run's sticky setup + reload, and hand_over_slm's rearrange) --
    shot = {"nsteps": nsteps, "step_period_ms": 1.0, "protocol": "rearrange",
            "client_scan_id": "timing",
            "extras": {"block_max_size": 256, "pattern": "sunflower", "kagome_crop": 0.88,
                       "precompute": True, "precompute_host": True, "z4": -5}}
    print("timing %d per-shot SLM call sequences (nsteps=%d, lease=%.0fs) ..." % (reps, nsteps, lease_s))
    for i in range(reps):
        timed("ensure_held", ses.ensure_held)
        timed("acquire_compute",
              lambda: client.acquire_lock("compute", "timing", timeout_s=10, block_timeout=1))
        timed("setup_rearrangement", lambda: client.setup_rearrangement(**shot))
        timed("reload_rearrange", client.reload_rearrange)
        timed("rearrange", lambda: client.rearrange("0", scan_id="timing", seq_id=i))
        timed("release_compute", lambda: client.release_lock("compute"))
        print("  shot %d done" % (i + 1))

    try:
        ses.done()
    except Exception as e:  # noqa: BLE001
        print("[session] done() failed: %s" % e)

    print("\n=== SLM per-call timing (ms): mean / median / min / max  [%d shots] ===" % reps)
    for k in (["begin", "initial_setup"] + _SHOT_ORDER):
        if timings[k]:
            mean, med, lo, hi = _stats(timings[k])
            print("  %-22s %9.1f %9.1f %9.1f %9.1f" % (k, mean, med, lo, hi))
    if errors:
        print("\n%d call error(s) during timing (call still timed):" % len(errors))
        for e in errors[:12]:
            print("  - %s" % e)
    return timings


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="SLM server URL (default: the configured server)")
    ap.add_argument("--reps", type=int, default=5, help="per-shot call sequences to time")
    ap.add_argument("--nsteps", type=int, default=100, help="rearrange nsteps for the per-shot setup")
    ap.add_argument("--lease", type=float, default=15.0, help="session lease seconds")
    a = ap.parse_args()
    time_slm_calls(url=a.url, reps=a.reps, nsteps=a.nsteps, lease_s=a.lease)
