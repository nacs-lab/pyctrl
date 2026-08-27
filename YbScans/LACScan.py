"""LACScan.py -- pyctrl port of ``matlab_new/YbScans/LACScan.m`` (run directly to submit).

Builds the LAC loading ScanGroup (seq = ``TweezerLoadingSeq``) and submits it to the RUNNING
pyctrl backend over ZMQ (``submit_scan_descriptor``). **Configured here as a green-MOT bias
X x Y (MOT position) grid**: it sweeps ``GreenMOT.BiasCoilCurrent.X`` and ``.Y`` to optimize
tweezer loading RATE + UNIFORMITY, leaving every other parameter at the ``expConfig.py``
defaults. Set loading params in expConfig, NOT as g() overrides here -- a stale override in
this file silently shadows config (it bit rounds 4/5 on 2026-08-27). This only BUILDS the
ScanGroup + sends the descriptor JSON -- it does NOT load the engine, so any interpreter with
pyctrl importable + zmq works (e.g. the yb_analysis env, base, or .venv-engine-py312).

Run it:
    cd pyctrl
    python YbScans/LACScan.py                 # StackNum=max(ceil(NumPerGroup/nseqs),2) passes
    python YbScans/LACScan.py --reps 1        # one pass over the sweep
    python YbScans/LACScan.py --reps 0        # run forever (continuous loading monitor)
    python YbScans/LACScan.py --url tcp://127.0.0.1:1408

Prereq: the pyctrl backend must be running at --url (default tcp://127.0.0.1:1408 -- the
monitor's URL). Submit, then watch the dashboard's Tweezer Array / loading.
"""

import argparse
import numpy as np

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

# (submits "TweezerLoadingSeq" by name -- see ybStartScan below; no seq import needed)


def LACScan(url=None, reps=None):
    """Build + submit the LAC loading scan. Returns the queued descriptor id."""
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from yb_start_scan import ybStartScan

    g = ScanGroup()

    # ===== SWEEP =====
    # 2026-08-27 loading campaign (33x33_feedback11). Committed to expConfig: base
    # GreenMOT.BiasCoilCurrent X 0.0353 / Y 0.244, ByPattern BlueMOT.LoadingTime 0.28 s,
    # PowerBroaden.HandoverTime left at 0.015 s (swept 0.005-0.050, broad plateau).
    # Full campaign record: Notion 08/27 + the loading-optimization runbook. Scans:
    #   0a LoadingTime curve   20260827151119  cliff <0.09 s, plateau from ~0.295 s
    #   1  bias X x Y coarse   20260827151415  X ~4 mA wide; in-use X was on the flank
    #   2  fine X              20260827152139  interior peak 0.0350-0.0355, corr_x null there
    #   3  Y at pinned X       20260827152326  broad; corr_y nulls ~0.2635
    #   0b handover/overlap    20260827152511  plateau 0.010-0.050 -> keep 0.015
    #      verify (0.28 s)     20260827152706  load 0.480, CV 0.25, 0/1068 dead sites
    #   4/5 joint X x Y        20260827160412/160629  Y edge-pinned low -> shifted down
    #
    # *** WARNING: a stray `g().BlueMOT.LoadingTime = 0.8` (a leftover 2026-07-15 override) was
    # ACTIVE here during rounds 4/5, so scan 20260827160629 ran at 0.8 s -- DEEP IN SATURATION,
    # not the 0.28 s work-point (descriptor confirms 0.8). The runbook is explicit that nothing
    # discriminates at a saturated LoadingTime, so that grid must be re-measured at 0.28 s before
    # its fitted centre (X 0.03526 / Y 0.2441) is trusted. Removed 2026-08-27. Set LoadingTime
    # ONLY via config, or deliberately here with a dated comment saying why. ***

    # ---- ACTIVE sweep -----------------------------------------------------
    #g().GreenMOT.BiasCoilCurrent.X.scan(1, np.linspace(0.0335, 0.0375, 9))
    #g().GreenMOT.BiasCoilCurrent.Y.scan(2, np.linspace(0.218, 0.258, 11))

    # ---- MENU: other sweeps / pins, uncomment as needed -------------------
    # Loading params normally come from expConfig (base GreenMOT.BiasCoilCurrent,
    # ByPattern BlueMOT.LoadingTime). Anything uncommented HERE SHADOWS config -- it silently
    # overrode LoadingTime during rounds 4/5 on 2026-08-27. Re-comment when done.
    #
    # BlueMOT (capture / timing)
    g().BlueMOT.LoadingTime.scan(1, np.linspace(0.2, 0.8, 12)) #= 0.8      # 2026-07-15 long blue load; 08-27 work-point was 0.28
    #g().BlueMOT.LoadingTime = 0.30
    #g().BlueMOT.LoadingTime.scan(1, np.linspace(0.05, 0.50, 12))   # 0a cliff curve
    #g().BlueMOT.FreqDetuning = -44e6   # the speed lever (plateau -44..-48 on feedback9)
    #g().BlueMOT.Amp = 0.6              # flat 0.4-0.7
    #
    # GreenMOT bias (MOT position: rate + uniformity)
    #g().GreenMOT.BiasCoilCurrent.X = 0.0353    # committed 08-27
    #g().GreenMOT.BiasCoilCurrent.X = 0.0387
    #g().GreenMOT.BiasCoilCurrent.X.scan(1, np.linspace(0.026, 0.044, 10))   # coarse, 2 mA
    #g().GreenMOT.BiasCoilCurrent.Y = 0.244     # committed 08-27 (provisional, see WARNING)
    #g().GreenMOT.BiasCoilCurrent.Y = 0.262     # the 0.28 s 1-D result
    #g().GreenMOT.BiasCoilCurrent.Y.scan(1, [0.265, 0.278])                  # head-to-head
    #g().GreenMOT.BiasCoilCurrent.Y.scan(1, np.linspace(0.235, 0.285, 11))
    #g().GreenMOT.BiasCoilCurrent.Z = 0.172 #.scan(1, np.linspace(0.16, 0.20, 11))
    #
    # GreenMOT handover / cooldown
    #g().GreenMOT.PowerBroaden.HandoverTime = 0.015   # blue->green overlap; plateau 0.010-0.050
    #g().GreenMOT.PowerBroaden.HandoverTime.scan(1, np.linspace(0.005, 0.050, 10))
    #g().GreenMOT.CoolDown.FreqDetuning = 0.5e6
    #g().GreenMOT.CoolDown.Amp.scan(1, np.linspace(0, 0.5, 10))   # 0.25; dead by 0.40
    #g().GreenMOT.CoolDown.HoldTime = 0.5
    #g().GreenMOT.CoolDown.RampdownTime = 0.05    # WARNING: 0.0 ERRORS (ramp(0) invalid), keep >=0.01
    #
    # LAC (blockade). Not limiting at a sub-saturation work-point (no 2-atom peak) -- 06-05.
    #g().LAC.Amp = 0 #.scan(1, np.linspace(0.04, 0.20, 7))            # LAC intensity
    #g().LAC.FreqDetuning.scan(2, np.linspace(0.10e6, 0.30e6, 7))     # LAC detuning (Hz)
    #g().LAC.Time = 0.02 #.scan(1, np.linspace(0.02, 0.05, 5))
    #g().LAC.DeadTime = 10e-3
    #
    # Imaging (needs NumImages=2 + a Pushout seq; this scan submits TweezerLoadingSeq, img1 only)
    #g().Pushout.Time = 0.001    # ~0 pushout: real survival + true separation
    #g().Imag399.Amp1.scan(1, np.linspace(0.2, 2, 10))    # DDS amps must stay 1 under the PID scheme
    #g().Imag399.Amp2.scan(1, np.linspace(0.2, 2, 10))
    #g().BlueMOT.Img1PIDSet.scan(1, np.linspace(0.4, 2.4, 10))
    #g().BlueMOT.Img2PIDSet.scan(2, np.linspace(0.2, 2.65, 10))
    #
    # Other
    #g().Init.VSLMservo = 3.5    # NOTE 08-11: VServo appeared NOT to move trap depth over 1.9-3.9
    rp = g.runp()
    rp.NumPerGroup = 100
    rp.NumImages = 1              # loading-rate readout (MOT-position opt); img1 only
    rp.isInit = 0
    rp.Scramble = 1   # randomize point order: run-start warmup would otherwise bias early points.
                      # Safe here -- this scan drives no 616 EOM, so the scramble-unlocks-616
                      # gotcha does not apply.
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    g.runp().loading_phase = "phase/33x33_feedback11.pt"   # server-side WGS phase path
    # 2026-08-10: loading plane now comes from the per-array config
    # (ByPattern[<pattern>].SLM.Loading.Defocus -> slm_runtime._pattern_defocus);
    # setting rp.loading_defocus here would override it, so it is left unset.
    #g().loading_defocus.scan(1, np.linspace(-20, 0, 5));                    # ANSI z4 loading defocus (rad): -20..20 step 5 (9 pts) -- tri_3013_v2 focus bracket

    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps

    desc = ("33x33_feedback11 (z4=-5) loading opt: GreenMOT bias X x Y grid at the committed "
            "LoadingTime work-point, img1 only. Reads loading RATE + UNIFORMITY (CV, x/y gradient) "
            "per cell.")
    did = ybStartScan("TweezerLoadingSeq", g, url=url, label="LACScan",
                      description=desc, **opts)
    # print("submitted LACScan sweep (%d pts %.3f..%.3f A) -> descriptor id %s (url=%s)"
    #       % (len(xvals), xvals[0], xvals[-1], did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit LACScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=20,
                    help="passes = shots/point (0 = forever)")
    args = ap.parse_args()
    LACScan(url=args.url, reps=args.reps)
