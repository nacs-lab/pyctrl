"""LACScan.py -- pyctrl port of ``matlab_new/YbScans/LACScan.m`` (run directly to submit).

Builds the LAC loading ScanGroup (seq = ``TweezerLoadingSeq``) and submits it to the RUNNING
pyctrl backend over ZMQ (``submit_scan_descriptor``). **Configured here as a green-MOT X-bias
(MOT position) sweep**: it sweeps ``GreenMOT.BiasCoilCurrent.X`` to optimize tweezer loading
RATE + UNIFORMITY, leaving every other parameter at the ``expConfig.py`` defaults (the old
single-point Phase-8 overrides are commented out in :func:`LACScan`). This only BUILDS the
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
import os
import sys
import numpy as np


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def LACScan(url=None, reps=None):
    """Build + submit the LAC loading scan. Returns the queued descriptor id."""
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from yb_start_scan import ybStartScan

    g = ScanGroup()

    # ===== SCAN: GreenMOT X bias coil (green-MOT position along x) ========
    # The X bias current shifts the green-MOT cloud along x relative to the
    # tweezer array, so it sets both how much the MOT overlaps the array (peak
    # loading RATE) and whether that overlap is centered (loading UNIFORMITY).
    # It is razor-sharp: +/-0.01 A off the optimum kills loading
    # (scan-methodology "fast+flat": GreenMOT bias Y x Z x X). Bracket the
    # expConfig default (0.040 A) by +/-0.01 A so both shoulders fall off,
    # 11 pts @ 2 mA. matlab_colon -> MATLAB-exact float64 (X feeds VBiasCoilX,
    # a byte-affecting analog value). Recenter / expand next run if the peak
    # pins to an edge or sits off-center.
    # g().GreenMOT.BiasCoilCurrent.X.scan(1, xvals)

    # ---- everything else at expConfig defaults (unscanned -> commented out) --
    # The 2026-06-05 "fast+flat" loading optimum is already the apparatus default
    # in expConfig.py, so we leave it there and sweep ONLY X. Two of the old
    # Phase-8 overrides differ slightly from expConfig and now fall back to it:
    # BlueMOT.LoadingTime 0.23 -> 0.30 s and GreenMOT.CoolDown.HoldTime 0.12 ->
    # 0.20 s (a touch more saturated/longer); the rest are identical to expConfig.
    g().Init.VSLMservo = 10
    # ===== PHASE 0: BlueMOT.LoadingTime curve (3270_tri loading opt) =====
    # Find the loading cliff + a sub-saturation work-point (~25-40% fill) on the
    # 3270_tri hologram (3270 sites, ~300 uK traps -- shallower than the 33x33
    # ~400 uK). On feedback9 the cliff sat ~0.13-0.14 s with a ~0.58 ceiling;
    # 3270_tri is a different/larger array so re-find it. Bracket wide (0.10-0.34 s,
    # 13 pts) so the rising edge and the ceiling both show. linspace is fine for a
    # single-axis time sweep (LoadingTime feeds s.wait()).
    # ===== PHASE 4: blue capture (BlueMOT.FreqDetuning x Amp) -- the RATE lever =====
    # Loading saturates early (~0.14 s) at a low ~1.8% ceiling that LoadingTime +
    # alignment do NOT raise, so the remaining rate lever is blue capture. Sweep
    # FreqDetuning (the speed lever -- feedback9 had a sharp plateau -44..-48 MHz;
    # default -40 sits on the rising edge) x Amp (was flat 0.4-0.7 on feedback9).
    # Fix LoadingTime just BELOW the cliff (0.13 s, partial fill) so a detuning that
    # shifts the cliff left shows up as more loading (at the 0.14 s ceiling nothing
    # discriminates). 7 detunings x 4 amps = 28 pts.
    # ===== DIAGNOSTIC: single fixed point, current status of 3270_tri loading =====
    # No sweep -- one operating point at the expConfig defaults to read the CURRENT
    # loading fraction + uniformity (rate, CV, x/y gradient) on 3270_tri. Use a
    # well-saturated LoadingTime (0.30 s, deep past the ~0.14 s cliff) so we see the
    # ceiling, not the rising edge. Everything else at expConfig defaults.
    # g().BlueMOT.LoadingTime = 0.30

    # ===== GREEN-MOT POSITION 2-D map (X x Y bias) -- loading uniformity =====
    # Move the green-MOT cloud relative to the 3270_tri array to test whether the
    # bottom-left low-loading/survival region is a MOT-POSITION offset (vs trap depth).
    # X bias is razor-sharp (~+/-0.01 A off-peak kills loading), Y broader; current
    # expConfig defaults X=0.0385, Y=0.265. Sweep both +/-0.05 A, 5 pts each (per user).
    # FINE 1-D X sweep -- X loading peak is sub-mA sharp (+/-0.01 and even +/-0.005
    # were too wide; the historical fine range was 0.037-0.041). Sweep X 0.0335-0.0435
    # at 1 mA step (11 pts) around the expConfig default 0.0385, Y fixed at 0.265.
    # Finds the X peak + whether it sits off-center (-> MOT-position fix for the
    # bottom-left starved region). Do Y next once X is centered.
    # HEAD-TO-HEAD: Y=0.265 (old) vs Y=0.278, interleaved (scramble), high reps,
    # to settle whether Y MOT-position actually fills the bottom-left (the 11-shot
    # sweep faked a gain; the 30-shot confirm did not reproduce it). X at peak.
    
    
    # g().Imag399.Amp1 = 0 #.scan(1, np.linspace(0.1, 1.0, 10))
    # g().Imag399.Amp2 = 0.5 #.scan(1, np.linspace(0.0, 1, 10))
    g().GreenMOT.BiasCoilCurrent.X = 0.0387
    # g().GreenMOT.BiasCoilCurrent.Y.scan(1, [0.265, 0.278])
    # g().BlueMOT.LoadingTime = 0.5
    # g().BlueMOT.FreqDetuning = -44e6
    # g().BlueMOT.Amp = 0.6
    # g().GreenMOT.BiasCoilCurrent.X.scan(1, np.linspace(0.037, 0.041, 12))
    # g().GreenMOT.BiasCoilCurrent.Y.scan(2, np.linspace(0.24, 0.28, 18))
    # g().GreenMOT.BiasCoilCurrent.Z.scan(1, np.linspace(0.16, 0.20, 11))
    # g().GreenMOT.PowerBroaden.HandoverTime = 0.015
    # g().GreenMOT.CoolDown.FreqDetuning = 0.5e6
    # g().GreenMOT.CoolDown.Amp = 0.25
    # g().GreenMOT.CoolDown.HoldTime = 0.2
    # g().GreenMOT.CoolDown.RampdownTime = 0.05
    # LAC
    # g().LAC.FreqDetuning = 0.11e6
    # g().LAC.Amp = 0.2
    # g().LAC.Time = 0.02 #.scan(1, np.linspace(0.02, 0.05, 5))
    # g().LAC.DeadTime = 10e-3

    # ---- run params (runp) ------------------------------------------------
    rp = g.runp()
    rp.NumPerGroup = 200          # image-batch cadence; rep (below) sets shots/point
    rp.NumImages = 1
    rp.isInit = 0
    rp.Scramble = 1   # randomize point order so run-start warmup doesn't bias low-time points
    rp.isHC = 0
    rp.isGrid2 = 0
    # --- optional per-scan SLM loading-pattern override (default from expConfig
    #     SLM.Loading: 33x33_uniform, defocus -5). Uncomment to load a different
    #     hologram for THIS scan (writes it + holds the SLM lock + detects with
    #     that pattern's per-pattern thresholds):
    g.runp().loading_phase = "phase/33x33_feedback9.pt"   # server-side WGS phase path
    g.runp().loading_defocus = -5
    # g().loading_defocus.scan(1, np.linspace(-10, 10, 10));                     # ANSI z4 loading defocus (rad) -- new focus (was -5; array was defocused)

    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps

    desc = ("33x33_feedback9 loading REFERENCE DIAGNOSTIC -- single fixed point at "
            "expConfig defaults (BlueMOT.LoadingTime 0.30 s, well saturated). Sanity "
            "check on the known-good 1068-site array (loaded ~58%, CV 8% earlier "
            "today) to confirm the camera/MOT/imaging chain is healthy and isolate "
            "whether 33x33_feedback9's near-zero loading is array-size / SLM-pattern specific "
            "vs a system-wide problem. feedback9's own registered grid/thresholds.")
    did = ybStartScan("TweezerLoadingSeq", g, url=url, label="LACScan_fb9_refdiag",
                      description=desc, **opts)
    # print("submitted LACScan sweep (%d pts %.3f..%.3f A) -> descriptor id %s (url=%s)"
    #       % (len(xvals), xvals[0], xvals[-1], did, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit LACScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=15,
                    help="passes = shots/point (0 = forever); default 15 -- more shots/point "
                         "than a survival scan because per-site CV/gradient (uniformity) needs "
                         "the statistics. Lower (~8) if you only care about the rate trend.")
    args = ap.parse_args()
    LACScan(url=args.url, reps=args.reps)
