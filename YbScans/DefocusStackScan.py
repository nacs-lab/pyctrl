"""DefocusStackScan.py -- ONE scan that sweeps the LOADING DEFOCUS (ANSI z4) per shot, so the
averaged images stack into a 3-D picture of a tweezer as the atom camera sees it.

seq = ``TweezerLoadingSeq`` (Init -> BlueMOT -> SLM -> GreenMOT -> LAC -> Imag399), NumImages=1.

HOW THE PER-SHOT DEFOCUS WORKS.  ``runp().loading_defocus`` is a scan CONSTANT (read once, applied
on the single scan-start SLM write), so it cannot sweep.  This scan declares the plane as a SCANNED
SEQUENCE PARAM instead::

    g().SLM.LoadingDefocus.scan(1, matlab_colon(-10, 0.1, 0))

No Step reads that param -- it never reaches ``serialize()``, so the byte oracles / THE ONE RULE
are untouched.  It exists so each scan point CARRIES its focal plane, exactly as ``g().AWG.*`` /
``g().QICK.*`` carry per-point device state.  The run loop's per-shot pre_cb
(``slm_runtime.make_slm_defocus_pre_cb`` -> ``SlmScanSession.set_defocus``) rewrites the loading
hologram's ``[0 0 0 0 z4]`` Zernike BEFORE the shot's sequence runs, so that shot's atoms are
LOADED at that plane -- physically identical to running one scan per defocus, minus 100x the
per-job overhead, and with ``Scramble`` available so slow drift cannot masquerade as axial
structure.

WHAT THE STACK IS.  The science camera's focal plane is FIXED; z4 moves the TRAP plane.  So the
z-stack of averaged frames is the imaging system's axial response to an atom (a point source)
walked through focus, multiplied by the loading/survival envelope at that plane.  1 rad of z4 =
0.8 um axially (rig value 2026-08-07; train3d.UM_PER_RAD models 0.905).

THRESHOLDS.  Defocused shots are DIM and they feed the per-pattern threshold accumulator
(bug-threshold-dim-scan-contamination).  Re-anchor with a bright run at the production defocus
after this scan (``tools/defocus_stack.py reanchor``).

Run it:
    cd pyctrl
    python YbScans/DefocusStackScan.py                    # 101 planes x 10 shots (scrambled)
    python YbScans/DefocusStackScan.py --lo -10 --hi 0 --step 0.1 --shots 10
    python YbScans/DefocusStackScan.py --shots 1 --step 2.0    # quick 6-plane smoke test

Analyse: python tools/defocus_stack.py analyze --scan-id <14-digit id>
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()


def DefocusStackScan(url=None, lo=-10.0, hi=0.0, step=0.1, shots=10,
                     phase="phase/33x33_feedback11.pt", scramble=1, label=None):
    """Build + submit the single-scan defocus stack. Returns the queued descriptor id."""
    from scan_group import ScanGroup
    from scan_export import matlab_colon
    from yb_start_scan import ybStartScan

    zs = [float(z) for z in matlab_colon(lo, step, hi)]

    g = ScanGroup()
    # ===== SCAN: loading defocus (ANSI z4, rad), one focal plane per point ==============
    # matlab_colon -> the MATLAB-exact float64 grid. dim1, so it varies fastest; every pass
    # covers all planes, and `shots` passes give `shots` shots per plane.
    g().SLM.LoadingDefocus.scan(1, zs)

    rp = g.runp()
    rp.NumPerGroup = float(len(zs))
    rp.NumImages = 1                 # img1 only -- this is a loading/imaging shape measurement
    rp.isInit = 0
    # Scramble the point order within each pass: the sweep takes ~35 min, and a slow loading /
    # 399-power drift over that window would otherwise print itself onto the axial envelope as a
    # fake asymmetry. Randomised order turns drift into noise instead of structure.
    rp.Scramble = int(scramble)
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.loading_phase = phase
    # Scan-start write + the plane the pre_cb starts from (harmless: point 1 rewrites it).
    rp.loading_defocus = zs[0]

    desc = ("Single-scan z4 axial stack on %s: loading defocus swept PER SHOT over %.2f..%.2f rad "
            "step %.2f (%d planes) x %d shots, img1 only, scrambled point order. The SLM loading "
            "hologram's z4 Zernike is rewritten before every shot (slm_runtime per-shot defocus "
            "pre_cb), so each shot LOADS at its own plane. Averaged frames reconstruct the 3-D "
            "atom-fluorescence image of a tweezer (axial PSF x loading envelope; 1 rad = 0.8 um "
            "axially). Analyse with tools/defocus_stack.py."
            % (phase, zs[0], zs[-1], step, len(zs), shots))
    did = ybStartScan("TweezerLoadingSeq", g, url=url,
                      label=label or "z4stack_%dplanes_x%d" % (len(zs), shots),
                      description=desc, rep=int(shots))
    print("submitted DefocusStackScan: %d planes (z4 %+0.2f..%+0.2f step %.2f) x %d shots = %d "
          "shots -> descriptor id %s" % (len(zs), zs[0], zs[-1], step, shots,
                                         len(zs) * shots, did))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit the single-scan loading-defocus stack.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--lo", type=float, default=-10.0)
    ap.add_argument("--hi", type=float, default=0.0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--shots", type=int, default=10, help="shots per plane (= passes)")
    ap.add_argument("--phase", default="phase/33x33_feedback11.pt")
    ap.add_argument("--scramble", type=int, default=1)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    DefocusStackScan(url=a.url, lo=a.lo, hi=a.hi, step=a.step, shots=a.shots,
                     phase=a.phase, scramble=a.scramble, label=a.label)
