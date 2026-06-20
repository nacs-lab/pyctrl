"""TwoLayer2x15ImagingScan.py -- initial loading / imaging tests for the 2-layer 15x15 array.

NEW scan (does not touch any existing scan/seq). Loads atoms into the two-layer "diamond"
SLM array ``phase/2x15x15_xyoffset_5um.pt`` and images BOTH layers at once in a single camera
frame at the stack midplane -- the protocol where the two layers sit at +-2.5 um = +-0.768 rad
(of the 2*rho^2-1 defocus map) about the imaging plane, close enough to both be in focus, and
are offset 5 um in xy so every spot resolves.

Why the midplane "just works" at the standard defocus: the phase is zernike-free / z=0
stack-centered (save_diamond_phase.py), so the SAME ``loading_defocus`` that centers a 2-D array
on the science camera (-5, the plane the global SLM->camera affine is calibrated against) drops
the stack MIDPLANE onto the camera -- no extra "between-layers" zernike needed.

How the pieces wire up (all standard pyctrl loading-scan machinery, no rearrangement):
  * ``runp().loading_phase`` = the 2x15x15 phase  -> SlmScanSession writes ``base + [0 0 0 0
    loading_defocus]`` once at scan start and holds the slm lock; AND it sets the scan-default
    per-pattern overlay (expConfig ByPattern["2x15x15_xyoffset_5um"]) so cooling/imaging/VSLMServo
    resolve from that array's config (an EXACT copy of 33x33_uniform -- see expConfig.py).
  * ``runp().imagePatternsJson`` declares the DETECTION pattern with ``planes_z_rad=[-0.768,
    0.768]`` -> the SLM server does a per-plane (3-D) extraction so both layers' sites are found
    and labelled. (On a server not yet restarted with the 3-D build, planes_z_rad is ignored and
    it falls back to one 2-D extraction; both layers are within DOF so the combined interleaved
    lattice still extracts -- detection still works, just without per-layer labels.)
  * ``loading_defocus`` is written ONCE per scan (pyctrl writes the loading phase at scan start,
    not per shot), so to bracket the midplane you RUN THE SCAN a few times at different --defocus
    values rather than sweeping it in one run.

Modes:
  --mode load      (default) TweezerLoadingSeq, NumImages=1: load -> single image. Per-site
                   loading rate across both layers. The basic "do atoms load + does detection
                   find all ~450 sites" test.
  --mode survival  ImagingSurvivalSeq, NumImages=2: load -> image -> hold (imaging light ON) ->
                   image. Survival vs hold time -> trap depth / imaging-lifetime health, and
                   confirms img1->img2 detection is stable for both layers.

First run recommendation: ``--isinit`` grabs raw images only (no per-site detection), so you can
eyeball that both layers load and resolve BEFORE trusting the affine-mapped grid on a brand-new
array. Then drop --isinit for the quantitative loading-rate / survival run.

Run (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/TwoLayer2x15ImagingScan.py --isinit                 # look first (raw images)
    python YbScans/TwoLayer2x15ImagingScan.py                          # load rate @ defocus -5
    python YbScans/TwoLayer2x15ImagingScan.py --defocus -4             # bracket the midplane
    python YbScans/TwoLayer2x15ImagingScan.py --mode survival          # survival vs hold
    python YbScans/TwoLayer2x15ImagingScan.py --mode survival --holds 0.005 0.1 0.5 1.0

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine.
"""

import argparse
import json
import os
import sys


PHASE_PATH = "phase/2x15x15_xyoffset_5um.pt"
PATTERN_NAME = "2x15x15_xyoffset_5um"      # == ByPattern key, phase basename, threshold folder
PLANES_Z_RAD = [-0.768, 0.768]            # layers +-2.5 um, radians of 2*rho^2-1 (1 rad ~ 3.26 um)
DEFAULT_DEFOCUS = -5.0                      # ANSI z4: the science-camera plane (== a 2-D array's)
DEFAULT_HOLDS = [0.005, 0.1, 0.5, 1.0]     # survival-mode hold times (s)


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build(mode="load", defocus=DEFAULT_DEFOCUS, isinit=False, holds=None):
    """Build the ScanGroup for the chosen mode. ``defocus`` (ANSI z4 rad) is written once for the
    whole scan; ``holds`` (survival mode) is the swept Pushout hold time list."""
    _bootstrap()
    from scan_group import ScanGroup
    from seq_config import SeqConfig

    # Ensure the REAL expConfig is active so Consts() + the ByPattern overlay resolve (an empty
    # test config has neither -> wrong/zero imaging+cooling+VSLMServo).
    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    g = ScanGroup()
    rp = g.runp()

    # --- SLM loading hologram (physical write + per-pattern overlay) -------------------------
    # loading_phase sets BOTH what SlmScanSession writes (base + [0 0 0 0 defocus]) AND the
    # scan-default ByPattern overlay name (basename of this path).
    rp.loading_phase = PHASE_PATH
    rp.loading_defocus = float(defocus)
    rp.useScanLongSlmLock = 1

    # --- detection pattern (per-frame); planes_z_rad -> 3-D per-layer extraction -------------
    num_images = 2 if mode == "survival" else 1
    pat = {"name": PATTERN_NAME, "base_phase_path": PHASE_PATH,
           "order": "col", "legacy_zerniked": False,
           "planes_z_rad": PLANES_Z_RAD}
    rp.imagePatternsJson = json.dumps([pat] * num_images)

    # --- run controls ------------------------------------------------------------------------
    rp.NumImages = num_images
    rp.isInit = 1 if isinit else 0      # 1 = images only (no per-site detection); look-first
    rp.Scramble = 1 if mode == "survival" else 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.NumPerGroup = 2000

    # --- survival mode: sweep the hold (ImagingSurvivalSeq images during Pushout.Time) -------
    if mode == "survival":
        holds = [float(h) for h in (holds if holds else DEFAULT_HOLDS)]
        g().Pushout.Time.scan(1, holds)

    return g


def TwoLayer2x15ImagingScan(url=None, mode="load", defocus=DEFAULT_DEFOCUS,
                            isinit=False, holds=None, reps=None):
    """Build + submit the 2x15x15 loading/imaging test. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    seq = "ImagingSurvivalSeq" if mode == "survival" else "TweezerLoadingSeq"
    g = build(mode=mode, defocus=defocus, isinit=isinit, holds=holds)

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = "TwoLayer2x15_%s_z%g%s" % (mode, defocus, "_isinit" if isinit else "")
    did = ybStartScan(seq, g, url=url, label=label, **opts)
    print("[TwoLayer2x15ImagingScan] submitted -> id %s  seq=%s  defocus=%g  isinit=%s  reps=%s"
          % (did, seq, defocus, isinit, reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 2x15x15 two-layer loading/imaging test to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--mode", choices=["load", "survival"], default="load",
                    help="load = TweezerLoadingSeq (1 img, loading rate); "
                         "survival = ImagingSurvivalSeq (2 img, survival vs hold)")
    ap.add_argument("--defocus", type=float, default=DEFAULT_DEFOCUS,
                    help="ANSI z4 loading defocus written for the whole scan (default %g; the "
                         "midplane plane). Re-run at -4/-6 to bracket." % DEFAULT_DEFOCUS)
    ap.add_argument("--isinit", action="store_true",
                    help="images only (no per-site detection) -- use for the FIRST look")
    ap.add_argument("--holds", type=float, nargs="+", default=None,
                    help="survival-mode hold times (s); default %s" % DEFAULT_HOLDS)
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the points (0 = forever); omit = StackNum from NumPerGroup")
    args = ap.parse_args()
    TwoLayer2x15ImagingScan(url=args.url, mode=args.mode, defocus=args.defocus,
                            isinit=args.isinit, holds=args.holds, reps=args.reps)
