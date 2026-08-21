"""TwoLayer2x11ImagingScan.py -- loading / per-layer readout tests for the 2x11x11 bifocal array.

NEW scan (adapted from TwoLayer2x15ImagingScan; does not touch any existing scan/seq). Loads atoms
into the two-layer BIFOCAL SLM array ``phase/2x11x11_5um.pt``: an 11x11 grid duplicated at TWO
axial planes z4 = +-2.7778 rad about the stack center, with the SAME xy positions in both layers
(verified by per-plane FFT extraction 2026-07-04: 121 spots per plane, inter-plane NN distance
< 0.07 knm px). Unlike the 2x15x15 "diamond" (xy-offset layers), the layers here overlap on the
camera, so ONE 121-site detection grid boxes BOTH layers and per-layer readout comes from moving
the LOADING DEFOCUS, not from per-layer site labels:

  defocus -5.0                stack midplane at the camera plane; both layers +-2.7778 rad
                              (out of focus by the layer offset) -> "see both at once" test
  defocus -5 - 2.7778 = -7.78 one layer AT the camera plane (sharp), other 5.56 rad away
  defocus -5 + 2.7778 = -2.22 the OTHER layer at the camera plane

The detection pattern declares ``planes_z_rad=[-2.7778, 2.7778]`` -> the server's 3-D per-plane
extraction; its cross-plane xy-dedup (radius 6 knm) collapses the 121 coincident pairs to the
single shared 121-site grid (the per-site plane labels that survive dedup are brightness-arbitrary
-- IGNORE them; layer identity comes from which defocus run you are in).

``loading_defocus`` is written ONCE per scan (SlmScanSession writes base + [0 0 0 0 defocus] at
scan start), so bracket by RE-RUNNING at different --defocus values rather than sweeping in-run.

Per-pattern config: expConfig ByPattern["2x11x11_5um"] = copy of 33x33_feedback9's current params
with Init.VSLMServo = 0.39; thresholds seeded flat 202.5 (2026-07-04).

Modes:
  --mode load      (default) TweezerLoadingSeq, NumImages=1: load -> single image. Per-site
                   loading rate.
  --mode survival  ImagingSurvivalSeq, NumImages=2: load -> image -> hold -> image.

Run (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/TwoLayer2x11ImagingScan.py --isinit                 # look first (raw images)
    python YbScans/TwoLayer2x11ImagingScan.py --defocus -5             # midplane (both layers)
    python YbScans/TwoLayer2x11ImagingScan.py --defocus -7.7778        # layer A in focus
    python YbScans/TwoLayer2x11ImagingScan.py --defocus -2.2222        # layer B in focus

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine.
"""

import argparse
import json
import os
import sys


PHASE_PATH = "phase/2x11x11_5um.pt"
PATTERN_NAME = "2x11x11_5um"          # == ByPattern key, phase basename, threshold folder
PLANES_Z_RAD = [-2.7778, 2.7778]      # layer offsets (rad of 2*rho^2-1) about the stack center
DEFAULT_DEFOCUS = -5.0                # ANSI z4: the science-camera plane (== a 2-D array's)
DEFAULT_HOLDS = [0.005, 0.1, 0.5, 1.0]


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

    if not SeqConfig.get().consts:
        SeqConfig.load_real()

    g = ScanGroup()
    rp = g.runp()

    # --- SLM loading hologram (physical write + per-pattern overlay) -------------------------
    rp.loading_phase = PHASE_PATH
    rp.loading_defocus = float(defocus)
    rp.useScanLongSlmLock = 1

    # --- detection pattern (per-frame); planes_z_rad -> 3-D per-plane extraction -------------
    num_images = 2 if mode == "survival" else 1
    pat = {"name": PATTERN_NAME, "base_phase_path": PHASE_PATH,
           "order": "col", "legacy_zerniked": False,
           "planes_z_rad": PLANES_Z_RAD}
    rp.imagePatternsJson = json.dumps([pat] * num_images)

    # --- run controls ------------------------------------------------------------------------
    rp.NumImages = num_images
    rp.isInit = 1 if isinit else 0
    rp.Scramble = 1 if mode == "survival" else 0
    rp.isHC = 0
    rp.isGrid2 = 0
    rp.NumPerGroup = 2000

    if mode == "survival":
        holds = [float(h) for h in (holds if holds else DEFAULT_HOLDS)]
        g().Pushout.Time.scan(1, holds)

    return g


def TwoLayer2x11ImagingScan(url=None, mode="load", defocus=DEFAULT_DEFOCUS,
                            isinit=False, holds=None, reps=None, description=None):
    """Build + submit the 2x11x11 bifocal loading/imaging test. Returns the descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    seq = "ImagingSurvivalSeq" if mode == "survival" else "TweezerLoadingSeq"
    g = build(mode=mode, defocus=defocus, isinit=isinit, holds=holds)

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    if description is None:
        description = (
            "2x11x11_5um two-layer BIFOCAL array (11x11 xy grid duplicated at z4 +-2.7778 rad, "
            "same xy both layers) first-light %s test at loading defocus %g. Layer readout via "
            "defocus: -5 = midplane (both layers defocused by 2.78 rad), -7.78/-2.22 = one layer "
            "at the camera plane. ByPattern copied from 33x33_feedback9 with VSLMServo 0.39; "
            "thresholds seeded flat 202.5. Campaign: 2026-07-04 two-layer loading/defocus "
            "characterization." % (mode, defocus))
    label = "TwoLayer2x11_%s_z%g%s" % (mode, defocus, "_isinit" if isinit else "")
    did = ybStartScan(seq, g, url=url, label=label, description=description, **opts)
    print("[TwoLayer2x11ImagingScan] submitted -> id %s  seq=%s  defocus=%g  isinit=%s  reps=%s"
          % (did, seq, defocus, isinit, reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 2x11x11 two-layer bifocal loading/imaging test.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--mode", choices=["load", "survival"], default="load")
    ap.add_argument("--defocus", type=float, default=DEFAULT_DEFOCUS,
                    help="ANSI z4 loading defocus written for the whole scan (default %g = "
                         "stack midplane; -7.7778/-2.2222 put one layer at the camera plane)"
                         % DEFAULT_DEFOCUS)
    ap.add_argument("--isinit", action="store_true",
                    help="images only (no per-site detection) -- use for the FIRST look")
    ap.add_argument("--holds", type=float, nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the points; omit = StackNum from NumPerGroup")
    ap.add_argument("--description", default=None,
                    help="override the auto-generated run description")
    args = ap.parse_args()
    TwoLayer2x11ImagingScan(url=args.url, mode=args.mode, defocus=args.defocus,
                            isinit=args.isinit, holds=args.holds, reps=args.reps,
                            description=args.description)
