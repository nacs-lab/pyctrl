"""TwoLayer2x11Back2umImagingScan.py -- one-shot both-layer imaging of the NEW 2x11x11 back-2um array.

NEW scan (adapted from TwoLayer2x11ImagingScan; does not touch any existing scan/seq). Loads the
two-layer BIFOCAL SLM array ``phase/2x11x11_5um_back2um.pt``: an 11x11 grid duplicated at TWO axial
planes z4 = +-2.7778 rad about the stack center. UNLIKE the older ``2x11x11_5um`` (whose two layers
sit at the SAME xy and therefore need per-layer defocus moves to read apart), THIS array's two layers
are OFFSET in xy by ~4.6 knm px (front/back laterally separated -- verified per-plane FFT extraction
2026-07-14: 121 spots/plane, inter-layer NN 4.58 knm px). Because the layers are laterally separated
they are BOTH resolved in ONE image, so per-layer readout comes from a 242-site NO-DEDUP detection grid
at the stack-midplane loading defocus -- no defocus bracketing needed.

Focus map (relative phase-z from a z-scan of the base phase, 2026-07-14):
  each layer's spots peak in brightness at z4 = -+2.7778 (amps ~2140), so at the stack midplane
  (loading defocus -5) BOTH layers are 2.7778 rad out of focus (amps ~334, ~6x dimmer than the sharp
  per-layer plane) but their xy offset keeps the 242 spots spatially resolvable. Per-layer sharp focus:
    defocus -5 - 2.7778 = -7.78   front (min-z) layer at the camera plane
    defocus -5 + 2.7778 = -2.22   back  (max-z) layer at the camera plane
  Default here is the MIDPOINT -5 (user 2026-07-14: measure between the layers, get both at once).

Detection: name "2x11x11_5um_back2um", planes_z_rad=[-2.7778, 2.7778] with cross-plane xy-dedup OFF
(the layers are only ~4.6 knm apart -- the server's default dedup radius 6 would MERGE the pairs to
121 sites). The 242-site registry record (dedup=0) is pre-seeded lab-side; the registry reuses it
without re-deriving (params_match ignores dedup). ByPattern["2x11x11_5um_back2um"] in expConfig is an
alias of the current 2x11x11_5um imaging params (user 2026-07-14: reuse the aligned-xy array's params:
VSLMServo 0.39, Imag399 Amp1 0.23/Amp2 0.22, cooling X(0.16,0.26)/h(0.16,0.14), 35 ms exposure).

``loading_defocus`` is written ONCE per scan (SlmScanSession writes base + [0 0 0 0 defocus] at scan
start), so to read a single layer sharp instead of both, RE-RUN at --defocus -7.7778 / -2.2222.

Modes:
  --mode survival  (default) ImagingPushoutSurvivalSeq, NumImages=2: load -> image -> cool ->
                   pushout(hold) -> cool -> image.
                   Per-site loading rate + real 2-image survival + (offline) per-layer fidelity.
  --mode load      TweezerLoadingSeq, NumImages=1: load -> single image. Per-site loading only.

Run (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/TwoLayer2x11Back2umImagingScan.py --isinit                # look first (raw images)
    python YbScans/TwoLayer2x11Back2umImagingScan.py --mode survival --reps 60
    python YbScans/TwoLayer2x11Back2umImagingScan.py --defocus -7.7778       # front layer sharp only
    python YbScans/TwoLayer2x11Back2umImagingScan.py --defocus -2.2222       # back  layer sharp only

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine.
"""

import argparse
import json
import os
import sys


PHASE_PATH = "phase/2x11x11_5um_back2um.pt"
PATTERN_NAME = "2x11x11_5um_back2um"   # == ByPattern key, phase basename, threshold folder
PLANES_Z_RAD = [-2.7778, 2.7778]       # layer offsets (rad of 2*rho^2-1) about the stack center
DEFAULT_DEFOCUS = -5.0                 # ANSI z4: the stack midplane (both layers, 2.78 rad defocused)
DEFAULT_HOLDS = [0.005, 0.1, 0.5, 1.0]


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for p in (root, os.path.join(root, "lib"), os.path.join(root, "YbExptCtrl")):
        if p not in sys.path:
            sys.path.insert(0, p)


def build(mode="survival", defocus=DEFAULT_DEFOCUS, isinit=False, holds=None):
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
    #     The 242-site no-dedup grid is pre-seeded in the pattern registry; declaring the same
    #     name + planes here re-uses that record (the merging default dedup is NOT re-applied).
    num_images = 2 if mode == "survival" else 1
    # order MUST stay "col_up": any other order mismatches the seeded 242-site registry record ->
    # re-derive with the server's default dedup MERGES the 4.6-knm layer pairs to 121 sites.
    pat = {"name": PATTERN_NAME, "base_phase_path": PHASE_PATH,
           "order": "col_up", "legacy_zerniked": False,
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


def TwoLayer2x11Back2umImagingScan(url=None, mode="survival", defocus=DEFAULT_DEFOCUS,
                                   isinit=False, holds=None, reps=None, description=None):
    """Build + submit the 2x11x11 back-2um both-layer imaging test. Returns the descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    # ImagingPushoutSurvivalSeq (canonical, campaign-tested) NOT the flagged ImagingSurvivalSeq
    # (that one references a bad TTL alias -> "Invalid channel name" on this backend, 2026-07-14).
    seq = "ImagingPushoutSurvivalSeq" if mode == "survival" else "TweezerLoadingSeq"
    g = build(mode=mode, defocus=defocus, isinit=isinit, holds=holds)

    opts = {}
    if reps is not None:
        opts["rep"] = reps
    if description is None:
        description = (
            "2x11x11_5um_back2um NEW two-layer BIFOCAL array (11x11 xy grid duplicated at z4 "
            "+-2.7778 rad, layers OFFSET ~4.6 knm px in xy) both-layer imaging %s test at loading "
            "defocus %g. Layers laterally separated -> 242-site no-dedup detection reads BOTH from "
            "ONE image at the stack midplane (-5); no defocus bracketing. Per-layer sharp = "
            "-7.78/-2.22. Imaging params copied from the aligned-xy 2x11x11_5um (VSLMServo 0.39, "
            "Amp1 0.23/Amp2 0.22, cool X(0.16,0.26)/h(0.16,0.14), 35 ms). Campaign: 2026-07-14 "
            "back-2um both-layer imaging fidelity characterization." % (mode, defocus))
    label = "TwoLayer2x11Back2um_%s_z%g%s" % (mode, defocus, "_isinit" if isinit else "")
    did = ybStartScan(seq, g, url=url, label=label, description=description, **opts)
    print("[TwoLayer2x11Back2umImagingScan] submitted -> id %s  seq=%s  defocus=%g  isinit=%s  reps=%s"
          % (did, seq, defocus, isinit, reps))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Submit the 2x11x11 back-2um two-layer both-layer imaging test.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--mode", choices=["load", "survival"], default="survival")
    ap.add_argument("--defocus", type=float, default=DEFAULT_DEFOCUS,
                    help="ANSI z4 loading defocus written for the whole scan (default %g = "
                         "stack midplane, BOTH layers; -7.7778/-2.2222 put one layer sharp)"
                         % DEFAULT_DEFOCUS)
    ap.add_argument("--isinit", action="store_true",
                    help="images only (no per-site detection) -- use for the FIRST look")
    ap.add_argument("--holds", type=float, nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the points; omit = StackNum from NumPerGroup")
    ap.add_argument("--description", default=None,
                    help="override the auto-generated run description")
    args = ap.parse_args()
    TwoLayer2x11Back2umImagingScan(url=args.url, mode=args.mode, defocus=args.defocus,
                                   isinit=args.isinit, holds=args.holds, reps=args.reps,
                                   description=args.description)
