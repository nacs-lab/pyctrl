"""ShiftLeftStepSweep.py -- ONE-WAY uniform lattice shift, nsteps sweep. The discriminator for the
ping-pong vs rearrange2 per-step-loss discrepancy.

WHY THIS SCAN EXISTS. The 2026-08-01/02 heating campaign measured a 1%/step survival cliff at
1.296 knm px (radial, 0.696 ms) using ``pingponggrating`` -- every atom driven out and back by the
same vector, 2n steps. The full rearrange2 -> every_other run (627/638) then measured only
~0.1%/step at that same per-frame stroke, and did not reach 1%/step until ~2.5-3 px. A factor of
ten in per-step loss, ~2x in tolerable stroke. Three candidate causes, and the two scans differ in
BOTH of the relevant ways at once, so neither can be blamed:

    ping-pong    GLOBAL uniform move   OUT-AND-BACK (turnaround, 2n steps, same trap revisited)
    rearrange2   SCATTERED movers      ONE-WAY (n steps, atom walks a path and stops)

``shift_left`` splits them: it is a GLOBAL uniform move like ping-pong, but ONE-WAY like
rearrange2. So it is a clean two-way discriminator --

    loss ~ rearrange2 (low)   -> the turnaround / repeated identical kicks are the cause
    loss ~ ping-pong  (high)  -> the global-uniform-move geometry is the cause

WHY ONE-WAY IS IMAGEABLE HERE (the reason this is shift_left and not ``pingpong --oneway``). A
plain one-way ping-pong parks every atom at nsteps x step_size, which is not generally a lattice
multiple, so img2's fixed detection grid has no site under the atom and survival cannot be read
out at all. ``shift_left`` moves every atom by exactly ONE LATTICE SITE and runs Hungarian
assignment, so the destinations ARE grid sites and the per-shot ``target_paired`` ledger records
them. Target-aware survival then works unmodified -- no hand-rolled index arithmetic, which is the
easy way to get a silent off-by-one across a rotated 33x33 grid.

THE nsteps AXIS IS THE STROKE AXIS. The move is one lattice constant, measured 24.456 knm px on
33x33_feedback11 (nearest-neighbour median, IQR 55.35-55.44 cam px at 2.2658 cam px per knm px),
so the per-frame stroke is 24.456 / nsteps:

    nsteps  10    12    15    20    25    30    40
    stroke  2.45  2.04  1.63  1.22  0.98  0.82  0.61   knm px
    x cliff 1.89  1.57  1.26  0.94  0.75  0.63  0.47

which brackets the 1.296 px cliff from both sides and overlays directly on the rearrange2
per-step-loss table.

Everything else -- array, phase, warmup model, wgs_warm producer, z4, cooling -- is copied from
WarmWGSRearrangeStepSweep so the three datasets are comparable without cross-calibration.

Run:
    cd pyctrl && python YbScans/RearrangeDiagnostics/ShiftLeftStepSweep.py --dry-run
    cd pyctrl && python YbScans/RearrangeDiagnostics/ShiftLeftStepSweep.py --force
"""
import argparse
import json
import os
import sys

PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0, 0, 0, 0, 0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

LATTICE_KNM_PX = 24.456        # measured on this array; the shift is exactly one of these
NSTEPS = [10, 12, 15, 20, 25, 30, 40]
STEP_PERIOD_MS = 0.696
WGS_PAD = 2048
WGS_ITERS = 3
DEFOCUS = -4
DEF_REPS = 60                  # never leave this None -- see WarmWGSRearrangeStepSweep's DEF_REPS


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build(nsteps=None, period=STEP_PERIOD_MS, wgs_warm=True, precompute=False):
    _bootstrap()
    from scan_group import ScanGroup

    ns = list(nsteps or NSTEPS)
    g = ScanGroup()
    rp = g.runp()

    # ---- warmup: identical to WarmWGSRearrangeStepSweep ----
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.extras.final_phase_zernike = [float(z) for z in BAKED_ZERNIKE]
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- one-way uniform lattice shift, nsteps swept ----
    g().rearrange_kwargs.protocol = "shift_left"
    g().rearrange_kwargs.nsteps.scan(1, [int(n) for n in ns])
    # STOP ON THE SHIFTED FRAME. Without this the schedule ends
    #     ... last transit frame (atoms at SHIFTED positions) -> WGS-final
    # and WGS-final is built from ``final_pattern``, i.e. the UNSHIFTED hologram -- so the last
    # thing written puts every trap one site back from where the atoms now are and they are all
    # dropped. Run 644 (`20260802_190931`) did exactly that: img1 loaded a normal 0.632 while img2
    # read 0.008 at every nsteps. ``stop_after`` = the model-frame count truncates the write
    # sequence before the WGS bookend, leaving the shifted frame on the SLM for img2. With both
    # bookends off the model-frame count is nsteps + 1, so it must be swept ON THE SAME DIM as
    # nsteps to stay matched point-for-point.
    g().rearrange_kwargs.extras.stop_after.scan(1, [int(n) + 1 for n in ns])
    g().rearrange_kwargs.step_period_ms = float(period)
    g().rearrange_kwargs.extras.n_rounds = 1
    g().rearrange_kwargs.extras.full_n = True        # shift the whole array, not a subset
    g().rearrange_kwargs.extras.wgs_warm = bool(wgs_warm)
    g().rearrange_kwargs.extras.wgs_pad = WGS_PAD
    g().rearrange_kwargs.extras.wgs_iters = WGS_ITERS
    g().rearrange_kwargs.extras.mover_boost = 0.0
    g().rearrange_kwargs.extras.prob_hungarian = True
    g().rearrange_kwargs.extras.overdrive = False
    g().rearrange_kwargs.extras.dynamic = False
    g().rearrange_kwargs.extras.ifEnhanced = True
    g().rearrange_kwargs.extras.z4 = DEFOCUS
    g().rearrange_kwargs.extras.initial_pattern = PATTERN
    g().rearrange_kwargs.extras.final_pattern = PATTERN
    # Server extras are STICKY -- write both precompute flags explicitly every build.
    g().rearrange_kwargs.extras.precompute = bool(precompute)
    g().rearrange_kwargs.extras.precompute_host = bool(precompute)
    g().rearrange_kwargs.extras.description = (
        "ONE-WAY uniform one-lattice-site shift (shift_left) + Hungarian assignment, warm-WGS "
        "transit frames (pad %d x %d iters), 33x33_feedback11, step_period_ms=%g, nsteps %s. "
        "Per-frame stroke = %.3f / nsteps knm px, spanning %.2f-%.2f x the 1.296 px ping-pong "
        "cliff. Discriminator for the ping-pong (global+round-trip, ~1%%/step at 1.296 px) vs "
        "rearrange2 (scattered+one-way, ~0.1%%/step at the same stroke) discrepancy: shift_left "
        "is global+uniform like ping-pong but one-way like rearrange2."
        % (WGS_PAD, WGS_ITERS, period, ",".join(str(n) for n in ns), LATTICE_KNM_PX,
           (LATTICE_KNM_PX / max(ns)) / 1.296, (LATTICE_KNM_PX / min(ns)) / 1.296))

    rp.NumPerGroup = 100000
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = json.dumps([
        {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
        {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
         "legacy_zerniked": False},
    ])
    return "RearrangeCommSeq", g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--nsteps", default=None, help="comma-separated nsteps override")
    ap.add_argument("--period", type=float, default=STEP_PERIOD_MS)
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    a = ap.parse_args()

    ns = [int(x) for x in a.nsteps.split(",")] if a.nsteps else list(NSTEPS)
    seq, g = build(nsteps=ns, period=a.period, precompute=a.precompute)
    label = a.label or ("ShiftLeftRearrangeStepSweep_p%g" % a.period)

    if a.dry_run:
        s0 = g.getseq(0)["rearrange_kwargs"]
        print("seq=%s nseq=%d protocol=%s period=%g full_n=%s wgs_warm=%s precompute=%s"
              % (seq, g.nseq(), s0["protocol"], s0["step_period_ms"],
                 s0["extras"]["full_n"], s0["extras"]["wgs_warm"],
                 s0["extras"]["precompute"]))
        print("nsteps  %s" % ns)
        print("stroke  %s knm px"
              % ["%.3f" % (LATTICE_KNM_PX / n) for n in ns])
        print("x cliff %s" % ["%.2f" % ((LATTICE_KNM_PX / n) / 1.296) for n in ns])
        return

    if not a.force:
        raise SystemExit("refusing to submit without --force (use --dry-run to inspect)")
    from yb_start_scan import ybStartScan
    did = ybStartScan(seq, g, url=a.url, label=label,
                      rep=DEF_REPS if a.reps is None else int(a.reps))
    print("submitted %s -> id %s (nsteps=%s, period=%g)" % (label, did, ns, a.period))


if __name__ == "__main__":
    main()
