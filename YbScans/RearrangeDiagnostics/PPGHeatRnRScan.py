"""PPGHeatRnRScan.py -- pingponggrating heating characterization via release-and-recapture.

2-D scans on RearrangeRnRHeatCommSeq (img1 -> pingponggrating motion -> [post-motion cooling
OFF by default] -> release-and-recapture -> img2):

    dim 1: ONE pingponggrating knob (step_x / step_y / step_z / nsteps / piston / period)
    dim 2: ReleaseRecapture.Time (release-time curve; t=0 = byte-clean "no release" baseline)

The release curve at each knob value is a thermometer for the post-motion atom temperature;
the t=0 row is the pure (motion-only) survival. The knob=0 / step0 column is the in-scan
no-motion control (frames still written -- controls for the write process itself).

Design decisions (2026-07-22 heating campaign):
  * seq = RearrangeRnRHeatCommSeq: pre-motion cooling = s.C.Cool556 (same prepared state as
    the baseline ReleaseRecaptureScan); post-motion cooling = s.C.PostRearrCool, default OFF
    here (amps 0 + short dark hold) so the release probes the post-motion temperature
    directly. --cool restores production 5 ms post-motion cooling (group left unset).
  * pattern 33x33_feedback11 (production depth-uniformized array), loading_defocus -5.
  * protocol pingponggrating, no model frames; precompute + precompute_host on so the paced
    loop is write-only and step_period_ms is realised.
  * no_depth_piston=True EXPLICIT (server default since 2026-07-15) with the DEFAULT beam
    params (fill 0.65, center (-35,+6) px) -- per campaign instruction.
  * step knobs swept as SCALAR extras (step_x/step_y/step_z) -> _fold_step_xyz builds the
    xyz step_size 3-vector per shot (list-valued axes break the lab-side live curve).

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGHeatRnRScan.py --knob step_x \
        --values 0,0.5,0.75,1.0,1.25,1.5,1.75 --times 0,5,10,15,20,30,40 --reps 10
    python ... --knob nsteps --values 5,10,20,50,100 --fixed-step-x 1.0
    python ... --knob piston --values 0,0.25,0.5,1.0,1.57,3.14 --fixed-step-x 0
    python ... --knob period --values 0.696,1.4,3.0 --fixed-step-x 1.0
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
LOADING_DEFOCUS = -5
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"

DEF_NSTEPS = 50
DEF_PERIOD_MS = 0.696
DEF_TIMES_US = "0,5,10,15,20,30,40"

KNOBS = ("step_x", "step_y", "step_z", "nsteps", "piston", "period")


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(knob, values, times_us, nsteps=DEF_NSTEPS, period_ms=DEF_PERIOD_MS,
          fixed_step_x=0.0, fixed_step_y=0.0, fixed_step_z=0.0, fixed_piston=0.0,
          cool=False, post_cool_hold_ms=0.5, fill_center=None):
    """Build (do NOT submit) the 2-D [knob x release-time] ScanGroup. Returns (seq_name, g)."""
    _bootstrap()
    from scan_group import ScanGroup

    if knob not in KNOBS:
        raise ValueError("knob must be one of %s, got %r" % (KNOBS, knob))
    values = [float(v) for v in values]
    times = [float(t) * 1e-6 for t in times_us]

    seq_name = "RearrangeRnRHeatCommSeq"
    g = ScanGroup()

    # ---- single-round rearrangement ----------------------------------------------------
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = BAKED_ZERNIKE
    rp.warmup_kwargs.extras.final_phase_zernike = BAKED_ZERNIKE
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs: pingponggrating ---------------------------------------------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.no_depth_piston = True     # EXPLICIT (server default); fill frac/center = defaults
    if fill_center is not None:
        # normalized [cx, cy] beam-center override (recenters blaze x/y + depth Z4).
        rk.extras.depth_fill_center = [float(fill_center[0]), float(fill_center[1])]
    rk.extras.ifEnhanced = False
    rk.extras.hw_sequence = False
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # Fixed step components (scalars; _fold_step_xyz -> step_size=[x,y,z] per shot). Set ALL
    # THREE explicitly every scan so the sticky server extras can never inherit a stale value
    # from a previous scan in the same backend session.
    fixed = {"step_x": float(fixed_step_x), "step_y": float(fixed_step_y),
             "step_z": float(fixed_step_z)}
    if knob in ("step_x", "step_y", "step_z"):
        fixed.pop(knob)
        getattr(rk.extras, knob).scan(1, values)
        for k, v in fixed.items():
            setattr(rk.extras, k, v)
        rk.nsteps = int(nsteps)
        rk.step_period_ms = float(period_ms)
        rk.extras.piston = float(fixed_piston)
    elif knob == "nsteps":
        rk.nsteps.scan(1, [int(v) for v in values])
        for k, v in fixed.items():
            setattr(rk.extras, k, v)
        rk.step_period_ms = float(period_ms)
        rk.extras.piston = float(fixed_piston)
    elif knob == "piston":
        rk.extras.piston.scan(1, values)
        for k, v in fixed.items():
            setattr(rk.extras, k, v)
        rk.nsteps = int(nsteps)
        rk.step_period_ms = float(period_ms)
    elif knob == "period":
        rk.step_period_ms.scan(1, values)
        for k, v in fixed.items():
            setattr(rk.extras, k, v)
        rk.nsteps = int(nsteps)
        rk.extras.piston = float(fixed_piston)

    # ---- POST-MOTION cooling (the seq's PostRearrCool group) ---------------------------
    # Default (cool=False): amps 0 => NO cooling between motion and release; short dark hold.
    # cool=True: leave the group UNSET -> falls back to ByPattern[pattern].Cool556 (the
    # production 5 ms RNR-optimized cooling), byte-identical to RearrangeRnRCommSeq.
    if not cool:
        g().PostRearrCool.X.Amp = 0
        g().PostRearrCool.h.Amp = 0
        g().PostRearrCool.Time = float(post_cool_hold_ms) * 1e-3

    # ---- dim 2: the release-and-recapture window ----------------------------------------
    g().ReleaseRecapture.Time.scan(2, times)
    g().ReleaseRecapture.Hold = 0   # set for faithfulness; UNREAD by RearrangeRnRStep

    # ---- run params (runp) ---------------------------------------------------------------
    rp.NumPerGroup = 100000
    rp.loading_defocus = LOADING_DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def submit(knob, values, times_us, url=None, reps=None, label=None, **kw):
    seq_name, g = build(knob, values, times_us, **kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    label = label or ("PPGHeatRnR_%s" % knob)
    did = ybStartScan(seq_name, g, url=url, label=label, **opts)
    print("submitted %s -> descriptor id %s (%d x %d = %d pts; knob=%s values=%s; url=%s)"
          % (label, did, len(values), len(times_us), g.nseq(), knob, values, url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="pingponggrating heating R&R 2-D scan (pyctrl).")
    ap.add_argument("--knob", required=True, choices=KNOBS)
    ap.add_argument("--values", required=True,
                    help="comma-separated dim-1 knob values (px / rad / count / ms)")
    ap.add_argument("--times", default=DEF_TIMES_US,
                    help="comma-separated release times in US (default %s)" % DEF_TIMES_US)
    ap.add_argument("--nsteps", type=int, default=DEF_NSTEPS)
    ap.add_argument("--period", type=float, default=DEF_PERIOD_MS,
                    help="step_period_ms when not the swept knob (default 0.696)")
    ap.add_argument("--fixed-step-x", type=float, default=0.0)
    ap.add_argument("--fixed-step-y", type=float, default=0.0)
    ap.add_argument("--fixed-step-z", type=float, default=0.0)
    ap.add_argument("--fixed-piston", type=float, default=0.0)
    ap.add_argument("--fill-center", default=None,
                    help="normalized 'cx,cy' beam-center override (default: server beam model)")
    ap.add_argument("--cool", action="store_true",
                    help="KEEP the production post-motion cooling (default: off = direct heating)")
    ap.add_argument("--post-cool-hold-ms", type=float, default=0.5,
                    help="dark hold (ms) replacing the post-motion cool when --cool absent")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()
    submit(args.knob,
           [float(x) for x in args.values.split(",")],
           [float(x) for x in args.times.split(",")],
           url=args.url, reps=args.reps, label=args.label,
           nsteps=args.nsteps, period_ms=args.period,
           fixed_step_x=args.fixed_step_x, fixed_step_y=args.fixed_step_y,
           fixed_step_z=args.fixed_step_z, fixed_piston=args.fixed_piston,
           cool=args.cool, post_cool_hold_ms=args.post_cool_hold_ms,
           fill_center=([float(x) for x in args.fill_center.split(",")]
                        if args.fill_center else None))
