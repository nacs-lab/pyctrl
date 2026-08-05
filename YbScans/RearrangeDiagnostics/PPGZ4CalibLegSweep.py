"""PPGZ4CalibLegSweep.py -- calibrate the Z4 rad -> um axial conversion with the ASYMMETRIC-LEG mode.

THE KNOB (new on the SLM server 2026-07-30, ``rearrange_actual`` / ``rearrange_protocols``):
``extras.leg_defocus`` runs the TWO LEGS of the pingponggrating triangle on DIFFERENT axial maps --

    "true_out"  outward leg = EXACT spherical map (k*(1-sqrt(1-NA^2 rho^2)), step in UM)
                return  leg = PARABOLIC ANSI Z4 (2*rho^2, step in RAD of Z4-PV)
    "z4_out"    the reverse
    None/"off"  both legs on whatever ``true_defocus`` selects (the old behaviour)

Both legs are commanded the SAME nominal per-step displacement: the return leg's amplitude is
converted through the code's own ``kappa = k*NA^2/4 = 1.2475 rad-of-Z4-PV per um`` (NA = 0.65, i.e.
the currently ASSUMED 0.8016 um/rad), then multiplied by ``extras.return_leg_scale``. So sweeping
``return_leg_scale`` sweeps the parabolic leg's PHYSICAL travel against the exact leg's, and the
survival maximum measures what a radian of Z4 ACTUALLY does in microns -- the rad/um conversion,
measured on atoms instead of derived from the lateral cal.

WHERE THE SIGNAL LIVES (and the ONE systematic that must be corrected)

The frame schedule is ``disp_idx = min(k, 2n-k)``, i.e. displacements 0,1,...,n,...,1,0 with the
turnaround frame belonging to the OUTWARD leg. So with outward physical step ``s`` (um) and return
physical step ``s*g`` (``g = scale * kappa_code/kappa_true``):

    outward frames   z_k = k*s            (k = 0..n)
    return  frames   z_k = (2n-k)*s*g     (k = n+1..2n; first return frame is at (n-1)*s*g)

The atoms are CARRIED by the trap, so only the JUMP SIZES matter -- and every return-leg jump is
``s*g ~ s`` except ONE: the leg boundary,

    Delta = n*s - (n-1)*s*g          <- the whole measurement is this single jump

Its sensitivity is amplified by ``(n-1)``: a 1% error in g moves Delta by ``0.01*(n-1)*s`` = 0.94 um
at n = 40, s = 2.4 um, i.e. half an axial cliff. Survival is therefore a PLATEAU in ``scale``, flat
while ``|Delta|`` stays under the single-jump axial cliff (~2 um settled) and falling off sharply on
both sides; the plateau CENTRE is where ``Delta = 0``:

    Delta = 0  <=>  g = n/(n-1)  <=>  scale_peak = r * n/(n-1),   r := kappa_true/kappa_code

    =>  r = scale_peak * (n-1)/n        and       um/rad = 0.8016 / r

The ``n/(n-1)`` factor is NOT a fudge: it is the exact geometry of "the return leg restarts one
displacement index below the peak". At n = 40 it is +2.56%, which is the SAME SIZE as the
calibration error being measured -- so it is measured, not assumed: the scan runs TWO nsteps values
whose predicted plateau centres differ (r*n/(n-1) each). If the two centres land on that spacing,
the geometry model is confirmed and both give the same ``r``. If they land on the SAME scale
instead, the boundary jump is not what I think it is and ``r`` must be read off differently (the
two-n pair is what makes the result falsifiable).

    dim 1: extras.return_leg_scale  -- per-mode grid (the null sits in a different place):
             true_out: 0.96, 0.98, 1.00..1.16 by 0.01, 1.19, 1.22   (21 values)
             z4_out:   0.88, 0.90, 0.92..1.04 by 0.01, 1.07, 1.10   (17 values)
    dim 2: rearrange_kwargs.nsteps = {40, 60}
    => 42 / 34 points.

WHAT JOB 408 (the first run of this scan, 2026-07-31, leg_defocus='true_out', nsteps {20, 40}) GAVE:
  * n = 20 is USELESS -- its plateau half-width is ~4.4% and the row came out flat (S = 0.96-0.996
    across the whole 0.96-1.16 grid), so it cannot localise a centre. Replaced by n = 60 here
    (half-width ~1.4%).
  * n = 40 argmax at scale 1.08 -> r = 1.053, i.e. **Z4 = 0.761 um/rad, ~5% BELOW the derived
    0.8016**. Sign is the expected one (the exact map's beam-weighted quadratic content exceeds its
    paraxial coefficient, so a radian of Z4 buys less travel than 8s^2/(pi*lambda) predicts).
  * the row is SIGN-ASYMMETRIC about the peak: the low-scale side decays gradually (S = 0.38 at
    0.96) while the high side collapses fast (0.99 at 1.10 -> 0.57 at 1.13) and then goes
    NON-MONOTONIC (0.60, 0.62, 0.58 at 1.14-1.16). Fitting a symmetric plateau therefore pulls the
    centre low (1.055 vs the 1.08 argmax); the analysis now fits separate cliff/width per sign of
    Delta and ignores the far tails. The asymmetry is physically expected -- Delta > 0 and Delta < 0
    are jumps in opposite axial directions, and the +z/-z per-step loss already differs 2-4x.

WHY BOTH LEG ORDERS: ``return_leg_scale`` always multiplies the RETURN leg, so in ``true_out`` it
rides the PARABOLIC map and in ``z4_out`` it rides the EXACT one. The null therefore inverts:

    true_out:  Delta = s*(n - (n-1)*scale/r),  null at scale = r*n/(n-1)
    z4_out:    Delta = s*(n/r - (n-1)*scale),  null at scale = n/((n-1)*r)

Both must return the SAME ``r``. Anything that depends on WHICH leg carries the mismatch -- LC
rise/fall asymmetry, the rho^4 residual sitting on the outward vs the return leg, a sign error in
the amplitude transfer -- breaks that equality, so the pair is the bias check.

WHAT THIS CANNOT DO: ``r`` is a single RATIO between the two maps. It does not separate "Z4's
paraxial formula is biased" (NA 0.65 correct, exact map = a true translation, so um/rad = 0.8016/r)
from "NA/lateral cal is off" (Z4 formula exact, NA_eff = 0.65*sqrt(r)). Both readings fit the same
number. Breaking that degeneracy needs an ABSOLUTE axial ruler (through-focus imaging against a
known objective/stage translation, or the atoms' own z_R from trap frequencies), not another
leg-ratio measurement.

If the derived 0.8016 um/rad were already right (r = 1) the centres would sit at 1.026 (n=40) and
1.017 (n=60); the grids here are centred on the job-408 value r = 1.053 (1.080 / 1.071 for
true_out, 0.974 / 0.966 for z4_out) with far anchors that pin the dead floor for the fit.

CONFIG NOTES
  * ``step_size = -2.4 um`` -- MINUS z on purpose: job 323 found the per-step loss 2-4x lower moving
    -z than +z under true_defocus, so the plateau top sits higher and the edges are cleaner. 2.4 um
    is the operating point where job 323 has S = 0.951 (-z) at nsteps 40. The ``rho^4`` residual that
    makes the two maps differ is DIRECTION-ODD, so ``r`` measured here is the -z number; the +z
    repeat (``--step-um +2.4``) is the natural follow-up and any difference between them IS the
    residual's fingerprint, not noise.
  * ``step_period_ms = 3.0`` (not the 0.696 ms production pacing): this is a GEOMETRIC measurement,
    and 0.696 ms is LC-settle-limited (2026-07-22 campaign: the whole lateral cliff moves from 1.28
    to 1.65 px between 0.696 and >=1.4 ms). At 3 ms every frame is the commanded frame, so the
    plateau edges are the true axial cliff and not a settle artifact. Cost is only 0.24 s/shot.
  * ``hold_ms = 5`` keeps the turnaround frame fully settled BEFORE the boundary jump -- the jump is
    the observable, so its starting point must not be a half-relaxed panel.
  * ``depth_piston_corr = 0.617`` rad/um -- the measured true-defocus piston null (job 320). Under
    ``true_defocus`` the server reads it directly in rad-per-UM. ``depth_piston_corr_alt`` is left
    None so the server AUTO-TRANSFERS it through kappa (0.617/1.2475 = 0.4946 rad per rad-of-Z4) and
    both legs apply the same uniform phase per step -- mandatory here, since piston is itself a
    survival ridge and mismatched legs would confound the very signal being measured.
  * ``depth_fill_frac = None`` -- MANDATORY: an explicit positive value activates the deprecated
    beam-model path and OVERRIDES depth_piston_corr, and server extras are STICKY across scans.
  * ``leg_defocus`` bypasses the setup-time uint8 cache (two frames per displacement, one per leg),
    so the frames are float-built and deduped by (displacement, leg): 2*(nsteps+1) = 82 unique
    frames at n=40, ~4 MB each. ``precompute`` + ``precompute_host`` stay ON so the paced hot loop is
    still write-only. Watch ``diag.paced_total_ms`` for overrun.
  * VERIFY IN THE DIAG, first shot: ``leg_defocus_active: true`` (it is stamped on EVERY shot and
    silently False if the mode was refused -- it needs return=True + an axial term), plus
    ``leg_out_true_defocus: true`` / ``leg_ret_true_defocus: false``, ``leg_ret_step`` in rad,
    ``leg_nominal_step_um: -2.4``, ``leg_kappa_z4_rad_per_um: 1.2475``.

Run (pyctrl backend live):
    cd pyctrl
    python YbScans/RearrangeDiagnostics/PPGZ4CalibLegSweep.py --dry-run
    python YbScans/RearrangeDiagnostics/PPGZ4CalibLegSweep.py --force
"""

import argparse
import json
import os
import sys


PATTERN = "33x33_feedback11"
PHASE_PATH = "phase/33x33_feedback11.pt"
BAKED_ZERNIKE = [0.0, 0.0, 0.0, 0.0, 0.0]
MODEL_FILENAME = "SLMnet/checkpoints/sinc_3x3_experiment/models/direct_flat/direct_flat_best.pth"
DEFOCUS = -4.0

LEG_DEFOCUS = "true_out"       # outward = exact spherical (um), return = parabolic Z4 (rad)
STEP_UM = -2.4                 # -z: 2-4x lower per-step loss than +z (job 323); S(-)=0.951 @ n=40
NSTEPS_LIST = [40, 60]         # job 408 showed n=20's plateau is far too broad to localise (flat)
PERIOD_MS = 3.0                # fully settled: the measurement is geometric, not settle-limited
HOLD_MS = 5.0                  # turnaround settled before the boundary jump
PISTON_CORR = 0.617            # rad of uniform phase per UM (true_defocus numeraire; job 320)
KAPPA_CODE = 1.2475            # rad of Z4-PV per um assumed by the server (NA 0.65 -> 0.8016 um/rad)
NUM_PER_GROUP = 650            # ~16 passes on 42 pts -> ~2000 loaded atoms/point
R_SEEN = 1.053                 # job 408 argmax (n=40 peak at scale 1.08) -> the grids are centred here

# dim 1: the return-leg amplitude scale. The null sits at a DIFFERENT place for the two leg orders
# (the scale rides the parabolic leg in true_out and the exact leg in z4_out):
#     true_out:  scale_null = r * n/(n-1)          (r = 1.053 -> 1.080 at n=40, 1.071 at n=60)
#     z4_out:    scale_null = n/((n-1) * r)        (r = 1.053 -> 0.974 at n=40, 0.966 at n=60)
# so each mode gets its own grid: 0.01 spacing across the plateau (half-width ~2.1% at n=40, ~1.4%
# at n=60) plus far anchors that pin the dead floor for the fit.
SCALES_TRUE_OUT = ([0.96, 0.98] + [round(1.00 + 0.01 * i, 3) for i in range(17)] + [1.19, 1.22])
SCALES_Z4_OUT = ([0.88, 0.90] + [round(0.92 + 0.01 * i, 3) for i in range(13)] + [1.07, 1.10])


def default_scales(leg_defocus=LEG_DEFOCUS):
    return list(SCALES_Z4_OUT if str(leg_defocus) == "z4_out" else SCALES_TRUE_OUT)


SCALES = SCALES_TRUE_OUT

def run_desc(leg_defocus=LEG_DEFOCUS, nsteps_list=None, scales=None, step_um=STEP_UM,
             period_ms=PERIOD_MS, hold_ms=HOLD_MS, piston_corr=PISTON_CORR):
    """Scan description (leg-order aware -- the null formula and the grid differ per mode)."""
    ns = [int(n) for n in (nsteps_list or NSTEPS_LIST)]
    sc = [float(s) for s in (scales or default_scales(leg_defocus))]
    z4o = str(leg_defocus) == "z4_out"
    legs = ("outward leg = PARABOLIC ANSI Z4 (amplitude s*kappa, rad), return leg = EXACT spherical "
            "(step in um) and return_leg_scale rides the EXACT leg"
            if z4o else
            "outward leg = EXACT spherical (step in um), return leg = PARABOLIC ANSI Z4 (amplitude "
            "s*kappa*scale, rad) and return_leg_scale rides the PARABOLIC leg")
    null = ("Delta = s*(n/r - (n-1)*scale), null at scale = n/((n-1)*r)" if z4o
            else "Delta = s*(n - (n-1)*scale/r), null at scale = r*n/(n-1)")
    cent = ", ".join("n=%d -> %.4f" % (n, c)
                     for n, c in predicted_centres(ns, r=R_SEEN, leg_defocus=leg_defocus))
    return (
        "Z4 rad->um CALIBRATION via the ASYMMETRIC-LEG pingponggrating mode: extras.leg_defocus='%s' "
        "(%s; the other leg's amplitude is converted through the server's kappa=1.2475 rad/um, i.e. "
        "the assumed 0.8016 um/rad) x return_leg_scale %.3f..%.3f (%d values) x nsteps %s = %d "
        "points, step_size=%.2f um, period %.3f ms, hold_ms=%.1f, depth_piston_corr=%.3f rad/um "
        "(auto-transferred through kappa to the parabolic leg so both legs carry equal piston per "
        "step), depth_fill_frac cleared, no_depth_piston=True, piston=0. "
        "MECHANISM: with disp_idx=min(k,2n-k) the return leg restarts at displacement n-1, so the "
        "ONLY jump that depends on the scale is the leg boundary: %s, with r = kappa_true/kappa_code "
        "and um/rad = 0.8016/r. Sensitivity is amplified by (n-1): 1%% of scale = %.2f um at n=%d. "
        "Predicted null for the job-408 estimate r=%.3f: %s. "
        "CONTEXT: job 408 (leg_defocus='true_out', nsteps {20,40}) put the n=40 argmax at scale 1.08 "
        "-> r = 1.053, i.e. Z4 = 0.761 um/rad, ~5%% BELOW the derived 0.8016; its n=20 row was flat "
        "(plateau half-width ~4.4%%, unlocalisable) which is why n=20 is replaced by n=60 here "
        "(half-width ~1.4%%), and the n=40 row was sign-ASYMMETRIC about the peak (low-scale side "
        "gradual, high side collapsing then a non-monotonic floor), so the extraction now fits "
        "separate cliff/width per sign of Delta. "
        "THIS PAIR of jobs runs BOTH leg orders ('true_out' and 'z4_out'): the scale rides the "
        "parabolic leg in one and the exact leg in the other, so any leg-order bias (LC rise/fall "
        "asymmetry, or the rho^4 residual sitting on the outward vs return leg) shows up as the two "
        "modes NOT returning the same r -- they cannot both be right. "
        "-z chosen (per-step loss 2-4x lower than +z under true_defocus, job 323); the rho^4 "
        "residual is direction-odd so this is the -z calibration and a +2.4 um repeat is the "
        "follow-up. 3 ms pacing (not 0.696) so the frames are the commanded frames: 0.696 ms is "
        "LC-settle-limited (07-22 campaign) and would put a settle artifact inside a geometric "
        "calibration. leg_defocus bypasses the uint8 cache -> deduped float build, 2*(n+1) unique "
        "frames (122 at n=60). Array 33x33_feedback11, z4 = loading_defocus = -4."
        % (leg_defocus, legs, min(sc), max(sc), len(sc), ns, len(sc) * len(ns), step_um,
           period_ms, hold_ms, piston_corr, null, 0.01 * (max(ns) - 1) * abs(step_um), max(ns),
           R_SEEN, cent))


# NOTE: no module-level RUN_DESC -- run_desc() depends on predicted_centres() (defined below) and
# on the per-mode grid, so submit() builds the description at call time.


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def _image_patterns_json():
    """Both camera frames are the SAME array (the ping-pong returns to the source sites). Name =
    the phase-file BASENAME -- what the detection registry + expConfig ByPattern are keyed by."""
    it = {"name": PATTERN, "base_phase_path": PHASE_PATH, "order": "col",
          "legacy_zerniked": False}
    return json.dumps([it, dict(it)])


def build(step_um=STEP_UM, nsteps_list=None, scales=None, period_ms=PERIOD_MS,
          hold_ms=HOLD_MS, piston_corr=PISTON_CORR, leg_defocus=LEG_DEFOCUS,
          num_per_group=NUM_PER_GROUP):
    """Build (do NOT submit) the 2-D [return_leg_scale x nsteps] ScanGroup. Returns (seq_name, g)."""
    _bootstrap()
    from scan_group import ScanGroup

    scl = [float(s) for s in (scales if scales else default_scales(leg_defocus))]
    ns = [int(n) for n in (nsteps_list if nsteps_list else NSTEPS_LIST)]

    seq_name = "RearrangeCommSeq"
    g = ScanGroup()
    g().rearrange_kwargs.extras.n_rounds = 1

    # ---- warmup_kwargs (runp; forwarded ONCE at dequeue with reset_params) -------------
    rp = g.runp()
    rp.warmup_kwargs.model_filename = MODEL_FILENAME
    rp.warmup_kwargs.initial_phase = PHASE_PATH
    rp.warmup_kwargs.final_phase = PHASE_PATH
    rp.warmup_kwargs.extras.grid_rotation = 90
    rp.warmup_kwargs.extras.initial_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.extras.final_phase_zernike = list(BAKED_ZERNIKE)
    rp.warmup_kwargs.compile_mode = "max-autotune-no-cudagraphs"
    rp.warmup_kwargs.use_fp16 = True
    rp.warmup_kwargs.use_channels_last = True
    rp.warmup_kwargs.use_compile = True
    rp.warmup_kwargs.compile_fullgraph = True
    rp.warmup_kwargs.cuda_graph = True
    rp.warmup_kwargs.derive_threshold = 0.35

    # ---- rearrange_kwargs: pingponggrating, TRUE-DEFOCUS depth mode, LEG SPLIT ---------
    rk = g().rearrange_kwargs
    rk.protocol = "pingponggrating"
    rk.step_period_ms = float(period_ms)

    # Fixed axial step, MICRONS (the outward/exact leg's numeraire under true_defocus).
    rk.extras.step_size = float(step_um)
    rk.extras.depth = True
    rk.extras.true_defocus = True          # outward leg = exact spherical map (step in um)

    # THE CALIBRATION MODE. String extra; explicit because server extras are sticky.
    rk.extras.leg_defocus = str(leg_defocus)
    # dim 1: the return (parabolic-Z4) leg's amplitude scale -- the null axis.
    rk.extras.return_leg_scale.scan(1, list(scl))
    # dim 2: nsteps -> the (n-1) lever on the boundary jump AND the geometry cross-check.
    rk.nsteps.scan(2, list(ns))

    # Piston: nulled on the exact leg at the measured value, AUTO-transferred through kappa to the
    # parabolic leg (depth_piston_corr_alt left unset) so both legs sit at the same point on the
    # piston survival ridge -- otherwise the ridge would masquerade as the calibration signal.
    rk.extras.depth_piston_corr = float(piston_corr)
    # MANDATORY: an explicit positive depth_fill_frac OVERRIDES depth_piston_corr (legacy
    # beam-model path) and is sticky across scans.
    rk.extras.depth_fill_frac = None
    rk.extras.no_depth_piston = True
    rk.extras.piston = 0.0
    # depth_fill_center: left UNSET -> the server's measured beam centroid (-0.0684, +0.0117).

    # Turnaround dwell: the boundary jump is the observable, so it must start from a settled panel.
    rk.extras.hold_ms = float(hold_ms)
    # A corrective/aberration Zernike is not in play here.
    rk.extras.reverse_zernike = False

    rk.extras.precompute = True
    rk.extras.precompute_host = True
    rk.extras.hw_sequence = False
    rk.extras.ifEnhanced = True
    rk.extras.z4 = DEFOCUS                      # rearrange focal plane == loading_defocus
    rk.extras.initial_pattern = PATTERN
    rk.extras.final_pattern = PATTERN

    # ---- run params (runp) --------------------------------------------------------------
    # 650 -> ~16 passes on a 42-point grid, ~2000 loaded atoms/point (sigma_S ~ 0.005-0.010). The
    # centre comes from the plateau EDGES, which swing ~0.5 in survival per 0.01 of scale, so this
    # is far more statistics than the centring needs -- job 408 used 3400/point and was overkill.
    rp.NumPerGroup = int(num_per_group)
    rp.loading_defocus = DEFOCUS
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isGrid2 = 0
    rp.isInit = 0
    rp.isHC = 0
    rp.useScanLongSlmLock = 1
    rp.imagePatternsJson = _image_patterns_json()

    return seq_name, g


def predicted_centres(nsteps_list=None, r=1.0, leg_defocus=LEG_DEFOCUS):
    """Plateau centre in ``return_leg_scale`` per nsteps for a given r = kappa_true/kappa_code.

    ``true_out``: the scale rides the PARABOLIC (return) leg -> centre = r*n/(n-1).
    ``z4_out``:   the scale rides the EXACT (return) leg     -> centre = n/((n-1)*r).
    """
    ns = [int(n) for n in (nsteps_list if nsteps_list else NSTEPS_LIST)]
    z4o = str(leg_defocus) == "z4_out"
    return [(n, (n / ((n - 1.0) * r)) if z4o else (r * n / (n - 1.0))) for n in ns]


def um_per_rad(scale_peak, nsteps, leg_defocus=LEG_DEFOCUS):
    """Invert the measurement: plateau centre + nsteps -> the calibrated Z4 um/rad."""
    n = int(nsteps)
    if str(leg_defocus) == "z4_out":
        r = n / ((n - 1.0) * float(scale_peak))
    else:
        r = float(scale_peak) * (n - 1.0) / n
    return (1.0 / KAPPA_CODE) / r


def submit(url=None, reps=None, **kw):
    seq_name, g = build(**kw)
    from yb_start_scan import ybStartScan
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan(seq_name, g, url=url, label="PPGZ4CalibLegSweep",
                      description=run_desc(
                          leg_defocus=kw.get("leg_defocus", LEG_DEFOCUS),
                          nsteps_list=kw.get("nsteps_list"), scales=kw.get("scales"),
                          step_um=kw.get("step_um", STEP_UM),
                          period_ms=kw.get("period_ms", PERIOD_MS),
                          hold_ms=kw.get("hold_ms", HOLD_MS),
                          piston_corr=kw.get("piston_corr", PISTON_CORR)), **opts)
    print("submitted PPGZ4CalibLegSweep -> descriptor id %s (%d pts; leg_defocus=%s, step=%.2f um, "
          "nsteps=%s, period=%.3f ms, hold=%.1f, corr=%.3f rad/um; url=%s)"
          % (did, g.nseq(), kw.get("leg_defocus", LEG_DEFOCUS), kw.get("step_um", STEP_UM),
             kw.get("nsteps_list") or NSTEPS_LIST, kw.get("period_ms", PERIOD_MS),
             kw.get("hold_ms", HOLD_MS), kw.get("piston_corr", PISTON_CORR), url or "default"))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Z4 rad->um calibration via the asymmetric-leg pingponggrating mode.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--step-um", type=float, default=STEP_UM,
                    help="signed axial step in um (default %.2f -- the lower-loss -z)" % STEP_UM)
    ap.add_argument("--nsteps-list", default=None,
                    help="comma-separated nsteps (default %s)" % NSTEPS_LIST)
    ap.add_argument("--scales", default=None,
                    help="comma-separated return_leg_scale values (default 0.96..1.16 by 0.01)")
    ap.add_argument("--period", type=float, default=PERIOD_MS)
    ap.add_argument("--hold-ms", type=float, default=HOLD_MS)
    ap.add_argument("--corr", type=float, default=PISTON_CORR,
                    help="depth_piston_corr in rad/um (default %.3f)" % PISTON_CORR)
    ap.add_argument("--leg", default=LEG_DEFOCUS, choices=("true_out", "z4_out"))
    ap.add_argument("--nper", type=int, default=NUM_PER_GROUP,
                    help="NumPerGroup (default %d)" % NUM_PER_GROUP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="required to submit")
    args = ap.parse_args()

    _sc = ([float(x) for x in args.scales.split(",") if x.strip()] if args.scales else None)
    _nl = ([int(x) for x in args.nsteps_list.split(",") if x.strip()]
           if args.nsteps_list else None)
    kw = dict(step_um=args.step_um, nsteps_list=_nl, scales=_sc, period_ms=args.period,
              hold_ms=args.hold_ms, piston_corr=args.corr, leg_defocus=args.leg,
              num_per_group=args.nper)
    if args.dry_run:
        _seq, _g = build(**kw)
        _ns = _nl or NSTEPS_LIST
        _s = _sc or default_scales(args.leg)
        print("seq=%s  nseq=%d  leg_defocus=%s  step=%.2f um  period=%g ms  hold=%g  corr=%g rad/um"
              % (_seq, _g.nseq(), args.leg, args.step_um, args.period, args.hold_ms, args.corr))
        print("scales  = %s (%d)" % (_s, len(_s)))
        print("nsteps  = %s -> frames %s, unique float frames %s"
              % (_ns, [2 * n + 1 for n in _ns], [2 * (n + 1) for n in _ns]))
        print("one-way excursion = %s um ; total path = %s um"
              % ([round(n * abs(args.step_um), 1) for n in _ns],
                 [round(2 * n * abs(args.step_um), 1) for n in _ns]))
        for n, c in predicted_centres(_ns, r=1.0, leg_defocus=args.leg):
            print("  null if 0.8016 um/rad exact (r=1):   nsteps=%d -> scale %.4f "
                  "(half-width ~%.3f for a 2 um single-jump cliff)"
                  % (n, c, 2.0 / ((n - 1) * abs(args.step_um))))
        for n, c in predicted_centres(_ns, r=R_SEEN, leg_defocus=args.leg):
            print("  null for the job-408 estimate r=%.3f: nsteps=%d -> scale %.4f" % (R_SEEN, n, c))
    elif not args.force:
        ap.error("refusing to submit without --force (use --dry-run to inspect)")
    else:
        submit(url=args.url, reps=args.reps, **kw)
