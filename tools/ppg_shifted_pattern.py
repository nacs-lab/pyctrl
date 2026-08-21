"""ppg_shifted_pattern.py -- register knm-shifted copies of a loading pattern.

``pingponggrating`` with ``return=False`` (outbound-only) leaves the WHOLE tweezer
array rigidly TRANSLATED by ``step_size * nsteps`` knm-px when the second image is
taken.  The frame-1 detection boxes therefore have to sit at the shifted positions,
not the loading positions.

The lab-side detection grid for camera frame ``k`` comes from ``imagePatternsJson``
entry ``k`` -> the pattern registry record's ``knm`` -> the global SLM->camera affine
(``data_manager._build_pattern_grids``, entry k -> frame k).  So the clean way to move
frame 1's boxes is to register a NEW pattern whose ``knm`` is the base pattern's knm
plus the commanded displacement, and name it as the frame-1 entry.

Why this is stable (and does not get clobbered):
``_build_pattern_grids`` calls ``fetch_or_refresh_pattern(name, base_phase_path=...)``,
which returns the CACHED record untouched whenever ``_params_match`` succeeds.  That
guard compares ``base_phase_path / order / fft_shape / threshold / min_dist /
legacy_zerniked / baked_zernike / planes_z_rad`` and requires a truthy ``base_sha256``
-- it does NOT look at ``knm``.  We copy all of those fields verbatim from the base
record, so a shifted record is a permanent cache hit and is never re-derived from the
(unshifted) phase file.

Coordinate convention
---------------------
Registry ``knm`` is stored ``[y, x]``; ``affine_transform._knm_to_xy`` swaps to
``[x, y]`` exactly once before the affine.  A ``+x`` blaze displacement is therefore
added to ``knm[:, 1]``.  Verified against the live affine: ``+1`` knm-x maps to
``dX = +2.268``, ``dY = -0.033`` camera px (a pure camera-X move).

Usage::

    python pyctrl/tools/ppg_shifted_pattern.py --list
    python pyctrl/tools/ppg_shifted_pattern.py --base 33x33_feedback11 \
        --shifts 0.25,0.5 --nsteps 1,2,5 --apply
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Fields that define extraction identity in pattern_registry._params_match. Copying
# every one of them verbatim is what makes a shifted record a permanent cache hit.
_IDENTITY_FIELDS = (
    "base_phase_path", "order", "fft_shape", "threshold", "min_dist",
    "legacy_zerniked", "baked_zernike", "planes_z_rad", "base_sha256",
)
# Carried so the shifted pattern behaves like the base everywhere else.
_CARRY_FIELDS = (
    "phases", "lattice", "n_sites", "is_3d", "z_rad", "positions_knm3d",
    "n_per_plane", "plane_of_site", "default_loading_zernike", "source_endpoint",
)

SHIFT_NAME_FMT = "%s__dx%s"


def shifted_name(base, dx):
    """Registry name for the base pattern displaced ``dx`` knm-px in +x.

    ``dx`` is formatted to 3 decimals with ``.``/``-`` made filename-safe, so
    12.5 -> ``33x33_feedback11__dx12p500``.
    """
    tag = ("%.3f" % float(dx)).replace("-", "m").replace(".", "p")
    return SHIFT_NAME_FMT % (base, tag)


def build_shifted_record(base_rec, dx, name):
    """Copy ``base_rec`` with ``knm[:, 1] += dx`` (knm is [y, x]) under a new name."""
    knm = np.asarray(base_rec["knm"], dtype=np.float64).copy()
    if knm.ndim != 2 or knm.shape[1] != 2:
        raise ValueError("unexpected knm shape %r for %r"
                         % (knm.shape, base_rec.get("name")))
    knm[:, 1] += float(dx)          # [y, x] -> shift the x column

    rec = {"name": name, "knm": [[float(a), float(b)] for a, b in knm]}
    for f in _IDENTITY_FIELDS + _CARRY_FIELDS:
        if f in base_rec:
            rec[f] = base_rec[f]
    # Provenance: not read by _params_match, purely so a human (or a later audit)
    # can tell this record was synthesized rather than derived from a phase file.
    rec["derived_from"] = base_rec.get("name")
    rec["knm_shift_x"] = float(dx)
    rec["synthetic_shift"] = True
    return rec


def offsensor_count(knm, roi, margin=6):
    """Sites whose detection box would fall outside ``roi = [Xoff, Yoff, W, H]``."""
    from yb_analysis.analysis import affine_transform as aff
    A = aff.load_matrix()
    if A is None:
        raise RuntimeError("no committed affine")
    cam = aff.apply_affine_cropped(aff._knm_to_xy(np.asarray(knm, float)), A, roi)
    cy, cx = cam[:, 0], cam[:, 1]
    H, W = float(roi[3]), float(roi[2])
    bad = (cy < margin) | (cy > H - margin - 1) | (cx < margin) | (cx > W - margin - 1)
    return int(bad.sum())


def ensure_shifted_patterns(base, shifts, roi=None, apply=False, margin=6):
    """Create (or verify) a registry record per displacement in ``shifts``.

    Returns a list of ``(dx, name, n_offsensor)``. With ``apply=False`` nothing is
    written -- it only reports what WOULD be written, including the off-sensor count
    so a scan can refuse to probe a displacement that walks atoms off the ROI.
    """
    from yb_analysis.analysis import pattern_registry as reg

    base_rec = reg.get_pattern(base)
    if base_rec is None:
        raise SystemExit("base pattern %r not in registry" % base)

    out = []
    for dx in shifts:
        name = shifted_name(base, dx)
        rec = build_shifted_record(base_rec, dx, name)
        n_off = (offsensor_count(rec["knm"], roi, margin) if roi is not None else -1)
        existing = reg.get_pattern(name)
        if apply and existing is None:
            reg.write_pattern(rec)
        out.append((dx, name, n_off, "exists" if existing is not None
                    else ("written" if apply else "would-write")))
    return out


def _parse_floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _parse_ints(s):
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="33x33_feedback11")
    ap.add_argument("--shifts", default=None,
                    help="explicit comma-separated knm-px displacements")
    ap.add_argument("--step-sizes", default=None,
                    help="comma-separated step sizes (knm-px/step)")
    ap.add_argument("--nsteps", default=None, help="comma-separated nsteps")
    ap.add_argument("--roi", default="1000,100,2100,2100",
                    help="Xoff,Yoff,W,H (default the production Orca ROI)")
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--list", action="store_true", help="list synthetic records")
    ap.add_argument("--apply", action="store_true", help="actually write records")
    args = ap.parse_args()

    from yb_analysis.analysis import pattern_registry as reg

    if args.list:
        for n, r in sorted(reg.list_patterns().items()):
            if r.get("synthetic_shift") or "__dx" in n:
                print("%-44s dx=%+8.3f  from %s"
                      % (n, r.get("knm_shift_x", float("nan")), r.get("derived_from")))
        return

    if args.shifts:
        shifts = sorted({round(v, 4) for v in _parse_floats(args.shifts)})
    elif args.step_sizes and args.nsteps:
        shifts = sorted({round(s * n, 4)
                         for s in _parse_floats(args.step_sizes)
                         for n in _parse_ints(args.nsteps)})
    else:
        ap.error("need --shifts, or both --step-sizes and --nsteps")

    roi = _parse_floats(args.roi)
    rows = ensure_shifted_patterns(args.base, shifts, roi=roi,
                                   apply=args.apply, margin=args.margin)
    print("base=%s  roi=%s  n=%d" % (args.base, roi, len(rows)))
    for dx, name, n_off, status in rows:
        flag = "  <-- OFF-SENSOR" if n_off > 0 else ""
        print("  dx=%+8.3f knm-px (%6.2f cam-px)  %-44s %-11s off=%d%s"
              % (dx, dx * 2.268, name, status, n_off, flag))
    worst = max((r[2] for r in rows), default=0)
    if worst > 0:
        print("\nWARNING: %d displacement(s) push sites off the ROI; those points would "
              "under-report survival. Cap the travel or widen the ROI." % worst)


if __name__ == "__main__":
    main()
