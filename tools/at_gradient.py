"""at_gradient.py -- half-array Autler-Townes splitting = the 308 beam-CENTRING metric.

The array-average AT splitting is a poor alignment observable when the array sits on a FLANK
of the 308 beam: the mean is dominated by the middle of the array and barely moves, while the
beam centre is still walking. Splitting the array in half and fitting each half separately is
far more sensitive -- the difference (high-half minus low-half) is the beam-centring error
along that axis, and it should be driven to ZERO.

    dx = split(x_high) - split(x_low)      dy = split(y_high) - split(y_low)

dx = 0 and dy = 0 means the 308 beam is centred on the array along that axis. The sign says
which way to move: dx < 0 means the low-x half is brighter, i.e. the beam sits toward low x.

Each half is fit with the same double-Lorentzian model used by fit_at_splitting.py, on the
loading-weighted pooled P11 of the sites in that half, so the errors are honest.

Run from PROJECT ROOT with the yb_analysis env:
    python pyctrl/tools/at_gradient.py <scan_id> [<scan_id> ...]

Prints one line per scan; with several scans it prints the trend so consecutive picomotor
steps can be compared directly.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _persite_common import load_combined  # noqa: E402


def _lor(x, A, x0, w):
    return A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def _double_dip(x, y0, A1, x1, w1, A2, x2, w2):
    """INDEPENDENT centre/width/depth per dip -- the AT doublet is NOT symmetric and the two
    dips do not share a width, so a shared-width / symmetric-about-centre model slides and
    manufactures scatter. Same model + seeding as tools/fit_at_splitting.py."""
    return y0 - _lor(x, A1, x1, w1) - _lor(x, A2, x2, w2)


def _smooth(y, k=3):
    m = np.isfinite(y)
    ys = np.where(m, y, 0.0)
    w = m.astype(float)
    num = np.convolve(ys, np.ones(k), "same")
    den = np.convolve(w, np.ones(k), "same")
    return np.where(den > 0, num / den, np.nan)


def _seed_two_dips(x, ys, min_sep=0.4):
    """Two deepest minima at least `min_sep` MHz apart (robust double-dip seed)."""
    finite = np.where(np.isfinite(ys))[0]
    o = finite[np.argsort(ys[finite])]
    d1 = o[0]
    d2 = next((k for k in o[1:] if abs(x[k] - x[d1]) > min_sep), o[1] if len(o) > 1 else o[0])
    return sorted([float(x[d1]), float(x[d2])])


def _fit_half(freq_mhz, p11, n_load, mask, min_sep=0.4):
    """Loading-weighted pooled P11 over `mask`, fit with the robust independent double dip.
    Returns (splitting, splitting_err, centre_of_pair)."""
    from scipy.optimize import curve_fit
    num = np.nansum(np.nan_to_num(p11[mask] * n_load[mask]), axis=0)
    den = np.nansum(n_load[mask], axis=0)
    y = num / np.maximum(den, 1)
    e = np.sqrt(np.maximum(y * (1 - y), 1e-6) / np.maximum(den, 1))
    m = np.isfinite(y)
    xf, yf, ef = freq_mhz[m], y[m], e[m]
    if xf.size < 7:
        return float("nan"), float("nan"), float("nan")
    ys = _smooth(y, 3)
    span = float(freq_mhz.max() - freq_mhz.min())
    y0g = float(np.nanpercentile(yf, 90))
    s1, s2 = _seed_two_dips(freq_mhz, ys, min_sep)
    Ag = max(y0g - float(np.nanmin(yf)), 0.05)
    p0 = [y0g, Ag, s1, 0.3, Ag, s2, 0.3]
    lo = [0, 0, freq_mhz.min(), 0.05, 0, freq_mhz.min(), 0.05]
    hi = [1.2, 1.5, freq_mhz.max(), span, 1.5, freq_mhz.max(), span]
    try:
        p, c = curve_fit(_double_dip, xf, yf, p0=p0, bounds=(lo, hi),
                         sigma=ef, absolute_sigma=True, maxfev=60000)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), float("nan")
    err = np.sqrt(np.diag(c))
    comps = sorted([(p[2], err[2]), (p[5], err[5])], key=lambda t: t[0])
    split = abs(comps[1][0] - comps[0][0])
    split_err = float(np.hypot(comps[0][1], comps[1][1]))
    centre = 0.5 * (comps[0][0] + comps[1][0])
    return float(split), split_err, float(centre)


def analyse(scan_id, seed_split=None, seed_centre=None):
    c = load_combined([scan_id])
    freq = np.asarray(c["freq_hz"], float) / 1e6
    p11, n_load = c["p11"], c["n_load"]
    xs = np.asarray(c["xs"], float)
    ys = np.asarray(c["ys"], float)
    if not c.get("have_xy", True) or not np.isfinite(xs).any():
        raise SystemExit("scan %s has no per-site x/y -- cannot split the array" % scan_id)
    out = {}
    for name, mask in (("x_low", xs < np.median(xs)), ("x_high", xs >= np.median(xs)),
                       ("y_low", ys < np.median(ys)), ("y_high", ys >= np.median(ys))):
        out[name] = _fit_half(freq, p11, n_load, mask)
    dx = out["x_high"][0] - out["x_low"][0]
    dy = out["y_high"][0] - out["y_low"][0]
    edx = float(np.hypot(out["x_high"][1], out["x_low"][1]))
    edy = float(np.hypot(out["y_high"][1], out["y_low"][1]))
    allmask = np.ones(len(xs), bool)
    mean = _fit_half(freq, p11, n_load, allmask)
    return {"scan_id": scan_id, "mean": mean[0], "mean_err": mean[1],
            "x_low": out["x_low"][0], "x_high": out["x_high"][0], "dx": dx, "dx_err": edx,
            "y_low": out["y_low"][0], "y_high": out["y_high"][0], "dy": dy, "dy_err": edy}


def main():
    ap = argparse.ArgumentParser(description="Half-array AT splitting (308 beam centring).")
    ap.add_argument("scan_ids", nargs="+")
    ap.add_argument("--seed-split", type=float, default=0.83)
    ap.add_argument("--seed-centre", type=float, default=97.35)
    a = ap.parse_args()
    rows = []
    for sid in a.scan_ids:
        r = analyse(sid, a.seed_split, a.seed_centre)
        rows.append(r)
        print("%s  mean %.4f+-%.4f | x_low %.4f x_high %.4f  dx %+.4f+-%.4f "
              "| y_low %.4f y_high %.4f  dy %+.4f+-%.4f"
              % (r["scan_id"], r["mean"], r["mean_err"], r["x_low"], r["x_high"],
                 r["dx"], r["dx_err"], r["y_low"], r["y_high"], r["dy"], r["dy_err"]))
    if len(rows) > 1:
        print("\ntrend (dx -> 0 and dy -> 0 = beam centred):")
        for r in rows:
            print("   %s  dx %+.4f  dy %+.4f  mean %.4f"
                  % (r["scan_id"], r["dx"], r["dy"], r["mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
