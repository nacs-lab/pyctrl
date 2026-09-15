"""Per-site detection health (d-prime) straight from a run's h5 -- shared by the fit tools.

Why this exists: ``analyze_scan``'s cached ``discrimination`` block comes back with
``source = scan_init`` and ``dprime``/``fill``/``gap`` all ``None`` on a normal run, so every
analysis that wanted detection health was recomputing it by hand from ``intensities_img1``
(two agent sessions did exactly that on 2026-09-15, ~1 min each). This makes it a one-line call
so the fit tools can print it and nobody has to re-derive it.

Judge detection by **d-prime**, not by absolute ADU: at the current camera gain the atom peak sits
only ~8-9 ADU above a ~200 pedestal, so "max ADU < 500 = blank" is useless (see the daily-system-scan
runbook). Healthy is d-prime >~ 4 with fill 0.25-0.75.
"""

import os

import numpy as np


def detection_health(scan_dir, sid, scan_json=None, n_shots=None, img="intensities_img1"):
    """Return a dict of per-site d-prime stats, or None if it cannot be computed.

    Splits each site's pooled intensities at that site's stored threshold and takes the
    normalized separation of the two populations. Never raises -- a fit must not die because
    the health read failed.

    Parameters
    ----------
    scan_dir : str   the run's data dir
    sid : str        file_id, e.g. "20260915_125339"
    scan_json : dict optional already-loaded data_<sid>.json (for ``initThresholds``)
    n_shots : int    optional cap (use the same shot count the fit used)
    img : str        which image's intensities to judge ("intensities_img1" or "_img2")
    """
    try:
        import json

        import h5py

        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        if scan_json is None:
            with open(os.path.join(scan_dir, "data_%s.json" % sid)) as fh:
                scan_json = json.load(fh)
        thr = np.asarray(scan_json.get("initThresholds"), float).ravel()
        with h5py.File(os.path.join(scan_dir, "data_%s.h5" % sid), "r", swmr=True) as f:
            if img not in f:
                return None
            I = f[img][: (n_shots or f[img].shape[0])]
        if thr.size != I.shape[1]:
            return None
        dp, fills = [], []
        for s in range(I.shape[1]):
            col = I[:, s]
            hi, lo = col[col >= thr[s]], col[col < thr[s]]
            fills.append(hi.size / float(col.size) if col.size else np.nan)
            if hi.size >= 3 and lo.size >= 3:
                sd = np.sqrt(0.5 * (hi.std() ** 2 + lo.std() ** 2))
                if sd > 0:
                    dp.append((hi.mean() - lo.mean()) / sd)
        if not dp:
            return None
        dp = np.asarray(dp)
        fills = np.asarray(fills, float)
        return {"dprime_median": float(np.median(dp)),
                "dprime_p10": float(np.percentile(dp, 10)),
                "n_sites": int(dp.size),
                "n_below_2": int((dp < 2).sum()),
                "fill_median": float(np.nanmedian(fills)),
                "healthy": bool(np.median(dp) >= 4.0 and 0.2 <= np.nanmedian(fills) <= 0.8)}
    except Exception:
        return None


def format_health(h):
    """One line for a report, or an honest note that it is unavailable."""
    if not h:
        return "detection: d-prime unavailable (no intensities/thresholds) -- judge from loading"
    return ("detection: per-site d-prime median %.2f (p10 %.2f), %d/%d sites < 2, fill %.2f -> %s"
            % (h["dprime_median"], h["dprime_p10"], h["n_below_2"], h["n_sites"],
               h["fill_median"], "HEALTHY" if h["healthy"] else "SUSPECT"))


# --------------------------------------------------------------------------------------------
# Run provenance: the operating point AS RUN.
# --------------------------------------------------------------------------------------------
# The operating-point rule (yb-basic) says never to assume the sublevel or the field. The run's
# own descriptor `set_params` are the authoritative record -- better than reading expConfig.py,
# which may have been edited since the run. Printing this saves every reader the archaeology.

# Params worth showing for a spectroscopy / Rydberg push-out run, in report order.
_PROV_KEYS = [
    ("Pushout.BiasCoilCurrent.Ryd", "Ryd bias", "%g G"),
    ("Init.EOM616.Freq", "616 park", "%.3f MHz"),
    ("Pushout.Ryd308.Amp", "Ryd308.Amp", "%g"),
    ("Pushout.Green.Amp", "556 push amp", "%g"),
    ("Pushout.Time", "push time", "%g s"),
    ("Pushout.Green.Freq", "556 push freq", "%.4f MHz"),
    ("AWG.AWG556.Ch1.carrier_freq_MHz", "556 carrier", "%.4f MHz"),
]


def run_provenance(scan_dir, sid, scan_json=None):
    """Return {'scan_name':..., 'pattern':..., 'params':[(label, text), ...]} or None.

    Reads the run's own data_<sid>.json. Never raises.
    """
    try:
        import json

        if scan_json is None:
            with open(os.path.join(scan_dir, "data_%s.json" % sid)) as fh:
                scan_json = json.load(fh)
        desc = scan_json.get("descriptor")
        if isinstance(desc, str):
            desc = json.loads(desc)
        params = (desc or {}).get("params") or {}

        name = scan_json.get("ScanName")
        if isinstance(name, dict) and name.get("scanname"):
            name = "".join(chr(c) for c in name["scanname"])
        elif not isinstance(name, str):
            name = (desc or {}).get("seq")

        pattern = None
        for key in ("calibrationSource", "pattern", "loading_pattern"):
            v = scan_json.get(key)
            if isinstance(v, str) and v:
                pattern = v
                break

        out = []
        for key, label, fmt in _PROV_KEYS:
            v = params.get(key)
            if isinstance(v, dict):          # a swept axis, not a fixed setting
                continue
            if v is None:
                continue
            if "MHz" in fmt and abs(float(v)) > 1e6:   # stored in Hz
                v = float(v) / 1e6
            try:
                out.append((label, fmt % float(v)))
            except Exception:
                out.append((label, str(v)))
        return {"scan_name": name, "pattern": pattern, "params": out}
    except Exception:
        return None


def format_provenance(p):
    """One line: the operating point as the run actually ran it."""
    if not p:
        return ("operating point: unavailable (no descriptor) -- state it from the caller, "
                "do NOT assume it")
    bits = ", ".join("%s %s" % (lab, txt) for lab, txt in p["params"])
    head = "as-run: %s" % (p.get("scan_name") or "?")
    if p.get("pattern"):
        head += " on %s" % p["pattern"]
    return head + (" | " + bits if bits else "") + (
        "  [sublevel/mj is NOT encoded in the run -- take it from the caller]")
