"""StirapGapDarkScan.py -- RearrangeSTIRAP with ALL pulses dark, sweeping only the STIRAP gap.

Created 2026-08-05 (user request). Reuses ``RearrangeSTIRAPScan.build()`` verbatim -- same
rearrangement, same frame layout, same warmup -- then applies two overrides:

  1. **Every AWG pulse amplitude_scale -> 0** (556 Ch1/Ch2, 308 Ch1/Ch2). The base scan already
     has 556 Ch1/Ch2 and 308 Ch1 at 0; only 308 Ch2 was live at 0.95. With all four at 0 no
     Rydberg pulse is applied at all, so the sequence is a pure *dark hold*.
  2. **``Pushout.STIRAPGap`` swept 0 .. 20 us** (the base scan pins it at 1e-6 and notes it is
     scannable for a Rydberg-lifetime sweep).

Readout is survival vs the dark hold. With no pulses this measures the BACKGROUND loss of the
STIRAP slot itself -- trap-off / hold-time loss, vacuum, and any un-gated field or light leak --
i.e. the floor that any real STIRAP gap-dependence sits on top of. A flat trace means the slot
costs nothing; a decay means the hold itself is losing atoms and any measured "Rydberg lifetime"
would be contaminated by it.

Run (pyctrl backend live; from pyctrl/):
    python YbScans/StirapGapDarkScan.py --reps 12
    python YbScans/StirapGapDarkScan.py --tmax 20e-6 --tstep 1e-6 --reps 12
"""

import argparse

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent; explicit so it's never stripped)

from RearrangeSTIRAPSeq import RearrangeSTIRAPSeq
import RearrangeSTIRAPScan as base


DEF_TSTEP = 1e-6       # STIRAP-gap step (s)
DEF_TMAX = 20e-6       # STIRAP-gap upper bound (s)


def build(tstep=DEF_TSTEP, tmax=DEF_TMAX):
    """The base RearrangeSTIRAP ScanGroup with all pulses dark and the gap swept."""
    from scan_export import matlab_colon

    g = base.build()

    # ---- (1) ALL pulses dark: zero every AWG channel amplitude_scale --------------------
    # 556 Ch1/Ch2 and 308 Ch1 are already 0 in the base scan; 308 Ch2 is the live one (0.95).
    g().AWG.AWG556.Ch1.amplitude_scale = 0
    g().AWG.AWG556.Ch2.amplitude_scale = 0
    g().AWG.AWG308.Ch1.amplitude_scale = 0
    g().AWG.AWG308.Ch2.amplitude_scale = 0

    # ---- (2) sweep the STIRAP gap (the dark hold) ---------------------------------------
    # The base scan PINS Pushout.STIRAPGap as a fixed scalar (1e-6), and ScanGroup refuses to
    # scan a fixed parameter ("Cannot scan a fixed parameter", _check_noconflict). There is no
    # public un-fix API, so drop the fixed entry from the base params tree before scanning it.
    _base_params = g._base["params"]
    if "Pushout" in _base_params:
        _base_params["Pushout"].pop("STIRAPGap", None)

    gaps = [v * 1e-6 for v in matlab_colon(0.0, tstep * 1e6, tmax * 1e6)]
    g().Pushout.STIRAPGap.scan(1, gaps)

    return g, gaps


def submit(url=None, reps=12, tstep=DEF_TSTEP, tmax=DEF_TMAX):
    from yb_start_scan import ybStartScan

    g, gaps = build(tstep=tstep, tmax=tmax)
    opts = {"rep": reps} if reps is not None else {}
    did = ybStartScan(RearrangeSTIRAPSeq, g, url=url, label="StirapGapDarkScan",
                      description=("STIRAP gap sweep with ALL pulses dark (556 Ch1/Ch2 + 308 "
                                   "Ch1/Ch2 amplitude_scale = 0): Pushout.STIRAPGap %.1f..%.1f us, "
                                   "%d pts. Survival vs dark hold = the background loss floor of "
                                   "the STIRAP slot (no Rydberg excitation applied)."
                                   % (gaps[0] * 1e6, gaps[-1] * 1e6, len(gaps))),
                      **opts)
    print("submitted StirapGapDarkScan -> descriptor id %s (url=%s, reps=%s, %d gap pts %.1f-%.1f us, "
          "ALL pulses amp_scale=0)"
          % (did, url or "default", reps, len(gaps), gaps[0] * 1e6, gaps[-1] * 1e6))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="RearrangeSTIRAP with all pulses dark, sweeping Pushout.STIRAPGap.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=12, help="passes over the sweep (0 = forever)")
    ap.add_argument("--tstep", type=float, default=DEF_TSTEP, help="gap step in s (default 1e-6)")
    ap.add_argument("--tmax", type=float, default=DEF_TMAX, help="gap max in s (default 20e-6)")
    a = ap.parse_args()
    submit(url=a.url, reps=a.reps, tstep=a.tstep, tmax=a.tmax)
