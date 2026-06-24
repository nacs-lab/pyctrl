"""TrapFreqScan.py -- pyctrl port of ``matlab_new/YbScans/TrapFreqScan.m``.

Parametric-heating trap-frequency measurement of an SLM tweezer array (here the
``33x33_feedback9`` production array). Sweeps the SLM trap-AOM modulation frequency;
survival (image 1 vs image 2) dips at the parametric resonance ``f_mod = 2*f_trap``. The
modulation time is set inverse-square in frequency so the heating "dose" (``t * f^2``) is
held constant across the sweep. Seq = ``SLMTrapModulationSeq``.

Modes (``--mode``), mirroring the active + commented blocks of TrapFreqScan.m:
  * ``radial`` (default) : 100..240 kHz @ 5 kHz (29 pts), AmpFactor 0.4
                           -> radial trap (f_trap ~ 50-120 kHz)
  * ``axial``            :  10.. 60 kHz @ 1 kHz (51 pts), AmpFactor 0.6
                           -> axial trap  (f_trap ~ 5-30 kHz)
  * ``both``             : radial + axial concatenated on ONE scan axis (80 pts), with
                           AmpFactor co-swept per range (0.4 radial, 0.6 axial)

Co-sweep: ``Freq`` and ``Time`` (and, in ``both``, ``AmpFactor``) share scan axis 1, so they
vary together point-for-point. ScanGroup enforces equal length along a dim, so the three
arrays must match (they do, by construction). The modulation time of EACH range is
normalized to ``t_shortest = 1 ms`` at THAT range's highest frequency (matches MATLAB's
``heatingRate = t_shortest * ScannedFreq(end)^2``); so the axial low-freq points reach
~36 ms of modulation per shot -- expected, not a bug.

AmpFactor scales the trap-AOM amplitude during modulation:
``Amp_real = AmpFactor * Consts().SLM.AOM.Amp`` (= 0.55). Edit ``_RADIAL`` / ``_AXIAL`` below
to change a range's frequency window, step, or AmpFactor.

Loading: ``runp().loading_phase`` points the scan at ``33x33_feedback9`` -- the runner writes
that hologram, holds the SLM lock for the scan, applies the feedback9 ByPattern config
overlay (VSLMServo 1.9 + tuned imaging/cooling), and detects with that pattern's per-pattern
thresholds. Change ``LOADING_PHASE`` / ``LOADING_DEFOCUS`` to measure a different array.

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine, so
any interpreter with pyctrl importable + zmq works (yb_analysis env, base, or the engine venv).

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/TrapFreqScan.py                    # radial, rep=3 passes over 29 pts
    python YbScans/TrapFreqScan.py --mode axial
    python YbScans/TrapFreqScan.py --mode both --reps 2
    python YbScans/TrapFreqScan.py --url tcp://127.0.0.1:1408
"""

import argparse
import os
import sys


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


# Modulation drive strength -- AmpFactor = fraction of the trap-AOM amplitude used as the
# parametric-shake tone (Amp_real = AmpFactor * SLM.AOM.Amp, = 0.55). THE main calibration
# knob: aim for ~50% loss on resonance with >90% survival in the wings, then back off to the
# smallest value that keeps the dip clean. Axial (looser trap) needs a stronger drive.
AMP_RADIAL = 0.4
AMP_AXIAL = 0.6

# (start_kHz, step_kHz, stop_kHz, AmpFactor) per modulation range.
_RADIAL = (100, 5, 240, AMP_RADIAL)
_AXIAL = (10, 1, 60, AMP_AXIAL)
_T_SHORTEST = 1e-3           # shortest modulation time (s), at a range's highest frequency

# SLM loading pattern under test (server-side path) + ANSI z4 loading defocus (rad).
LOADING_PHASE = "phase/33x33_feedback9.pt"
LOADING_DEFOCUS = -5
# Bare array name derived from LOADING_PHASE (single source of truth for the label +
# description, e.g. "phase/33x33_feedback9.pt" -> "33x33_feedback9"). Editing LOADING_PHASE
# above retargets the scan AND its label/description together -- no second hardcoded name.
PATTERN_NAME = os.path.splitext(os.path.basename(LOADING_PHASE.replace("\\", "/")))[0]


def _range_arrays(start_khz, step_khz, stop_khz, amp_factor, t_shortest=_T_SHORTEST):
    """(freqs[Hz], times[s], amps) for one modulation range.

    times = heating_rate / f^2 with heating_rate = t_shortest * f_max^2 (constant-dose;
    MATLAB ``ScannedTime = heatingRate ./ ScannedFreq.^2``, normalized to the highest freq)."""
    freqs = [f * 1e3 for f in range(start_khz, stop_khz + 1, step_khz)]
    heating_rate = t_shortest * freqs[-1] ** 2
    times = [heating_rate / f ** 2 for f in freqs]
    amps = [amp_factor] * len(freqs)
    return freqs, times, amps


def build(mode="radial"):
    """The TrapFreqScan ScanGroup (single group, modulation Freq/Time co-swept on axis 1)."""
    _bootstrap()
    from scan_group import ScanGroup

    if mode == "radial":
        freqs, times, amps = _range_arrays(*_RADIAL)
    elif mode == "axial":
        freqs, times, amps = _range_arrays(*_AXIAL)
    elif mode == "both":
        fr, tr, ar = _range_arrays(*_RADIAL)
        fa, ta, aa = _range_arrays(*_AXIAL)
        freqs, times, amps = fr + fa, tr + ta, ar + aa
    else:
        raise ValueError("mode must be radial/axial/both, got %r" % (mode,))

    g = ScanGroup()

    # ---- modulation drive: Freq + Time co-swept on axis 1 -----------------
    g().SLMTrapModulation.Freq.scan(1, freqs)
    g().SLMTrapModulation.Time.scan(1, times)
    # AmpFactor: a fixed leaf when uniform across the sweep (radial / axial); co-swept on
    # axis 1 when it differs per point (both -> 0.4 radial, 0.6 axial).
    if len(set(amps)) == 1:
        g().SLMTrapModulation.AmpFactor = amps[0]
    else:
        g().SLMTrapModulation.AmpFactor.scan(1, amps)

    # ---- fixed trap-lowering params (lower depth, let hot atoms fly out) --
    g().SLMTrapModulation.lowerTrapDepth.Vservo = 0.1
    g().SLMTrapModulation.lowerTrapDepth.Time = 1e-3

    # ---- run params (runp); no byte effect, drive the live run ------------
    rp = g.runp()
    rp.NumPerGroup = 1000        # TOTAL-shots target when no rep is passed (reconciled below)
    rp.NumImages = 2             # img1 (before) + img2 (after) -> survival
    rp.Scramble = 1              # randomize point order (interleaves the two ranges in `both`)
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    # Measure the 33x33_feedback9 production array: writes the hologram, holds the SLM lock,
    # applies the feedback9 ByPattern overlay (VSLMServo 1.9 + tuned imaging/cooling), and
    # detects with that pattern's per-pattern thresholds.
    rp.loading_phase = LOADING_PHASE
    rp.loading_defocus = LOADING_DEFOCUS
    return g


def TrapFreqScan(url=None, reps=3, mode="radial"):
    """Build + submit the trap-frequency scan (radial/axial/both). Returns the descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build(mode=mode)
    npts = g.scansize(1)         # authoritative point count (derived from the built sweep)
    opts = {}
    if reps is not None:
        # rep=0 -> run forever; rep>=1 -> that many passes; omit -> StackNum from NumPerGroup.
        opts["rep"] = reps
        if reps > 0:
            # Make the dashboard's scheduled-shots total agree with the real cap
            # (rep counts PASSES; total shots = reps * npts). See the experiment-running
            # skill's "Shots vs reps vs NumPerGroup" footgun note.
            g.runp().NumPerGroup = reps * npts

    label = "TrapFreqScan_%s_%s" % (PATTERN_NAME, mode)
    description = (
        "Parametric-heating trap-frequency measurement of the %s production "
        "array (%s modulation range%s). Survival dips at f_mod = 2*f_trap; modulation time "
        "held inverse-square in frequency for a constant heating dose. %d frequency points."
        % (PATTERN_NAME, mode, ", AmpFactor co-swept 0.4/0.6" if mode == "both" else "", npts))
    did = ybStartScan("SLMTrapModulationSeq", g, url=url, label=label,
                      description=description, **opts)
    print("submitted %s -> descriptor id %s (url=%s, reps=%s, %d freq pts)"
          % (label, did, url or "default", reps, npts))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit TrapFreqScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=3,
                    help="passes over the sweep (0 = forever); default 3")
    ap.add_argument("--mode", default="radial", choices=("radial", "axial", "both"),
                    help="radial (default, 100-240 kHz @ AmpFactor 0.4), "
                         "axial (10-60 kHz @ 0.6), or both (concatenated, AmpFactor co-swept)")
    args = ap.parse_args()
    TrapFreqScan(url=args.url, reps=args.reps, mode=args.mode)
