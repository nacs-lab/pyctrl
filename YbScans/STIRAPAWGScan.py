"""STIRAPAWGScan.py -- pyctrl port of ``matlab_new/YbScans/STIRAPAWGScan.m``.

Builds the STIRAP push-out survival ScanGroup (seq = ``PushoutSurvivalAWGSeq``) and submits it to
the RUNNING pyctrl backend over ZMQ, mirroring the .m's
``ybStartScan(STIRAPAWGScanParams(), @PushoutSurvivalAWGSeq)``.

The two-photon STIRAP (556 + 308) shaped pulses come from the Siglent SDG6X AWGs (per-AWG
``shape``: half-Gaussian 556 = rise_gaussian / 308 = fall_gaussian; see
``devices/sigilent_awg/pulse_waveform.py``). The AWG params
are declared under ``g().AWG.AWG556.*`` / ``g().AWG.AWG308.*`` and activated by ``runp().AWGs`` --
the run loop's ``AWGManager`` pre-stores every unique waveform at scan start and switches per shot
(``ARWV NAME`` on fw >= 38R3, re-upload fallback otherwise; ``devices/sigilent_awg``). Here the AWG
params are FIXED -> one waveform per AWG, uploaded once, never switched.

Swept axis = ``Pushout.STIRAP.gap`` over (100:100:10000) ns (100 pts): the STIRAP forward/reverse
gap, which ``STIRAPPushoutStep`` uses as the ``TTLQickTrig`` high duration. Two ``Imag399Step``
calls => NumImages=2 (survival vs gap).

QICK note: the .m's ``server_pre_run`` uploaded a QICK Ramsey microwave program keyed on
``Pushout.MRabi.*`` + ``Ramsey.Phase``; that port is DEFERRED, so those params are set for the
future but currently unused (no QICK program is loaded -> ``TTLQickTrig`` fires but no microwave).

This only BUILDS the ScanGroup + sends the descriptor JSON; it does NOT load the engine.

Run it (pyctrl backend must already be live at --url):
    cd pyctrl
    python YbScans/STIRAPAWGScan.py                 # short A/B run: rep=3 over 100 gap pts
    python YbScans/STIRAPAWGScan.py --reps 0        # run forever
"""

import argparse
import os
import sys
import numpy as np

def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    if root not in sys.path:
        sys.path.insert(0, root)


def build():
    """The STIRAPAWGScan ScanGroup (1-D Pushout.STIRAP.gap sweep; fixed AWG waveforms)."""
    _bootstrap()
    from scan_group import ScanGroup
    from scan_export import matlab_colon

    g = ScanGroup()

    # ---- Siglent AWG config (out-of-band; AWGManager reads g().AWG.<name>.*; runp().AWGs activates).
    # FIXED here (not scanned) -> one waveform per AWG, pre-stored once.
    # shape (devices/sigilent_awg/pulse_waveform.py; gallery pyctrl/tmp/pulse_10_examples.png):
    #   gaussian / rise_gaussian / fall_gaussian / rise_linear / fall_linear.
    #   pulse_width_us = MAIN window; smooth_width_us = EXTRA half-cosine edge (0 = sharp step).
    #   Half-Gaussian STIRAP: 556 = rise_gaussian (peak at END of its window),
    #   308 = fall_gaussian (peak at START, i.e. at its trigger + smooth_width_us).
    #   NOTE: total playback = pulse_width_us + smooth_width_us -> the TTL gate must cover it.
    g().AWG.AWG556.shape = "gaussian"
    #g().AWG.AWG556.smooth_width_us = 0.1
    # parked on the 20um-array 2D dip center (scans 20260707_180931 + _183401, 17x17_20um):
    # 616=234.46 / 556=143.33, dip survival ~0.06-0.09 (~90-93% transfer, deepest of the spacing
    # series 10/14.5/20um -> Rydberg blockade). was 142.62 (14.5um scan 20260707_170113)
    g().AWG.AWG556.carrier_freq_MHz = 143.33 #.scan(2, np.linspace(141.4, 144.4, 11))
    g().AWG.AWG556.pulse_width_us = 4    # old params: steepness 4, pw 4us
    g().AWG.AWG556.steepness = 4 #1.2*2
    g().AWG.AWG556.max_amplitude_vpp = 11 #11
    g().AWG.AWG556.amplitude_scale = 1

    g().AWG.AWG308.shape = "gaussian"
    #g().AWG.AWG308.smooth_width_us = 0.1
    g().AWG.AWG308.carrier_freq_MHz = 200
    g().AWG.AWG308.pulse_width_us = 4    # old params: steepness 4, pw 4us
    g().AWG.AWG308.steepness = 4 #1.5*2
    g().AWG.AWG308.max_amplitude_vpp = 5.5 #6.5 #.scan(1, np.linspace(6, 7, 6)) #
    g().AWG.AWG308.amplitude_scale = 1

    # Which AWGs to activate (the run loop reads runp().AWGs).
    g.runp().AWGs = ["AWG556", "AWG308"]

    # ---- QICK microwave params (consumed by the DEFERRED QICK upload; no effect until
    #      FPGAAWGManager is wired -- kept for the future port).
    g().Pushout.MRabi.Freq = 10863.04
    g().Pushout.MRabi.Gain = 3000
    g().Pushout.MRabi.FreqRabi = 7.187e6
    g().Pushout.Ramsey.Phase = 0

    # ---- STIRAP push-out params (STIRAPPushoutStep reads these) ----
    # parked on the 20um-array 2D dip center (scans 20260707_180931 + _183401, 17x17_20um):
    # 616=234.46 / 556=143.33. was 233.88 (14.5um 20260707_170113); 233.97 (10um _132639)
    g().Init.EOM616.Freq = 234.46e6 #.scan(1, np.linspace(233.5e6, 235.5e6, 11))
    g().Pushout.VRydTrap = 0.5
    # TTL gate width = pulse width (else the gaussian is clipped/repeated); fixed at 4us
    g().Pushout.STIRAP.guassian_pulse_width = 4e-6
    # scan the 308->556 overlap delay at fixed pw=4us, ZOOM 0.01-2us, 10 pts
    g().Pushout.STIRAP.delay = 0.9e-6
    g().Pushout.STIRAP.ifReverse = True
    g().Pushout.STIRAP.reverse_delay.scan(1, np.linspace(0.01e-6, 2e-6, 10)) #= 0.7e-6
    g().Pushout.STIRAP.waitTime = 0e-6
    g().Pushout.Amp369 = 1
    g().Pushout.Time369 = 1e-6
    g().Pushout.BiasCoilCurrent.Ryd = 30

    # ---- swept axis: STIRAP.gap over (100:100:10000) ns = 100 pts ----
    gaps = [v * 1e-9 for v in matlab_colon(100, 100, 10000)]
    g().Pushout.STIRAP.gap = 1e-6 #.scan(1, gaps)

    # ---- per-scan SLM loading-pattern override (default from expConfig.py ---
    g.runp().loading_phase = "phase/17x17_20um.pt"   # server-side WGS phase path
    g.runp().loading_defocus = -5                         # ANSI z4 loading defocus (rad)
    
    # ---- run params (runp); no byte effect, drive the live run ----
    rp = g.runp()
    rp.NumPerGroup = 2000
    rp.NumImages = 2
    rp.Scramble = 1
    rp.isInit = 0
    rp.isHC = 0
    rp.isGrid2 = 0
    return g


def STIRAPAWGScan(url=None, reps=3):
    """Build + submit the STIRAP AWG scan. Returns the queued descriptor id."""
    _bootstrap()
    from yb_start_scan import ybStartScan

    g = build()
    npts = g().Pushout.STIRAP.gap.size(1)
    opts = {}
    if reps is not None:
        opts["rep"] = reps
    did = ybStartScan("PushoutSurvivalAWGSeq", g, url=url, label="STIRAPAWGScan", **opts)
    print("submitted STIRAPAWGScan -> descriptor id %s (url=%s, reps=%s, %d gap pts)"
          % (did, url or "default", reps, npts))
    return did


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Submit STIRAPAWGScan to the pyctrl backend.")
    ap.add_argument("--url", default=None,
                    help="ExptServer URL (default: $NACS_RUNNER_URL or tcp://127.0.0.1:1408)")
    ap.add_argument("--reps", type=int, default=10,
                    help="passes over the sweep (0 = forever); default 3 for a short A/B run")
    args = ap.parse_args()
    STIRAPAWGScan(url=args.url, reps=args.reps)
