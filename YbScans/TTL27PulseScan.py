"""TTL27PulseScan.py -- submit ``TTL27PulseSeq`` to the RUNNING pyctrl backend, once.

Fires the 50 us TTL27 output pulse (``YbSeqs/TTL27PulseSeq.py``) a SINGLE time through the live
backend (shares the engine the backend already owns -- no second-process engine collision). One
scan point, ``rep=1`` -> exactly ONE shot. No atoms, no imaging, no SLM.

The 60 Hz line trigger is enabled in expConfig (consts["LineTrigger"], Channel 0), so the shot
still waits for the mains edge before firing -- expected.

Only BUILDS the 1-point ScanGroup + sends the descriptor JSON; does NOT load the engine, so any
interpreter with pyctrl importable + zmq works.

Run it (pyctrl backend must already be live):
    cd pyctrl
    python YbScans/TTL27PulseScan.py
"""

import scan_bootstrap
scan_bootstrap.bootstrap()   # pyctrl dirs on sys.path (idempotent)

from TTL27PulseSeq import TTL27PulseSeq


def build():
    """A 1-point ScanGroup: nothing to sweep, the seq is self-contained."""
    from scan_group import ScanGroup

    g = ScanGroup()
    rp = g.runp()
    rp.NumPerGroup = 1          # one shot per group
    rp.NumImages = 0            # no imaging step
    rp.Scramble = 0
    rp.useScanLongSlmLock = 0   # does not touch the SLM: skip the lock + loading-phase write
    return g


def TTL27PulseScan(url=None):
    """Build + submit ONE TTL27 pulse shot (rep=1). Returns the queued descriptor id."""
    from yb_start_scan import ybStartScan

    g = build()
    description = "one-shot 50 us TTL27 (FPGA1/TTL27) output pulse; bench trigger, no atoms."
    did = ybStartScan(TTL27PulseSeq, g, url=url, label="TTL27Pulse",
                      description=description,
                      rep=100)   # rep=1 + 1 scan point -> exactly one shot (rep=0 would be forever)
    print("submitted TTL27PulseSeq -> descriptor id %s (url=%s, 1 shot)" % (did, url or "default"))
    return did


if __name__ == "__main__":
    TTL27PulseScan()
