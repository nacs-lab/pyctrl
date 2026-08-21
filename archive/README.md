# archive/

Retired bench/test seqs and scans, kept under version control for reference.

Nothing here is on a live scan path. These are no-atoms bench utilities and one-off
diagnostic scans that were pulled out of `YbSeqs/` and `YbScans/` once the work they
supported finished. They are parked rather than deleted because several of them are the
only record of how a hardware measurement was actually driven.

| File | What it was for |
| --- | --- |
| `YbSeqs/TTL27PulseSeq.py` | One TTL pulse on `FPGA1/TTL27`. Drove the 2026-08-16 scope measurement that established the inverted FPGA wait-trigger edge (see `engine_run._MOLECUBE2_TRIG_EDGE_INVERTED`). |
| `YbSeqs/TTL27PulseSeq_50ms_addstep.py` | Earlier variant of the same seq: 50 ms via two `add_step`s instead of 5 ms via `wait`. Kept because the pulse width and timing construction both differ. |
| `YbSeqs/AWGSwitchTestSeq.py` | Bench test: two AWG trigger edges 10 us apart with the C1/C2 RF switch. |
| `YbSeqs/StrobeBeamTestSeq.py` | Minimal beam-response test for `StrobeImag399Step` (no atoms). |
| `YbScans/StrobeImagingScan.py` | Strobe-imaging optimization scan. Superseded by the round driver below, which builds the same scan inline. |
| `YbSeqs/StrobeImagingPushoutSeq.py` | Strobe-imaging twin of `ImagingPushoutSurvivalSeq`. The seq the round driver submitted. |
| `tools/strobe_imaging_round.py` | Round-by-round driver for the strobe-imaging optimization (`--pulse-time`, `cool` mode). Retired with the rest of the strobe path. |
| `YbScans/StirapGapDarkScan.py` | STIRAP gap/dark-time scan. |
| `YbScans/TTL27PulseScan.py` | Scan wrapper around `TTL27PulseSeq`. |

Some of these seqs were briefly moved to a gitignored scratch dir before landing here; where
the scratch copy had diverged, it is the one archived, since that is the version that was
actually in use (both are kept where they differ meaningfully -- see the TTL27 pair above).

To bring one back, copy it to `YbSeqs/`, `YbScans/`, or `tools/` — imports are plain module-name
imports, so no path rewriting is needed. Note that `archive/` is not on the seq search path,
so a scan cannot import from here as-is.
