# RearrangeDiagnostics

Diagnostic / sweep scans for SLM atom rearrangement. Every scan here builds a `ScanGroup`
and submits the `RearrangeCommSeq` sequence to the running pyctrl backend -- they only differ
in what they sweep (step size, blur, grating/Zernike distortion, beta/z-weight, prob-Hungarian,
direction, ...). The production rearrange scans (`SLMRearrangementScan.py`,
`SLMRearrangement3DScan.py`) stay one level up in `YbScans/`.

Run from the `pyctrl/` dir, e.g.:

    python YbScans/RearrangeDiagnostics/SLMPingpongGratingZernikeDistortionSweep.py

(Each file's `_bootstrap()` adds the pyctrl root to sys.path via three `dirname` levels,
accounting for this subfolder.)
