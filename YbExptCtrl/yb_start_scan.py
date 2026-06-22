"""yb_start_scan.py -- MATLAB-style scan submission for pyctrl (build g(), then run).

``ybStartScan(seq, g, **opts)`` is the pyctrl entry that lets a "scan file" read like the
MATLAB one (``YbScans/scanConfig/ybStartScan.m``): build a :class:`ScanGroup` field by field,
then submit it. Under the hood it EXPORTS the group to a descriptor JSON
(:func:`scan_export.scangroup_to_descriptor`) and submits it over the SAME ZMQ verb the new
monitor uses (``submit_scan_descriptor``) -- so it rides the one intra-backend payload the
run loop already consumes (runner.py / dispatch_descriptor.py). This is **Option A** (chosen
2026-06-02): MATLAB ergonomics on the single descriptor contract.

This DIVERGES from MATLAB's ``ybStartScan.m`` mechanism on purpose. MATLAB serializes the
ScanGroup into a MATLAB byte-stream job (``ybBuildScanPayload`` -> ``getByteStreamFromArray``
-> ``submit_job``); that payload is MATLAB-proprietary and needs a live MATLAB, neither of
which exists in scenario 3. pyctrl instead emits the portable descriptor JSON.

Example scan file (``YbScans``-style)::

    from scan_group import ScanGroup
    from scan_export import linspace
    from yb_start_scan import ybStartScan
    from Spectrum556Seq import Spectrum556Seq

    def Spectrum556Scan():
        g = ScanGroup()
        g().Cooling.Detuning = 25e6                       # fixed param (fallback)
        g(1).Pushout.Green.Freq.scan(linspace(105e6, 107e6, 23))   # 1-D sweep (dim 1)
        g.runp().NumPerGroup = 200
        g.runp().Scramble = 1                             # per-pass scramble (default off)
        return ybStartScan(Spectrum556Seq, g, rep=5)      # submits; returns the descriptor id

**Scan run order / scramble live in the runp, not the opts.** The runner builds the realized
run order (ybBuildScanJob's ``Scan.Params``: ``StackNum`` passes over the sweep) in the prep
layer from the ScanGroup's ``runp``:
  * ``g.runp().NumPerGroup`` -> ``StackNum = max(ceil(NumPerGroup / nseqs), 2)`` passes (unless
    an explicit ``rep`` opt overrides the pass count; ``rep=0`` = run forever).
  * ``g.runp().Scramble`` -> per-pass scramble (``scrambleGroups``: shuffle WITHIN each pass).
    **Defaults OFF**; set ``g.runp().Scramble = 1`` in the scan file to enable it. The ``random``
    opt does NOT control this -- it only selects runSeq2's legacy global-shuffle on the
    ``rep=0`` forever path.

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import json

from scan_export import scangroup_to_descriptor


def ybStartScan(seq, scangroup, *, url=None, label=None, description=None,
                background=False, cycle=True, submit=None, **opts):
    """Export ``scangroup`` to a descriptor and submit it; return the descriptor id (int).

    Args:
        seq: the seq function (callable) or its name (str).
        scangroup: the :class:`ScanGroup` (single group; built imperatively).
        url: ExptServer ZMQ URL (default: ``runner.resolve_url`` -- ``$NACS_RUNNER_URL`` or
            the canonical default).
        label: queue-UI label (defaults to the seq name).
        description: free-text PURPOSE/CONTEXT for this run (NOT the name) -- why it was run,
            which test group it belongs to, what it's measuring. Be verbose; it is stamped into
            the scan-config sidecar so the analysis dashboard shows it (collapsible) and lets you
            search runs by it. Defaults to ``None`` (blank). ALWAYS pass one for a real scan.
        background: queue this as a low-priority BACKGROUND (calibration) scan -- it runs only
            when no foreground scan is running or queued, yields cleanly at the next shot
            boundary the moment foreground work is queued, and (when ``cycle``) re-queues itself
            so calibrations cycle continuously. Default ``False`` (an ordinary foreground scan).
        cycle: when ``background``, re-queue this scan after each finite slice / yield so it
            runs round-robin with any other background scans. Set ``False`` for a one-shot
            background scan. Ignored when ``background`` is ``False``.
        submit: optional ``(desc_json, label) -> id`` override (injected in tests so no
            socket is bound). Default: :func:`submit_descriptor` over a one-shot REQ socket.
        **opts: run options forwarded as descriptor ``opts`` (``rep`` = explicit pass count,
            ``0`` = forever / ``random`` = forever-path global shuffle / ``tstartwait`` /
            ``pre_cb`` / ``post_cb``); callables become ``{"@": name}``. NOTE: per-pass scramble
            is ``g.runp().Scramble`` (default off), NOT the ``random`` opt -- see the module
            docstring.
    """
    desc = scangroup_to_descriptor(scangroup, seq, opts=opts or None, label=label,
                                   description=description, background=background, cycle=cycle)
    desc_json = json.dumps(desc, ensure_ascii=False)
    lbl = label or desc["seq"]
    if submit is None:
        from runner import resolve_url
        target = resolve_url([url] if url else [])
        return submit_descriptor(target, desc_json, lbl)
    return int(submit(desc_json, lbl))


def submit_descriptor(url, descriptor_json, label="", timeout_ms=2000):
    """Send one ``submit_scan_descriptor`` to the ExptServer at ``url``; return the new id.

    Mirrors ``ExptClient.submit_scan_descriptor``'s framing exactly (REQ socket; verb +
    descriptor + label as three string frames; reply = an 8-byte little-endian id). The
    socket/context are always torn down -- this is a fire-and-return one-shot, not a cached
    client. Raises :class:`TimeoutError` if the server does not reply in ``timeout_ms``.
    """
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(url)
        sock.send_string("submit_scan_descriptor", zmq.SNDMORE)
        sock.send_string(descriptor_json or "", zmq.SNDMORE)
        sock.send_string(label or "")
        if sock.poll(timeout_ms) == 0:
            raise TimeoutError("submit_scan_descriptor: no reply from %s" % url)
        return int.from_bytes(sock.recv(), byteorder="little")
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass
