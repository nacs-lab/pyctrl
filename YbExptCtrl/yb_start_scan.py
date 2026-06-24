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
import os
import time

from scan_export import scangroup_to_descriptor

# Process/import anchors for client-submit timing: perf_counter for durations, wall for the
# absolute "descriptor enqueued at" stamp that cross-correlates with the backend's bucket-B log.
# For scans that import yb_start_scan before build() (e.g. BeamProfilePushoutScan), the gap from
# this import to the ybStartScan() call captures build()+arg-parse too.
_IMPORT_PERF = time.perf_counter()


def _client_timing_on():
    """Same toggle as the run loop: env ``YB_RUN_TIMING`` truthy OR the ``RUN_TIMING_ON`` file.

    Zero added cost on a normal submit when OFF (an env read + at most one ``os.path.exists``)."""
    v = str(os.environ.get("YB_RUN_TIMING", "")).strip().lower()
    if v not in ("", "0", "false", "no", "off"):
        return True
    try:
        from run_timing import log_dir
        return os.path.exists(os.path.join(log_dir(), "RUN_TIMING_ON"))
    except Exception:  # noqa: BLE001 - any probe failure -> treated as OFF
        return False


def _emit_client_timing(label, did, t_enter, t_export, t_json, t_send, wall_send, desc_bytes):
    """Print + append one client-submit timing row (best-effort; never raises into a submit).

    ``t_enter`` is perf_counter at ybStartScan entry (after build); ``wall_send`` is ``time.time()``
    right before the ZMQ send -> the absolute moment the descriptor is enqueued on the backend, the
    anchor that lines up with the runner log's ``setup timing (bucket B ...)`` stamp."""
    try:
        ms = lambda a, b: (b - a) * 1e3
        pre_ms = (t_enter - _IMPORT_PERF) * 1e3   # import(yb_start_scan) -> ybStartScan(): build()+args
        export_ms, json_ms, send_ms = ms(t_enter, t_export), ms(t_export, t_json), ms(t_json, t_send)
        # Full submit-script wall-time from PROCESS SPAWN to send (interpreter + ALL imports +
        # build + export + json + send) via the OS process create-time. Lets the analyzer split the
        # pre-send gap into "human delay before launch" vs "slow script launch" without guessing.
        try:
            import psutil
            proc_age_ms = (wall_send - psutil.Process().create_time()) * 1e3
        except Exception:  # noqa: BLE001 - psutil missing/blocked -> -1 sentinel
            proc_age_ms = -1.0
        lt = time.localtime(wall_send)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", lt) + (".%03d" % int((wall_send % 1) * 1000))
        print("[client_timing] %s id=%s | proc_age=%.0fms (spawn->send) build+args=%.1fms "
              "export=%.1fms json=%.1fms send=%.1fms desc_bytes=%d | descriptor ENQUEUED at %s"
              % (label, did, proc_age_ms, pre_ms, export_ms, json_ms, send_ms, desc_bytes, stamp))
        from run_timing import log_dir
        path = os.path.join(log_dir(), "client_submit_timing.csv")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as fh:
            if new:
                fh.write("wall_enqueued,label,id,proc_age_ms,build_args_ms,export_ms,json_ms,"
                         "send_ms,desc_bytes\n")
            fh.write("%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n"
                     % (stamp, str(label).replace(",", " "), did, proc_age_ms, pre_ms, export_ms,
                        json_ms, send_ms, desc_bytes))
    except Exception:  # noqa: BLE001 - diagnostics must never break a submit
        pass


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
    timing = _client_timing_on()
    t_enter = time.perf_counter()
    desc = scangroup_to_descriptor(scangroup, seq, opts=opts or None, label=label,
                                   description=description, background=background, cycle=cycle)
    t_export = time.perf_counter()
    desc_json = json.dumps(desc, ensure_ascii=False)
    t_json = time.perf_counter()
    lbl = label or desc["seq"]
    wall_send = time.time()           # absolute moment the descriptor is handed to the backend
    if submit is None:
        from runner import resolve_url
        target = resolve_url([url] if url else [])
        did = submit_descriptor(target, desc_json, lbl)
    else:
        did = int(submit(desc_json, lbl))
    if timing:
        _emit_client_timing(lbl, did, t_enter, t_export, t_json, time.perf_counter(),
                            wall_send, len(desc_json))
    return did


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
