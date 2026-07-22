"""awg_runtime.py -- per-scan AWG glue over devices/sigilent_awg (and soon devices/qick_awg).

Batch-uploads the unique waveforms at scan start (:func:`setup`), recalls the active waveform
per shot (:func:`make_pre_cb`), and disconnects at scan end (:func:`cleanup`). :func:`awg_names`
reports which AWGs a scan activates.

The ordering contract (AWG setup BEFORE the scan-long SLM lock, for lease freshness) is owned by
engine_run, not this module. QICK wiring will land here next to the Siglent code.

Split out of runner.py 2026-07-22.
"""


def awg_names(scangroup):
    """The AWGs this scan activates: ``runp().AWGs`` (e.g. ``["AWG556"]``), [] if unset.

    Mirrors MATLAB ``scanp.AWGs({})`` gating AWGManager.setup. A bare string is wrapped; a
    missing/empty field -> [] so non-AWG scans skip the AWG path entirely.
    """
    try:
        v = scangroup.runp().AWGs([])
    except Exception:  # noqa: BLE001
        return []
    if isinstance(v, str):
        return [v]
    try:
        return [str(x) for x in v if x]
    except TypeError:
        return []


def setup(names, scangroup):
    """Batch-upload every unique waveform for the AWGs named in ``names`` from the ScanGroup."""
    from devices.sigilent_awg import AWGManager
    AWGManager.setup(names, scangroup)


def make_pre_cb(scangroup):
    """Return the per-shot pre_cb that recalls this point's active AWG waveform(s)."""
    def _awg_pre_cb(_seq_num, arg0):
        from devices.sigilent_awg import AWGManager
        pt = scangroup.getseq(arg0)
        AWGManager.recall_for_seq(pt.get("AWG", {}) if isinstance(pt, dict) else {})
    return _awg_pre_cb


def cleanup():
    """Disconnect all AWGs at scan end (best-effort; swallow exceptions)."""
    try:
        from devices.sigilent_awg import AWGManager
        AWGManager.cleanup()                         # disconnect all AWGs
    except Exception:  # noqa: BLE001
        pass
