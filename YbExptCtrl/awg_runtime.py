"""awg_runtime.py -- per-scan AWG glue over devices/sigilent_awg (and soon devices/qick_awg).

Batch-uploads the unique waveforms at scan start (:func:`setup`), recalls the active waveform
per shot (:func:`make_pre_cb`), and disconnects at scan end (:func:`cleanup`). :func:`awg_names`
reports which AWGs a scan activates.

The ordering contract (AWG setup BEFORE the scan-long SLM lock, for lease freshness) is owned by
engine_run, not this module. QICK wiring will land here next to the Siglent code.

Split out of runner.py (now run_loop.py) 2026-07-22.
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


# --------------------------------------------------------------------------- #
# QICK FPGA_AWG (RFSoC4x2 microwave) -- the analog of the Siglent glue above.
#
# Differs from Siglent in two ways the board forces (see devices/qick_awg + the plan):
#   * upload-all-once: every unique program is batch-uploaded at scan start (setup); nothing is
#     re-uploaded per shot (the Siglent re-sends a WVDT per shot).
#   * arm-EVERY-shot: the board is one-shot, so the per-shot pre_cb ALWAYS re-arms (stop+start) via
#     FPGAAWGManager.arm_for_seq -- not the skip-on-unchanged recall_for_seq.
# A scan opts in with g().runp().QICK = True and declares the sequence via g().QICK.* (expConfig
# c["QICK"]). Trigger is external (TTLQickTrig = FPGA1/TTL14), set once in setup.
# --------------------------------------------------------------------------- #
def qick_enabled(scangroup):
    """Whether this scan activates the QICK board (``runp().QICK``); [] non-QICK scans skip it."""
    from devices.qick_awg import qick_enabled as _qick_enabled
    return _qick_enabled(scangroup)


def qick_setup(scangroup):
    """Batch-upload every unique QICK program for the scan + arm external-trigger mode (once)."""
    from devices.qick_awg import FPGAAWGManager, build_programs
    from devices.qick_awg.fpga_awg_client import DEFAULT_HOST, DEFAULT_PORT
    host, port = _qick_host_port(scangroup)
    FPGAAWGManager.setup(build_programs(scangroup),
                         host=host or DEFAULT_HOST, port=port or DEFAULT_PORT,
                         trigger_mode="external")


def make_qick_pre_cb(scangroup):
    """Return the per-shot pre_cb that ARMS this point's QICK program (stop+start EVERY shot)."""
    def _qick_pre_cb(_seq_num, arg0):
        from devices.qick_awg import FPGAAWGManager, seq_qick_key
        pt = scangroup.getseq(arg0)
        FPGAAWGManager.arm_for_seq(seq_qick_key(pt))
    return _qick_pre_cb


def qick_cleanup():
    """Stop the running program + disconnect the QICK board at scan end (best-effort)."""
    try:
        from devices.qick_awg import FPGAAWGManager
        FPGAAWGManager.cleanup()
    except Exception:  # noqa: BLE001
        pass


def _qick_host_port(scangroup):
    """Resolve (host, port) for the QICK server from c["QICK"] (None,None -> client defaults)."""
    try:
        from seq_config import SeqConfig
        q = SeqConfig.get().consts.get("QICK", {})
        return q.get("host"), q.get("port")
    except Exception:  # noqa: BLE001
        return None, None
