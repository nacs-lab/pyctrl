"""qick_awg -- the QICK FPGA_AWG (RFSoC4x2) RF AWG for Rydberg/STIRAP/microwave pulses.

A SEPARATE board from the libnacs ``zynq`` FPGA1: not in ``config.yml``, armed out-of-band, and
**TTL-triggered** (``TTLQickTrig = FPGA1/TTL14``). It runs its own socket server with a small
length-prefixed protocol. Three pieces, ported from ``matlab_new/YbExptCtrl/qick/``:

  * :class:`FPGAAWGClient` / :class:`FPGAABSClient` -- the socket client (1:1 with the upstream
    ``cliulinnaeus/FPGA_AWG@rfsoc4x2`` ``FPGA_AWG_client.py`` and the MATLAB reimpl). Stdlib socket
    only -- import-safe with no vendor package.
  * :func:`simple_pulse_cfg` / :func:`compile_chn` / :func:`simple_prog_cfg` -- pure pulse + program
    JSON builders (== ``uploadSimplePulse``/``compileCHN``/``uploadSimpleProg``).
  * :class:`FPGAAWGManager` -- the scan coordinator: BATCH-upload every unique program at scan start,
    PICK the active program per shot (``start_program``, skip on no-change), clean up at scan end.
    The QICK protocol supports both; the existing MATLAB seqs re-upload every shot instead.

⚠ Out-of-band: QICK output is NOT in the serialized seq byte blob, so THE ONE RULE does not apply.
"""
from .fpga_awg_client import (DEFAULT_HOST, DEFAULT_PORT, FPGAABSClient,
                              FPGAAWGClient)
from .fpga_awg_manager import FPGAAWGManager, QickProgram
from .simple_pulse import (Loop, compile_chn, loop, render_tokens,
                           simple_prog_cfg, simple_pulse_cfg)

__all__ = [
    "FPGAABSClient", "FPGAAWGClient", "DEFAULT_HOST", "DEFAULT_PORT",
    "FPGAAWGManager", "QickProgram",
    "simple_pulse_cfg", "simple_prog_cfg", "compile_chn", "render_tokens", "loop", "Loop",
]

# Registry: create("qick_awg") -> an FPGAAWGClient (NEEDS-HARDWARE on connect()).
# The FPGAAWGManager is the scan-level orchestrator.
from ..device_registry import register

register("qick_awg")(FPGAAWGClient)
